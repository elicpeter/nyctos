//! Claude Code adapter (`agent_loop`).
//!
//! Spawns the `claude` CLI as a subprocess so the rest of the agent
//! does not have to embed Anthropic's tool-use loop. The adapter
//! detects the binary on `PATH` at construction time and refuses to
//! run if it is missing; callers fall back to the Anthropic adapter
//! for `one_shot` work.
//!
//! Wire shape:
//! 1. Write `agent_task.md` into a per-task scratch directory.
//! 2. Invoke `claude --print --output-format stream-json --verbose` with
//!    the instruction file content piped on stdin (the public CLI does
//!    not currently expose a `--instruction-file` flag; the scratch
//!    file still lands on disk so traces stay auditable).
//! 3. Parse the NDJSON event stream into structured events,
//!    republishing tool-use blocks as `AiEvent::ToolCallStarted`/
//!    `Finished` on the shared event bus.
//! 4. Lift recognised tool calls into `ExtractedAgentResult` variants
//!    so downstream tasks (PayloadSynthesis, SpecExtraction,
//!    ChainRanking, Exploration) consume a typed agent-loop result
//!    rather than re-parsing Claude Code's raw transcript.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use nyx_agent_types::agent::{
    classify_tool_use, AgentResult, AgentTask, AiError, Budget, CacheStats, CostEstimate,
    ExtractedAgentResult, HaltReason, Prompt, Response, TokenUsage,
};
use nyx_agent_types::event::{AgentEvent, AiEvent, EventSink};
use semver::Version;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStderr, Command};
use tokio::task::JoinHandle;

use crate::runtime::{AiRuntime, SharedBudgetTracker};

/// Preferred binary name. Falls back to `claude-code` when `claude` is
/// absent so operators who installed the older alias still work.
pub const DEFAULT_CLAUDE_BINARY: &str = "claude";
const FALLBACK_CLAUDE_BINARY: &str = "claude-code";

/// Built-in minimum supported Claude Code version. The adapter's
/// `--print --output-format stream-json --verbose --max-turns N` call
/// shape and the `assistant` / `result` event types parsed in
/// [`parse_stream_json`] have been stable since this release. Raise
/// the floor when a load-bearing CLI flag or stream-json field
/// shape change forces a hard cutover.
pub const MINIMUM_CLAUDE_VERSION: &str = "1.0.0";

/// Upper bound on the number of stderr bytes the drain task retains.
/// `claude --verbose` can write multi-megabyte transcripts; an
/// unbounded buffer would let a runaway child OOM the daemon. The
/// drain keeps the trailing window (most recent bytes) because the
/// proximate failure reason is almost always at the end of the stream.
const MAX_STDERR_CAPTURE_BYTES: usize = 64 * 1024;

/// Path + version string captured at adapter-construction time. Surfaced
/// by `nyx-agent doctor` so operators can confirm which binary the
/// daemon will spawn.
#[derive(Clone, Debug)]
pub struct ClaudeBinary {
    pub path: PathBuf,
    pub version: String,
}

/// Resolve the Claude Code binary on `PATH` and capture its
/// `--version` output. Returns `AiError::AdapterUnavailable` when no
/// candidate is found or the version probe fails.
pub async fn detect_claude_binary() -> Result<ClaudeBinary, AiError> {
    let path = which::which(DEFAULT_CLAUDE_BINARY)
        .or_else(|_| which::which(FALLBACK_CLAUDE_BINARY))
        .map_err(|_| {
            AiError::AdapterUnavailable(format!(
                "`{DEFAULT_CLAUDE_BINARY}` (or `{FALLBACK_CLAUDE_BINARY}`) not on PATH"
            ))
        })?;

    let output = Command::new(&path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| {
            AiError::AdapterUnavailable(format!(
                "failed to invoke {} --version: {e}",
                path.display()
            ))
        })?;

    if !output.status.success() {
        return Err(AiError::AdapterUnavailable(format!(
            "{} --version exited {}: {}",
            path.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let parsed = parse_claude_version(&version).ok_or_else(|| {
        AiError::AdapterUnavailable(format!(
            "claude-code version output `{version}` could not be parsed as semver"
        ))
    })?;
    let floor = Version::parse(MINIMUM_CLAUDE_VERSION).expect("built-in floor parses");
    if parsed < floor {
        return Err(AiError::AdapterUnavailable(format!(
            "claude-code v{parsed} below required minimum v{floor}"
        )));
    }
    Ok(ClaudeBinary { path, version })
}

/// Extract a `semver::Version` from the `claude --version` stdout. The
/// CLI prints either `<X.Y.Z> (Claude Code)` or a bare `<X.Y.Z>` token;
/// the parser accepts both shapes and returns `None` when no
/// semver-shaped token is present.
fn parse_claude_version(raw: &str) -> Option<Version> {
    let candidate = raw.split_whitespace().next()?;
    Version::parse(candidate).ok()
}

/// Claude Code agent-loop adapter.
#[derive(Clone)]
pub struct ClaudeCodeAdapter {
    binary: ClaudeBinary,
    tracker: SharedBudgetTracker,
    default_model: String,
    effort: Option<String>,
    context_window: Option<u32>,
    /// Wall-clock cap on a single agent-loop invocation. Defaults to
    /// 15 minutes; operators can override via `with_timeout`.
    timeout: Duration,
}

impl ClaudeCodeAdapter {
    /// Build an adapter from a pre-detected binary. Callers typically
    /// run [`detect_claude_binary`] first and surface the result through
    /// the doctor, then construct one of these.
    pub fn new(binary: ClaudeBinary, tracker: SharedBudgetTracker) -> Self {
        Self {
            binary,
            tracker,
            default_model: "claude-opus-4-7".to_string(),
            effort: None,
            context_window: None,
            timeout: Duration::from_secs(15 * 60),
        }
    }

    /// Convenience: detect + construct in one shot.
    pub async fn discover(tracker: SharedBudgetTracker) -> Result<Self, AiError> {
        let binary = detect_claude_binary().await?;
        Ok(Self::new(binary, tracker))
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    pub fn with_effort(mut self, effort: impl Into<String>) -> Self {
        let effort = effort.into();
        if !effort.trim().is_empty() {
            self.effort = Some(effort);
        }
        self
    }

    pub fn with_context_window(mut self, context_window: u32) -> Self {
        if context_window > 0 {
            self.context_window = Some(context_window);
        }
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn binary(&self) -> &ClaudeBinary {
        &self.binary
    }
}

#[async_trait]
impl AiRuntime for ClaudeCodeAdapter {
    fn name(&self) -> &'static str {
        "claude-code"
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn supports_agent_loop(&self) -> bool {
        true
    }

    fn supports_prompt_cache(&self) -> bool {
        true
    }

    fn supports_deterministic_sampling(&self) -> bool {
        false
    }

    async fn one_shot(
        &self,
        prompt: Prompt,
        budget: Budget,
        sink: EventSink,
    ) -> Result<Response, AiError> {
        let requested_model = prompt.model.clone().unwrap_or_else(|| self.default_model.clone());
        let model = claude_model_for_context(&requested_model, self.context_window);

        // Mirror the Anthropic adapter's cap semantics for structured
        // one-shot work: the effective ceiling is the tighter of the
        // tracker row and the per-call envelope.
        let spent_before = self.tracker.current_spend(&budget.run_id, budget.kind).await?;
        let tracker_cap = self.tracker.cap(&budget.run_id, budget.kind).await?;
        let cap = effective_cap(tracker_cap, budget.cap_usd_micros);
        if spent_before > cap {
            let _ = sink.send(AgentEvent::Ai {
                data: AiEvent::TaskHalted {
                    task_id: prompt.task_id.clone(),
                    reason: HaltReason::BudgetCapReached,
                },
            });
            return Err(AiError::BudgetExceeded {
                cap_usd_micros: cap,
                spent_usd_micros: spent_before,
            });
        }

        let prompt_body = render_one_shot_prompt(&prompt);
        let mut child = Command::new(&self.binary.path)
            .arg("--print")
            .arg("--output-format")
            .arg("stream-json")
            .arg("--verbose")
            .arg("--max-turns")
            .arg("1")
            .arg("--model")
            .arg(&model)
            .args(effort_args(self.effort.as_deref()))
            // TODO(release-hardening): make this opt-in/configured before
            // shipping beyond local testing.
            .arg("--dangerously-skip-permissions")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| AiError::Transport(format!("spawn claude: {e}")))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(prompt_body.as_bytes())
                .await
                .map_err(|e| AiError::Transport(format!("write stdin: {e}")))?;
            stdin.shutdown().await.map_err(|e| AiError::Transport(format!("close stdin: {e}")))?;
        }

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AiError::Transport("claude stdout missing".to_string()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| AiError::Transport("claude stderr missing".to_string()))?;
        let stderr_handle = spawn_stderr_drain(stderr);
        let mut reader = BufReader::new(stdout).lines();

        let mut content = String::new();
        let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut cache = CacheStats { cache_creation_tokens: 0, cache_read_tokens: 0 };
        let mut cost_usd_micros: i64 = 0;
        let mut reported_model: Option<String> = None;
        let mut result_error: Option<String> = None;

        let read_loop = async {
            while let Some(line) = reader
                .next_line()
                .await
                .map_err(|e| AiError::Transport(format!("read stdout: {e}")))?
            {
                if line.trim().is_empty() {
                    continue;
                }
                let Some(event) = parse_stream_json(&line) else {
                    continue;
                };
                match event {
                    ClaudeEvent::Assistant(msg) => {
                        for block in msg.content {
                            if let ContentBlock::Text { text } = block {
                                content.push_str(&text);
                                let _ = sink.send(AgentEvent::Ai {
                                    data: AiEvent::TokenReceived {
                                        task_id: prompt.task_id.clone(),
                                        token: text,
                                    },
                                });
                            }
                        }
                    }
                    ClaudeEvent::Result(r) => {
                        if r.is_error == Some(true) {
                            result_error = Some(format_result_error(&r.subtype, &r.result));
                        }
                        if let Some(model) = r.model {
                            reported_model = Some(model);
                        }
                        if let Some(u) = r.usage {
                            usage.input_tokens =
                                usage.input_tokens.saturating_add(u.input_tokens.unwrap_or(0));
                            usage.output_tokens =
                                usage.output_tokens.saturating_add(u.output_tokens.unwrap_or(0));
                            cache.cache_creation_tokens = cache
                                .cache_creation_tokens
                                .saturating_add(u.cache_creation_input_tokens.unwrap_or(0));
                            cache.cache_read_tokens = cache
                                .cache_read_tokens
                                .saturating_add(u.cache_read_input_tokens.unwrap_or(0));
                        }
                        if let Some(c) = r.total_cost_usd {
                            cost_usd_micros = (c * 1_000_000.0).round() as i64;
                        }
                        if let Some(text) = r.result {
                            if content.is_empty() {
                                content = text;
                            }
                        }
                    }
                    ClaudeEvent::Other => {}
                }
            }
            Ok::<(), AiError>(())
        };

        match tokio::time::timeout(self.timeout, read_loop).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = child.kill().await;
                let stderr_text = await_stderr_drain(stderr_handle).await;
                return Err(annotate_with_stderr(e, &stderr_text));
            }
            Err(_) => {
                let _ = child.kill().await;
                let _ = sink.send(AgentEvent::Ai {
                    data: AiEvent::TaskHalted {
                        task_id: prompt.task_id.clone(),
                        reason: HaltReason::OperatorCancelled,
                    },
                });
                let stderr_text = await_stderr_drain(stderr_handle).await;
                let base = format!("claude one_shot timed out after {}s", self.timeout.as_secs());
                return Err(AiError::Transport(append_stderr(&base, &stderr_text)));
            }
        }

        let status =
            child.wait().await.map_err(|e| AiError::Transport(format!("wait claude: {e}")))?;
        if !status.success() {
            let stderr_text = await_stderr_drain(stderr_handle).await;
            let base = format!("claude exited {status}");
            let base = append_result_error(&base, &result_error);
            return Err(AiError::UpstreamRefused(append_stderr(&base, &stderr_text)));
        }
        drop(stderr_handle);

        if cache.cache_creation_tokens > 0 {
            let _ = sink.send(AgentEvent::Ai {
                data: AiEvent::CacheMiss {
                    task_id: prompt.task_id.clone(),
                    tokens: cache.cache_creation_tokens,
                },
            });
        }
        if cache.cache_read_tokens > 0 {
            let _ = sink.send(AgentEvent::Ai {
                data: AiEvent::CacheHit {
                    task_id: prompt.task_id.clone(),
                    tokens: cache.cache_read_tokens,
                },
            });
        }

        let spent_after =
            self.tracker.add_spend(&budget.run_id, budget.kind, cost_usd_micros).await?;
        let _ = sink.send(AgentEvent::Ai {
            data: AiEvent::BudgetTick {
                task_id: prompt.task_id.clone(),
                run_id: budget.run_id.clone(),
                spent_usd_micros: spent_after,
            },
        });

        let tracker_cap = self.tracker.cap(&budget.run_id, budget.kind).await?;
        let cap = effective_cap(tracker_cap, budget.cap_usd_micros);
        if spent_after > cap {
            let _ = sink.send(AgentEvent::Ai {
                data: AiEvent::TaskHalted {
                    task_id: prompt.task_id.clone(),
                    reason: HaltReason::BudgetCapReached,
                },
            });
            return Err(AiError::BudgetExceeded {
                cap_usd_micros: cap,
                spent_usd_micros: spent_after,
            });
        }

        Ok(Response {
            prompt_version: prompt.prompt_version,
            task_id: prompt.task_id,
            model: reported_model.unwrap_or(model),
            content,
            usage,
            cache: Some(cache),
            cost_usd_micros,
        })
    }

    async fn agent_loop(
        &self,
        task: AgentTask,
        budget: Budget,
        sink: EventSink,
    ) -> Result<AgentResult, AiError> {
        // Pre-call budget check uses `>` to match the post-call check
        // below; cap is the spendable ceiling, not a hard refuse-when-equal.
        let spent_before = self.tracker.current_spend(&budget.run_id, budget.kind).await?;
        if let Some(cap) = self.tracker.cap(&budget.run_id, budget.kind).await? {
            if spent_before > cap {
                let _ = sink.send(AgentEvent::Ai {
                    data: AiEvent::TaskHalted {
                        task_id: task.task_id.clone(),
                        reason: HaltReason::BudgetCapReached,
                    },
                });
                return Err(AiError::BudgetExceeded {
                    cap_usd_micros: cap,
                    spent_usd_micros: spent_before,
                });
            }
        }

        let scratch = tempfile::Builder::new()
            .prefix("nyx-claude-task-")
            .tempdir()
            .map_err(|e| AiError::Transport(format!("scratch dir: {e}")))?;
        let task_path = scratch.path().join("agent_task.md");
        let task_body = render_task_markdown(&task);
        tokio::fs::write(&task_path, &task_body)
            .await
            .map_err(|e| AiError::Transport(format!("write agent_task.md: {e}")))?;

        let model = claude_model_for_context(&self.default_model, self.context_window);
        let mut cmd = Command::new(&self.binary.path);
        cmd.arg("--print")
            .arg("--output-format")
            .arg("stream-json")
            .arg("--verbose")
            .arg("--max-turns")
            .arg(task.max_turns.to_string())
            .arg("--model")
            .arg(&model)
            .args(effort_args(self.effort.as_deref()))
            // TODO(release-hardening): make this opt-in/configured before
            // shipping beyond local testing.
            .arg("--dangerously-skip-permissions")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Pipe stderr so failure paths can surface the upstream reason.
            // A sibling task drains the pipe into a bounded buffer
            // (`MAX_STDERR_CAPTURE_BYTES`) in parallel with stdout, so
            // `--verbose` output cannot block the child on a full pipe.
            .stderr(Stdio::piped())
            // Ensure SIGKILL fires and the child is reaped if a future error
            // path drops `child` before `wait().await` runs. The timeout arm
            // below calls `kill().await` (which reaps), but `kill_on_drop`
            // covers panic/early-return paths too.
            .kill_on_drop(true);
        if let Some(dir) = task.working_directory.as_deref() {
            cmd.current_dir(dir);
        }
        let mut child =
            cmd.spawn().map_err(|e| AiError::Transport(format!("spawn claude: {e}")))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(task_body.as_bytes())
                .await
                .map_err(|e| AiError::Transport(format!("write stdin: {e}")))?;
            stdin.shutdown().await.map_err(|e| AiError::Transport(format!("close stdin: {e}")))?;
        }

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AiError::Transport("claude stdout missing".to_string()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| AiError::Transport("claude stderr missing".to_string()))?;
        let stderr_handle = spawn_stderr_drain(stderr);
        let mut reader = BufReader::new(stdout).lines();

        let mut turns: u32 = 0;
        let mut final_message = String::new();
        let mut extracted: Vec<ExtractedAgentResult> = Vec::new();
        let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut cost_usd_micros: i64 = 0;
        let mut result_error: Option<String> = None;

        let read_loop = async {
            while let Some(line) = reader
                .next_line()
                .await
                .map_err(|e| AiError::Transport(format!("read stdout: {e}")))?
            {
                if line.trim().is_empty() {
                    continue;
                }
                let Some(event) = parse_stream_json(&line) else {
                    continue;
                };
                match event {
                    ClaudeEvent::Assistant(msg) => {
                        turns = turns.saturating_add(1);
                        for block in msg.content {
                            match block {
                                ContentBlock::Text { text } => {
                                    final_message.push_str(&text);
                                    let _ = sink.send(AgentEvent::Ai {
                                        data: AiEvent::TokenReceived {
                                            task_id: task.task_id.clone(),
                                            token: text,
                                        },
                                    });
                                }
                                ContentBlock::ToolUse { name, input, .. } => {
                                    let _ = sink.send(AgentEvent::Ai {
                                        data: AiEvent::ToolCallStarted {
                                            task_id: task.task_id.clone(),
                                            name: name.clone(),
                                        },
                                    });
                                    if let Some(result) = classify_tool_use(&name, &input) {
                                        extracted.push(result);
                                    }
                                    let _ = sink.send(AgentEvent::Ai {
                                        data: AiEvent::ToolCallFinished {
                                            task_id: task.task_id.clone(),
                                            name,
                                            ok: true,
                                        },
                                    });
                                }
                                ContentBlock::Other => {}
                            }
                        }
                    }
                    ClaudeEvent::Result(r) => {
                        if r.is_error == Some(true) {
                            result_error = Some(format_result_error(&r.subtype, &r.result));
                        }
                        if let Some(u) = r.usage {
                            usage.input_tokens =
                                usage.input_tokens.saturating_add(u.input_tokens.unwrap_or(0));
                            usage.output_tokens =
                                usage.output_tokens.saturating_add(u.output_tokens.unwrap_or(0));
                        }
                        if let Some(c) = r.total_cost_usd {
                            cost_usd_micros = (c * 1_000_000.0).round() as i64;
                        }
                        if let Some(text) = r.result {
                            if final_message.is_empty() {
                                final_message = text;
                            }
                        }
                    }
                    ClaudeEvent::Other => {}
                }
            }
            Ok::<(), AiError>(())
        };

        match tokio::time::timeout(self.timeout, read_loop).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = child.kill().await;
                let stderr_text = await_stderr_drain(stderr_handle).await;
                return Err(annotate_with_stderr(e, &stderr_text));
            }
            Err(_) => {
                let _ = child.kill().await;
                let _ = sink.send(AgentEvent::Ai {
                    data: AiEvent::TaskHalted {
                        task_id: task.task_id.clone(),
                        reason: HaltReason::OperatorCancelled,
                    },
                });
                let stderr_text = await_stderr_drain(stderr_handle).await;
                let base = format!("claude agent_loop timed out after {}s", self.timeout.as_secs());
                return Err(AiError::Transport(append_stderr(&base, &stderr_text)));
            }
        }

        let status =
            child.wait().await.map_err(|e| AiError::Transport(format!("wait claude: {e}")))?;
        if !status.success() {
            let stderr_text = await_stderr_drain(stderr_handle).await;
            let base = format!("claude exited {status}");
            let base = append_result_error(&base, &result_error);
            return Err(AiError::UpstreamRefused(append_stderr(&base, &stderr_text)));
        }
        // Success: detach the drain. The child has exited so the pipe is
        // closed; the task completes naturally and its buffer is dropped.
        drop(stderr_handle);

        let spent_after =
            self.tracker.add_spend(&budget.run_id, budget.kind, cost_usd_micros).await?;
        let _ = sink.send(AgentEvent::Ai {
            data: AiEvent::BudgetTick {
                task_id: task.task_id.clone(),
                run_id: budget.run_id.clone(),
                spent_usd_micros: spent_after,
            },
        });

        if let Some(cap) = self.tracker.cap(&budget.run_id, budget.kind).await? {
            if spent_after > cap {
                let _ = sink.send(AgentEvent::Ai {
                    data: AiEvent::TaskHalted {
                        task_id: task.task_id.clone(),
                        reason: HaltReason::BudgetCapReached,
                    },
                });
                return Err(AiError::BudgetExceeded {
                    cap_usd_micros: cap,
                    spent_usd_micros: spent_after,
                });
            }
        }

        extracted.extend(extract_tool_markers_from_text(&final_message));

        Ok(AgentResult {
            prompt_version: task.prompt_version,
            task_id: task.task_id,
            model,
            final_message,
            turns,
            usage,
            cache: None,
            cost_usd_micros,
            extracted,
        })
    }

    fn cost_estimate(&self, _prompt: &Prompt) -> Option<CostEstimate> {
        None
    }
}

fn effort_args(effort: Option<&str>) -> Vec<&str> {
    match effort.map(str::trim).filter(|v| !v.is_empty()) {
        Some(effort) => vec!["--effort", effort],
        None => Vec::new(),
    }
}

fn claude_model_for_context(model: &str, context_window: Option<u32>) -> String {
    let trimmed = model.trim();
    let without_suffix = trimmed.strip_suffix("[1m]").unwrap_or(trimmed);
    match context_window {
        Some(window) if window >= 1_000_000 => format!("{without_suffix}[1m]"),
        Some(_) => without_suffix.to_string(),
        None => trimmed.to_string(),
    }
}

fn spawn_stderr_drain(mut stderr: ChildStderr) -> JoinHandle<Vec<u8>> {
    tokio::spawn(async move {
        let mut buf: Vec<u8> = Vec::with_capacity(1024);
        let mut chunk = [0u8; 4096];
        loop {
            match stderr.read(&mut chunk).await {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.len() > MAX_STDERR_CAPTURE_BYTES {
                        let drop_n = buf.len() - MAX_STDERR_CAPTURE_BYTES;
                        buf.drain(..drop_n);
                    }
                }
                Err(_) => break,
            }
        }
        buf
    })
}

async fn await_stderr_drain(handle: JoinHandle<Vec<u8>>) -> String {
    let bytes = match tokio::time::timeout(Duration::from_secs(1), handle).await {
        Ok(Ok(b)) => b,
        _ => return String::new(),
    };
    String::from_utf8_lossy(&bytes).trim().to_string()
}

fn append_stderr(base: &str, stderr_text: &str) -> String {
    if stderr_text.is_empty() {
        base.to_string()
    } else {
        format!("{base}: stderr: {stderr_text}")
    }
}

/// Build a human-readable reason from a `stream-json` error result. In
/// `--print` mode claude reports API failures (e.g. a too-long prompt)
/// as a result event with `is_error: true` on stdout, not on stderr; we
/// surface its subtype and any message so the exit-1 error is not
/// opaque.
fn format_result_error(subtype: &Option<String>, result: &Option<String>) -> String {
    let subtype = subtype.as_deref().unwrap_or("error");
    match result.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(text) => {
            let mut text = text.to_string();
            if text.len() > MAX_STDERR_CAPTURE_BYTES {
                text.truncate(MAX_STDERR_CAPTURE_BYTES);
            }
            format!("result({subtype}): {text}")
        }
        None => format!("result({subtype})"),
    }
}

fn append_result_error(base: &str, result_error: &Option<String>) -> String {
    match result_error {
        Some(reason) if !reason.is_empty() => format!("{base}: {reason}"),
        _ => base.to_string(),
    }
}

fn annotate_with_stderr(err: AiError, stderr_text: &str) -> AiError {
    if stderr_text.is_empty() {
        return err;
    }
    match err {
        AiError::Transport(msg) => AiError::Transport(append_stderr(&msg, stderr_text)),
        AiError::UpstreamRefused(msg) => AiError::UpstreamRefused(append_stderr(&msg, stderr_text)),
        other => other,
    }
}

fn render_task_markdown(task: &AgentTask) -> String {
    let tools_block = if task.tools.is_empty() {
        "(none; answer from context)".to_string()
    } else {
        task.tools.iter().map(|t| format!("- {t}")).collect::<Vec<_>>().join("\n")
    };
    let working_directory =
        task.working_directory.as_deref().unwrap_or("(not set; adapter process cwd will be used)");
    format!(
        "# Agent task\n\
         \n\
         **prompt_version**: `{pv}`  \n\
         **task_id**: `{tid}`  \n\
         **working_directory**: `{working_directory}`  \n\
         **max_turns**: {max_turns}\n\
         \n\
         ## System\n{system}\n\
         \n\
         ## Objective\n{objective}\n\
         \n\
         ## Tools available\n{tools}\n\
         \n\
         When you need to record a structured Nyx Agent artifact, emit a JSON object on its own line \
         using this shape: {{\"tool\":\"<listed tool name>\",\"input\":{{...}}}}. Use one of the \
         listed `record_*` tool names as the `tool` value and place its arguments in `input`.\n",
        pv = task.prompt_version,
        tid = task.task_id,
        working_directory = working_directory,
        max_turns = task.max_turns,
        system = task.system,
        objective = task.objective,
        tools = tools_block,
    )
}

fn extract_tool_markers_from_text(text: &str) -> Vec<ExtractedAgentResult> {
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('{') {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        extract_tool_marker_value(&value, &mut out);
    }
    out
}

fn extract_tool_marker_value(value: &serde_json::Value, out: &mut Vec<ExtractedAgentResult>) {
    if let Some(calls) = value.get("tool_calls").and_then(|v| v.as_array()) {
        for call in calls {
            extract_tool_marker_value(call, out);
        }
        return;
    }
    let Some(name) = value.get("tool").or_else(|| value.get("name")).and_then(|v| v.as_str())
    else {
        return;
    };
    let input = value.get("input").cloned().unwrap_or_else(|| serde_json::json!({}));
    if let Some(result) = classify_tool_use(name, &input) {
        out.push(result);
    }
}

fn render_one_shot_prompt(prompt: &Prompt) -> String {
    format!(
        "# Nyx Agent one-shot task\n\
         \n\
         **prompt_version**: `{pv}`  \n\
         **task_id**: `{tid}`  \n\
         **max_output_tokens**: {max_tokens}  \n\
         **temperature**: {temperature}\n\
         \n\
         Return the requested content only. Preserve any JSON contract from the system prompt; \
         do not add explanation, Markdown fences, or surrounding prose unless the prompt explicitly \
         asks for them.\n\
         \n\
         ## System\n{system}\n\
         \n\
         ## User\n{user}\n",
        pv = prompt.prompt_version,
        tid = prompt.task_id,
        max_tokens = prompt.max_output_tokens,
        temperature = prompt.temperature,
        system = prompt.system,
        user = prompt.user,
    )
}

fn effective_cap(tracker_cap: Option<i64>, envelope_cap: i64) -> i64 {
    match tracker_cap {
        Some(t) => t.min(envelope_cap),
        None => envelope_cap,
    }
}

/// Parse one NDJSON line from `claude --output-format stream-json` into
/// a typed event. Unknown shapes return `None` so callers can skip
/// gracefully across Claude Code CLI revisions.
pub fn parse_stream_json(line: &str) -> Option<ClaudeEvent> {
    let raw: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    let kind = raw.get("type")?.as_str()?;
    match kind {
        "assistant" => {
            let msg = raw.get("message")?;
            let content_raw = msg.get("content").cloned().unwrap_or(serde_json::json!([]));
            let content: Vec<ContentBlock> = serde_json::from_value(content_raw).ok()?;
            Some(ClaudeEvent::Assistant(AssistantMessage { content }))
        }
        "result" => {
            let parsed: ResultPayload = serde_json::from_value(raw).ok()?;
            Some(ClaudeEvent::Result(parsed))
        }
        _ => Some(ClaudeEvent::Other),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClaudeEvent {
    Assistant(AssistantMessage),
    Result(ResultPayload),
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        #[serde(default)]
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ResultPayload {
    #[serde(default)]
    pub result: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub total_cost_usd: Option<f64>,
    #[serde(default)]
    pub usage: Option<ResultUsage>,
    /// `true` when claude failed (e.g. an API 400 such as a too-long
    /// prompt). In `stream-json` mode this error lands here on stdout,
    /// not stderr, so the failure path must read it to report a reason.
    #[serde(default)]
    pub is_error: Option<bool>,
    /// e.g. `"success"`, `"error_during_execution"`,
    /// `"error_max_turns"`.
    #[serde(default)]
    pub subtype: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ResultUsage {
    #[serde(default)]
    pub input_tokens: Option<u32>,
    #[serde(default)]
    pub output_tokens: Option<u32>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::runtime::InMemoryBudgetTracker;
    use nyx_agent_types::agent::BudgetKind;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use tokio::sync::broadcast;

    fn sample_task() -> AgentTask {
        AgentTask {
            prompt_version: "phase13.test.v1".to_string(),
            task_id: "task-claude-1".to_string(),
            system: "you are a static analysis exploration agent".to_string(),
            objective: "enumerate sinks in fixture.py".to_string(),
            tools: vec!["fs.read".to_string(), "record_payload".to_string()],
            working_directory: None,
            max_turns: 3,
        }
    }

    fn budget(cap_usd_micros: i64) -> Budget {
        Budget { run_id: "run-claude-1".to_string(), kind: BudgetKind::AgentLoop, cap_usd_micros }
    }

    fn one_shot_budget(cap_usd_micros: i64) -> Budget {
        Budget { run_id: "run-claude-1".to_string(), kind: BudgetKind::OneShot, cap_usd_micros }
    }

    fn fake_binary() -> ClaudeBinary {
        ClaudeBinary { path: PathBuf::from("/usr/bin/false"), version: "0.0.0-test".to_string() }
    }

    fn fake_cli_script(body: &str) -> (tempfile::TempDir, ClaudeBinary) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude");
        std::fs::write(&path, body).expect("write fake claude");
        #[cfg(unix)]
        {
            let mut perms = std::fs::metadata(&path).expect("metadata").permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&path, perms).expect("chmod");
        }
        let binary = ClaudeBinary { path, version: "2.1.146-test".to_string() };
        (dir, binary)
    }

    #[test]
    fn capability_flags_match_phase13_contract() {
        let tracker = Arc::new(InMemoryBudgetTracker::new()) as SharedBudgetTracker;
        let adapter = ClaudeCodeAdapter::new(fake_binary(), tracker);
        assert_eq!(adapter.name(), "claude-code");
        assert!(adapter.supports_agent_loop());
        assert!(adapter.supports_prompt_cache());
        assert!(!adapter.supports_deterministic_sampling());
    }

    #[tokio::test]
    async fn one_shot_parses_stream_json_and_records_budget() {
        let script = r#"#!/bin/sh
cat >/dev/null
printf '%s\n' '{"type":"assistant","message":{"content":[{"type":"text","text":"{\"ok\":true}"}]}}'
printf '%s\n' '{"type":"result","result":"{\"ok\":true}","model":"claude-sonnet-4-6","total_cost_usd":0.001234,"usage":{"input_tokens":10,"output_tokens":3,"cache_creation_input_tokens":4,"cache_read_input_tokens":5}}'
"#;
        let (_dir, binary) = fake_cli_script(script);
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        let adapter = ClaudeCodeAdapter::new(binary, tracker.clone() as SharedBudgetTracker);
        let (tx, _rx) = broadcast::channel::<AgentEvent>(4);
        let prompt = Prompt {
            prompt_version: "v1".to_string(),
            task_id: "t".to_string(),
            model: None,
            system: "s".to_string(),
            user: "u".to_string(),
            max_output_tokens: 8,
            temperature: 0.0,
            seed: None,
        };
        let resp = adapter.one_shot(prompt, one_shot_budget(10_000), tx).await.expect("one_shot");
        assert_eq!(resp.content, "{\"ok\":true}");
        assert_eq!(resp.model, "claude-sonnet-4-6");
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 3);
        assert_eq!(resp.cache.unwrap().cache_creation_tokens, 4);
        assert_eq!(resp.cost_usd_micros, 1_234);
        assert_eq!(tracker.spent("run-claude-1", BudgetKind::OneShot), 1_234);
    }

    #[tokio::test]
    async fn one_shot_forwards_model_effort_and_context_to_cli() {
        let script = r#"#!/bin/sh
found_model=0
found_effort=0
prev=
for arg in "$@"; do
  if [ "$prev" = "--model" ] && [ "$arg" = "opus[1m]" ]; then found_model=1; fi
  if [ "$prev" = "--effort" ] && [ "$arg" = "xhigh" ]; then found_effort=1; fi
  prev="$arg"
done
[ "$found_model" = 1 ] || exit 7
[ "$found_effort" = 1 ] || exit 8
cat >/dev/null
printf '%s\n' '{"type":"result","result":"done","usage":{"input_tokens":1,"output_tokens":1}}'
"#;
        let (_dir, binary) = fake_cli_script(script);
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        let adapter = ClaudeCodeAdapter::new(binary, tracker.clone() as SharedBudgetTracker)
            .with_default_model("opus")
            .with_effort("xhigh")
            .with_context_window(1_000_000);
        let (tx, _rx) = broadcast::channel::<AgentEvent>(4);
        let prompt = Prompt {
            prompt_version: "v1".to_string(),
            task_id: "t".to_string(),
            model: None,
            system: "s".to_string(),
            user: "u".to_string(),
            max_output_tokens: 8,
            temperature: 0.0,
            seed: None,
        };

        let resp = adapter.one_shot(prompt, one_shot_budget(10_000), tx).await.expect("one_shot");

        assert_eq!(resp.model, "opus[1m]");
        assert_eq!(resp.content, "done");
    }

    #[tokio::test]
    async fn one_shot_enforces_post_call_budget_cap() {
        let script = r#"#!/bin/sh
cat >/dev/null
printf '%s\n' '{"type":"result","result":"done","total_cost_usd":0.0002,"usage":{"input_tokens":1,"output_tokens":1}}'
"#;
        let (_dir, binary) = fake_cli_script(script);
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        let adapter = ClaudeCodeAdapter::new(binary, tracker.clone() as SharedBudgetTracker);
        let (tx, _rx) = broadcast::channel::<AgentEvent>(4);
        let prompt = Prompt {
            prompt_version: "v1".to_string(),
            task_id: "t".to_string(),
            model: None,
            system: "s".to_string(),
            user: "u".to_string(),
            max_output_tokens: 8,
            temperature: 0.0,
            seed: None,
        };
        let err = adapter
            .one_shot(prompt, one_shot_budget(100), tx)
            .await
            .expect_err("budget should trip");
        assert!(matches!(
            err,
            AiError::BudgetExceeded { cap_usd_micros: 100, spent_usd_micros: 200 }
        ));
        assert_eq!(tracker.spent("run-claude-1", BudgetKind::OneShot), 200);
    }

    #[test]
    fn parse_claude_version_accepts_bare_semver() {
        let v = parse_claude_version("1.2.3").expect("bare semver parses");
        assert_eq!(v, Version::new(1, 2, 3));
    }

    #[test]
    fn parse_claude_version_strips_claude_code_suffix() {
        let v = parse_claude_version("2.0.5 (Claude Code)").expect("suffixed semver parses");
        assert_eq!(v, Version::new(2, 0, 5));
    }

    #[test]
    fn parse_claude_version_rejects_non_semver_output() {
        assert!(parse_claude_version("").is_none());
        assert!(parse_claude_version("not-a-version").is_none());
        assert!(parse_claude_version("v1.2.3").is_none());
    }

    #[test]
    fn minimum_claude_version_constant_parses() {
        Version::parse(MINIMUM_CLAUDE_VERSION).expect("MINIMUM_CLAUDE_VERSION literal parses");
    }

    #[tokio::test]
    async fn detect_returns_adapter_unavailable_when_binary_missing() {
        // Clear PATH so `which` fails deterministically.
        let prior = std::env::var_os("PATH");
        // SAFETY: tests run single-threaded under tokio::test; the env
        // mutation is reset before the test returns.
        unsafe {
            std::env::set_var("PATH", "");
        }
        let result = detect_claude_binary().await;
        if let Some(p) = prior {
            unsafe {
                std::env::set_var("PATH", p);
            }
        } else {
            unsafe {
                std::env::remove_var("PATH");
            }
        }
        assert!(matches!(result, Err(AiError::AdapterUnavailable(_))));
    }

    #[test]
    fn parse_stream_json_lifts_text_blocks() {
        let line = serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [
                    { "type": "text", "text": "scanning fixture.py" }
                ]
            }
        })
        .to_string();
        let ev = parse_stream_json(&line).expect("parsed");
        match ev {
            ClaudeEvent::Assistant(a) => {
                assert_eq!(a.content.len(), 1);
                assert!(matches!(
                    a.content[0],
                    ContentBlock::Text { ref text } if text == "scanning fixture.py"
                ));
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn parse_stream_json_lifts_tool_use_block() {
        let line = serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "record_payload",
                        "input": { "rule_id": "py.cmdi.os_system", "body": "ls -la; #" }
                    }
                ]
            }
        })
        .to_string();
        let ev = parse_stream_json(&line).expect("parsed");
        let ClaudeEvent::Assistant(a) = ev else { panic!("expected Assistant") };
        let block = a.content.into_iter().next().expect("one block");
        let ContentBlock::ToolUse { name, input, .. } = block else { panic!("expected ToolUse") };
        assert_eq!(name, "record_payload");
        let extracted = classify_tool_use(&name, &input).expect("extract");
        assert!(matches!(
            extracted,
            ExtractedAgentResult::PayloadFound { ref rule_id, ref body }
                if rule_id == "py.cmdi.os_system" && body == "ls -la; #"
        ));
    }

    #[test]
    fn parse_stream_json_lifts_result_event() {
        let line = serde_json::json!({
            "type": "result",
            "subtype": "success",
            "result": "done",
            "total_cost_usd": 0.0125,
            "usage": { "input_tokens": 100, "output_tokens": 50 },
            "model": "claude-opus-4-7"
        })
        .to_string();
        let ev = parse_stream_json(&line).expect("parsed");
        let ClaudeEvent::Result(r) = ev else { panic!("expected Result") };
        assert_eq!(r.result.as_deref(), Some("done"));
        assert_eq!(r.model.as_deref(), Some("claude-opus-4-7"));
        assert!((r.total_cost_usd.unwrap() - 0.0125).abs() < f64::EPSILON);
        assert_eq!(r.usage.unwrap().input_tokens, Some(100));
    }

    #[test]
    fn classify_tool_use_falls_back_to_exploration_event() {
        let input = serde_json::json!({ "path": "src/main.rs" });
        let ev = classify_tool_use("fs.read", &input).expect("extracted");
        match ev {
            ExtractedAgentResult::ExplorationEvent { message } => {
                assert!(message.starts_with("tool fs.read"));
            }
            other => panic!("expected ExplorationEvent, got {other:?}"),
        }
    }

    #[test]
    fn classify_tool_use_records_spec_and_chains() {
        let spec_input = serde_json::json!({
            "capability": "fs.write",
            "spec": "writes to /tmp"
        });
        let ev = classify_tool_use("record_spec", &spec_input).expect("extracted");
        assert!(matches!(
            ev,
            ExtractedAgentResult::SpecFound { ref capability, ref spec }
                if capability == "fs.write" && spec == "writes to /tmp"
        ));

        let chain_input = serde_json::json!({
            "chain_ids": ["c1", "c2"],
            "rationale": "shorter sink reachability"
        });
        let ev = classify_tool_use("record_chains", &chain_input).expect("extracted");
        assert!(matches!(
            ev,
            ExtractedAgentResult::ChainsRanked { ref chain_ids, .. }
                if chain_ids == &vec!["c1".to_string(), "c2".to_string()]
        ));
    }

    #[test]
    fn classify_tool_use_records_exploration_finding() {
        let input = serde_json::json!({
            "path": "<api:/api/admin/orders>",
            "line": 42,
            "cap": "AUTH_BYPASS",
            "rationale": "GET admin endpoint accepts unauthenticated requests",
            "endpoint": "GET /api/admin/orders",
            "suggested_payload_hint": "curl -i http://127.0.0.1:3000/api/admin/orders",
        });
        let ev = classify_tool_use("record_exploration_finding", &input).expect("extracted");
        assert!(matches!(
            ev,
            ExtractedAgentResult::ExplorationFinding {
                ref path, line: Some(42), ref cap, ref endpoint, ..
            } if path == "<api:/api/admin/orders>"
                && cap == "AUTH_BYPASS"
                && endpoint.as_deref() == Some("GET /api/admin/orders")
        ));
        // Empty required fields fall back to None (gating against
        // malformed tool-use blocks).
        let bad = serde_json::json!({ "path": "", "cap": "X", "rationale": "y" });
        assert!(classify_tool_use("record_exploration_finding", &bad).is_none());
    }

    #[test]
    fn render_task_markdown_includes_all_fields() {
        let md = render_task_markdown(&sample_task());
        assert!(md.contains("phase13.test.v1"));
        assert!(md.contains("task-claude-1"));
        assert!(md.contains("enumerate sinks"));
        assert!(md.contains("- fs.read"));
        assert!(md.contains("- record_payload"));
    }

    #[test]
    fn append_stderr_skips_when_empty() {
        assert_eq!(append_stderr("base", ""), "base");
        assert_eq!(append_stderr("base", "boom"), "base: stderr: boom");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn agent_loop_surfaces_stderr_on_nonzero_exit() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let script = dir.path().join("fake_claude");
        let body = "#!/bin/sh\n\
                    cat > /dev/null\n\
                    printf 'API authentication failed: invalid bearer token\\n' >&2\n\
                    exit 7\n";
        tokio::fs::write(&script, body).await.expect("write script");
        let mut perms = tokio::fs::metadata(&script).await.expect("meta").permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(&script, perms).await.expect("perms");

        let tracker = Arc::new(InMemoryBudgetTracker::new()) as SharedBudgetTracker;
        let adapter = ClaudeCodeAdapter::new(
            ClaudeBinary { path: script, version: "test".to_string() },
            tracker,
        )
        .with_timeout(Duration::from_secs(10));
        let (tx, _rx) = broadcast::channel::<AgentEvent>(4);
        let err = adapter
            .agent_loop(sample_task(), budget(1_000_000), tx)
            .await
            .expect_err("expected upstream refused");
        match err {
            AiError::UpstreamRefused(msg) => {
                assert!(msg.starts_with("claude exited"), "missing exit prefix: {msg}");
                assert!(msg.contains("API authentication failed"), "stderr missing: {msg}");
                assert!(msg.contains("invalid bearer token"), "trailing stderr trimmed: {msg}");
            }
            other => panic!("expected UpstreamRefused, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn agent_loop_surfaces_stderr_on_timeout() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let script = dir.path().join("fake_claude_slow");
        // Write to stderr before the stall so the drain task captures
        // a line for sure before the adapter timeout fires. `exec sleep`
        // avoids leaving an extra shell waiting on the sleep child.
        let body = "#!/bin/sh\n\
                    printf 'pre-stall diagnostic line\\n' >&2\n\
                    exec sleep 30\n";
        tokio::fs::write(&script, body).await.expect("write script");
        let mut perms = tokio::fs::metadata(&script).await.expect("meta").permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(&script, perms).await.expect("perms");

        let tracker = Arc::new(InMemoryBudgetTracker::new()) as SharedBudgetTracker;
        let adapter = ClaudeCodeAdapter::new(
            ClaudeBinary { path: script, version: "test".to_string() },
            tracker,
        )
        .with_timeout(Duration::from_millis(2_000));
        let (tx, _rx) = broadcast::channel::<AgentEvent>(4);
        let err = adapter
            .agent_loop(sample_task(), budget(1_000_000), tx)
            .await
            .expect_err("expected timeout");
        match err {
            AiError::Transport(msg) => {
                assert!(msg.contains("timed out"), "missing timeout prefix: {msg}");
                assert!(msg.contains("pre-stall diagnostic line"), "stderr missing: {msg}");
            }
            other => panic!("expected Transport timeout, got {other:?}"),
        }
    }
}
