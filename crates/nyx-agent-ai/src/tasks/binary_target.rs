//! Binary / CLI target pentest task.
//!
//! Drives an AI agent against an operator-pinned local binary, hunting
//! crashes / memory-safety trips / input-validation failures / path
//! traversal / DoS by crafting malformed inputs and running the target
//! against them inside `nyx-agent-sandbox`.
//!
//! Transport: **Option B** from `docs/binary-target-pentest.md` §6 — a
//! host-orchestrated deterministic micro-loop. The agent never runs the
//! target itself and has no host shell. Each turn the model emits one
//! JSON tool action (`sandbox.write_file` / `sandbox.exec` /
//! `record_binary_finding` / `stop`); the host executes it via the
//! [`SandboxExecutor`] and appends a redacted result to the running
//! transcript. This reuses the existing `one_shot` plumbing and needs no
//! new tool-callback transport; Option A (MCP/agent-loop tool results)
//! can replace it later without changing the runner, types, or this
//! task's external shape.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use nyx_agent_sandbox::{BinaryExecRequest, BinaryExecResult, BinaryRunner, BinaryRunnerError};
use nyx_agent_types::agent::{
    classify_tool_use, AiError, Budget, BudgetKind, ExtractedAgentResult, Prompt,
};
use nyx_agent_types::event::EventSink;
use nyx_agent_types::target::LocalBinaryTarget;
use serde::Serialize;
use serde_json::Value;
use std::time::Duration;

pub const BINARY_TARGET_PROMPT_VERSION: &str = "phase-pre-mvp.binary-target.v1";
pub const DEFAULT_BINARY_TARGET_MAX_TURNS: u32 = 40;
/// Hard ceiling on output tokens per micro-loop turn.
const TURN_MAX_OUTPUT_TOKENS: u32 = 2048;

/// Host-side executor the micro-loop drives. Abstracted so the task can
/// stay free of a concrete backend and be unit-tested with a fake. The
/// production impl is [`BinaryRunner`] (below).
#[async_trait]
pub trait SandboxExecutor: Send + Sync {
    /// Run one contained exec of the target, returning the redacted
    /// result or a host-side error string (surfaced to the model).
    async fn exec(
        &self,
        workspace: &Path,
        req: BinaryExecRequest,
    ) -> Result<BinaryExecResult, String>;
}

#[async_trait]
impl SandboxExecutor for BinaryRunner {
    async fn exec(
        &self,
        workspace: &Path,
        req: BinaryExecRequest,
    ) -> Result<BinaryExecResult, String> {
        BinaryRunner::exec(self, workspace, req).await.map_err(|e: BinaryRunnerError| e.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct BinaryTargetLead {
    pub source: String,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct BinaryTargetScope {
    pub run_id: String,
    pub project_id: String,
    pub task_id: String,
    pub target: LocalBinaryTarget,
    /// Sandbox workspace for this run (inputs staged here, target cwd).
    pub workspace_root: String,
    /// Where contained crash repros / evidence land.
    pub artifact_dir: String,
    pub known_leads: Vec<BinaryTargetLead>,
    pub max_turns: u32,
    pub run_cap_usd_micros: Option<i64>,
}

impl BinaryTargetScope {
    pub fn new(
        run_id: impl Into<String>,
        project_id: impl Into<String>,
        target: LocalBinaryTarget,
    ) -> Self {
        let run_id = run_id.into();
        Self {
            task_id: format!("binary-target-{run_id}"),
            run_id,
            project_id: project_id.into(),
            target,
            workspace_root: String::new(),
            artifact_dir: String::new(),
            known_leads: Vec::new(),
            max_turns: DEFAULT_BINARY_TARGET_MAX_TURNS,
            run_cap_usd_micros: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BinaryFinding {
    pub title: String,
    pub vuln_class: String,
    pub severity: String,
    pub confidence: u8,
    pub affected_components: Vec<serde_json::Value>,
    pub business_impact: String,
    pub evidence_summary: String,
    pub repro_steps: String,
    pub remediation: String,
    pub proof_artifact_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BinaryTargetAuditEntry {
    pub action: String,
    pub summary: String,
}

#[derive(Debug)]
pub struct BinaryTargetOutcome {
    pub findings: Vec<BinaryFinding>,
    pub audit: Vec<BinaryTargetAuditEntry>,
    pub exec_count: u32,
    pub turns: u32,
    pub spent_usd_micros: i64,
    pub prompt_version: String,
    pub halted: bool,
}

/// Why the micro-loop ended early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopEnd {
    Stopped,
    BudgetExhausted,
    TurnsExhausted,
    ParseGaveUp,
}

use crate::runtime::AiRuntime;

pub async fn run<R: AiRuntime + ?Sized, E: SandboxExecutor + ?Sized>(
    runtime: &R,
    scope: &BinaryTargetScope,
    executor: &E,
    sink: EventSink,
) -> Result<BinaryTargetOutcome, AiError> {
    // Resolve and pin the program path host-side before the first exec
    // (§8.5: the agent never picks the executable).
    let resolved_program = resolve_program(&scope.target.program)
        .map_err(|e| AiError::AdapterUnavailable(format!("binary target program resolve failed: {e}")))?;

    let workspace = PathBuf::from(&scope.workspace_root);
    if let Err(e) = std::fs::create_dir_all(&workspace) {
        return Err(AiError::AdapterUnavailable(format!("create workspace {}: {e}", workspace.display())));
    }
    if !scope.artifact_dir.is_empty() {
        let _ = std::fs::create_dir_all(&scope.artifact_dir);
    }

    let system = include_str!("../prompts/binary_target.v1.system.md").to_string();
    let objective = build_objective(scope, &resolved_program);

    let mut state = LoopState::new(scope, executor, &workspace, resolved_program);
    let mut transcript = String::new();
    let mut end = LoopEnd::TurnsExhausted;

    for turn in 0..scope.max_turns {
        // Budget gate before each model round-trip.
        if let Some(cap) = scope.run_cap_usd_micros {
            if state.spent_usd_micros >= cap {
                end = LoopEnd::BudgetExhausted;
                break;
            }
        }

        let user = if transcript.is_empty() {
            objective.clone()
        } else {
            format!("{objective}\n\nTRANSCRIPT (most recent last)\n{transcript}")
        };
        let prompt = Prompt {
            prompt_version: BINARY_TARGET_PROMPT_VERSION.to_string(),
            task_id: scope.task_id.clone(),
            model: None,
            system: system.clone(),
            user,
            max_output_tokens: TURN_MAX_OUTPUT_TOKENS,
            temperature: 0.0,
            seed: None,
        };

        let budget = Budget {
            run_id: scope.run_id.clone(),
            kind: BudgetKind::OneShot,
            cap_usd_micros: scope.run_cap_usd_micros.unwrap_or(i64::MAX),
        };
        let response = runtime.one_shot(prompt, budget, sink.clone()).await?;
        state.spent_usd_micros += response.cost_usd_micros;
        state.turns = turn + 1;

        let Some(action) = first_tool_action(&response.content) else {
            // Model produced no parsable action. One nudge, then give up
            // to avoid burning the whole budget on malformed turns.
            transcript.push_str(&format!(
                "\nMODEL (turn {}):\n{}\n[host: no JSON tool action found; reply with exactly one tool object]\n",
                turn + 1,
                truncate(&response.content, 600),
            ));
            if state.consecutive_parse_failures >= 2 {
                end = LoopEnd::ParseGaveUp;
                break;
            }
            state.consecutive_parse_failures += 1;
            continue;
        };
        state.consecutive_parse_failures = 0;

        let (tool, input) = action;
        transcript.push_str(&format!(
            "\nMODEL (turn {}): {}\n",
            turn + 1,
            serde_json::json!({"tool": tool, "input": input})
        ));

        match tool.as_str() {
            "sandbox.write_file" => {
                let result = state.handle_write_file(&input);
                transcript.push_str(&format!("HOST RESULT: {result}\n"));
            }
            "sandbox.exec" => {
                let result = state.handle_exec(&input).await;
                transcript.push_str(&format!("HOST RESULT: {result}\n"));
            }
            "record_binary_finding" => {
                let ok = state.handle_record(&input);
                transcript.push_str(&format!(
                    "HOST RESULT: {}\n",
                    if ok {
                        serde_json::json!({"recorded": true})
                    } else {
                        serde_json::json!({"recorded": false, "error": "missing required finding fields"})
                    }
                ));
            }
            "stop" => {
                state.audit.push(BinaryTargetAuditEntry {
                    action: "stop".to_string(),
                    summary: input
                        .get("reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                });
                end = LoopEnd::Stopped;
                break;
            }
            other => {
                transcript.push_str(&format!(
                    "HOST RESULT: {}\n",
                    serde_json::json!({"error": format!("unknown tool {other}")})
                ));
            }
        }
    }

    Ok(BinaryTargetOutcome {
        findings: state.findings,
        audit: state.audit,
        exec_count: state.exec_count,
        turns: state.turns,
        spent_usd_micros: state.spent_usd_micros,
        prompt_version: BINARY_TARGET_PROMPT_VERSION.to_string(),
        halted: matches!(end, LoopEnd::BudgetExhausted | LoopEnd::ParseGaveUp),
    })
}

/// Per-run mutable state threaded through the micro-loop.
struct LoopState<'a, E: SandboxExecutor + ?Sized> {
    scope: &'a BinaryTargetScope,
    executor: &'a E,
    workspace: &'a Path,
    resolved_program: String,
    /// Staged input files, logical-name -> (workspace-rel path, bytes).
    staged: Vec<(String, PathBuf, Vec<u8>)>,
    findings: Vec<BinaryFinding>,
    audit: Vec<BinaryTargetAuditEntry>,
    exec_count: u32,
    turns: u32,
    spent_usd_micros: i64,
    consecutive_parse_failures: u32,
}

impl<'a, E: SandboxExecutor + ?Sized> LoopState<'a, E> {
    fn new(
        scope: &'a BinaryTargetScope,
        executor: &'a E,
        workspace: &'a Path,
        resolved_program: String,
    ) -> Self {
        Self {
            scope,
            executor,
            workspace,
            resolved_program,
            staged: Vec::new(),
            findings: Vec::new(),
            audit: Vec::new(),
            exec_count: 0,
            turns: 0,
            spent_usd_micros: 0,
            consecutive_parse_failures: 0,
        }
    }

    fn handle_write_file(&mut self, input: &Value) -> Value {
        let name = match input.get("name").and_then(|v| v.as_str()).map(str::trim) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => return serde_json::json!({"error": "write_file needs a non-empty name"}),
        };
        let bytes = if let Some(b64) = input.get("content_base64").and_then(|v| v.as_str()) {
            match base64_decode(b64) {
                Ok(b) => b,
                Err(e) => return serde_json::json!({"error": format!("bad base64: {e}")}),
            }
        } else if let Some(text) = input.get("content_text").and_then(|v| v.as_str()) {
            text.as_bytes().to_vec()
        } else {
            return serde_json::json!({"error": "exactly one of content_base64/content_text required"});
        };
        // Workspace-relative path = sanitised basename of the logical
        // name; the agent never picks a directory.
        let rel = PathBuf::from(sanitize_name(&name));
        // Replace any prior staging under the same logical name.
        self.staged.retain(|(n, _, _)| n != &name);
        let bytes_len = bytes.len();
        self.staged.push((name.clone(), rel.clone(), bytes));
        self.audit.push(BinaryTargetAuditEntry {
            action: "sandbox.write_file".to_string(),
            summary: format!("staged {name} ({bytes_len} bytes)"),
        });
        serde_json::json!({"path": rel.to_string_lossy(), "bytes": bytes_len})
    }

    async fn handle_exec(&mut self, input: &Value) -> Value {
        let model_args: Vec<String> = input
            .get("args")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default();

        let argv = match self.build_argv(&model_args) {
            Ok(a) => a,
            Err(e) => return serde_json::json!({"error": e}),
        };

        let stdin = input
            .get("stdin_base64")
            .and_then(|v| v.as_str())
            .and_then(|b| base64_decode(b).ok());
        let capture: Vec<PathBuf> = input
            .get("capture")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter().filter_map(|v| v.as_str()).map(|s| PathBuf::from(sanitize_name(s))).collect()
            })
            .unwrap_or_default();

        let mut req = BinaryExecRequest::new(argv.clone());
        req.staged_files =
            self.staged.iter().map(|(_, rel, bytes)| (rel.clone(), bytes.clone())).collect();
        req.stdin = stdin;
        req.capture_files = capture;
        req.timeout = Duration::from_secs(self.scope.target.timeout_secs());
        req.allow_loopback = self.scope.target.allow_loopback;
        req.allow_read = self.scope.target.allow_read.iter().map(PathBuf::from).collect();

        match self.executor.exec(self.workspace, req).await {
            Ok(result) => {
                self.exec_count += 1;
                let n = self.exec_count;
                // Persist a replayable evidence bundle when the run was
                // contained (anything but a clean exit(0)) so a finding
                // can cite a byte-exact repro.
                let proof_dir = if result.contained() {
                    self.persist_evidence(n, &argv, &result)
                } else {
                    None
                };
                let mut out = serde_json::to_value(&result).unwrap_or(serde_json::json!({}));
                if let Some(dir) = proof_dir {
                    out["proof_dir"] = serde_json::json!(dir);
                }
                out["resolved_argv"] = serde_json::json!(argv);
                self.audit.push(BinaryTargetAuditEntry {
                    action: "sandbox.exec".to_string(),
                    summary: format!(
                        "argv={argv:?} status={:?} crash={:?}",
                        result.status, result.crash_signal
                    ),
                });
                out
            }
            Err(e) => serde_json::json!({"error": e}),
        }
    }

    /// Resolve the realized argv from the model's slot fillers.
    ///
    /// - No template: `[program] + base_args + model_args` (with
    ///   `@FILE:<name>` entries resolved to the staged workspace path).
    /// - Template present: walk the template — `@ARG` consumes the next
    ///   model arg in order, `@FILE:<name>` resolves to the staged path,
    ///   anything else is a literal — then prefix `[program] + base_args`.
    fn build_argv(&self, model_args: &[String]) -> Result<Vec<String>, String> {
        let mut argv = vec![self.resolved_program.clone()];
        argv.extend(self.scope.target.base_args.iter().cloned());
        match &self.scope.target.argv_template {
            None => {
                for a in model_args {
                    argv.push(self.resolve_arg(a)?);
                }
            }
            Some(template) => {
                let mut next = model_args.iter();
                for slot in template {
                    if slot == "@ARG" {
                        let a = next
                            .next()
                            .ok_or_else(|| "argv_template @ARG slot not filled".to_string())?;
                        argv.push(self.resolve_arg(a)?);
                    } else if let Some(name) = slot.strip_prefix("@FILE:") {
                        argv.push(self.staged_path(name)?);
                    } else {
                        argv.push(slot.clone());
                    }
                }
            }
        }
        Ok(argv)
    }

    fn resolve_arg(&self, arg: &str) -> Result<String, String> {
        if let Some(name) = arg.strip_prefix("@FILE:") {
            self.staged_path(name)
        } else {
            Ok(arg.to_string())
        }
    }

    fn staged_path(&self, name: &str) -> Result<String, String> {
        self.staged
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, rel, _)| rel.to_string_lossy().to_string())
            .ok_or_else(|| format!("@FILE:{name} references a file not staged via sandbox.write_file"))
    }

    /// Write the staged inputs + output previews under
    /// `<artifact_dir>/exec-<n>/` so a recorded finding has a
    /// sandbox-replayable repro. Returns the directory path.
    fn persist_evidence(&self, n: u32, argv: &[String], result: &BinaryExecResult) -> Option<String> {
        if self.scope.artifact_dir.is_empty() {
            return None;
        }
        let dir = PathBuf::from(&self.scope.artifact_dir).join(format!("exec-{n}"));
        std::fs::create_dir_all(&dir).ok()?;
        let _ = std::fs::write(dir.join("argv.json"), serde_json::json!(argv).to_string());
        let _ = std::fs::write(dir.join("stdout.txt"), &result.stdout_preview);
        let _ = std::fs::write(dir.join("stderr.txt"), &result.stderr_preview);
        for (_, rel, bytes) in &self.staged {
            if let Some(fname) = rel.file_name() {
                let _ = std::fs::write(dir.join(fname), bytes);
            }
        }
        Some(dir.to_string_lossy().to_string())
    }

    fn handle_record(&mut self, input: &Value) -> bool {
        // Reuse the shared lift so binary findings land in the same
        // vulnerability shape as live attack findings (doc §4.3).
        match classify_tool_use("record_binary_finding", input) {
            Some(ExtractedAgentResult::AttackVulnerability {
                title,
                vuln_class,
                severity,
                confidence,
                affected_components,
                business_impact,
                evidence_summary,
                repro_steps,
                remediation,
                proof_artifact_paths,
                ..
            }) => {
                self.audit.push(BinaryTargetAuditEntry {
                    action: "record_binary_finding".to_string(),
                    summary: format!("{title} class={vuln_class} confidence={confidence}%"),
                });
                self.findings.push(BinaryFinding {
                    title,
                    vuln_class,
                    severity,
                    confidence,
                    affected_components,
                    business_impact,
                    evidence_summary,
                    repro_steps,
                    remediation,
                    proof_artifact_paths,
                });
                true
            }
            _ => false,
        }
    }
}

fn build_objective(scope: &BinaryTargetScope, resolved_program: &str) -> String {
    let mut objective = include_str!("../prompts/binary_target.v1.objective.md").to_string();
    objective = objective.replace("@@RUN_ID@@", &scope.run_id);
    objective = objective.replace("@@PROJECT_ID@@", &scope.project_id);
    objective = objective.replace("@@TARGET@@", &render_target(&scope.target, resolved_program));
    objective = objective.replace("@@ARGV_TEMPLATE@@", &render_argv_template(&scope.target));
    objective = objective.replace("@@KNOWN_LEADS@@", &render_leads(&scope.known_leads));
    objective = objective.replace("@@ARTIFACT_DIR@@", &scope.artifact_dir);
    objective = objective.replace("@@MAX_TURNS@@", &scope.max_turns.to_string());
    objective
}

fn render_target(target: &LocalBinaryTarget, resolved_program: &str) -> String {
    let mut lines = vec![
        format!("- program (resolved): {resolved_program}"),
        format!("- per-exec timeout: {}s", target.timeout_secs()),
        format!("- loopback network: {}", target.allow_loopback),
    ];
    if !target.base_args.is_empty() {
        lines.push(format!("- fixed base args (you may NOT change): {:?}", target.base_args));
    }
    if let Some(oracle) = &target.oracle_program {
        lines.push(format!("- differential oracle build: {oracle}"));
    }
    if !target.allow_read.is_empty() {
        lines.push(format!("- extra readable paths: {:?}", target.allow_read));
    }
    lines.join("\n")
}

fn render_argv_template(target: &LocalBinaryTarget) -> String {
    match &target.argv_template {
        Some(t) => format!(
            "{t:?}\n(@ARG slots you fill in order; @FILE:<name> resolves to a file you staged)"
        ),
        None => "(no template — supply the full argv tail in `args`; use @FILE:<name> for staged files)"
            .to_string(),
    }
}

fn render_leads(leads: &[BinaryTargetLead]) -> String {
    if leads.is_empty() {
        return "(none — inspect the target's behaviour from first principles)".to_string();
    }
    leads
        .iter()
        .take(40)
        .map(|l| format!("- [{}] {}", l.source, truncate(&l.note, 240)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Resolve a program reference to an absolute path. A reference
/// containing a path separator is taken as a path; a bare name is looked
/// up on `PATH` via `which`.
fn resolve_program(program: &str) -> Result<String, String> {
    let program = program.trim();
    if program.is_empty() {
        return Err("empty program".to_string());
    }
    let candidate = Path::new(program);
    if candidate.is_absolute() {
        if candidate.exists() {
            return Ok(program.to_string());
        }
        return Err(format!("{program} does not exist"));
    }
    if program.contains('/') {
        // Relative path: resolve against cwd.
        let abs = std::env::current_dir().map_err(|e| e.to_string())?.join(candidate);
        if abs.exists() {
            return Ok(abs.to_string_lossy().to_string());
        }
        return Err(format!("{program} does not exist"));
    }
    which::which(program).map(|p| p.to_string_lossy().to_string()).map_err(|e| e.to_string())
}

/// Extract the first `{"tool": "...", "input": {...}}` object from model
/// text. Tolerates the object being embedded in prose. `input` defaults
/// to `{}` when omitted.
fn first_tool_action(text: &str) -> Option<(String, Value)> {
    for value in crate::tasks::structured_output::json_values_from_text(text) {
        if let Some(tool) = value.get("tool").and_then(|v| v.as_str()) {
            let input = value.get("input").cloned().unwrap_or_else(|| serde_json::json!({}));
            return Some((tool.to_string(), input));
        }
    }
    None
}

/// Reduce a logical file name to a safe single-segment basename: drop
/// any directory components and reject empties.
fn sanitize_name(name: &str) -> String {
    let base = Path::new(name).file_name().and_then(|s| s.to_str()).unwrap_or("input");
    if base.is_empty() || base == "." || base == ".." {
        "input".to_string()
    } else {
        base.to_string()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// Minimal standard-alphabet base64 decoder (RFC 4648, with `=`
/// padding). Avoids pulling in a base64 crate for the one decode path
/// the agent uses to stage binary input. Whitespace is ignored.
fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut bits: u32 = 0;
    let mut nbits = 0;
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    for &c in input.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = val(c).ok_or_else(|| format!("invalid base64 byte 0x{c:02x}"))?;
        bits = (bits << 6) | v as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((bits >> nbits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use nyx_agent_sandbox::BinaryExecStatus;
    use nyx_agent_types::agent::{
        AgentResult, AgentTask, CacheStats, CostEstimate, Response, TokenUsage,
    };
    use tokio::sync::broadcast;

    use super::*;

    #[test]
    fn base64_round_trips_known_vectors() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("AAEC/w==").unwrap(), vec![0u8, 1, 2, 255]);
        assert_eq!(base64_decode("").unwrap(), Vec::<u8>::new());
        assert!(base64_decode("****").is_err());
    }

    #[test]
    fn first_tool_action_extracts_embedded_object() {
        let (tool, input) = first_tool_action(
            "I'll stage it.\n{\"tool\":\"sandbox.write_file\",\"input\":{\"name\":\"x\"}}\nok",
        )
        .unwrap();
        assert_eq!(tool, "sandbox.write_file");
        assert_eq!(input["name"], "x");
    }

    #[test]
    fn sanitize_name_strips_dirs() {
        assert_eq!(sanitize_name("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_name("sub/dir/input.bin"), "input.bin");
        assert_eq!(sanitize_name(""), "input");
    }

    // --- fake executor + runtime to drive the micro-loop ---

    struct FakeExec {
        // returns Signaled(11) once a file named "crash" is staged & exec'd
        last_argv: Mutex<Vec<Vec<String>>>,
    }

    #[async_trait]
    impl SandboxExecutor for FakeExec {
        async fn exec(
            &self,
            _workspace: &Path,
            req: BinaryExecRequest,
        ) -> Result<BinaryExecResult, String> {
            self.last_argv.lock().unwrap().push(req.argv.clone());
            let crashed = req.staged_files.iter().any(|(_, b)| b == b"BOOM");
            Ok(BinaryExecResult {
                backend: "process".to_string(),
                status: if crashed {
                    BinaryExecStatus::Signaled { signal: 11 }
                } else {
                    BinaryExecStatus::Exited { code: 0 }
                },
                stdout_preview: String::new(),
                stderr_preview: if crashed { "segfault".into() } else { String::new() },
                duration_ms: 1,
                crash_signal: if crashed {
                    Some(nyx_agent_sandbox::CrashSignal::Segfault)
                } else {
                    None
                },
                artifacts_present: vec![],
                refusals: vec![],
            })
        }
    }

    struct ScriptedRuntime {
        turns: Mutex<std::collections::VecDeque<String>>,
    }

    #[async_trait]
    impl AiRuntime for ScriptedRuntime {
        fn name(&self) -> &'static str {
            "scripted"
        }
        fn default_model(&self) -> &str {
            "scripted"
        }
        fn supports_agent_loop(&self) -> bool {
            false
        }
        fn supports_prompt_cache(&self) -> bool {
            false
        }
        fn supports_deterministic_sampling(&self) -> bool {
            true
        }
        async fn one_shot(
            &self,
            prompt: Prompt,
            _budget: Budget,
            _sink: EventSink,
        ) -> Result<Response, AiError> {
            let content = self.turns.lock().unwrap().pop_front().unwrap_or_else(|| {
                "{\"tool\":\"stop\",\"input\":{\"reason\":\"out of script\"}}".to_string()
            });
            Ok(Response {
                prompt_version: prompt.prompt_version,
                task_id: prompt.task_id,
                model: "scripted".to_string(),
                content,
                usage: TokenUsage { input_tokens: 1, output_tokens: 1 },
                cache: Some(CacheStats::default()),
                cost_usd_micros: 10,
            })
        }
        async fn agent_loop(
            &self,
            _task: AgentTask,
            _budget: Budget,
            _sink: EventSink,
        ) -> Result<AgentResult, AiError> {
            Err(AiError::UnsupportedMode("agent_loop"))
        }
        fn cost_estimate(&self, _prompt: &Prompt) -> Option<CostEstimate> {
            None
        }
    }

    fn scope_with(dir: &Path, target: LocalBinaryTarget) -> BinaryTargetScope {
        let mut scope = BinaryTargetScope::new("run-1", "proj-1", target);
        scope.workspace_root = dir.join("ws").to_string_lossy().to_string();
        scope.artifact_dir = dir.join("artifacts").to_string_lossy().to_string();
        scope.max_turns = 10;
        scope
    }

    #[tokio::test]
    async fn micro_loop_stages_execs_and_records_finding() {
        let dir = tempfile::tempdir().unwrap();
        // /bin/echo is resolvable; the fake executor ignores it anyway.
        let target = LocalBinaryTarget::new("/bin/echo");
        let scope = scope_with(dir.path(), target);

        let script = [
            // base64("BOOM") = Qk9PTQ==
            "{\"tool\":\"sandbox.write_file\",\"input\":{\"name\":\"crash\",\"content_base64\":\"Qk9PTQ==\"}}",
            "{\"tool\":\"sandbox.exec\",\"input\":{\"args\":[\"@FILE:crash\"]}}",
            "{\"tool\":\"record_binary_finding\",\"input\":{\"title\":\"segfault on crafted input\",\"vuln_class\":\"memory_safety\",\"severity\":\"High\",\"confidence\":90,\"business_impact\":\"crash\",\"evidence_summary\":\"SIGSEGV\",\"repro_steps\":\"echo crash\",\"proof_artifact_paths\":[\"exec-1\"]}}",
            "{\"tool\":\"stop\",\"input\":{\"reason\":\"done\"}}",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<std::collections::VecDeque<_>>();

        let runtime = ScriptedRuntime { turns: Mutex::new(script) };
        let exec = FakeExec { last_argv: Mutex::new(Vec::new()) };
        let (tx, _) = broadcast::channel(8);

        let outcome = run(&runtime, &scope, &exec, tx).await.expect("run");
        assert_eq!(outcome.exec_count, 1);
        assert_eq!(outcome.findings.len(), 1);
        assert_eq!(outcome.findings[0].vuln_class, "memory_safety");
        assert!(!outcome.halted);
        // argv[0] is the resolved program; tail is the staged file path.
        let argvs = exec.last_argv.lock().unwrap();
        assert_eq!(argvs[0].last().unwrap(), "crash");
        // evidence bundle was written for the contained run.
        assert!(dir.path().join("artifacts/exec-1/crash").exists());
    }

    #[tokio::test]
    async fn argv_template_fills_arg_and_file_slots() {
        let dir = tempfile::tempdir().unwrap();
        let mut target = LocalBinaryTarget::new("/bin/echo");
        target.argv_template = Some(vec!["@ARG".into(), "@FILE:input".into()]);
        let scope = scope_with(dir.path(), target);

        let script = [
            "{\"tool\":\"sandbox.write_file\",\"input\":{\"name\":\"input\",\"content_text\":\"hi\"}}",
            "{\"tool\":\"sandbox.exec\",\"input\":{\"args\":[\"--flag\"]}}",
            "{\"tool\":\"stop\",\"input\":{}}",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<std::collections::VecDeque<_>>();

        let runtime = ScriptedRuntime { turns: Mutex::new(script) };
        let exec = FakeExec { last_argv: Mutex::new(Vec::new()) };
        let (tx, _) = broadcast::channel(8);

        let outcome = run(&runtime, &scope, &exec, tx).await.expect("run");
        assert_eq!(outcome.exec_count, 1);
        let argvs = exec.last_argv.lock().unwrap();
        // [program, "--flag" (from @ARG), "input" (from @FILE:input)]
        assert_eq!(argvs[0][1], "--flag");
        assert_eq!(argvs[0][2], "input");
    }

    #[tokio::test]
    async fn budget_exhaustion_halts() {
        let dir = tempfile::tempdir().unwrap();
        let scope = {
            let mut s = scope_with(dir.path(), LocalBinaryTarget::new("/bin/echo"));
            s.run_cap_usd_micros = Some(15); // 10 micros/turn -> halts after 1
            s
        };
        // Script never stops; loop must halt on budget.
        let script = std::iter::repeat(
            "{\"tool\":\"sandbox.exec\",\"input\":{\"args\":[]}}".to_string(),
        )
        .take(20)
        .collect::<std::collections::VecDeque<_>>();
        let runtime = ScriptedRuntime { turns: Mutex::new(script) };
        let exec = FakeExec { last_argv: Mutex::new(Vec::new()) };
        let (tx, _) = broadcast::channel(8);
        let outcome = run(&runtime, &scope, &exec, tx).await.expect("run");
        assert!(outcome.halted);
        assert!(outcome.spent_usd_micros >= 15);
    }
}
