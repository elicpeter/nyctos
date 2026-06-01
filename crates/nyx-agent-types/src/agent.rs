//! Shared types for the AI runtime layer.
//!
//! Every adapter (Anthropic SDK, Claude Code, OpenAI, ...) consumes the
//! same `Prompt` / `Response` envelope so the rest of the agent never
//! depends on a vendor SDK shape. The `prompt_version` field on `Prompt`
//! is persisted with every trace so a verdict can always be traced back
//! to the exact prompt that produced it.
//!
//! All types derive `ts_rs::TS` so the frontend consumes them directly
//! through the `build.rs`-generated bindings.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;
use ts_rs::TS;

use crate::product::{ProjectLaunchProfileInput, SeedSetupPlan};
use crate::project::{ProjectAuthMode, ProjectAuthProfile};

/// Single-turn prompt envelope. The `prompt_version` field is the only
/// load-bearing constant - every adapter persists it alongside any
/// response so traces remain explainable across prompt edits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, TS)]
pub struct Prompt {
    /// Stable identifier of the prompt template. Adapters and the trace
    /// store both persist this so a verdict can be tied back to the
    /// exact prompt revision that produced it.
    pub prompt_version: String,
    /// Logical task identifier used to namespace streaming events on
    /// the bus. The caller supplies it; adapters echo it back in every
    /// emitted `AiEvent`.
    pub task_id: String,
    /// Model override. When `None`, the adapter's `default_model()` is
    /// used.
    pub model: Option<String>,
    /// System prompt. Adapters that support prompt caching may attach a
    /// `cache_control` block to this slot.
    pub system: String,
    /// User message body.
    pub user: String,
    /// Hard ceiling on output tokens. Adapters clamp to vendor limits.
    pub max_output_tokens: u32,
    /// Sampling temperature. `0.0` for deterministic decoding.
    pub temperature: f32,
    /// Adapter-specific seed for deterministic sampling. Adapters that
    /// do not expose a seed ignore this.
    #[ts(type = "number | null")]
    pub seed: Option<u64>,
}

/// Adapter response envelope. Carries the model's final text plus the
/// accounting needed to persist a trace and reconcile budgets.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct Response {
    /// Echoes the `prompt_version` from the request.
    pub prompt_version: String,
    /// Echoes the `task_id` from the request.
    pub task_id: String,
    /// Model name as reported by the vendor (which may differ from the
    /// requested alias).
    pub model: String,
    /// Final completion text.
    pub content: String,
    /// Token accounting.
    pub usage: TokenUsage,
    /// Prompt-cache statistics, if the adapter reports them. `None`
    /// when the runtime does not support caching.
    pub cache: Option<CacheStats>,
    /// Total cost charged for this call, in USD micros (1e-6 USD).
    #[ts(type = "number")]
    pub cost_usd_micros: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct CacheStats {
    pub cache_creation_tokens: u32,
    pub cache_read_tokens: u32,
}

/// Per-call observability bundle carried from each task's outcome
/// envelope back to the binary's trace-row builder. Adapters populate
/// `usage` / `cache` / `model` from the underlying `Response` or
/// `AgentResult`; tasks accumulate across retry attempts via
/// [`AgentTraceMetrics::merge`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct AgentTraceMetrics {
    pub usage: TokenUsage,
    pub cache: Option<CacheStats>,
    pub model: Option<String>,
}

impl AgentTraceMetrics {
    /// Build a metrics bundle from a single `one_shot` response.
    pub fn from_response(resp: &Response) -> Self {
        Self { usage: resp.usage, cache: resp.cache, model: Some(resp.model.clone()) }
    }

    /// Build a metrics bundle from an `agent_loop` result. `cache` is
    /// `None` until the adapter learns to parse cache_* delta events.
    pub fn from_agent_result(r: &AgentResult) -> Self {
        let model = if r.model.is_empty() { None } else { Some(r.model.clone()) };
        Self { usage: r.usage, cache: r.cache, model }
    }

    /// Saturating-merge another metrics bundle into self. Used to sum
    /// across retry attempts in tasks that may issue two `one_shot`
    /// calls. `model` and `cache` from `other` win when present.
    pub fn merge(mut self, other: AgentTraceMetrics) -> Self {
        self.usage.input_tokens = self.usage.input_tokens.saturating_add(other.usage.input_tokens);
        self.usage.output_tokens =
            self.usage.output_tokens.saturating_add(other.usage.output_tokens);
        match (self.cache, other.cache) {
            (Some(a), Some(b)) => {
                self.cache = Some(CacheStats {
                    cache_creation_tokens: a
                        .cache_creation_tokens
                        .saturating_add(b.cache_creation_tokens),
                    cache_read_tokens: a.cache_read_tokens.saturating_add(b.cache_read_tokens),
                });
            }
            (None, Some(b)) => self.cache = Some(b),
            (a, None) => self.cache = a,
        }
        if other.model.is_some() {
            self.model = other.model;
        }
        self
    }
}

/// Per-call budget contract. The adapter checks `cap_usd_micros` against
/// the per-run spend tracked by the host before and after every model
/// call; on cap-exceeded it emits `AiEvent::TaskHalted` and returns
/// `AiError::BudgetExceeded`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct Budget {
    /// Run identifier used as the budget-store key.
    pub run_id: String,
    /// Budget bucket. `OneShot` for `one_shot` calls; `AgentLoop` for
    /// multi-turn loops; `Total` is reserved for the per-run aggregate
    /// the host writes itself.
    pub kind: BudgetKind,
    /// Hard cap, in USD micros. Exceeding this cap halts the task.
    #[ts(type = "number")]
    pub cap_usd_micros: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum BudgetKind {
    OneShot,
    AgentLoop,
    Total,
}

impl BudgetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BudgetKind::OneShot => "OneShot",
            BudgetKind::AgentLoop => "AgentLoop",
            BudgetKind::Total => "Total",
        }
    }
}

/// Pre-call cost prediction. Adapters that price deterministically
/// return this from `cost_estimate`; the host uses it for an early
/// halt check before the round-trip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct CostEstimate {
    #[ts(type = "number")]
    pub min_usd_micros: i64,
    #[ts(type = "number")]
    pub max_usd_micros: i64,
}

/// Multi-turn agent task. The Anthropic adapter returns
/// `AiError::UnsupportedMode` from `agent_loop`; the Claude Code
/// adapter implements it. The type is defined here so both adapters
/// share one schema.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct AgentTask {
    pub prompt_version: String,
    pub task_id: String,
    pub system: String,
    pub objective: String,
    pub tools: Vec<String>,
    /// Optional working directory for CLI-backed agent loops. When set,
    /// adapters launch the agent from this directory so native file,
    /// search, and shell tools operate on the target repository rather
    /// than the daemon's process cwd.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub working_directory: Option<String>,
    pub max_turns: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct AgentResult {
    pub prompt_version: String,
    pub task_id: String,
    /// Model name as reported by the vendor. Empty when the adapter
    /// cannot extract a per-call model id from the agent-loop stream.
    #[serde(default)]
    pub model: String,
    pub final_message: String,
    pub turns: u32,
    pub usage: TokenUsage,
    /// Prompt-cache statistics. `None` when the adapter does not parse
    /// per-turn cache deltas yet.
    #[serde(default)]
    pub cache: Option<CacheStats>,
    #[ts(type = "number")]
    pub cost_usd_micros: i64,
    /// Structured artefacts the adapter lifted out of the agent loop's
    /// tool-use trace. The Claude Code adapter populates these from
    /// recognised function calls; PayloadSynthesis, SpecDerivation,
    /// ChainReasoning, and exploration consume them as the typed
    /// agent-loop output.
    #[serde(default)]
    pub extracted: Vec<ExtractedAgentResult>,
}

/// Typed view of the structured artefacts an agent loop produced.
///
/// The agent-loop trace is non-deterministic by nature; consumers do
/// not depend on event order, only on the set of `ExtractedAgentResult`
/// values the adapter recognised. Each variant carries the smallest
/// stable payload its consumer needs; richer per-variant schemas live
/// alongside the consuming task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
#[allow(clippy::large_enum_variant)]
#[serde(tag = "kind")]
pub enum ExtractedAgentResult {
    /// PayloadSynthesis output: exploit payload candidate (rule id +
    /// payload body).
    PayloadFound { rule_id: String, body: String },
    /// SpecDerivation output: capability spec inferred for a sink.
    SpecFound { capability: String, spec: String },
    /// ChainReasoning output: ranked chain ids with a short rationale.
    ChainsRanked { chain_ids: Vec<String>, rationale: String },
    /// Exploration output: AI-discovered candidate finding. Emitted by
    /// the Claude Code agent loop when it identifies a new
    /// vulnerability while exploring the workspace (shadow API, state
    /// machine flaw, CORS misconfiguration, ...).
    ExplorationFinding {
        path: String,
        line: Option<u32>,
        cap: String,
        rationale: String,
        #[serde(default)]
        endpoint: Option<String>,
        #[serde(default)]
        suggested_payload_hint: Option<String>,
    },
    /// Unsafe local attack-agent output: the agent claims it exploited
    /// or broke something in the running development app and provides
    /// enough material to create a user-facing vulnerability row.
    AttackVulnerability {
        title: String,
        vuln_class: String,
        severity: String,
        confidence: u8,
        #[serde(default)]
        #[ts(type = "Array<unknown>")]
        affected_components: Vec<serde_json::Value>,
        business_impact: String,
        evidence_summary: String,
        repro_steps: String,
        remediation: String,
        #[serde(default)]
        source_candidate_ids: Vec<String>,
        #[serde(default)]
        source_signal_ids: Vec<String>,
        #[serde(default)]
        proof_artifact_paths: Vec<String>,
    },
    /// Free-form exploration trace event. Captures anything the agent
    /// surfaced that does not fit a more specific variant.
    ExplorationEvent { message: String },
    /// Auth setup output: one secret-safe auth profile generated after
    /// inspecting the target repository.
    AuthProfileDiscovered { profile: ProjectAuthProfile, rationale: String },
    /// Auth setup output: the agent's self-check after comparing the
    /// generated profiles back to the repository's auth routes and
    /// session flow.
    AuthSetupVerification { status: String, checks: Vec<String>, warnings: Vec<String> },
    /// Project setup output: a launch profile produced after the agent
    /// inspected and exercised the repository's local dev workflow.
    ProjectSetupProfile {
        profile: ProjectLaunchProfileInput,
        summary: String,
        checks: Vec<String>,
        warnings: Vec<String>,
    },
    /// Seed setup output: deterministic local fixtures and launch hooks
    /// that create roles, owned objects, tenants, and other app-native
    /// state for live authorization checks.
    SeedSetupPlan { plan: SeedSetupPlan },
    /// Auth session output: an agent completed login and saved a
    /// Playwright storageState file that the host can import without
    /// logging raw cookies or bearer tokens.
    AuthSessionAcquired { storage_state_path: String, summary: String },
}

/// Classify a tool-use block emitted by an agent-loop adapter into a
/// typed `ExtractedAgentResult`. Recognised names (`record_payload`,
/// `record_spec`, `record_chains`) lift their inputs into the typed
/// variant; any other tool name folds into `ExplorationEvent`. Adapters
/// that surface tool calls (Claude Code today, Anthropic agent-loop if
/// it ever ships) share this mapping so the trace store sees a single
/// stable shape.
pub fn classify_tool_use(name: &str, input: &serde_json::Value) -> Option<ExtractedAgentResult> {
    match name {
        "record_payload" => {
            let rule_id = input.get("rule_id")?.as_str()?.to_string();
            let body = input.get("body")?.as_str()?.to_string();
            Some(ExtractedAgentResult::PayloadFound { rule_id, body })
        }
        "record_spec" => {
            let capability = input.get("capability")?.as_str()?.to_string();
            let spec = match input.get("spec") {
                Some(v) => match v.as_str() {
                    Some(s) => s.to_string(),
                    None => v.to_string(),
                },
                None => return None,
            };
            Some(ExtractedAgentResult::SpecFound { capability, spec })
        }
        "record_chains" => {
            let chain_ids = input
                .get("chain_ids")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>();
            let rationale =
                input.get("rationale").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            Some(ExtractedAgentResult::ChainsRanked { chain_ids, rationale })
        }
        "record_exploration_finding" => {
            let path = input.get("path")?.as_str()?.to_string();
            let cap = input.get("cap")?.as_str()?.to_string();
            let rationale = input.get("rationale")?.as_str()?.to_string();
            if path.trim().is_empty() || cap.trim().is_empty() || rationale.trim().is_empty() {
                return None;
            }
            let line = input.get("line").and_then(|v| v.as_u64()).map(|n| n as u32);
            let endpoint = input.get("endpoint").and_then(|v| v.as_str()).map(|s| s.to_string());
            let suggested_payload_hint =
                input.get("suggested_payload_hint").and_then(|v| v.as_str()).map(|s| s.to_string());
            Some(ExtractedAgentResult::ExplorationFinding {
                path,
                line,
                cap,
                rationale,
                endpoint,
                suggested_payload_hint,
            })
        }
        // Live attack-agent findings and binary-target findings share
        // one lifted shape (`AttackVulnerability`): a binary finding is
        // just a vulnerability with binary-flavoured evidence and a
        // binary `vuln_class` (see docs/binary-target-pentest.md §4.3).
        // `record_binary_finding` defaults `vuln_class` to
        // `memory_safety` when the agent omits it.
        "record_attack_vulnerability" => attack_vulnerability_from_input(input, None),
        "record_binary_finding" => {
            attack_vulnerability_from_input(input, Some("memory_safety"))
        }
        "record_auth_profile" => {
            let profile = auth_profile_from_tool_input(input)?;
            let rationale = optional_string(input, "rationale")
                .unwrap_or_else(|| "repo auth profile".to_string());
            Some(ExtractedAgentResult::AuthProfileDiscovered { profile, rationale })
        }
        "record_auth_verification" => {
            let status =
                optional_string(input, "status").unwrap_or_else(|| "needs_review".to_string());
            let checks = string_array(input, "checks");
            let warnings = string_array(input, "warnings");
            Some(ExtractedAgentResult::AuthSetupVerification { status, checks, warnings })
        }
        "record_project_setup" => {
            let profile_value = input.get("profile")?;
            let profile: ProjectLaunchProfileInput =
                serde_json::from_value(profile_value.clone()).ok()?;
            let summary = optional_string(input, "summary")
                .unwrap_or_else(|| "local project setup".to_string());
            Some(ExtractedAgentResult::ProjectSetupProfile {
                profile,
                summary,
                checks: string_array(input, "checks"),
                warnings: string_array(input, "warnings"),
            })
        }
        "record_seed_setup" => {
            let plan_value = input.get("plan").unwrap_or(input);
            let plan: SeedSetupPlan = serde_json::from_value(plan_value.clone()).ok()?;
            Some(ExtractedAgentResult::SeedSetupPlan { plan })
        }
        "record_auth_session" => {
            let storage_state_path = optional_string(input, "storage_state_path")?;
            let summary =
                optional_string(input, "summary").unwrap_or_else(|| "session captured".to_string());
            Some(ExtractedAgentResult::AuthSessionAcquired { storage_state_path, summary })
        }
        _ => Some(ExtractedAgentResult::ExplorationEvent {
            message: format!("tool {name} input={input}"),
        }),
    }
}

/// Lift a `record_attack_vulnerability` / `record_binary_finding` tool
/// input into the shared [`ExtractedAgentResult::AttackVulnerability`]
/// variant. `default_vuln_class` supplies the class when the agent omits
/// both `vuln_class` and `cap` (binary findings default to a binary
/// class; live attack findings have no default and must name one).
/// Returns `None` when a required field is missing or empty.
fn attack_vulnerability_from_input(
    input: &serde_json::Value,
    default_vuln_class: Option<&str>,
) -> Option<ExtractedAgentResult> {
    let title = input.get("title")?.as_str()?.trim().to_string();
    let vuln_class = match input
        .get("vuln_class")
        .or_else(|| input.get("cap"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
    {
        Some(s) if !s.is_empty() => s,
        _ => default_vuln_class?.to_string(),
    };
    let severity =
        input.get("severity").and_then(|v| v.as_str()).unwrap_or("High").trim().to_string();
    let business_impact = input.get("business_impact")?.as_str()?.trim().to_string();
    let evidence_summary = input.get("evidence_summary")?.as_str()?.trim().to_string();
    let repro_steps = input.get("repro_steps")?.as_str()?.trim().to_string();
    let remediation = input
        .get("remediation")
        .and_then(|v| v.as_str())
        .unwrap_or("Review the vulnerable flow and apply a targeted fix.")
        .trim()
        .to_string();
    if title.is_empty()
        || vuln_class.is_empty()
        || business_impact.is_empty()
        || evidence_summary.is_empty()
        || repro_steps.is_empty()
    {
        return None;
    }
    let confidence = input.get("confidence").map(confidence_percent_from_value).unwrap_or(90);
    let affected_components =
        input.get("affected_components").and_then(|v| v.as_array()).cloned().unwrap_or_default();
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
        source_candidate_ids: string_array(input, "source_candidate_ids"),
        source_signal_ids: string_array(input, "source_signal_ids"),
        proof_artifact_paths: string_array(input, "proof_artifact_paths"),
    })
}

fn confidence_percent_from_value(value: &serde_json::Value) -> u8 {
    if let Some(raw) = value.as_u64() {
        return raw.min(100) as u8;
    }
    if let Some(raw) = value.as_f64().filter(|v| v.is_finite()) {
        let percent = if raw <= 1.0 { raw * 100.0 } else { raw };
        return percent.round().clamp(0.0, 100.0) as u8;
    }
    90
}

fn auth_profile_from_tool_input(input: &serde_json::Value) -> Option<ProjectAuthProfile> {
    if let Some(profile) = input.get("profile") {
        let mut parsed: ProjectAuthProfile = serde_json::from_value(profile.clone()).ok()?;
        parsed.role = parsed.role.trim().to_string();
        if parsed.role.is_empty() || parsed.role.eq_ignore_ascii_case("anonymous") {
            return None;
        }
        return Some(parsed);
    }

    let role = optional_string(input, "role")?;
    if role.is_empty() || role.eq_ignore_ascii_case("anonymous") {
        return None;
    }
    let mode = parse_field(input, "mode").unwrap_or(ProjectAuthMode::AiAuto);
    Some(ProjectAuthProfile {
        role,
        role_aliases: string_array(input, "role_aliases"),
        mode,
        label: optional_string(input, "label"),
        tenant: optional_string(input, "tenant"),
        session_cache_ttl_seconds: input.get("session_cache_ttl_seconds").and_then(|v| v.as_u64()),
        session_import_path: optional_string(input, "session_import_path"),
        login_url: optional_string(input, "login_url"),
        username: optional_string(input, "username"),
        username_env: optional_string(input, "username_env"),
        login_email_env: optional_string(input, "login_email_env"),
        password_env: optional_string(input, "password_env"),
        password_secret_ref: optional_string(input, "password_secret_ref"),
        cookie_env: optional_string(input, "cookie_env"),
        bearer_token_env: optional_string(input, "bearer_token_env"),
        headers: parse_vec(input, "headers"),
        otp_source: parse_field(input, "otp_source"),
        post_login_assertions: parse_vec(input, "post_login_assertions"),
        post_login_assertion: optional_string(input, "post_login_assertion"),
        custom_command: optional_string(input, "custom_command"),
        owned_objects: parse_vec(input, "owned_objects"),
    })
}

fn optional_string(input: &serde_json::Value, field: &str) -> Option<String> {
    let value = input.get(field)?.as_str()?.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn string_array(input: &serde_json::Value, field: &str) -> Vec<String> {
    input
        .get(field)
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_vec<T: DeserializeOwned>(input: &serde_json::Value, field: &str) -> Vec<T> {
    input.get(field).cloned().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
}

fn parse_field<T: DeserializeOwned>(input: &serde_json::Value, field: &str) -> Option<T> {
    input.get(field).cloned().and_then(|v| serde_json::from_value(v).ok())
}

/// Reason the adapter halted a task. Surfaced both on the event bus
/// (`AiEvent::TaskHalted`) and as part of the typed error.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum HaltReason {
    BudgetCapReached,
    OperatorCancelled,
    UpstreamRefused,
}

#[derive(Debug, Error)]
pub enum AiError {
    #[error("budget cap of {cap_usd_micros} usd-micros reached (spent {spent_usd_micros})")]
    BudgetExceeded { cap_usd_micros: i64, spent_usd_micros: i64 },
    #[error("adapter does not support {0}")]
    UnsupportedMode(&'static str),
    #[error("upstream refused: {0}")]
    UpstreamRefused(String),
    #[error("upstream returned malformed response: {0}")]
    MalformedResponse(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("budget tracker error: {0}")]
    BudgetTracker(String),
    #[error("adapter unavailable: {0}")]
    AdapterUnavailable(String),
}
