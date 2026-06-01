//! Typed configuration loaded from `nyx-agent.toml`.
//!
//! Missing sections fall back to defaults so that `nyx-agent doctor` and
//! other read-only operations work in a fresh checkout with no config
//! file on disk.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use nyx_agent_types::product::{
    LaunchEnvRef, LaunchHealthCheck, LaunchStep, ProjectLaunchProfileInput,
};
use nyx_agent_types::project::ProjectRuntimeProfile;
use nyx_agent_types::target::{LocalBinaryTarget, PentestTarget};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config at {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config at {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("failed to serialise config: {0}")]
    Serialise(#[from] toml::ser::Error),
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub general: GeneralConfig,
    pub performance: PerformanceConfig,
    pub sandbox: SandboxConfig,
    pub ai: AiConfig,
    pub ui: UiConfig,
    pub triggers: TriggersConfig,
    pub nyx: NyxConfig,
    pub run: RunConfig,
    pub env: EnvConfig,
    /// Projects own repos. Each `[[project]]` block declares one
    /// product (e.g. backend + frontend) and groups its repos under
    /// `[[project.repo]]`. The top-level `[[repo]]` shape is rejected;
    /// every repo must live under a project.
    #[serde(rename = "project", default)]
    pub projects: Vec<ProjectConfig>,
    /// Cron-driven scan schedule entries. Each entry pairs a 5-field
    /// cron expression with an optional repo filter (`None` scans
    /// every enabled repo). The daemon's scheduler task evaluates
    /// every entry once per minute.
    #[serde(rename = "schedule", default)]
    pub schedules: Vec<ScheduleConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneralConfig {
    pub log_level: String,
    pub state_dir: Option<PathBuf>,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self { log_level: "info".to_string(), state_dir: None }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PerformanceConfig {
    pub max_parallel_scans: u32,
    pub scan_timeout_secs: u64,
    /// Explicit override for the per-run static-pass fan-out.
    /// `None` -> dispatcher computes `min(num_cpus / 2, len(repos))`.
    /// `Some(n)` -> use exactly `n.max(1)` parallel jobs.
    #[serde(default)]
    pub static_concurrency: Option<usize>,
    /// Per-repo budget for the static-pass scan. A scan that exceeds
    /// the budget is killed and its repo bundle records
    /// `Inconclusive(StaticPassTimeout)` while the rest of the run
    /// continues. `None` -> 30 minutes.
    #[serde(default)]
    pub per_repo_timeout_secs: Option<u64>,
    /// Cadence at which the cron scheduler wakes to evaluate
    /// `[[schedule]]` entries. `None` -> 60 seconds (matches the
    /// granularity of the standard 5-field cron expression).
    /// Floored at 1 second by [`PerformanceConfig::scheduler_tick`].
    /// Operators should only lower this when a sub-minute cron
    /// granularity is required; tighter polling spends more CPU.
    #[serde(default)]
    pub scheduler_tick_secs: Option<u64>,
    /// Per-lane simultaneous-spinup cap for the chain lane. `None`
    /// -> built-in default (2). Mirrors
    /// `nyx_agent_sandbox::LaneConcurrency::DEFAULT_CHAIN`. A configured
    /// `0` is floored to `1` by
    /// [`PerformanceConfig::chain_lane_concurrency_resolved`].
    #[serde(default)]
    pub chain_lane_concurrency: Option<usize>,
    /// Per-lane simultaneous-spinup cap for the fast lane. `None`
    /// -> built-in default (8). Mirrors
    /// `nyx_agent_sandbox::LaneConcurrency::DEFAULT_FAST`. A configured
    /// `0` is floored to `1` by
    /// [`PerformanceConfig::fast_lane_concurrency_resolved`].
    #[serde(default)]
    pub fast_lane_concurrency: Option<usize>,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_parallel_scans: 4,
            scan_timeout_secs: 600,
            static_concurrency: None,
            per_repo_timeout_secs: None,
            scheduler_tick_secs: None,
            chain_lane_concurrency: None,
            fast_lane_concurrency: None,
        }
    }
}

impl PerformanceConfig {
    /// Resolved per-repo timeout. Falls back to 30 minutes when the
    /// operator has not set `[performance] per_repo_timeout_secs`.
    pub fn per_repo_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.per_repo_timeout_secs.unwrap_or(30 * 60))
    }

    /// Resolved static-pass fan-out. Returns `None` when the operator
    /// has not set `[performance] static_concurrency`; the dispatcher
    /// then derives the default from CPU count and repo count.
    pub fn static_concurrency_override(&self) -> Option<usize> {
        self.static_concurrency.map(|n| n.max(1))
    }

    /// Resolved scheduler wake cadence. Falls back to 60 seconds when
    /// the operator has not set `[performance] scheduler_tick_secs`;
    /// a configured `0` is floored to `1` so the loop cannot busy-wait.
    pub fn scheduler_tick(&self) -> std::time::Duration {
        let secs = self.scheduler_tick_secs.unwrap_or(60).max(1);
        std::time::Duration::from_secs(secs)
    }

    /// Built-in fallback for the chain-lane simultaneous-spinup cap.
    /// Mirrors `nyx_agent_sandbox::LaneConcurrency::DEFAULT_CHAIN`; kept
    /// duplicated here so this crate has no reverse dep on the sandbox
    /// crate. The two values must stay in sync.
    pub const DEFAULT_CHAIN_LANE_CONCURRENCY: usize = 2;

    /// Built-in fallback for the fast-lane simultaneous-spinup cap.
    /// Mirrors `nyx_agent_sandbox::LaneConcurrency::DEFAULT_FAST`.
    pub const DEFAULT_FAST_LANE_CONCURRENCY: usize = 8;

    /// Resolved chain-lane spinup cap. Falls back to
    /// [`Self::DEFAULT_CHAIN_LANE_CONCURRENCY`] when the operator has
    /// not set `[performance] chain_lane_concurrency`; a configured
    /// `0` is floored to `1` so the worker pool cannot deadlock.
    pub fn chain_lane_concurrency_resolved(&self) -> usize {
        self.chain_lane_concurrency
            .map(|n| n.max(1))
            .unwrap_or(Self::DEFAULT_CHAIN_LANE_CONCURRENCY)
    }

    /// Resolved fast-lane spinup cap. Falls back to
    /// [`Self::DEFAULT_FAST_LANE_CONCURRENCY`] when the operator has
    /// not set `[performance] fast_lane_concurrency`; a configured
    /// `0` is floored to `1`.
    pub fn fast_lane_concurrency_resolved(&self) -> usize {
        self.fast_lane_concurrency.map(|n| n.max(1)).unwrap_or(Self::DEFAULT_FAST_LANE_CONCURRENCY)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SandboxConfig {
    pub enabled: bool,
    pub allow_network: bool,
    /// The first-launch wizard records the operator's preferred
    /// sandbox backend here; the launcher reads it to pick a backend.
    #[serde(default)]
    pub backend: SandboxBackend,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self { enabled: true, allow_network: false, backend: SandboxBackend::default() }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxBackend {
    /// Pick the strongest available backend at runtime.
    #[default]
    Auto,
    /// No kernel isolation. Static-pass only. Always works.
    Process,
    /// macOS Seatbelt profile shipped with the agent.
    Birdcage,
    /// Lightweight microVM on Linux via libkrun.
    Libkrun,
    /// Lightweight microVM on Linux via Firecracker.
    Firecracker,
    /// Docker container fallback. Slowest, requires the docker daemon.
    Docker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AiConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    /// Optional reasoning effort for CLI-backed AI runtimes. The daemon
    /// validates values written through the UI, then forwards the string
    /// to the selected local CLI using that CLI's native flag/config key.
    #[serde(default)]
    pub effort: Option<String>,
    /// Optional context-window hint in tokens for CLI-backed runtimes.
    /// For Claude Code this maps to model aliases such as `opus[1m]`;
    /// for Codex this maps to `model_context_window`.
    #[serde(default)]
    pub context_window: Option<u32>,
    pub api_base: Option<String>,
    /// Operator-selected AI runtime. The wizard writes this; the run
    /// dispatcher reads it to pick which provider client to build.
    /// The API key itself is stored in the OS keychain, not in TOML.
    #[serde(default)]
    pub runtime: AiRuntime,
    /// Maximum number of in-flight `one_shot` AI calls per run.
    /// PayloadSynthesis / SpecDerivation / ChainReasoning all share
    /// this cap. `0` is floored to `1` by
    /// [`AiConfig::max_concurrent_one_shot_resolved`].
    #[serde(default = "default_max_concurrent_one_shot")]
    pub max_concurrent_one_shot: u32,
    /// Per-run AI budget cap (in USD micros) stamped on brand-new
    /// `(run_id, kind)` rows the `BudgetStoreTracker` auto-creates.
    /// `None` leaves the run uncapped. Operators can enable a cap via
    /// `[ai] default_run_budget_usd_micros` in `nyx-agent.toml`.
    #[serde(default)]
    pub default_run_budget_usd_micros: Option<i64>,
    /// Optional per-model pricing overrides. Each entry maps an exact
    /// Anthropic model id (e.g. `claude-opus-4-7-20260101`) to a
    /// per-million-token USD rate sheet. The Anthropic adapter
    /// consults this map first; unmatched models fall back to the
    /// built-in pricing table. Configured via
    /// `[ai.pricing.<model>]` blocks in `nyx-agent.toml`.
    #[serde(default)]
    pub pricing: HashMap<String, AiPricingOverride>,
    /// Per-call cap forwarded into each PayloadSynthesis `Budget`. The
    /// adapter checks the call against `min(per_call_cap, run_cap -
    /// spent_so_far)`, so this knob lets an operator clamp a single
    /// PayloadSynthesis call below the shared per-run bucket. `None`
    /// leaves the call uncapped.
    /// Configured via
    /// `[ai] payload_synthesis_per_call_cap_usd_micros`.
    #[serde(default)]
    pub payload_synthesis_per_call_cap_usd_micros: Option<i64>,
    /// Per-call cap forwarded into each SpecDerivation `Budget`. See
    /// `payload_synthesis_per_call_cap_usd_micros` for semantics.
    /// Configured via
    /// `[ai] spec_derivation_per_call_cap_usd_micros`.
    #[serde(default)]
    pub spec_derivation_per_call_cap_usd_micros: Option<i64>,
    /// Per-call cap forwarded into the single ChainReasoning `Budget`.
    /// Same shape as the PayloadSynthesis / SpecDerivation knobs.
    /// Configured via
    /// `[ai] chain_reasoning_per_call_cap_usd_micros`.
    #[serde(default)]
    pub chain_reasoning_per_call_cap_usd_micros: Option<i64>,
    /// Per-call cap forwarded into each NovelFindingDiscovery batch.
    /// Same shape; defaults to the run cap so a single batch may use
    /// the full bucket when no earlier pass has spent yet. Configured
    /// via `[ai] novel_discovery_per_call_cap_usd_micros`.
    #[serde(default)]
    pub novel_discovery_per_call_cap_usd_micros: Option<i64>,
    /// Per-task soft cap for AI Exploration. Crossing the cap emits
    /// a single operator warning but does not halt the run; the hard
    /// cap below is the only ceiling that aborts an in-progress
    /// exploration. `None` falls back to the caller-supplied default
    /// (crate-level constant in `nyx-agent-ai`). Configured via
    /// `[ai] exploration_soft_cap_usd_micros`.
    #[serde(default)]
    pub exploration_soft_cap_usd_micros: Option<i64>,
    /// Per-run hard cap for AI Exploration. Sized for Claude Opus
    /// pricing on a Phase-23 exploration loop. `None` falls back to
    /// the caller-supplied default. Configured via
    /// `[ai] exploration_run_cap_usd_micros`.
    #[serde(default)]
    pub exploration_run_cap_usd_micros: Option<i64>,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            provider: None,
            model: None,
            effort: None,
            context_window: None,
            api_base: None,
            runtime: AiRuntime::default(),
            max_concurrent_one_shot: default_max_concurrent_one_shot(),
            default_run_budget_usd_micros: None,
            pricing: HashMap::new(),
            payload_synthesis_per_call_cap_usd_micros: None,
            spec_derivation_per_call_cap_usd_micros: None,
            chain_reasoning_per_call_cap_usd_micros: None,
            novel_discovery_per_call_cap_usd_micros: None,
            exploration_soft_cap_usd_micros: None,
            exploration_run_cap_usd_micros: None,
        }
    }
}

/// Operator-friendly pricing override for one Anthropic model. All
/// fields are USD per million tokens; the adapter converts to
/// micros-per-token at construction time. Cache fields default to
/// zero so an override that only sets `input`/`output` keeps the
/// no-cache pricing of the haiku/sonnet defaults.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AiPricingOverride {
    pub input_per_mtok_usd: i64,
    pub output_per_mtok_usd: i64,
    pub cache_write_per_mtok_usd: i64,
    pub cache_read_per_mtok_usd: i64,
}

fn default_max_concurrent_one_shot() -> u32 {
    4
}

impl AiConfig {
    /// Sentinel used for the built-in uncapped AI budget. The budget
    /// plumbing still stores an integer cap, so `i64::MAX` gives the
    /// existing adapters a practical "unlimited" ceiling without a
    /// schema migration.
    pub const DEFAULT_RUN_BUDGET_USD_MICROS: i64 = i64::MAX;

    /// Floored fan-out used by run-time dispatchers. A configured `0`
    /// would deadlock a semaphore acquire so we floor to `1`.
    pub fn max_concurrent_one_shot_resolved(&self) -> usize {
        self.max_concurrent_one_shot.max(1) as usize
    }

    /// Resolved per-run AI budget cap, honouring the operator override
    /// when set. Missing, negative, or zero values resolve to the
    /// built-in uncapped sentinel.
    pub fn default_run_budget_usd_micros_resolved(&self) -> i64 {
        match self.default_run_budget_usd_micros {
            Some(v) if v > 0 => v,
            _ => Self::DEFAULT_RUN_BUDGET_USD_MICROS,
        }
    }

    /// Resolved per-call cap for PayloadSynthesis. Falls back to the
    /// built-in default when the operator did not set
    /// `[ai] payload_synthesis_per_call_cap_usd_micros` or set a
    /// non-positive value.
    pub fn payload_synthesis_per_call_cap_usd_micros_resolved(&self) -> i64 {
        match self.payload_synthesis_per_call_cap_usd_micros {
            Some(v) if v > 0 => v,
            _ => Self::DEFAULT_RUN_BUDGET_USD_MICROS,
        }
    }

    /// Resolved per-call cap for SpecDerivation. Same fall-back rules
    /// as `payload_synthesis_per_call_cap_usd_micros_resolved`.
    pub fn spec_derivation_per_call_cap_usd_micros_resolved(&self) -> i64 {
        match self.spec_derivation_per_call_cap_usd_micros {
            Some(v) if v > 0 => v,
            _ => Self::DEFAULT_RUN_BUDGET_USD_MICROS,
        }
    }

    /// Resolved per-call cap for ChainReasoning. Same fall-back rules.
    pub fn chain_reasoning_per_call_cap_usd_micros_resolved(&self) -> i64 {
        match self.chain_reasoning_per_call_cap_usd_micros {
            Some(v) if v > 0 => v,
            _ => Self::DEFAULT_RUN_BUDGET_USD_MICROS,
        }
    }

    /// Resolved per-call cap for NovelFindingDiscovery batches. Same
    /// fall-back rules; the per-call cap defaults to the run cap so a
    /// single batch may consume the entire bucket when no earlier pass
    /// has spent yet.
    pub fn novel_discovery_per_call_cap_usd_micros_resolved(&self) -> i64 {
        match self.novel_discovery_per_call_cap_usd_micros {
            Some(v) if v > 0 => v,
            _ => Self::DEFAULT_RUN_BUDGET_USD_MICROS,
        }
    }

    /// Resolved per-task soft cap for AI Exploration. Falls back to
    /// the caller-supplied default when the operator did not set
    /// `[ai] exploration_soft_cap_usd_micros` or set a non-positive
    /// value. The default lives in the `nyx-agent-ai` crate
    /// (`DEFAULT_EXPLORATION_SOFT_CAP_USD_MICROS`); core does not
    /// depend on `nyx-agent-ai`, so the caller passes the value in.
    pub fn exploration_soft_cap_usd_micros_resolved(&self, default: i64) -> i64 {
        match self.exploration_soft_cap_usd_micros {
            Some(v) if v > 0 => v,
            _ => default,
        }
    }

    /// Resolved per-run hard cap for AI Exploration. Same fall-back
    /// rules as `exploration_soft_cap_usd_micros_resolved`.
    pub fn exploration_run_cap_usd_micros_resolved(&self, default: i64) -> i64 {
        match self.exploration_run_cap_usd_micros {
            Some(v) if v > 0 => v,
            _ => default,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AiRuntime {
    /// AI features off. Static scans, route modelling, live checks,
    /// evidence storage, and triage still run.
    #[default]
    None,
    /// Hosted Anthropic API. The wizard prompts for an API key and
    /// stashes it in the OS keychain under `secrets::ACCOUNT_AI_ANTHROPIC`.
    Anthropic,
    /// Local OpenAI-compatible runtime (LM Studio, Ollama, vLLM, ...).
    /// The endpoint URL goes in `api_base`; any embedded bearer goes
    /// in the keychain under `secrets::ACCOUNT_AI_LOCAL_LLM`.
    LocalLlm,
    /// Optional local adapter that drives an already-installed `claude`
    /// CLI on `$PATH`.
    ClaudeCode,
    /// Optional local adapter that drives an already-installed `codex`
    /// CLI on `$PATH`.
    Codex,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UiConfig {
    pub listen_addr: String,
    pub open_browser: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        // Plan: serve opens a browser on startup unless --no-open /
        // --headless. Default this to true so users who never write
        // `[ui].open_browser` keep the documented behaviour, and those
        // who set it to false in nyx-agent.toml suppress the launch.
        Self { listen_addr: "127.0.0.1:8765".to_string(), open_browser: true }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NyxConfig {
    /// Override the discovered `nyx` binary. When `None`, the runner falls
    /// back to a `PATH` lookup.
    pub binary_path: Option<PathBuf>,
    /// Override the built-in minimum-supported `nyx` version. Useful in
    /// integration tests; production deployments should leave it unset.
    pub min_version: Option<String>,
}

/// `[env]` section: env-builder (docker-compose) knobs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnvConfig {
    /// Image-pull policy forwarded to `docker compose up --pull <policy>`.
    /// `None` falls back to [`EnvPullPolicy::default`] (`Missing`: pull
    /// only when the local store is missing the image), matching the
    /// docker daemon's own default. Operators on a CI lane with a warm
    /// image cache can set `[env] pull_policy = "never"` to skip the
    /// per-spin-up pull RTT.
    pub pull_policy: Option<EnvPullPolicy>,
}

impl EnvConfig {
    /// Resolved image-pull policy: the operator override when set,
    /// otherwise [`EnvPullPolicy::default`] (`Missing`).
    pub fn pull_policy_resolved(&self) -> EnvPullPolicy {
        self.pull_policy.unwrap_or_default()
    }
}

/// Operator-friendly mirror of `nyx_agent_sandbox::env::PullPolicy`.
/// Defined here so the `[env]` toml block parses without
/// `nyx-agent-core` taking a reverse dep on the sandbox crate. The
/// binary glue converts to the runtime type at the EnvBuilder seam.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EnvPullPolicy {
    /// Pull only when the local store does not have the image.
    #[default]
    Missing,
    /// Re-pull on every spin-up.
    Always,
    /// Never pull; fail spin-up if the image is missing locally.
    Never,
}

fn default_optional_scanner_enabled() -> bool {
    true
}

/// `[run]` section: verifier knobs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    /// When `true`, the deterministic payload runner re-executes each
    /// (vuln, benign) pair a second time and stamps `replay_stable` on
    /// the resulting `VerifyResult`. Adds ~2× cost per verify; default
    /// is `false` so the verifier stays fast on the happy path.
    pub replay_stable_check: bool,
    /// Allow live verification plans to send methods that are likely to
    /// mutate target state (`POST`, `PUT`, `PATCH`, `DELETE`). Defaults
    /// to false so Nyx Agent only runs safe probes unless the operator
    /// explicitly opts in for their local app.
    #[serde(default)]
    pub allow_state_changing_live_probes: bool,
    /// Opt in to browser-driven checks when a local Playwright runtime is
    /// available. When false, browser plans are recorded as skipped with
    /// an explicit reason.
    #[serde(default)]
    pub browser_checks_enabled: bool,
    /// Master opt-in for exploit mode. Defaults to false; when false,
    /// live verification stays non-destructive even if an older config
    /// sets `allow_state_changing_live_probes = true`.
    #[serde(default)]
    pub exploit_mode_enabled: bool,
    /// Evaluate guarded live probes and write audit records without
    /// sending HTTP/browser traffic. Static analysis and source-only
    /// scanners still run normally.
    #[serde(default)]
    pub exploit_dry_run: bool,
    /// Generate first-class business-logic pentest candidates from the
    /// route/auth model. The generated plans still pass through the
    /// normal live-verifier safety gates.
    #[serde(default = "default_true")]
    pub business_logic_templates_enabled: bool,
    /// Enable deeper authorized product-logic research. This adds
    /// invariant-focused candidate hypotheses and gives AI planning /
    /// exploration a broader product-logic brief. It does not relax
    /// live execution safety gates.
    #[serde(default)]
    pub research_mode_enabled: bool,
    /// Enable the pre-MVP unsafe local attack-agent phase. Once this
    /// final phase is invoked it does not route actions through the
    /// guarded live-verifier policy; it relies on the configured local
    /// development environment and CLI-backed sandbox boundary.
    #[serde(default)]
    pub unsafe_attack_agent_enabled: bool,
    /// Enable the binary / CLI target pentest phase. Off unless the
    /// operator opts in (mirrors `unsafe_attack_agent_enabled`). When on,
    /// the pass drives any configured `local_binary` targets against
    /// agent-crafted malformed inputs inside the sandbox. The pass
    /// refuses to run on the unisolated `process` backend.
    #[serde(default)]
    pub binary_target_pentest_enabled: bool,
    /// Run-scoped local-binary targets for the binary-target pentest
    /// pass. Populated from the `--binary` CLI flag (and/or higher-level
    /// project config). Only used when `binary_target_pentest_enabled`.
    #[serde(default)]
    pub binary_targets: Vec<LocalBinaryTarget>,
    /// Optional allowlist of business-logic template ids. Empty means
    /// every registered template is considered.
    #[serde(default)]
    pub business_logic_template_ids: Vec<String>,
    /// Per-candidate cap on guarded live HTTP/browser actions. `None`
    /// falls back to [`RunConfig::DEFAULT_EXPLOIT_REQUEST_CAP`]; a
    /// configured `0` is floored to `1`.
    #[serde(default)]
    pub exploit_request_cap: Option<u32>,
    /// Per-candidate rate limit for guarded live requests/actions.
    /// `None` falls back to
    /// [`RunConfig::DEFAULT_EXPLOIT_REQUESTS_PER_SECOND`]; `0` floors
    /// to `1`.
    #[serde(default)]
    pub exploit_requests_per_second: Option<u32>,
    /// After an allowed state-changing probe, ask the environment
    /// orchestration layer to reset/rollback when it supports that
    /// operation. Defaults true so opt-in exploit runs clean up after
    /// themselves when docker-compose orchestration is available.
    #[serde(default = "default_true")]
    pub exploit_reset_after_state_changing: bool,
    /// Optional passive ZAP baseline orchestration. Enabled by default,
    /// but the binary is only used when present on PATH; findings become
    /// candidates.
    #[serde(default = "default_optional_scanner_enabled")]
    pub enable_zap_baseline: bool,
    /// Optional Nuclei orchestration. Enabled by default, but the binary
    /// is only used when present on PATH; findings become candidates.
    #[serde(default = "default_optional_scanner_enabled")]
    pub enable_nuclei: bool,
    /// Optional Trivy repository/filesystem scan. Enabled by default,
    /// but the binary is only used when present on PATH; findings become
    /// source-context candidates for AI exploration.
    #[serde(default = "default_optional_scanner_enabled")]
    pub enable_trivy: bool,
    /// Optional OSV-Scanner dependency scan. Enabled by default, but the
    /// binary is only used when present on PATH; findings become
    /// source-context candidates for AI exploration.
    #[serde(default = "default_optional_scanner_enabled")]
    pub enable_osv_scanner: bool,
    /// Optional secret scanning. Enabled by default; Nyx Agent prefers
    /// `gitleaks` when present and falls back to `detect-secrets`.
    #[serde(default = "default_optional_scanner_enabled")]
    pub enable_secret_scanning: bool,
    /// Optional Katana crawler orchestration. Enabled by default, but
    /// the binary is only used when present on PATH; sensitive routes
    /// become live-test candidates.
    #[serde(default = "default_optional_scanner_enabled")]
    pub enable_katana: bool,
    /// Optional ProjectDiscovery httpx probe orchestration. Enabled by
    /// default, but the binary is only used when present on PATH;
    /// interesting live metadata becomes candidates.
    #[serde(default = "default_optional_scanner_enabled")]
    pub enable_httpx: bool,
    /// Aggressive external tooling is off unless this explicit gate is
    /// true. Nyx Agent does not run sqlmap by default.
    #[serde(default)]
    pub enable_aggressive_sqlmap: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            replay_stable_check: false,
            allow_state_changing_live_probes: false,
            browser_checks_enabled: false,
            exploit_mode_enabled: false,
            exploit_dry_run: false,
            business_logic_templates_enabled: true,
            research_mode_enabled: false,
            unsafe_attack_agent_enabled: false,
            binary_target_pentest_enabled: false,
            binary_targets: Vec::new(),
            business_logic_template_ids: Vec::new(),
            exploit_request_cap: None,
            exploit_requests_per_second: None,
            exploit_reset_after_state_changing: true,
            enable_zap_baseline: true,
            enable_nuclei: true,
            enable_trivy: true,
            enable_osv_scanner: true,
            enable_secret_scanning: true,
            enable_katana: true,
            enable_httpx: true,
            enable_aggressive_sqlmap: false,
        }
    }
}

impl RunConfig {
    /// Default request/action cap for one candidate live-verification
    /// attempt. This keeps malformed or model-generated workflows from
    /// fanning out indefinitely.
    pub const DEFAULT_EXPLOIT_REQUEST_CAP: u32 = 10;
    /// Default per-candidate live request/action rate. Optional
    /// scanners have their own CLI-level throttles; this cap covers
    /// the built-in verifier.
    pub const DEFAULT_EXPLOIT_REQUESTS_PER_SECOND: u32 = 5;

    pub fn exploit_request_cap_resolved(&self) -> u32 {
        self.exploit_request_cap.unwrap_or(Self::DEFAULT_EXPLOIT_REQUEST_CAP).max(1)
    }

    pub fn exploit_requests_per_second_resolved(&self) -> u32 {
        self.exploit_requests_per_second.unwrap_or(Self::DEFAULT_EXPLOIT_REQUESTS_PER_SECOND).max(1)
    }

    pub fn state_changing_live_probes_allowed(&self) -> bool {
        self.exploit_mode_enabled && self.allow_state_changing_live_probes
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TriggersConfig {
    pub on_push: bool,
    pub on_pr: bool,
    pub schedule_cron: Option<String>,
    /// HMAC-SHA256 secret for `POST /webhook/git`. When
    /// unset, the webhook handler returns 503 so a misconfigured host
    /// cannot accept unauthenticated triggers.
    #[serde(default)]
    pub webhook_secret_ref: Option<String>,
    /// Optional branch filter for the webhook. When set, the handler
    /// only triggers a scan if the payload's branch ref matches.
    /// `None` accepts any branch.
    #[serde(default)]
    pub webhook_branch: Option<String>,
    /// Selects the body decoder for `POST /webhook/git`. Accepted
    /// values: `refheads` (default, covers GitHub / Gitea / Forgejo /
    /// Gogs / GitLab top-level `ref`), `bitbucket` (Bitbucket Server /
    /// Data Center `changes[].refId`), `sourcehut` (nested
    /// `event.refs[0].name`). Unknown values fall back to `refheads`
    /// with a warning so a typo never silently disables webhooks.
    #[serde(default)]
    pub webhook_provider: Option<String>,
    /// Cap on simultaneous in-flight `POST /webhook/git` handlers.
    /// `None` falls back to the in-crate default. Set to a small
    /// integer to bound the worst-case parallel HMAC + body-buffer
    /// cost a flood of valid-signed deliveries can impose on the
    /// daemon. Non-positive values fall back to the default.
    #[serde(default)]
    pub webhook_max_concurrent: Option<usize>,
    /// Per-source-IP token bucket size for `POST /webhook/git`,
    /// expressed in deliveries per minute. `None` falls back to the
    /// in-crate default. Non-positive values fall back to the
    /// default. The token-bucket burst depth matches this value so a
    /// fresh sender can fire that many requests back-to-back before
    /// throttling kicks in.
    #[serde(default)]
    pub webhook_rate_limit_per_minute: Option<u32>,
}

impl TriggersConfig {
    /// Resolved cap on simultaneous in-flight webhook handlers.
    /// Falls back to the crate-level default when the operator left
    /// the knob unset or stamped a non-positive value.
    pub fn webhook_max_concurrent_resolved(&self, default: usize) -> usize {
        match self.webhook_max_concurrent {
            Some(n) if n > 0 => n,
            _ => default,
        }
    }

    /// Resolved per-IP rate limit in deliveries per minute. Falls
    /// back to the crate-level default when the operator left the
    /// knob unset or stamped a non-positive value.
    pub fn webhook_rate_limit_per_minute_resolved(&self, default: u32) -> u32 {
        match self.webhook_rate_limit_per_minute {
            Some(n) if n > 0 => n,
            _ => default,
        }
    }
}

/// One `[[schedule]]` entry. A 5-field cron expression plus
/// an optional repo filter. When `repo` is `None` the scheduler runs
/// against every enabled repo (i.e. the same shape as the API's
/// manual-scan endpoint with no `repo=` query).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleConfig {
    /// 5-field cron expression (minute hour day-of-month month day-of-week).
    /// Example: `0 3 * * 1` = 03:00 every Monday.
    pub cron: String,
    /// Limit the run to a single configured repo. `None` scans every
    /// enabled repo.
    #[serde(default)]
    pub repo: Option<String>,
    /// Operator-readable label surfaced in tracing spans and the UI.
    /// Default `"scheduled"`.
    #[serde(default = "default_schedule_label")]
    pub label: String,
}

fn default_schedule_label() -> String {
    "scheduled".to_string()
}

/// A project groups one or more repos that belong to the same
/// product. Scan/run/env-builder/chain-runner operate per-project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectConfig {
    /// Unique project name. Used as the human-facing identifier and
    /// as the workspace directory prefix.
    pub name: String,
    /// Optional free-form description surfaced in the UI.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional base URL the sandbox env-builder dials when running
    /// dynamic checks against the running stack.
    #[serde(default)]
    pub target_base_url: Option<String>,
    /// Optional structured env overrides merged into the project's
    /// docker-compose / sandbox runtime. Stored as opaque TOML so each
    /// stack can carry whatever keys it needs.
    #[serde(default)]
    pub env_config: Option<toml::Value>,
    /// Optional first-class launch recipe for local-app orchestration.
    /// When set, CLI/API scans start this profile before Nyx/static
    /// analysis and stop it after the pentest run.
    #[serde(default)]
    pub launch: Option<ProjectLaunchConfig>,
    /// Optional auth/session profile. Auth material is referenced via
    /// env vars or session files; raw secrets should not be stored in
    /// TOML.
    #[serde(default)]
    pub runtime_profile: Option<ProjectRuntimeProfile>,
    /// Repos that belong to this project. Use `[[project.repo]]`
    /// blocks in TOML.
    #[serde(rename = "repo", default)]
    pub repos: Vec<RepoConfig>,
    /// Explicit pentest targets. Each `[[project.target]]` block is a
    /// [`PentestTarget`] (`kind = "http_app"` or `kind = "local_binary"`).
    /// Backward compatible: when absent, the project synthesises a single
    /// [`PentestTarget::HttpApp`] from the launch profile / `target_base_url`
    /// (see [`ProjectConfig::resolved_targets`]). The legacy `target_urls`
    /// path is left untouched.
    #[serde(rename = "target", default)]
    pub targets: Vec<PentestTarget>,
}

impl ProjectConfig {
    /// Resolve the run's targets. Returns the explicitly-configured
    /// `[[project.target]]` blocks verbatim when present; otherwise
    /// synthesises a single [`PentestTarget::HttpApp`] from the launch
    /// profile's `target_urls` (falling back to `target_base_url`) so
    /// existing HTTP-only projects keep working with no config change.
    pub fn resolved_targets(&self) -> Vec<PentestTarget> {
        if !self.targets.is_empty() {
            return self.targets.clone();
        }
        let mut urls: Vec<String> = self
            .launch
            .as_ref()
            .map(|l| l.target_urls.clone())
            .unwrap_or_default()
            .into_iter()
            .filter(|u| !u.trim().is_empty())
            .collect();
        if urls.is_empty() {
            if let Some(base) = self.target_base_url.as_deref().filter(|u| !u.trim().is_empty()) {
                urls.push(base.to_string());
            }
        }
        vec![PentestTarget::HttpApp { urls }]
    }

    /// Local-binary targets only (the subset the binary-target pentest
    /// pass drives). Empty when the project has no binary targets.
    pub fn local_binary_targets(&self) -> Vec<LocalBinaryTarget> {
        self.resolved_targets()
            .into_iter()
            .filter_map(|t| match t {
                PentestTarget::LocalBinary(b) => Some(b),
                PentestTarget::HttpApp { .. } => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectLaunchConfig {
    pub name: Option<String>,
    /// `auto`, `already-running`, `custom-commands`, or `docker-compose`.
    /// `auto` is resolved before the profile is persisted.
    pub mode: Option<String>,
    pub target_urls: Vec<String>,
    #[serde(default)]
    pub build: Vec<ProjectLaunchCommandConfig>,
    #[serde(default)]
    pub start: Vec<ProjectLaunchCommandConfig>,
    #[serde(default)]
    pub seed: Vec<ProjectLaunchCommandConfig>,
    #[serde(default)]
    pub reset: Vec<ProjectLaunchCommandConfig>,
    #[serde(default)]
    pub login: Vec<ProjectLaunchCommandConfig>,
    #[serde(default)]
    pub stop: Vec<ProjectLaunchCommandConfig>,
    #[serde(default)]
    pub health: Vec<ProjectLaunchHealthConfig>,
    #[serde(default)]
    pub env_files: Vec<String>,
    #[serde(default)]
    pub env_vars: Vec<ProjectLaunchEnvVarConfig>,
}

impl ProjectLaunchConfig {
    pub fn to_profile_input(&self, fallback_target_url: Option<&str>) -> ProjectLaunchProfileInput {
        let mut target_urls = self.target_urls.clone();
        if target_urls.is_empty() {
            if let Some(target) = fallback_target_url.filter(|target| !target.trim().is_empty()) {
                target_urls.push(target.to_string());
            }
        }
        let env_refs = self
            .env_files
            .iter()
            .filter(|path| !path.trim().is_empty())
            .map(|path| LaunchEnvRef {
                kind: "env-file".to_string(),
                value: path.trim().to_string(),
                secret: true,
            })
            .chain(self.env_vars.iter().filter_map(ProjectLaunchEnvVarConfig::to_env_ref))
            .collect();
        ProjectLaunchProfileInput {
            name: self.name.clone(),
            mode: self.mode.clone(),
            build_steps: self.build.iter().map(ProjectLaunchCommandConfig::to_step).collect(),
            start_steps: self.start.iter().map(ProjectLaunchCommandConfig::to_step).collect(),
            seed_steps: self.seed.iter().map(ProjectLaunchCommandConfig::to_step).collect(),
            reset_steps: self.reset.iter().map(ProjectLaunchCommandConfig::to_step).collect(),
            login_steps: self.login.iter().map(ProjectLaunchCommandConfig::to_step).collect(),
            stop_steps: self.stop.iter().map(ProjectLaunchCommandConfig::to_step).collect(),
            health_checks: self.health.iter().map(ProjectLaunchHealthConfig::to_check).collect(),
            target_urls,
            env_refs,
            working_dirs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectLaunchCommandConfig {
    pub command: String,
    #[serde(default, alias = "repo")]
    pub repo_name: Option<String>,
    #[serde(default)]
    pub working_directory: Option<String>,
    #[serde(default, alias = "timeout_secs")]
    pub timeout_seconds: Option<u64>,
    #[serde(default, alias = "input")]
    pub stdin: Option<String>,
}

impl ProjectLaunchCommandConfig {
    pub fn to_step(&self) -> LaunchStep {
        LaunchStep {
            command: self.command.clone(),
            repo_id: None,
            repo_name: self.repo_name.clone(),
            working_directory: self.working_directory.clone(),
            timeout_seconds: self.timeout_seconds,
            stdin: self.stdin.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectLaunchHealthConfig {
    pub kind: Option<String>,
    pub url: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub command: Option<ProjectLaunchCommandConfig>,
    #[serde(alias = "timeout_secs")]
    pub timeout_seconds: Option<u64>,
}

impl ProjectLaunchHealthConfig {
    pub fn to_check(&self) -> LaunchHealthCheck {
        LaunchHealthCheck {
            kind: self.kind.clone().unwrap_or_else(|| {
                if self.command.is_some() {
                    "command".to_string()
                } else {
                    "http".to_string()
                }
            }),
            url: self.url.clone(),
            host: self.host.clone(),
            port: self.port,
            command: self.command.as_ref().map(ProjectLaunchCommandConfig::to_step),
            timeout_seconds: self.timeout_seconds,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectLaunchEnvVarConfig {
    pub name: String,
    pub secret: bool,
}

impl Default for ProjectLaunchEnvVarConfig {
    fn default() -> Self {
        Self { name: String::new(), secret: true }
    }
}

impl ProjectLaunchEnvVarConfig {
    fn to_env_ref(&self) -> Option<LaunchEnvRef> {
        let name = self.name.trim();
        (!name.is_empty()).then(|| LaunchEnvRef {
            kind: "env-var".to_string(),
            value: name.to_string(),
            secret: self.secret,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepoConfig {
    pub name: String,
    /// Operator attestation that they own this repo and consent to scanning.
    /// The daemon refuses to ingest a repo without `i_own_this = true`.
    #[serde(default)]
    pub i_own_this: bool,
    pub source: RepoSourceConfig,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum RepoSourceConfig {
    /// Read-only clone of a remote git URL into `<state>/repos/<name>/`.
    Git {
        url: String,
        #[serde(default)]
        branch: Option<String>,
        /// Auth descriptor: `ssh-key:<path>`, `token-env:<var>`, `gh-app:<id>`.
        #[serde(default)]
        auth: Option<String>,
    },
    /// Read-only snapshot of a directory already present on disk.
    LocalPath { path: PathBuf },
}

fn default_true() -> bool {
    true
}

impl Config {
    #[tracing::instrument(skip_all, fields(path = %path.display()))]
    pub fn load_from(path: &Path) -> Result<Self, ConfigError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|source| ConfigError::Read { path: path.to_path_buf(), source })?;
        Self::parse(&raw, path)
    }

    #[tracing::instrument(skip_all, fields(path = %path.display()))]
    pub fn load_or_default(path: &Path) -> Result<Self, ConfigError> {
        match std::fs::read_to_string(path) {
            Ok(raw) => Self::parse(&raw, path),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(ConfigError::Read { path: path.to_path_buf(), source }),
        }
    }

    pub fn parse(raw: &str, path: &Path) -> Result<Self, ConfigError> {
        toml::from_str(raw)
            .map_err(|source| ConfigError::Parse { path: path.to_path_buf(), source })
    }

    pub fn to_toml_string(&self) -> Result<String, ConfigError> {
        Ok(toml::to_string_pretty(self)?)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn defaults_roundtrip_through_toml() {
        let cfg = Config::default();
        let rendered = cfg.to_toml_string().expect("serialise defaults");
        let parsed = Config::parse(&rendered, &PathBuf::from("<test>")).expect("parse defaults");
        assert_eq!(parsed, cfg);
    }

    #[test]
    fn run_defaults_enable_recommended_optional_scanners() {
        let cfg = Config::parse("[run]\n", &PathBuf::from("<test>")).expect("parse run defaults");
        assert!(cfg.run.enable_zap_baseline);
        assert!(cfg.run.enable_nuclei);
        assert!(cfg.run.enable_trivy);
        assert!(cfg.run.enable_osv_scanner);
        assert!(cfg.run.enable_secret_scanning);
        assert!(cfg.run.enable_katana);
        assert!(cfg.run.enable_httpx);
        assert!(!cfg.run.enable_aggressive_sqlmap);
        assert!(!cfg.run.exploit_mode_enabled);
        assert!(!cfg.run.exploit_dry_run);
        assert!(cfg.run.business_logic_templates_enabled);
        assert!(cfg.run.business_logic_template_ids.is_empty());
        assert_eq!(cfg.run.exploit_request_cap_resolved(), RunConfig::DEFAULT_EXPLOIT_REQUEST_CAP);
        assert_eq!(
            cfg.run.exploit_requests_per_second_resolved(),
            RunConfig::DEFAULT_EXPLOIT_REQUESTS_PER_SECOND
        );
        assert!(cfg.run.exploit_reset_after_state_changing);
        assert!(!cfg.run.state_changing_live_probes_allowed());
    }

    #[test]
    fn exploit_policy_config_parses_and_resolves() {
        let raw = r#"
[run]
exploit_mode_enabled = true
allow_state_changing_live_probes = true
exploit_dry_run = true
business_logic_templates_enabled = false
business_logic_template_ids = ["tenant_object_isolation", "webhook_callback_trust_boundary"]
exploit_request_cap = 0
exploit_requests_per_second = 0
exploit_reset_after_state_changing = false
"#;
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse exploit policy");
        assert!(cfg.run.exploit_mode_enabled);
        assert!(cfg.run.allow_state_changing_live_probes);
        assert!(cfg.run.state_changing_live_probes_allowed());
        assert!(cfg.run.exploit_dry_run);
        assert!(!cfg.run.business_logic_templates_enabled);
        assert_eq!(
            cfg.run.business_logic_template_ids,
            vec![
                "tenant_object_isolation".to_string(),
                "webhook_callback_trust_boundary".to_string()
            ]
        );
        assert_eq!(cfg.run.exploit_request_cap_resolved(), 1);
        assert_eq!(cfg.run.exploit_requests_per_second_resolved(), 1);
        assert!(!cfg.run.exploit_reset_after_state_changing);
    }

    #[test]
    fn populated_config_roundtrips() {
        let cfg = Config {
            general: GeneralConfig {
                log_level: "debug".to_string(),
                state_dir: Some(PathBuf::from("/tmp/nyx")),
            },
            performance: PerformanceConfig {
                max_parallel_scans: 8,
                scan_timeout_secs: 1200,
                static_concurrency: Some(2),
                per_repo_timeout_secs: Some(45),
                scheduler_tick_secs: Some(10),
                chain_lane_concurrency: Some(3),
                fast_lane_concurrency: Some(12),
            },
            sandbox: SandboxConfig {
                enabled: false,
                allow_network: true,
                backend: SandboxBackend::Birdcage,
            },
            ai: AiConfig {
                provider: Some("anthropic".to_string()),
                model: Some("claude-opus-4-7".to_string()),
                api_base: None,
                runtime: AiRuntime::Anthropic,
                max_concurrent_one_shot: 2,
                default_run_budget_usd_micros: None,
                pricing: {
                    let mut m = HashMap::new();
                    m.insert(
                        "claude-opus-4-7-20260101".to_string(),
                        AiPricingOverride {
                            input_per_mtok_usd: 12,
                            output_per_mtok_usd: 60,
                            cache_write_per_mtok_usd: 15,
                            cache_read_per_mtok_usd: 1,
                        },
                    );
                    m
                },
                payload_synthesis_per_call_cap_usd_micros: Some(2_500_000),
                spec_derivation_per_call_cap_usd_micros: Some(1_500_000),
                chain_reasoning_per_call_cap_usd_micros: Some(3_000_000),
                novel_discovery_per_call_cap_usd_micros: Some(4_000_000),
                exploration_soft_cap_usd_micros: Some(3_500_000),
                exploration_run_cap_usd_micros: Some(8_000_000),
                effort: None,
                context_window: None,
            },
            ui: UiConfig { listen_addr: "0.0.0.0:9999".to_string(), open_browser: true },
            triggers: TriggersConfig {
                on_push: true,
                on_pr: true,
                schedule_cron: Some("0 * * * *".to_string()),
                webhook_secret_ref: Some("env:NYX_WEBHOOK_SECRET".to_string()),
                webhook_branch: Some("main".to_string()),
                webhook_provider: Some("github".to_string()),
                webhook_max_concurrent: Some(4),
                webhook_rate_limit_per_minute: Some(60),
            },
            nyx: NyxConfig {
                binary_path: Some(PathBuf::from("/opt/nyx/bin/nyx")),
                min_version: Some("0.2.0".to_string()),
            },
            run: RunConfig {
                replay_stable_check: true,
                allow_state_changing_live_probes: true,
                browser_checks_enabled: true,
                exploit_mode_enabled: true,
                exploit_dry_run: true,
                exploit_request_cap: Some(3),
                exploit_requests_per_second: Some(2),
                exploit_reset_after_state_changing: false,
                ..RunConfig::default()
            },
            env: EnvConfig { pull_policy: Some(EnvPullPolicy::Never) },
            projects: vec![ProjectConfig {
                name: "acme-app".to_string(),
                description: Some("Acme web product".to_string()),
                target_base_url: Some("http://localhost:3000".to_string()),
                env_config: None,
                launch: None,
                runtime_profile: None,
                targets: Vec::new(),
                repos: vec![
                    RepoConfig {
                        name: "acme-backend".to_string(),
                        i_own_this: true,
                        source: RepoSourceConfig::Git {
                            url: "git@github.com:acme/acme-backend.git".to_string(),
                            branch: Some("main".to_string()),
                            auth: Some("ssh-key:~/.ssh/work_ed25519".to_string()),
                        },
                        enabled: true,
                    },
                    RepoConfig {
                        name: "monolith".to_string(),
                        i_own_this: true,
                        source: RepoSourceConfig::LocalPath {
                            path: PathBuf::from("/Users/eli/code/monolith"),
                        },
                        enabled: true,
                    },
                ],
            }],
            schedules: vec![ScheduleConfig {
                cron: "0 3 * * 1".to_string(),
                repo: Some("acme-backend".to_string()),
                label: "weekly-monday-3am".to_string(),
            }],
        };
        let rendered = cfg.to_toml_string().expect("serialise");
        let parsed = Config::parse(&rendered, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(parsed, cfg);
    }

    #[test]
    fn missing_file_returns_default() {
        let path = PathBuf::from("/definitely/does/not/exist/nyx-agent.toml");
        let cfg = Config::load_or_default(&path).expect("missing file -> default");
        assert_eq!(cfg, Config::default());
    }

    #[test]
    fn empty_string_parses_to_default() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("empty parses");
        assert_eq!(cfg, Config::default());
    }

    #[test]
    fn repo_enabled_defaults_to_true_when_omitted() {
        let raw = "[[project]]\nname = \"p\"\n\n[[project.repo]]\nname = \"acme-backend\"\n\
                   i_own_this = true\n\
                   source = { kind = \"local-path\", path = \"/srv/repos/acme-backend\" }\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.projects.len(), 1);
        assert_eq!(cfg.projects[0].repos.len(), 1);
        assert!(
            cfg.projects[0].repos[0].enabled,
            "declared repo without explicit enabled must default to true"
        );
    }

    #[test]
    fn repo_source_git_parses_with_inline_table() {
        let raw = "[[project]]\nname = \"p\"\n\n[[project.repo]]\nname = \"billing\"\n\
                   i_own_this = true\n\
                   source = { kind = \"git\", url = \"git@github.com:org/billing.git\", \
                              branch = \"main\", auth = \"ssh-key:~/.ssh/work_ed25519\" }\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        match &cfg.projects[0].repos[0].source {
            RepoSourceConfig::Git { url, branch, auth } => {
                assert_eq!(url, "git@github.com:org/billing.git");
                assert_eq!(branch.as_deref(), Some("main"));
                assert_eq!(auth.as_deref(), Some("ssh-key:~/.ssh/work_ed25519"));
            }
            other => panic!("expected git source, got {other:?}"),
        }
    }

    #[test]
    fn repo_source_local_path_parses() {
        let raw = "[[project]]\nname = \"p\"\n\n[[project.repo]]\nname = \"monolith\"\n\
                   i_own_this = true\n\
                   source = { kind = \"local-path\", path = \"/home/eli/code/monolith\" }\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        match &cfg.projects[0].repos[0].source {
            RepoSourceConfig::LocalPath { path } => {
                assert_eq!(path, &PathBuf::from("/home/eli/code/monolith"));
            }
            other => panic!("expected local-path source, got {other:?}"),
        }
    }

    #[test]
    fn repo_source_unknown_kind_rejected() {
        let raw = "[[project]]\nname = \"p\"\n\n[[project.repo]]\nname = \"x\"\n\
                   i_own_this = true\n\
                   source = { kind = \"hg\", path = \"/srv/x\" }\n";
        let err = Config::parse(raw, &PathBuf::from("<test>")).expect_err("must reject");
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn repo_i_own_this_defaults_to_false_when_omitted() {
        let raw = "[[project]]\nname = \"p\"\n\n[[project.repo]]\nname = \"x\"\n\
                   source = { kind = \"local-path\", path = \"/srv/x\" }\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert!(
            !cfg.projects[0].repos[0].i_own_this,
            "i_own_this must default to false so the daemon refuses unattested repos"
        );
    }

    #[test]
    fn top_level_repo_block_rejected() {
        // Bare `[[repo]]` is no longer accepted. The TOML must
        // declare a `[[project]]` first and nest repos under it.
        let raw = "[[repo]]\nname = \"x\"\ni_own_this = true\n\
                   source = { kind = \"local-path\", path = \"/srv/x\" }\n";
        let err = Config::parse(raw, &PathBuf::from("<test>")).expect_err("must reject");
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn project_groups_multiple_repos() {
        let raw = "[[project]]\nname = \"acme\"\ndescription = \"Acme product\"\n\
                   target_base_url = \"http://localhost:3000\"\n\n\
                   [[project.repo]]\nname = \"acme-backend\"\ni_own_this = true\nenabled = true\n\
                   source = { kind = \"local-path\", path = \"/p/backend\" }\n\n\
                   [[project.repo]]\nname = \"acme-frontend\"\ni_own_this = true\nenabled = true\n\
                   source = { kind = \"local-path\", path = \"/p/frontend\" }\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.projects.len(), 1);
        let p = &cfg.projects[0];
        assert_eq!(p.name, "acme");
        assert_eq!(p.description.as_deref(), Some("Acme product"));
        assert_eq!(p.target_base_url.as_deref(), Some("http://localhost:3000"));
        assert_eq!(p.repos.len(), 2);
        assert_eq!(p.repos[0].name, "acme-backend");
        assert_eq!(p.repos[1].name, "acme-frontend");
    }

    #[test]
    fn project_launch_profile_parses_lifecycle_hooks() {
        let raw = r#"
            [[project]]
            name = "acme"
            target_base_url = "http://localhost:3000"

            [project.launch]
            mode = "custom-commands"
            env_files = [".env.test"]

            [[project.launch.build]]
            command = "npm ci"
            repo = "web"
            timeout_secs = 120

            [[project.launch.start]]
            command = "npm run dev"
            repo_name = "web"

            [[project.launch.seed]]
            command = "npm run seed:test"

            [[project.launch.reset]]
            command = "npm run db:reset"

            [[project.launch.login]]
            command = "npm run session:nyx-agent"

            [[project.launch.stop]]
            command = "npm run stop"

            [[project.launch.health]]
            url = "http://localhost:3000/health"
            timeout_seconds = 15

            [[project.launch.env_vars]]
            name = "NYX_AGENT_TEST_TOKEN"
            secret = true
        "#;
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        let launch = cfg.projects[0].launch.as_ref().expect("launch config");
        let input = launch.to_profile_input(cfg.projects[0].target_base_url.as_deref());

        assert_eq!(input.mode.as_deref(), Some("custom-commands"));
        assert_eq!(input.target_urls, vec!["http://localhost:3000"]);
        assert_eq!(input.build_steps[0].repo_name.as_deref(), Some("web"));
        assert_eq!(input.build_steps[0].timeout_seconds, Some(120));
        assert_eq!(input.seed_steps[0].command, "npm run seed:test");
        assert_eq!(input.reset_steps[0].command, "npm run db:reset");
        assert_eq!(input.login_steps[0].command, "npm run session:nyx-agent");
        assert_eq!(input.stop_steps[0].command, "npm run stop");
        assert_eq!(input.health_checks[0].url.as_deref(), Some("http://localhost:3000/health"));
        assert_eq!(input.env_refs.len(), 2);
    }

    #[test]
    fn unknown_field_rejected() {
        let raw = "garbage_field = true\n";
        let err = Config::parse(raw, &PathBuf::from("<test>")).expect_err("must reject");
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn performance_static_concurrency_and_per_repo_timeout_round_trip() {
        let raw = "[performance]\nstatic_concurrency = 3\nper_repo_timeout_secs = 5\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.performance.static_concurrency, Some(3));
        assert_eq!(cfg.performance.per_repo_timeout_secs, Some(5));
        assert_eq!(cfg.performance.per_repo_timeout(), std::time::Duration::from_secs(5));
        assert_eq!(cfg.performance.static_concurrency_override(), Some(3));
    }

    #[test]
    fn performance_omitted_overrides_fall_back() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("parse");
        assert!(cfg.performance.static_concurrency.is_none());
        assert!(cfg.performance.per_repo_timeout_secs.is_none());
        assert_eq!(cfg.performance.per_repo_timeout(), std::time::Duration::from_secs(30 * 60));
        assert!(cfg.performance.static_concurrency_override().is_none());
    }

    #[test]
    fn ai_max_concurrent_one_shot_default_is_four() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.ai.max_concurrent_one_shot, 4);
        assert_eq!(cfg.ai.max_concurrent_one_shot_resolved(), 4);
    }

    #[test]
    fn ai_max_concurrent_one_shot_zero_floors_to_one() {
        let raw = "[ai]\nmax_concurrent_one_shot = 0\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.ai.max_concurrent_one_shot, 0);
        assert_eq!(cfg.ai.max_concurrent_one_shot_resolved(), 1);
    }

    #[test]
    fn ai_max_concurrent_one_shot_roundtrips_through_toml() {
        let raw = "[ai]\nmax_concurrent_one_shot = 8\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.ai.max_concurrent_one_shot, 8);
        let rendered = cfg.to_toml_string().expect("ser");
        let back = Config::parse(&rendered, &PathBuf::from("<test>")).expect("roundtrip");
        assert_eq!(back.ai.max_concurrent_one_shot, 8);
    }

    #[test]
    fn performance_static_concurrency_zero_floors_to_one() {
        let cfg = Config {
            performance: PerformanceConfig {
                static_concurrency: Some(0),
                ..PerformanceConfig::default()
            },
            ..Config::default()
        };
        assert_eq!(cfg.performance.static_concurrency_override(), Some(1));
    }

    #[test]
    fn performance_scheduler_tick_defaults_to_sixty_seconds() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("parse");
        assert!(cfg.performance.scheduler_tick_secs.is_none());
        assert_eq!(cfg.performance.scheduler_tick(), std::time::Duration::from_secs(60));
    }

    #[test]
    fn performance_scheduler_tick_roundtrips_through_toml() {
        let raw = "[performance]\nscheduler_tick_secs = 5\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.performance.scheduler_tick_secs, Some(5));
        assert_eq!(cfg.performance.scheduler_tick(), std::time::Duration::from_secs(5));
    }

    #[test]
    fn performance_scheduler_tick_zero_floors_to_one_second() {
        let cfg = Config {
            performance: PerformanceConfig {
                scheduler_tick_secs: Some(0),
                ..PerformanceConfig::default()
            },
            ..Config::default()
        };
        assert_eq!(cfg.performance.scheduler_tick(), std::time::Duration::from_secs(1));
    }

    #[test]
    fn performance_lane_concurrency_defaults_match_sandbox_constants() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("parse");
        assert!(cfg.performance.chain_lane_concurrency.is_none());
        assert!(cfg.performance.fast_lane_concurrency.is_none());
        assert_eq!(cfg.performance.chain_lane_concurrency_resolved(), 2);
        assert_eq!(cfg.performance.fast_lane_concurrency_resolved(), 8);
    }

    #[test]
    fn performance_lane_concurrency_roundtrips_through_toml() {
        let raw = "[performance]\nchain_lane_concurrency = 4\nfast_lane_concurrency = 16\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.performance.chain_lane_concurrency, Some(4));
        assert_eq!(cfg.performance.fast_lane_concurrency, Some(16));
        assert_eq!(cfg.performance.chain_lane_concurrency_resolved(), 4);
        assert_eq!(cfg.performance.fast_lane_concurrency_resolved(), 16);
    }

    #[test]
    fn ai_pricing_override_defaults_empty() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("parse");
        assert!(cfg.ai.pricing.is_empty());
    }

    #[test]
    fn ai_pricing_override_parses_per_model_block() {
        let raw = "[ai.pricing.\"claude-opus-4-7\"]\n\
                   input_per_mtok_usd = 12\n\
                   output_per_mtok_usd = 60\n\
                   cache_write_per_mtok_usd = 15\n\
                   cache_read_per_mtok_usd = 1\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        let entry =
            cfg.ai.pricing.get("claude-opus-4-7").expect("override for claude-opus-4-7 must parse");
        assert_eq!(entry.input_per_mtok_usd, 12);
        assert_eq!(entry.output_per_mtok_usd, 60);
        assert_eq!(entry.cache_write_per_mtok_usd, 15);
        assert_eq!(entry.cache_read_per_mtok_usd, 1);
    }

    #[test]
    fn ai_pricing_override_omitted_cache_fields_default_to_zero() {
        let raw = "[ai.pricing.\"claude-haiku-4-5\"]\n\
                   input_per_mtok_usd = 1\n\
                   output_per_mtok_usd = 5\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        let entry = cfg.ai.pricing.get("claude-haiku-4-5").expect("override must parse");
        assert_eq!(entry.input_per_mtok_usd, 1);
        assert_eq!(entry.output_per_mtok_usd, 5);
        assert_eq!(entry.cache_write_per_mtok_usd, 0);
        assert_eq!(entry.cache_read_per_mtok_usd, 0);
    }

    #[test]
    fn ai_pricing_override_unknown_field_rejected() {
        let raw = "[ai.pricing.\"claude-haiku-4-5\"]\n\
                   input_per_mtok_usd = 1\n\
                   garbage = true\n";
        let err = Config::parse(raw, &PathBuf::from("<test>")).expect_err("must reject");
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn ai_run_budget_defaults_to_uncapped_sentinel() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("parse");
        assert!(cfg.ai.default_run_budget_usd_micros.is_none());
        assert_eq!(cfg.ai.default_run_budget_usd_micros_resolved(), i64::MAX);
    }

    #[test]
    fn ai_per_call_caps_default_to_run_budget_constant() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("parse");
        assert!(cfg.ai.payload_synthesis_per_call_cap_usd_micros.is_none());
        assert!(cfg.ai.spec_derivation_per_call_cap_usd_micros.is_none());
        assert!(cfg.ai.chain_reasoning_per_call_cap_usd_micros.is_none());
        assert!(cfg.ai.novel_discovery_per_call_cap_usd_micros.is_none());
        let fallback = AiConfig::DEFAULT_RUN_BUDGET_USD_MICROS;
        assert_eq!(cfg.ai.payload_synthesis_per_call_cap_usd_micros_resolved(), fallback);
        assert_eq!(cfg.ai.spec_derivation_per_call_cap_usd_micros_resolved(), fallback);
        assert_eq!(cfg.ai.chain_reasoning_per_call_cap_usd_micros_resolved(), fallback);
        assert_eq!(cfg.ai.novel_discovery_per_call_cap_usd_micros_resolved(), fallback);
    }

    #[test]
    fn ai_per_call_caps_parse_per_task_overrides() {
        let raw = "[ai]\n\
                   payload_synthesis_per_call_cap_usd_micros = 2500000\n\
                   spec_derivation_per_call_cap_usd_micros = 1500000\n\
                   chain_reasoning_per_call_cap_usd_micros = 3000000\n\
                   novel_discovery_per_call_cap_usd_micros = 4000000\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.ai.payload_synthesis_per_call_cap_usd_micros_resolved(), 2_500_000);
        assert_eq!(cfg.ai.spec_derivation_per_call_cap_usd_micros_resolved(), 1_500_000);
        assert_eq!(cfg.ai.chain_reasoning_per_call_cap_usd_micros_resolved(), 3_000_000);
        assert_eq!(cfg.ai.novel_discovery_per_call_cap_usd_micros_resolved(), 4_000_000);
    }

    #[test]
    fn ai_per_call_caps_non_positive_overrides_fall_back_to_default() {
        let raw = "[ai]\n\
                   payload_synthesis_per_call_cap_usd_micros = 0\n\
                   spec_derivation_per_call_cap_usd_micros = -1\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        let fallback = AiConfig::DEFAULT_RUN_BUDGET_USD_MICROS;
        assert_eq!(cfg.ai.payload_synthesis_per_call_cap_usd_micros_resolved(), fallback);
        assert_eq!(cfg.ai.spec_derivation_per_call_cap_usd_micros_resolved(), fallback);
    }

    #[test]
    fn ai_exploration_caps_default_to_caller_default() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("parse");
        assert!(cfg.ai.exploration_soft_cap_usd_micros.is_none());
        assert!(cfg.ai.exploration_run_cap_usd_micros.is_none());
        // Caller default round-trips through the resolved getter.
        assert_eq!(cfg.ai.exploration_soft_cap_usd_micros_resolved(5_000_000), 5_000_000);
        assert_eq!(cfg.ai.exploration_run_cap_usd_micros_resolved(10_000_000), 10_000_000);
    }

    #[test]
    fn ai_exploration_caps_parse_operator_overrides() {
        let raw = "[ai]\n\
                   exploration_soft_cap_usd_micros = 3500000\n\
                   exploration_run_cap_usd_micros = 8000000\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.ai.exploration_soft_cap_usd_micros_resolved(5_000_000), 3_500_000);
        assert_eq!(cfg.ai.exploration_run_cap_usd_micros_resolved(10_000_000), 8_000_000);
    }

    #[test]
    fn ai_exploration_caps_non_positive_overrides_fall_back_to_caller_default() {
        let raw = "[ai]\n\
                   exploration_soft_cap_usd_micros = 0\n\
                   exploration_run_cap_usd_micros = -1\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.ai.exploration_soft_cap_usd_micros_resolved(5_000_000), 5_000_000);
        assert_eq!(cfg.ai.exploration_run_cap_usd_micros_resolved(10_000_000), 10_000_000);
    }

    #[test]
    fn env_pull_policy_defaults_to_missing() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("parse");
        assert!(cfg.env.pull_policy.is_none());
        assert_eq!(cfg.env.pull_policy_resolved(), EnvPullPolicy::Missing);
    }

    #[test]
    fn env_pull_policy_parses_kebab_case_variants() {
        for (raw, expected) in [
            ("[env]\npull_policy = \"missing\"\n", EnvPullPolicy::Missing),
            ("[env]\npull_policy = \"always\"\n", EnvPullPolicy::Always),
            ("[env]\npull_policy = \"never\"\n", EnvPullPolicy::Never),
        ] {
            let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
            assert_eq!(cfg.env.pull_policy, Some(expected));
            assert_eq!(cfg.env.pull_policy_resolved(), expected);
        }
    }

    #[test]
    fn env_pull_policy_unknown_value_rejected() {
        let raw = "[env]\npull_policy = \"sometimes\"\n";
        let err = Config::parse(raw, &PathBuf::from("<test>")).expect_err("must reject");
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn env_pull_policy_unknown_field_rejected() {
        let raw = "[env]\nmystery = true\n";
        let err = Config::parse(raw, &PathBuf::from("<test>")).expect_err("must reject");
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn webhook_limit_knobs_default_to_none_and_fall_back_to_caller_default() {
        let cfg = Config::parse("", &PathBuf::from("<test>")).expect("parse");
        assert!(cfg.triggers.webhook_max_concurrent.is_none());
        assert!(cfg.triggers.webhook_rate_limit_per_minute.is_none());
        assert_eq!(cfg.triggers.webhook_max_concurrent_resolved(8), 8);
        assert_eq!(cfg.triggers.webhook_rate_limit_per_minute_resolved(30), 30);
    }

    #[test]
    fn webhook_limit_knobs_parse_operator_overrides() {
        let raw = "[triggers]\n\
                   webhook_max_concurrent = 16\n\
                   webhook_rate_limit_per_minute = 120\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.triggers.webhook_max_concurrent, Some(16));
        assert_eq!(cfg.triggers.webhook_rate_limit_per_minute, Some(120));
        assert_eq!(cfg.triggers.webhook_max_concurrent_resolved(8), 16);
        assert_eq!(cfg.triggers.webhook_rate_limit_per_minute_resolved(30), 120);
    }

    #[test]
    fn webhook_limit_knobs_non_positive_overrides_fall_back_to_default() {
        let raw = "[triggers]\n\
                   webhook_max_concurrent = 0\n\
                   webhook_rate_limit_per_minute = 0\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        assert_eq!(cfg.triggers.webhook_max_concurrent_resolved(8), 8);
        assert_eq!(cfg.triggers.webhook_rate_limit_per_minute_resolved(30), 30);
    }

    #[test]
    fn resolved_targets_synthesises_http_app_from_launch_urls() {
        let raw = "[[project]]\n\
                   name = \"demo\"\n\
                   [project.launch]\n\
                   target_urls = [\"http://localhost:3000\"]\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        let project = &cfg.projects[0];
        let targets = project.resolved_targets();
        assert_eq!(targets.len(), 1);
        match &targets[0] {
            PentestTarget::HttpApp { urls } => assert_eq!(urls, &["http://localhost:3000"]),
            other => panic!("expected synthesised HttpApp, got {other:?}"),
        }
        assert!(project.local_binary_targets().is_empty());
    }

    #[test]
    fn resolved_targets_parses_local_binary_block() {
        let raw = "[[project]]\n\
                   name = \"demo\"\n\
                   [[project.target]]\n\
                   kind = \"local_binary\"\n\
                   program = \"curl\"\n\
                   argv_template = [\"@ARG\", \"@FILE:input\"]\n";
        let cfg = Config::parse(raw, &PathBuf::from("<test>")).expect("parse");
        let project = &cfg.projects[0];
        let bins = project.local_binary_targets();
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].program, "curl");
        assert_eq!(bins[0].argv_template.as_deref().unwrap(), &["@ARG", "@FILE:input"]);
    }

    #[test]
    fn performance_lane_concurrency_zero_floors_to_one() {
        let cfg = Config {
            performance: PerformanceConfig {
                chain_lane_concurrency: Some(0),
                fast_lane_concurrency: Some(0),
                ..PerformanceConfig::default()
            },
            ..Config::default()
        };
        assert_eq!(cfg.performance.chain_lane_concurrency_resolved(), 1);
        assert_eq!(cfg.performance.fast_lane_concurrency_resolved(), 1);
    }
}
