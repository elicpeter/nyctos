//! AI runtime + agent-task pipeline glue.
//!
//! The module wires together:
//!
//! 1. [`BudgetStoreTracker`]: adapts `nyx-agent-core`'s SQLite
//!    `BudgetStore` to the `nyx-agent-ai::BudgetTracker` host port the
//!    adapters call on every successful round trip. The trait surface
//!    lives in `nyx-agent-ai`; the SQLite backend lives in
//!    `nyx-agent-core`; this binary owns the wiring.
//! 2. [`run_payload_synthesis_pass`]: scans a finished
//!    `RunBundle<Diag>` for diags carrying
//!    `Unsupported(NoPayloadsForCap)` and fans out one PayloadSynthesis
//!    task per finding. Concurrency is capped by
//!    `[ai] max_concurrent_one_shot`; spend is recorded against the
//!    run's `budgets` row.
//! 3. [`run_spec_derivation_pass`]: same shape as the payload pass
//!    but firing on `Inconclusive(SpecDerivationFailed)` diags.
//!    Successful outcomes land in the `harness_specs` table and the
//!    parent finding's `spec_id` back-link is stamped.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use nyx_agent_ai::{
    read_spec_excerpt, reduce_chain_graph, run_agentic_chain_reasoning, run_attack_agent,
    run_chain_reasoning, run_exploration, run_live_evidence_review, run_novel_findings,
    run_payload_synthesis, run_spec_derivation, AiRuntime, AnthropicSdkAdapter,
    AttackAgentKnownLead, AttackAgentOutcome, AttackAgentProfile, AttackAgentScope,
    AttackAgentVulnerability, AttackWorkspace, BudgetTracker, ChainReasoningOutcome,
    ChainReasoningWorkspace, ClaudeCodeAdapter, CodexCliAdapter, EscapeSuiteGate,
    EscapeSuiteVerdict, ExistingVulnerabilitySummary, ExplorationAuditEntry, ExplorationEndpoint,
    ExplorationFinding, ExplorationHaltReason, ExplorationKnownLead, ExplorationOutcome,
    ExplorationScope, GraphBudget, LiveEvidenceReviewInput, LiveEvidenceReviewOutput,
    LocalLlmAdapter, NovelFindingDiscoveryOutcome, PayloadSynthesisOutcome, Pricing,
    SharedBudgetTracker, SpecDerivationOutcome, DEFAULT_ATTACK_AGENT_MAX_TURNS,
    DEFAULT_ATTACK_AGENT_PROFILES, DEFAULT_EXPLORATION_RUN_CAP_USD_MICROS,
    DEFAULT_EXPLORATION_SOFT_CAP_USD_MICROS,
};
use nyx_agent_core::store::{
    canonical_risk_rating, clamp_risk_score, compact_memory_for_prompt, finding_id_hash,
    AgentTraceRecord, CandidateFindingRecord, CandidateStatus, ChainRecord, ExplorationMemoryInput,
    ExplorationMemoryRecord, FindingOrigin, FindingRecord, HarnessSpecRecord, PayloadRecord,
    PentestCandidateRecord, Store, TaskKind, VerificationAttemptRecord,
    VerifiedVulnerabilityRecord,
};
use nyx_agent_core::{
    ids::short_token, now_epoch_ms, AiConfig, AiRuntime as ConfigAiRuntime, RepoOutcome, RunBundle,
    RunConfig, SandboxBackend, SandboxConfig, SecretStore, WorkspaceHandle,
};
use nyx_agent_nyx::Diag;
use nyx_agent_sandbox::payload_runner::{
    HarnessSource, HarnessSpecInput, PayloadRun, PayloadRunner,
};
use nyx_agent_sandbox::BackendKind;
use nyx_agent_types::agent::{AgentTraceMetrics, AiError, Budget, BudgetKind, Prompt};
use nyx_agent_types::attack_graph::{
    NODE_CANDIDATE, NODE_SIGNAL, NODE_VERIFICATION_ATTEMPT, NODE_VERIFIED_VULNERABILITY,
};
use nyx_agent_types::chain::{
    ChainCandidate, ChainReasoningEdge, ChainReasoningInput, ChainReasoningNode,
    CHAIN_REASONING_DEFAULT_MAX, CHAIN_REASONING_PROMPT_VERSION, NODE_KIND_ENTRY,
    NODE_KIND_FRAMEWORK, NODE_KIND_SINK,
};
use nyx_agent_types::event::{AgentEvent, EventSink, RunEvent, SandboxEvent};
use nyx_agent_types::live_plan::{
    EnvCapabilityReport, LiveTestPlan, NoPlanReason, NoPlanReasonCode,
};
use nyx_agent_types::novel::{
    FileForReview, NovelFindingDiscoveryInput, PriorFinding, DEFAULT_FILES_PER_BATCH,
    DEFAULT_NOVEL_DISCOVERY_RUN_CAP_USD_MICROS, NOVEL_FINDING_DISCOVERY_PROMPT_VERSION,
};
use nyx_agent_types::payload::{
    AttackProvenance, PayloadSynthesisInput, PAYLOAD_SYNTHESIS_PROMPT_VERSION,
};
use nyx_agent_types::product::RouteModel;
use nyx_agent_types::project::ProjectAuthProfile;
use nyx_agent_types::spec::{SpecDerivationInput, SPEC_DERIVATION_PROMPT_VERSION};
use nyx_agent_types::verify::{Oracle, VerifyResult, VerifyVerdict};
use tokio::sync::Semaphore;

use crate::{ai_terminal::wrap_runtime, live_planning, pentest_tools, route_model};

// Per-call PayloadSynthesis / SpecDerivation caps now live on
// `AiConfig` as
// `[ai] payload_synthesis_per_call_cap_usd_micros` /
// `[ai] spec_derivation_per_call_cap_usd_micros`. The tracker-side
// cap (resolved per-run from `[ai] default_run_budget_usd_micros`,
// falling back to `AiConfig::DEFAULT_RUN_BUDGET_USD_MICROS`) is the
// authoritative bucket the adapter checks against; the per-call
// value is informational on the wire and bounds a single call below
// the run cap when the operator wants tighter clamps.

/// Radius (in lines) of each excerpt the SpecDerivation prompt
/// receives. The vendored `HarnessSpec` only needs a few lines around
/// the call site, sink, and framework binding; a wide window would
/// blow the prompt budget without adding useful signal.
const SPEC_DERIVATION_EXCERPT_RADIUS: u32 = 4;

/// Maximum upstream files the SpecDerivation pre-fetch attaches to a
/// prompt. Capped at three relevant files (call site, sink, framework
/// binding) so the prompt envelope stays bounded.
const SPEC_DERIVATION_MAX_EXCERPTS: usize = 3;

const ATTACK_PLANNING_PROMPT_VERSION: &str = "phase-live.attack-planning.v1";
const ATTACK_PLANNING_RESEARCH_PROMPT_VERSION: &str = "phase-live.attack-planning.research.v1";
const LIVE_TEST_PLAN_PROMPT_VERSION: &str = "phase24.live_test_plan.v1";
const LIVE_TEST_PLAN_EXCERPT_RADIUS: u32 = 18;
const EXPLORATION_KNOWN_LEADS_MAX: usize = 24;

/// Pass-scoped budget-bucket namespaces. A pass with its own dedicated
/// cap knob (distinct from the run-wide `default_run_budget`) gets a
/// private `(run_id, kind@bucket)` row so its tighter cap cannot pin the
/// shared run-wide bucket that other agent-loop / one-shot passes draw
/// on. See [`BudgetStoreTracker::with_bucket`].
const EXPLORATION_BUDGET_BUCKET: &str = "exploration";
const NOVEL_DISCOVERY_BUDGET_BUCKET: &str = "novel-discovery";

/// Wall-clock cap for each unsafe attack-agent profile invocation.
///
/// The generic agent-loop adapters default to 15 minutes, but each
/// attack specialist needs a deeper budget because the phase runs
/// serially and failed profiles should not be clipped at the adapter
/// default.
const ATTACK_AGENT_PER_PASS_TIMEOUT_SECS: u64 = 30 * 60;

/// `BudgetTracker` impl backed by the SQLite `budgets` table.
///
/// `(run_id, kind)` rows are inserted lazily on first observation so
/// callers do not have to pre-seed them. The lazy init goes through
/// `BudgetStore::ensure_default` (`INSERT OR IGNORE`), which is a
/// single SQL statement so concurrent fan-out tasks cannot clobber a
/// peer's `spent_usd_micros`. Subsequent `add_spend` calls take the
/// `UPDATE ... RETURNING` fast path inside `BudgetStore::add_spend`.
pub struct BudgetStoreTracker {
    store: Store,
    default_cap_usd_micros: i64,
    /// Optional pass-scoped namespace appended to the stored `kind`
    /// column so a pass with its own dedicated cap knob does not share
    /// (and pollute) the run-wide `(run_id, kind)` bucket. `None` keeps
    /// the bare kind string, i.e. the shared run-wide budget bucket.
    ///
    /// This rides the `kind` dimension rather than `run_id` because
    /// `budgets.run_id` carries a `REFERENCES runs(id)` foreign key; a
    /// namespaced run_id would fail the constraint, whereas `kind` is
    /// free text. The `run_id` carried in `BudgetTick` events therefore
    /// stays the real run id and event routing is unaffected.
    bucket: Option<String>,
}

impl BudgetStoreTracker {
    pub fn new(store: Store, default_cap_usd_micros: i64) -> Self {
        Self { store, default_cap_usd_micros, bucket: None }
    }

    /// Namespace this tracker's budget bucket so the pass keeps its own
    /// cap and spend independent of the shared run-wide bucket. Passes
    /// that draw on the run-wide `default_run_budget` leave this unset.
    pub fn with_bucket(mut self, bucket: impl Into<String>) -> Self {
        self.bucket = Some(bucket.into());
        self
    }

    fn kind_str(kind: BudgetKind) -> &'static str {
        match kind {
            BudgetKind::OneShot => "OneShot",
            BudgetKind::AgentLoop => "AgentLoop",
            BudgetKind::Total => "Total",
        }
    }

    /// Storage key for the `kind` column: the bare kind string, or
    /// `"<kind>@<bucket>"` when this tracker is pass-scoped.
    fn keyed_kind(&self, kind: BudgetKind) -> String {
        match &self.bucket {
            Some(bucket) => format!("{}@{}", Self::kind_str(kind), bucket),
            None => Self::kind_str(kind).to_string(),
        }
    }

    async fn ensure_row(&self, run_id: &str, kind: BudgetKind) -> Result<(), AiError> {
        self.store
            .budgets()
            .ensure_default(run_id, &self.keyed_kind(kind), self.default_cap_usd_micros)
            .await
            .map_err(store_err)
    }
}

fn store_err(e: nyx_agent_core::StoreError) -> AiError {
    AiError::BudgetTracker(format!("{e}"))
}

fn ai_error_should_halt_pass(err: &AiError) -> bool {
    matches!(
        err,
        AiError::BudgetExceeded { .. }
            | AiError::UpstreamRefused(_)
            | AiError::Transport(_)
            | AiError::AdapterUnavailable(_)
    )
}

/// Convert `[ai.pricing.<model>]` overrides from `AiConfig` into the
/// `HashMap<String, Pricing>` shape the Anthropic adapter consumes.
/// Operator-friendly per-million-token USD rates collapse to
/// micros-per-token via [`Pricing::from_per_mtok_usd`]. Returns an
/// empty map when the operator has not declared any overrides; the
/// adapter then falls back to its built-in pricing table.
fn pricing_overrides_from_config(config: &AiConfig) -> HashMap<String, Pricing> {
    config
        .pricing
        .iter()
        .map(|(model, override_)| {
            (
                model.clone(),
                Pricing::from_per_mtok_usd(
                    override_.input_per_mtok_usd,
                    override_.output_per_mtok_usd,
                    override_.cache_write_per_mtok_usd,
                    override_.cache_read_per_mtok_usd,
                ),
            )
        })
        .collect()
}

async fn selected_one_shot_runtime(
    config: &AiConfig,
    store: &Store,
    secrets: &SecretStore,
    default_cap_usd_micros: i64,
    pass_name: &str,
    bucket: Option<&str>,
) -> anyhow::Result<Option<Arc<dyn AiRuntime>>> {
    let mut tracker_impl = BudgetStoreTracker::new(store.clone(), default_cap_usd_micros);
    if let Some(bucket) = bucket {
        tracker_impl = tracker_impl.with_bucket(bucket);
    }
    let tracker: SharedBudgetTracker = Arc::new(tracker_impl);
    match config.runtime {
        ConfigAiRuntime::None => Ok(None),
        ConfigAiRuntime::LocalLlm => {
            let Some(api_base) =
                config.api_base.as_deref().map(str::trim).filter(|v| !v.is_empty())
            else {
                tracing::info!(
                    pass = pass_name,
                    "selected local-llm runtime has no API base configured; skipping"
                );
                return Ok(None);
            };
            let bearer_token = match secrets.get(nyx_agent_core::secrets::ACCOUNT_AI_LOCAL_LLM) {
                Ok(token) => token,
                Err(err) => {
                    tracing::warn!(
                        pass = pass_name,
                        "local-llm bearer token unavailable ({err}); continuing without bearer"
                    );
                    None
                }
            };
            let mut adapter = LocalLlmAdapter::new(api_base.to_string(), bearer_token, tracker);
            if let Some(model) = &config.model {
                adapter = adapter.with_default_model(model.clone());
            }
            Ok(Some(wrap_runtime(adapter)))
        }
        ConfigAiRuntime::Anthropic => {
            let api_key = match secrets.get(nyx_agent_core::secrets::ACCOUNT_AI_ANTHROPIC) {
                Ok(Some(k)) => k,
                Ok(None) => {
                    tracing::info!(
                        pass = pass_name,
                        "selected Anthropic runtime has no API key configured; skipping"
                    );
                    return Ok(None);
                }
                Err(e) => return Err(anyhow::anyhow!("secret store error: {e}")),
            };
            let mut adapter = AnthropicSdkAdapter::new(api_key, tracker)
                .with_pricing_overrides(pricing_overrides_from_config(config));
            if let Some(model) = &config.model {
                adapter = adapter.with_default_model(model.clone());
            }
            Ok(Some(wrap_runtime(adapter)))
        }
        ConfigAiRuntime::ClaudeCode => {
            let mut adapter = match ClaudeCodeAdapter::discover(tracker).await {
                Ok(a) => a,
                Err(err) => {
                    tracing::info!(
                        pass = pass_name,
                        "selected Claude Code runtime unavailable ({err}); skipping"
                    );
                    return Ok(None);
                }
            };
            if let Some(model) = &config.model {
                adapter = adapter.with_default_model(model.clone());
            }
            if let Some(effort) = &config.effort {
                adapter = adapter.with_effort(effort.clone());
            }
            if let Some(context_window) = config.context_window {
                adapter = adapter.with_context_window(context_window);
            }
            Ok(Some(wrap_runtime(adapter)))
        }
        ConfigAiRuntime::Codex => {
            let mut adapter = match CodexCliAdapter::discover(tracker).await {
                Ok(a) => a,
                Err(err) => {
                    tracing::info!(
                        pass = pass_name,
                        "selected Codex runtime unavailable ({err}); skipping"
                    );
                    return Ok(None);
                }
            };
            if let Some(model) = &config.model {
                adapter = adapter.with_default_model(model.clone());
            }
            if let Some(effort) = &config.effort {
                adapter = adapter.with_effort(effort.clone());
            }
            if let Some(context_window) = config.context_window {
                adapter = adapter.with_context_window(context_window);
            }
            Ok(Some(wrap_runtime(adapter)))
        }
    }
}

async fn selected_agent_loop_runtime(
    config: &AiConfig,
    store: &Store,
    run_cap_usd_micros: i64,
    timeout: Option<std::time::Duration>,
    bucket: Option<&str>,
) -> Option<Arc<dyn AiRuntime>> {
    let mut tracker_impl = BudgetStoreTracker::new(store.clone(), run_cap_usd_micros);
    if let Some(bucket) = bucket {
        tracker_impl = tracker_impl.with_bucket(bucket);
    }
    let tracker: SharedBudgetTracker = Arc::new(tracker_impl);
    match config.runtime {
        ConfigAiRuntime::ClaudeCode => {
            let mut adapter = match ClaudeCodeAdapter::discover(tracker).await {
                Ok(a) => a,
                Err(err) => {
                    tracing::info!(
                        "selected Claude Code agent-loop runtime unavailable ({err}); skipping pass"
                    );
                    return None;
                }
            };
            if let Some(model) = &config.model {
                adapter = adapter.with_default_model(model.clone());
            }
            if let Some(effort) = &config.effort {
                adapter = adapter.with_effort(effort.clone());
            }
            if let Some(context_window) = config.context_window {
                adapter = adapter.with_context_window(context_window);
            }
            if let Some(timeout) = timeout {
                adapter = adapter.with_timeout(timeout);
            }
            Some(wrap_runtime(adapter))
        }
        ConfigAiRuntime::Codex => {
            let mut adapter = match CodexCliAdapter::discover(tracker).await {
                Ok(a) => a,
                Err(err) => {
                    tracing::info!(
                        "selected Codex agent-loop runtime unavailable ({err}); skipping pass"
                    );
                    return None;
                }
            };
            if let Some(model) = &config.model {
                adapter = adapter.with_default_model(model.clone());
            }
            if let Some(effort) = &config.effort {
                adapter = adapter.with_effort(effort.clone());
            }
            if let Some(context_window) = config.context_window {
                adapter = adapter.with_context_window(context_window);
            }
            if let Some(timeout) = timeout {
                adapter = adapter.with_timeout(timeout);
            }
            Some(wrap_runtime(adapter))
        }
        ConfigAiRuntime::Anthropic => {
            tracing::info!(
                "selected Anthropic API runtime does not support agent-loop tasks; skipping pass"
            );
            None
        }
        ConfigAiRuntime::LocalLlm => {
            tracing::info!(
                "selected local-llm runtime does not support agent-loop tasks; skipping pass"
            );
            None
        }
        ConfigAiRuntime::None => None,
    }
}

#[async_trait]
impl BudgetTracker for BudgetStoreTracker {
    async fn cap(&self, run_id: &str, kind: BudgetKind) -> Result<Option<i64>, AiError> {
        self.ensure_row(run_id, kind).await?;
        let row =
            self.store.budgets().get(run_id, &self.keyed_kind(kind)).await.map_err(store_err)?;
        Ok(row.map(|r| r.cap_usd_micros))
    }

    async fn current_spend(&self, run_id: &str, kind: BudgetKind) -> Result<i64, AiError> {
        self.ensure_row(run_id, kind).await?;
        let row =
            self.store.budgets().get(run_id, &self.keyed_kind(kind)).await.map_err(store_err)?;
        Ok(row.map(|r| r.spent_usd_micros).unwrap_or(0))
    }

    async fn add_spend(&self, run_id: &str, kind: BudgetKind, micros: i64) -> Result<i64, AiError> {
        self.ensure_row(run_id, kind).await?;
        self.store
            .budgets()
            .add_spend(run_id, &self.keyed_kind(kind), micros)
            .await
            .map_err(store_err)
    }
}

/// Counts surfaced by [`run_payload_synthesis_pass`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PayloadSynthesisPassReport {
    pub synthesised: u32,
    pub quarantined: u32,
    pub failed: u32,
    pub total_attempts: u64,
    pub spend_usd_micros: i64,
}

/// Fan-out PayloadSynthesis across every `Unsupported(NoPayloadsForCap)`
/// finding in `bundle`. No-op (returns a default report) when the
/// selected runtime does not support one-shot tasks or is unavailable.
pub async fn run_payload_synthesis_pass(
    config: &AiConfig,
    store: &Store,
    secrets: &SecretStore,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    events: EventSink,
) -> anyhow::Result<PayloadSynthesisPassReport> {
    let inputs = build_inputs(bundle, workspaces);
    if inputs.is_empty() {
        return Ok(PayloadSynthesisPassReport::default());
    }
    let adapter = match selected_one_shot_runtime(
        config,
        store,
        secrets,
        config.default_run_budget_usd_micros_resolved(),
        "payload synthesis",
        None,
    )
    .await?
    {
        Some(adapter) => adapter,
        None => return Ok(PayloadSynthesisPassReport::default()),
    };
    tracing::info!(count = inputs.len(), "payload synthesis: fanning out");

    let semaphore = Arc::new(Semaphore::new(config.max_concurrent_one_shot_resolved()));
    let per_call_cap = config.payload_synthesis_per_call_cap_usd_micros_resolved();
    let runtime_name = adapter.name();
    let runtime_model = adapter.default_model().to_string();
    let mut handles = Vec::with_capacity(inputs.len());
    for input in inputs {
        let rt = Arc::clone(&adapter);
        let sem = Arc::clone(&semaphore);
        let sink = events.clone();
        handles.push(tokio::spawn(async move {
            let permit = sem.acquire_owned().await.expect("semaphore closed");
            let started_at = now_epoch_ms();
            let outcome = run_payload_synthesis(rt.as_ref(), &input, sink, per_call_cap).await;
            drop(permit);
            outcome.map(|o| (started_at, o))
        }));
    }

    let mut report = PayloadSynthesisPassReport::default();
    for handle in handles {
        match handle.await {
            Ok(Ok((started_at, outcome))) => {
                apply_outcome(store, outcome, &mut report, runtime_name, &runtime_model, started_at)
                    .await?
            }
            Ok(Err(err)) => {
                tracing::warn!(error = %err, "payload synthesis call failed");
                report.failed += 1;
            }
            Err(join) => {
                tracing::warn!(error = %join, "payload synthesis task join error");
                report.failed += 1;
            }
        }
    }
    Ok(report)
}

/// Walk `bundle` + `workspaces` and turn each `Unsupported(NoPayloadsForCap)`
/// diag into a `PayloadSynthesisInput`. Public to keep the inner
/// filter unit-testable without spinning up an adapter.
pub fn build_inputs(
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
) -> Vec<PayloadSynthesisInput> {
    let mut out = Vec::new();
    for repo_bundle in &bundle.per_repo {
        let RepoOutcome::Success(diags) = &repo_bundle.outcome else {
            continue;
        };
        let Some(workspace) = workspaces.get(&repo_bundle.repo) else {
            continue;
        };
        for diag in diags {
            if !diag.is_unsupported_no_payloads() {
                continue;
            }
            let line = i64::from(diag.line);
            let finding_id = nyx_agent_core::store::finding_id_hash(
                &repo_bundle.repo,
                &diag.path,
                Some(line),
                &diag.cap,
                &diag.rule,
            );
            let lang = infer_lang_for_file(workspace.workspace(), &diag.path);
            let sink_ctx = diag.sink_ctx(workspace.workspace());
            out.push(PayloadSynthesisInput {
                finding_id,
                run_id: bundle.run_id.clone(),
                cap: diag.cap.clone(),
                lang,
                sink_ctx,
            });
        }
    }
    out
}

async fn apply_outcome(
    store: &Store,
    outcome: PayloadSynthesisOutcome,
    report: &mut PayloadSynthesisPassReport,
    runtime_name: &str,
    runtime_model: &str,
    started_at_ms: i64,
) -> anyhow::Result<()> {
    let finished_at = now_epoch_ms();
    match outcome {
        PayloadSynthesisOutcome::Synthesised {
            finding_id,
            cap,
            lang,
            output,
            prompt_version,
            spent_usd_micros,
            attempts,
            metrics,
        } => {
            let provenance = AttackProvenance::LlmSynthesised.as_str().to_string();
            let rec = PayloadRecord {
                id: format!("payload-{finding_id}-{finished_at:x}-{}", short_token()),
                finding_id: finding_id.clone(),
                cap,
                lang,
                vuln_bytes: output.vuln_payload.into_bytes(),
                benign_bytes: Some(output.benign_payload.into_bytes()),
                oracle_blob: Some(output.vuln_oracle),
                attack_provenance: Some(provenance.clone()),
                prompt_version: Some(prompt_version.clone()),
                created_at: finished_at,
            };
            // Atomic dual-write so a partial failure of the finding
            // stamp does not leave an orphaned payload row behind.
            store
                .payloads()
                .insert_with_finding_provenance(&rec, &finding_id, &provenance, &prompt_version)
                .await?;
            let trace = build_trace_row(
                TaskKind::PayloadSynthesis,
                Some(finding_id),
                runtime_name,
                runtime_model,
                &prompt_version,
                spent_usd_micros,
                started_at_ms,
                finished_at,
                Some(&metrics),
            );
            persist_trace_row(store, trace).await;
            report.synthesised += 1;
            report.spend_usd_micros += spent_usd_micros;
            report.total_attempts += u64::from(attempts);
        }
        PayloadSynthesisOutcome::Quarantined {
            finding_id,
            reason,
            spent_usd_micros,
            attempts,
            metrics,
        } => {
            let blob = serde_json::json!({
                "kind": "PayloadSynthesisQuarantined",
                "task": "PayloadSynthesis",
                "reason": reason,
            })
            .to_string();
            store.findings().quarantine(&finding_id, &blob).await?;
            let trace = build_trace_row(
                TaskKind::PayloadSynthesis,
                Some(finding_id),
                runtime_name,
                runtime_model,
                PAYLOAD_SYNTHESIS_PROMPT_VERSION,
                spent_usd_micros,
                started_at_ms,
                finished_at,
                Some(&metrics),
            );
            persist_trace_row(store, trace).await;
            report.quarantined += 1;
            report.spend_usd_micros += spent_usd_micros;
            report.total_attempts += u64::from(attempts);
        }
    }
    Ok(())
}

/// Map a source path to a language tag the prompt can quote. Keeps the
/// table small (PayloadSynthesis only ships for the languages nyx
/// already supports); unknown extensions land as `unknown`.
pub fn infer_lang(path: &str) -> String {
    let lower = path.to_lowercase();
    let basename = lower.rsplit_once('/').map(|(_, b)| b).unwrap_or(&lower);
    let ext = basename.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    let lang = match ext {
        "py" => "python",
        "js" | "mjs" | "cjs" => "javascript",
        "ts" | "tsx" => "typescript",
        "go" => "go",
        "rs" => "rust",
        "java" => "java",
        "rb" => "ruby",
        "php" => "php",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" => "cpp",
        _ => "unknown",
    };
    lang.to_string()
}

/// Per-file language inference that prefers the extension table but falls
/// back to a shebang lookup when the file has no recognised extension.
/// Handles the bin/foo + `#!/usr/bin/env python3` case the bare
/// [`infer_lang`] cannot reach. Reads at most the first 256 bytes from
/// disk; unreadable / non-existent files degrade to the extension result.
pub fn infer_lang_for_file(workspace_root: &std::path::Path, rel_path: &str) -> String {
    let from_ext = infer_lang(rel_path);
    if from_ext != "unknown" {
        return from_ext;
    }
    let basename = rel_path.rsplit_once('/').map(|(_, b)| b).unwrap_or(rel_path);
    if basename.contains('.') {
        return from_ext;
    }
    let mut buf = [0_u8; 256];
    let abs = workspace_root.join(rel_path);
    let Ok(mut file) = std::fs::File::open(&abs) else {
        return from_ext;
    };
    use std::io::Read;
    let Ok(read) = file.read(&mut buf) else {
        return from_ext;
    };
    if read < 2 || &buf[..2] != b"#!" {
        return from_ext;
    }
    let line_end = buf[..read].iter().position(|&b| b == b'\n').unwrap_or(read);
    let line = String::from_utf8_lossy(&buf[..line_end]);
    lang_from_shebang(&line).unwrap_or(from_ext)
}

/// Parse a shebang line (without the trailing newline) into one of the
/// language tokens [`infer_lang`] can also return. Returns `None` when
/// the interpreter name is unrecognised. Handles three common shapes:
///   * `#!/usr/bin/python3 ...`         → basename of the first token
///   * `#!/usr/bin/env python3 ...`     → first non-flag token after `env`
///   * `#!/usr/bin/perl -w`             → ignore trailing flags
fn lang_from_shebang(line: &str) -> Option<String> {
    let lower = line.to_lowercase();
    let trimmed = lower.trim_start_matches("#!").trim_start();
    let mut tokens = trimmed.split_whitespace();
    let first = tokens.next()?;
    let first_leaf = first.rsplit('/').next().unwrap_or(first);
    let leaf = if first_leaf == "env" {
        tokens.find(|tok| !tok.starts_with('-')).map(|tok| tok.rsplit('/').next().unwrap_or(tok))?
    } else {
        first_leaf
    };
    let lang = if leaf.starts_with("python") {
        "python"
    } else if leaf == "node" || leaf == "nodejs" {
        "javascript"
    } else if leaf == "deno" {
        "typescript"
    } else if leaf == "ruby" {
        "ruby"
    } else if leaf == "php" {
        "php"
    } else if leaf == "perl" {
        "perl"
    } else {
        return None;
    };
    Some(lang.to_string())
}

/// Process-local sequence number so two trace rows minted in the same
/// millisecond produce distinct ids. Reset per process; the resulting
/// id is `trace-<task_kind>-<finished_ms hex>-<seq hex>`.
static TRACE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Build a fresh `AgentTraceRecord` describing one AI task call.
///
/// `metrics` carries the per-call `TokenUsage` + `CacheStats` + model
/// each task's `*Outcome` envelope lifted out of the underlying
/// `Response` / `AgentResult`. Pass `None` for deterministic callers
/// (e.g. the payload verifier) that have no AI metrics to record.
#[allow(clippy::too_many_arguments)]
fn build_trace_row(
    task_kind: TaskKind,
    finding_id: Option<String>,
    runtime_name: &str,
    model: &str,
    prompt_version: &str,
    spent_usd_micros: i64,
    started_at_ms: i64,
    finished_at_ms: i64,
    metrics: Option<&AgentTraceMetrics>,
) -> AgentTraceRecord {
    let seq = TRACE_SEQ.fetch_add(1, Ordering::Relaxed);
    let id = format!("trace-{}-{:x}-{:08x}", task_kind.as_str(), finished_at_ms, seq);
    let duration = (finished_at_ms - started_at_ms).max(0);
    let (tokens_in, tokens_out, cache_hits, cache_misses, resolved_model) = match metrics {
        Some(m) => {
            let cache = m.cache.unwrap_or_default();
            let model_str = m.model.clone().unwrap_or_else(|| model.to_string());
            (
                i64::from(m.usage.input_tokens),
                i64::from(m.usage.output_tokens),
                i64::from(cache.cache_read_tokens),
                i64::from(cache.cache_creation_tokens),
                model_str,
            )
        }
        None => (0, 0, 0, 0, model.to_string()),
    };
    AgentTraceRecord {
        id,
        finding_id,
        task_kind: task_kind.as_str().to_string(),
        runtime_name: runtime_name.to_string(),
        model: resolved_model,
        prompt_version: Some(prompt_version.to_string()),
        conversation_jsonl_path: None,
        tokens_in,
        tokens_out,
        cost_usd_micros: spent_usd_micros,
        cache_hits,
        cache_misses,
        duration_ms: Some(duration),
        started_at: started_at_ms,
        finished_at: Some(finished_at_ms),
        verifier_blob: None,
    }
}

async fn persist_trace_row(store: &Store, row: AgentTraceRecord) {
    let task_kind = row.task_kind.clone();
    let finding_id = row.finding_id.clone();
    if let Err(err) = store.agent_traces().insert(&row).await {
        tracing::warn!(
            error = %err,
            task_kind = %task_kind,
            finding_id = ?finding_id,
            "failed to persist agent trace row"
        );
    }
}

/// Counts surfaced by [`run_live_test_plan_synthesis_pass`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LiveTestPlanSynthesisPassReport {
    pub candidates_seen: u32,
    pub planned: u32,
    pub no_plan: u32,
    pub failed: u32,
    pub attempts: u32,
    pub spend_usd_micros: i64,
}

#[derive(Debug)]
struct AiLivePlanCandidate {
    candidate: PentestCandidateRecord,
    fallback_no_plan: Option<LiveTestPlan>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AttackPlanningPassReport {
    pub candidates_seen: u32,
    pub candidates_planned: u32,
    pub skipped: u32,
    pub failed: u32,
    pub attempts: u32,
    pub spend_usd_micros: i64,
    pub plan_context: Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub async fn run_attack_planning_pass(
    config: &AiConfig,
    run_config: &RunConfig,
    store: &Store,
    secrets: &SecretStore,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    route_model: &RouteModel,
    auth_profiles: &[ProjectAuthProfile],
    target_urls: &[String],
    events: EventSink,
) -> anyhow::Result<AttackPlanningPassReport> {
    let candidates = store.pentest_candidates().list_by_run(&bundle.run_id).await?;
    let mut candidates: Vec<_> = candidates
        .into_iter()
        .filter(|c| matches!(c.status.as_str(), "Proposed" | "NeedsLiveTest"))
        .collect();
    prioritize_candidates_for_planning(&mut candidates, run_config.research_mode_enabled);
    let mut report = AttackPlanningPassReport {
        candidates_seen: candidates.len() as u32,
        ..AttackPlanningPassReport::default()
    };
    if candidates.is_empty() {
        return Ok(report);
    }
    if target_urls.is_empty() {
        report.skipped = candidates.len() as u32;
        report.plan_context = Some("attack planning skipped: no live target URL".to_string());
        return Ok(report);
    }
    let adapter = match selected_one_shot_runtime(
        config,
        store,
        secrets,
        config.default_run_budget_usd_micros_resolved(),
        "attack planning",
        None,
    )
    .await?
    {
        Some(adapter) => adapter,
        None => {
            report.skipped = candidates.len() as u32;
            report.plan_context =
                Some("attack planning skipped: no configured one-shot AI runtime".to_string());
            return Ok(report);
        }
    };

    let started_at = now_epoch_ms();
    let prompt = build_attack_planning_prompt(
        &candidates,
        workspaces,
        route_model,
        auth_profiles,
        target_urls,
        run_config.research_mode_enabled,
    );
    let budget = Budget {
        run_id: bundle.run_id.clone(),
        kind: BudgetKind::OneShot,
        cap_usd_micros: config.default_run_budget_usd_micros_resolved(),
    };
    let runtime_name = adapter.name();
    let runtime_model = adapter.default_model().to_string();
    let resp = match adapter.one_shot(prompt, budget, events).await {
        Ok(resp) => resp,
        Err(err) => {
            report.failed = candidates.len() as u32;
            if matches!(err, AiError::BudgetExceeded { .. }) {
                report.plan_context =
                    Some("attack planning halted: AI budget exhausted".to_string());
                return Ok(report);
            }
            return Err(anyhow::anyhow!(err.to_string()));
        }
    };
    let finished_at = now_epoch_ms();
    report.attempts = 1;
    report.spend_usd_micros = resp.cost_usd_micros;
    report.candidates_planned = candidates.len() as u32;
    report.plan_context = Some(compact_attack_plan_context(&resp.content));

    let metrics = AgentTraceMetrics::from_response(&resp);
    let mut trace = build_trace_row(
        TaskKind::AttackPlanning,
        None,
        runtime_name,
        &runtime_model,
        &resp.prompt_version,
        resp.cost_usd_micros,
        started_at,
        finished_at,
        Some(&metrics),
    );
    trace.verifier_blob = Some(
        serde_json::json!({
            "kind": "attack_plan",
            "run_id": &bundle.run_id,
            "project_id": &bundle.project_id,
            "content": resp.content,
        })
        .to_string(),
    );
    persist_trace_row(store, trace).await;
    Ok(report)
}

fn build_attack_planning_prompt(
    candidates: &[PentestCandidateRecord],
    workspaces: &HashMap<String, WorkspaceHandle>,
    route_model: &RouteModel,
    auth_profiles: &[ProjectAuthProfile],
    target_urls: &[String],
    research_mode_enabled: bool,
) -> Prompt {
    let targets = target_urls.iter().map(|u| format!("- {u}")).collect::<Vec<_>>().join("\n");
    let routes = route_model::compact_route_model_for_prompt(route_model, 80);
    let auth = pentest_tools::auth_profiles_summary(auth_profiles);
    let candidates_json = serde_json::to_string_pretty(
        &candidates
            .iter()
            .take(30)
            .map(|c| {
                serde_json::json!({
                    "id": c.id,
                    "source": c.source,
                    "source_ids": c.source_ids,
                    "title": c.title,
                    "class": c.vuln_class,
                    "severity": c.severity_guess,
                    "status": c.status,
                    "confidence": c.confidence,
                    "hypothesis": c.hypothesis,
                    "affected_components": c.affected_components,
                    "source_excerpt": candidate_source_excerpt(c, workspaces).map(|ex| serde_json::json!({
                        "path": ex.path,
                        "line": ex.line,
                        "kind": ex.kind,
                        "body": ex.body,
                    })),
                })
            })
            .collect::<Vec<_>>(),
    )
    .unwrap_or_else(|_| "[]".to_string());
    let system = if research_mode_enabled {
        r#"You are Nyx Agent's senior pentest planner operating in Vuln Research Mode. Produce a ranked, safe attack plan for authorized testing of the operator's local application.

Research mode changes planning depth, not execution safety. Favor product-invariant hypotheses: lifecycle bugs, stale access, replay, downgrade/entitlement mismatch, invite/team/org transitions, webhook/event consistency, AI-agent indirect actions, and background job side effects. Use ResearchMode candidates and prior scanner leads as hypotheses to organize deeper exploration, not as proof.

Return exactly one JSON object and no Markdown. Do not claim a vulnerability is verified. For each hypothesis, specify the deterministic evidence the verifier must collect and the rejecting evidence that would disprove it. Prefer harmless probes. Mark destructive or aggressive probes clearly so Nyx Agent can block them unless explicitly enabled."#
            .to_string()
    } else {
        r#"You are Nyx Agent's senior pentest planner. Produce a ranked, safe attack plan for authorized testing of the operator's local application.

Return exactly one JSON object and no Markdown. Do not claim a vulnerability is verified. For each hypothesis, specify the deterministic evidence the verifier must collect and the rejecting evidence that would disprove it. Prefer harmless probes. Mark destructive or aggressive probes clearly so Nyx Agent can block them unless explicitly enabled."#
            .to_string()
    };
    let research_mode = if research_mode_enabled {
        "enabled - prioritize product invariants and ResearchMode provenance"
    } else {
        "disabled"
    };
    let user = format!(
        "TARGET BASE URLS\n{targets}\n\nAUTH PROFILES\n{auth}\n\nRESEARCH MODE\n{research_mode}\n\nROUTE MODEL\n{routes}\n\nCANDIDATES\n{candidates_json}\n\nRequired JSON shape:\n{{\"threat_model_summary\":\"...\",\"top_hypotheses\":[{{\"candidate_id\":\"...\",\"rank\":1,\"mapped_routes\":[\"GET /api/...\"],\"needed_auth_roles\":[\"anonymous\"],\"safest_probe\":\"...\",\"fallback_probe\":\"...\",\"destructiveness\":\"safe|state-changing|aggressive\",\"expected_confirming_evidence\":\"...\",\"expected_rejecting_evidence\":\"...\"}}]}}\n"
    );
    Prompt {
        prompt_version: if research_mode_enabled {
            ATTACK_PLANNING_RESEARCH_PROMPT_VERSION
        } else {
            ATTACK_PLANNING_PROMPT_VERSION
        }
        .to_string(),
        task_id: format!("attack-plan-{}", short_candidate_id(&candidates[0].run_id)),
        model: None,
        system,
        user,
        max_output_tokens: if research_mode_enabled { 5000 } else { 3000 },
        temperature: 0.0,
        seed: None,
    }
}

fn prioritize_candidates_for_planning(
    candidates: &mut [PentestCandidateRecord],
    research_mode_enabled: bool,
) {
    candidates.sort_by(|a, b| {
        research_candidate_rank(b, research_mode_enabled)
            .cmp(&research_candidate_rank(a, research_mode_enabled))
            .then_with(|| severity_rank(&b.severity_guess).cmp(&severity_rank(&a.severity_guess)))
            .then_with(|| {
                b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.id.cmp(&b.id))
    });
}

fn compact_attack_plan_context(raw: &str) -> String {
    let compact =
        raw.lines().map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ");
    compact.chars().take(4_000).collect()
}

/// Convert static `PentestCandidate` rows into executable HTTP plans
/// before the live verifier runs. This is the missing bridge between
/// Nyx's source signal and the local dev site: the verifier deliberately
/// refuses to guess at routes unless an AI pass has turned source context
/// into method/url/body/oracle JSON.
pub async fn run_live_test_plan_synthesis_pass(
    config: &AiConfig,
    store: &Store,
    secrets: &SecretStore,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    target_urls: &[String],
    route_model: Option<&RouteModel>,
    auth_profiles: &[ProjectAuthProfile],
    browser_checks_enabled: bool,
    allow_state_changing: bool,
    capabilities: Option<&EnvCapabilityReport>,
    attack_plan_context: Option<&str>,
    events: EventSink,
) -> anyhow::Result<LiveTestPlanSynthesisPassReport> {
    let mut report = LiveTestPlanSynthesisPassReport::default();
    if target_urls.is_empty() {
        return Ok(report);
    }

    let candidates = store.pentest_candidates().list_by_run(&bundle.run_id).await?;
    let candidates: Vec<_> = candidates
        .into_iter()
        .filter(|c| matches!(c.status.as_str(), "Proposed" | "NeedsLiveTest"))
        .filter(|c| !candidate_has_executable_live_plan(c, target_urls, browser_checks_enabled))
        .collect();
    if candidates.is_empty() {
        return Ok(report);
    }
    report.candidates_seen = candidates.len() as u32;

    if let Some(capability_report) = capabilities {
        let _ = events.send(AgentEvent::Run {
            data: RunEvent::LiveVerificationCapabilities {
                run_id: bundle.run_id.clone(),
                project_id: bundle.project_id.clone(),
                report: serde_json::to_value(capability_report).unwrap_or_default(),
                ts_ms: now_epoch_ms(),
            },
        });
    }
    let synthesizer =
        live_planning::LiveTestPlanSynthesizer::new(live_planning::LiveTestPlanSynthesisContext {
            route_model,
            target_urls,
            auth_profiles,
            browser_checks_enabled,
            allow_state_changing,
            capabilities,
        });
    let mut ai_candidates = Vec::new();
    for candidate in candidates {
        let plan = synthesizer.synthesize(&candidate);
        let finished_at = now_epoch_ms();
        match plan {
            LiveTestPlan::NoPlan(no_plan) => {
                let fallback = LiveTestPlan::NoPlan(no_plan);
                if fallback.no_plan_reason().is_some_and(should_try_ai_live_plan_after_no_plan) {
                    ai_candidates
                        .push(AiLivePlanCandidate { candidate, fallback_no_plan: Some(fallback) });
                } else {
                    persist_no_plan_candidate(store, &candidate, fallback, finished_at).await?;
                    report.no_plan += 1;
                }
            }
            executable => {
                let plan_blob = serde_json::to_string(&executable)?;
                match normalise_live_test_plan(&plan_blob, target_urls) {
                    Ok(Some(_)) => {
                        store
                            .pentest_candidates()
                            .set_test_plan(&candidate.id, &plan_blob, "Proposed", None, finished_at)
                            .await?;
                        report.planned += 1;
                    }
                    Ok(None) => {
                        ai_candidates
                            .push(AiLivePlanCandidate { candidate, fallback_no_plan: None });
                    }
                    Err(err) => {
                        tracing::warn!(
                            candidate_id = %candidate.id,
                            error = %err,
                            "deterministic live test plan synthesis produced unusable plan"
                        );
                        ai_candidates
                            .push(AiLivePlanCandidate { candidate, fallback_no_plan: None });
                    }
                }
            }
        }
    }
    if ai_candidates.is_empty() {
        return Ok(report);
    }

    let adapter = match selected_one_shot_runtime(
        config,
        store,
        secrets,
        config.default_run_budget_usd_micros_resolved(),
        "live test plan synthesis",
        None,
    )
    .await?
    {
        Some(adapter) => adapter,
        None => {
            for queued in ai_candidates {
                if let Some(no_plan) = queued.fallback_no_plan {
                    persist_no_plan_candidate(store, &queued.candidate, no_plan, now_epoch_ms())
                        .await?;
                    report.no_plan += 1;
                }
            }
            return Ok(report);
        }
    };

    let runtime_name = adapter.name();
    let runtime_model = adapter.default_model().to_string();
    for queued in ai_candidates {
        let candidate = queued.candidate;
        let started_at = now_epoch_ms();
        let prompt = build_live_test_plan_prompt(
            &candidate,
            workspaces,
            target_urls,
            route_model,
            auth_profiles,
            browser_checks_enabled,
            allow_state_changing,
            attack_plan_context,
        );
        let budget = Budget {
            run_id: bundle.run_id.clone(),
            kind: BudgetKind::OneShot,
            cap_usd_micros: config.default_run_budget_usd_micros_resolved(),
        };
        let resp = match adapter.one_shot(prompt, budget, events.clone()).await {
            Ok(resp) => resp,
            Err(err) => {
                tracing::warn!(
                    candidate_id = %candidate.id,
                    error = %err,
                    "live test plan synthesis call failed"
                );
                report.failed += 1;
                if ai_error_should_halt_pass(&err) {
                    break;
                }
                continue;
            }
        };
        let finished_at = now_epoch_ms();
        report.attempts += 1;
        report.spend_usd_micros += resp.cost_usd_micros;
        let metrics = AgentTraceMetrics::from_response(&resp);
        let trace = build_trace_row(
            TaskKind::LiveTestPlan,
            None,
            runtime_name,
            &runtime_model,
            &resp.prompt_version,
            resp.cost_usd_micros,
            started_at,
            finished_at,
            Some(&metrics),
        );
        let trace_id = trace.id.clone();
        persist_trace_row(store, trace).await;

        match normalise_live_test_plan(&resp.content, target_urls) {
            Ok(Some(plan)) => {
                let plan_blob = serde_json::to_string(&plan)?;
                store
                    .pentest_candidates()
                    .set_test_plan(
                        &candidate.id,
                        &plan_blob,
                        "Proposed",
                        Some(&trace_id),
                        finished_at,
                    )
                    .await?;
                report.planned += 1;
            }
            Ok(None) => {
                let reason = queued
                    .fallback_no_plan
                    .as_ref()
                    .and_then(LiveTestPlan::no_plan_reason)
                    .map(|reason| reason.message.clone())
                    .unwrap_or_else(|| {
                        "AI live test planning returned no executable plan".to_string()
                    });
                if let Some(fallback) = queued.fallback_no_plan {
                    persist_no_plan_candidate(store, &candidate, fallback, finished_at).await?;
                } else {
                    store
                        .pentest_candidates()
                        .set_status(&candidate.id, "NeedsReview", Some(&reason), finished_at)
                        .await?;
                    persist_no_plan_memory(store, &candidate, &reason, finished_at).await;
                }
                report.no_plan += 1;
            }
            Err(err) => {
                tracing::warn!(
                    candidate_id = %candidate.id,
                    error = %err,
                    response = %resp.content,
                    "live test plan synthesis returned an unusable plan"
                );
                report.failed += 1;
            }
        }
    }
    Ok(report)
}

async fn persist_no_plan_candidate(
    store: &Store,
    candidate: &PentestCandidateRecord,
    plan: LiveTestPlan,
    finished_at: i64,
) -> anyhow::Result<()> {
    let reason = plan
        .no_plan_reason()
        .map(|reason| reason.message.clone())
        .unwrap_or_else(|| "candidate has no executable live test plan".to_string());
    let plan_blob = serde_json::to_string(&plan)?;
    store
        .pentest_candidates()
        .set_test_plan(&candidate.id, &plan_blob, "NeedsReview", None, finished_at)
        .await?;
    store
        .pentest_candidates()
        .set_status(&candidate.id, "NeedsReview", Some(&reason), finished_at)
        .await?;
    persist_no_plan_memory(store, candidate, &reason, finished_at).await;
    Ok(())
}

fn should_try_ai_live_plan_after_no_plan(reason: &NoPlanReason) -> bool {
    matches!(
        reason.code,
        NoPlanReasonCode::BadEndpoint
            | NoPlanReasonCode::WeakOracle
            | NoPlanReasonCode::NoExecutablePlan
            | NoPlanReasonCode::UnsupportedClass
            | NoPlanReasonCode::RouteNotInferred
            | NoPlanReasonCode::Other
    )
}

fn build_live_test_plan_prompt(
    candidate: &PentestCandidateRecord,
    workspaces: &HashMap<String, WorkspaceHandle>,
    target_urls: &[String],
    route_model: Option<&RouteModel>,
    auth_profiles: &[ProjectAuthProfile],
    browser_checks_enabled: bool,
    allow_state_changing: bool,
    attack_plan_context: Option<&str>,
) -> Prompt {
    let excerpt = candidate_source_excerpt(candidate, workspaces)
        .map(|ex| {
            format!(
                "path: {path}\nline: {line}\nkind: {kind}\n```\n{body}```",
                path = ex.path,
                line = ex.line.map(|l| l.to_string()).unwrap_or_else(|| "unknown".to_string()),
                kind = ex.kind,
                body = ex.body,
            )
        })
        .unwrap_or_else(|| {
            "source excerpt unavailable; infer only from candidate metadata".to_string()
        });
    let components = serde_json::to_string_pretty(&candidate.affected_components)
        .unwrap_or_else(|_| "[]".to_string());
    let source_ids = serde_json::to_string_pretty(&candidate.source_ids).unwrap_or_default();
    let targets = target_urls.iter().map(|u| format!("- {u}")).collect::<Vec<_>>().join("\n");
    let routes = route_model
        .map(|m| route_model::compact_route_model_for_prompt(m, 40))
        .unwrap_or_else(|| "route model unavailable".to_string());
    let auth = pentest_tools::auth_profiles_summary(auth_profiles);
    let browser = if browser_checks_enabled {
        "enabled"
    } else {
        "disabled; return no_plan_reason for browser-only/client-side-only candidates"
    };
    let state_changing = if allow_state_changing {
        "state-changing probes are explicitly allowed by run policy"
    } else {
        "state-changing probes are not allowed; return no_plan_reason for mutation-only verification"
    };
    let attack_plan =
        attack_plan_context.unwrap_or("no prior attack-planning trace available for this run");
    let system = r#"You are Nyx Agent's live-test-plan synthesizer. Work like a senior application security tester converting a static signal into one safe, executable HTTP verification plan for a local development app.

Return exactly one JSON object and no Markdown. Prefer harmless probes that demonstrate reachability, authorization bypass, reflection, unsafe redirect, exposed data, or other vulnerability-specific evidence without destroying data. Use only the supplied target base URLs.

Nyx Agent can execute these deterministic tools in the verifier: http.request and auth.login_as(role) through configured header/cookie/token injection. Browser plans are allowed only when the operator enables browser verification and Playwright is installed, so prefer HTTP unless the source is clearly client-side.

The oracle you return must be a confirming oracle: if all predicates pass, the candidate should be vulnerable. Do not encode rejecting/safety evidence as success. A 401/403/404 response, escaped output, no reflection, or absence of sensitive data is rejecting evidence; return {"no_plan_reason":"..."} when that is the only safe probe.

Do not fetch static source assets such as .js/.css/.map files to prove a sink string exists. A served bundle or source snippet is not live exploit evidence. For client-side-only DOM issues, return a browser plan only when a real browser workflow can exercise attacker-controlled input; otherwise return {"no_plan_reason":"..."}.

If you cannot derive a meaningful exploit-confirming live test from the source context, return {"no_plan_reason":"..."}.

Executable plan schema:
{
  "kind": "http",
  "method": "GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS",
  "url": "absolute URL under one supplied target base URL",
  "path": "optional path, useful for audit only",
  "headers": {"Header": "value"},
  "body": "optional raw request body",
  "json": {"optional": "JSON request body"},
  "expect_status": 200,
  "status_range": "2xx|3xx|4xx|5xx",
  "body_contains": "positive exploit marker or sensitive marker that must appear",
  "body_not_contains": "optional guard string or array; never the primary confirming evidence",
  "header_contains": {"Header": "substring"},
  "rationale": "brief reason this would confirm the candidate"
}

For authorization boundaries, prefer first-class authz plans when matching auth roles exist. Use role comparison for vertical checks (for example user vs admin on the same action):
{
  "kind": "authz_role_comparison",
  "hypothesis": "...",
  "allowed_role": "admin",
  "challenged_role": "user",
  "request": {"method": "GET", "path": "/api/admin/report"},
  "oracle": {
    "type": "role_comparison_break",
    "forbidden_status": [401,403,404],
    "positive_markers": ["admin report", "accountId"]
  }
}

Use object ownership for horizontal checks (for example user_b reading user_a's seeded object). Include a configured object id or seed and capture one before comparing owner vs accessor:
{
  "kind": "authz_object_ownership",
  "hypothesis": "...",
  "object": {"name": "project", "owner_role": "user_a", "id": "123", "positive_markers": ["nyx-agent-owned-project"]},
  "accessor_role": "user_b",
  "owner_request": {"method": "GET", "path": "/api/projects/123"},
  "accessor_request": {"method": "GET", "path": "/api/projects/123"},
  "oracle": {
    "type": "object_ownership_break",
    "forbidden_status": [401,403,404],
    "positive_markers": ["nyx-agent-owned-project", "123"]
  }
}

For client-side authorization checks where the role boundary only appears in the rendered app, compare the same browser workflow under two roles:
{
  "kind": "authz_browser_role_comparison",
  "hypothesis": "...",
  "allowed_role": "admin",
  "challenged_role": "user",
  "workflow": {
    "url": "/app/admin",
    "steps": [{"action": "wait_for_selector", "selector": "main"}],
    "oracle": {"text_contains": "Admin Console"}
  }
}

If you cannot provide positive role/object markers, use a legacy differential plan only for non-authz comparisons and expect it to be rejected as weak without sensitive_body_markers:
{
  "kind": "differential_http",
  "hypothesis": "...",
  "steps": [
    {"as": "user_a", "method": "GET", "path": "/api/accounts/123"},
    {"as": "user_b", "method": "GET", "path": "/api/accounts/123"}
  ],
  "oracle": {
    "type": "forbidden_equivalence_break",
    "expected_allowed_step": 0,
    "expected_forbidden_step": 1,
    "forbidden_status": [401,403,404],
    "sensitive_body_markers": ["email", "accountId"]
  }
}

For multi-step stateful bugs, use an HTTP workflow. Captures can extract response values and later steps can reference them as {{name}}:
{
  "kind": "http_workflow",
  "steps": [
    {"as": "user_a", "method": "POST", "path": "/api/projects", "json": {"name": "nyx-agent-probe"}, "captures": {"project_id": {"from": "json", "path": "id"}}},
    {"as": "user_b", "method": "GET", "path": "/api/projects/{{project_id}}"}
  ],
  "oracle": {"step": 1, "status_range": "2xx", "body_contains": "nyx-agent-probe"}
}

For client-side-only bugs, use a browser plan only when browser verification is enabled:
{
  "kind": "browser",
  "url": "/app/search?q=%3Cimg%20src%3Dx%20onerror%3Dalert(1)%3E",
  "steps": [{"action": "wait_for_selector", "selector": "body"}],
  "oracle": {"alert_contains": "nyx-agent-probe"}
}

At least one positive live evidence oracle is required: body_contains/header_contains for HTTP, positive_markers for authz role/object probes, or text_contains/html_contains/selector_exists/selector_text_contains/url_contains/title_contains/console_contains/alert_contains for browser and authz browser probes. expect_status/status_range/body_not_contains may be included only as guards around that positive evidence. Do not return generic homepage checks, blocked-request checks, no-reflection checks, or static bundle/source checks."#
        .to_string();
    let user = format!(
        "TARGET BASE URLS\n{targets}\n\nAUTH PROFILES\n{auth}\n\nBROWSER VERIFICATION\n{browser}\n\nSTATE-CHANGING POLICY\n{state_changing}\n\nROUTE MODEL\n{routes}\n\nSENIOR ATTACK PLAN CONTEXT\n{attack_plan}\n\nCANDIDATE\nid: {id}\ntitle: {title}\nclass: {vuln_class}\nseverity: {severity}\nstatus: {status}\nhypothesis: {hypothesis}\nsource_ids: {source_ids}\naffected_components:\n{components}\n\nSOURCE EXCERPT\n{excerpt}\n",
        id = candidate.id,
        title = candidate.title,
        vuln_class = candidate.vuln_class,
        severity = candidate.severity_guess,
        status = candidate.status,
        hypothesis = candidate.hypothesis,
    );
    Prompt {
        prompt_version: LIVE_TEST_PLAN_PROMPT_VERSION.to_string(),
        task_id: format!("live-plan-{}", short_candidate_id(&candidate.id)),
        model: None,
        system,
        user,
        max_output_tokens: 1600,
        temperature: 0.0,
        seed: None,
    }
}

fn candidate_source_excerpt(
    candidate: &PentestCandidateRecord,
    workspaces: &HashMap<String, WorkspaceHandle>,
) -> Option<nyx_agent_types::spec::FileExcerpt> {
    for component in &candidate.affected_components {
        let Some(obj) = component.as_object() else {
            continue;
        };
        let Some(path) = obj.get("path").and_then(|v| v.as_str()) else {
            continue;
        };
        let repo = obj.get("repo").and_then(|v| v.as_str());
        let Some(workspace) = repo.and_then(|r| workspaces.get(r)).or_else(|| {
            if workspaces.len() == 1 {
                workspaces.values().next()
            } else {
                None
            }
        }) else {
            continue;
        };
        let line = obj.get("line").and_then(|v| v.as_i64()).and_then(|v| {
            if v > 0 {
                Some(v as u32)
            } else {
                None
            }
        });
        if let Some(excerpt) = read_spec_excerpt(
            workspace.workspace(),
            path,
            line,
            "candidate",
            LIVE_TEST_PLAN_EXCERPT_RADIUS,
        ) {
            return Some(excerpt);
        }
    }
    None
}

async fn persist_no_plan_memory(
    store: &Store,
    candidate: &PentestCandidateRecord,
    reason: &str,
    now_ms: i64,
) {
    let input = ExplorationMemoryInput {
        project_id: candidate.project_id.clone(),
        repo: candidate_repo_hint(candidate),
        run_id: candidate.run_id.clone(),
        source: "live_plan_no_plan".to_string(),
        hypothesis: candidate.hypothesis.clone(),
        endpoint: candidate_endpoint_hint(candidate),
        role_context: candidate_role_hint(candidate),
        object_context: candidate_object_hint(candidate),
        result: "blocked".to_string(),
        reason: reason.to_string(),
        useful_markers: candidate.source_ids.clone(),
        auth_session_notes: None,
        follow_up_ideas: vec![
            "derive missing auth/session/object context before retrying".to_string()
        ],
        candidate_id: Some(candidate.id.clone()),
        verification_attempt_id: None,
        trace_id: candidate.trace_id.clone(),
        created_at: now_ms,
    };
    if let Err(err) = store.exploration_memory().upsert(&input).await {
        tracing::warn!(
            candidate_id = %candidate.id,
            error = %err,
            "failed to persist no-plan exploration memory"
        );
    }
}

fn candidate_repo_hint(candidate: &PentestCandidateRecord) -> String {
    candidate
        .affected_components
        .iter()
        .find_map(|c| c.get("repo").and_then(|v| v.as_str()))
        .unwrap_or("unknown")
        .to_string()
}

fn candidate_endpoint_hint(candidate: &PentestCandidateRecord) -> Option<String> {
    candidate.affected_components.iter().find_map(|c| {
        c.get("endpoint")
            .or_else(|| c.get("route"))
            .or_else(|| c.get("url"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    })
}

fn candidate_role_hint(candidate: &PentestCandidateRecord) -> Option<String> {
    candidate.affected_components.iter().find_map(|c| {
        c.get("role")
            .or_else(|| c.get("auth_role"))
            .or_else(|| c.get("needed_role"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    })
}

fn candidate_object_hint(candidate: &PentestCandidateRecord) -> Option<String> {
    candidate.affected_components.iter().find_map(|c| {
        c.get("object")
            .or_else(|| c.get("resource"))
            .or_else(|| c.get("model"))
            .or_else(|| c.get("path"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    })
}

fn candidate_has_executable_live_plan(
    candidate: &PentestCandidateRecord,
    target_urls: &[String],
    browser_checks_enabled: bool,
) -> bool {
    let Some(plan) = normalise_live_test_plan(&candidate.test_plan, target_urls).ok().flatten()
    else {
        return false;
    };
    let kind = plan.get("kind").and_then(|v| v.as_str()).unwrap_or("http");
    browser_checks_enabled || !matches!(kind, "browser" | "browser_workflow")
}

fn normalise_live_test_plan(
    raw: &str,
    target_urls: &[String],
) -> Result<Option<serde_json::Value>, String> {
    pentest_tools::normalise_live_test_plan(raw, target_urls)
}

fn short_candidate_id(id: &str) -> String {
    id.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').take(48).collect()
}

#[allow(clippy::too_many_arguments)]
pub async fn run_live_evidence_review_pass(
    config: &AiConfig,
    store: &Store,
    secrets: &SecretStore,
    run_id: &str,
    candidate: &PentestCandidateRecord,
    proposed_plan: serde_json::Value,
    live_evidence: serde_json::Value,
    oracle_result: serde_json::Value,
    deterministic_review: LiveEvidenceReviewOutput,
    events: EventSink,
) -> anyhow::Result<Option<LiveEvidenceReviewOutput>> {
    let adapter = match selected_one_shot_runtime(
        config,
        store,
        secrets,
        config.default_run_budget_usd_micros_resolved(),
        "live evidence review",
        None,
    )
    .await?
    {
        Some(adapter) => adapter,
        None => return Ok(None),
    };
    let runtime_name = adapter.name();
    let runtime_model = adapter.default_model().to_string();
    let started_at = now_epoch_ms();
    let input = LiveEvidenceReviewInput {
        run_id: run_id.to_string(),
        candidate: candidate.clone(),
        proposed_plan,
        live_evidence,
        oracle_result,
        deterministic_review,
    };
    let outcome = run_live_evidence_review(
        adapter.as_ref(),
        &input,
        events,
        config.default_run_budget_usd_micros_resolved(),
    )
    .await?;
    let finished_at = now_epoch_ms();
    let review = outcome.output.clone();
    let mut trace = build_trace_row(
        TaskKind::LiveEvidenceReview,
        None,
        runtime_name,
        &runtime_model,
        &outcome.prompt_version,
        outcome.spent_usd_micros,
        started_at,
        finished_at,
        Some(&outcome.metrics),
    );
    trace.verifier_blob = Some(
        serde_json::json!({
            "kind": "LiveEvidenceReview",
            "run_id": run_id,
            "candidate_id": &candidate.id,
            "decision": review.decision.as_str(),
            "review": &review,
        })
        .to_string(),
    );
    persist_trace_row(store, trace).await;
    Ok(Some(review))
}

/// Counts surfaced by [`run_spec_derivation_pass`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpecDerivationPassReport {
    pub synthesised: u32,
    pub quarantined: u32,
    pub failed: u32,
    pub total_attempts: u64,
    pub spend_usd_micros: i64,
}

/// Fan-out SpecDerivation across every `Inconclusive(SpecDerivationFailed)`
/// finding in `bundle`. No-op (returns a default report) when the
/// selected runtime does not support one-shot tasks or is unavailable.
pub async fn run_spec_derivation_pass(
    config: &AiConfig,
    store: &Store,
    secrets: &SecretStore,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    events: EventSink,
) -> anyhow::Result<SpecDerivationPassReport> {
    let inputs = build_spec_inputs(bundle, workspaces);
    if inputs.is_empty() {
        return Ok(SpecDerivationPassReport::default());
    }
    let adapter = match selected_one_shot_runtime(
        config,
        store,
        secrets,
        config.default_run_budget_usd_micros_resolved(),
        "spec derivation",
        None,
    )
    .await?
    {
        Some(adapter) => adapter,
        None => return Ok(SpecDerivationPassReport::default()),
    };
    tracing::info!(count = inputs.len(), "spec derivation: fanning out");

    let semaphore = Arc::new(Semaphore::new(config.max_concurrent_one_shot_resolved()));
    let per_call_cap = config.spec_derivation_per_call_cap_usd_micros_resolved();
    let runtime_name = adapter.name();
    let runtime_model = adapter.default_model().to_string();
    let mut handles = Vec::with_capacity(inputs.len());
    for input in inputs {
        let rt = Arc::clone(&adapter);
        let sem = Arc::clone(&semaphore);
        let sink = events.clone();
        handles.push(tokio::spawn(async move {
            let permit = sem.acquire_owned().await.expect("semaphore closed");
            let started_at = now_epoch_ms();
            let outcome = run_spec_derivation(rt.as_ref(), &input, sink, per_call_cap).await;
            drop(permit);
            outcome.map(|o| (started_at, o))
        }));
    }

    let mut report = SpecDerivationPassReport::default();
    for handle in handles {
        match handle.await {
            Ok(Ok((started_at, outcome))) => {
                apply_spec_outcome(
                    store,
                    outcome,
                    &mut report,
                    runtime_name,
                    &runtime_model,
                    started_at,
                )
                .await?
            }
            Ok(Err(err)) => {
                tracing::warn!(error = %err, "spec derivation call failed");
                report.failed += 1;
            }
            Err(join) => {
                tracing::warn!(error = %join, "spec derivation task join error");
                report.failed += 1;
            }
        }
    }
    Ok(report)
}

/// Walk `bundle` + `workspaces` and turn each `Inconclusive(SpecDerivationFailed)`
/// diag into a `SpecDerivationInput` pre-populated with up to three
/// file excerpts (sink, call-site, framework). Public so the inner
/// filter + pre-fetch can be unit-tested without spinning up an
/// adapter.
pub fn build_spec_inputs(
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
) -> Vec<SpecDerivationInput> {
    let mut out = Vec::new();
    for repo_bundle in &bundle.per_repo {
        let RepoOutcome::Success(diags) = &repo_bundle.outcome else {
            continue;
        };
        let Some(workspace) = workspaces.get(&repo_bundle.repo) else {
            continue;
        };
        for diag in diags {
            if !diag.is_spec_derivation_failed() {
                continue;
            }
            let line = i64::from(diag.line);
            let finding_id = nyx_agent_core::store::finding_id_hash(
                &repo_bundle.repo,
                &diag.path,
                Some(line),
                &diag.cap,
                &diag.rule,
            );
            let lang = infer_lang_for_file(workspace.workspace(), &diag.path);
            let sink_ctx = diag.sink_ctx(workspace.workspace());
            let excerpts = collect_spec_excerpts(workspace, diag, SPEC_DERIVATION_MAX_EXCERPTS);
            out.push(SpecDerivationInput {
                finding_id,
                run_id: bundle.run_id.clone(),
                cap: diag.cap.clone(),
                lang,
                callee: sink_ctx.callee,
                excerpts,
            });
        }
    }
    out
}

/// Pre-fetch up to `max` excerpts for SpecDerivation: the sink line
/// first, then each distinct flow-step file (labelled `call_site` for
/// the first hop and `framework` for subsequent ones). Excerpts that
/// cannot be read are silently skipped; the agent tolerates an empty
/// list and produces a `Quarantined` outcome if it cannot infer the
/// harness shape.
fn collect_spec_excerpts(
    workspace: &WorkspaceHandle,
    diag: &Diag,
    max: usize,
) -> Vec<nyx_agent_types::spec::FileExcerpt> {
    let mut out = Vec::new();
    if let Some(ex) = read_spec_excerpt(
        workspace.workspace(),
        &diag.path,
        Some(diag.line),
        "sink",
        SPEC_DERIVATION_EXCERPT_RADIUS,
    ) {
        out.push(ex);
    }
    let mut first_upstream = true;
    for path in diag.flow_step_files_ranked() {
        if out.len() >= max {
            break;
        }
        let kind = if first_upstream {
            first_upstream = false;
            "call_site"
        } else {
            "framework"
        };
        if let Some(ex) = read_spec_excerpt(
            workspace.workspace(),
            path,
            None,
            kind,
            SPEC_DERIVATION_EXCERPT_RADIUS,
        ) {
            out.push(ex);
        }
    }
    out
}

async fn apply_spec_outcome(
    store: &Store,
    outcome: SpecDerivationOutcome,
    report: &mut SpecDerivationPassReport,
    runtime_name: &str,
    runtime_model: &str,
    started_at_ms: i64,
) -> anyhow::Result<()> {
    let finished_at = now_epoch_ms();
    match outcome {
        SpecDerivationOutcome::Synthesised {
            finding_id,
            cap,
            lang,
            spec: _,
            spec_blob,
            prompt_version,
            spent_usd_micros,
            attempts,
            metrics,
        } => {
            let provenance = AttackProvenance::LlmSynthesised.as_str().to_string();
            let rec = HarnessSpecRecord {
                id: format!("spec-{finding_id}-{finished_at:x}-{}", short_token()),
                cap,
                lang,
                spec_blob,
                attack_provenance: Some(provenance.clone()),
                prompt_version: Some(prompt_version.clone()),
                created_at: finished_at,
            };
            // Atomic dual-write so a partial failure of the finding
            // back-link does not orphan the harness_specs row.
            store
                .harness_specs()
                .insert_with_finding_spec_link(&rec, &finding_id, &provenance, &prompt_version)
                .await?;
            let trace = build_trace_row(
                TaskKind::SpecDerivation,
                Some(finding_id),
                runtime_name,
                runtime_model,
                &prompt_version,
                spent_usd_micros,
                started_at_ms,
                finished_at,
                Some(&metrics),
            );
            persist_trace_row(store, trace).await;
            report.synthesised += 1;
            report.spend_usd_micros += spent_usd_micros;
            report.total_attempts += u64::from(attempts);
        }
        SpecDerivationOutcome::Quarantined {
            finding_id,
            reason,
            spent_usd_micros,
            attempts,
            metrics,
        } => {
            let blob = serde_json::json!({
                "kind": "SpecDerivationQuarantined",
                "task": "SpecDerivation",
                "reason": reason,
            })
            .to_string();
            store.findings().quarantine(&finding_id, &blob).await?;
            let trace = build_trace_row(
                TaskKind::SpecDerivation,
                Some(finding_id),
                runtime_name,
                runtime_model,
                SPEC_DERIVATION_PROMPT_VERSION,
                spent_usd_micros,
                started_at_ms,
                finished_at,
                Some(&metrics),
            );
            persist_trace_row(store, trace).await;
            report.quarantined += 1;
            report.spend_usd_micros += spent_usd_micros;
            report.total_attempts += u64::from(attempts);
        }
    }
    Ok(())
}

// Per-call ChainReasoning cap now lives on `AiConfig` as
// `[ai] chain_reasoning_per_call_cap_usd_micros`. Same shape as the
// PayloadSynthesis / SpecDerivation knobs.

/// Heuristic path fragments that mark a file as a vendored framework
/// binding. The ChainReasoning prompt tags nodes whose source path
/// matches any of these as `framework` so the model can recognise
/// glue code that is not under the operator's control.
const FRAMEWORK_PATH_FRAGMENTS: &[&str] = &[
    "site-packages/",
    "node_modules/",
    "vendor/",
    "/lib/",
    "/framework/",
    "/frameworks/",
    ".cargo/registry/",
    "_vendor/",
];

/// Counts surfaced by [`run_chain_reasoning_pass`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ChainReasoningPassReport {
    pub chains_persisted: u32,
    pub chains_verified: u32,
    pub chains_needing_verification: u32,
    pub vulnerabilities_recorded: u32,
    pub cross_repo_chains: u32,
    pub members_stamped: u32,
    pub spend_usd_micros: i64,
    pub attempts: u64,
    pub failed: u32,
}

/// Fan-out (single-call) ChainReasoning over the run's finding graph.
/// No-op (returns a default report) when the selected runtime does not
/// support one-shot tasks, is unavailable, or the bundle has fewer than
/// two findings.
pub async fn run_chain_reasoning_pass(
    config: &AiConfig,
    store: &Store,
    secrets: &SecretStore,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    events: EventSink,
) -> anyhow::Result<ChainReasoningPassReport> {
    let input = match store
        .attack_graph()
        .chain_planning_input(&bundle.run_id, CHAIN_REASONING_DEFAULT_MAX)
        .await?
        .or_else(|| build_chain_input(bundle))
    {
        Some(i) => i,
        None => return Ok(ChainReasoningPassReport::default()),
    };

    // Reduce the run graph to a budget-bounded subgraph plus seed
    // chains. The full graph can be tens of thousands of edges, which
    // overflows the model context window (the call is refused before
    // any reasoning) and is impossible for the model to traverse
    // reliably. The deterministic path engine does the traversal and
    // hands the model the strongest entry->impact paths to verify.
    let reduced = reduce_chain_graph(
        &input,
        GraphBudget::for_context_window(config.context_window.map(|w| w as usize)),
    );
    tracing::info!(
        orig_nodes = input.nodes.len(),
        orig_edges = input.edges.len(),
        nodes = reduced.input.nodes.len(),
        edges = reduced.input.edges.len(),
        seeds = reduced.seeds.len(),
        dropped_nodes = reduced.dropped_nodes,
        dropped_edges = reduced.dropped_edges,
        fell_back = reduced.fell_back,
        repos = reduced.input.repos.len(),
        "chain reasoning: dispatching"
    );
    let input = reduced.input;
    let seeds = reduced.seeds;

    let started_at = now_epoch_ms();
    let (outcome, runtime_name, runtime_model) = if let Some(agent_adapter) =
        selected_agent_loop_runtime(
            config,
            store,
            config.default_run_budget_usd_micros_resolved(),
            None,
            // Shares the run-wide AgentLoop budget bucket (no dedicated
            // cap knob) alongside the attack-agent pass.
            None,
        )
        .await
    {
        let workspace_roots = chain_reasoning_workspaces(workspaces);
        let runtime_name = agent_adapter.name();
        let runtime_model = agent_adapter.default_model().to_string();
        let outcome = match run_agentic_chain_reasoning(
            agent_adapter.as_ref(),
            &input,
            &seeds,
            &workspace_roots,
            events,
            config.chain_reasoning_per_call_cap_usd_micros_resolved(),
        )
        .await
        {
            Ok(o) => o,
            Err(err) => {
                tracing::warn!(error = %err, "agentic chain reasoning call failed");
                return Ok(ChainReasoningPassReport { failed: 1, ..Default::default() });
            }
        };
        (outcome, runtime_name, runtime_model)
    } else {
        let adapter = match selected_one_shot_runtime(
            config,
            store,
            secrets,
            config.default_run_budget_usd_micros_resolved(),
            "chain reasoning",
            None,
        )
        .await?
        {
            Some(adapter) => adapter,
            None => return Ok(ChainReasoningPassReport::default()),
        };
        let runtime_name = adapter.name();
        let runtime_model = adapter.default_model().to_string();
        let outcome = match run_chain_reasoning(
            adapter.as_ref(),
            &input,
            &seeds,
            events,
            config.chain_reasoning_per_call_cap_usd_micros_resolved(),
        )
        .await
        {
            Ok(o) => o,
            Err(err) => {
                tracing::warn!(error = %err, "chain reasoning call failed");
                return Ok(ChainReasoningPassReport { failed: 1, ..Default::default() });
            }
        };
        (outcome, runtime_name, runtime_model)
    };

    let mut report = ChainReasoningPassReport::default();
    apply_chain_outcome(
        store,
        &input,
        outcome,
        &mut report,
        bundle.project_id.as_str(),
        runtime_name,
        &runtime_model,
        started_at,
    )
    .await?;
    Ok(report)
}

fn chain_reasoning_workspaces(
    workspaces: &HashMap<String, WorkspaceHandle>,
) -> Vec<ChainReasoningWorkspace> {
    let mut out = workspaces
        .iter()
        .map(|(repo, workspace)| ChainReasoningWorkspace {
            repo: repo.clone(),
            root: workspace.workspace().display().to_string(),
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| a.repo.cmp(&b.repo));
    out
}

/// Walk `bundle` and turn each diag into a `ChainReasoningNode`. Edges
/// are derived from `Diag::flow_steps`: each step inside a diag that
/// resolves to *another* diag's `(path, line)` produces a directed
/// `Reaches` edge, with `cross_repo = true` when the two diags live in
/// different repos. Public so the inner graph builder is unit-testable
/// without spinning up an adapter.
pub fn build_chain_input(bundle: &RunBundle<Diag>) -> Option<ChainReasoningInput> {
    // Collect every node + an index keyed by (repo, path, line) so flow
    // steps can resolve to a finding id without a quadratic scan.
    let mut nodes: Vec<ChainReasoningNode> = Vec::new();
    let mut by_location: HashMap<(String, String, u32), String> = HashMap::new();
    let mut repos: Vec<String> = Vec::new();
    for repo_bundle in &bundle.per_repo {
        if !repos.contains(&repo_bundle.repo) {
            repos.push(repo_bundle.repo.clone());
        }
        let RepoOutcome::Success(diags) = &repo_bundle.outcome else {
            continue;
        };
        for diag in diags {
            let id = nyx_agent_core::store::finding_id_hash(
                &repo_bundle.repo,
                &diag.path,
                Some(i64::from(diag.line)),
                &diag.cap,
                &diag.rule,
            );
            let kind = classify_node_kind(diag);
            by_location
                .insert((repo_bundle.repo.clone(), diag.path.clone(), diag.line), id.clone());
            nodes.push(ChainReasoningNode {
                id,
                graph_kind: None,
                label: None,
                ref_id: None,
                repo: repo_bundle.repo.clone(),
                path: diag.path.clone(),
                line: Some(diag.line),
                cap: diag.cap.clone(),
                rule: diag.rule.clone(),
                severity: diag.severity.clone(),
                kind: kind.to_string(),
                routes: Vec::new(),
                roles: Vec::new(),
                objects: Vec::new(),
                evidence_refs: Vec::new(),
            });
        }
    }
    if nodes.len() < 2 {
        return None;
    }

    // Edges: per diag, walk its flow_steps; whenever a step lands on a
    // location that resolves to another known diag, link that diag to
    // the current diag. The edge direction goes "from upstream step ->
    // sink diag" so the model sees an entry-to-sink traversal.
    let mut edges: Vec<ChainReasoningEdge> = Vec::new();
    let mut edge_keys: HashSet<(String, String)> = HashSet::new();
    for repo_bundle in &bundle.per_repo {
        let RepoOutcome::Success(diags) = &repo_bundle.outcome else {
            continue;
        };
        for diag in diags {
            let sink_id =
                match by_location.get(&(repo_bundle.repo.clone(), diag.path.clone(), diag.line)) {
                    Some(id) => id.clone(),
                    None => continue,
                };
            // Walk every step; match by (repo, path, line) first, then
            // by (any repo, path, line) so a cross-repo step finds the
            // diag whose path matches even when the step itself does
            // not name a repo.
            for step in &diag.flow_steps {
                let same_repo_key = (repo_bundle.repo.clone(), step.path.clone(), step.line);
                if let Some(from_id) = by_location.get(&same_repo_key) {
                    push_edge(&mut edges, &mut edge_keys, from_id, &sink_id, false);
                    continue;
                }
                // Cross-repo: scan for any other repo whose diag
                // matches (path, line).
                for (other_repo, _, _) in by_location
                    .keys()
                    .filter(|(r, p, l)| {
                        r != &repo_bundle.repo && p == &step.path && *l == step.line
                    })
                    .cloned()
                    .collect::<Vec<_>>()
                {
                    let key = (other_repo, step.path.clone(), step.line);
                    if let Some(from_id) = by_location.get(&key) {
                        push_edge(&mut edges, &mut edge_keys, from_id, &sink_id, true);
                    }
                }
            }
        }
    }

    Some(ChainReasoningInput {
        run_id: bundle.run_id.clone(),
        repos,
        nodes,
        edges,
        max_chains: CHAIN_REASONING_DEFAULT_MAX,
    })
}

fn push_edge(
    edges: &mut Vec<ChainReasoningEdge>,
    keys: &mut HashSet<(String, String)>,
    from: &str,
    to: &str,
    cross_repo: bool,
) {
    if from == to {
        return;
    }
    let key = (from.to_string(), to.to_string());
    if keys.insert(key) {
        edges.push(ChainReasoningEdge {
            from: from.to_string(),
            to: to.to_string(),
            label: "Reaches".to_string(),
            cross_repo,
            edge_id: None,
            evidence_ref: None,
            source: None,
        });
    }
}

/// Coarse role tag for a node. The static pass's flow_steps drive the
/// `entry` decision; framework detection is a path-fragment heuristic;
/// every remaining diag is a `sink`. The classification is advisory
/// for the prompt; the model is free to override.
fn classify_node_kind(diag: &Diag) -> &'static str {
    let lower = diag.path.to_lowercase();
    if FRAMEWORK_PATH_FRAGMENTS.iter().any(|frag| lower.contains(frag)) {
        return NODE_KIND_FRAMEWORK;
    }
    if diag.flow_steps.iter().any(|s| s.kind.as_deref() == Some("source")) {
        return NODE_KIND_ENTRY;
    }
    if diag.flow_steps.iter().any(|s| s.kind.as_deref() == Some("sink")) {
        return NODE_KIND_SINK;
    }
    // Default: diags surface where the static pass landed, so bare
    // diags without an explicit `source` step lean toward `sink`. The
    // `other` bucket exported by `nyx-agent-types::chain` is reserved
    // for clearly non-source / non-sink nodes a later phase may add.
    NODE_KIND_SINK
}

/// Persist a `ChainReasoningOutcome` against the store. Writes a
/// `chains` row (carrying the chain-level `attack_provenance` /
/// `prompt_version` on the chain itself) and stamps `findings.chain_id`
/// on every member via `set_chain`.
///
/// Provenance ladder: ChainReasoning is the lowest-priority writer for
/// `findings.attack_provenance` / `findings.prompt_version` (below
/// PayloadSynthesis and SpecDerivation) and intentionally does not
/// touch those columns on member findings. ChainReasoning is a graph-
/// level synthesis pass; it does not produce the payload the verifier
/// executes, nor the harness spec the verifier wraps the payload in,
/// so the chain's prompt version is not the canonical attribution for
/// any individual member finding. Per-chain provenance is recorded on
/// `chains.attack_provenance` / `chains.prompt_version` instead, where
/// the UI can render it without colliding with the per-finding writers.
async fn apply_chain_outcome(
    store: &Store,
    input: &ChainReasoningInput,
    outcome: ChainReasoningOutcome,
    report: &mut ChainReasoningPassReport,
    project_id: &str,
    runtime_name: &str,
    runtime_model: &str,
    started_at_ms: i64,
) -> anyhow::Result<()> {
    let finished_at = now_epoch_ms();
    match outcome {
        ChainReasoningOutcome::Ranked {
            run_id,
            output,
            prompt_version,
            spent_usd_micros,
            attempts,
            metrics,
        } => {
            report.spend_usd_micros += spent_usd_micros;
            report.attempts += u64::from(attempts);
            let provenance = AttackProvenance::LlmSynthesised.as_str().to_string();
            let repo_by_id: HashMap<String, String> =
                input.nodes.iter().map(|n| (n.id.clone(), n.repo.clone())).collect();
            let created_at = finished_at;
            let trace = build_trace_row(
                TaskKind::ChainReasoning,
                None,
                runtime_name,
                runtime_model,
                &prompt_version,
                spent_usd_micros,
                started_at_ms,
                finished_at,
                Some(&metrics),
            );
            persist_trace_row(store, trace).await;
            for (rank, chain) in output.chains.iter().enumerate() {
                let cross_repo = chain
                    .member_ids
                    .iter()
                    .filter_map(|m| repo_by_id.get(m))
                    .collect::<HashSet<_>>()
                    .len()
                    > 1;
                let member_ids_blob = match serde_json::to_string(&chain.member_ids) {
                    Ok(b) => b,
                    Err(err) => {
                        tracing::warn!(error = %err, "chain reasoning: dropping chain with unserialisable member_ids");
                        continue;
                    }
                };
                let rationale_blob = serde_json::json!({
                    "rationale": chain.rationale,
                })
                .to_string();
                let evidence_blob = build_chain_evidence_blob(input, chain);
                let severity = chain
                    .member_ids
                    .iter()
                    .filter_map(|id| input.nodes.iter().find(|n| &n.id == id))
                    .map(|n| n.severity.as_str())
                    .max_by_key(|severity| chain_severity_rank(severity))
                    .map(str::to_string);
                let chain_id = format!("chain-{run_id}-{rank:02}-{created_at:x}-{}", short_token());
                let proof = chain_terminal_live_proof(input, chain);
                let chain_status = if proof.terminal_live_proof
                    && chain.missing_verification_steps.is_empty()
                    && chain.confidence >= 60
                {
                    "Verified"
                } else {
                    "NeedsChainVerification"
                };
                let rec = ChainRecord {
                    id: chain_id.clone(),
                    run_id: run_id.clone(),
                    cross_repo,
                    member_ids: member_ids_blob,
                    rationale_blob: Some(rationale_blob),
                    attack_provenance: Some(provenance.clone()),
                    prompt_version: Some(prompt_version.clone()),
                    status: chain_status.to_string(),
                    verification_attempt_id: proof.verification_attempt_id.clone(),
                    evidence_blob: Some(evidence_blob),
                    severity: severity.clone(),
                };
                store.chains().insert(&rec).await?;
                report.chains_persisted += 1;
                if chain_status == "Verified" {
                    report.chains_verified += 1;
                    let mut vuln = verified_vulnerability_from_chain(
                        input,
                        chain,
                        &rec,
                        project_id,
                        proof.verification_attempt_id.as_deref(),
                        proof.verified_vulnerability_id.as_deref(),
                        created_at,
                    );
                    if let Some(existing_id) = proof.verified_vulnerability_id.as_deref() {
                        if let Some(existing) =
                            store.verified_vulnerabilities().get(existing_id).await?
                        {
                            vuln = merge_chain_vulnerability_with_existing(vuln, existing, &rec);
                        }
                    }
                    store.verified_vulnerabilities().upsert(&vuln).await?;
                    report.vulnerabilities_recorded += 1;
                } else {
                    report.chains_needing_verification += 1;
                }
                if cross_repo {
                    report.cross_repo_chains += 1;
                }
                for member_id in &chain.member_ids {
                    match store.findings().set_chain(member_id, &chain_id).await {
                        Ok(()) => report.members_stamped += 1,
                        Err(err) => tracing::warn!(
                            error = %err,
                            chain = %chain_id,
                            finding = %member_id,
                            "chain reasoning: failed to stamp finding back-link"
                        ),
                    }
                }
            }
        }
        ChainReasoningOutcome::NoChains {
            run_id: _,
            reason,
            spent_usd_micros,
            attempts,
            metrics,
        } => {
            tracing::info!(reason = %reason, "chain reasoning: no chains produced");
            let trace = build_trace_row(
                TaskKind::ChainReasoning,
                None,
                runtime_name,
                runtime_model,
                CHAIN_REASONING_PROMPT_VERSION,
                spent_usd_micros,
                started_at_ms,
                finished_at,
                Some(&metrics),
            );
            persist_trace_row(store, trace).await;
            report.spend_usd_micros += spent_usd_micros;
            report.attempts += u64::from(attempts);
        }
    }
    Ok(())
}

fn build_chain_evidence_blob(input: &ChainReasoningInput, chain: &ChainCandidate) -> String {
    let nodes_by_id: HashMap<&str, &ChainReasoningNode> =
        input.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mut edge_records = Vec::new();
    let mut missing_gaps = chain.missing_verification_steps.clone();
    for pair in chain.member_ids.windows(2) {
        let edge = input.edges.iter().find(|e| e.from == pair[0] && e.to == pair[1]);
        if let Some(edge) = edge {
            edge_records.push(serde_json::json!({
                "from": edge.from,
                "to": edge.to,
                "kind": edge.label,
                "edge_id": edge.edge_id,
                "evidence_ref": edge.evidence_ref,
                "source": edge.source,
                "cross_repo": edge.cross_repo,
            }));
        } else {
            missing_gaps.push(format!("No graph edge proves {} -> {}", pair[0], pair[1]));
        }
    }
    let proof = chain_terminal_live_proof(input, chain);
    let has_live_proof =
        chain.member_ids.iter().filter_map(|id| nodes_by_id.get(id.as_str())).any(|n| {
            matches!(
                n.graph_kind.as_deref(),
                Some("verification_attempt") | Some("verified_vulnerability")
            )
        });
    if !has_live_proof {
        missing_gaps.push(
            "Live verification attempt or confirmed vulnerability not yet attached".to_string(),
        );
    }
    if has_live_proof && !proof.terminal_live_proof {
        missing_gaps.push(
            "Live proof is present but not at the terminal impact node; replay the full chain to terminal impact"
                .to_string(),
        );
    }
    missing_gaps.sort();
    missing_gaps.dedup();

    let members = chain
        .member_ids
        .iter()
        .filter_map(|id| {
            nodes_by_id.get(id.as_str()).map(|n| {
                serde_json::json!({
                    "id": n.id,
                    "graph_kind": n.graph_kind,
                    "label": n.label,
                    "ref_id": n.ref_id,
                    "repo": n.repo,
                    "path": n.path,
                    "line": n.line,
                    "cap": n.cap,
                    "rule": n.rule,
                    "severity": n.severity,
                    "routes": n.routes,
                    "roles": n.roles,
                    "objects": n.objects,
                    "evidence_refs": n.evidence_refs,
                })
            })
        })
        .collect::<Vec<_>>();

    serde_json::json!({
        "schema_version": 1,
        "graph_backed": true,
        "member_ids": chain.member_ids,
        "members": members,
        "edge_provenance": edge_records,
        "model_edge_provenance": chain.edge_provenance,
        "prerequisites": chain.prerequisites,
        "evidence": chain.evidence,
        "blast_radius": chain.blast_radius,
        "confidence": chain.confidence,
        "terminal_live_proof": proof.terminal_live_proof,
        "missing_verification_steps": missing_gaps,
    })
    .to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChainLiveProof {
    terminal_live_proof: bool,
    verification_attempt_id: Option<String>,
    verified_vulnerability_id: Option<String>,
}

fn chain_terminal_live_proof(
    input: &ChainReasoningInput,
    chain: &ChainCandidate,
) -> ChainLiveProof {
    let nodes_by_id: HashMap<&str, &ChainReasoningNode> =
        input.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let verification_attempt_id =
        chain.member_ids.iter().filter_map(|id| nodes_by_id.get(id.as_str())).find_map(|n| {
            if n.graph_kind.as_deref() == Some(NODE_VERIFICATION_ATTEMPT) {
                n.ref_id.clone()
            } else {
                None
            }
        });
    let verified_vulnerability_id =
        chain.member_ids.iter().filter_map(|id| nodes_by_id.get(id.as_str())).find_map(|n| {
            if n.graph_kind.as_deref() == Some(NODE_VERIFIED_VULNERABILITY) {
                n.ref_id.clone()
            } else {
                None
            }
        });
    let terminal_live_proof =
        chain.member_ids.last().and_then(|id| nodes_by_id.get(id.as_str())).is_some_and(|n| {
            matches!(
                n.graph_kind.as_deref(),
                Some(NODE_VERIFICATION_ATTEMPT) | Some(NODE_VERIFIED_VULNERABILITY)
            )
        });
    ChainLiveProof { terminal_live_proof, verification_attempt_id, verified_vulnerability_id }
}

fn verified_vulnerability_from_chain(
    input: &ChainReasoningInput,
    chain: &ChainCandidate,
    rec: &ChainRecord,
    project_id: &str,
    verification_attempt_id: Option<&str>,
    terminal_vulnerability_id: Option<&str>,
    now_ms: i64,
) -> VerifiedVulnerabilityRecord {
    let nodes_by_id: HashMap<&str, &ChainReasoningNode> =
        input.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let members = chain
        .member_ids
        .iter()
        .filter_map(|id| nodes_by_id.get(id.as_str()).copied())
        .collect::<Vec<_>>();
    let source_candidate_ids = members
        .iter()
        .filter(|n| n.graph_kind.as_deref() == Some(NODE_CANDIDATE))
        .filter_map(|n| n.ref_id.clone())
        .collect::<Vec<_>>();
    let source_signal_ids = members
        .iter()
        .filter(|n| n.graph_kind.as_deref() == Some(NODE_SIGNAL))
        .filter_map(|n| n.ref_id.clone())
        .collect::<Vec<_>>();
    let affected_components = members
        .iter()
        .map(|n| {
            serde_json::json!({
                "kind": "chain_member",
                "graph_kind": n.graph_kind,
                "id": n.id,
                "ref_id": n.ref_id,
                "repo": n.repo,
                "path": n.path,
                "line": n.line,
                "routes": n.routes,
                "roles": n.roles,
                "objects": n.objects,
            })
        })
        .collect::<Vec<_>>();
    let base_severity = rec.severity.as_deref().unwrap_or("High");
    let severity = chain_report_severity(base_severity, rec.cross_repo, members.len());
    let confidence = f64::from(chain.confidence).clamp(0.0, 100.0) / 100.0;
    let risk_score = chain_risk_score(&severity, confidence, rec.cross_repo, members.len());
    let proof_note = verification_attempt_id
        .map(|id| format!(" Terminal live proof is attached to verification attempt `{id}`."))
        .unwrap_or_else(|| {
            " Terminal live proof is attached to a verified vulnerability node.".to_string()
        });
    VerifiedVulnerabilityRecord {
        id: terminal_vulnerability_id
            .filter(|id| !id.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("vuln-{}", rec.id)),
        run_id: rec.run_id.clone(),
        project_id: project_id.to_string(),
        title: chain_title(rec.cross_repo, members.len()),
        severity: severity.clone(),
        confidence,
        risk_score,
        risk_rating: canonical_risk_rating("", risk_score),
        risk_score_source: "chain-reasoning".to_string(),
        risk_score_rationale: format!(
            "Verified chain reasoning: terminal live proof plus {} member(s), cross-repo = {}.",
            members.len(),
            rec.cross_repo
        ),
        vuln_class: "ExploitChain".to_string(),
        affected_components,
        business_impact: chain.blast_radius.join("; "),
        evidence_summary: format!("{}{}", chain.rationale, proof_note),
        repro_steps: chain_repro_steps(chain),
        remediation: "Fix the terminal impact and every upstream precondition in the chain; then rerun chain verification before closing.".to_string(),
        source_candidate_ids,
        source_signal_ids,
        verification_attempt_ids: verification_attempt_id.map(|id| vec![id.to_string()]).unwrap_or_default(),
        chain_id: Some(rec.id.clone()),
        status: "Open".to_string(),
        first_seen: now_ms,
        last_seen: now_ms,
    }
}

fn merge_chain_vulnerability_with_existing(
    chain_vuln: VerifiedVulnerabilityRecord,
    existing: VerifiedVulnerabilityRecord,
    chain: &ChainRecord,
) -> VerifiedVulnerabilityRecord {
    let severity =
        if chain_severity_rank(&chain_vuln.severity) > chain_severity_rank(&existing.severity) {
            chain_vuln.severity.clone()
        } else {
            existing.severity.clone()
        };
    let confidence = existing.confidence.max(chain_vuln.confidence);
    let risk_score = clamp_risk_score(existing.risk_score.max(chain_vuln.risk_score));
    let mut affected_components =
        union_json_values(existing.affected_components, chain_vuln.affected_components);
    affected_components.push(serde_json::json!({
        "kind": "chain_evidence",
        "chain_id": chain.id,
    }));
    let evidence_summary = append_unique_paragraph(
        existing.evidence_summary,
        format!("Chain evidence: {}", chain_vuln.evidence_summary),
    );
    let repro_steps = append_unique_paragraph(
        existing.repro_steps,
        format!("Chain replay: {}", chain_vuln.repro_steps),
    );
    let remediation = append_unique_paragraph(
        existing.remediation,
        "Also fix the upstream chain preconditions and rerun chain verification before closing."
            .to_string(),
    );

    VerifiedVulnerabilityRecord {
        id: existing.id,
        run_id: existing.run_id,
        project_id: existing.project_id,
        title: existing.title,
        severity,
        confidence,
        risk_score,
        risk_rating: canonical_risk_rating("", risk_score),
        risk_score_source: if risk_score >= existing.risk_score {
            chain_vuln.risk_score_source
        } else {
            existing.risk_score_source
        },
        risk_score_rationale: append_unique_paragraph(
            existing.risk_score_rationale,
            chain_vuln.risk_score_rationale,
        ),
        vuln_class: existing.vuln_class,
        affected_components,
        business_impact: append_unique_paragraph(
            existing.business_impact,
            chain_vuln.business_impact,
        ),
        evidence_summary,
        repro_steps,
        remediation,
        source_candidate_ids: union_strings(
            existing.source_candidate_ids,
            chain_vuln.source_candidate_ids,
        ),
        source_signal_ids: union_strings(existing.source_signal_ids, chain_vuln.source_signal_ids),
        verification_attempt_ids: union_strings(
            existing.verification_attempt_ids,
            chain_vuln.verification_attempt_ids,
        ),
        chain_id: Some(chain.id.clone()),
        status: existing.status,
        first_seen: existing.first_seen,
        last_seen: chain_vuln.last_seen,
    }
}

fn chain_report_severity(base: &str, cross_repo: bool, members: usize) -> String {
    if cross_repo || members >= 3 || chain_severity_rank(base) >= chain_severity_rank("High") {
        "Critical".to_string()
    } else {
        base.to_string()
    }
}

fn chain_risk_score(severity: &str, confidence: f64, cross_repo: bool, members: usize) -> f64 {
    let mut score: f64 = match severity.to_ascii_lowercase().as_str() {
        "critical" => 9.2,
        "high" => 7.8,
        "medium" => 5.4,
        "low" => 2.4,
        _ => 1.0,
    };
    if confidence >= 0.85 {
        score += 0.4;
    }
    if cross_repo {
        score += 0.3;
    }
    if members >= 3 {
        score += 0.2;
    }
    clamp_risk_score(score)
}

fn chain_title(cross_repo: bool, members: usize) -> String {
    if cross_repo {
        format!("Verified cross-repo exploit chain across {members} graph nodes")
    } else {
        format!("Verified exploit chain across {members} graph nodes")
    }
}

fn chain_repro_steps(chain: &ChainCandidate) -> String {
    let mut steps = Vec::new();
    if !chain.prerequisites.is_empty() {
        steps.push(format!("Prerequisites: {}", chain.prerequisites.join("; ")));
    }
    if !chain.edge_provenance.is_empty() {
        steps.push(format!("Graph edges: {}", chain.edge_provenance.join(", ")));
    }
    if !chain.evidence.is_empty() {
        steps.push(format!("Evidence: {}", chain.evidence.join("; ")));
    }
    if steps.is_empty() {
        "Replay the chain in member order and confirm the terminal live proof still fires."
            .to_string()
    } else {
        steps.join("\n")
    }
}

fn chain_severity_rank(severity: &str) -> u8 {
    match severity.to_ascii_lowercase().as_str() {
        "critical" => 5,
        "high" => 4,
        "medium" => 3,
        "low" => 2,
        "info" | "informational" => 1,
        _ => 0,
    }
}

// ----- NovelFindingDiscovery ----------------------------------------------

// Per-call NovelFindingDiscovery cap now lives on `AiConfig` as
// `[ai] novel_discovery_per_call_cap_usd_micros`. The pass halts
// further batches once the cumulative `(run_id, OneShot)` spend
// crosses the run cap; the per-call value bounds a single batch
// below the run cap when the operator wants tighter clamps.

/// Maximum bytes of source per file forwarded into the batch prompt.
/// Files above this are truncated and the `truncated` flag is set so
/// the model knows not to invent line numbers past the visible region.
const NOVEL_DISCOVERY_FILE_TRUNCATE_BYTES: usize = 8 * 1024;

/// Hard ceiling on the raw on-disk size of a candidate file before the
/// walker skips it outright. Above this size the file is almost always
/// generated, vendored, or otherwise low-signal; truncating an enormous
/// file would waste the upstream tokens.
const NOVEL_DISCOVERY_MAX_RAW_BYTES: u64 = 256 * 1024;

/// Directories the file walker refuses to descend into. Vendored or
/// generated trees would dominate the priority list and burn budget
/// on code outside the operator's control.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "vendor",
    "_vendor",
    "__pycache__",
    "dist",
    "build",
    ".venv",
    "venv",
    "env",
    ".next",
    ".nuxt",
    "site-packages",
    "third_party",
];

/// Path-keyword score table the priority heuristic uses. Routes,
/// controllers, models and DB layer files float to the top of the
/// batch queue. The table is intentionally short and language-neutral.
const PRIORITY_KEYWORDS: &[(&str, i64)] = &[
    ("route", 6),
    ("controller", 6),
    ("handler", 5),
    ("view", 4),
    ("api", 4),
    ("model", 4),
    ("auth", 4),
    ("login", 4),
    ("query", 3),
    ("sql", 3),
    ("db", 2),
    ("exec", 3),
];

/// Maximum boost applied to [`priority_for`] when a path's historical
/// AI-promotion rate is at the ceiling (rate = 1.0). At rate = 0.0 the
/// boost is zero; the boost scales linearly in between. Sized so the
/// strongest converters can outrank a plain `route`-keyword hit (which
/// scores +6 today) without drowning out the keyword signal entirely.
const PROMOTION_RATE_WEIGHT: f64 = 10.0;

/// Counts surfaced by [`run_novel_finding_discovery_pass`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NovelFindingDiscoveryPassReport {
    pub batches_dispatched: u32,
    pub batches_halted: u32,
    pub candidates_persisted: u32,
    pub failed: u32,
    pub spend_usd_micros: i64,
    pub attempts: u64,
}

/// Fan-out NovelFindingDiscovery across every successfully-ingested
/// repo in `bundle`. No-op (returns a default report) when the
/// selected runtime does not support one-shot tasks or is unavailable.
///
/// This is the most expensive AI pass; a per-run cap (default $5
/// model spend, sourced from
/// [`DEFAULT_NOVEL_DISCOVERY_RUN_CAP_USD_MICROS`]) halts further
/// batches once the cumulative `(run_id, OneShot)` spend crosses it.
/// All output starts in Quarantine (`candidate_findings.status =
/// 'Pending'`); promotion to a real finding lands with the payload
/// verifier.
pub async fn run_novel_finding_discovery_pass(
    config: &AiConfig,
    store: &Store,
    secrets: &SecretStore,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    events: EventSink,
) -> anyhow::Result<NovelFindingDiscoveryPassReport> {
    let tracker: SharedBudgetTracker = Arc::new(
        BudgetStoreTracker::new(store.clone(), DEFAULT_NOVEL_DISCOVERY_RUN_CAP_USD_MICROS)
            .with_bucket(NOVEL_DISCOVERY_BUDGET_BUCKET),
    );
    let adapter = match selected_one_shot_runtime(
        config,
        store,
        secrets,
        DEFAULT_NOVEL_DISCOVERY_RUN_CAP_USD_MICROS,
        "novel finding discovery",
        Some(NOVEL_DISCOVERY_BUDGET_BUCKET),
    )
    .await?
    {
        Some(adapter) => adapter,
        None => return Ok(NovelFindingDiscoveryPassReport::default()),
    };

    drive_novel_finding_pass(
        adapter.as_ref(),
        tracker.as_ref(),
        store,
        bundle,
        workspaces,
        events,
        DEFAULT_NOVEL_DISCOVERY_RUN_CAP_USD_MICROS,
        config.novel_discovery_per_call_cap_usd_micros_resolved(),
        DEFAULT_FILES_PER_BATCH,
    )
    .await
}

/// Inner driver, generic over `AiRuntime` + `BudgetTracker` so tests
/// can wire a scripted runtime + in-memory tracker without going
/// through the production Anthropic adapter. The pass runs each repo's
/// batches sequentially (against one shared `(run_id, OneShot)` budget
/// bucket) so the cap check has a deterministic ordering.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn drive_novel_finding_pass<R: AiRuntime + ?Sized>(
    runtime: &R,
    tracker: &dyn BudgetTracker,
    store: &Store,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    events: EventSink,
    run_cap_usd_micros: i64,
    per_call_cap_usd_micros: i64,
    files_per_batch: usize,
) -> anyhow::Result<NovelFindingDiscoveryPassReport> {
    let mut report = NovelFindingDiscoveryPassReport::default();
    let mut halted = false;
    for repo_bundle in &bundle.per_repo {
        let RepoOutcome::Success(diags) = &repo_bundle.outcome else {
            continue;
        };
        let Some(workspace) = workspaces.get(&repo_bundle.repo) else {
            continue;
        };
        // Historical AI-promotion rate per source path in this repo,
        // used to bias the priority heuristic toward files the verifier
        // has previously confirmed. A store error degrades to "no
        // boost" rather than failing the pass; the keyword + size
        // heuristic still produces a usable ordering on its own.
        let promotion_rates =
            match store.findings().per_path_promotion_rate(&repo_bundle.repo).await {
                Ok(map) => Some(map),
                Err(err) => {
                    tracing::warn!(
                        repo = %repo_bundle.repo,
                        error = %err,
                        "novel finding discovery: per-path promotion rate lookup failed; \
                         falling back to keyword + size heuristic only"
                    );
                    None
                }
            };
        let inputs = build_novel_inputs_for_repo(
            &bundle.run_id,
            &repo_bundle.repo,
            workspace.workspace(),
            diags,
            files_per_batch,
            promotion_rates.as_ref(),
        );
        if inputs.is_empty() {
            continue;
        }
        tracing::info!(
            repo = %repo_bundle.repo,
            batches = inputs.len(),
            "novel finding discovery: dispatching repo batches"
        );
        let runtime_name = runtime.name();
        let runtime_model = runtime.default_model().to_string();
        for input in inputs {
            let spent_before = tracker
                .current_spend(&bundle.run_id, BudgetKind::OneShot)
                .await
                .map_err(|e| anyhow::anyhow!("budget tracker error: {e}"))?;
            if spent_before >= run_cap_usd_micros {
                halted = true;
                report.batches_halted += 1;
                tracing::info!(
                    spent_usd_micros = spent_before,
                    cap_usd_micros = run_cap_usd_micros,
                    "novel finding discovery: budget cap reached; halting further batches"
                );
                continue;
            }
            report.batches_dispatched += 1;
            let started_at = now_epoch_ms();
            let outcome =
                match run_novel_findings(runtime, &input, events.clone(), per_call_cap_usd_micros)
                    .await
                {
                    Ok(o) => o,
                    Err(err) => {
                        tracing::warn!(error = %err, "novel finding discovery call failed");
                        report.failed += 1;
                        if ai_error_should_halt_pass(&err) {
                            halted = true;
                            break;
                        }
                        continue;
                    }
                };
            apply_novel_outcome(
                store,
                outcome,
                &mut report,
                runtime_name,
                &runtime_model,
                started_at,
            )
            .await?;
        }
        if halted {
            // Once the run-wide cap has tripped no further repo's
            // batches should fire either, but record their count so
            // operators see the full halted surface.
            break;
        }
    }
    Ok(report)
}

/// Pure data path: walk the repo workspace, prioritise files by the
/// route/controller/model/db keyword heuristic, partition into batches
/// of `files_per_batch`, and attach the matching nyx priors per batch.
/// `promotion_rates`, when set, maps repo-relative paths to their
/// historical AI-promotion rate (see
/// `FindingStore::per_path_promotion_rate`); a non-empty entry boosts
/// the file's priority. Public so the prioritisation + batching can be
/// unit-tested without spinning up an adapter.
pub fn build_novel_inputs_for_repo(
    run_id: &str,
    repo: &str,
    workspace: &std::path::Path,
    diags: &[Diag],
    files_per_batch: usize,
    promotion_rates: Option<&HashMap<String, f64>>,
) -> Vec<NovelFindingDiscoveryInput> {
    let files = walk_source_files(workspace);
    if files.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(i64, std::path::PathBuf, u64)> = files
        .into_iter()
        .map(|p| {
            let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            let rate = promotion_rates.and_then(|map| {
                p.strip_prefix(workspace)
                    .ok()
                    .and_then(|rel| rel.to_str())
                    .and_then(|rel| map.get(rel))
                    .copied()
            });
            let score = priority_for(&p, size, rate);
            (score, p, size)
        })
        .collect();
    scored.sort_by(|a, b| {
        // Higher score first; tie-break on path to keep ordering
        // deterministic across runs.
        b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1))
    });

    // Group priors by file path so each batch only sees the priors for
    // files it actually contains.
    let mut priors_by_path: HashMap<String, Vec<PriorFinding>> = HashMap::new();
    for diag in diags {
        priors_by_path.entry(diag.path.clone()).or_default().push(PriorFinding {
            path: diag.path.clone(),
            line: diag.line,
            cap: diag.cap.clone(),
            rule: diag.rule.clone(),
        });
    }

    let mut out = Vec::new();
    let batch_size = files_per_batch.max(1);
    for (batch_idx, chunk) in scored.chunks(batch_size).enumerate() {
        let mut files = Vec::with_capacity(chunk.len());
        let mut priors = Vec::new();
        for (_, abs_path, _size) in chunk {
            let rel = match abs_path.strip_prefix(workspace) {
                Ok(r) => r.to_string_lossy().to_string(),
                Err(_) => continue,
            };
            let Some((content, truncated)) = read_truncated(abs_path) else {
                continue;
            };
            if let Some(p) = priors_by_path.get(&rel) {
                priors.extend(p.iter().cloned());
            }
            files.push(FileForReview { path: rel, content, truncated });
        }
        if files.is_empty() {
            continue;
        }
        out.push(NovelFindingDiscoveryInput {
            run_id: run_id.to_string(),
            repo: repo.to_string(),
            batch_id: format!("{repo}:{batch_idx}"),
            files,
            priors,
        });
    }
    out
}

fn walk_source_files(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    // `ignore::WalkBuilder` honours .gitignore / .ignore / global excludes
    // and skips hidden entries by default. SKIP_DIRS layers in the
    // hardcoded skips that often are absent from a repo's gitignore
    // (target, vendor, __pycache__, site-packages, etc.).
    let walker = ignore::WalkBuilder::new(root)
        .standard_filters(true)
        .hidden(true)
        .parents(false)
        .require_git(false)
        .filter_entry(|entry| {
            entry.file_name().to_str().is_some_and(|name| !SKIP_DIRS.contains(&name))
        })
        .build();
    let mut out = Vec::new();
    for result in walker {
        let Ok(entry) = result else { continue };
        let Some(ft) = entry.file_type() else { continue };
        if !ft.is_file() {
            continue;
        }
        let path = entry.into_path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !accepts_source_file(name) {
            continue;
        }
        let raw_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if raw_size > NOVEL_DISCOVERY_MAX_RAW_BYTES {
            continue;
        }
        out.push(path);
    }
    out
}

fn accepts_source_file(name: &str) -> bool {
    infer_lang(name) != "unknown"
}

fn priority_for(path: &std::path::Path, size: u64, promotion_rate: Option<f64>) -> i64 {
    let lower = path.to_string_lossy().to_lowercase();
    let mut score = 0_i64;
    for (kw, w) in PRIORITY_KEYWORDS {
        if lower.contains(kw) {
            score += *w;
        }
    }
    let s = size as i64;
    if s < 256 {
        score -= 5;
    } else if s < 2_048 {
        score += 1;
    } else if s < 50_000 {
        score += 3;
    } else if s > 200_000 {
        score -= 5;
    }
    if let Some(rate) = promotion_rate {
        let clamped = rate.clamp(0.0, 1.0);
        score += (clamped * PROMOTION_RATE_WEIGHT).round() as i64;
    }
    score
}

fn read_truncated(path: &std::path::Path) -> Option<(String, bool)> {
    let raw = std::fs::read(path).ok()?;
    let utf8 = match std::str::from_utf8(&raw) {
        Ok(s) => s.to_string(),
        Err(_) => return None,
    };
    if utf8.len() <= NOVEL_DISCOVERY_FILE_TRUNCATE_BYTES {
        Some((utf8, false))
    } else {
        // Truncate on a char boundary so we never split a UTF-8 sequence.
        let mut cut = NOVEL_DISCOVERY_FILE_TRUNCATE_BYTES;
        while cut > 0 && !utf8.is_char_boundary(cut) {
            cut -= 1;
        }
        let mut head = utf8[..cut].to_string();
        if !head.ends_with('\n') {
            head.push('\n');
        }
        head.push_str("... <file truncated>\n");
        Some((head, true))
    }
}

async fn apply_novel_outcome(
    store: &Store,
    outcome: NovelFindingDiscoveryOutcome,
    report: &mut NovelFindingDiscoveryPassReport,
    runtime_name: &str,
    runtime_model: &str,
    started_at_ms: i64,
) -> anyhow::Result<()> {
    let finished_at = now_epoch_ms();
    match outcome {
        NovelFindingDiscoveryOutcome::Discovered {
            run_id,
            repo,
            batch_id: _,
            output,
            prompt_version,
            spent_usd_micros,
            attempts,
            metrics,
        } => {
            report.spend_usd_micros += spent_usd_micros;
            report.attempts += u64::from(attempts);
            let trace = build_trace_row(
                TaskKind::NovelFindings,
                None,
                runtime_name,
                runtime_model,
                &prompt_version,
                spent_usd_micros,
                started_at_ms,
                finished_at,
                Some(&metrics),
            );
            let trace_id = trace.id.clone();
            persist_trace_row(store, trace).await;
            let created_at = finished_at;
            for (idx, c) in output.candidates.iter().enumerate() {
                let id = candidate_id(&run_id, &repo, c, created_at, idx);
                let rec = CandidateFindingRecord {
                    id,
                    run_id: run_id.clone(),
                    repo: repo.clone(),
                    path: c.path.clone(),
                    line: Some(i64::from(c.line)),
                    cap: c.cap.clone(),
                    rule_hint: c.rule_hint.clone(),
                    rationale: Some(c.rationale.clone()),
                    suggested_payload_hint: c.suggested_payload_hint.clone(),
                    // Pending = quarantined for AI proposals; promotion
                    // to a real finding requires the payload verifier
                    // to confirm via PayloadSynthesis + dynamic verify.
                    status: nyx_agent_core::store::CandidateStatus::Pending.as_str().to_string(),
                    prompt_version: Some(prompt_version.clone()),
                    trace_id: Some(trace_id.clone()),
                };
                match store.candidate_findings().insert(&rec).await {
                    Ok(()) => report.candidates_persisted += 1,
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            "novel finding discovery: failed to persist candidate"
                        );
                    }
                }
            }
        }
        NovelFindingDiscoveryOutcome::NoCandidates {
            run_id: _,
            repo: _,
            batch_id,
            reason,
            spent_usd_micros,
            attempts,
            metrics,
        } => {
            tracing::info!(
                batch = %batch_id,
                reason = %reason,
                "novel finding discovery: no candidates produced"
            );
            let trace = build_trace_row(
                TaskKind::NovelFindings,
                None,
                runtime_name,
                runtime_model,
                NOVEL_FINDING_DISCOVERY_PROMPT_VERSION,
                spent_usd_micros,
                started_at_ms,
                finished_at,
                Some(&metrics),
            );
            persist_trace_row(store, trace).await;
            report.spend_usd_micros += spent_usd_micros;
            report.attempts += u64::from(attempts);
        }
    }
    Ok(())
}

fn candidate_id(
    run_id: &str,
    repo: &str,
    c: &nyx_agent_types::novel::CandidateFinding,
    created_at_ms: i64,
    rank: usize,
) -> String {
    // The stable half reuses `finding_id_hash`'s 8-byte BLAKE3 truncation
    // so the candidate id mirrors the eventual `findings.id` shape if
    // the payload verifier promotes it. `run_id` + `rationale` are
    // folded into the `rule` slot so two candidates that differ only
    // in rationale do not collide.
    let folded_rule = format!(
        "{run_id}\0{rule_hint}\0{rationale}",
        rule_hint = c.rule_hint.as_deref().unwrap_or(""),
        rationale = c.rationale,
    );
    let stable = nyx_agent_core::store::finding_id_hash(
        repo,
        &c.path,
        Some(i64::from(c.line)),
        &c.cap,
        &folded_rule,
    );
    // Append created-at-ms + rank + a random 8-hex suffix so a
    // deterministic-replay path (same prompt response twice in the
    // same ms with identical rank ordering) still produces a unique
    // row.
    format!("cand-{stable}-{created_at_ms:x}-{rank:02}-{}", short_token())
}

// ----- Payload verification ----------------------------------------------

/// Counts surfaced by [`run_payload_verification_pass`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PayloadVerificationPassReport {
    /// Findings that produced a [`VerifyVerdict::Confirmed`] verdict.
    pub confirmed: u32,
    /// Findings that produced a [`VerifyVerdict::NotConfirmed`] verdict.
    pub not_confirmed: u32,
    /// Findings whose verify run surfaced an `Errored` verdict.
    pub errored: u32,
    /// Pending candidate_findings flipped to Promoted after a
    /// Confirmed verdict.
    pub candidates_promoted: u32,
    /// Verifier invocations that bubbled an unrecoverable error
    /// (workspace setup failed, sandbox spawn refused, etc.).
    pub failed: u32,
    /// Findings (including candidates) that were considered but had no
    /// payload+spec pair to verify against. Surfaces "wired up but not
    /// yet exercised" so operators can spot a missing synthesis hand-off.
    pub skipped_no_payload: u32,
}

/// Per-run timeout for each sandbox run inside the verifier. Vuln +
/// benign each get this budget; a replay-stable check doubles it.
/// Tuned for the canned shell harness fixture; production tuning lives
/// with the operator-facing knob that arrives in a later phase.
const VERIFIER_PER_RUN_TIMEOUT_SECS: u64 = 10;

/// Marker prefix in [`nyx_agent_nyx::HarnessSpec::oracle`] that selects
/// [`Oracle::SinkProbe`] over the default [`Oracle::OutputContains`].
/// Format: `sink-probe:<sentinel-path>[#<expect-contains>]`.
const SINK_PROBE_ORACLE_PREFIX: &str = "sink-probe:";

/// Drive the deterministic payload runner across every finding (and
/// AI-discovered candidate) for this run that has both a `payloads`
/// row and a `harness_specs` row available. Verdicts land back on the
/// `findings` table: `Verified` for [`VerifyVerdict::Confirmed`],
/// `Closed` for [`VerifyVerdict::NotConfirmed`]. Errored verdicts leave
/// status untouched but stamp `verdict_blob` so the UI can render the
/// failure mode. Promoted candidates land a new `findings` row with
/// `finding_origin = AiExploration`.
pub async fn run_payload_verification_pass(
    run_config: &RunConfig,
    sandbox_config: &SandboxConfig,
    store: &Store,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    events: EventSink,
) -> anyhow::Result<PayloadVerificationPassReport> {
    let runner = PayloadRunner {
        backend: pick_verifier_backend(sandbox_config),
        per_run_timeout: std::time::Duration::from_secs(VERIFIER_PER_RUN_TIMEOUT_SECS),
        replay_stable_check: run_config.replay_stable_check,
        shim_path: None,
    };
    let mut report = PayloadVerificationPassReport::default();

    // 1. Static + LLM-synthesised findings persisted under this run.
    let findings = store.findings().list_by_run(&bundle.run_id).await?;
    for finding in findings {
        let Some(workspace) = workspaces.get(&finding.repo) else {
            continue;
        };
        match drive_verify_for_finding(&runner, store, &finding, workspace, &events, &bundle.run_id)
            .await
        {
            Ok(VerifyOutcome::Skipped) => report.skipped_no_payload += 1,
            Ok(VerifyOutcome::Verdict(verdict)) => bump_verdict(&mut report, verdict),
            Err(err) => {
                tracing::warn!(error = %err, finding = %finding.id, "verifier failed");
                report.failed += 1;
            }
        }
    }

    // 2. AI-discovered candidates that have a payload+spec pre-staged
    //    against their candidate id. The synthesis hand-off that
    //    creates those rows is deferred; this pass picks them up the
    //    moment a future phase lands the synthesis side.
    let pending = store.candidate_findings().list_pending().await?;
    let now_ms = now_epoch_ms();
    for cand in pending {
        if cand.run_id != bundle.run_id {
            continue;
        }
        let Some(workspace) = workspaces.get(&cand.repo) else {
            continue;
        };
        match drive_verify_for_candidate(
            &runner,
            store,
            &cand,
            workspace,
            now_ms,
            &events,
            &bundle.run_id,
        )
        .await
        {
            Ok(VerifyOutcome::Skipped) => report.skipped_no_payload += 1,
            Ok(VerifyOutcome::Verdict(verdict)) => {
                bump_verdict(&mut report, verdict);
                if matches!(verdict, VerifyVerdict::Confirmed) {
                    report.candidates_promoted += 1;
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, candidate = %cand.id, "verifier failed");
                report.failed += 1;
            }
        }
    }

    Ok(report)
}

/// Best-effort fan-out of a `SandboxEvent` over the run-wide bus. The
/// underlying `broadcast::Sender::send` returns `Err` only when no
/// receiver is alive, which is not actionable for the verifier pass;
/// log nothing and drop the error so the pass continues.
fn emit_sandbox(events: &EventSink, event: SandboxEvent) {
    let _ = events.send(AgentEvent::Sandbox { data: event });
}

fn pick_verifier_backend(sandbox_config: &SandboxConfig) -> BackendKind {
    match sandbox_config.backend {
        SandboxBackend::Birdcage => BackendKind::Birdcage,
        // Process is the unhardened fallback; every other backend
        // lands in a later phase that grows the sandbox crate's
        // launchers, so route them through Process today.
        _ => BackendKind::Process,
    }
}

fn bump_verdict(report: &mut PayloadVerificationPassReport, verdict: VerifyVerdict) {
    match verdict {
        VerifyVerdict::Confirmed => report.confirmed += 1,
        VerifyVerdict::NotConfirmed => report.not_confirmed += 1,
        VerifyVerdict::Errored => report.errored += 1,
    }
}

enum VerifyOutcome {
    Skipped,
    Verdict(VerifyVerdict),
}

async fn drive_verify_for_finding(
    runner: &PayloadRunner,
    store: &Store,
    finding: &FindingRecord,
    workspace: &WorkspaceHandle,
    events: &EventSink,
    run_id: &str,
) -> anyhow::Result<VerifyOutcome> {
    let Some((payload, spec)) = load_payload_and_spec(store, &finding.id).await? else {
        return Ok(VerifyOutcome::Skipped);
    };
    let prompt_version = payload.prompt_version.clone();
    let spec_id = spec.id.clone();
    let started_at = now_epoch_ms();
    emit_sandbox(
        events,
        SandboxEvent::VerifierStarted {
            run_id: run_id.to_string(),
            finding_id: finding.id.clone(),
            repo: finding.repo.clone(),
            started_at_ms: started_at,
        },
    );
    let result = run_one_verify(runner, &finding.id, payload, spec, workspace).await?;
    let finished_at = now_epoch_ms();
    emit_sandbox(
        events,
        SandboxEvent::VerifierFinished {
            run_id: run_id.to_string(),
            finding_id: finding.id.clone(),
            repo: finding.repo.clone(),
            verdict: result.verdict.as_str().to_string(),
            replay_stable: result.replay_stable,
            elapsed_ms: finished_at - started_at,
        },
    );
    persist_finding_verdict(store, finding, &result).await?;
    persist_verifier_trace(
        store,
        Some(finding.id.clone()),
        runner.backend.as_str(),
        prompt_version.as_deref().unwrap_or(&spec_id),
        started_at,
        finished_at,
        Some(&spec_id),
        Some(&result),
    )
    .await;
    Ok(VerifyOutcome::Verdict(result.verdict))
}

#[allow(clippy::too_many_arguments)]
async fn persist_verifier_trace(
    store: &Store,
    finding_id: Option<String>,
    backend: &str,
    prompt_version: &str,
    started_at: i64,
    finished_at: i64,
    spec_id: Option<&str>,
    result: Option<&VerifyResult>,
) {
    let mut row = build_trace_row(
        TaskKind::Verifier,
        finding_id,
        backend,
        backend,
        prompt_version,
        0,
        started_at,
        finished_at,
        None,
    );
    row.verifier_blob = build_verifier_blob(spec_id, result);
    persist_trace_row(store, row).await;
}

/// Render the `agent_traces.verifier_blob` JSON for a Verifier row.
///
/// Returns `None` when no inputs are available (i.e. the runner failed
/// before producing a `VerifyResult` AND no spec id was known). The
/// shape matches the `agent_traces.verifier_blob` contract: every field
/// is independently optional so callers stamp whatever they have.
fn build_verifier_blob(spec_id: Option<&str>, result: Option<&VerifyResult>) -> Option<String> {
    use sha2::{Digest, Sha256};
    let mut obj = serde_json::Map::new();
    if let Some(id) = spec_id {
        obj.insert("spec_id".into(), serde_json::Value::String(id.to_string()));
    }
    if let Some(r) = result {
        let vuln_hash = hex::encode(Sha256::digest(&r.vuln_run.payload));
        let benign_hash = hex::encode(Sha256::digest(&r.benign_run.payload));
        obj.insert("vuln_payload_sha256".into(), serde_json::Value::String(vuln_hash));
        obj.insert("vuln_exit_code".into(), serde_json::Value::from(r.vuln_run.exit_code));
        obj.insert("benign_payload_sha256".into(), serde_json::Value::String(benign_hash));
        obj.insert("benign_exit_code".into(), serde_json::Value::from(r.benign_run.exit_code));
    }
    if obj.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(obj).to_string())
    }
}

async fn drive_verify_for_candidate(
    runner: &PayloadRunner,
    store: &Store,
    candidate: &CandidateFindingRecord,
    workspace: &WorkspaceHandle,
    now_ms: i64,
    events: &EventSink,
    run_id: &str,
) -> anyhow::Result<VerifyOutcome> {
    // Candidates do not yet flow through PayloadSynthesis +
    // SpecDerivation (that hand-off is the deferred
    // candidate-confirmation pipeline). The verifier picks up the
    // promotion side using a built-in per-cap harness template
    // seeded by `suggested_payload_hint` plus a constant benign
    // control. When the synthesis hand-off lands, this path swaps
    // over to the stored `payloads` / `harness_specs` rows the same
    // way [`drive_verify_for_finding`] already does.
    let Some(hint) = candidate.suggested_payload_hint.as_deref() else {
        return Ok(VerifyOutcome::Skipped);
    };
    let Some(spec_input) = builtin_harness_for_cap(&candidate.cap) else {
        return Ok(VerifyOutcome::Skipped);
    };
    let oracle = builtin_oracle_for_cap(&candidate.cap);
    let benign = builtin_benign_for_cap(&candidate.cap);
    let run = PayloadRun {
        finding_id: candidate.id.clone(),
        spec: spec_input,
        harness_source: HarnessSource::Synthesised,
        vuln_payload: hint.as_bytes().to_vec(),
        benign_payload: benign.as_bytes().to_vec(),
        oracle,
        attack_provenance: AttackProvenance::LlmSynthesised,
        workspace: workspace.workspace().to_path_buf(),
    };
    let prompt_version = candidate
        .prompt_version
        .clone()
        .unwrap_or_else(|| format!("builtin-cap:{}", candidate.cap));
    let started_at = now_epoch_ms();
    emit_sandbox(
        events,
        SandboxEvent::VerifierStarted {
            run_id: run_id.to_string(),
            finding_id: candidate.id.clone(),
            repo: candidate.repo.clone(),
            started_at_ms: started_at,
        },
    );
    let result = match runner.verify(run).await {
        Ok(r) => r,
        Err(err) => {
            tracing::warn!(error = %err, candidate = %candidate.id, "verifier errored on candidate");
            let finished_at = now_epoch_ms();
            emit_sandbox(
                events,
                SandboxEvent::VerifierFinished {
                    run_id: run_id.to_string(),
                    finding_id: candidate.id.clone(),
                    repo: candidate.repo.clone(),
                    verdict: VerifyVerdict::Errored.as_str().to_string(),
                    replay_stable: None,
                    elapsed_ms: finished_at - started_at,
                },
            );
            persist_verifier_trace(
                store,
                None,
                runner.backend.as_str(),
                &prompt_version,
                started_at,
                finished_at,
                None,
                None,
            )
            .await;
            return Ok(VerifyOutcome::Verdict(VerifyVerdict::Errored));
        }
    };
    let finished_at = now_epoch_ms();
    emit_sandbox(
        events,
        SandboxEvent::VerifierFinished {
            run_id: run_id.to_string(),
            finding_id: candidate.id.clone(),
            repo: candidate.repo.clone(),
            verdict: result.verdict.as_str().to_string(),
            replay_stable: result.replay_stable,
            elapsed_ms: finished_at - started_at,
        },
    );
    if matches!(result.verdict, VerifyVerdict::Confirmed) {
        promote_candidate(store, candidate, &result, now_ms).await?;
    } else {
        tracing::info!(
            candidate = %candidate.id,
            verdict = %result.verdict.as_str(),
            "verifier ran on candidate but did not promote"
        );
    }
    // Candidate trace row carries `finding_id = None` until the
    // data-model convergence stitches candidate ids into trace rows
    // (deferred). When promoted, the same call's verdict is also
    // attributed via the freshly minted `findings` row's
    // `verdict_blob`, so operators still have a path back to this
    // call.
    persist_verifier_trace(
        store,
        None,
        runner.backend.as_str(),
        &prompt_version,
        started_at,
        finished_at,
        None,
        Some(&result),
    )
    .await;
    persist_candidate_finding_verifier_memory(store, candidate, &result, now_ms).await;
    Ok(VerifyOutcome::Verdict(result.verdict))
}

async fn persist_candidate_finding_verifier_memory(
    store: &Store,
    candidate: &CandidateFindingRecord,
    result: &VerifyResult,
    now_ms: i64,
) {
    let Some(project_id) =
        store.runs().get(&candidate.run_id).await.ok().flatten().and_then(|run| run.project_id)
    else {
        return;
    };
    let memory_result = match result.verdict {
        VerifyVerdict::Confirmed => "confirmed",
        VerifyVerdict::NotConfirmed => "rejected",
        VerifyVerdict::Errored => "inconclusive",
    };
    let input = ExplorationMemoryInput {
        project_id,
        repo: candidate.repo.clone(),
        run_id: candidate.run_id.clone(),
        source: "candidate_finding_verifier".to_string(),
        hypothesis: candidate.rationale.clone().unwrap_or_else(|| candidate.cap.clone()),
        endpoint: None,
        role_context: None,
        object_context: Some(candidate.path.clone()),
        result: memory_result.to_string(),
        reason: format!("candidate verifier returned {}", result.verdict.as_str()),
        useful_markers: candidate.suggested_payload_hint.clone().into_iter().collect(),
        auth_session_notes: None,
        follow_up_ideas: if matches!(result.verdict, VerifyVerdict::Confirmed) {
            vec!["look for adjacent source paths with the same vulnerable pattern".to_string()]
        } else {
            vec!["do not retry this payload/spec pair without changing the harness".to_string()]
        },
        candidate_id: None,
        verification_attempt_id: None,
        trace_id: candidate.trace_id.clone(),
        created_at: now_ms,
    };
    if let Err(err) = store.exploration_memory().upsert(&input).await {
        tracing::warn!(
            candidate_id = %candidate.id,
            error = %err,
            "failed to persist candidate verifier exploration memory"
        );
    }
}

/// Built-in per-cap shell/python harness templates the candidate
/// verifier uses until the candidate-confirmation pipeline starts
/// driving real spec-derivation on candidates. Returning `None` makes
/// the verifier skip the candidate (it cannot construct a deterministic
/// harness for a cap it does not recognise). Caps that need a richer
/// runtime than a single `sh -c` invocation (XSS / SSRF / XXE / ...)
/// stay `None` until the candidate-confirmation pipeline lands real
/// per-candidate spec derivation.
fn builtin_harness_for_cap(cap: &str) -> Option<HarnessSpecInput> {
    match cap {
        "SQL_QUERY" | "SQLi" => Some(HarnessSpecInput {
            cap: cap.to_string(),
            lang: "shell".to_string(),
            setup: vec!["STORED='alice:pw1\\nbob:pw2\\nadmin:TOP_SECRET'".to_string()],
            invoke: "printf '%b\\n' \"$STORED\" | grep -E @PAYLOAD || true".to_string(),
            teardown: vec![],
        }),
        "OS_COMMAND" | "CMDi" => Some(HarnessSpecInput {
            cap: cap.to_string(),
            lang: "shell".to_string(),
            setup: vec![],
            // Evaluate the payload as a shell command. The sentinel-file
            // oracle (`OSCMD_SENTINEL`) fires when the payload performs
            // an injection side effect (touches the flag file); the
            // benign control is a shell no-op that leaves the workspace
            // untouched.
            invoke: "sh -c @PAYLOAD".to_string(),
            teardown: vec![],
        }),
        "PATH_TRAVERSAL" => Some(HarnessSpecInput {
            cap: cap.to_string(),
            lang: "shell".to_string(),
            // Stand up a `safe_dir/` the application "intends" to keep
            // payload-supplied filenames inside, and stage an out-of-band
            // secret (`oracle_secret.txt`) in the workspace root.
            // `oracle_secret.txt` is reachable only by traversing out of
            // `safe_dir/` (`safe_dir/../oracle_secret.txt`), so an
            // AI-suggested payload that escapes the intended container
            // leaks `TOP_SECRET_TRAV` to stdout and trips the oracle.
            setup: vec![
                "mkdir -p safe_dir".to_string(),
                "printf 'workspace\\n' > safe_dir/innocent.txt".to_string(),
                "printf 'TOP_SECRET_TRAV\\n' > oracle_secret.txt".to_string(),
            ],
            invoke: "cat -- safe_dir/@PAYLOAD 2>/dev/null || true".to_string(),
            teardown: vec![],
        }),
        "XXE" => Some(HarnessSpecInput {
            cap: cap.to_string(),
            lang: "python".to_string(),
            // Drive expat directly so the harness exercises the
            // DOCTYPE + ENTITY parse path that defines the XXE class.
            // CharacterDataHandler captures element text so an
            // entity-expanded marker reaches stdout; a malformed or
            // entity-free payload simply prints nothing.
            setup: vec![
                "import xml.parsers.expat".to_string(),
                "parser = xml.parsers.expat.ParserCreate()".to_string(),
                "out = []".to_string(),
                "parser.CharacterDataHandler = lambda d: out.append(d)".to_string(),
            ],
            invoke: "try:\n    parser.Parse(@PAYLOAD, True)\n    print(''.join(out))\nexcept Exception as e:\n    print('parse-error:', e)".to_string(),
            teardown: vec![],
        }),
        "DESERIALISATION" => Some(HarnessSpecInput {
            cap: cap.to_string(),
            lang: "python".to_string(),
            // Feed the payload bytes verbatim into `pickle.loads`. A
            // protocol-0 pickle that REDUCEs `builtins.print` with the
            // `TOP_SECRET_PICKLE` marker writes the marker to stdout
            // during unpickling; a benign pickle (`b"N."` = NONE+STOP)
            // returns `None` with no side effect.
            setup: vec!["import pickle".to_string()],
            invoke: "try:\n    pickle.loads(@PAYLOAD)\nexcept Exception as e:\n    print('deserialise-error:', e)".to_string(),
            teardown: vec![],
        }),
        _ => None,
    }
}

/// Sentinel filename the `OS_COMMAND` / `CMDi` builtin harness probes
/// to confirm an injection. Kept in sync with the `SinkProbe` oracle
/// returned by [`builtin_oracle_for_cap`].
const OSCMD_SENTINEL: &str = "nyx_oscmd.flag";

fn builtin_oracle_for_cap(cap: &str) -> Oracle {
    match cap {
        "SQL_QUERY" | "SQLi" => Oracle::OutputContains { marker: "TOP_SECRET".to_string() },
        "OS_COMMAND" | "CMDi" => {
            Oracle::SinkProbe { sentinel_path: OSCMD_SENTINEL.to_string(), expect_contains: None }
        }
        "PATH_TRAVERSAL" => Oracle::OutputContains { marker: "TOP_SECRET_TRAV".to_string() },
        "XXE" => Oracle::OutputContains { marker: "TOP_SECRET_XXE".to_string() },
        "DESERIALISATION" => Oracle::OutputContains { marker: "TOP_SECRET_PICKLE".to_string() },
        _ => Oracle::OutputContains { marker: "ORACLE_FIRED".to_string() },
    }
}

fn builtin_benign_for_cap(cap: &str) -> &'static str {
    match cap {
        "SQL_QUERY" | "SQLi" => "^alice$",
        // POSIX shell no-op: parses, executes, no filesystem effect.
        "OS_COMMAND" | "CMDi" => ":",
        // Workspace-local file the harness's `safe_dir/` actually
        // contains; resolves to `safe_dir/innocent.txt` and leaks no
        // secret marker.
        "PATH_TRAVERSAL" => "innocent.txt",
        // Well-formed XML with no DOCTYPE / entities: expat parses it
        // and CharacterDataHandler captures "workspace", which is
        // distinct from the `TOP_SECRET_XXE` marker the oracle expects.
        "XXE" => "<r>workspace</r>",
        // Protocol-0 pickle of `None` (NONE + STOP): unpickles cleanly
        // with no `__reduce__` side effect, so stdout stays empty.
        "DESERIALISATION" => "N.",
        _ => "__nyx_benign_control__",
    }
}

async fn load_payload_and_spec(
    store: &Store,
    finding_id: &str,
) -> anyhow::Result<Option<(PayloadRecord, HarnessSpecRecord)>> {
    let payloads = store.payloads().list_for_finding(finding_id).await?;
    let Some(payload) = payloads.into_iter().next() else {
        return Ok(None);
    };
    // The spec back-link lives on `findings.spec_id`, but candidates do
    // not yet have a back-link column (deferred). Fall back to picking
    // the most-recent spec for the payload's cap; in production each
    // finding has exactly one spec for the cap so this is unambiguous
    // until cross-cap variants land.
    let specs = store.harness_specs().list_by_cap(&payload.cap).await?;
    let Some(spec) = specs.into_iter().last() else {
        return Ok(None);
    };
    Ok(Some((payload, spec)))
}

async fn run_one_verify(
    runner: &PayloadRunner,
    finding_id: &str,
    payload: PayloadRecord,
    spec: HarnessSpecRecord,
    workspace: &WorkspaceHandle,
) -> anyhow::Result<VerifyResult> {
    let parsed = match nyx_agent_nyx::HarnessSpec::from_json(&spec.spec_blob) {
        Ok((p, _)) => p,
        Err(err) => {
            return Ok(VerifyResult::errored(
                finding_id.to_string(),
                derive_oracle("output-contains-error"),
                empty_verify_run(&payload.vuln_bytes),
                empty_verify_run(payload.benign_bytes.as_deref().unwrap_or_default()),
                attack_provenance_from(payload.attack_provenance.as_deref()),
                format!("harness spec parse failed: {err}"),
            ));
        }
    };
    let spec_input = HarnessSpecInput {
        cap: parsed.cap.clone(),
        lang: parsed.lang.clone(),
        setup: parsed.setup.clone(),
        invoke: parsed.invoke.clone(),
        teardown: parsed.teardown.clone(),
    };
    let oracle = derive_oracle(&parsed.oracle);
    let attack_provenance = attack_provenance_from(payload.attack_provenance.as_deref());
    if let Some(reason) = degenerate_oracle_reason(&oracle) {
        return Ok(VerifyResult::errored(
            finding_id.to_string(),
            oracle,
            empty_verify_run(&payload.vuln_bytes),
            empty_verify_run(payload.benign_bytes.as_deref().unwrap_or_default()),
            attack_provenance,
            format!("oracle degenerate: {reason}"),
        ));
    }
    let benign_bytes = payload.benign_bytes.clone().unwrap_or_default();
    let run = PayloadRun {
        finding_id: finding_id.to_string(),
        spec: spec_input,
        harness_source: HarnessSource::Synthesised,
        vuln_payload: payload.vuln_bytes.clone(),
        benign_payload: benign_bytes,
        oracle,
        attack_provenance,
        workspace: workspace.workspace().to_path_buf(),
    };
    match runner.verify(run).await {
        Ok(r) => Ok(r),
        Err(err) => Ok(VerifyResult::errored(
            finding_id.to_string(),
            derive_oracle(&parsed.oracle),
            empty_verify_run(&payload.vuln_bytes),
            empty_verify_run(payload.benign_bytes.as_deref().unwrap_or_default()),
            attack_provenance,
            format!("payload runner: {err}"),
        )),
    }
}

fn attack_provenance_from(label: Option<&str>) -> AttackProvenance {
    match label {
        Some("LlmSynthesised") => AttackProvenance::LlmSynthesised,
        _ => AttackProvenance::Curated,
    }
}

fn empty_verify_run(payload: &[u8]) -> nyx_agent_types::verify::VerifyRun {
    nyx_agent_types::verify::VerifyRun {
        payload: payload.to_vec(),
        oracle_fired: false,
        exit_code: -1,
        timed_out: false,
        stdout: Vec::new(),
        stderr: Vec::new(),
        duration_ms: 0,
    }
}

/// Convert the harness spec's free-form `oracle` string into a typed
/// [`Oracle`] predicate. The default is `Oracle::OutputContains` with
/// the entire string as the marker; an oracle prefixed by
/// `sink-probe:<sentinel-path>[#<expect-contains>]` selects
/// [`Oracle::SinkProbe`].
fn derive_oracle(raw: &str) -> Oracle {
    if let Some(rest) = raw.strip_prefix(SINK_PROBE_ORACLE_PREFIX) {
        let (path, expect) = match rest.split_once('#') {
            Some((p, e)) => (p.to_string(), Some(e.to_string())),
            None => (rest.to_string(), None),
        };
        Oracle::SinkProbe { sentinel_path: path, expect_contains: expect }
    } else {
        Oracle::OutputContains { marker: raw.to_string() }
    }
}

/// Detect oracle shapes that would silently coerce every verify to
/// `NotConfirmed` because their predicate can never fire: an empty
/// `OutputContains` marker, or a `SinkProbe` with no sentinel path.
/// Returns a short diagnostic when degenerate so the caller can stamp
/// `VerifyVerdict::Errored` instead of running the sandbox.
fn degenerate_oracle_reason(oracle: &Oracle) -> Option<&'static str> {
    match oracle {
        Oracle::OutputContains { marker } if marker.trim().is_empty() => {
            Some("OutputContains marker is empty")
        }
        Oracle::SinkProbe { sentinel_path, .. } if sentinel_path.trim().is_empty() => {
            Some("SinkProbe sentinel_path is empty")
        }
        _ => None,
    }
}

/// Serialise a `VerifyResult` and stamp `kind = "VerifyResult"` at the
/// top level so the UI can distinguish verifier output from the
/// legacy free-form `{"message": ...}` verdict blob without sniffing
/// field names. The original `VerifyResult` fields remain at the top
/// level so direct `serde_json::from_str::<VerifyResult>` consumers
/// keep working.
fn stamp_verdict_kind(result: &VerifyResult) -> anyhow::Result<String> {
    let mut value = serde_json::to_value(result)?;
    if let Some(obj) = value.as_object_mut() {
        obj.insert("kind".to_string(), serde_json::Value::String("VerifyResult".to_string()));
    }
    Ok(serde_json::to_string(&value)?)
}

async fn persist_finding_verdict(
    store: &Store,
    finding: &FindingRecord,
    result: &VerifyResult,
) -> anyhow::Result<()> {
    // Stamp the `VerifyResult` JSON with a typed `kind` discriminator so
    // the API and UI can distinguish verifier output from the legacy
    // free-form `{"message": ...}` verdict blob without sniffing
    // field names. The fields remain at the top level so direct
    // `serde_json::from_str::<VerifyResult>` consumers still parse.
    let verdict_blob = stamp_verdict_kind(result)?;
    let new_status = match result.verdict {
        // Verified = the verifier confirmed an actual exploit landed.
        VerifyVerdict::Confirmed => "Verified",
        // Closed = the verifier ran cleanly but the differential rule
        // rejected the finding. Operators can re-open by retriaging.
        VerifyVerdict::NotConfirmed => "Closed",
        // Errored leaves the row's status alone so a transient failure
        // does not bury an open finding; verdict_blob carries the
        // diagnostic for triage.
        VerifyVerdict::Errored => finding.status.as_str(),
    };
    let attack_provenance = result.attack_provenance.as_str();
    store
        .findings()
        .set_verify_result(&finding.id, new_status, &verdict_blob, attack_provenance)
        .await?;
    Ok(())
}

async fn promote_candidate(
    store: &Store,
    candidate: &CandidateFindingRecord,
    result: &VerifyResult,
    now_ms: i64,
) -> anyhow::Result<()> {
    let line = candidate.line.unwrap_or(-1);
    let rule =
        candidate.rule_hint.clone().unwrap_or_else(|| format!("ai-exploration:{}", candidate.cap));
    let id = nyx_agent_core::store::finding_id_hash(
        &candidate.repo,
        &candidate.path,
        Some(line),
        &candidate.cap,
        &rule,
    );
    let verdict_blob = stamp_verdict_kind(result)?;
    let rec = FindingRecord {
        id,
        run_id: candidate.run_id.clone(),
        repo: candidate.repo.clone(),
        path: candidate.path.clone(),
        line: candidate.line,
        cap: candidate.cap.clone(),
        rule,
        severity: "High".to_string(),
        status: "Verified".to_string(),
        finding_origin: FindingOrigin::AiExploration.as_str().to_string(),
        first_seen: now_ms,
        last_seen: now_ms,
        superseded_by: None,
        triage_state: "Open".to_string(),
        triage_assigned_to: None,
        verdict_blob: Some(verdict_blob),
        repro_path: None,
        attack_provenance: Some(result.attack_provenance.as_str().to_string()),
        prompt_version: candidate.prompt_version.clone(),
        chain_id: None,
        spec_id: None,
    };
    store.findings().upsert(&rec).await?;
    store
        .candidate_findings()
        .set_status(&candidate.id, CandidateStatus::Promoted.as_str())
        .await?;
    Ok(())
}

// ----- AI Exploration ----------------------------------------------------

/// Counts surfaced by [`run_ai_exploration_pass`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AiExplorationPassReport {
    /// Repos for which the exploration driver dispatched the agent
    /// loop (escape gate green, budget available).
    pub explorations_dispatched: u32,
    /// Repos that were skipped because the escape-suite gate returned
    /// red. The driver halted before the agent loop fired; operators
    /// see the failing fixture name in the structured log.
    pub halted_escape_suite_red: u32,
    /// Repos that were skipped because the per-run budget cap was
    /// already exhausted before the call.
    pub halted_budget_exhausted: u32,
    /// `findings` rows written with `finding_origin = AiExploration`
    /// and `status = Quarantine`. The payload verifier promotes them
    /// to `Verified` when a payload + spec pair confirms.
    pub findings_quarantined: u32,
    /// Exploration calls that bubbled an unrecoverable upstream error
    /// (transport, malformed response).
    pub failed: u32,
    /// Sum of `cost_usd_micros` reported by every dispatched call.
    pub spend_usd_micros: i64,
}

/// Static escape-suite gate. Pre-records a verdict the binary supplies
/// at startup so the AI driver can refer to a recent escape-suite
/// run's result without spinning up a fresh probe on every
/// exploration. Wiring this to a real periodic probe lives with the
/// release-pipeline phase that already needs to think about the
/// `nyx-sandbox-shim` install path; until then operators rely on CI's
/// own escape-suite run.
#[derive(Debug, Clone)]
pub struct StaticEscapeSuiteGate {
    verdict: EscapeSuiteVerdict,
}

impl StaticEscapeSuiteGate {
    pub fn green() -> Self {
        Self { verdict: EscapeSuiteVerdict::Green }
    }

    #[cfg(test)]
    pub fn red(fixture: impl Into<String>, reason: impl Into<String>) -> Self {
        Self { verdict: EscapeSuiteVerdict::Red { fixture: fixture.into(), reason: reason.into() } }
    }
}

#[async_trait]
impl EscapeSuiteGate for StaticEscapeSuiteGate {
    async fn check(&self) -> Result<EscapeSuiteVerdict, AiError> {
        Ok(self.verdict.clone())
    }
}

/// Fan-out AI Exploration across every successfully-ingested repo in
/// `bundle`. No-op (returns a default report) when the selected runtime
/// does not support agent loops or its CLI binary is unavailable.
///
/// Each repo gets one exploration call routed through the Claude Code
/// agent loop. Findings the model records via the
/// `record_exploration_finding` tool land in the `findings` table with
/// `finding_origin = AiExploration` and `status = Quarantine`; the
/// payload verifier promotes them to `Verified` when a payload + spec
/// pair confirms (the same dynamic-confirm gate NovelFindingDiscovery
/// candidates flow through).
///
/// The escape suite is a precondition: a red fixture halts the driver
/// before any agent loop fires. A run-wide hard cap (default $10 in
/// USD micros, tuned for Claude Opus pricing) bounds spend; a per-task
/// soft cap emits a warning frame on the event bus without halting
/// the run.
pub async fn run_ai_exploration_pass(
    config: &AiConfig,
    run_config: &RunConfig,
    store: &Store,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    target_urls: &[String],
    escape_gate: &dyn EscapeSuiteGate,
    events: EventSink,
    traces_dir: &std::path::Path,
) -> anyhow::Result<AiExplorationPassReport> {
    let run_cap_usd_micros =
        config.exploration_run_cap_usd_micros_resolved(DEFAULT_EXPLORATION_RUN_CAP_USD_MICROS);
    let soft_cap_usd_micros =
        config.exploration_soft_cap_usd_micros_resolved(DEFAULT_EXPLORATION_SOFT_CAP_USD_MICROS);
    // Exploration has its own tight `exploration_run_cap` knob, so it
    // gets a private budget bucket; otherwise its low cap would pin the
    // shared AgentLoop bucket and starve the later attack-agent pass.
    let adapter = match selected_agent_loop_runtime(
        config,
        store,
        run_cap_usd_micros,
        None,
        Some(EXPLORATION_BUDGET_BUCKET),
    )
    .await
    {
        Some(adapter) => adapter,
        None => return Ok(AiExplorationPassReport::default()),
    };

    drive_ai_exploration_pass(
        adapter.as_ref(),
        store,
        bundle,
        workspaces,
        target_urls,
        escape_gate,
        events,
        traces_dir,
        soft_cap_usd_micros,
        run_cap_usd_micros,
        run_config.research_mode_enabled,
    )
    .await
}

/// Inner driver, generic over `AiRuntime` so tests can supply a
/// scripted agent-loop runtime without going through the production
/// Claude Code adapter. Shape mirrors the `drive_novel_finding_pass`
/// inner driver.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn drive_ai_exploration_pass<R: AiRuntime + ?Sized>(
    runtime: &R,
    store: &Store,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    target_urls: &[String],
    escape_gate: &dyn EscapeSuiteGate,
    events: EventSink,
    traces_dir: &std::path::Path,
    soft_cap_usd_micros: i64,
    run_cap_usd_micros: i64,
    research_mode_enabled: bool,
) -> anyhow::Result<AiExplorationPassReport> {
    let mut report = AiExplorationPassReport::default();
    let runtime_name = runtime.name();
    let runtime_model = runtime.default_model().to_string();
    let candidate_leads = match store.pentest_candidates().list_by_run(&bundle.run_id).await {
        Ok(candidates) => candidates,
        Err(err) => {
            tracing::warn!(
                run_id = %bundle.run_id,
                error = %err,
                "ai exploration: failed to load known scanner leads; continuing without them"
            );
            Vec::new()
        }
    };
    for repo_bundle in &bundle.per_repo {
        let RepoOutcome::Success(_) = &repo_bundle.outcome else {
            continue;
        };
        let Some(workspace) = workspaces.get(&repo_bundle.repo) else {
            continue;
        };
        let memory_hints = candidate_memory_hints_for_repo(&candidate_leads, &repo_bundle.repo);
        let prior_memory_rows = match store
            .exploration_memory()
            .relevant_for_repo(
                &bundle.project_id,
                &repo_bundle.repo,
                EXPLORATION_KNOWN_LEADS_MAX,
                &memory_hints,
            )
            .await
        {
            Ok(rows) => rows,
            Err(err) => {
                tracing::warn!(
                    repo = %repo_bundle.repo,
                    error = %err,
                    "ai exploration: failed to load prior exploration memory"
                );
                Vec::new()
            }
        };
        let known_leads = exploration_known_leads_for_repo_with_memory(
            &candidate_leads,
            &repo_bundle.repo,
            EXPLORATION_KNOWN_LEADS_MAX,
            research_mode_enabled,
            &prior_memory_rows,
        );
        let prior_memory = compact_memory_for_prompt(&prior_memory_rows, 12);
        let scope = build_exploration_scope(
            &bundle.run_id,
            &repo_bundle.repo,
            workspace.workspace(),
            target_urls,
            known_leads,
            prior_memory,
            soft_cap_usd_micros,
            run_cap_usd_micros,
            research_mode_enabled,
        );

        let started_at = now_epoch_ms();
        let outcome = match run_exploration(runtime, &scope, escape_gate, events.clone()).await {
            Ok(o) => o,
            Err(err) => {
                tracing::warn!(
                    repo = %repo_bundle.repo,
                    error = %err,
                    "ai exploration call failed"
                );
                report.failed += 1;
                if ai_error_should_halt_pass(&err) {
                    break;
                }
                continue;
            }
        };
        apply_exploration_outcome(
            store,
            &bundle.run_id,
            &repo_bundle.repo,
            &scope.task_id,
            &scope.research_focus,
            outcome,
            &mut report,
            runtime_name,
            &runtime_model,
            started_at,
            traces_dir,
            research_mode_enabled,
        )
        .await?;
    }
    Ok(report)
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AttackAgentPassReport {
    pub dispatched: u32,
    pub vulnerabilities_recorded: u32,
    pub candidates_promoted: u32,
    pub failed: u32,
    pub spend_usd_micros: i64,
}

#[allow(clippy::too_many_arguments)]
pub async fn run_attack_agent_pass(
    config: &AiConfig,
    store: &Store,
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    target_urls: &[String],
    environment_run_id: &str,
    artifact_root: &std::path::Path,
    events: EventSink,
) -> anyhow::Result<AttackAgentPassReport> {
    if target_urls.is_empty() || workspaces.is_empty() {
        return Ok(AttackAgentPassReport::default());
    }
    // Draws on the run-wide AgentLoop budget bucket (the operator's
    // `default_run_budget`, uncapped by default). No private bucket, so
    // it is never pinned by exploration's tighter cap.
    let adapter = match selected_agent_loop_runtime(
        config,
        store,
        config.default_run_budget_usd_micros_resolved(),
        Some(std::time::Duration::from_secs(ATTACK_AGENT_PER_PASS_TIMEOUT_SECS)),
        None,
    )
    .await
    {
        Some(adapter) => adapter,
        None => return Ok(AttackAgentPassReport::default()),
    };
    let runtime_name = adapter.name();
    let runtime_model = adapter.default_model().to_string();
    std::fs::create_dir_all(artifact_root)?;

    let mut report = AttackAgentPassReport::default();
    for profile in DEFAULT_ATTACK_AGENT_PROFILES {
        let profile_artifact_root = artifact_root.join(profile.slug());
        std::fs::create_dir_all(&profile_artifact_root)?;
        let candidates = store.pentest_candidates().list_by_run(&bundle.run_id).await?;
        let existing_vulnerabilities =
            store.verified_vulnerabilities().list_by_project(&bundle.project_id).await?;
        let scope = build_attack_agent_scope(
            bundle,
            workspaces,
            target_urls,
            &candidates,
            &existing_vulnerabilities,
            &profile_artifact_root,
            config.default_run_budget_usd_micros_resolved(),
            *profile,
        );
        let started_at = now_epoch_ms();
        let outcome = match run_attack_agent(adapter.as_ref(), &scope, events.clone()).await {
            Ok(outcome) => outcome,
            Err(err) => {
                report.failed += 1;
                tracing::warn!(
                    error = %err,
                    profile = profile.slug(),
                    "unsafe attack agent failed"
                );
                continue;
            }
        };
        let finished_at = now_epoch_ms();
        let AttackAgentOutcome::Completed {
            vulnerabilities,
            audit,
            final_message,
            turns,
            spent_usd_micros,
            prompt_version,
            metrics,
        } = outcome;
        report.dispatched += 1;
        report.spend_usd_micros += spent_usd_micros;

        let mut trace = build_trace_row(
            TaskKind::AttackAgent,
            None,
            runtime_name,
            &runtime_model,
            &prompt_version,
            spent_usd_micros,
            started_at,
            finished_at,
            Some(&metrics),
        );
        let trace_id = trace.id.clone();
        let structured_replay_plans = vulnerabilities
            .iter()
            .map(|vulnerability| {
                serde_json::json!({
                    "title": vulnerability.title,
                    "fingerprint": attack_vulnerability_fingerprint(vulnerability),
                    "structured_replay_plan": attack_agent_structured_replay_plan(
                        vulnerability,
                        target_urls,
                    ),
                })
            })
            .collect::<Vec<_>>();
        trace.verifier_blob = Some(
            serde_json::json!({
                "kind": "unsafe_attack_agent",
                "agent_profile": profile.slug(),
                "agent_profile_title": profile.title(),
                "run_id": &bundle.run_id,
                "project_id": &bundle.project_id,
                "turns": turns,
                "artifact_dir": profile_artifact_root.to_string_lossy(),
                "audit": audit,
                "final_message": final_message,
                "vulnerabilities": vulnerabilities.clone(),
                "structured_replay_plans": structured_replay_plans,
            })
            .to_string(),
        );
        persist_trace_row(store, trace).await;

        for vulnerability in vulnerabilities {
            match persist_attack_agent_vulnerability(
                store,
                bundle,
                environment_run_id,
                &trace_id,
                &candidates,
                &existing_vulnerabilities,
                target_urls,
                vulnerability,
                finished_at,
            )
            .await
            {
                Ok(promoted_existing) => {
                    report.vulnerabilities_recorded += 1;
                    if promoted_existing {
                        report.candidates_promoted += 1;
                    }
                }
                Err(err) => {
                    report.failed += 1;
                    tracing::warn!(
                        error = %err,
                        profile = profile.slug(),
                        "failed to persist unsafe attack-agent vulnerability"
                    );
                }
            }
        }
    }
    Ok(report)
}

fn build_attack_agent_scope(
    bundle: &RunBundle<Diag>,
    workspaces: &HashMap<String, WorkspaceHandle>,
    target_urls: &[String],
    candidates: &[PentestCandidateRecord],
    existing_vulnerabilities: &[VerifiedVulnerabilityRecord],
    artifact_root: &std::path::Path,
    run_cap_usd_micros: i64,
    profile: AttackAgentProfile,
) -> AttackAgentScope {
    let mut scope = AttackAgentScope::new(&bundle.run_id, &bundle.project_id);
    scope.profile = profile;
    scope.task_id = format!("attack-agent-{}-{}", profile.slug(), bundle.run_id);
    let mut workspaces_vec = workspaces
        .iter()
        .map(|(repo, workspace)| AttackWorkspace {
            repo: repo.clone(),
            root: workspace.workspace().to_string_lossy().to_string(),
        })
        .collect::<Vec<_>>();
    workspaces_vec.sort_by(|a, b| a.repo.cmp(&b.repo));
    scope.target_urls = target_urls.to_vec();
    scope.workspaces = workspaces_vec;
    scope.known_leads = candidates
        .iter()
        .filter(|candidate| !matches!(candidate.status.as_str(), "Rejected" | "Errored"))
        .take(40)
        .map(candidate_to_attack_agent_known_lead)
        .collect();
    scope.existing_vulnerabilities =
        existing_vulnerabilities.iter().take(30).map(existing_vulnerability_summary).collect();
    scope.artifact_dir = artifact_root.to_string_lossy().to_string();
    scope.max_turns = DEFAULT_ATTACK_AGENT_MAX_TURNS;
    scope.run_cap_usd_micros = run_cap_usd_micros;
    scope
}

fn candidate_to_attack_agent_known_lead(
    candidate: &PentestCandidateRecord,
) -> AttackAgentKnownLead {
    AttackAgentKnownLead {
        id: candidate.id.clone(),
        source: candidate.source.clone(),
        title: candidate.title.clone(),
        vuln_class: candidate.vuln_class.clone(),
        severity: candidate.severity_guess.clone(),
        status: candidate.status.clone(),
        location: candidate_location(candidate),
        hypothesis: candidate.hypothesis.clone(),
    }
}

fn existing_vulnerability_summary(
    vuln: &VerifiedVulnerabilityRecord,
) -> ExistingVulnerabilitySummary {
    ExistingVulnerabilitySummary {
        id: vuln.id.clone(),
        title: vuln.title.clone(),
        vuln_class: vuln.vuln_class.clone(),
        severity: vuln.severity.clone(),
        confidence_percent: (vuln.confidence * 100.0).round().clamp(0.0, 100.0) as u8,
        status: vuln.status.clone(),
        evidence_summary: vuln.evidence_summary.clone(),
    }
}

async fn persist_attack_agent_vulnerability(
    store: &Store,
    bundle: &RunBundle<Diag>,
    environment_run_id: &str,
    trace_id: &str,
    candidates: &[PentestCandidateRecord],
    existing_vulnerabilities: &[VerifiedVulnerabilityRecord],
    target_urls: &[String],
    vulnerability: AttackAgentVulnerability,
    now_ms: i64,
) -> anyhow::Result<bool> {
    let matched_candidate = attack_agent_candidate_match(&vulnerability, candidates);
    let matched_vulnerability =
        attack_agent_existing_vulnerability_match(&vulnerability, existing_vulnerabilities);
    let candidate_id =
        matched_candidate.as_ref().map(|candidate| candidate.id.clone()).unwrap_or_else(|| {
            format!("pc-attack-{}", attack_vulnerability_fingerprint(&vulnerability))
        });
    let promoted_existing = matched_candidate.is_some();
    let mut source_candidate_ids =
        union_strings(vulnerability.source_candidate_ids.clone(), vec![candidate_id.clone()]);
    let mut source_signal_ids = vulnerability.source_signal_ids.clone();
    let affected_components = attack_components_or_default(&vulnerability);
    let severity = attack_agent_contextual_severity(&vulnerability);
    let reported_confidence = confidence_fraction(vulnerability.confidence);
    let structured_replay_plan = attack_agent_structured_replay_plan(&vulnerability, target_urls);
    let verifier_test_plan = structured_replay_plan
        .get("plan")
        .map(|plan| plan.to_string())
        .unwrap_or_else(|| vulnerability.repro_steps.clone());
    let candidate = match matched_candidate {
        Some(existing) => PentestCandidateRecord {
            title: vulnerability.title.clone(),
            vuln_class: vulnerability.vuln_class.clone(),
            severity_guess: severity.clone(),
            affected_components: affected_components.clone(),
            hypothesis: vulnerability.business_impact.clone(),
            test_plan: verifier_test_plan.clone(),
            status: "Verified".to_string(),
            rejection_reason: None,
            confidence: reported_confidence.max(existing.confidence),
            trace_id: Some(trace_id.to_string()),
            updated_at: now_ms,
            ..existing.clone()
        },
        None => PentestCandidateRecord {
            id: candidate_id.clone(),
            run_id: bundle.run_id.clone(),
            project_id: bundle.project_id.clone(),
            source: "AttackAgent".to_string(),
            source_ids: source_signal_ids.clone(),
            title: vulnerability.title.clone(),
            vuln_class: vulnerability.vuln_class.clone(),
            severity_guess: severity.clone(),
            affected_components: affected_components.clone(),
            hypothesis: vulnerability.business_impact.clone(),
            test_plan: verifier_test_plan,
            status: "Verified".to_string(),
            rejection_reason: None,
            confidence: reported_confidence,
            trace_id: Some(trace_id.to_string()),
            created_at: now_ms,
            updated_at: now_ms,
        },
    };
    store.pentest_candidates().insert(&candidate).await?;
    store.pentest_candidates().set_status(&candidate.id, "Verified", None, now_ms).await?;

    let attempt_id = format!("va-attack-{}-{now_ms}", safe_id_fragment(&candidate.id));
    let attempt = VerificationAttemptRecord {
        id: attempt_id.clone(),
        run_id: bundle.run_id.clone(),
        project_id: bundle.project_id.clone(),
        environment_run_id: environment_run_id.to_string(),
        candidate_id: Some(candidate.id.clone()),
        chain_id: None,
        method: "agent".to_string(),
        status: "Confirmed".to_string(),
        started_at: now_ms,
        finished_at: Some(now_ms),
        duration_ms: Some(0),
        request: Some(serde_json::json!({
            "kind": "unsafe_attack_agent",
            "trace_id": trace_id,
            "target": "local dev app",
            "safety_policy": "disabled",
            "structured_replay_plan": structured_replay_plan,
        })),
        response: Some(serde_json::json!({
            "title": vulnerability.title.clone(),
            "evidence_summary": vulnerability.evidence_summary.clone(),
            "proof_artifact_paths": vulnerability.proof_artifact_paths.clone(),
        })),
        oracle: Some(serde_json::json!({
            "success": true,
            "type": "agent_reported_live_proof",
            "confidence": vulnerability.confidence,
        })),
        artifact_paths: vulnerability.proof_artifact_paths.clone(),
        error: None,
        replay_stable: None,
    };
    store.verification_attempts().insert(&attempt).await?;

    let vuln_id = if let Some(existing) = matched_vulnerability {
        existing.id.clone()
    } else if candidate.id.starts_with("pc-") {
        format!("vuln-{}", candidate.id.trim_start_matches("pc-"))
    } else {
        format!("vuln-{}", candidate.id)
    };
    let mut verification_attempt_ids = vec![attempt_id.clone()];
    let mut first_seen = now_ms;
    if let Some(existing) = store.verified_vulnerabilities().get(&vuln_id).await? {
        verification_attempt_ids =
            union_strings(existing.verification_attempt_ids, verification_attempt_ids);
        source_candidate_ids = union_strings(existing.source_candidate_ids, source_candidate_ids);
        source_signal_ids = union_strings(existing.source_signal_ids, source_signal_ids);
        first_seen = existing.first_seen;
    }
    let confidence = reported_confidence.max(candidate.confidence);
    let risk_score = attack_agent_risk_score(&severity, confidence, &vulnerability);
    let risk_score_rationale =
        attack_agent_risk_rationale(&vulnerability, &severity, confidence, risk_score);
    let vuln = VerifiedVulnerabilityRecord {
        id: vuln_id,
        run_id: bundle.run_id.clone(),
        project_id: bundle.project_id.clone(),
        title: vulnerability.title,
        severity,
        confidence,
        risk_score,
        risk_rating: canonical_risk_rating("", risk_score),
        risk_score_source: "unsafe-attack-agent".to_string(),
        risk_score_rationale,
        vuln_class: vulnerability.vuln_class,
        affected_components,
        business_impact: vulnerability.business_impact,
        evidence_summary: vulnerability.evidence_summary,
        repro_steps: vulnerability.repro_steps,
        remediation: vulnerability.remediation,
        source_candidate_ids,
        source_signal_ids,
        verification_attempt_ids,
        chain_id: None,
        status: "Open".to_string(),
        first_seen,
        last_seen: now_ms,
    };
    store.verified_vulnerabilities().upsert(&vuln).await?;
    Ok(promoted_existing)
}

fn attack_agent_structured_replay_plan(
    vulnerability: &AttackAgentVulnerability,
    target_urls: &[String],
) -> serde_json::Value {
    let policy_note = "agent-derived replay candidates must be executed only by the normal live verifier with LiveVerifierOptions and ExploitSafetyPolicy";
    match extract_attack_agent_replay_plan(vulnerability, target_urls) {
        Ok(plan) => serde_json::json!({
            "source": "agent-derived",
            "status": "available",
            "policy_enforcement": "LiveVerifierOptions/ExploitSafetyPolicy",
            "plan": plan,
            "notes": [policy_note],
        }),
        Err(reason) => serde_json::json!({
            "source": "agent-derived",
            "status": "replay_unavailable",
            "reason": reason,
            "policy_enforcement": "LiveVerifierOptions/ExploitSafetyPolicy",
            "notes": [policy_note],
        }),
    }
}

fn extract_attack_agent_replay_plan(
    vulnerability: &AttackAgentVulnerability,
    target_urls: &[String],
) -> Result<serde_json::Value, String> {
    for raw in attack_agent_json_plan_candidates(vulnerability) {
        if let Ok(Some(plan)) = normalise_live_test_plan(&raw, target_urls) {
            return Ok(plan);
        }
    }

    let steps = attack_agent_http_steps(vulnerability);
    if steps.is_empty() {
        return Err("agent proof did not include extractable HTTP endpoint/method/body/role steps"
            .to_string());
    }
    let body_contains = attack_agent_expected_markers(vulnerability);
    if body_contains.is_empty() && !steps.iter().any(|step| step.get("oracle").is_some()) {
        return Err(
            "agent proof did not include a deterministic expected result/body/header marker"
                .to_string(),
        );
    }

    let plan = serde_json::json!({
        "kind": "http_workflow",
        "hypothesis": vulnerability.business_impact,
        "steps": steps,
        "oracle": {
            "body_contains": body_contains,
        },
        "why_this_confirms": "Sanitized deterministic replay metadata extracted from unsafe attack-agent proof; execution remains gated by the normal live verifier policy.",
    });
    normalise_live_test_plan(&plan.to_string(), target_urls)?.ok_or_else(|| {
        "agent-derived HTTP workflow was rejected by live-plan normalization".to_string()
    })
}

fn attack_agent_json_plan_candidates(vulnerability: &AttackAgentVulnerability) -> Vec<String> {
    let mut candidates = Vec::new();
    for component in &vulnerability.affected_components {
        for key in ["structured_replay_plan", "replay_plan", "live_test_plan", "plan"] {
            if let Some(value) = component.get(key) {
                candidates.push(value.to_string());
            }
        }
    }
    for raw in [&vulnerability.repro_steps, &vulnerability.evidence_summary] {
        let trimmed = raw.trim();
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            candidates.push(trimmed.to_string());
        }
        for block in fenced_json_blocks(trimmed) {
            candidates.push(block);
        }
    }
    candidates
}

fn fenced_json_blocks(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = raw;
    while let Some(start) = rest.find("```") {
        rest = &rest[start + 3..];
        if let Some(newline) = rest.find('\n') {
            let lang = rest[..newline].trim();
            rest = &rest[newline + 1..];
            if let Some(end) = rest.find("```") {
                let body = rest[..end].trim();
                if lang.eq_ignore_ascii_case("json") || body.starts_with('{') {
                    out.push(body.to_string());
                }
                rest = &rest[end + 3..];
            } else {
                break;
            }
        } else {
            break;
        }
    }
    out
}

fn attack_agent_http_steps(vulnerability: &AttackAgentVulnerability) -> Vec<serde_json::Value> {
    let mut steps = Vec::new();
    for component in &vulnerability.affected_components {
        if let Some(step) = attack_agent_step_from_component(component) {
            steps.push(step);
        }
    }
    if steps.is_empty() {
        steps.extend(attack_agent_steps_from_text(&vulnerability.repro_steps));
    }
    steps
}

fn attack_agent_step_from_component(component: &serde_json::Value) -> Option<serde_json::Value> {
    let endpoint_raw = first_string_field(component, &["url", "endpoint", "path", "route"])?;
    let (endpoint_method, endpoint) = split_method_endpoint(&endpoint_raw);
    let method = first_string_field(component, &["method", "http_method"])
        .or(endpoint_method)
        .unwrap_or_else(|| "GET".to_string());
    let mut step = serde_json::Map::new();
    step.insert("method".to_string(), serde_json::Value::String(method));
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        step.insert("url".to_string(), serde_json::Value::String(endpoint));
    } else {
        step.insert("path".to_string(), serde_json::Value::String(endpoint));
    }
    if let Some(role) = first_string_field(component, &["role", "as", "auth_role"]) {
        step.insert("as".to_string(), serde_json::Value::String(role));
    }
    if let Some(body) = first_value_field(component, &["json", "body", "request_body", "payload"]) {
        if body.is_object() || body.is_array() {
            step.insert("json".to_string(), body.clone());
        } else if let Some(body) = body.as_str() {
            step.insert("body".to_string(), serde_json::Value::String(body.to_string()));
        }
    }
    let destructive = method_is_state_changing(step.get("method").and_then(|v| v.as_str()));
    step.insert("destructive".to_string(), serde_json::Value::Bool(destructive));
    if let Some(marker) =
        first_string_field(component, &["expected_result", "expected", "body_contains", "marker"])
    {
        step.insert("oracle".to_string(), serde_json::json!({ "body_contains": [marker] }));
    }
    Some(serde_json::Value::Object(step))
}

fn attack_agent_steps_from_text(raw: &str) -> Vec<serde_json::Value> {
    raw.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            if line.contains("curl ") || line.starts_with("curl") {
                return attack_agent_step_from_curl(line);
            }
            attack_agent_step_from_http_line(line)
        })
        .collect()
}

fn attack_agent_step_from_http_line(line: &str) -> Option<serde_json::Value> {
    let mut parts = line.split_whitespace();
    let method = parts.next()?.trim_matches(|c: char| !c.is_ascii_alphabetic());
    if !is_http_method(method) {
        return None;
    }
    let endpoint = parts
        .find(|part| {
            part.starts_with('/') || part.starts_with("http://") || part.starts_with("https://")
        })?
        .trim_matches(|c: char| c == '`' || c == '"' || c == '\'' || c == ',' || c == ';')
        .to_string();
    Some(attack_agent_step_json(method, &endpoint, None, role_from_text(line)))
}

fn attack_agent_step_from_curl(line: &str) -> Option<serde_json::Value> {
    let tokens = shell_like_tokens(line);
    let mut method: Option<String> = None;
    let mut body: Option<String> = None;
    let mut url: Option<String> = None;
    let mut idx = 0;
    while idx < tokens.len() {
        match tokens[idx].as_str() {
            "-X" | "--request" => {
                if let Some(next) = tokens.get(idx + 1) {
                    method = Some(next.to_ascii_uppercase());
                    idx += 1;
                }
            }
            "-d" | "--data" | "--data-raw" | "--data-binary" | "--json" => {
                if let Some(next) = tokens.get(idx + 1) {
                    body = Some(next.clone());
                    if method.is_none() {
                        method = Some("POST".to_string());
                    }
                    idx += 1;
                }
            }
            token
                if token.starts_with("http://")
                    || token.starts_with("https://")
                    || token.starts_with('/') =>
            {
                url = Some(token.to_string());
            }
            _ => {}
        }
        idx += 1;
    }
    let url = url?;
    let method = method.unwrap_or_else(|| "GET".to_string());
    Some(attack_agent_step_json(&method, &url, body, role_from_text(line)))
}

fn attack_agent_step_json(
    method: &str,
    endpoint: &str,
    body: Option<String>,
    role: Option<String>,
) -> serde_json::Value {
    let mut step = serde_json::Map::new();
    step.insert("method".to_string(), serde_json::Value::String(method.to_ascii_uppercase()));
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        step.insert("url".to_string(), serde_json::Value::String(endpoint.to_string()));
    } else {
        step.insert("path".to_string(), serde_json::Value::String(endpoint.to_string()));
    }
    if let Some(role) = role {
        step.insert("as".to_string(), serde_json::Value::String(role));
    }
    if let Some(body) = body {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
            step.insert("json".to_string(), json);
        } else {
            step.insert("body".to_string(), serde_json::Value::String(body));
        }
    }
    step.insert(
        "destructive".to_string(),
        serde_json::Value::Bool(method_is_state_changing(Some(method))),
    );
    serde_json::Value::Object(step)
}

fn attack_agent_expected_markers(vulnerability: &AttackAgentVulnerability) -> Vec<String> {
    let mut markers = Vec::new();
    for component in &vulnerability.affected_components {
        for key in ["expected_result", "expected", "body_contains", "marker"] {
            if let Some(value) = first_string_field(component, &[key]) {
                markers.push(value);
            }
        }
    }
    for raw in [&vulnerability.evidence_summary, &vulnerability.repro_steps] {
        markers.extend(quoted_markers(raw));
    }
    let mut seen = HashSet::new();
    markers
        .into_iter()
        .map(|marker| marker.trim().to_string())
        .filter(|marker| marker.len() >= 4 && marker.len() <= 200)
        .filter(|marker| seen.insert(marker.clone()))
        .take(8)
        .collect()
}

fn quoted_markers(raw: &str) -> Vec<String> {
    let mut markers = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    for ch in raw.chars() {
        if let Some(q) = quote {
            if ch == q {
                let marker = current.trim();
                if marker.len() >= 4
                    && !marker.starts_with("http://")
                    && !marker.starts_with("https://")
                    && !marker.starts_with('/')
                {
                    markers.push(marker.to_string());
                }
                current.clear();
                quote = None;
            } else {
                current.push(ch);
            }
        } else if matches!(ch, '"' | '\'' | '`') {
            quote = Some(ch);
        }
    }
    markers
}

fn first_string_field(value: &serde_json::Value, keys: &[&str]) -> Option<String> {
    let obj = value.as_object()?;
    for key in keys {
        if let Some(value) = obj.get(*key).and_then(|v| v.as_str()) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn first_value_field<'a>(
    value: &'a serde_json::Value,
    keys: &[&str],
) -> Option<&'a serde_json::Value> {
    let obj = value.as_object()?;
    keys.iter().find_map(|key| obj.get(*key).filter(|value| !value.is_null()))
}

fn is_http_method(method: &str) -> bool {
    matches!(
        method.trim().to_ascii_uppercase().as_str(),
        "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD" | "OPTIONS"
    )
}

fn split_method_endpoint(raw: &str) -> (Option<String>, String) {
    let mut parts = raw.split_whitespace();
    let Some(first) = parts.next() else {
        return (None, raw.to_string());
    };
    if !is_http_method(first) {
        return (None, raw.to_string());
    }
    let endpoint = parts.next().unwrap_or(raw).to_string();
    (Some(first.to_ascii_uppercase()), endpoint)
}

fn method_is_state_changing(method: Option<&str>) -> bool {
    !matches!(
        method.unwrap_or("GET").trim().to_ascii_uppercase().as_str(),
        "GET" | "HEAD" | "OPTIONS"
    )
}

fn role_from_text(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    for prefix in ["role=", "role:", "as=", "as:"] {
        if let Some(pos) = lower.find(prefix) {
            let value =
                line[pos + prefix.len()..].split_whitespace().next().unwrap_or("").trim_matches(
                    |c: char| c == '"' || c == '\'' || c == '`' || c == ',' || c == ';',
                );
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn shell_like_tokens(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            } else if ch == '\\' {
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            } else {
                current.push(ch);
            }
        } else if matches!(ch, '"' | '\'') {
            quote = Some(ch);
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn attack_agent_candidate_match<'a>(
    vulnerability: &AttackAgentVulnerability,
    candidates: &'a [PentestCandidateRecord],
) -> Option<&'a PentestCandidateRecord> {
    for id in &vulnerability.source_candidate_ids {
        if let Some(candidate) = candidates.iter().find(|candidate| candidate.id == *id) {
            return Some(candidate);
        }
    }
    let vuln_location = first_attack_location(vulnerability);
    candidates.iter().find(|candidate| {
        candidate.vuln_class == vulnerability.vuln_class
            && vuln_location.as_deref().is_some_and(|loc| {
                candidate_location(candidate)
                    .as_deref()
                    .is_some_and(|candidate_loc| same_attack_location(loc, candidate_loc))
            })
    })
}

fn attack_agent_existing_vulnerability_match<'a>(
    vulnerability: &AttackAgentVulnerability,
    existing_vulnerabilities: &'a [VerifiedVulnerabilityRecord],
) -> Option<&'a VerifiedVulnerabilityRecord> {
    for id in &vulnerability.source_candidate_ids {
        if let Some(vuln) = existing_vulnerabilities
            .iter()
            .find(|vuln| vuln.source_candidate_ids.iter().any(|source_id| source_id == id))
        {
            return Some(vuln);
        }
    }
    let vuln_location = first_attack_location(vulnerability);
    existing_vulnerabilities.iter().find(|existing| {
        existing.vuln_class == vulnerability.vuln_class
            && vuln_location.as_deref().is_some_and(|loc| {
                first_verified_vulnerability_location(existing)
                    .as_deref()
                    .is_some_and(|existing_loc| same_attack_location(loc, existing_loc))
            })
    })
}

fn attack_vulnerability_fingerprint(vulnerability: &AttackAgentVulnerability) -> String {
    let location = first_attack_location(vulnerability)
        .map(|location| normalized_attack_location(&location))
        .unwrap_or_else(|| vulnerability.title.clone());
    finding_id_hash("attack-agent", &location, None, &vulnerability.vuln_class, "confirmed")
}

fn first_attack_location(vulnerability: &AttackAgentVulnerability) -> Option<String> {
    vulnerability.affected_components.iter().find_map(first_location_from_value)
}

fn first_location_from_value(component: &serde_json::Value) -> Option<String> {
    let obj = component.as_object()?;
    for key in ["endpoint", "url", "path", "route"] {
        if let Some(value) = obj.get(key).and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
            return Some(value.to_string());
        }
    }
    None
}

fn first_verified_vulnerability_location(
    vulnerability: &VerifiedVulnerabilityRecord,
) -> Option<String> {
    vulnerability.affected_components.iter().find_map(first_location_from_value)
}

fn same_attack_location(left: &str, right: &str) -> bool {
    normalized_attack_location(left) == normalized_attack_location(right)
}

fn normalized_attack_location(raw: &str) -> String {
    let (_, endpoint) = split_method_endpoint(raw.trim());
    let without_origin = strip_url_origin(&endpoint);
    without_origin.trim_end_matches('/').to_ascii_lowercase()
}

fn strip_url_origin(raw: &str) -> &str {
    let Some(scheme_pos) = raw.find("://") else {
        return raw;
    };
    let after_scheme = &raw[scheme_pos + 3..];
    match after_scheme.find('/') {
        Some(path_pos) => &after_scheme[path_pos..],
        None => "/",
    }
}

fn attack_components_or_default(
    vulnerability: &AttackAgentVulnerability,
) -> Vec<serde_json::Value> {
    if vulnerability.affected_components.is_empty() {
        vec![serde_json::json!({
            "source": "unsafe_attack_agent",
            "title": vulnerability.title,
        })]
    } else {
        vulnerability.affected_components.clone()
    }
}

fn confidence_fraction(confidence: u8) -> f64 {
    (f64::from(confidence) / 100.0).clamp(0.0, 1.0)
}

fn canonical_severity(severity: &str) -> String {
    match severity.trim().to_ascii_lowercase().as_str() {
        "critical" => "Critical",
        "high" => "High",
        "medium" => "Medium",
        "low" => "Low",
        "info" | "informational" => "Info",
        _ => "High",
    }
    .to_string()
}

fn attack_agent_contextual_severity(vulnerability: &AttackAgentVulnerability) -> String {
    let text = attack_agent_risk_text(vulnerability);
    if attack_agent_contains_any(
        &text,
        &[
            "dev-only",
            "development-only",
            "local dev",
            "local development",
            "test-only",
            "non-production",
            "not production",
            "never in prod",
            "never in production",
            "dev mail",
            "dev-mail",
            "dev_mail",
            "/api/dev/mail",
        ],
    ) {
        return "Low".to_string();
    }
    if attack_agent_contains_any(
        &text,
        &[
            "event ingestion",
            "alert ingestion",
            "alerts",
            "telemetry",
            "audit event",
            "logging endpoint",
        ],
    ) && !attack_agent_contains_any(
        &text,
        &["leak secret", "password", "token", "account takeover", "admin", "privilege", "tenant"],
    ) {
        return "Low".to_string();
    }
    canonical_severity(&vulnerability.severity)
}

fn attack_agent_risk_score(
    severity: &str,
    confidence: f64,
    vulnerability: &AttackAgentVulnerability,
) -> f64 {
    let base: f64 = match severity {
        "Critical" => 9.2,
        "High" => 7.8,
        "Medium" => 5.5,
        "Low" => 3.0,
        _ => 1.5,
    };
    let text = attack_agent_risk_text(vulnerability);
    let mut score = (base * 0.75) + (confidence.clamp(0.0, 1.0) * 2.5);
    if attack_agent_contains_any(
        &text,
        &[
            "dev-only",
            "development-only",
            "local dev",
            "local development",
            "test-only",
            "non-production",
            "not production",
            "never in prod",
            "never in production",
            "dev mail",
            "dev-mail",
            "dev_mail",
            "/api/dev/mail",
        ],
    ) {
        score -= 2.4;
    }
    if attack_agent_contains_any(
        &text,
        &[
            "event ingestion",
            "alert ingestion",
            "alerts",
            "telemetry",
            "audit event",
            "logging endpoint",
        ],
    ) && !attack_agent_contains_any(
        &text,
        &["leak secret", "password", "token", "account takeover", "admin", "privilege", "tenant"],
    ) {
        score -= 1.8;
    }
    clamp_risk_score(score)
}

fn attack_agent_risk_rationale(
    vulnerability: &AttackAgentVulnerability,
    severity: &str,
    confidence: f64,
    risk_score: f64,
) -> String {
    let text = attack_agent_risk_text(vulnerability);
    let mut factors = vec![format!(
        "unsafe local attack agent reported live behavior with {}% confidence",
        (confidence * 100.0).round() as u8
    )];
    if canonical_severity(&vulnerability.severity) != severity {
        factors.push(format!(
            "severity adjusted from `{}` to `{severity}` after contextual scoring",
            vulnerability.severity
        ));
    }
    if attack_agent_contains_any(
        &text,
        &[
            "dev-only",
            "development-only",
            "local dev",
            "local development",
            "test-only",
            "non-production",
            "not production",
            "never in prod",
            "never in production",
            "dev mail",
            "dev-mail",
            "dev_mail",
            "/api/dev/mail",
        ],
    ) {
        factors.push("development-only exposure lowers production risk".to_string());
    }
    if attack_agent_contains_any(
        &text,
        &[
            "event ingestion",
            "alert ingestion",
            "alerts",
            "telemetry",
            "audit event",
            "logging endpoint",
        ],
    ) {
        factors.push(
            "alerting or telemetry impact is lower than direct data or privilege impact"
                .to_string(),
        );
    }
    format!("{} Risk score {:.1}.", factors.join("; "), risk_score)
}

fn attack_agent_risk_text(vulnerability: &AttackAgentVulnerability) -> String {
    let mut text = format!(
        "{} {} {} {} {} {} {}",
        vulnerability.title,
        vulnerability.vuln_class,
        vulnerability.severity,
        vulnerability.business_impact,
        vulnerability.evidence_summary,
        vulnerability.repro_steps,
        vulnerability.remediation
    )
    .to_ascii_lowercase();
    for component in &vulnerability.affected_components {
        text.push(' ');
        text.push_str(&component.to_string().to_ascii_lowercase());
    }
    text
}

fn attack_agent_contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn safe_id_fragment(raw: &str) -> String {
    raw.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '-' { ch } else { '-' }).collect()
}

fn append_unique_paragraph(left: String, right: String) -> String {
    let left = left.trim();
    let right = right.trim();
    if right.is_empty() || left.contains(right) {
        return left.to_string();
    }
    if left.is_empty() {
        right.to_string()
    } else {
        format!("{left}\n\n{right}")
    }
}

fn union_json_values(
    mut left: Vec<serde_json::Value>,
    right: Vec<serde_json::Value>,
) -> Vec<serde_json::Value> {
    for item in right {
        if !left.iter().any(|existing| existing == &item) {
            left.push(item);
        }
    }
    left
}

fn union_strings(mut left: Vec<String>, right: Vec<String>) -> Vec<String> {
    for item in right {
        if !left.iter().any(|existing| existing == &item) {
            left.push(item);
        }
    }
    left
}

fn build_exploration_scope(
    run_id: &str,
    repo: &str,
    workspace_root: &std::path::Path,
    target_urls: &[String],
    known_leads: Vec<ExplorationKnownLead>,
    prior_memory: Vec<String>,
    soft_cap_usd_micros: i64,
    run_cap_usd_micros: i64,
    research_mode_enabled: bool,
) -> ExplorationScope {
    let mut research_focus = prior_memory;
    if research_mode_enabled {
        research_focus.extend(research_focus_from_known_leads(&known_leads));
    }
    let mut scope = ExplorationScope::new(run_id, format!("expl-{repo}"));
    scope.workspace_root = Some(workspace_root.to_string_lossy().to_string());
    scope.allowed_hosts = target_urls.iter().filter_map(|url| host_from_url(url)).collect();
    scope.target_endpoints = target_urls
        .iter()
        .map(|url| ExplorationEndpoint {
            method: "GET".to_string(),
            url: url.clone(),
            description: Some("launch profile target".to_string()),
        })
        .collect();
    scope.known_leads = known_leads;
    scope.research_mode_enabled = research_mode_enabled;
    scope.research_focus = research_focus;
    if research_mode_enabled {
        scope.max_actions = scope.max_actions.max(40);
    }
    scope.soft_cap_usd_micros = soft_cap_usd_micros;
    scope.run_cap_usd_micros = run_cap_usd_micros;
    scope
}

fn exploration_known_leads_for_repo_with_memory(
    candidates: &[PentestCandidateRecord],
    repo: &str,
    limit: usize,
    research_mode_enabled: bool,
    memory: &[ExplorationMemoryRecord],
) -> Vec<ExplorationKnownLead> {
    let mut candidates = candidates
        .iter()
        .filter(|c| matches!(c.status.as_str(), "Proposed" | "NeedsLiveTest" | "Observed"))
        .filter(|c| candidate_applies_to_repo(c, repo))
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        memory_adjusted_candidate_rank(b, research_mode_enabled, memory)
            .cmp(&memory_adjusted_candidate_rank(a, research_mode_enabled, memory))
            .then_with(|| severity_rank(&b.severity_guess).cmp(&severity_rank(&a.severity_guess)))
            .then_with(|| {
                b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.id.cmp(&b.id))
    });
    candidates.into_iter().take(limit).map(candidate_to_exploration_known_lead).collect()
}

fn candidate_memory_hints_for_repo(
    candidates: &[PentestCandidateRecord],
    repo: &str,
) -> Vec<String> {
    candidates
        .iter()
        .filter(|candidate| candidate_applies_to_repo(candidate, repo))
        .flat_map(|candidate| {
            [
                candidate.vuln_class.clone(),
                candidate.hypothesis.clone(),
                candidate_location(candidate).unwrap_or_default(),
            ]
        })
        .collect()
}

fn memory_adjusted_candidate_rank(
    candidate: &PentestCandidateRecord,
    research_mode_enabled: bool,
    memory: &[ExplorationMemoryRecord],
) -> i16 {
    let mut rank = i16::from(research_candidate_rank(candidate, research_mode_enabled)) * 100;
    rank += i16::from(severity_rank(&candidate.severity_guess)) * 4;
    for entry in memory.iter().filter(|entry| memory_matches_candidate(entry, candidate)) {
        rank += match entry.result.as_str() {
            "confirmed" => 45,
            "inconclusive" => 18,
            "blocked" => -28,
            "rejected" => -45,
            _ => 0,
        };
    }
    rank
}

fn memory_matches_candidate(
    memory: &ExplorationMemoryRecord,
    candidate: &PentestCandidateRecord,
) -> bool {
    if memory.candidate_id.as_deref() == Some(candidate.id.as_str()) {
        return true;
    }
    let haystack = format!(
        "{} {} {} {}",
        candidate.hypothesis,
        candidate.vuln_class,
        candidate.title,
        candidate_location(candidate).unwrap_or_default()
    )
    .to_ascii_lowercase();
    let endpoint = memory.endpoint.as_deref().unwrap_or("");
    [memory.hypothesis.as_str(), endpoint]
        .iter()
        .map(|s| s.split_whitespace().take(8).collect::<Vec<_>>().join(" ").to_ascii_lowercase())
        .any(|needle| needle.len() >= 12 && haystack.contains(&needle))
}

fn research_focus_from_known_leads(leads: &[ExplorationKnownLead]) -> Vec<String> {
    leads
        .iter()
        .filter(|lead| {
            lead.source == "ResearchMode" || research_vuln_class_rank(&lead.vuln_class) > 0
        })
        .take(16)
        .map(|lead| {
            format!(
                "{} {} {} - {}",
                lead.source,
                lead.vuln_class,
                lead.location.as_deref().unwrap_or("unknown location"),
                lead.hypothesis
            )
        })
        .collect()
}

fn research_candidate_rank(candidate: &PentestCandidateRecord, research_mode_enabled: bool) -> u8 {
    if !research_mode_enabled {
        return 0;
    }
    let source_rank = if candidate.source == "ResearchMode" { 3 } else { 0 };
    source_rank.max(research_vuln_class_rank(&candidate.vuln_class))
}

fn research_vuln_class_rank(vuln_class: &str) -> u8 {
    match vuln_class {
        "LIFECYCLE_INVARIANT_BYPASS"
        | "STALE_ACCESS"
        | "REPLAY_OR_TOKEN_REUSE"
        | "ENTITLEMENT_MISMATCH"
        | "INVITE_OR_MEMBERSHIP_TRANSITION"
        | "WEBHOOK_EVENT_CONSISTENCY"
        | "AI_AGENT_INDIRECT_ACTION"
        | "BACKGROUND_JOB_SIDE_EFFECT" => 2,
        _ => 0,
    }
}

fn candidate_applies_to_repo(candidate: &PentestCandidateRecord, repo: &str) -> bool {
    let repos = candidate
        .affected_components
        .iter()
        .filter_map(|component| {
            component.as_object().and_then(|obj| obj.get("repo")).and_then(|value| value.as_str())
        })
        .collect::<Vec<_>>();
    repos.is_empty() || repos.contains(&repo)
}

fn candidate_to_exploration_known_lead(candidate: &PentestCandidateRecord) -> ExplorationKnownLead {
    ExplorationKnownLead {
        id: candidate.id.clone(),
        source: candidate.source.clone(),
        title: candidate.title.clone(),
        vuln_class: candidate.vuln_class.clone(),
        severity: candidate.severity_guess.clone(),
        status: candidate.status.clone(),
        location: candidate_location(candidate),
        hypothesis: candidate.hypothesis.clone(),
    }
}

fn candidate_location(candidate: &PentestCandidateRecord) -> Option<String> {
    for component in &candidate.affected_components {
        let Some(obj) = component.as_object() else {
            continue;
        };
        if let Some(url) = obj
            .get("url")
            .or_else(|| obj.get("matched_at"))
            .or_else(|| obj.get("target"))
            .and_then(|value| value.as_str())
        {
            let method = obj.get("method").and_then(|value| value.as_str());
            return Some(match method {
                Some(method) => format!("{} {}", method.to_ascii_uppercase(), url),
                None => url.to_string(),
            });
        }
        if let Some(path) = obj.get("path").and_then(|value| value.as_str()) {
            let repo = obj.get("repo").and_then(|value| value.as_str());
            let line = obj.get("line").and_then(|value| value.as_i64());
            return Some(match (repo, line) {
                (Some(repo), Some(line)) => format!("{repo}:{path}:{line}"),
                (Some(repo), None) => format!("{repo}:{path}"),
                (None, Some(line)) => format!("{path}:{line}"),
                (None, None) => path.to_string(),
            });
        }
    }
    None
}

fn severity_rank(severity: &str) -> u8 {
    match severity.to_ascii_lowercase().as_str() {
        "critical" => 5,
        "high" => 4,
        "medium" => 3,
        "low" => 2,
        "info" | "informational" => 1,
        _ => 0,
    }
}

fn host_from_url(url: &str) -> Option<String> {
    let after_scheme = url.split_once("://")?.1;
    let host_port = after_scheme.split('/').next().unwrap_or(after_scheme);
    Some(host_port.to_string())
}

#[allow(clippy::too_many_arguments)]
async fn apply_exploration_outcome(
    store: &Store,
    run_id: &str,
    repo: &str,
    task_id: &str,
    prompt_memory: &[String],
    outcome: ExplorationOutcome,
    report: &mut AiExplorationPassReport,
    runtime_name: &str,
    runtime_model: &str,
    started_at_ms: i64,
    traces_dir: &std::path::Path,
    research_mode_enabled: bool,
) -> anyhow::Result<()> {
    let finished_at = now_epoch_ms();
    match outcome {
        ExplorationOutcome::Halted { reason } => match reason {
            ExplorationHaltReason::EscapeSuiteRed { fixture, reason } => {
                tracing::warn!(
                    repo = %repo,
                    fixture = %fixture,
                    reason = %reason,
                    "ai exploration: escape suite RED; halting driver"
                );
                report.halted_escape_suite_red += 1;
            }
            ExplorationHaltReason::BudgetCapAlreadyReached { cap_usd_micros, spent_usd_micros } => {
                tracing::info!(
                    repo = %repo,
                    cap_usd_micros,
                    spent_usd_micros,
                    "ai exploration: run-wide budget cap already reached; skipping repo"
                );
                report.halted_budget_exhausted += 1;
            }
        },
        ExplorationOutcome::Completed {
            findings,
            audit,
            final_message: _final_message,
            turns: _turns,
            spent_usd_micros,
            prompt_version,
            soft_cap_exceeded,
            metrics,
        } => {
            report.explorations_dispatched += 1;
            report.spend_usd_micros += spent_usd_micros;
            if soft_cap_exceeded {
                tracing::info!(
                    repo = %repo,
                    spent_usd_micros,
                    "ai exploration: soft cap exceeded; operator warned, run continues"
                );
            }
            let now_ms = finished_at;
            // Persist the proposed findings first, collecting the ids
            // that survived. The split-cost step below needs the actual
            // success count, not the proposal count.
            let mut successful: Vec<String> = Vec::with_capacity(findings.len());
            for finding in findings {
                match persist_exploration_finding(
                    store,
                    run_id,
                    repo,
                    &finding,
                    &prompt_version,
                    now_ms,
                    research_mode_enabled,
                )
                .await
                {
                    Ok(finding_id) => {
                        report.findings_quarantined += 1;
                        successful.push(finding_id);
                    }
                    Err(err) => {
                        tracing::warn!(
                            repo = %repo,
                            error = %err,
                            "ai exploration: failed to persist finding"
                        );
                    }
                }
            }
            // Split the call cost evenly across emitted findings so the
            // AiTraceViewer's per-finding "Total $..." sums to a real
            // share of the call cost instead of $0. When zero findings
            // survived, keep the cost on the parent row so the run-card
            // roll-up still observes the spend through its time-window
            // join. Token/cache metrics always live on the parent.
            let n_succ = successful.len() as i64;
            let (parent_cost, per_finding_costs): (i64, Vec<i64>) = if n_succ == 0 {
                (spent_usd_micros, Vec::new())
            } else {
                let base = spent_usd_micros / n_succ;
                let leftover = spent_usd_micros - base * n_succ;
                let costs: Vec<i64> =
                    (0..n_succ).map(|i| if i < leftover { base + 1 } else { base }).collect();
                (0, costs)
            };
            let audit_path = if audit.is_empty() {
                None
            } else {
                match write_exploration_audit_jsonl(traces_dir, run_id, task_id, &audit) {
                    Ok(path) => Some(path.to_string_lossy().to_string()),
                    Err(err) => {
                        tracing::warn!(
                            run_id = %run_id,
                            task_id = %task_id,
                            error = %err,
                            "ai exploration: failed to write audit jsonl"
                        );
                        None
                    }
                }
            };
            let mut parent_trace = build_trace_row(
                TaskKind::Exploration,
                None,
                runtime_name,
                runtime_model,
                &prompt_version,
                parent_cost,
                started_at_ms,
                finished_at,
                Some(&metrics),
            );
            parent_trace.conversation_jsonl_path = audit_path;
            if !prompt_memory.is_empty() {
                parent_trace.verifier_blob = Some(
                    serde_json::json!({
                        "kind": "ExplorationPromptContext",
                        "learned_from_prior_runs": prompt_memory,
                    })
                    .to_string(),
                );
            }
            persist_exploration_audit_memory(store, run_id, repo, &audit, now_ms).await;
            persist_trace_row(store, parent_trace).await;
            for (finding_id, cost) in successful.into_iter().zip(per_finding_costs) {
                let per_trace = build_trace_row(
                    TaskKind::Exploration,
                    Some(finding_id),
                    runtime_name,
                    runtime_model,
                    &prompt_version,
                    cost,
                    started_at_ms,
                    finished_at,
                    None,
                );
                persist_trace_row(store, per_trace).await;
            }
        }
    }
    Ok(())
}

/// Persist the exploration audit log as JSONL under
/// `<traces_dir>/<run_id>/<task_id>.jsonl`. One JSON object per line
/// keyed on `action` / `summary`. Returns the absolute path that the
/// caller stamps on `agent_traces.conversation_jsonl_path`.
fn write_exploration_audit_jsonl(
    traces_dir: &std::path::Path,
    run_id: &str,
    task_id: &str,
    audit: &[ExplorationAuditEntry],
) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;

    let run_dir = traces_dir.join(run_id);
    std::fs::create_dir_all(&run_dir)?;
    let path = run_dir.join(format!("{task_id}.jsonl"));
    let mut file =
        std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(&path)?;
    for entry in audit {
        let line = serde_json::to_string(entry).map_err(std::io::Error::other)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
    }
    file.flush()?;
    Ok(path)
}

async fn persist_exploration_audit_memory(
    store: &Store,
    run_id: &str,
    repo: &str,
    audit: &[ExplorationAuditEntry],
    now_ms: i64,
) {
    let Some(project_id) = store.runs().get(run_id).await.ok().flatten().and_then(|r| r.project_id)
    else {
        return;
    };
    for entry in audit.iter().filter(|entry| entry.action != "record_exploration_finding") {
        let input = ExplorationMemoryInput {
            project_id: project_id.clone(),
            repo: repo.to_string(),
            run_id: run_id.to_string(),
            source: "ai_exploration_audit".to_string(),
            hypothesis: entry.summary.clone(),
            endpoint: extract_endpoint_like(&entry.summary),
            role_context: extract_labeled_value(&entry.summary, "role"),
            object_context: extract_labeled_value(&entry.summary, "object"),
            result: "inconclusive".to_string(),
            reason: format!(
                "AI exploration audit action `{}` did not produce a confirmed finding",
                entry.action
            ),
            useful_markers: Vec::new(),
            auth_session_notes: extract_labeled_value(&entry.summary, "auth"),
            follow_up_ideas: vec![
                "resume from this audit observation if related leads recur".to_string()
            ],
            candidate_id: None,
            verification_attempt_id: None,
            trace_id: None,
            created_at: now_ms,
        };
        if let Err(err) = store.exploration_memory().upsert(&input).await {
            tracing::warn!(
                run_id = %run_id,
                repo = %repo,
                error = %err,
                "failed to persist exploration audit memory"
            );
        }
    }
}

fn extract_endpoint_like(raw: &str) -> Option<String> {
    raw.split_whitespace()
        .find(|part| {
            part.starts_with("http://")
                || part.starts_with("https://")
                || part.starts_with('/')
                || matches!(*part, "GET" | "POST" | "PUT" | "PATCH" | "DELETE")
        })
        .map(|s| s.trim_matches(|c: char| c == ',' || c == ';').to_string())
}

fn extract_labeled_value(raw: &str, label: &str) -> Option<String> {
    let prefix = format!("{label}=");
    raw.split_whitespace()
        .find_map(|part| part.strip_prefix(&prefix))
        .map(|s| s.trim_matches(|c: char| c == ',' || c == ';').to_string())
        .filter(|s| !s.is_empty())
}

async fn persist_exploration_finding(
    store: &Store,
    run_id: &str,
    repo: &str,
    finding: &ExplorationFinding,
    prompt_version: &str,
    now_ms: i64,
    research_mode_enabled: bool,
) -> anyhow::Result<String> {
    let line = finding.line.map(i64::from);
    let rule = format!("ai-exploration:{}", finding.cap);
    let id = nyx_agent_core::store::finding_id_hash(repo, &finding.path, line, &finding.cap, &rule);
    let verdict_blob = serde_json::to_string(&serde_json::json!({
        "kind": "AiExploration",
        "rationale": finding.rationale,
        "endpoint": finding.endpoint,
        "suggested_payload_hint": finding.suggested_payload_hint,
        "prompt_version": prompt_version,
        "research_mode": research_mode_enabled,
        "research_mode_provenance": if research_mode_enabled {
            serde_json::json!({
                "mode": "research",
                "source": "ai_exploration",
                "prompt_version": prompt_version,
            })
        } else {
            serde_json::Value::Null
        },
    }))?;
    let rec = FindingRecord {
        id: id.clone(),
        run_id: run_id.to_string(),
        repo: repo.to_string(),
        path: finding.path.clone(),
        line,
        cap: finding.cap.clone(),
        rule,
        // Severity defaults to High pending verifier promotion; the
        // verifier can downgrade or close the row on `NotConfirmed`.
        severity: "High".to_string(),
        status: "Quarantine".to_string(),
        finding_origin: FindingOrigin::AiExploration.as_str().to_string(),
        first_seen: now_ms,
        last_seen: now_ms,
        superseded_by: None,
        triage_state: "Open".to_string(),
        triage_assigned_to: None,
        verdict_blob: Some(verdict_blob),
        repro_path: None,
        attack_provenance: Some(AttackProvenance::AiExploration.as_str().to_string()),
        prompt_version: Some(prompt_version.to_string()),
        chain_id: None,
        spec_id: None,
    };
    store.findings().upsert(&rec).await?;
    if let Some(project_id) = store.runs().get(run_id).await?.and_then(|run| run.project_id) {
        let memory = ExplorationMemoryInput {
            project_id,
            repo: repo.to_string(),
            run_id: run_id.to_string(),
            source: "ai_exploration_finding".to_string(),
            hypothesis: finding.rationale.clone(),
            endpoint: finding.endpoint.clone(),
            role_context: None,
            object_context: Some(finding.path.clone()),
            result: "inconclusive".to_string(),
            reason: "AI exploration proposed a finding that still needs deterministic confirmation"
                .to_string(),
            useful_markers: finding.suggested_payload_hint.clone().into_iter().collect(),
            auth_session_notes: None,
            follow_up_ideas: vec![
                "derive and run a deterministic live verification plan".to_string()
            ],
            candidate_id: None,
            verification_attempt_id: None,
            trace_id: None,
            created_at: now_ms,
        };
        store.exploration_memory().upsert(&memory).await?;
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use nyx_agent_core::run::{CrossRepoCallgraphStub, RepoBundle};
    use nyx_agent_types::verify::{Oracle, VerifyRun, VerifyVerdict};

    use super::*;

    fn fake_verify_run(payload: &[u8], oracle_fired: bool, exit_code: i32) -> VerifyRun {
        VerifyRun {
            payload: payload.to_vec(),
            oracle_fired,
            exit_code,
            timed_out: false,
            stdout: vec![],
            stderr: vec![],
            duration_ms: 1,
        }
    }

    #[test]
    fn live_test_plan_normalises_relative_url_and_requires_oracle() {
        let targets = vec!["http://localhost:8787".to_string()];
        let raw = r#"{"method":"post","path":"/api/search","json":{"q":"nyx-agent-probe"},"status_range":"2xx","body_contains":"nyx-agent-probe"}"#;
        let plan = normalise_live_test_plan(raw, &targets).expect("valid").expect("plan");
        assert_eq!(plan["method"], "POST");
        assert_eq!(plan["url"], "http://localhost:8787/api/search");

        let raw_without_oracle = r#"{"method":"GET","path":"/"}"#;
        let err = normalise_live_test_plan(raw_without_oracle, &targets).expect_err("oracle");
        assert!(err.contains("no explicit oracle"));
    }

    #[test]
    fn live_test_plan_no_plan_policy_escalates_planner_gaps_only() {
        for code in [
            NoPlanReasonCode::BadEndpoint,
            NoPlanReasonCode::WeakOracle,
            NoPlanReasonCode::NoExecutablePlan,
            NoPlanReasonCode::UnsupportedClass,
            NoPlanReasonCode::RouteNotInferred,
            NoPlanReasonCode::Other,
        ] {
            let reason = NoPlanReason::new(code, "planner needs source-aware help");
            assert!(should_try_ai_live_plan_after_no_plan(&reason));
        }

        for code in [
            NoPlanReasonCode::SetupMissing,
            NoPlanReasonCode::StateChangingBlocked,
            NoPlanReasonCode::TargetOutOfScope,
            NoPlanReasonCode::DependencyReviewOnly,
            NoPlanReasonCode::UnsafeProbe,
            NoPlanReasonCode::BrowserDisabled,
        ] {
            let reason = NoPlanReason::new(code, "hard live verification blocker");
            assert!(!should_try_ai_live_plan_after_no_plan(&reason));
        }
    }

    #[test]
    fn live_test_plan_rejects_urls_outside_target_base() {
        let targets = vec!["http://localhost:8787".to_string()];
        let raw =
            r#"{"method":"GET","url":"http://localhost:8787.evil.test/","expect_status":200}"#;
        let err = normalise_live_test_plan(raw, &targets).expect_err("outside target");
        assert!(err.contains("outside supplied target"));
    }

    #[test]
    fn verifier_blob_carries_spec_id_and_payload_hashes() {
        let result = VerifyResult {
            finding_id: "f-1".to_string(),
            verdict: VerifyVerdict::Confirmed,
            oracle: Oracle::OutputContains { marker: "x".to_string() },
            vuln_run: fake_verify_run(b"VULN", true, 7),
            benign_run: fake_verify_run(b"BENIGN", false, 0),
            attack_provenance: AttackProvenance::Curated,
            replay_stable: None,
            error_message: None,
        };
        let blob = build_verifier_blob(Some("spec-abc"), Some(&result)).expect("populated");
        let v: serde_json::Value = serde_json::from_str(&blob).expect("json");
        assert_eq!(v["spec_id"], "spec-abc");
        assert_eq!(v["vuln_exit_code"], 7);
        assert_eq!(v["benign_exit_code"], 0);
        // sha256("VULN") = ad9a82ba23ddccd8...
        let vuln = v["vuln_payload_sha256"].as_str().expect("hex string");
        assert_eq!(vuln.len(), 64, "sha256 hex is 64 chars");
        let benign = v["benign_payload_sha256"].as_str().expect("hex string");
        assert_eq!(benign.len(), 64);
        assert_ne!(vuln, benign, "distinct payloads hash distinctly");
    }

    #[test]
    fn verifier_blob_is_none_when_no_inputs() {
        assert!(build_verifier_blob(None, None).is_none());
    }

    #[test]
    fn verifier_blob_with_only_spec_id() {
        let blob = build_verifier_blob(Some("spec-1"), None).expect("populated");
        let v: serde_json::Value = serde_json::from_str(&blob).expect("json");
        assert_eq!(v["spec_id"], "spec-1");
        assert!(v.get("vuln_payload_sha256").is_none());
    }

    fn diag_unsupported(path: &str, line: u32, cap: &str, rule: &str) -> Diag {
        serde_json::from_value(serde_json::json!({
            "path": path,
            "line": line,
            "severity": "High",
            "id": rule,
            "category": cap,
            "evidence": {
                "unsupported": "NoPayloadsForCap",
                "sink": {"callee": "cursor.execute", "args": ["q"]}
            }
        }))
        .unwrap()
    }

    fn diag_supported(path: &str, line: u32, cap: &str, rule: &str) -> Diag {
        serde_json::from_value(serde_json::json!({
            "path": path,
            "line": line,
            "severity": "Low",
            "id": rule,
            "category": cap,
        }))
        .unwrap()
    }

    fn make_bundle(run_id: &str, repo: &str, diags: Vec<Diag>) -> RunBundle<Diag> {
        RunBundle {
            run_id: run_id.to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: vec![RepoBundle {
                repo: repo.to_string(),
                outcome: RepoOutcome::Success(diags),
                started_at_ms: 0,
                finished_at_ms: 0,
                elapsed_ms: 0,
            }],
            callgraph: CrossRepoCallgraphStub::default(),
        }
    }

    fn handle(name: &str, path: &std::path::Path) -> WorkspaceHandle {
        WorkspaceHandle::for_local_path_test(name, path.to_path_buf())
    }

    #[test]
    fn infer_lang_handles_common_extensions() {
        assert_eq!(infer_lang("src/foo.py"), "python");
        assert_eq!(infer_lang("src/bar.ts"), "typescript");
        assert_eq!(infer_lang("Main.JAVA"), "java");
        assert_eq!(infer_lang("noext"), "unknown");
    }

    #[test]
    fn infer_lang_ignores_directory_dots_when_basename_has_no_extension() {
        assert_eq!(infer_lang("path.with.dots/bin/foo"), "unknown");
        assert_eq!(infer_lang("path.with.dots/bin/foo.py"), "python");
    }

    #[test]
    fn infer_lang_for_file_reads_shebang_when_extensionless() {
        let tmp = tempfile::tempdir().unwrap();
        let bin_dir = tmp.path().join("bin");
        std::fs::create_dir(&bin_dir).unwrap();
        std::fs::write(
            bin_dir.join("ingest"),
            "#!/usr/bin/env python3\nimport sys\nprint(sys.argv)\n",
        )
        .unwrap();

        assert_eq!(infer_lang_for_file(tmp.path(), "bin/ingest"), "python");
    }

    #[test]
    fn infer_lang_for_file_recognises_common_interpreters() {
        let tmp = tempfile::tempdir().unwrap();
        for (name, line, expected) in [
            ("py3", "#!/usr/bin/python3", "python"),
            ("node", "#!/usr/bin/env node", "javascript"),
            ("deno", "#!/usr/bin/env deno", "typescript"),
            ("rb", "#!/usr/bin/env ruby", "ruby"),
            ("php", "#!/usr/bin/env php", "php"),
            ("pl", "#!/usr/bin/perl -w", "perl"),
        ] {
            std::fs::write(tmp.path().join(name), format!("{line}\n# rest")).unwrap();
            assert_eq!(infer_lang_for_file(tmp.path(), name), expected, "shebang `{line}`");
        }
    }

    #[test]
    fn infer_lang_for_file_does_not_overwrite_a_recognised_extension() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("foo.py"), "#!/usr/bin/env ruby\nputs 'x'\n").unwrap();
        // The .py extension wins; we do not re-read the file when the
        // extension table already produces a non-`unknown` answer.
        assert_eq!(infer_lang_for_file(tmp.path(), "foo.py"), "python");
    }

    #[test]
    fn infer_lang_for_file_returns_unknown_when_shebang_interp_is_unrecognised() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("oddball"), "#!/usr/bin/env mystery\n").unwrap();
        assert_eq!(infer_lang_for_file(tmp.path(), "oddball"), "unknown");
    }

    #[test]
    fn infer_lang_for_file_returns_unknown_when_file_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(infer_lang_for_file(tmp.path(), "ghost"), "unknown");
    }

    #[test]
    fn build_inputs_filters_to_unsupported_only() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.py"), "x = 1\ny = 2\nz = 3\n").unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-1".to_string(), handle("repo-1", tmp.path()));

        let bundle = make_bundle(
            "run-X",
            "repo-1",
            vec![
                diag_unsupported("a.py", 2, "SQL_QUERY", "rule-a"),
                diag_supported("a.py", 3, "SQL_QUERY", "rule-b"),
            ],
        );

        let inputs = build_inputs(&bundle, &workspaces);
        assert_eq!(inputs.len(), 1, "only the unsupported diag should fan out");
        assert_eq!(inputs[0].cap, "SQL_QUERY");
        assert_eq!(inputs[0].lang, "python");
        assert_eq!(inputs[0].run_id, "run-X");
        assert_eq!(inputs[0].sink_ctx.callee, "cursor.execute");
    }

    #[test]
    fn build_inputs_skips_failed_repos() {
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let bundle = RunBundle::<Diag> {
            run_id: "r".to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: vec![RepoBundle {
                repo: "broken".to_string(),
                outcome: RepoOutcome::Failed("scanner crashed".to_string()),
                started_at_ms: 0,
                finished_at_ms: 0,
                elapsed_ms: 0,
            }],
            callgraph: CrossRepoCallgraphStub::default(),
        };
        let inputs = build_inputs(&bundle, &workspaces);
        assert!(inputs.is_empty());
    }

    #[tokio::test]
    async fn budget_store_tracker_creates_row_lazily_and_records_spend() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        // Seed a run row; budgets FK requires it.
        let run = nyx_agent_core::store::RunRecord {
            id: "run-bt".to_string(),
            project_id: None,
            kind: "Scan".to_string(),
            started_at: 0,
            finished_at: None,
            status: "Running".to_string(),
            triggered_by: "Manual".to_string(),
            git_ref: None,
            parent_run_id: None,
            wall_clock_ms: None,
            total_ai_spend_usd_micros: 0,
        };
        store.runs().insert(&run).await.unwrap();

        let tracker = BudgetStoreTracker::new(store.clone(), 1_000_000);
        let cap = tracker.cap("run-bt", BudgetKind::OneShot).await.unwrap();
        assert_eq!(cap, Some(1_000_000));

        let after_a = tracker.add_spend("run-bt", BudgetKind::OneShot, 2_500).await.unwrap();
        let after_b = tracker.add_spend("run-bt", BudgetKind::OneShot, 1_000).await.unwrap();
        assert_eq!(after_a, 2_500);
        assert_eq!(after_b, 3_500);

        // Row was persisted via the public store API.
        let row = store.budgets().get("run-bt", "OneShot").await.unwrap().expect("row");
        assert_eq!(row.cap_usd_micros, 1_000_000);
        assert_eq!(row.spent_usd_micros, 3_500);
    }

    #[tokio::test]
    async fn budget_tracker_concurrent_ensure_row_preserves_spend() {
        // Regression: concurrent ensure_row calls used to clobber
        // spent_usd_micros to 0 via upsert's DO UPDATE clause; the
        // INSERT OR IGNORE path keeps spend from prior add_spend calls
        // intact even when a peer task fans in just after.
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        let run = nyx_agent_core::store::RunRecord {
            id: "run-cc".to_string(),
            project_id: None,
            kind: "Scan".to_string(),
            started_at: 0,
            finished_at: None,
            status: "Running".to_string(),
            triggered_by: "Manual".to_string(),
            git_ref: None,
            parent_run_id: None,
            wall_clock_ms: None,
            total_ai_spend_usd_micros: 0,
        };
        store.runs().insert(&run).await.unwrap();
        let tracker = Arc::new(BudgetStoreTracker::new(store.clone(), 1_000_000));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let t = Arc::clone(&tracker);
            handles.push(tokio::spawn(async move {
                let _ = t.cap("run-cc", BudgetKind::OneShot).await.unwrap();
                t.add_spend("run-cc", BudgetKind::OneShot, 1_000).await.unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let row = store.budgets().get("run-cc", "OneShot").await.unwrap().expect("row");
        assert_eq!(row.spent_usd_micros, 8_000, "every concurrent add_spend must persist");
        assert_eq!(row.cap_usd_micros, 1_000_000);
    }

    #[tokio::test]
    async fn budget_buckets_isolate_cap_and_spend_per_pass() {
        // Regression: exploration's tight `exploration_run_cap` used to
        // seed the shared `(run_id, AgentLoop)` row first (INSERT OR
        // IGNORE), pinning the attack-agent pass to that low cap and
        // halting it with BudgetCapReached even though the operator left
        // `default_run_budget` uncapped. A per-pass bucket keeps the two
        // budgets on independent rows.
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        let run = nyx_agent_core::store::RunRecord {
            id: "run-bucket".to_string(),
            project_id: None,
            kind: "Scan".to_string(),
            started_at: 0,
            finished_at: None,
            status: "Running".to_string(),
            triggered_by: "Manual".to_string(),
            git_ref: None,
            parent_run_id: None,
            wall_clock_ms: None,
            total_ai_spend_usd_micros: 0,
        };
        store.runs().insert(&run).await.unwrap();

        // Exploration runs first with its own bucket and a tight cap.
        let exploration = BudgetStoreTracker::new(store.clone(), 10_000_000)
            .with_bucket(EXPLORATION_BUDGET_BUCKET);
        let cap = exploration.cap("run-bucket", BudgetKind::AgentLoop).await.unwrap();
        assert_eq!(cap, Some(10_000_000));
        exploration.add_spend("run-bucket", BudgetKind::AgentLoop, 9_999_999).await.unwrap();

        // Attack agent runs later on the shared (unbucketed) row. It must
        // see its own uncapped ceiling and zero inherited spend.
        let attack = BudgetStoreTracker::new(store.clone(), i64::MAX);
        let attack_cap = attack.cap("run-bucket", BudgetKind::AgentLoop).await.unwrap();
        assert_eq!(attack_cap, Some(i64::MAX), "attack must not inherit exploration's cap");
        let attack_spend = attack.current_spend("run-bucket", BudgetKind::AgentLoop).await.unwrap();
        assert_eq!(attack_spend, 0, "attack must not inherit exploration's spend");

        // Distinct kind strings landed in storage.
        let expl_row = store
            .budgets()
            .get("run-bucket", "AgentLoop@exploration")
            .await
            .unwrap()
            .expect("exploration row");
        assert_eq!(expl_row.spent_usd_micros, 9_999_999);
        let attack_row =
            store.budgets().get("run-bucket", "AgentLoop").await.unwrap().expect("attack row");
        assert_eq!(attack_row.cap_usd_micros, i64::MAX);
        assert_eq!(attack_row.spent_usd_micros, 0);
    }

    #[tokio::test]
    async fn run_pass_is_noop_when_runtime_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        let secrets = SecretStore::memory();
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let bundle = RunBundle::<Diag> {
            run_id: "r".to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: Vec::new(),
            callgraph: CrossRepoCallgraphStub::default(),
        };
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let cfg = AiConfig::default();
        assert!(matches!(cfg.runtime, ConfigAiRuntime::None));
        let report = run_payload_synthesis_pass(&cfg, &store, &secrets, &bundle, &workspaces, tx)
            .await
            .unwrap();
        assert_eq!(report, PayloadSynthesisPassReport::default());
    }

    #[tokio::test]
    async fn run_pass_is_noop_when_anthropic_but_no_key() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        let secrets = SecretStore::memory();
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let bundle = RunBundle::<Diag> {
            run_id: "r".to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: Vec::new(),
            callgraph: CrossRepoCallgraphStub::default(),
        };
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let cfg = AiConfig { runtime: ConfigAiRuntime::Anthropic, ..AiConfig::default() };
        let report = run_payload_synthesis_pass(&cfg, &store, &secrets, &bundle, &workspaces, tx)
            .await
            .unwrap();
        assert_eq!(report, PayloadSynthesisPassReport::default());
    }

    #[tokio::test]
    async fn deterministic_live_plan_synthesis_plans_reclassified_nyx_candidate_without_ai() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        store.projects().create("project-live", "Live", None, None, None, 1).await.unwrap();
        store
            .runs()
            .insert(&nyx_agent_core::store::RunRecord {
                id: "run-live".to_string(),
                project_id: Some("project-live".to_string()),
                kind: "Scan".to_string(),
                started_at: 1,
                finished_at: None,
                status: "Running".to_string(),
                triggered_by: "Manual".to_string(),
                git_ref: None,
                parent_run_id: None,
                wall_clock_ms: None,
                total_ai_spend_usd_micros: 0,
            })
            .await
            .unwrap();
        store
            .pentest_candidates()
            .insert(&PentestCandidateRecord {
                id: "pc-nyx-open-redirect".to_string(),
                run_id: "run-live".to_string(),
                project_id: "project-live".to_string(),
                source: "NyxSignal".to_string(),
                source_ids: vec!["sig-nyx-open-redirect".to_string()],
                title: "Potential open redirect: /login/callback via `next`".to_string(),
                vuln_class: "OPEN_REDIRECT".to_string(),
                severity_guess: "Medium".to_string(),
                affected_components: vec![serde_json::json!({
                    "kind": "nyx_signal",
                    "path": "src/auth/callback.ts",
                    "route": "/login/callback",
                    "url_path": "/login/callback",
                    "method": "GET",
                    "param": "next",
                    "sink": "redirect",
                    "nyx_signal_id": "sig-nyx-open-redirect",
                    "cap": "Security",
                    "rule": "taint-unsanitised-flow",
                })],
                hypothesis: "Nyx Agent reclassified the generic Nyx signal as OPEN_REDIRECT."
                    .to_string(),
                test_plan:
                    "Derive a live HTTP/browser test from the affected route before confirmation."
                        .to_string(),
                status: "NeedsLiveTest".to_string(),
                rejection_reason: None,
                confidence: 0.7,
                trace_id: None,
                created_at: 2,
                updated_at: 2,
            })
            .await
            .unwrap();
        let bundle = RunBundle::<Diag> {
            run_id: "run-live".to_string(),
            project_id: "project-live".to_string(),
            started_at_ms: 1,
            finished_at_ms: 2,
            wall_clock_ms: 1,
            per_repo: Vec::new(),
            callgraph: CrossRepoCallgraphStub::default(),
        };
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let secrets = SecretStore::memory();
        let targets = vec!["http://localhost:3000".to_string()];
        let auth = Vec::new();
        let (tx, _rx) = tokio::sync::broadcast::channel(4);

        let report = run_live_test_plan_synthesis_pass(
            &AiConfig::default(),
            &store,
            &secrets,
            &bundle,
            &workspaces,
            &targets,
            None,
            &auth,
            false,
            false,
            None,
            None,
            tx,
        )
        .await
        .unwrap();

        assert_eq!(report.candidates_seen, 1);
        assert_eq!(report.planned, 1);
        assert_eq!(report.no_plan, 0);
        assert_eq!(report.attempts, 0, "deterministic plan should avoid AI");
        let updated = store.pentest_candidates().list_by_run("run-live").await.unwrap();
        assert_eq!(updated[0].status, "Proposed");
        let plan = normalise_live_test_plan(&updated[0].test_plan, &targets).unwrap().unwrap();
        assert_eq!(plan["kind"], "single_http");
        assert!(plan["url"].as_str().unwrap().contains("next="));
    }

    // -------- spec-derivation pass coverage --------

    fn diag_spec_failed(
        path: &str,
        line: u32,
        cap: &str,
        rule: &str,
        flow_files: &[(&str, u32)],
    ) -> Diag {
        let mut steps: Vec<serde_json::Value> = Vec::new();
        for (i, (f, l)) in flow_files.iter().enumerate() {
            steps.push(serde_json::json!({
                "step": i + 1,
                "kind": if i == 0 { "source" } else { "call" },
                "file": f,
                "line": l,
            }));
        }
        // Final sink step.
        steps.push(serde_json::json!({
            "step": flow_files.len() + 1,
            "kind": "sink",
            "file": path,
            "line": line,
        }));
        let mut diag: Diag = serde_json::from_value(serde_json::json!({
            "path": path,
            "line": line,
            "severity": "Medium",
            "id": rule,
            "category": cap,
            "evidence": {
                "inconclusive": "SpecDerivationFailed",
                "sink": {"callee": "cursor.execute", "args": ["q"]},
                "flow_steps": steps,
            }
        }))
        .unwrap();
        diag.lift_flow_steps();
        diag
    }

    #[test]
    fn build_spec_inputs_filters_and_attaches_excerpts() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("sink.py"),
            "1\n2\n3\n4\ncursor.execute('SELECT * FROM u WHERE n=' + q)\n6\n7\n",
        )
        .unwrap();
        std::fs::create_dir_all(tmp.path().join("framework")).unwrap();
        std::fs::write(tmp.path().join("framework/orm.py"), "a\nb\nc\nd\n").unwrap();
        std::fs::write(tmp.path().join("router.py"), "r1\nr2\nr3\nr4\n").unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-1".to_string(), handle("repo-1", tmp.path()));

        let diag = diag_spec_failed(
            "sink.py",
            5,
            "SQL_QUERY",
            "rule-spec",
            &[("router.py", 2), ("framework/orm.py", 3)],
        );
        let skipped = diag_supported("sink.py", 6, "SQL_QUERY", "rule-ok");
        let bundle = make_bundle("run-S", "repo-1", vec![diag, skipped]);

        let inputs = build_spec_inputs(&bundle, &workspaces);
        assert_eq!(inputs.len(), 1, "only the SpecDerivationFailed diag fans out");
        let input = &inputs[0];
        assert_eq!(input.cap, "SQL_QUERY");
        assert_eq!(input.lang, "python");
        assert_eq!(input.callee, "cursor.execute");
        // sink first, then call_site (router.py), then framework (orm.py).
        let kinds: Vec<&str> = input.excerpts.iter().map(|e| e.kind.as_str()).collect();
        assert_eq!(kinds, vec!["sink", "call_site", "framework"]);
        assert_eq!(input.excerpts[0].path, "sink.py");
        assert!(input.excerpts[0].body.contains("cursor.execute"));
        assert_eq!(input.excerpts[1].path, "router.py");
        assert_eq!(input.excerpts[2].path, "framework/orm.py");
        assert!(input.excerpts.len() <= SPEC_DERIVATION_MAX_EXCERPTS);
    }

    #[test]
    fn build_spec_inputs_skips_failed_repos() {
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let bundle = RunBundle::<Diag> {
            run_id: "r".to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: vec![RepoBundle {
                repo: "broken".to_string(),
                outcome: RepoOutcome::Failed("scanner crashed".to_string()),
                started_at_ms: 0,
                finished_at_ms: 0,
                elapsed_ms: 0,
            }],
            callgraph: CrossRepoCallgraphStub::default(),
        };
        assert!(build_spec_inputs(&bundle, &workspaces).is_empty());
    }

    #[tokio::test]
    async fn spec_pass_is_noop_when_runtime_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        let secrets = SecretStore::memory();
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let bundle = RunBundle::<Diag> {
            run_id: "r".to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: Vec::new(),
            callgraph: CrossRepoCallgraphStub::default(),
        };
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let cfg = AiConfig::default();
        let report = run_spec_derivation_pass(&cfg, &store, &secrets, &bundle, &workspaces, tx)
            .await
            .unwrap();
        assert_eq!(report, SpecDerivationPassReport::default());
    }

    #[tokio::test]
    async fn spec_pass_is_noop_when_anthropic_but_no_key() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        let secrets = SecretStore::memory();
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let bundle = RunBundle::<Diag> {
            run_id: "r".to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: Vec::new(),
            callgraph: CrossRepoCallgraphStub::default(),
        };
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let cfg = AiConfig { runtime: ConfigAiRuntime::Anthropic, ..AiConfig::default() };
        let report = run_spec_derivation_pass(&cfg, &store, &secrets, &bundle, &workspaces, tx)
            .await
            .unwrap();
        assert_eq!(report, SpecDerivationPassReport::default());
    }

    fn seed_run(id: &str) -> nyx_agent_core::store::RunRecord {
        nyx_agent_core::store::RunRecord {
            id: id.to_string(),
            project_id: None,
            kind: "Scan".to_string(),
            started_at: 0,
            finished_at: None,
            status: "Running".to_string(),
            triggered_by: "Manual".to_string(),
            git_ref: None,
            parent_run_id: None,
            wall_clock_ms: None,
            total_ai_spend_usd_micros: 0,
        }
    }

    fn seed_repo(name: &str) -> nyx_agent_core::store::RepoRecord {
        nyx_agent_core::store::RepoRecord {
            id: format!("repo-default-{name}"),
            name: name.to_string(),
            project_id: nyx_agent_core::store::DEFAULT_PROJECT_ID.to_string(),
            source_kind: "local".to_string(),
            source_url_or_path: format!("/tmp/{name}"),
            branch: Some("main".to_string()),
            auth_ref: None,
            i_own_this: true,
            last_scan_run_id: None,
            last_scan_finished_at: None,
            created_at: 1_000,
            updated_at: 1_000,
        }
    }

    fn seed_finding(
        run_id: &str,
        repo: &str,
        path: &str,
        rule: &str,
    ) -> nyx_agent_core::store::FindingRecord {
        let id = nyx_agent_core::store::finding_id_hash(repo, path, Some(10), "SQL_QUERY", rule);
        nyx_agent_core::store::FindingRecord {
            id,
            run_id: run_id.to_string(),
            repo: repo.to_string(),
            path: path.to_string(),
            line: Some(10),
            cap: "SQL_QUERY".to_string(),
            rule: rule.to_string(),
            severity: "High".to_string(),
            status: "Open".to_string(),
            finding_origin: "Static".to_string(),
            first_seen: 3_000,
            last_seen: 3_000,
            superseded_by: None,
            triage_state: "Open".to_string(),
            triage_assigned_to: None,
            verdict_blob: None,
            repro_path: None,
            attack_provenance: None,
            prompt_version: None,
            chain_id: None,
            spec_id: None,
        }
    }

    #[tokio::test]
    async fn apply_spec_outcome_persists_record_and_stamps_finding() {
        // Acceptance: a finding whose strategies all failed in nyx now
        // produces a usable spec, which the store materialises so the
        // verifier can consume it.
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-S")).await.unwrap();
        store.runs().insert(&seed_run("run-S")).await.unwrap();
        let finding = seed_finding("run-S", "repo-S", "src/sink.py", "rule-spec");
        let fid = finding.id.clone();
        store.findings().upsert(&finding).await.unwrap();

        let body = serde_json::json!({
            "schema_version": 1,
            "cap": "SQL_QUERY",
            "lang": "python",
            "entry": "app.handlers:run_query",
            "invoke": "db.execute('SELECT * FROM users WHERE n=' + @PAYLOAD)",
            "payload_arg": 0,
            "oracle": "row count > 0",
        })
        .to_string();
        let (spec, canonical) = nyx_agent_nyx::HarnessSpec::from_json(&body).unwrap();
        let outcome = SpecDerivationOutcome::Synthesised {
            finding_id: fid.clone(),
            cap: "SQL_QUERY".to_string(),
            lang: "python".to_string(),
            spec: Box::new(spec),
            spec_blob: canonical,
            prompt_version: nyx_agent_types::spec::SPEC_DERIVATION_PROMPT_VERSION.to_string(),
            spent_usd_micros: 3_500,
            attempts: 1,
            metrics: AgentTraceMetrics::default(),
        };
        let mut report = SpecDerivationPassReport::default();
        apply_spec_outcome(&store, outcome, &mut report, "test-runtime", "test-model", 0)
            .await
            .unwrap();
        assert_eq!(report.synthesised, 1);
        assert_eq!(report.spend_usd_micros, 3_500);

        let updated = store.findings().get(&fid).await.unwrap().expect("finding");
        assert_eq!(updated.attack_provenance.as_deref(), Some("LlmSynthesised"));
        assert_eq!(updated.prompt_version.as_deref(), Some("phase15.spec_derivation.v1"));
        // Spec row exists and round-trips through the vendored schema.
        let specs = store.harness_specs().list_by_cap("SQL_QUERY").await.unwrap();
        assert_eq!(specs.len(), 1);
        let (parsed, _) = nyx_agent_nyx::HarnessSpec::from_json(&specs[0].spec_blob).unwrap();
        parsed.validate().expect("vendored schema accepts persisted blob");
        assert_eq!(specs[0].attack_provenance.as_deref(), Some("LlmSynthesised"));
    }

    #[tokio::test]
    async fn apply_spec_outcome_quarantines_on_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-Q")).await.unwrap();
        store.runs().insert(&seed_run("run-Q")).await.unwrap();
        let finding = seed_finding("run-Q", "repo-Q", "src/sink.py", "rule-bad");
        let fid = finding.id.clone();
        store.findings().upsert(&finding).await.unwrap();

        let outcome = SpecDerivationOutcome::Quarantined {
            finding_id: fid.clone(),
            reason: "spec derivation failed twice (attempt 1: ...; attempt 2: ...)".to_string(),
            spent_usd_micros: 1_200,
            attempts: 2,
            metrics: AgentTraceMetrics::default(),
        };
        let mut report = SpecDerivationPassReport::default();
        apply_spec_outcome(&store, outcome, &mut report, "test-runtime", "test-model", 0)
            .await
            .unwrap();
        assert_eq!(report.quarantined, 1);
        let row = store.findings().get(&fid).await.unwrap().expect("finding");
        assert_eq!(row.status, "Quarantine");
        let blob = row.verdict_blob.unwrap();
        assert!(blob.contains("SpecDerivation"), "blob: {blob}");
        assert!(blob.contains("failed twice"));
    }

    #[tokio::test]
    async fn spec_derivation_end_to_end_through_build_run_apply() {
        // Acceptance: a real Diag carrying Inconclusive(SpecDerivationFailed)
        // travels build_spec_inputs -> run_spec_derivation (scripted runtime)
        // -> apply_spec_outcome, lands a `harness_specs` row, and stamps
        // the parent `findings.spec_id` back-link. Pins the seam the per-half
        // unit tests already cover in isolation.
        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-E2E")).await.unwrap();
        store.runs().insert(&seed_run("run-E2E")).await.unwrap();
        let seed = seed_finding("run-E2E", "repo-E2E", "sink.py", "rule-e2e-spec");
        let fid = seed.id.clone();
        store.findings().upsert(&seed).await.unwrap();
        assert!(seed.spec_id.is_none(), "seed finding starts with no spec back-link");

        // Workspace lays out a sink whose `line: 10` matches seed_finding's
        // BLAKE3-keyed line, plus one flow-step file the excerpt collector
        // attaches as `call_site`.
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("sink.py"),
            "import sqlite3\ndb = sqlite3.connect(':memory:')\n\
             # padding\n# padding\n# padding\n# padding\n# padding\n# padding\n# padding\n\
             cursor.execute('SELECT * FROM users WHERE n=' + q)\n",
        )
        .unwrap();
        std::fs::write(workspace.path().join("router.py"), "def route(q):\n    handler(q)\n")
            .unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert(
            "repo-E2E".to_string(),
            WorkspaceHandle::for_local_path_test("repo-E2E", workspace.path().to_path_buf()),
        );

        let diag =
            diag_spec_failed("sink.py", 10, "SQL_QUERY", "rule-e2e-spec", &[("router.py", 2)]);
        let bundle = make_bundle("run-E2E", "repo-E2E", vec![diag]);

        let inputs = build_spec_inputs(&bundle, &workspaces);
        assert_eq!(inputs.len(), 1, "the SpecDerivationFailed diag must fan out");
        assert_eq!(inputs[0].finding_id, fid, "input finding_id must match seeded row");
        assert_eq!(inputs[0].cap, "SQL_QUERY");
        assert_eq!(inputs[0].lang, "python");

        let body = serde_json::json!({
            "schema_version": 1,
            "cap": "SQL_QUERY",
            "lang": "python",
            "entry": "app.handlers:run_query",
            "setup": ["import sqlite3", "db = sqlite3.connect(':memory:')"],
            "invoke": "db.execute('SELECT * FROM users WHERE n=' + @PAYLOAD)",
            "payload_arg": 0,
            "oracle": "row count > 0",
            "teardown": ["db.close()"],
        })
        .to_string();
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-E2E", BudgetKind::OneShot, 5_000_000);
        let runtime = ScriptedNovelRuntime::new(vec![Ok(body)], 4_200, tracker.clone());

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let outcome = nyx_agent_ai::run_spec_derivation(&runtime, &inputs[0], tx, 5_000_000)
            .await
            .expect("scripted runtime produces a synthesised outcome");

        let mut report = SpecDerivationPassReport::default();
        apply_spec_outcome(&store, outcome, &mut report, "scripted-novel", "scripted-model", 0)
            .await
            .unwrap();
        assert_eq!(report.synthesised, 1);
        assert_eq!(report.quarantined, 0);
        assert_eq!(report.failed, 0);

        let specs = store.harness_specs().list_by_cap("SQL_QUERY").await.unwrap();
        assert_eq!(specs.len(), 1, "exactly one harness_specs row persisted");
        let spec_row = &specs[0];
        assert_eq!(spec_row.cap, "SQL_QUERY");
        assert_eq!(spec_row.lang, "python");
        assert_eq!(spec_row.attack_provenance.as_deref(), Some("LlmSynthesised"));
        let (parsed, _) = nyx_agent_nyx::HarnessSpec::from_json(&spec_row.spec_blob).unwrap();
        parsed.validate().expect("vendored schema accepts persisted blob");

        let updated = store.findings().get(&fid).await.unwrap().expect("finding");
        assert_eq!(
            updated.spec_id.as_deref(),
            Some(spec_row.id.as_str()),
            "findings.spec_id must back-link to the persisted harness_specs row"
        );
        assert_eq!(updated.attack_provenance.as_deref(), Some("LlmSynthesised"));
        assert_eq!(updated.prompt_version.as_deref(), Some(SPEC_DERIVATION_PROMPT_VERSION));
    }

    #[tokio::test]
    async fn payload_synthesis_end_to_end_through_build_run_apply() {
        // Acceptance: a real Diag carrying Unsupported(NoPayloadsForCap)
        // travels build_inputs -> run_payload_synthesis (scripted runtime)
        // -> apply_outcome, lands a `payloads` row keyed to the seeded
        // finding with `attack_provenance = LlmSynthesised`, and stamps
        // the parent finding's provenance + prompt_version columns.
        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-P")).await.unwrap();
        store.runs().insert(&seed_run("run-P")).await.unwrap();
        let seed = seed_finding("run-P", "repo-P", "sink.py", "rule-e2e-payload");
        let fid = seed.id.clone();
        store.findings().upsert(&seed).await.unwrap();
        assert!(seed.attack_provenance.is_none(), "seed finding starts with no AI provenance");

        // Workspace with a python file at line 10 to match seed_finding's
        // BLAKE3-keyed line.
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("sink.py"),
            "import sqlite3\n# pad\n# pad\n# pad\n# pad\n# pad\n# pad\n# pad\n# pad\n\
             cursor.execute('SELECT * FROM users WHERE n=' + q)\n",
        )
        .unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert(
            "repo-P".to_string(),
            WorkspaceHandle::for_local_path_test("repo-P", workspace.path().to_path_buf()),
        );

        let diag = diag_unsupported("sink.py", 10, "SQL_QUERY", "rule-e2e-payload");
        let bundle = make_bundle("run-P", "repo-P", vec![diag]);

        let inputs = build_inputs(&bundle, &workspaces);
        assert_eq!(inputs.len(), 1, "the unsupported diag must fan out");
        assert_eq!(inputs[0].finding_id, fid, "input finding_id must match seeded row");
        assert_eq!(inputs[0].cap, "SQL_QUERY");
        assert_eq!(inputs[0].lang, "python");

        let body = serde_json::json!({
            "vuln_payload": "' OR 1=1 --",
            "vuln_oracle": "row count > 0 OR error contains 'SQL'",
            "benign_payload": "alice",
        })
        .to_string();
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-P", BudgetKind::OneShot, 5_000_000);
        let runtime = ScriptedNovelRuntime::new(vec![Ok(body)], 6_400, tracker.clone());

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let outcome = nyx_agent_ai::run_payload_synthesis(&runtime, &inputs[0], tx, 5_000_000)
            .await
            .expect("scripted runtime produces a synthesised payload");

        let mut report = PayloadSynthesisPassReport::default();
        apply_outcome(&store, outcome, &mut report, "scripted-novel", "scripted-model", 0)
            .await
            .unwrap();
        assert_eq!(report.synthesised, 1);
        assert_eq!(report.quarantined, 0);

        let payloads = store.payloads().list_for_finding(&fid).await.unwrap();
        assert_eq!(payloads.len(), 1, "exactly one payloads row persisted");
        let row = &payloads[0];
        assert_eq!(row.finding_id, fid);
        assert_eq!(row.cap, "SQL_QUERY");
        assert_eq!(row.lang, "python");
        assert_eq!(row.vuln_bytes, b"' OR 1=1 --");
        assert_eq!(row.benign_bytes.as_deref(), Some(b"alice".as_ref()));
        assert_eq!(row.attack_provenance.as_deref(), Some("LlmSynthesised"));
        assert_eq!(
            row.prompt_version.as_deref(),
            Some(nyx_agent_types::payload::PAYLOAD_SYNTHESIS_PROMPT_VERSION)
        );

        let updated = store.findings().get(&fid).await.unwrap().expect("finding");
        assert_eq!(
            updated.attack_provenance.as_deref(),
            Some("LlmSynthesised"),
            "finding's provenance must be stamped by the dual-write"
        );
        assert_eq!(
            updated.prompt_version.as_deref(),
            Some(nyx_agent_types::payload::PAYLOAD_SYNTHESIS_PROMPT_VERSION)
        );
    }

    // -------- chain-reasoning pass coverage --------

    fn diag_with_flow_step(
        path: &str,
        line: u32,
        cap: &str,
        rule: &str,
        flow: &[(&str, u32, &str)],
    ) -> Diag {
        let mut steps: Vec<serde_json::Value> = Vec::new();
        for (i, (f, l, k)) in flow.iter().enumerate() {
            steps.push(serde_json::json!({
                "step": i + 1,
                "kind": k,
                "file": f,
                "line": l,
            }));
        }
        let mut diag: Diag = serde_json::from_value(serde_json::json!({
            "path": path,
            "line": line,
            "severity": "High",
            "id": rule,
            "category": cap,
            "evidence": {
                "flow_steps": steps,
            },
        }))
        .unwrap();
        diag.lift_flow_steps();
        diag
    }

    fn two_repo_bundle() -> RunBundle<Diag> {
        // repo-A controller (entry, has a `source` flow_step) reaches
        // repo-B sink. The sink's flow_step points at the controller's
        // (path, line) tuple in repo-A, so the graph builder produces a
        // cross-repo `Reaches` edge.
        let entry = diag_with_flow_step(
            "controller.py",
            5,
            "SQL_QUERY",
            "rule-entry",
            &[("controller.py", 5, "source")],
        );
        let sink = diag_with_flow_step(
            "db.py",
            42,
            "SQL_QUERY",
            "rule-sink",
            &[("controller.py", 5, "call"), ("db.py", 42, "sink")],
        );
        RunBundle {
            run_id: "run-X".to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: vec![
                RepoBundle {
                    repo: "repo-A".to_string(),
                    outcome: RepoOutcome::Success(vec![entry]),
                    started_at_ms: 0,
                    finished_at_ms: 0,
                    elapsed_ms: 0,
                },
                RepoBundle {
                    repo: "repo-B".to_string(),
                    outcome: RepoOutcome::Success(vec![sink]),
                    started_at_ms: 0,
                    finished_at_ms: 0,
                    elapsed_ms: 0,
                },
            ],
            callgraph: CrossRepoCallgraphStub::default(),
        }
    }

    #[test]
    fn build_chain_input_emits_cross_repo_edge() {
        let bundle = two_repo_bundle();
        let input = build_chain_input(&bundle).expect("graph");
        assert_eq!(input.run_id, "run-X");
        assert_eq!(input.repos, vec!["repo-A".to_string(), "repo-B".to_string()]);
        assert_eq!(input.nodes.len(), 2);
        // Entry node classification picks up the `source` flow_step.
        let entry_node = input.nodes.iter().find(|n| n.repo == "repo-A").expect("entry");
        assert_eq!(entry_node.kind, NODE_KIND_ENTRY);
        let sink_node = input.nodes.iter().find(|n| n.repo == "repo-B").expect("sink");
        assert_eq!(sink_node.kind, NODE_KIND_SINK);
        // One cross-repo edge: entry -> sink.
        let cross: Vec<_> = input.edges.iter().filter(|e| e.cross_repo).collect();
        assert_eq!(cross.len(), 1, "edges: {:?}", input.edges);
        assert_eq!(cross[0].from, entry_node.id);
        assert_eq!(cross[0].to, sink_node.id);
        assert_eq!(cross[0].label, "Reaches");
    }

    #[test]
    fn build_chain_input_classifies_framework_path() {
        let mut bundle = two_repo_bundle();
        // Replace the entry diag with one whose path looks like a
        // vendored framework binding.
        let fw = diag_with_flow_step(
            "vendor/orm/query.py",
            10,
            "SQL_QUERY",
            "rule-fw",
            &[("vendor/orm/query.py", 10, "call")],
        );
        bundle.per_repo[0].outcome = RepoOutcome::Success(vec![fw]);
        let input = build_chain_input(&bundle).expect("graph");
        let node = input.nodes.iter().find(|n| n.repo == "repo-A").expect("fw");
        assert_eq!(node.kind, NODE_KIND_FRAMEWORK);
    }

    #[test]
    fn build_chain_input_returns_none_below_two_nodes() {
        let bundle = RunBundle::<Diag> {
            run_id: "r".to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: vec![RepoBundle {
                repo: "repo-A".to_string(),
                outcome: RepoOutcome::Success(vec![diag_with_flow_step(
                    "a.py",
                    1,
                    "SQL_QUERY",
                    "rule-1",
                    &[],
                )]),
                started_at_ms: 0,
                finished_at_ms: 0,
                elapsed_ms: 0,
            }],
            callgraph: CrossRepoCallgraphStub::default(),
        };
        assert!(build_chain_input(&bundle).is_none());
    }

    #[tokio::test]
    async fn apply_chain_outcome_persists_cross_repo_chain() {
        // Acceptance: a two-repo run with controller-in-repo-A
        // reaches-sink-in-repo-B fixture produces at least one cross-repo
        // chain row, with rationale stored and members back-linked.
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-A")).await.unwrap();
        store.repos().upsert(&seed_repo("repo-B")).await.unwrap();
        store.runs().insert(&seed_run("run-X")).await.unwrap();
        let bundle = two_repo_bundle();
        let input = build_chain_input(&bundle).expect("graph");

        // Seed the two finding rows the chain will link.
        let entry_node = input.nodes.iter().find(|n| n.repo == "repo-A").unwrap().clone();
        let sink_node = input.nodes.iter().find(|n| n.repo == "repo-B").unwrap().clone();
        for n in [&entry_node, &sink_node] {
            let f = nyx_agent_core::store::FindingRecord {
                id: n.id.clone(),
                run_id: "run-X".to_string(),
                repo: n.repo.clone(),
                path: n.path.clone(),
                line: n.line.map(i64::from),
                cap: n.cap.clone(),
                rule: n.rule.clone(),
                severity: n.severity.clone(),
                status: "Open".to_string(),
                finding_origin: "Static".to_string(),
                first_seen: 1_000,
                last_seen: 1_000,
                superseded_by: None,
                triage_state: "Open".to_string(),
                triage_assigned_to: None,
                verdict_blob: None,
                repro_path: None,
                attack_provenance: None,
                prompt_version: None,
                chain_id: None,
                spec_id: None,
            };
            store.findings().upsert(&f).await.unwrap();
        }

        let output = nyx_agent_types::chain::ChainReasoningOutput {
            chains: vec![nyx_agent_types::chain::ChainCandidate {
                member_ids: vec![entry_node.id.clone(), sink_node.id.clone()],
                rationale: "controller in repo-A reaches SQL sink in repo-B".to_string(),
                prerequisites: Vec::new(),
                evidence: Vec::new(),
                blast_radius: Vec::new(),
                confidence: 82,
                missing_verification_steps: Vec::new(),
                edge_provenance: Vec::new(),
            }],
        };
        let outcome = ChainReasoningOutcome::Ranked {
            run_id: "run-X".to_string(),
            output,
            prompt_version: nyx_agent_types::chain::CHAIN_REASONING_PROMPT_VERSION.to_string(),
            spent_usd_micros: 12_000,
            attempts: 1,
            metrics: AgentTraceMetrics::default(),
        };
        let mut report = ChainReasoningPassReport::default();
        apply_chain_outcome(
            &store,
            &input,
            outcome,
            &mut report,
            "project-test",
            "test-runtime",
            "test-model",
            0,
        )
        .await
        .unwrap();
        assert_eq!(report.chains_persisted, 1);
        assert_eq!(report.cross_repo_chains, 1);
        assert_eq!(report.members_stamped, 2);
        assert_eq!(report.spend_usd_micros, 12_000);

        // Chain row landed with cross_repo + LlmSynthesised provenance.
        let chains = store.chains().list_by_run("run-X").await.unwrap();
        assert_eq!(chains.len(), 1);
        let c = &chains[0];
        assert!(c.cross_repo);
        assert_eq!(c.attack_provenance.as_deref(), Some("LlmSynthesised"));
        assert_eq!(
            c.prompt_version.as_deref(),
            Some(nyx_agent_types::chain::CHAIN_REASONING_PROMPT_VERSION),
        );
        let rationale = c.rationale_blob.as_deref().unwrap();
        assert!(rationale.contains("controller in repo-A"), "rationale: {rationale}");
        let members: Vec<String> = serde_json::from_str(&c.member_ids).unwrap();
        assert_eq!(members, vec![entry_node.id.clone(), sink_node.id.clone()]);
        let evidence: serde_json::Value =
            serde_json::from_str(c.evidence_blob.as_deref().expect("evidence blob")).unwrap();
        assert_eq!(evidence.get("graph_backed").and_then(|v| v.as_bool()), Some(true));
        assert!(evidence
            .get("edge_provenance")
            .and_then(|v| v.as_array())
            .is_some_and(|edges| !edges.is_empty()));
        assert!(evidence
            .get("missing_verification_steps")
            .and_then(|v| v.as_array())
            .is_some_and(|gaps| !gaps.is_empty()));

        // Both findings have chain_id back-link stamped.
        let entry_row = store.findings().get(&entry_node.id).await.unwrap().unwrap();
        let sink_row = store.findings().get(&sink_node.id).await.unwrap().unwrap();
        assert_eq!(entry_row.chain_id.as_deref(), Some(c.id.as_str()));
        assert_eq!(sink_row.chain_id.as_deref(), Some(c.id.as_str()));
    }

    #[tokio::test]
    async fn apply_chain_outcome_promotes_terminal_live_chain_to_vulnerability() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        store
            .projects()
            .create("project-test", "Project Test", None, None, None, 1_000)
            .await
            .unwrap();
        let mut run = seed_run("run-live-chain");
        run.project_id = Some("project-test".to_string());
        store.runs().insert(&run).await.unwrap();
        store
            .verified_vulnerabilities()
            .upsert(&VerifiedVulnerabilityRecord {
                id: "vuln-export".to_string(),
                run_id: "run-live-chain".to_string(),
                project_id: "project-test".to_string(),
                title: "Tenant data export confirmed".to_string(),
                severity: "High".to_string(),
                confidence: 0.88,
                risk_score: 8.2,
                risk_rating: canonical_risk_rating("", 8.2),
                risk_score_source: "test".to_string(),
                risk_score_rationale: "existing live proof".to_string(),
                vuln_class: "ACCESS_CONTROL".to_string(),
                affected_components: vec![serde_json::json!({
                    "endpoint": "GET /api/export",
                    "path": "src/export.ts",
                })],
                business_impact: "tenant export data".to_string(),
                evidence_summary: "existing export proof".to_string(),
                repro_steps: "GET /api/export as another tenant".to_string(),
                remediation: "Enforce tenant authorization.".to_string(),
                source_candidate_ids: Vec::new(),
                source_signal_ids: Vec::new(),
                verification_attempt_ids: vec!["va-export".to_string()],
                chain_id: None,
                status: "Open".to_string(),
                first_seen: 500,
                last_seen: 500,
            })
            .await
            .unwrap();

        let candidate_node = ChainReasoningNode {
            id: "node-candidate".to_string(),
            graph_kind: Some(NODE_CANDIDATE.to_string()),
            label: Some("IDOR candidate".to_string()),
            ref_id: Some("pc-idor".to_string()),
            repo: "web".to_string(),
            path: "src/routes/projects.ts".to_string(),
            line: Some(17),
            cap: "IDOR".to_string(),
            rule: "candidate".to_string(),
            severity: "Low".to_string(),
            kind: NODE_KIND_ENTRY.to_string(),
            routes: vec!["GET /api/projects/:id".to_string()],
            roles: vec!["authenticated".to_string()],
            objects: vec!["project".to_string()],
            evidence_refs: vec!["pc-idor".to_string()],
        };
        let verified_node = ChainReasoningNode {
            id: "node-verified".to_string(),
            graph_kind: Some(NODE_VERIFIED_VULNERABILITY.to_string()),
            label: Some("Tenant data export confirmed".to_string()),
            ref_id: Some("vuln-export".to_string()),
            repo: "api".to_string(),
            path: "src/export.ts".to_string(),
            line: Some(88),
            cap: "ACCESS_CONTROL".to_string(),
            rule: "verified_vulnerability".to_string(),
            severity: "High".to_string(),
            kind: NODE_KIND_SINK.to_string(),
            routes: vec!["GET /api/export".to_string()],
            roles: vec!["authenticated".to_string()],
            objects: vec!["tenant export".to_string()],
            evidence_refs: vec!["vuln-export".to_string()],
        };
        let input = ChainReasoningInput {
            run_id: "run-live-chain".to_string(),
            repos: vec!["web".to_string(), "api".to_string()],
            nodes: vec![candidate_node.clone(), verified_node.clone()],
            edges: vec![ChainReasoningEdge {
                from: candidate_node.id.clone(),
                to: verified_node.id.clone(),
                label: "verified_as".to_string(),
                cross_repo: true,
                edge_id: Some("edge-live".to_string()),
                evidence_ref: Some("va-export".to_string()),
                source: Some("verification_attempt".to_string()),
            }],
            max_chains: 10,
        };
        let outcome = ChainReasoningOutcome::Ranked {
            run_id: "run-live-chain".to_string(),
            output: nyx_agent_types::chain::ChainReasoningOutput {
                chains: vec![ChainCandidate {
                    member_ids: vec![candidate_node.id.clone(), verified_node.id.clone()],
                    rationale:
                        "Low-severity IDOR setup reaches a live-confirmed tenant export impact."
                            .to_string(),
                    prerequisites: vec!["authenticated user".to_string()],
                    evidence: vec!["verified export proof".to_string()],
                    blast_radius: vec!["tenant export data".to_string()],
                    confidence: 91,
                    missing_verification_steps: Vec::new(),
                    edge_provenance: vec!["edge-live".to_string()],
                }],
            },
            prompt_version: CHAIN_REASONING_PROMPT_VERSION.to_string(),
            spent_usd_micros: 7_000,
            attempts: 1,
            metrics: AgentTraceMetrics::default(),
        };
        let mut report = ChainReasoningPassReport::default();
        apply_chain_outcome(
            &store,
            &input,
            outcome,
            &mut report,
            "project-test",
            "test-runtime",
            "test-model",
            0,
        )
        .await
        .unwrap();

        assert_eq!(report.chains_persisted, 1);
        assert_eq!(report.chains_verified, 1);
        assert_eq!(report.chains_needing_verification, 0);
        assert_eq!(report.vulnerabilities_recorded, 1);

        let chains = store.chains().list_by_run("run-live-chain").await.unwrap();
        assert_eq!(chains[0].status, "Verified");
        assert_eq!(chains[0].severity.as_deref(), Some("High"));
        let evidence: serde_json::Value =
            serde_json::from_str(chains[0].evidence_blob.as_deref().unwrap()).unwrap();
        assert_eq!(evidence.get("terminal_live_proof").and_then(|v| v.as_bool()), Some(true));

        let vulns = store.verified_vulnerabilities().list_by_run("run-live-chain").await.unwrap();
        assert_eq!(vulns.len(), 1);
        assert_eq!(vulns[0].id, "vuln-export");
        assert_eq!(vulns[0].title, "Tenant data export confirmed");
        assert_eq!(vulns[0].chain_id.as_deref(), Some(chains[0].id.as_str()));
        assert_eq!(vulns[0].severity, "Critical");
        assert_eq!(vulns[0].source_candidate_ids, vec!["pc-idor".to_string()]);
        assert_eq!(vulns[0].verification_attempt_ids, vec!["va-export".to_string()]);
        assert_eq!(vulns[0].first_seen, 500);
        assert!(vulns[0].evidence_summary.contains("Chain evidence:"));
    }

    #[tokio::test]
    async fn apply_chain_outcome_handles_no_chains_without_writes() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        store.runs().insert(&seed_run("run-X")).await.unwrap();
        let input = nyx_agent_types::chain::ChainReasoningInput {
            run_id: "run-X".to_string(),
            repos: vec!["repo-A".to_string()],
            nodes: vec![],
            edges: vec![],
            max_chains: 10,
        };
        let outcome = ChainReasoningOutcome::NoChains {
            run_id: "run-X".to_string(),
            reason: "chain reasoning failed twice (...; ...)".to_string(),
            spent_usd_micros: 1_000,
            attempts: 2,
            metrics: AgentTraceMetrics::default(),
        };
        let mut report = ChainReasoningPassReport::default();
        apply_chain_outcome(
            &store,
            &input,
            outcome,
            &mut report,
            "project-test",
            "test-runtime",
            "test-model",
            0,
        )
        .await
        .unwrap();
        assert_eq!(report.chains_persisted, 0);
        assert_eq!(report.cross_repo_chains, 0);
        assert_eq!(report.members_stamped, 0);
        assert_eq!(report.attempts, 2);
        assert_eq!(report.spend_usd_micros, 1_000);
        assert!(store.chains().list_by_run("run-X").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn chain_pass_is_noop_when_runtime_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        let secrets = SecretStore::memory();
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let bundle = two_repo_bundle();
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let cfg = AiConfig::default();
        let report = run_chain_reasoning_pass(&cfg, &store, &secrets, &bundle, &workspaces, tx)
            .await
            .unwrap();
        assert_eq!(report, ChainReasoningPassReport::default());
    }

    #[tokio::test]
    async fn chain_pass_is_noop_when_anthropic_but_no_key() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        let secrets = SecretStore::memory();
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let bundle = two_repo_bundle();
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let cfg = AiConfig { runtime: ConfigAiRuntime::Anthropic, ..AiConfig::default() };
        let report = run_chain_reasoning_pass(&cfg, &store, &secrets, &bundle, &workspaces, tx)
            .await
            .unwrap();
        assert_eq!(report, ChainReasoningPassReport::default());
    }

    // -------- novel-finding-discovery pass coverage --------

    use nyx_agent_ai::{AiRuntime, InMemoryBudgetTracker};
    use nyx_agent_types::agent::{
        AgentResult, AgentTask, Budget, CacheStats, CostEstimate, Prompt, Response, TokenUsage,
    };
    use std::sync::Mutex as StdMutex;

    /// Scripted runtime mirroring the per-task fixtures. Each `one_shot`
    /// pops the next response from the back of the queue.
    struct ScriptedNovelRuntime {
        responses: StdMutex<Vec<Result<String, AiError>>>,
        cost_per_call: i64,
        tracker: Arc<dyn BudgetTracker>,
    }

    impl ScriptedNovelRuntime {
        fn new(
            responses: Vec<Result<String, AiError>>,
            cost_per_call: i64,
            tracker: Arc<dyn BudgetTracker>,
        ) -> Self {
            Self { responses: StdMutex::new(responses), cost_per_call, tracker }
        }
    }

    #[async_trait]
    impl AiRuntime for ScriptedNovelRuntime {
        fn name(&self) -> &'static str {
            "scripted-novel"
        }
        fn default_model(&self) -> &str {
            "scripted-model"
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
            budget: Budget,
            _sink: nyx_agent_types::event::EventSink,
        ) -> Result<Response, AiError> {
            let next = self
                .responses
                .lock()
                .unwrap()
                .pop()
                .expect("scripted novel runtime: no more responses");
            let content = next?;
            let cost = self.cost_per_call;
            self.tracker.add_spend(&budget.run_id, budget.kind, cost).await?;
            Ok(Response {
                prompt_version: prompt.prompt_version,
                task_id: prompt.task_id,
                model: "scripted-model".to_string(),
                content,
                usage: TokenUsage { input_tokens: 500, output_tokens: 200 },
                cache: Some(CacheStats::default()),
                cost_usd_micros: cost,
            })
        }

        async fn agent_loop(
            &self,
            _task: AgentTask,
            _budget: Budget,
            _sink: nyx_agent_types::event::EventSink,
        ) -> Result<AgentResult, AiError> {
            Err(AiError::UnsupportedMode("agent_loop"))
        }

        fn cost_estimate(&self, _prompt: &Prompt) -> Option<CostEstimate> {
            Some(CostEstimate { min_usd_micros: 0, max_usd_micros: self.cost_per_call })
        }
    }

    fn two_python_workspace() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("app")).unwrap();
        std::fs::create_dir_all(tmp.path().join("models")).unwrap();
        // handlers.py: line 3 is the known SQL sink (the prior), line 6
        // is the intentionally similar second sink the agent should
        // flag.
        std::fs::write(
            tmp.path().join("app/handlers.py"),
            "def list_users(q):\n    sql = 'SELECT * FROM u WHERE n=' + q\n    cursor.execute(sql)\n\ndef list_admins(q):\n    sql2 = 'SELECT * FROM admin WHERE n=' + q\n    cursor.execute(sql2)\n",
        )
        .unwrap();
        // A lower-priority untouched model file so the walker has more
        // than one source file to choose from.
        std::fs::write(tmp.path().join("models/user.py"), "class User:\n    pass\n").unwrap();
        // A directory that must be skipped: ensure the walker doesn't
        // descend into node_modules.
        std::fs::create_dir_all(tmp.path().join("node_modules/junk")).unwrap();
        std::fs::write(tmp.path().join("node_modules/junk/index.js"), "module.exports = {}\n")
            .unwrap();
        tmp
    }

    #[test]
    fn priority_for_prefers_route_controller_handler() {
        let routes = priority_for(std::path::Path::new("app/routes/users.py"), 4_096, None);
        let plain = priority_for(std::path::Path::new("misc/notes.py"), 4_096, None);
        assert!(routes > plain, "routes={routes} plain={plain}");
    }

    #[test]
    fn priority_for_boosts_historical_promotion_rate() {
        // Two identically-keyworded paths: one with no rate, one with a
        // strong historical promotion rate. The boosted one must
        // outrank the unboosted one.
        let base = priority_for(std::path::Path::new("models/user.py"), 4_096, None);
        let zero = priority_for(std::path::Path::new("models/user.py"), 4_096, Some(0.0));
        let high = priority_for(std::path::Path::new("models/user.py"), 4_096, Some(0.9));
        assert_eq!(base, zero, "rate=0.0 must not change the score");
        assert!(high > base, "high={high} base={base}");
        // Cap saturation: rate above 1.0 collapses to the same boost
        // as rate = 1.0 (defence against a corrupt rate map).
        let saturated = priority_for(std::path::Path::new("models/user.py"), 4_096, Some(5.0));
        let ceiling = priority_for(std::path::Path::new("models/user.py"), 4_096, Some(1.0));
        assert_eq!(saturated, ceiling);
    }

    #[test]
    fn walk_source_files_skips_node_modules() {
        let tmp = two_python_workspace();
        let files = walk_source_files(tmp.path());
        let stems: Vec<String> = files.iter().map(|p| p.to_string_lossy().to_string()).collect();
        let any_nm = stems.iter().any(|s| s.contains("node_modules"));
        assert!(!any_nm, "node_modules must be skipped: {stems:?}");
    }

    #[test]
    fn walk_source_files_respects_repo_gitignore() {
        // A custom build dir that is NOT in SKIP_DIRS but IS in the
        // operator's .gitignore must be skipped. This is the close-out
        // case for the deferred "swap hardcoded SKIP_DIRS for the
        // `ignore` crate" item.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::create_dir_all(tmp.path().join("custom_artifacts")).unwrap();
        std::fs::write(tmp.path().join("src/main.py"), "x = 1\n").unwrap();
        std::fs::write(tmp.path().join("custom_artifacts/gen.py"), "x = 2\n").unwrap();
        std::fs::write(tmp.path().join(".gitignore"), "custom_artifacts/\n").unwrap();

        let files = walk_source_files(tmp.path());
        let stems: Vec<String> = files.iter().map(|p| p.to_string_lossy().to_string()).collect();
        assert!(
            stems.iter().any(|s| s.ends_with("src/main.py")),
            "main.py must surface: {stems:?}",
        );
        assert!(
            !stems.iter().any(|s| s.contains("custom_artifacts")),
            "gitignored dir must be skipped: {stems:?}",
        );
    }

    #[test]
    fn build_novel_inputs_attaches_priors_per_file() {
        // One known SQL sink on handlers.py at line 3 -> the only prior;
        // the second sink at line 6 is intentionally NOT in priors so
        // the model has something to find.
        let tmp = two_python_workspace();
        let diag = diag_supported("app/handlers.py", 3, "SQL_QUERY", "py.sql.exec");
        let inputs = build_novel_inputs_for_repo(
            "run-N",
            "repo-1",
            tmp.path(),
            &[diag],
            DEFAULT_FILES_PER_BATCH,
            None,
        );
        assert!(!inputs.is_empty(), "walker must produce at least one batch");
        let first = &inputs[0];
        assert_eq!(first.run_id, "run-N");
        assert_eq!(first.repo, "repo-1");
        assert_eq!(first.batch_id, "repo-1:0");
        let paths: Vec<&str> = first.files.iter().map(|f| f.path.as_str()).collect();
        assert!(
            paths.contains(&"app/handlers.py"),
            "handlers.py must surface in the batch: {paths:?}",
        );
        // The prior must be forwarded so the model knows to skip line 3.
        assert!(first
            .priors
            .iter()
            .any(|p| p.path == "app/handlers.py" && p.line == 3 && p.cap == "SQL_QUERY"));
    }

    #[test]
    fn build_novel_inputs_chunks_into_batches() {
        // Force a tiny batch size so the chunker fires even on a small
        // workspace.
        let tmp = two_python_workspace();
        let inputs = build_novel_inputs_for_repo("run-N", "repo-1", tmp.path(), &[], 1, None);
        assert!(inputs.len() >= 2, "got: {}", inputs.len());
        for (i, b) in inputs.iter().enumerate() {
            assert_eq!(b.batch_id, format!("repo-1:{i}"));
            assert_eq!(b.files.len(), 1);
        }
    }

    #[test]
    fn build_novel_inputs_promotion_rates_boost_path_to_top_batch() {
        // Two source files: a low-keyword path with a strong historical
        // promotion rate vs a high-keyword path with no history. The
        // boost (PROMOTION_RATE_WEIGHT = 10 at rate = 1.0) must outrank
        // the strongest keyword hit (+6 for "route"/"controller") so
        // the historically-converting file lands in the first batch.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("misc")).unwrap();
        std::fs::create_dir_all(tmp.path().join("app/routes")).unwrap();
        let body = "def x():\n    pass\n".repeat(40);
        std::fs::write(tmp.path().join("misc/notes.py"), &body).unwrap();
        std::fs::write(tmp.path().join("app/routes/users.py"), &body).unwrap();

        let mut rates = HashMap::new();
        // Boost the otherwise-low-priority misc/notes.py path to a
        // near-ceiling rate. "misc/notes.py" matches "exec" in the
        // keyword table (because the body contains "pass"? no; keyword
        // table operates on the lowercased path, which does contain "ex" but not
        // "exec"; the path "misc/notes.py" alone scores 0 for keywords).
        rates.insert("misc/notes.py".to_string(), 1.0);
        let inputs =
            build_novel_inputs_for_repo("run-N", "repo-1", tmp.path(), &[], 1, Some(&rates));
        assert!(inputs.len() >= 2, "expected at least 2 batches; got {}", inputs.len());
        assert_eq!(
            inputs[0].files[0].path, "misc/notes.py",
            "promotion-rate boost must lift misc/notes.py above route-keyword path",
        );
    }

    #[tokio::test]
    async fn drive_novel_finding_pass_persists_candidate_for_similar_second_sink() {
        // NovelFindingDiscovery acceptance: a repo with one nyx-finding
        // (line 3) and an intentionally-similar second vulnerability
        // (line 6) produces a CandidateFinding for the second one. The
        // candidate lands as `candidate_findings.Pending` so nothing
        // surfaces to the operator without the payload verifier
        // confirming it.
        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-1")).await.unwrap();
        store.runs().insert(&seed_run("run-N")).await.unwrap();

        let workspace = two_python_workspace();
        let mut workspaces = HashMap::new();
        workspaces.insert(
            "repo-1".to_string(),
            WorkspaceHandle::for_local_path_test("repo-1", workspace.path().to_path_buf()),
        );

        let diag = diag_supported("app/handlers.py", 3, "SQL_QUERY", "py.sql.exec");
        let bundle = make_bundle("run-N", "repo-1", vec![diag]);

        let body = serde_json::json!({
            "candidates": [{
                "path": "app/handlers.py",
                "line": 6,
                "cap": "SQL_QUERY",
                "rule_hint": "py.sql.exec",
                "rationale": "list_admins reuses the same SQL-concat pattern as the prior at line 3",
                "suggested_payload_hint": "' OR 1=1 --"
            }]
        })
        .to_string();
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-N", BudgetKind::OneShot, 5_000_000);
        let runtime = ScriptedNovelRuntime::new(vec![Ok(body)], 7_500, tracker.clone());

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let report = drive_novel_finding_pass(
            &runtime,
            tracker.as_ref(),
            &store,
            &bundle,
            &workspaces,
            tx,
            5_000_000,
            5_000_000,
            DEFAULT_FILES_PER_BATCH,
        )
        .await
        .unwrap();

        assert_eq!(report.candidates_persisted, 1);
        assert!(report.batches_dispatched >= 1);
        assert_eq!(report.batches_halted, 0);
        assert_eq!(report.failed, 0);

        let pending = store.candidate_findings().list_pending().await.unwrap();
        assert_eq!(pending.len(), 1, "exactly one CandidateFinding must be quarantined");
        let row = &pending[0];
        assert_eq!(row.repo, "repo-1");
        assert_eq!(row.path, "app/handlers.py");
        assert_eq!(row.line, Some(6));
        assert_eq!(row.cap, "SQL_QUERY");
        assert_eq!(row.status, "Pending");
        assert_eq!(
            row.prompt_version.as_deref(),
            Some(nyx_agent_types::novel::NOVEL_FINDING_DISCOVERY_PROMPT_VERSION)
        );
        assert!(row.rationale.as_deref().unwrap_or("").contains("list_admins"));

        // The proposing AI call's trace row is back-linked on the
        // candidate via `candidate_findings.trace_id`. The quarantine UI's
        // trace viewer reads this back-link so the operator can see the
        // call that proposed a Pending candidate without joining on
        // task_kind = NovelFindings alone.
        let trace_id = row.trace_id.clone().expect("candidate must carry trace_id back-link");
        let traces = store.agent_traces().list_for_candidate(&row.id).await.unwrap();
        assert_eq!(traces.len(), 1, "back-linked trace must be reachable via list_for_candidate");
        assert_eq!(traces[0].id, trace_id);
        assert_eq!(traces[0].task_kind, TaskKind::NovelFindings.as_str());
    }

    #[tokio::test]
    async fn drive_novel_finding_pass_halts_on_budget_cap() {
        // Acceptance: the per-run cap halts further batches once spend
        // crosses the cap. With `files_per_batch = 1` and a two-file
        // workspace, the first call exhausts the cap, so the second
        // batch is marked halted instead of dispatched. The scripted
        // runtime is queued with exactly one response; if the halt
        // logic broke and a second one_shot fired, it would panic.
        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-B")).await.unwrap();
        store.runs().insert(&seed_run("run-Bg")).await.unwrap();

        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("controller.py"), "def f():\n    pass\n").unwrap();
        std::fs::write(workspace.path().join("api.py"), "def g():\n    pass\n").unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert(
            "repo-B".to_string(),
            WorkspaceHandle::for_local_path_test("repo-B", workspace.path().to_path_buf()),
        );

        // Use a small cap so a single call lands us at the ceiling.
        let cap = 1_000_i64;
        let body = serde_json::json!({ "candidates": [] }).to_string();
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-Bg", BudgetKind::OneShot, cap);
        // Only ONE scripted response, even though we expect TWO batches:
        // the second batch must short-circuit on the cap before issuing
        // a one_shot call, otherwise the runtime would panic on an
        // empty response queue.
        let runtime = ScriptedNovelRuntime::new(vec![Ok(body)], cap, tracker.clone());

        let bundle = make_bundle("run-Bg", "repo-B", Vec::new());
        let (tx, _rx) = tokio::sync::broadcast::channel(4);

        // Sanity: with files_per_batch=1 the walker must emit >=2
        // batches so the halt path is exercised.
        let inputs =
            build_novel_inputs_for_repo("run-Bg", "repo-B", workspace.path(), &[], 1, None);
        assert!(inputs.len() >= 2, "fixture must produce >=2 batches; got {}", inputs.len());

        let report = drive_novel_finding_pass(
            &runtime,
            tracker.as_ref(),
            &store,
            &bundle,
            &workspaces,
            tx,
            cap,
            cap,
            1,
        )
        .await
        .unwrap();

        // The first call exhausts the cap; every subsequent batch in
        // the same repo must be halted before issuing a one_shot.
        assert_eq!(
            report.batches_dispatched, 1,
            "exactly one batch should fire before the cap halts further dispatch"
        );
        assert!(
            report.batches_halted >= 1,
            "at least one batch must record a halt; got {}",
            report.batches_halted
        );
        assert_eq!(
            report.failed, 0,
            "no scripted errors expected; failure means runtime tried a second call"
        );
        let spent = tracker.spent("run-Bg", BudgetKind::OneShot);
        assert_eq!(spent, cap, "exactly one call's worth of spend should land in the bucket");
    }

    #[tokio::test]
    async fn novel_pass_is_noop_when_runtime_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        let secrets = SecretStore::memory();
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let bundle = RunBundle::<Diag> {
            run_id: "r".to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: Vec::new(),
            callgraph: CrossRepoCallgraphStub::default(),
        };
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let cfg = AiConfig::default();
        let report =
            run_novel_finding_discovery_pass(&cfg, &store, &secrets, &bundle, &workspaces, tx)
                .await
                .unwrap();
        assert_eq!(report, NovelFindingDiscoveryPassReport::default());
    }

    #[tokio::test]
    async fn novel_pass_is_noop_when_anthropic_but_no_key() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path()).await.unwrap();
        let secrets = SecretStore::memory();
        let workspaces: HashMap<String, WorkspaceHandle> = HashMap::new();
        let bundle = RunBundle::<Diag> {
            run_id: "r".to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: Vec::new(),
            callgraph: CrossRepoCallgraphStub::default(),
        };
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let cfg = AiConfig { runtime: ConfigAiRuntime::Anthropic, ..AiConfig::default() };
        let report =
            run_novel_finding_discovery_pass(&cfg, &store, &secrets, &bundle, &workspaces, tx)
                .await
                .unwrap();
        assert_eq!(report, NovelFindingDiscoveryPassReport::default());
    }

    // -------- payload verification pass coverage --------------------

    fn shell_spec_blob() -> String {
        // Canned SQLi-style shell harness. Same fixture as the
        // payload_runner unit tests, in JSON form for the
        // `harness_specs` table.
        serde_json::json!({
            "schema_version": 1,
            "cap": "SQL_QUERY",
            "lang": "shell",
            "entry": "harness:run",
            "setup": ["STORED='alice:pw1\\nbob:pw2\\nadmin:TOP_SECRET'"],
            "invoke": "printf '%b\\n' \"$STORED\" | grep -E @PAYLOAD || true",
            "payload_arg": 0,
            "oracle": "TOP_SECRET",
            "teardown": [],
        })
        .to_string()
    }

    fn seed_payload(finding_id: &str, vuln: &[u8], benign: &[u8]) -> PayloadRecord {
        PayloadRecord {
            id: format!("payload-{finding_id}"),
            finding_id: finding_id.to_string(),
            cap: "SQL_QUERY".to_string(),
            lang: "shell".to_string(),
            vuln_bytes: vuln.to_vec(),
            benign_bytes: Some(benign.to_vec()),
            oracle_blob: Some("TOP_SECRET".to_string()),
            attack_provenance: Some("LlmSynthesised".to_string()),
            prompt_version: Some("phase14.payload_synthesis.v1".to_string()),
            created_at: 5_000,
        }
    }

    fn seed_spec(id: &str) -> HarnessSpecRecord {
        HarnessSpecRecord {
            id: id.to_string(),
            cap: "SQL_QUERY".to_string(),
            lang: "shell".to_string(),
            spec_blob: shell_spec_blob(),
            attack_provenance: Some("LlmSynthesised".to_string()),
            prompt_version: Some("phase15.spec_derivation.v1".to_string()),
            created_at: 6_000,
        }
    }

    async fn ws_handle_for(repo: &str) -> (tempfile::TempDir, WorkspaceHandle) {
        let dir = tempfile::tempdir().unwrap();
        let handle = WorkspaceHandle::for_local_path_test(repo, dir.path().to_path_buf());
        (dir, handle)
    }

    fn empty_bundle(run_id: &str) -> RunBundle<Diag> {
        RunBundle {
            run_id: run_id.to_string(),
            project_id: "default-project".to_string(),
            started_at_ms: 0,
            finished_at_ms: 0,
            wall_clock_ms: 0,
            per_repo: Vec::new(),
            callgraph: CrossRepoCallgraphStub::default(),
        }
    }

    #[tokio::test]
    async fn verifier_confirms_finding_with_llm_payload() {
        // Verifier acceptance: an LLM-synthesised payload for a test
        // finding flows through the verifier and lands a verdict.
        let (_ws_tmp, ws_handle) = ws_handle_for("repo-V").await;
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-V".to_string(), ws_handle);

        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-V")).await.unwrap();
        store.runs().insert(&seed_run("run-V")).await.unwrap();
        let finding = seed_finding("run-V", "repo-V", "src/sink.sh", "rule-sqli");
        let fid = finding.id.clone();
        store.findings().upsert(&finding).await.unwrap();
        store.payloads().insert(&seed_payload(&fid, b".*", b"^alice$")).await.unwrap();
        store.harness_specs().insert(&seed_spec("spec-V")).await.unwrap();

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let report = run_payload_verification_pass(
            &RunConfig::default(),
            &SandboxConfig::default(),
            &store,
            &empty_bundle("run-V"),
            &workspaces,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(report.confirmed, 1, "{report:?}");
        assert_eq!(report.not_confirmed, 0);
        assert_eq!(report.errored, 0);

        let row = store.findings().get(&fid).await.unwrap().expect("row");
        assert_eq!(row.status, "Verified");
        assert_eq!(row.attack_provenance.as_deref(), Some("LlmSynthesised"));
        let blob = row.verdict_blob.expect("blob");
        let result: VerifyResult = serde_json::from_str(&blob).unwrap();
        assert_eq!(result.verdict, VerifyVerdict::Confirmed);
        assert!(result.vuln_run.oracle_fired);
        assert!(!result.benign_run.oracle_fired);

        let traces = store.agent_traces().list_for_finding(&fid).await.unwrap();
        let verifier_rows: Vec<_> =
            traces.into_iter().filter(|t| t.task_kind == "Verifier").collect();
        assert_eq!(verifier_rows.len(), 1, "expected one Verifier trace row");
        let trace = &verifier_rows[0];
        assert_eq!(trace.runtime_name, "process");
        assert_eq!(trace.cost_usd_micros, 0);
        assert!(trace.duration_ms.is_some());
        assert!(trace.finished_at.is_some());
    }

    #[tokio::test]
    async fn verifier_closes_finding_when_payload_is_benign() {
        // Verifier acceptance: replacing the vuln payload with the
        // benign one yields NotConfirmed; the finding flips to Closed.
        let (_ws_tmp, ws_handle) = ws_handle_for("repo-B").await;
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-B".to_string(), ws_handle);

        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-B")).await.unwrap();
        store.runs().insert(&seed_run("run-B")).await.unwrap();
        let finding = seed_finding("run-B", "repo-B", "src/sink.sh", "rule-sqli");
        let fid = finding.id.clone();
        store.findings().upsert(&finding).await.unwrap();
        // Both payloads are the benign control; neither trips the oracle.
        store.payloads().insert(&seed_payload(&fid, b"^alice$", b"^alice$")).await.unwrap();
        store.harness_specs().insert(&seed_spec("spec-B")).await.unwrap();

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let report = run_payload_verification_pass(
            &RunConfig::default(),
            &SandboxConfig::default(),
            &store,
            &empty_bundle("run-B"),
            &workspaces,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(report.confirmed, 0);
        assert_eq!(report.not_confirmed, 1, "{report:?}");
        let row = store.findings().get(&fid).await.unwrap().expect("row");
        assert_eq!(row.status, "Closed");
    }

    #[tokio::test]
    async fn verifier_promotes_quarantined_candidate_on_confirmed() {
        // Verifier acceptance: an AI-discovered candidate gets
        // promoted from Quarantined to Confirmed when its verify
        // passes. The promoted row lands with `finding_origin =
        // AiExploration`.
        let (_ws_tmp, ws_handle) = ws_handle_for("repo-C").await;
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-C".to_string(), ws_handle);

        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-C")).await.unwrap();
        store.runs().insert(&seed_run("run-C")).await.unwrap();

        let cand = CandidateFindingRecord {
            id: "cand-c1".to_string(),
            run_id: "run-C".to_string(),
            repo: "repo-C".to_string(),
            path: "app/handlers.sh".to_string(),
            line: Some(42),
            cap: "SQL_QUERY".to_string(),
            rule_hint: Some("sh.sql.exec".to_string()),
            rationale: Some("similar SQL-concat pattern".to_string()),
            suggested_payload_hint: Some(".*".to_string()),
            status: "Pending".to_string(),
            prompt_version: Some(
                nyx_agent_types::novel::NOVEL_FINDING_DISCOVERY_PROMPT_VERSION.to_string(),
            ),
            trace_id: None,
        };
        store.candidate_findings().insert(&cand).await.unwrap();
        // Candidate promotion uses the built-in per-cap harness
        // template seeded by `suggested_payload_hint`; no
        // payload / spec rows are pre-staged. The candidate-confirmation
        // pipeline (deferred) swaps this to real per-candidate
        // synthesis output.

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let report = run_payload_verification_pass(
            &RunConfig::default(),
            &SandboxConfig::default(),
            &store,
            &empty_bundle("run-C"),
            &workspaces,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(report.candidates_promoted, 1, "{report:?}");
        assert_eq!(report.confirmed, 1);

        // The candidate flipped to Promoted.
        let promoted = store.candidate_findings().get(&cand.id).await.unwrap().expect("row");
        assert_eq!(promoted.status, "Promoted");

        // A new findings row appeared with finding_origin = AiExploration
        // and status = Verified.
        let findings = store.findings().list_by_run("run-C").await.unwrap();
        assert_eq!(findings.len(), 1);
        let f = &findings[0];
        assert_eq!(f.finding_origin, "AiExploration");
        assert_eq!(f.status, "Verified");
        assert_eq!(f.repo, "repo-C");
        assert_eq!(f.path, "app/handlers.sh");
        assert_eq!(f.line, Some(42));
        assert_eq!(f.cap, "SQL_QUERY");
    }

    #[tokio::test]
    async fn verifier_promotes_quarantined_candidate_for_os_command_cap() {
        // OS_COMMAND candidate promotion: the built-in shell harness
        // evaluates the `suggested_payload_hint` as a shell command via
        // `sh -c`. A vuln payload that creates the sentinel file trips
        // `Oracle::SinkProbe`; the constant benign control (`:`) is a
        // shell no-op that leaves the workspace untouched.
        let (_ws_tmp, ws_handle) = ws_handle_for("repo-D").await;
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-D".to_string(), ws_handle);

        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-D")).await.unwrap();
        store.runs().insert(&seed_run("run-D")).await.unwrap();

        let cand = CandidateFindingRecord {
            id: "cand-d1".to_string(),
            run_id: "run-D".to_string(),
            repo: "repo-D".to_string(),
            path: "app/spawn.sh".to_string(),
            line: Some(17),
            cap: "OS_COMMAND".to_string(),
            rule_hint: Some("sh.subprocess.shell-true".to_string()),
            rationale: Some("user input flows into Popen(..., shell=True)".to_string()),
            suggested_payload_hint: Some(": > nyx_oscmd.flag".to_string()),
            status: "Pending".to_string(),
            prompt_version: Some(
                nyx_agent_types::novel::NOVEL_FINDING_DISCOVERY_PROMPT_VERSION.to_string(),
            ),
            trace_id: None,
        };
        store.candidate_findings().insert(&cand).await.unwrap();

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let report = run_payload_verification_pass(
            &RunConfig::default(),
            &SandboxConfig::default(),
            &store,
            &empty_bundle("run-D"),
            &workspaces,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(report.candidates_promoted, 1, "{report:?}");
        assert_eq!(report.confirmed, 1);

        let promoted = store.candidate_findings().get(&cand.id).await.unwrap().expect("row");
        assert_eq!(promoted.status, "Promoted");

        let findings = store.findings().list_by_run("run-D").await.unwrap();
        assert_eq!(findings.len(), 1);
        let f = &findings[0];
        assert_eq!(f.finding_origin, "AiExploration");
        assert_eq!(f.status, "Verified");
        assert_eq!(f.cap, "OS_COMMAND");
    }

    #[tokio::test]
    async fn verifier_promotes_quarantined_candidate_for_cmdi_cap() {
        // `CMDi` shares the OS_COMMAND template / sentinel oracle. This
        // test pins the alias so a future rename of the cap label does
        // not silently downgrade `CMDi` candidates back to "skipped".
        let (_ws_tmp, ws_handle) = ws_handle_for("repo-E").await;
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-E".to_string(), ws_handle);

        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-E")).await.unwrap();
        store.runs().insert(&seed_run("run-E")).await.unwrap();

        let cand = CandidateFindingRecord {
            id: "cand-e1".to_string(),
            run_id: "run-E".to_string(),
            repo: "repo-E".to_string(),
            path: "app/exec.js".to_string(),
            line: Some(8),
            cap: "CMDi".to_string(),
            rule_hint: Some("js.child_process.exec".to_string()),
            rationale: Some("user input concatenated into child_process.exec".to_string()),
            suggested_payload_hint: Some("touch nyx_oscmd.flag".to_string()),
            status: "Pending".to_string(),
            prompt_version: Some(
                nyx_agent_types::novel::NOVEL_FINDING_DISCOVERY_PROMPT_VERSION.to_string(),
            ),
            trace_id: None,
        };
        store.candidate_findings().insert(&cand).await.unwrap();

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let report = run_payload_verification_pass(
            &RunConfig::default(),
            &SandboxConfig::default(),
            &store,
            &empty_bundle("run-E"),
            &workspaces,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(report.candidates_promoted, 1, "{report:?}");
        assert_eq!(report.confirmed, 1);

        let findings = store.findings().list_by_run("run-E").await.unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].cap, "CMDi");
    }

    #[tokio::test]
    async fn verifier_promotes_quarantined_candidate_for_path_traversal_cap() {
        // PATH_TRAVERSAL candidate promotion: the built-in shell harness
        // stages a `safe_dir/innocent.txt` (the application's intended
        // container) plus an out-of-band `oracle_secret.txt` next to it.
        // The vuln payload `../oracle_secret.txt` escapes `safe_dir/`,
        // so the `cat -- safe_dir/<payload>` invocation leaks the
        // `TOP_SECRET_TRAV` marker to stdout and trips the
        // `OutputContains` oracle. The constant benign control
        // (`innocent.txt`) reads the in-`safe_dir` file and emits
        // "workspace", which the oracle does not match.
        let (_ws_tmp, ws_handle) = ws_handle_for("repo-PT").await;
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-PT".to_string(), ws_handle);

        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-PT")).await.unwrap();
        store.runs().insert(&seed_run("run-PT")).await.unwrap();

        let cand = CandidateFindingRecord {
            id: "cand-pt1".to_string(),
            run_id: "run-PT".to_string(),
            repo: "repo-PT".to_string(),
            path: "app/serve_file.py".to_string(),
            line: Some(42),
            cap: "PATH_TRAVERSAL".to_string(),
            rule_hint: Some("py.flask.send_file-userinput".to_string()),
            rationale: Some(
                "request filename concatenated into send_file path without normalisation"
                    .to_string(),
            ),
            suggested_payload_hint: Some("../oracle_secret.txt".to_string()),
            status: "Pending".to_string(),
            prompt_version: Some(
                nyx_agent_types::novel::NOVEL_FINDING_DISCOVERY_PROMPT_VERSION.to_string(),
            ),
            trace_id: None,
        };
        store.candidate_findings().insert(&cand).await.unwrap();

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let report = run_payload_verification_pass(
            &RunConfig::default(),
            &SandboxConfig::default(),
            &store,
            &empty_bundle("run-PT"),
            &workspaces,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(report.candidates_promoted, 1, "{report:?}");
        assert_eq!(report.confirmed, 1);

        let promoted = store.candidate_findings().get(&cand.id).await.unwrap().expect("row");
        assert_eq!(promoted.status, "Promoted");

        let findings = store.findings().list_by_run("run-PT").await.unwrap();
        assert_eq!(findings.len(), 1);
        let f = &findings[0];
        assert_eq!(f.finding_origin, "AiExploration");
        assert_eq!(f.status, "Verified");
        assert_eq!(f.cap, "PATH_TRAVERSAL");
    }

    #[tokio::test]
    async fn verifier_promotes_quarantined_candidate_for_xxe_cap() {
        // XXE candidate promotion: the built-in python harness drives
        // expat directly. The vuln payload declares an internal general
        // entity `x` whose value is the `TOP_SECRET_XXE` marker and
        // expands it inside `<r>&x;</r>`; expat substitutes the entity
        // at parse time and the CharacterDataHandler captures the
        // marker text, which the `OutputContains` oracle matches. The
        // benign control `<r>workspace</r>` parses cleanly with no
        // entity declaration, so the captured text is "workspace" and
        // the oracle does not fire.
        let (_ws_tmp, ws_handle) = ws_handle_for("repo-XXE").await;
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-XXE".to_string(), ws_handle);

        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-XXE")).await.unwrap();
        store.runs().insert(&seed_run("run-XXE")).await.unwrap();

        let cand = CandidateFindingRecord {
            id: "cand-xxe1".to_string(),
            run_id: "run-XXE".to_string(),
            repo: "repo-XXE".to_string(),
            path: "app/parse_doc.py".to_string(),
            line: Some(57),
            cap: "XXE".to_string(),
            rule_hint: Some("py.expat.parse-userinput".to_string()),
            rationale: Some(
                "request body parsed via expat without entity-resolution lockdown".to_string(),
            ),
            suggested_payload_hint: Some(
                "<?xml version=\"1.0\"?><!DOCTYPE r [<!ENTITY x \"TOP_SECRET_XXE\">]><r>&x;</r>"
                    .to_string(),
            ),
            status: "Pending".to_string(),
            prompt_version: Some(
                nyx_agent_types::novel::NOVEL_FINDING_DISCOVERY_PROMPT_VERSION.to_string(),
            ),
            trace_id: None,
        };
        store.candidate_findings().insert(&cand).await.unwrap();

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let report = run_payload_verification_pass(
            &RunConfig::default(),
            &SandboxConfig::default(),
            &store,
            &empty_bundle("run-XXE"),
            &workspaces,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(report.candidates_promoted, 1, "{report:?}");
        assert_eq!(report.confirmed, 1);

        let promoted = store.candidate_findings().get(&cand.id).await.unwrap().expect("row");
        assert_eq!(promoted.status, "Promoted");

        let findings = store.findings().list_by_run("run-XXE").await.unwrap();
        assert_eq!(findings.len(), 1);
        let f = &findings[0];
        assert_eq!(f.finding_origin, "AiExploration");
        assert_eq!(f.status, "Verified");
        assert_eq!(f.cap, "XXE");
    }

    #[tokio::test]
    async fn verifier_promotes_quarantined_candidate_for_deserialisation_cap() {
        // DESERIALISATION candidate promotion: the built-in python
        // harness feeds the payload bytes verbatim into `pickle.loads`.
        // The vuln payload is a protocol-0 pickle (`cbuiltins\nprint\n
        // (VTOP_SECRET_PICKLE\ntR.`) that REDUCEs `builtins.print` with
        // the `TOP_SECRET_PICKLE` marker; pickle.loads invokes the
        // REDUCE during unpickling, writing the marker to stdout and
        // tripping the oracle. The benign control `N.` (NONE + STOP)
        // returns `None` with no side effect, so stdout stays empty.
        let (_ws_tmp, ws_handle) = ws_handle_for("repo-PK").await;
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-PK".to_string(), ws_handle);

        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-PK")).await.unwrap();
        store.runs().insert(&seed_run("run-PK")).await.unwrap();

        let cand = CandidateFindingRecord {
            id: "cand-pk1".to_string(),
            run_id: "run-PK".to_string(),
            repo: "repo-PK".to_string(),
            path: "app/load_blob.py".to_string(),
            line: Some(91),
            cap: "DESERIALISATION".to_string(),
            rule_hint: Some("py.pickle.loads-userinput".to_string()),
            rationale: Some(
                "request body passed to pickle.loads without a safe-allowlist Unpickler"
                    .to_string(),
            ),
            suggested_payload_hint: Some("cbuiltins\nprint\n(VTOP_SECRET_PICKLE\ntR.".to_string()),
            status: "Pending".to_string(),
            prompt_version: Some(
                nyx_agent_types::novel::NOVEL_FINDING_DISCOVERY_PROMPT_VERSION.to_string(),
            ),
            trace_id: None,
        };
        store.candidate_findings().insert(&cand).await.unwrap();

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let report = run_payload_verification_pass(
            &RunConfig::default(),
            &SandboxConfig::default(),
            &store,
            &empty_bundle("run-PK"),
            &workspaces,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(report.candidates_promoted, 1, "{report:?}");
        assert_eq!(report.confirmed, 1);

        let promoted = store.candidate_findings().get(&cand.id).await.unwrap().expect("row");
        assert_eq!(promoted.status, "Promoted");

        let findings = store.findings().list_by_run("run-PK").await.unwrap();
        assert_eq!(findings.len(), 1);
        let f = &findings[0];
        assert_eq!(f.finding_origin, "AiExploration");
        assert_eq!(f.status, "Verified");
        assert_eq!(f.cap, "DESERIALISATION");
    }

    #[test]
    fn builtin_harness_table_covers_known_caps_and_skips_unknown() {
        // Sentinel test against silent drift in the per-cap table that
        // backs `drive_verify_for_candidate`. Every covered cap must
        // also return a non-`OutputContains{ORACLE_FIRED}` (i.e.
        // non-default) oracle and a benign control distinct from the
        // catch-all marker; every uncovered cap (XSS / SSRF, both
        // need infra we do not have: DOM runtime, loopback listener)
        // must return `None` so the verifier skips it instead of
        // confirming on a generic template that has no chance of
        // matching the real sink.
        for cap in
            ["SQL_QUERY", "SQLi", "OS_COMMAND", "CMDi", "PATH_TRAVERSAL", "XXE", "DESERIALISATION"]
        {
            let spec = builtin_harness_for_cap(cap).expect(cap);
            assert_eq!(spec.cap, cap);
            assert!(
                matches!(spec.lang.as_str(), "shell" | "python"),
                "cap {cap} lang {} unexpected",
                spec.lang
            );
            assert!(!matches!(
                builtin_oracle_for_cap(cap),
                Oracle::OutputContains { ref marker } if marker == "ORACLE_FIRED"
            ));
            assert_ne!(builtin_benign_for_cap(cap), "__nyx_benign_control__");
        }
        for cap in ["XSS", "SSRF"] {
            assert!(
                builtin_harness_for_cap(cap).is_none(),
                "cap {cap} should fall through to the candidate-confirmation pipeline"
            );
        }
    }

    #[tokio::test]
    async fn verifier_skips_findings_without_payload_or_spec() {
        // No payload/spec rows -> the finding is left alone and the
        // pass reports it as skipped-no-payload.
        let (_ws_tmp, ws_handle) = ws_handle_for("repo-S").await;
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-S".to_string(), ws_handle);

        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-S")).await.unwrap();
        store.runs().insert(&seed_run("run-S")).await.unwrap();
        let finding = seed_finding("run-S", "repo-S", "src/sink.sh", "rule-orphan");
        let fid = finding.id.clone();
        store.findings().upsert(&finding).await.unwrap();

        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let report = run_payload_verification_pass(
            &RunConfig::default(),
            &SandboxConfig::default(),
            &store,
            &empty_bundle("run-S"),
            &workspaces,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(report.skipped_no_payload, 1);
        let row = store.findings().get(&fid).await.unwrap().expect("row");
        assert_eq!(row.status, "Open", "status untouched without a payload");
    }

    #[tokio::test]
    async fn verifier_pass_emits_started_and_finished_sandbox_events() {
        // Subscribers on the run-wide bus see one VerifierStarted +
        // one VerifierFinished frame per finding the pass drives. Skipped
        // findings (no payload/spec) produce no event.
        let (_ws_tmp_a, ws_a) = ws_handle_for("repo-E").await;
        let (_ws_tmp_b, ws_b) = ws_handle_for("repo-F").await;
        let mut workspaces = HashMap::new();
        workspaces.insert("repo-E".to_string(), ws_a);
        workspaces.insert("repo-F".to_string(), ws_b);

        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-E")).await.unwrap();
        store.repos().upsert(&seed_repo("repo-F")).await.unwrap();
        store.runs().insert(&seed_run("run-E")).await.unwrap();

        // Driven finding: has payload + spec.
        let driven = seed_finding("run-E", "repo-E", "src/sink.sh", "rule-sqli");
        let driven_id = driven.id.clone();
        store.findings().upsert(&driven).await.unwrap();
        store.payloads().insert(&seed_payload(&driven_id, b".*", b"^alice$")).await.unwrap();
        store.harness_specs().insert(&seed_spec("spec-E")).await.unwrap();

        // Skipped finding: same run, no payload row.
        let skipped = seed_finding("run-E", "repo-F", "src/other.sh", "rule-orphan");
        store.findings().upsert(&skipped).await.unwrap();

        let (tx, mut rx) = tokio::sync::broadcast::channel(16);
        let report = run_payload_verification_pass(
            &RunConfig::default(),
            &SandboxConfig::default(),
            &store,
            &empty_bundle("run-E"),
            &workspaces,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(report.confirmed, 1);
        assert_eq!(report.skipped_no_payload, 1);

        let mut started = Vec::new();
        let mut finished = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let AgentEvent::Sandbox { data } = ev {
                match data {
                    SandboxEvent::VerifierStarted { finding_id, repo, run_id, .. } => {
                        started.push((run_id, finding_id, repo));
                    }
                    SandboxEvent::VerifierFinished {
                        finding_id,
                        repo,
                        run_id,
                        verdict,
                        replay_stable,
                        elapsed_ms,
                    } => {
                        finished.push((
                            run_id,
                            finding_id,
                            repo,
                            verdict,
                            replay_stable,
                            elapsed_ms,
                        ));
                    }
                }
            }
        }
        assert_eq!(started.len(), 1, "skipped findings must not emit Started");
        assert_eq!(finished.len(), 1);
        assert_eq!(started[0], ("run-E".into(), driven_id.clone(), "repo-E".into()));
        let (run_id, fid, repo, verdict, replay_stable, elapsed_ms) = finished[0].clone();
        assert_eq!(run_id, "run-E");
        assert_eq!(fid, driven_id);
        assert_eq!(repo, "repo-E");
        assert_eq!(verdict, "Confirmed");
        assert!(replay_stable.is_none(), "replay_stable_check is off by default");
        assert!(elapsed_ms >= 0);
    }

    #[test]
    fn derive_oracle_picks_sink_probe_when_prefixed() {
        match derive_oracle("sink-probe:flags/seen.txt") {
            Oracle::SinkProbe { sentinel_path, expect_contains } => {
                assert_eq!(sentinel_path, "flags/seen.txt");
                assert!(expect_contains.is_none());
            }
            other => panic!("expected SinkProbe, got {other:?}"),
        }
        match derive_oracle("sink-probe:flags/seen.txt#leaked") {
            Oracle::SinkProbe { sentinel_path, expect_contains } => {
                assert_eq!(sentinel_path, "flags/seen.txt");
                assert_eq!(expect_contains.as_deref(), Some("leaked"));
            }
            other => panic!("expected SinkProbe, got {other:?}"),
        }
        match derive_oracle("TOP_SECRET") {
            Oracle::OutputContains { marker } => assert_eq!(marker, "TOP_SECRET"),
            other => panic!("expected OutputContains, got {other:?}"),
        }
    }

    #[test]
    fn degenerate_oracle_reason_flags_empty_marker_and_empty_sentinel() {
        assert_eq!(
            degenerate_oracle_reason(&Oracle::OutputContains { marker: String::new() }),
            Some("OutputContains marker is empty"),
        );
        assert_eq!(
            degenerate_oracle_reason(&Oracle::OutputContains { marker: "  ".into() }),
            Some("OutputContains marker is empty"),
        );
        assert_eq!(
            degenerate_oracle_reason(&Oracle::SinkProbe {
                sentinel_path: String::new(),
                expect_contains: None,
            }),
            Some("SinkProbe sentinel_path is empty"),
        );
        assert_eq!(
            degenerate_oracle_reason(&Oracle::SinkProbe {
                sentinel_path: "  ".into(),
                expect_contains: Some("leaked".into()),
            }),
            Some("SinkProbe sentinel_path is empty"),
        );
        assert_eq!(
            degenerate_oracle_reason(&Oracle::OutputContains { marker: "TOP_SECRET".into() }),
            None,
        );
        assert_eq!(
            degenerate_oracle_reason(&Oracle::SinkProbe {
                sentinel_path: "flags/seen.txt".into(),
                expect_contains: None,
            }),
            None,
        );
    }

    /// Scripted agent-loop runtime that mirrors the per-task
    /// fixture. Each `agent_loop` call pops the next outcome off the
    /// back of the queue; `one_shot` returns `UnsupportedMode` because
    /// the exploration pass only drives the agent-loop surface.
    struct ScriptedExplorationRuntime {
        outcomes: StdMutex<Vec<Result<AgentResult, AiError>>>,
        cost_per_call: i64,
        tracker: Arc<dyn BudgetTracker>,
        tasks_seen: StdMutex<Vec<AgentTask>>,
    }

    impl ScriptedExplorationRuntime {
        fn new(
            outcomes: Vec<Result<AgentResult, AiError>>,
            cost_per_call: i64,
            tracker: Arc<dyn BudgetTracker>,
        ) -> Self {
            Self {
                outcomes: StdMutex::new(outcomes),
                cost_per_call,
                tracker,
                tasks_seen: StdMutex::new(Vec::new()),
            }
        }

        fn tasks_seen(&self) -> Vec<AgentTask> {
            self.tasks_seen.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl AiRuntime for ScriptedExplorationRuntime {
        fn name(&self) -> &'static str {
            "scripted-exploration"
        }
        fn default_model(&self) -> &str {
            "scripted-model"
        }
        fn supports_agent_loop(&self) -> bool {
            true
        }
        fn supports_prompt_cache(&self) -> bool {
            false
        }
        fn supports_deterministic_sampling(&self) -> bool {
            false
        }

        async fn one_shot(
            &self,
            _prompt: Prompt,
            _budget: Budget,
            _sink: nyx_agent_types::event::EventSink,
        ) -> Result<Response, AiError> {
            Err(AiError::UnsupportedMode("one_shot"))
        }

        async fn agent_loop(
            &self,
            task: AgentTask,
            budget: Budget,
            _sink: nyx_agent_types::event::EventSink,
        ) -> Result<AgentResult, AiError> {
            self.tasks_seen.lock().unwrap().push(task.clone());
            let mut next = self
                .outcomes
                .lock()
                .unwrap()
                .pop()
                .expect("scripted exploration runtime: no more outcomes");
            let cost = self.cost_per_call;
            self.tracker.add_spend(&budget.run_id, budget.kind, cost).await?;
            if let Ok(ref mut r) = next {
                r.task_id = task.task_id.clone();
                r.cost_usd_micros = cost;
            }
            next
        }

        fn cost_estimate(&self, _prompt: &Prompt) -> Option<CostEstimate> {
            Some(CostEstimate { min_usd_micros: 0, max_usd_micros: self.cost_per_call })
        }
    }

    fn empty_exploration_result() -> AgentResult {
        AgentResult {
            prompt_version: nyx_agent_ai::EXPLORATION_PROMPT_VERSION.to_string(),
            task_id: String::new(),
            model: "scripted-model".to_string(),
            final_message: "ok".to_string(),
            turns: 1,
            usage: TokenUsage { input_tokens: 100, output_tokens: 50 },
            cache: None,
            cost_usd_micros: 0,
            extracted: Vec::new(),
        }
    }

    fn pentest_candidate(
        id: &str,
        source: &str,
        severity: &str,
        status: &str,
        component: serde_json::Value,
    ) -> PentestCandidateRecord {
        PentestCandidateRecord {
            id: id.to_string(),
            run_id: "run-expl-leads".to_string(),
            project_id: "default-project".to_string(),
            source: source.to_string(),
            source_ids: vec![format!("{source}:{id}")],
            title: format!("{source} lead {id}"),
            vuln_class: "AUTH_BYPASS".to_string(),
            severity_guess: severity.to_string(),
            affected_components: vec![component],
            hypothesis: format!("{source} reported {id}; verify with live evidence."),
            test_plan: "Derive a safe live HTTP/browser confirmation.".to_string(),
            status: status.to_string(),
            rejection_reason: None,
            confidence: 0.75,
            trace_id: None,
            created_at: 1_000,
            updated_at: 1_000,
        }
    }

    #[test]
    fn exploration_known_leads_are_repo_scoped_and_prioritised() {
        let candidates = vec![
            pentest_candidate(
                "pc-zap",
                "ZAPBaseline",
                "Medium",
                "NeedsLiveTest",
                serde_json::json!({
                    "scanner": "zap-baseline",
                    "url": "http://localhost:3000/login",
                    "method": "GET"
                }),
            ),
            pentest_candidate(
                "pc-nyx-a",
                "NyxSignal",
                "High",
                "NeedsLiveTest",
                serde_json::json!({
                    "repo": "repo-a",
                    "path": "src/admin.ts",
                    "line": 42
                }),
            ),
            pentest_candidate(
                "pc-nyx-b",
                "NyxSignal",
                "Critical",
                "NeedsLiveTest",
                serde_json::json!({
                    "repo": "repo-b",
                    "path": "src/payments.ts",
                    "line": 7
                }),
            ),
            pentest_candidate(
                "pc-rejected",
                "Nuclei",
                "Critical",
                "Rejected",
                serde_json::json!({
                    "matched_at": "http://localhost:3000/admin"
                }),
            ),
            pentest_candidate(
                "pc-trivy",
                "Trivy",
                "High",
                "Observed",
                serde_json::json!({
                    "repo": "repo-a",
                    "path": "package-lock.json"
                }),
            ),
        ];

        let leads =
            exploration_known_leads_for_repo_with_memory(&candidates, "repo-a", 8, false, &[]);
        assert_eq!(
            leads.len(),
            3,
            "repo-a should see its Nyx lead, Trivy context, plus global ZAP lead"
        );
        assert_eq!(leads[0].id, "pc-nyx-a", "higher severity repo lead should rank first");
        assert_eq!(leads[0].location.as_deref(), Some("repo-a:src/admin.ts:42"));
        assert_eq!(leads[1].source, "Trivy");
        assert_eq!(leads[1].status, "Observed");
        assert_eq!(leads[2].source, "ZAPBaseline");
        assert_eq!(leads[2].location.as_deref(), Some("GET http://localhost:3000/login"));
        assert!(leads.iter().all(|lead| lead.id != "pc-nyx-b"));
        assert!(leads.iter().all(|lead| lead.id != "pc-rejected"));
    }

    #[test]
    fn exploration_scope_includes_prior_memory_in_prompt_focus() {
        let scope = build_exploration_scope(
            "run-memory",
            "repo-a",
            std::path::Path::new("/tmp/repo-a"),
            &["http://localhost:3000".to_string()],
            Vec::new(),
            vec!["prior rejected: admin export [GET /admin/export] - 403 blocked".to_string()],
            100_000,
            1_000_000,
            false,
        );
        assert!(scope
            .research_focus
            .iter()
            .any(|line| line.contains("prior rejected: admin export")));
    }

    #[test]
    fn exploration_known_leads_rank_uses_prior_memory() {
        let rejected = pentest_candidate(
            "pc-rejected-memory",
            "NyxSignal",
            "Critical",
            "NeedsLiveTest",
            serde_json::json!({"repo": "repo-a", "path": "src/admin.ts"}),
        );
        let mut promising = pentest_candidate(
            "pc-promising-memory",
            "NyxSignal",
            "Medium",
            "NeedsLiveTest",
            serde_json::json!({"repo": "repo-a", "path": "src/billing.ts"}),
        );
        promising.hypothesis = "billing export leaks invoices".to_string();
        let memory = vec![ExplorationMemoryRecord {
            id: "mem-1".to_string(),
            project_id: "project".to_string(),
            repo: "repo-a".to_string(),
            run_id: "old-run".to_string(),
            source: "live_verifier".to_string(),
            hypothesis: rejected.hypothesis.clone(),
            endpoint: None,
            role_context: None,
            object_context: None,
            result: "rejected".to_string(),
            reason: "prior verifier rejected the same probe".to_string(),
            useful_markers: Vec::new(),
            auth_session_notes: None,
            follow_up_ideas: Vec::new(),
            candidate_id: None,
            verification_attempt_id: None,
            trace_id: None,
            memory_key: "key".to_string(),
            created_at: 1,
            updated_at: 1,
        }];
        let leads = exploration_known_leads_for_repo_with_memory(
            &[rejected, promising],
            "repo-a",
            8,
            false,
            &memory,
        );
        assert_eq!(leads[0].id, "pc-promising-memory");
    }

    #[test]
    fn attack_planning_prompt_includes_candidate_source_attribution() {
        let candidate = PentestCandidateRecord {
            source: "RouteDiscovery+JavaScriptBundle".to_string(),
            source_ids: vec![
                "RouteDiscovery:api:GET:/api/admin/debug".to_string(),
                "JavaScriptBundle:web:GET:/api/admin/debug".to_string(),
            ],
            affected_components: vec![serde_json::json!({
                "kind": "route",
                "repo": "api",
                "method": "GET",
                "url_path": "/api/admin/debug"
            })],
            ..pentest_candidate(
                "pc-weak-admin",
                "RouteDiscovery",
                "Medium",
                "NeedsLiveTest",
                serde_json::json!({"repo":"api","path":"src/routes.rs","line":42}),
            )
        };
        let prompt = build_attack_planning_prompt(
            &[candidate],
            &HashMap::new(),
            &RouteModel::default(),
            &[],
            &["http://localhost:3000".to_string()],
            false,
        );

        assert!(prompt.user.contains("\"source\": \"RouteDiscovery+JavaScriptBundle\""));
        assert!(prompt.user.contains("JavaScriptBundle:web:GET:/api/admin/debug"));
        assert!(prompt.user.contains("\"confidence\""));
    }

    #[test]
    fn research_mode_prioritizes_research_candidates_in_attack_planning() {
        let research = pentest_candidate(
            "pc-research-entitlement",
            "ResearchMode",
            "Medium",
            "NeedsLiveTest",
            serde_json::json!({
                "kind": "research_mode_hypothesis",
                "repo": "api",
                "method": "POST",
                "url_path": "/api/billing/subscriptions/{id}/downgrade"
            }),
        );
        let critical = pentest_candidate(
            "pc-critical-scanner",
            "Nuclei",
            "Critical",
            "NeedsLiveTest",
            serde_json::json!({
                "matched_at": "http://localhost:3000/debug"
            }),
        );
        let mut normal = vec![research.clone(), critical.clone()];
        prioritize_candidates_for_planning(&mut normal, false);
        assert_eq!(normal[0].id, "pc-critical-scanner");

        let mut research_mode = vec![critical, research];
        prioritize_candidates_for_planning(&mut research_mode, true);
        assert_eq!(research_mode[0].id, "pc-research-entitlement");

        let prompt = build_attack_planning_prompt(
            &research_mode,
            &HashMap::new(),
            &RouteModel::default(),
            &[],
            &["http://localhost:3000".to_string()],
            true,
        );
        assert_eq!(prompt.prompt_version, ATTACK_PLANNING_RESEARCH_PROMPT_VERSION);
        assert_eq!(prompt.max_output_tokens, 5000);
        assert!(prompt.system.contains("Vuln Research Mode"));
        assert!(prompt.user.contains("ResearchMode"));
    }

    #[tokio::test]
    async fn drive_ai_exploration_persists_quarantined_finding() {
        // Exploration acceptance: an AI-discovered finding flows
        // into `findings` with `finding_origin = AiExploration` and
        // `status = Quarantine`.
        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-X")).await.unwrap();
        let mut run_row = seed_run("run-expl-1");
        run_row.id = "run-expl-1".into();
        store.runs().insert(&run_row).await.unwrap();
        let mut scanner_lead = pentest_candidate(
            "pc-zap-expl",
            "ZAPBaseline",
            "Medium",
            "NeedsLiveTest",
            serde_json::json!({
                "scanner": "zap-baseline",
                "url": "http://127.0.0.1:3000/login",
                "method": "GET"
            }),
        );
        scanner_lead.run_id = "run-expl-1".to_string();
        store.pentest_candidates().insert(&scanner_lead).await.unwrap();

        let workspace = tempfile::tempdir().unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert(
            "repo-X".to_string(),
            WorkspaceHandle::for_local_path_test("repo-X", workspace.path().to_path_buf()),
        );

        let mut result = empty_exploration_result();
        result.extracted.push(nyx_agent_types::agent::ExtractedAgentResult::ExplorationFinding {
            path: "<api:/api/admin/orders>".into(),
            line: None,
            cap: "AUTH_BYPASS".into(),
            rationale: "Admin endpoint accepts unauthenticated GET".into(),
            endpoint: Some("GET /api/admin/orders".into()),
            suggested_payload_hint: Some("curl -i http://127.0.0.1:3000/api/admin/orders".into()),
        });

        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-expl-1", BudgetKind::AgentLoop, 10_000_000);
        let runtime = ScriptedExplorationRuntime::new(vec![Ok(result)], 250_000, tracker.clone());

        let bundle = make_bundle("run-expl-1", "repo-X", Vec::new());
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let gate = StaticEscapeSuiteGate::green();

        let traces_root = tempfile::tempdir().unwrap();
        let report = drive_ai_exploration_pass(
            &runtime,
            &store,
            &bundle,
            &workspaces,
            &[],
            &gate,
            tx,
            traces_root.path(),
            DEFAULT_EXPLORATION_SOFT_CAP_USD_MICROS,
            DEFAULT_EXPLORATION_RUN_CAP_USD_MICROS,
            false,
        )
        .await
        .unwrap();

        assert_eq!(report.explorations_dispatched, 1);
        assert_eq!(report.findings_quarantined, 1);
        assert_eq!(report.halted_escape_suite_red, 0);
        assert_eq!(report.halted_budget_exhausted, 0);
        assert_eq!(report.failed, 0);
        assert_eq!(report.spend_usd_micros, 250_000);
        let tasks_seen = runtime.tasks_seen();
        assert_eq!(tasks_seen.len(), 1);
        assert!(tasks_seen[0].objective.contains("KNOWN SCANNER LEADS"));
        assert!(tasks_seen[0].objective.contains("pc-zap-expl"));
        assert!(tasks_seen[0].objective.contains("ZAPBaseline"));
        assert!(tasks_seen[0].objective.contains("GET http://127.0.0.1:3000/login"));

        // The finding landed in the `findings` table with the right
        // origin + status. We do not call list_by_run because the
        // finding's run_id may differ from the bundle's (the persister
        // picks the repo's last_scan_run_id, which is None on a fresh
        // seed). Query by repo via the active-list helper with a
        // quarantine-inclusive filter.
        let filter = nyx_agent_core::store::FindingFilter {
            repo: Some("repo-X"),
            include_quarantine: true,
            ..nyx_agent_core::store::FindingFilter::default()
        };
        let rows = store.findings().list_filtered(&filter).await.unwrap();
        assert_eq!(rows.len(), 1, "expected one quarantined finding, got {}", rows.len());
        let row = &rows[0];
        assert_eq!(row.finding_origin, "AiExploration");
        assert_eq!(row.status, "Quarantine");
        assert_eq!(row.cap, "AUTH_BYPASS");
        assert_eq!(row.path, "<api:/api/admin/orders>");
        assert_eq!(row.attack_provenance.as_deref(), Some("AiExploration"));
        assert_eq!(row.prompt_version.as_deref(), Some(nyx_agent_ai::EXPLORATION_PROMPT_VERSION));
        let blob = row.verdict_blob.as_deref().expect("verdict blob");
        assert!(blob.contains("AiExploration"));
        assert!(blob.contains("unauthenticated"));
    }

    #[tokio::test]
    async fn drive_ai_exploration_research_mode_persists_research_provenance() {
        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-research")).await.unwrap();
        store.runs().insert(&seed_run("run-expl-research")).await.unwrap();

        let workspace = tempfile::tempdir().unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert(
            "repo-research".to_string(),
            WorkspaceHandle::for_local_path_test("repo-research", workspace.path().to_path_buf()),
        );

        let mut result = empty_exploration_result();
        result.prompt_version = nyx_agent_ai::EXPLORATION_RESEARCH_PROMPT_VERSION.to_string();
        result.extracted.push(nyx_agent_types::agent::ExtractedAgentResult::ExplorationFinding {
            path: "<api:/api/billing/subscriptions/1/downgrade>".into(),
            line: None,
            cap: "ENTITLEMENT_MISMATCH".into(),
            rationale: "Downgrade does not update access to paid export job".into(),
            endpoint: Some("POST /api/billing/subscriptions/{id}/downgrade".into()),
            suggested_payload_hint: None,
        });

        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-expl-research", BudgetKind::AgentLoop, 10_000_000);
        let runtime = ScriptedExplorationRuntime::new(vec![Ok(result)], 125_000, tracker.clone());
        let bundle = make_bundle("run-expl-research", "repo-research", Vec::new());
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let gate = StaticEscapeSuiteGate::green();
        let traces_root = tempfile::tempdir().unwrap();

        let report = drive_ai_exploration_pass(
            &runtime,
            &store,
            &bundle,
            &workspaces,
            &["http://127.0.0.1:3000".to_string()],
            &gate,
            tx,
            traces_root.path(),
            DEFAULT_EXPLORATION_SOFT_CAP_USD_MICROS,
            DEFAULT_EXPLORATION_RUN_CAP_USD_MICROS,
            true,
        )
        .await
        .unwrap();

        assert_eq!(report.findings_quarantined, 1);
        let tasks_seen = runtime.tasks_seen();
        assert_eq!(tasks_seen[0].prompt_version, nyx_agent_ai::EXPLORATION_RESEARCH_PROMPT_VERSION);
        assert_eq!(tasks_seen[0].max_turns, 40);
        assert!(tasks_seen[0].system.contains("Vuln Research Mode"));

        let filter = nyx_agent_core::store::FindingFilter {
            repo: Some("repo-research"),
            include_quarantine: true,
            ..nyx_agent_core::store::FindingFilter::default()
        };
        let rows = store.findings().list_filtered(&filter).await.unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(
            row.prompt_version.as_deref(),
            Some(nyx_agent_ai::EXPLORATION_RESEARCH_PROMPT_VERSION)
        );
        let blob = row.verdict_blob.as_deref().expect("verdict blob");
        assert!(blob.contains("\"research_mode\":true"));
        assert!(blob.contains("research_mode_provenance"));
    }

    #[tokio::test]
    async fn drive_ai_exploration_red_gate_halts_with_banner() {
        // Exploration acceptance: a red escape-suite fixture halts
        // the AI driver. The agent loop must not fire (the scripted
        // runtime's
        // queue is empty, so a stray dispatch would panic), and the
        // report counts the halt.
        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-Y")).await.unwrap();
        store.runs().insert(&seed_run("run-expl-2")).await.unwrap();

        let workspace = tempfile::tempdir().unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert(
            "repo-Y".to_string(),
            WorkspaceHandle::for_local_path_test("repo-Y", workspace.path().to_path_buf()),
        );

        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-expl-2", BudgetKind::AgentLoop, 10_000_000);
        // Empty outcomes queue: a dispatched agent_loop would panic.
        let runtime = ScriptedExplorationRuntime::new(Vec::new(), 0, tracker.clone());

        let bundle = make_bundle("run-expl-2", "repo-Y", Vec::new());
        let (tx, mut rx) = tokio::sync::broadcast::channel(8);
        let gate = StaticEscapeSuiteGate::red(
            "write_outside_workspace_is_contained",
            "wrote /tmp/escaped during regression suite",
        );

        let traces_root = tempfile::tempdir().unwrap();
        let report = drive_ai_exploration_pass(
            &runtime,
            &store,
            &bundle,
            &workspaces,
            &[],
            &gate,
            tx,
            traces_root.path(),
            DEFAULT_EXPLORATION_SOFT_CAP_USD_MICROS,
            DEFAULT_EXPLORATION_RUN_CAP_USD_MICROS,
            false,
        )
        .await
        .unwrap();
        assert_eq!(report.halted_escape_suite_red, 1);
        assert_eq!(report.explorations_dispatched, 0);
        assert_eq!(report.findings_quarantined, 0);
        assert_eq!(report.spend_usd_micros, 0);

        // Banner frame on the event bus name-checks the failing fixture.
        let mut saw_banner = false;
        while let Ok(frame) = rx.try_recv() {
            if let nyx_agent_types::event::AgentEvent::Ai {
                data: nyx_agent_types::event::AiEvent::TokenReceived { token, .. },
            } = frame
            {
                if token.contains("escape-suite RED")
                    && token.contains("write_outside_workspace_is_contained")
                {
                    saw_banner = true;
                    break;
                }
            }
        }
        assert!(saw_banner, "expected escape-suite RED banner on the bus");
    }

    #[tokio::test]
    async fn drive_ai_exploration_splits_cost_across_emitted_findings() {
        // The Exploration call cost must be split across the
        // per-finding `agent_traces` rows so the AiTraceViewer's
        // per-finding "Total $..." sums to the proportional share of
        // the call. The parent row (finding_id = NULL) keeps the
        // token/cache metrics but carries cost = 0 to avoid double
        // counting in any join that touches both.
        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-split")).await.unwrap();
        let mut run_row = seed_run("run-split");
        run_row.id = "run-split".into();
        store.runs().insert(&run_row).await.unwrap();

        let workspace = tempfile::tempdir().unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert(
            "repo-split".to_string(),
            WorkspaceHandle::for_local_path_test("repo-split", workspace.path().to_path_buf()),
        );

        let mut result = empty_exploration_result();
        for i in 0..3 {
            result.extracted.push(
                nyx_agent_types::agent::ExtractedAgentResult::ExplorationFinding {
                    path: format!("<api:/api/admin/endpoint-{i}>"),
                    line: None,
                    cap: "AUTH_BYPASS".into(),
                    rationale: format!("Admin endpoint {i} accepts unauthenticated GET"),
                    endpoint: Some(format!("GET /api/admin/endpoint-{i}")),
                    suggested_payload_hint: None,
                },
            );
        }

        // 1_000_001 / 3 = 333_333 with leftover 2; first two rows get
        // 333_334, third gets 333_333. Total stays exact.
        let cost = 1_000_001_i64;
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-split", BudgetKind::AgentLoop, 10_000_000);
        let runtime = ScriptedExplorationRuntime::new(vec![Ok(result)], cost, tracker.clone());

        let bundle = make_bundle("run-split", "repo-split", Vec::new());
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let gate = StaticEscapeSuiteGate::green();

        let traces_root = tempfile::tempdir().unwrap();
        let report = drive_ai_exploration_pass(
            &runtime,
            &store,
            &bundle,
            &workspaces,
            &[],
            &gate,
            tx,
            traces_root.path(),
            DEFAULT_EXPLORATION_SOFT_CAP_USD_MICROS,
            DEFAULT_EXPLORATION_RUN_CAP_USD_MICROS,
            false,
        )
        .await
        .unwrap();
        assert_eq!(report.findings_quarantined, 3);
        assert_eq!(report.spend_usd_micros, cost);

        let parent_rows: Vec<_> = store
            .agent_traces()
            .list_by_task_kind("Exploration")
            .await
            .unwrap()
            .into_iter()
            .filter(|t| t.finding_id.is_none())
            .collect();
        assert_eq!(parent_rows.len(), 1, "expected one parent Exploration trace");
        assert_eq!(
            parent_rows[0].cost_usd_micros, 0,
            "parent cost must be zero when findings split it"
        );
        assert_eq!(parent_rows[0].tokens_in, 100);
        assert_eq!(parent_rows[0].tokens_out, 50);

        let per_finding_rows: Vec<_> = store
            .agent_traces()
            .list_by_task_kind("Exploration")
            .await
            .unwrap()
            .into_iter()
            .filter(|t| t.finding_id.is_some())
            .collect();
        assert_eq!(per_finding_rows.len(), 3, "expected three per-finding rows");
        let total: i64 = per_finding_rows.iter().map(|t| t.cost_usd_micros).sum();
        assert_eq!(total, cost, "per-finding split must sum to the call cost");
        // Token metrics stay on the parent so totals views do not
        // double-count usage when joining both kinds of rows.
        for row in &per_finding_rows {
            assert_eq!(row.tokens_in, 0);
            assert_eq!(row.tokens_out, 0);
        }
    }

    #[tokio::test]
    async fn drive_ai_exploration_keeps_cost_on_parent_when_zero_findings_emitted() {
        // When the call surfaces zero findings, the cost must stay on
        // the parent row so the run-card spend roll-up still observes
        // the spend through its time-window join.
        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-empty")).await.unwrap();
        let mut run_row = seed_run("run-empty");
        run_row.id = "run-empty".into();
        store.runs().insert(&run_row).await.unwrap();

        let workspace = tempfile::tempdir().unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert(
            "repo-empty".to_string(),
            WorkspaceHandle::for_local_path_test("repo-empty", workspace.path().to_path_buf()),
        );

        let result = empty_exploration_result();
        let cost = 250_000_i64;
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-empty", BudgetKind::AgentLoop, 10_000_000);
        let runtime = ScriptedExplorationRuntime::new(vec![Ok(result)], cost, tracker.clone());

        let bundle = make_bundle("run-empty", "repo-empty", Vec::new());
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let gate = StaticEscapeSuiteGate::green();

        let traces_root = tempfile::tempdir().unwrap();
        let report = drive_ai_exploration_pass(
            &runtime,
            &store,
            &bundle,
            &workspaces,
            &[],
            &gate,
            tx,
            traces_root.path(),
            DEFAULT_EXPLORATION_SOFT_CAP_USD_MICROS,
            DEFAULT_EXPLORATION_RUN_CAP_USD_MICROS,
            false,
        )
        .await
        .unwrap();
        assert_eq!(report.findings_quarantined, 0);

        let rows = store.agent_traces().list_by_task_kind("Exploration").await.unwrap();
        assert_eq!(rows.len(), 1, "expected a single parent row, no per-finding rows");
        assert!(rows[0].finding_id.is_none());
        assert_eq!(rows[0].cost_usd_micros, cost);
        // Audit was empty (no extracted items), so no JSONL gets stamped.
        assert!(rows[0].conversation_jsonl_path.is_none());
    }

    #[tokio::test]
    async fn drive_ai_exploration_writes_audit_jsonl_and_stamps_trace_path() {
        // Acceptance: when the agent reports tool invocations, the
        // exploration pass writes one JSONL entry per `AuditEntry` under
        // `<traces_dir>/<run_id>/<task_id>.jsonl` and stamps the path on
        // the parent `agent_traces.conversation_jsonl_path` column.
        let tmp_db = tempfile::tempdir().unwrap();
        let store = Store::open(tmp_db.path()).await.unwrap();
        store.repos().upsert(&seed_repo("repo-audit")).await.unwrap();
        let mut run_row = seed_run("run-audit");
        run_row.id = "run-audit".into();
        store.runs().insert(&run_row).await.unwrap();

        let workspace = tempfile::tempdir().unwrap();
        let mut workspaces = HashMap::new();
        workspaces.insert(
            "repo-audit".to_string(),
            WorkspaceHandle::for_local_path_test("repo-audit", workspace.path().to_path_buf()),
        );

        let mut result = empty_exploration_result();
        // One ExplorationFinding (audit: "record_exploration_finding"),
        // one ExplorationEvent (audit: "<other>"). Two JSONL lines
        // expected.
        result.extracted.push(nyx_agent_types::agent::ExtractedAgentResult::ExplorationFinding {
            path: "<api:/api/users/me>".into(),
            line: None,
            cap: "AUTH_BYPASS".into(),
            rationale: "Endpoint accepts no token".into(),
            endpoint: Some("GET /api/users/me".into()),
            suggested_payload_hint: None,
        });
        result.extracted.push(nyx_agent_types::agent::ExtractedAgentResult::ExplorationEvent {
            message: "probed /api/health and saw 200".into(),
        });

        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-audit", BudgetKind::AgentLoop, 10_000_000);
        let runtime = ScriptedExplorationRuntime::new(vec![Ok(result)], 100_000, tracker.clone());

        let bundle = make_bundle("run-audit", "repo-audit", Vec::new());
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        let gate = StaticEscapeSuiteGate::green();

        let traces_root = tempfile::tempdir().unwrap();
        let report = drive_ai_exploration_pass(
            &runtime,
            &store,
            &bundle,
            &workspaces,
            &[],
            &gate,
            tx,
            traces_root.path(),
            DEFAULT_EXPLORATION_SOFT_CAP_USD_MICROS,
            DEFAULT_EXPLORATION_RUN_CAP_USD_MICROS,
            false,
        )
        .await
        .unwrap();
        assert_eq!(report.findings_quarantined, 1);

        let expected_path = traces_root.path().join("run-audit").join("expl-repo-audit.jsonl");
        assert!(expected_path.exists(), "expected {} to exist", expected_path.display());

        let body = std::fs::read_to_string(&expected_path).expect("read jsonl");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "expected two JSONL audit lines, got {}", lines.len());
        let first: serde_json::Value = serde_json::from_str(lines[0]).expect("line 0 json");
        assert_eq!(first["action"], "record_exploration_finding");
        assert!(first["summary"].as_str().unwrap().contains("<api:/api/users/me>"));
        let second: serde_json::Value = serde_json::from_str(lines[1]).expect("line 1 json");
        assert_eq!(second["action"], "<other>");

        let parent_rows: Vec<_> = store
            .agent_traces()
            .list_by_task_kind("Exploration")
            .await
            .unwrap()
            .into_iter()
            .filter(|t| t.finding_id.is_none())
            .collect();
        assert_eq!(parent_rows.len(), 1, "expected one parent Exploration trace row");
        assert_eq!(
            parent_rows[0].conversation_jsonl_path.as_deref(),
            Some(expected_path.to_string_lossy().as_ref()),
            "parent trace must stamp the JSONL path"
        );

        // Per-finding child rows must NOT stamp the path; the audit log
        // is a per-call artefact, not per-finding.
        let per_finding_rows: Vec<_> = store
            .agent_traces()
            .list_by_task_kind("Exploration")
            .await
            .unwrap()
            .into_iter()
            .filter(|t| t.finding_id.is_some())
            .collect();
        assert_eq!(per_finding_rows.len(), 1);
        assert!(per_finding_rows[0].conversation_jsonl_path.is_none());
    }

    #[test]
    fn attack_agent_risk_downgrades_dev_only_mailbox_proof() {
        let vulnerability = AttackAgentVulnerability {
            title: "Unauthenticated dev mailbox exposes OTPs and enables account takeover in dev"
                .to_string(),
            vuln_class: "DEBUG_EXPOSURE".to_string(),
            severity: "High".to_string(),
            confidence: 99,
            affected_components: vec![serde_json::json!({
                "service": "dev-mailer",
                "endpoint": "GET /api/dev/mail",
            })],
            business_impact:
                "Dev-only mailbox can expose local OTPs, but this service is never in production."
                    .to_string(),
            evidence_summary: "Live local dev app returned mailbox contents.".to_string(),
            repro_steps: "curl http://127.0.0.1:3000/api/dev/mail".to_string(),
            remediation: "Keep the mailbox disabled outside local development.".to_string(),
            source_candidate_ids: Vec::new(),
            source_signal_ids: Vec::new(),
            proof_artifact_paths: Vec::new(),
        };

        let severity = attack_agent_contextual_severity(&vulnerability);
        let risk = attack_agent_risk_score(&severity, 0.99, &vulnerability);
        let rationale = attack_agent_risk_rationale(&vulnerability, &severity, 0.99, risk);

        assert_eq!(severity, "Low");
        assert!(risk < 4.0);
        assert!(rationale.contains("development-only exposure lowers production risk"));
    }

    #[test]
    fn attack_agent_fingerprint_ignores_title_drift_for_same_location() {
        let first = AttackAgentVulnerability {
            title: "Tenant export leaks data".to_string(),
            vuln_class: "ACCESS_CONTROL".to_string(),
            severity: "High".to_string(),
            confidence: 95,
            affected_components: vec![serde_json::json!({
                "endpoint": "GET /api/export/",
            })],
            business_impact: "Tenant data exposure.".to_string(),
            evidence_summary: "Export returned another tenant.".to_string(),
            repro_steps: "GET /api/export".to_string(),
            remediation: "Enforce tenant authorization.".to_string(),
            source_candidate_ids: Vec::new(),
            source_signal_ids: Vec::new(),
            proof_artifact_paths: Vec::new(),
        };
        let second = AttackAgentVulnerability {
            title: "Critical cross-tenant export bypass".to_string(),
            affected_components: vec![serde_json::json!({
                "url": "http://localhost:8787/api/export",
            })],
            ..first.clone()
        };

        assert!(same_attack_location("GET /api/export/", "http://localhost:8787/api/export"));
        assert_eq!(
            attack_vulnerability_fingerprint(&first),
            attack_vulnerability_fingerprint(&second)
        );
    }

    #[test]
    fn attack_agent_replay_extracts_agent_derived_http_workflow() {
        let vulnerability = AttackAgentVulnerability {
            title: "Tenant export bypass".to_string(),
            vuln_class: "Broken Access Control".to_string(),
            severity: "High".to_string(),
            confidence: 95,
            affected_components: vec![serde_json::json!({
                "endpoint": "/api/admin/export",
                "method": "GET",
                "role": "anonymous",
                "expected_result": "customer_email",
            })],
            business_impact: "Anonymous users can export tenant data.".to_string(),
            evidence_summary: "Response included `customer_email`.".to_string(),
            repro_steps: "GET /api/admin/export as:anonymous".to_string(),
            remediation: "Require admin authorization.".to_string(),
            source_candidate_ids: Vec::new(),
            source_signal_ids: Vec::new(),
            proof_artifact_paths: Vec::new(),
        };

        let replay =
            attack_agent_structured_replay_plan(&vulnerability, &["http://localhost:8787".into()]);

        assert_eq!(replay["source"], "agent-derived");
        assert_eq!(replay["status"], "available");
        assert_eq!(replay["policy_enforcement"], "LiveVerifierOptions/ExploitSafetyPolicy");
        assert_eq!(replay["plan"]["kind"], "http_workflow");
        assert_eq!(replay["plan"]["steps"][0]["url"], "http://localhost:8787/api/admin/export");
        assert_eq!(replay["plan"]["steps"][0]["as"], "anonymous");
        assert_eq!(replay["plan"]["oracle"]["body_contains"][0], "customer_email");
    }

    #[test]
    fn attack_agent_replay_marks_unavailable_without_expected_marker() {
        let vulnerability = AttackAgentVulnerability {
            title: "Vague proof".to_string(),
            vuln_class: "Unknown".to_string(),
            severity: "Low".to_string(),
            confidence: 50,
            affected_components: Vec::new(),
            business_impact: "Maybe reachable.".to_string(),
            evidence_summary: "It seemed to work.".to_string(),
            repro_steps: "curl http://localhost:8787/api/thing".to_string(),
            remediation: "Investigate.".to_string(),
            source_candidate_ids: Vec::new(),
            source_signal_ids: Vec::new(),
            proof_artifact_paths: Vec::new(),
        };

        let replay =
            attack_agent_structured_replay_plan(&vulnerability, &["http://localhost:8787".into()]);

        assert_eq!(replay["status"], "replay_unavailable");
        assert!(replay["reason"].as_str().unwrap().contains("expected result"));
    }
}
