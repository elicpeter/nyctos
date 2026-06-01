//! Axum router exposing the loopback HTTP and WebSocket surface.
//!
//! Routes live under `/api/v1/`; the WebSocket lives at
//! `/api/v1/events`. Subscribers connect with an optional
//! `?run_id=<id>` query parameter to filter the broadcast stream to a
//! single run; without the filter every `AgentEvent` lands on the
//! socket.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::panic::AssertUnwindSafe;
use std::path::{Path as FsPath, PathBuf};
use std::time::{Duration, Instant};

use axum::{
    body::{Body, Bytes},
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, Request, State,
    },
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{sse::Event as SseEvent, sse::Sse, IntoResponse, Response},
    routing::{get, patch, post},
    Json, Router,
};
use futures_util::{FutureExt, SinkExt, Stream, StreamExt};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::AsyncReadExt;
use tokio::sync::broadcast::error::RecvError;
use tower_http::trace::TraceLayer;

use nyx_agent_core::report::{
    build_bundle, build_run_card, render_html as render_run_card_html,
    render_markdown as render_run_card_markdown, verify_sha256 as verify_bundle_sha256,
    BundleError, BundleManifest, RunCard, RunCardError,
};
use nyx_agent_core::store::{
    CandidateFindingRecord, CandidateStatus, ChainRecord, FindingFilter, FindingRecord,
    ProjectIntegrationInsert, ProjectIntegrationPatch, ProjectPatch, ProjectPatchOption,
    ProjectRecord, RepoRecord, RunRecord,
};
use nyx_agent_core::{
    now_epoch_ms, parse_git_auth, run_event_log_path, safe_run_log_segment, AiRuntime, IngestError,
    SandboxBackend, ACCOUNT_AI_ANTHROPIC, ACCOUNT_AI_LOCAL_LLM,
};
use nyx_agent_types::api::{
    AgentTraceRow, DoctorCheck, DoctorRequest, DoctorResponse, FindingDiffStatus, FindingWithDiff,
    HealthResponse, QuarantineItem, QuarantineKind, RunFindingsResponse, SetupRequest,
    SetupStatusResponse,
};
use nyx_agent_types::business_logic::{
    business_logic_template_by_id, business_logic_template_metadata, BusinessLogicRunSummary,
    BusinessLogicTemplateMetadata,
};
use nyx_agent_types::event::{AgentEvent, AiEvent, ReproEvent, RunEvent, SandboxEvent};
use nyx_agent_types::integration::{
    CreateProjectIntegrationRequest, PatchProjectIntegrationRequest, ProjectIntegrationRecord,
    TestProjectIntegrationResponse,
};
use nyx_agent_types::product::{
    ProjectLaunchProfile, ProjectLaunchProfileInput, ProjectSetupError,
    ProjectSetupJobListResponse, ProjectSetupJobRecord, ProjectSetupPhase, ProjectSetupRequest,
    ProjectSetupResponse, ProjectSetupStartResponse, ProjectSetupVerification,
    ProjectSetupVerificationStatus, SeedSetupPlan, SeedSetupResponse, StartPentestRequest,
    StartPentestResponse, TestLaunchTargetRequest, TestLaunchTargetResponse,
};
use nyx_agent_types::project::{
    AuthSetupError, AuthSetupJobRecord, AuthSetupPhase, AuthSetupRequest, AuthSetupResponse,
    AuthSetupStartResponse, AuthSetupVerification, AuthSetupVerificationStatus,
    CreateProjectRequest, PatchProjectRequest, ProjectAuthMode, ProjectAuthOwnedObject,
    ProjectAuthProfile, ProjectOtpSourceConfig, ProjectOtpSourceKind, ProjectRuntimeEnvVar,
    ProjectRuntimeProfile, TriStateJson, TriStateProjectRuntimeProfile,
};
use nyx_agent_types::repo::{
    CreateRepoRequest, PatchRepoRequest, TestRepoRequest, TestRepoResponse,
};

use crate::state::{
    ApiError, AuthSetupAgentError, AuthSetupAgentOutput, AuthSetupAgentRequest,
    ProjectSetupAgentError, ProjectSetupAgentRequest, RemediationAgentRequest, RemediationJobError,
    ScanRunOverrides, ScanTriggerSource, SeedSetupAgentError, SeedSetupAgentRequest, ServerState,
};

/// Build the production router with every `/api/v1/...` route attached.
pub fn build_router(state: ServerState) -> Router {
    Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/setup/status", get(setup_status))
        .route("/api/v1/setup", post(submit_setup))
        .route("/api/v1/setup/doctor", post(setup_doctor))
        .route("/api/v1/business-logic/templates", get(business_logic_templates))
        .route("/api/v1/launch-target/test", post(test_launch_target))
        .route("/api/v1/projects", get(list_projects).post(create_project))
        .route(
            "/api/v1/projects/{project_id}",
            get(get_project).patch(patch_project).delete(delete_project),
        )
        .route("/api/v1/projects/{project_id}/auth/auto-setup", post(start_auth_auto_setup_project))
        .route(
            "/api/v1/projects/{project_id}/auth/auto-setup/{job_id}",
            get(get_auth_auto_setup_job),
        )
        .route(
            "/api/v1/projects/{project_id}/setup/ai",
            get(list_ai_project_setup_jobs).post(start_ai_project_setup),
        )
        .route("/api/v1/projects/{project_id}/setup/ai/{job_id}", get(get_ai_project_setup_job))
        .route(
            "/api/v1/projects/{project_id}/repos",
            get(list_project_repos).post(create_project_repo),
        )
        .route("/api/v1/projects/{project_id}/repos/test", post(test_repo_connectivity))
        .route(
            "/api/v1/projects/{project_id}/repos/{name}",
            get(get_project_repo).patch(patch_project_repo).delete(delete_project_repo),
        )
        .route("/api/v1/projects/{project_id}/scan", post(scan_project))
        .route("/api/v1/projects/{project_id}/pentest", post(start_pentest_project))
        .route(
            "/api/v1/projects/{project_id}/integrations",
            get(list_project_integrations).post(create_project_integration),
        )
        .route(
            "/api/v1/projects/{project_id}/integrations/{integration_id}",
            get(get_project_integration)
                .patch(patch_project_integration)
                .delete(delete_project_integration),
        )
        .route(
            "/api/v1/projects/{project_id}/integrations/{integration_id}/test",
            post(test_project_integration),
        )
        .route(
            "/api/v1/projects/{project_id}/launch-profile/default",
            get(get_default_launch_profile).patch(patch_default_launch_profile),
        )
        .route("/api/v1/projects/{project_id}/vulnerabilities", get(project_vulnerabilities))
        .route("/api/v1/runs", get(list_runs))
        .route("/api/v1/runs/{id}", get(get_run))
        .route("/api/v1/runs/{id}/findings", get(findings_for_run))
        .route("/api/v1/runs/{id}/signals", get(signals_for_run))
        .route("/api/v1/runs/{id}/candidates", get(candidates_for_run))
        .route("/api/v1/runs/{id}/route-model", get(route_model_for_run))
        .route("/api/v1/runs/{id}/environment-runs", get(environment_runs_for_run))
        .route("/api/v1/runs/{id}/events.jsonl", get(run_event_log))
        .route("/api/v1/runs/{id}/verification-attempts", get(verification_attempts_for_run))
        .route("/api/v1/runs/{id}/authz-matrix", get(authz_matrix_for_run))
        .route("/api/v1/runs/{id}/exploration-memory", get(exploration_memory_for_run))
        .route("/api/v1/runs/{id}/vulnerabilities", get(run_vulnerabilities))
        .route("/api/v1/runs/{id}/summary", get(run_summary))
        .route("/api/v1/runs/{id}/business-logic", get(run_business_logic))
        .route("/api/v1/runs/{id}/summary.md", get(run_summary_markdown))
        .route("/api/v1/runs/{id}/summary.html", get(run_summary_html))
        .route("/api/v1/findings", get(list_findings))
        .route("/api/v1/vulnerabilities", get(list_vulnerabilities))
        .route("/api/v1/vulnerabilities/status", patch(bulk_update_vulnerability_status))
        .route("/api/v1/vulnerabilities/{id}", get(get_vulnerability))
        .route("/api/v1/vulnerabilities/{id}/fix", post(start_vulnerability_fix))
        .route("/api/v1/vulnerabilities/{id}/fix/{job_id}", get(get_vulnerability_fix_job))
        .route("/api/v1/vulnerabilities/{id}/status", patch(update_vulnerability_status))
        .route("/api/v1/findings/{id}", get(get_finding))
        .route("/api/v1/findings/{id}/repro-bundle", post(create_repro_bundle))
        .route("/api/v1/findings/{id}/repro-bundle.tar", get(download_repro_bundle))
        .route("/api/v1/findings/{id}/replay", post(replay_repro_bundle))
        .route("/api/v1/chains", get(list_chains))
        .route("/api/v1/chains/{id}", get(get_chain))
        .route("/api/v1/findings/{id}/traces", get(traces_for_finding))
        .route("/api/v1/traces/{id}", get(get_trace))
        .route("/api/v1/quarantine", get(list_quarantine))
        .route("/api/v1/quarantine/{id}/promote", post(promote_quarantine))
        .route("/api/v1/quarantine/{id}/dismiss", post(dismiss_quarantine))
        .route("/api/v1/events", get(events_ws))
        .route("/webhook/git", post(crate::webhook::webhook_git))
        .layer(middleware::from_fn_with_state(state.clone(), auth_layer))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn business_logic_templates() -> Json<Vec<BusinessLogicTemplateMetadata>> {
    Json(business_logic_template_metadata())
}

/// Bearer-token gate. Skipped entirely when [`AuthConfig::token`] is
/// unset (the `--headless` path), and skipped on a per-route basis for
/// `/health` plus the read-only setup status endpoint. The mutating
/// wizard endpoints stay open only while setup is still pending. Once
/// `nyx-agent.toml` exists, setup writes require the bearer token like
/// every other mutation endpoint so an attacker cannot overwrite the
/// operator's config.
async fn auth_layer(
    State(state): State<ServerState>,
    req: Request,
    next: Next,
) -> Result<Response, ApiError> {
    if !state.auth.is_enforced() {
        return Ok(next.run(req).await);
    }
    let path = req.uri().path();
    if is_always_open(path) {
        return Ok(next.run(req).await);
    }
    if is_setup_status_path(path) {
        return Ok(next.run(req).await);
    }
    if is_setup_path(path) && !state.setup.is_complete() {
        return Ok(next.run(req).await);
    }
    let token = state.auth.token.as_deref().unwrap_or_default();
    if check_bearer(&req, token) || check_query_token(&req, token) {
        return Ok(next.run(req).await);
    }
    Err(ApiError::Unauthorized)
}

fn is_always_open(path: &str) -> bool {
    // `/webhook/git` carries its own HMAC auth so bypass the bearer
    // gate; the handler refuses on bad signature.
    path == "/api/v1/health" || path == "/webhook/git"
}

fn is_setup_path(path: &str) -> bool {
    matches!(path, "/api/v1/setup" | "/api/v1/setup/status" | "/api/v1/setup/doctor")
}

fn is_setup_status_path(path: &str) -> bool {
    path == "/api/v1/setup/status"
}

fn check_bearer(req: &Request, expected: &str) -> bool {
    let Some(value) = req.headers().get(axum::http::header::AUTHORIZATION) else {
        return false;
    };
    let Ok(text) = value.to_str() else { return false };
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("Bearer ") else { return false };
    constant_eq(rest.trim(), expected)
}

fn check_query_token(req: &Request, expected: &str) -> bool {
    let Some(q) = req.uri().query() else { return false };
    for pair in q.split('&') {
        if let Some(rest) = pair.strip_prefix("token=") {
            let decoded = urlencoded_decode(rest);
            if constant_eq(&decoded, expected) {
                return true;
            }
        }
    }
    false
}

fn constant_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.bytes().zip(b.bytes()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn urlencoded_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = hex_digit(bytes[i + 1]);
                let lo = hex_digit(bytes[i + 2]);
                if let (Some(h), Some(l)) = (hi, lo) {
                    out.push((h * 16 + l) as char);
                    i += 3;
                } else {
                    out.push(bytes[i] as char);
                    i += 1;
                }
            }
            b => {
                out.push(b as char);
                i += 1;
            }
        }
    }
    out
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn test_launch_target(
    Json(req): Json<TestLaunchTargetRequest>,
) -> Result<Json<TestLaunchTargetResponse>, ApiError> {
    let raw = req.url.trim();
    let url = local_http_url(raw).ok_or_else(|| {
        ApiError::BadRequest(
            "app URL must be local http:// or https:// (localhost, 127.0.0.1, or ::1)".to_string(),
        )
    })?;
    let timeout = Duration::from_secs(req.timeout_seconds.unwrap_or(3).clamp(1, 15));
    let started = Instant::now();
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| ApiError::Internal(format!("build URL test client: {e}")))?;

    let response = match client.get(url.clone()).send().await {
        Ok(resp) => {
            let status = resp.status();
            let ok = status.is_success();
            TestLaunchTargetResponse {
                ok,
                url: url.to_string(),
                message: if ok {
                    format!("Reachable in {}ms", started.elapsed().as_millis())
                } else {
                    format!("Responded with HTTP {}", status.as_u16())
                },
                status: Some(status.as_u16()),
                elapsed_ms: millis_u64(started.elapsed()),
            }
        }
        Err(err) => TestLaunchTargetResponse {
            ok: false,
            url: url.to_string(),
            message: if err.is_timeout() {
                format!("Timed out after {}s", timeout.as_secs())
            } else {
                format!("Could not reach app: {err}")
            },
            status: None,
            elapsed_ms: millis_u64(started.elapsed()),
        },
    };

    Ok(Json(response))
}

fn millis_u64(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

// ---- /setup -----------------------------------------------------------------

async fn setup_status(State(s): State<ServerState>) -> Result<Json<SetupStatusResponse>, ApiError> {
    let cfg = s.setup.config.read().await;
    Ok(Json(SetupStatusResponse {
        complete: s.setup.is_complete(),
        config_path: s.setup.config_path.display().to_string(),
        ai_runtime: ai_runtime_label(cfg.ai.runtime).to_string(),
        ai_provider: cfg.ai.provider.clone(),
        ai_model: cfg.ai.model.clone(),
        ai_effort: cfg.ai.effort.clone(),
        ai_context_window: cfg.ai.context_window,
        ai_api_base: cfg.ai.api_base.clone(),
        default_run_budget_usd_micros: cfg.ai.default_run_budget_usd_micros,
        sandbox_backend: sandbox_backend_label(cfg.sandbox.backend).to_string(),
        sandbox_enabled: cfg.sandbox.enabled,
        sandbox_allow_network: cfg.sandbox.allow_network,
        ui_listen_addr: cfg.ui.listen_addr.clone(),
        ui_open_browser: cfg.ui.open_browser,
        log_level: cfg.general.log_level.clone(),
        state_dir: cfg.general.state_dir.as_ref().map(|p| p.display().to_string()),
        max_parallel_scans: cfg.performance.max_parallel_scans,
        scan_timeout_secs: cfg.performance.scan_timeout_secs,
    }))
}

fn ai_runtime_label(r: AiRuntime) -> &'static str {
    match r {
        AiRuntime::None => "none",
        AiRuntime::Anthropic => "anthropic",
        AiRuntime::LocalLlm => "local-llm",
        AiRuntime::ClaudeCode => "claude-code",
        AiRuntime::Codex => "codex",
    }
}

fn sandbox_backend_label(b: SandboxBackend) -> &'static str {
    match b {
        SandboxBackend::Auto => "auto",
        SandboxBackend::Process => "process",
        SandboxBackend::Birdcage => "birdcage",
        SandboxBackend::Libkrun => "libkrun",
        SandboxBackend::Firecracker => "firecracker",
        SandboxBackend::Docker => "docker",
    }
}

#[derive(Debug, Serialize)]
struct SetupResponse {
    ok: bool,
    config_path: String,
}

async fn submit_setup(
    State(s): State<ServerState>,
    Json(req): Json<SetupRequest>,
) -> Result<Json<SetupResponse>, ApiError> {
    if !req.i_own_this {
        return Err(ApiError::BadRequest(
            "i_own_this must be true before the daemon will write a config".to_string(),
        ));
    }

    let ai_runtime = parse_ai_runtime(&req.ai_runtime)?;
    let sandbox_backend = parse_sandbox_backend(&req.sandbox_backend)?;
    let ai_model = parse_optional_string_patch(req.ai_model);
    let ai_effort = parse_optional_effort_patch(req.ai_effort, ai_runtime)?;
    let ai_context_window = parse_optional_context_window_patch(req.ai_context_window)?;
    let default_run_budget_usd_micros = parse_optional_positive_micros(
        req.default_run_budget_usd_micros,
        "default_run_budget_usd_micros",
    )?;
    let mut cfg = s.setup.config.read().await.clone();
    let anthropic_api_key =
        req.anthropic_api_key.as_deref().map(str::trim).filter(|v| !v.is_empty());
    let local_llm_url = req.local_llm_url.as_deref().map(str::trim).filter(|v| !v.is_empty());

    if matches!(ai_runtime, AiRuntime::Anthropic) && anthropic_api_key.is_none() {
        let has_existing_key = s
            .setup
            .secrets
            .get(ACCOUNT_AI_ANTHROPIC)
            .map_err(|e| ApiError::Internal(format!("read Anthropic key: {e}")))?
            .is_some();
        if !has_existing_key {
            return Err(ApiError::BadRequest(
                "anthropic_api_key is required when ai_runtime = \"anthropic\"".to_string(),
            ));
        }
    }
    if matches!(ai_runtime, AiRuntime::LocalLlm) && local_llm_url.is_none() {
        let missing_existing_url =
            cfg.ai.api_base.as_deref().map(str::trim).unwrap_or("").is_empty();
        if missing_existing_url {
            return Err(ApiError::BadRequest(
                "local_llm_url is required when ai_runtime = \"local-llm\"".to_string(),
            ));
        }
    }

    // Persist secrets first so a failure there does not orphan a
    // half-written config file. The keychain may legitimately reject
    // calls in non-interactive environments (e.g. CI); surface that as
    // a 500 with the precise reason.
    if let Some(key) = anthropic_api_key {
        s.setup
            .secrets
            .set(ACCOUNT_AI_ANTHROPIC, key)
            .map_err(|e| ApiError::Internal(format!("store Anthropic key: {e}")))?;
    } else if matches!(
        ai_runtime,
        AiRuntime::None | AiRuntime::LocalLlm | AiRuntime::ClaudeCode | AiRuntime::Codex
    ) {
        let _ = s.setup.secrets.delete(ACCOUNT_AI_ANTHROPIC);
    }
    if let Some(tok) = req.local_llm_token.as_deref().filter(|v| !v.trim().is_empty()) {
        s.setup
            .secrets
            .set(ACCOUNT_AI_LOCAL_LLM, tok.trim())
            .map_err(|e| ApiError::Internal(format!("store local-llm token: {e}")))?;
    } else if !matches!(ai_runtime, AiRuntime::LocalLlm) {
        let _ = s.setup.secrets.delete(ACCOUNT_AI_LOCAL_LLM);
    }

    cfg.ai.runtime = ai_runtime;
    if let Some(next) = ai_model {
        cfg.ai.model = next;
    }
    if let Some(next) = ai_effort {
        cfg.ai.effort = next;
    }
    if let Some(next) = ai_context_window {
        cfg.ai.context_window = next;
    }
    cfg.ai.provider = match ai_runtime {
        AiRuntime::None => None,
        AiRuntime::Anthropic => Some("anthropic".to_string()),
        AiRuntime::LocalLlm => Some("local-llm".to_string()),
        AiRuntime::ClaudeCode => Some("claude-code".to_string()),
        AiRuntime::Codex => Some("codex".to_string()),
    };
    cfg.ai.api_base = match ai_runtime {
        AiRuntime::LocalLlm => {
            local_llm_url.map(str::to_string).or_else(|| cfg.ai.api_base.clone())
        }
        _ => cfg.ai.api_base.clone(),
    };
    cfg.ai.default_run_budget_usd_micros = default_run_budget_usd_micros;
    cfg.sandbox.backend = sandbox_backend;

    let rendered =
        cfg.to_toml_string().map_err(|e| ApiError::Internal(format!("render toml: {e}")))?;
    write_config_atomic(&s.setup.config_path, &rendered)
        .map_err(|e| ApiError::Internal(format!("write {}: {e}", s.setup.config_path.display())))?;
    *s.setup.config.write().await = cfg;
    s.setup.mark_complete();
    Ok(Json(SetupResponse { ok: true, config_path: s.setup.config_path.display().to_string() }))
}

fn parse_ai_runtime(raw: &str) -> Result<AiRuntime, ApiError> {
    match raw.trim() {
        "none" => Ok(AiRuntime::None),
        "anthropic" => Ok(AiRuntime::Anthropic),
        "local-llm" => Ok(AiRuntime::LocalLlm),
        "claude-code" => Ok(AiRuntime::ClaudeCode),
        "codex" => Ok(AiRuntime::Codex),
        other => Err(ApiError::BadRequest(format!("unknown ai_runtime `{other}`"))),
    }
}

fn parse_optional_positive_micros(raw: Option<i64>, field: &str) -> Result<Option<i64>, ApiError> {
    match raw {
        Some(v) if v <= 0 => {
            Err(ApiError::BadRequest(format!("{field} must be a positive integer or null")))
        }
        other => Ok(other),
    }
}

fn parse_optional_string_patch(raw: Option<Option<String>>) -> Option<Option<String>> {
    raw.map(|value| value.and_then(|v| non_empty_trimmed(&v).map(str::to_string)))
}

fn parse_optional_effort_patch(
    raw: Option<Option<String>>,
    ai_runtime: AiRuntime,
) -> Result<Option<Option<String>>, ApiError> {
    let Some(value) = raw else {
        return Ok(None);
    };
    let Some(raw_effort) = value else {
        return Ok(Some(None));
    };
    let Some(trimmed) = non_empty_trimmed(&raw_effort) else {
        return Ok(Some(None));
    };
    let effort = trimmed.to_ascii_lowercase();
    let valid = matches!(effort.as_str(), "low" | "medium" | "high" | "xhigh" | "max");
    if !valid {
        return Err(ApiError::BadRequest(format!("unknown ai_effort `{trimmed}`")));
    }
    if effort == "max" && matches!(ai_runtime, AiRuntime::Codex) {
        return Err(ApiError::BadRequest(
            "ai_effort = \"max\" is only supported by the Claude Code runtime".to_string(),
        ));
    }
    Ok(Some(Some(effort)))
}

fn parse_optional_context_window_patch(
    raw: Option<Option<u32>>,
) -> Result<Option<Option<u32>>, ApiError> {
    match raw {
        Some(Some(0)) => {
            Err(ApiError::BadRequest("ai_context_window must be positive or null".to_string()))
        }
        other => Ok(other),
    }
}

fn non_empty_trimmed(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn parse_sandbox_backend(raw: &str) -> Result<SandboxBackend, ApiError> {
    match raw.trim() {
        "auto" => Ok(SandboxBackend::Auto),
        "process" => Ok(SandboxBackend::Process),
        "birdcage" => Ok(SandboxBackend::Birdcage),
        "libkrun" => Ok(SandboxBackend::Libkrun),
        "firecracker" => Ok(SandboxBackend::Firecracker),
        "docker" => Ok(SandboxBackend::Docker),
        other => Err(ApiError::BadRequest(format!("unknown sandbox_backend `{other}`"))),
    }
}

fn write_config_atomic(path: &std::path::Path, body: &str) -> std::io::Result<()> {
    use std::io::Write;
    let parent = path.parent().unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent)?;
    let tmp = path.with_extension("toml.tmp");
    {
        let mut f =
            std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(&tmp)?;
        f.write_all(body.as_bytes())?;
        f.flush()?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp, path)
}

/// Lightweight check pass invoked by the wizard's step 3 to surface
/// problems before the operator commits a config. Reports a list of
/// per-check results rather than a single pass/fail so the UI can
/// render targeted remediation hints.
async fn setup_doctor(
    State(s): State<ServerState>,
    Json(req): Json<DoctorRequest>,
) -> Result<Json<DoctorResponse>, ApiError> {
    let mut checks = Vec::new();
    checks.push(DoctorCheck {
        name: "state-dir".to_string(),
        passed: s.setup.config_path.parent().is_some(),
        message: "state directory writable".to_string(),
    });
    let ai_runtime = parse_ai_runtime(&req.ai_runtime)?;
    match ai_runtime {
        AiRuntime::None => checks.push(DoctorCheck {
            name: "ai".to_string(),
            passed: true,
            message: "AI disabled: static pass only".to_string(),
        }),
        AiRuntime::Anthropic => checks.push(anthropic_doctor_check(&s, &req)),
        AiRuntime::LocalLlm => checks.push(local_llm_doctor_check(&s, &req).await),
        AiRuntime::ClaudeCode => {
            let found = which_on_path("claude");
            checks.push(DoctorCheck {
                name: "ai-claude-code".to_string(),
                passed: found.is_some(),
                message: match found {
                    Some(p) => format!(
                        "Claude Code binary found at {p}; optional local CLI adapter enabled. Use provider-authorized credentials; Nyx Agent does not include or resell model access."
                    ),
                    None => "`claude` not found on PATH; install Claude Code only if you want the optional local CLI adapter".to_string(),
                },
            });
        }
        AiRuntime::Codex => checks.push(codex_doctor_check().await),
    }

    let sandbox_backend = parse_sandbox_backend(&req.sandbox_backend)?;
    let (sandbox_pass, sandbox_msg) = sandbox_backend_probe(sandbox_backend);
    checks.push(DoctorCheck {
        name: "sandbox".to_string(),
        passed: sandbox_pass,
        message: sandbox_msg,
    });

    let (bin_pass, bin_msg) = binary_lane_probe();
    checks.push(DoctorCheck {
        name: "binary-target-lane".to_string(),
        passed: bin_pass,
        message: bin_msg,
    });

    Ok(Json(DoctorResponse { checks }))
}

fn anthropic_doctor_check(s: &ServerState, req: &DoctorRequest) -> DoctorCheck {
    let provided = req.anthropic_api_key.as_deref().map(str::trim).is_some_and(|v| !v.is_empty());
    if provided {
        return DoctorCheck {
            name: "ai-anthropic".to_string(),
            passed: true,
            message: "Anthropic API key provided for this check; save settings to store it"
                .to_string(),
        };
    }

    match s.setup.secrets.get(ACCOUNT_AI_ANTHROPIC) {
        Ok(Some(_)) => DoctorCheck {
            name: "ai-anthropic".to_string(),
            passed: true,
            message: "Anthropic API key found in the OS keychain".to_string(),
        },
        Ok(None) => DoctorCheck {
            name: "ai-anthropic".to_string(),
            passed: false,
            message: "Anthropic API key is not set; enter one before saving this runtime"
                .to_string(),
        },
        Err(e) => DoctorCheck {
            name: "ai-anthropic".to_string(),
            passed: false,
            message: format!("Could not read Anthropic API key from the OS keychain: {e}"),
        },
    }
}

async fn local_llm_doctor_check(s: &ServerState, req: &DoctorRequest) -> DoctorCheck {
    let provided_url = req.local_llm_url.as_deref().map(str::trim).filter(|v| !v.is_empty());
    let configured_url = if provided_url.is_none() {
        let cfg = s.setup.config.read().await;
        cfg.ai.api_base.clone()
    } else {
        None
    };
    let url = provided_url.or_else(|| configured_url.as_deref().map(str::trim));
    match url.filter(|v| !v.is_empty()) {
        Some(url) => DoctorCheck {
            name: "ai-local-llm".to_string(),
            passed: true,
            message: format!(
                "Local OpenAI-compatible endpoint configured at {url}; one-shot helpers enabled. Set [ai].model if the server requires a specific model id."
            ),
        },
        None => DoctorCheck {
            name: "ai-local-llm".to_string(),
            passed: false,
            message: "Local LLM endpoint is not set; enter a /v1 URL before saving this runtime"
                .to_string(),
        },
    }
}

async fn codex_doctor_check() -> DoctorCheck {
    let Some(path) = which_on_path("codex") else {
        return DoctorCheck {
            name: "ai-codex".to_string(),
            passed: false,
            message: "`codex` not found on PATH; install Codex CLI only if you want the optional local CLI adapter".to_string(),
        };
    };

    let mut cmd = tokio::process::Command::new(&path);
    cmd.arg("doctor")
        .arg("--json")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let output = match tokio::time::timeout(Duration::from_secs(5), cmd.output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(err)) => {
            return DoctorCheck {
                name: "ai-codex".to_string(),
                passed: false,
                message: format!("Codex binary found at {path}, but doctor failed to run: {err}"),
            };
        }
        Err(_) => {
            return DoctorCheck {
                name: "ai-codex".to_string(),
                passed: false,
                message: format!("Codex binary found at {path}, but doctor timed out"),
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = serde_json::from_str::<serde_json::Value>(&stdout);
    let Ok(report) = parsed else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = if stderr.trim().is_empty() { stdout.trim() } else { stderr.trim() };
        return DoctorCheck {
            name: "ai-codex".to_string(),
            passed: false,
            message: format!(
                "Codex binary found at {path}, but doctor did not return JSON: {detail}"
            ),
        };
    };

    let version = report.get("codexVersion").and_then(|v| v.as_str()).unwrap_or("unknown version");
    let overall = report.get("overallStatus").and_then(|v| v.as_str()).unwrap_or("unknown");
    let auth = doctor_check_status(&report, "auth.credentials");
    let install = doctor_check_status(&report, "installation");
    let runtime = doctor_check_status(&report, "runtime.provenance");
    let passed = matches!(auth, Some("ok")) && matches!(install, Some("ok"));
    let auth_msg = match auth {
        Some("ok") => "auth configured",
        Some(other) => other,
        None => "auth status unavailable",
    };
    let runtime_msg = match runtime {
        Some("ok") => "runtime healthy",
        Some(other) => other,
        None => "runtime status unavailable",
    };
    DoctorCheck {
        name: "ai-codex".to_string(),
        passed,
        message: format!(
            "Codex CLI {version} found at {path}; {auth_msg}; {runtime_msg}; doctor overall {overall}; optional local CLI adapter enabled. Use provider-authorized credentials."
        ),
    }
}

fn doctor_check_status<'a>(report: &'a serde_json::Value, id: &str) -> Option<&'a str> {
    report.get("checks")?.get(id)?.get("status")?.as_str()
}

fn which_on_path(bin: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for entry in std::env::split_paths(&path) {
        let candidate = entry.join(bin);
        if candidate.is_file() {
            return Some(candidate.display().to_string());
        }
    }
    None
}

/// Can the binary-target pentest lane run on this host? It runs on the
/// Chain lane and refuses the unisolated `process` backend (§8.2), so it
/// needs at least birdcage and prefers libkrun/firecracker/docker.
fn binary_lane_probe() -> (bool, String) {
    let selection = nyx_agent_sandbox::select_backend(
        nyx_agent_sandbox::BackendChoice::Auto,
        nyx_agent_sandbox::Lane::Chain,
    );
    if selection.backend == nyx_agent_sandbox::BackendKind::Process {
        (
            false,
            "no isolation backend stronger than `process` is available; the binary-target pass \
             will refuse to run. Install libkrun/firecracker/docker, or enable birdcage."
                .to_string(),
        )
    } else {
        (
            true,
            format!(
                "binary-target lane ready on {} ({})",
                selection.backend.as_str(),
                selection.reason
            ),
        )
    }
}

fn sandbox_backend_probe(b: SandboxBackend) -> (bool, String) {
    // Auto stays advisory because the chosen backend depends on the lane
    // (chain vs fast) and is resolved at scan dispatch time. Every other
    // variant routes through `nyx_agent_sandbox::probe` so the wizard's
    // readiness tile shares its source of truth with the doctor and the
    // run-time auto-selector.
    if matches!(b, SandboxBackend::Auto) {
        return (true, "Backend will be chosen at scan time".to_string());
    }
    let kind = match b {
        SandboxBackend::Process => nyx_agent_sandbox::BackendKind::Process,
        SandboxBackend::Birdcage => nyx_agent_sandbox::BackendKind::Birdcage,
        SandboxBackend::Libkrun => nyx_agent_sandbox::BackendKind::Libkrun,
        SandboxBackend::Firecracker => nyx_agent_sandbox::BackendKind::Firecracker,
        SandboxBackend::Docker => nyx_agent_sandbox::BackendKind::Docker,
        SandboxBackend::Auto => unreachable!("Auto handled above"),
    };
    match nyx_agent_sandbox::probe(kind) {
        Ok(()) => (true, format!("{} ready on this host", kind.as_str())),
        Err(err) => (false, err.to_string()),
    }
}

// ---- /projects --------------------------------------------------------------

async fn list_projects(State(s): State<ServerState>) -> Result<Json<Vec<ProjectRecord>>, ApiError> {
    let rows = s.store.projects().list().await?;
    Ok(Json(rows))
}

async fn create_project(
    State(s): State<ServerState>,
    Json(req): Json<CreateProjectRequest>,
) -> Result<Json<ProjectRecord>, ApiError> {
    let name = req.name.trim();
    if name.is_empty() {
        return Err(ApiError::BadRequest("name is required".to_string()));
    }
    if s.store.projects().get_by_name(name).await?.is_some() {
        return Err(ApiError::BadRequest(format!("project `{name}` already exists")));
    }
    let id = format!("proj-{}", uuid_like(name, now_epoch_ms()));
    let env_config_json = match req.env_config.as_ref() {
        Some(v) => Some(serde_json::to_string(v).map_err(|e| {
            ApiError::BadRequest(format!("env_config must serialize to JSON: {e}"))
        })?),
        None => None,
    };
    let mut runtime_profile = req.runtime_profile;
    let target_base_url =
        normalize_create_target_base_url(req.target_base_url, &mut runtime_profile)?;
    let launch_profile = req.default_launch_profile.or_else(|| {
        runtime_profile
            .as_ref()
            .map(|profile| launch_profile_input_from_runtime(profile, target_base_url.as_deref()))
    });
    let runtime_profile_json = match runtime_profile.as_ref() {
        Some(v) => Some(serde_json::to_string(v).map_err(|e| {
            ApiError::BadRequest(format!("runtime_profile must serialize to JSON: {e}"))
        })?),
        None => None,
    };
    let _rec = s
        .store
        .projects()
        .create_with_runtime_profile(
            &id,
            name,
            req.description.as_deref(),
            target_base_url.as_deref(),
            env_config_json.as_deref(),
            runtime_profile_json.as_deref(),
            now_epoch_ms(),
        )
        .await?;
    if let Some(input) = launch_profile.as_ref() {
        s.store.launch_profiles().upsert_default(&id, input, now_epoch_ms()).await?;
    }
    let rec = s
        .store
        .projects()
        .get(&id)
        .await?
        .ok_or_else(|| ApiError::Internal("project vanished after create".to_string()))?;
    Ok(Json(rec))
}

async fn get_project(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<ProjectRecord>, ApiError> {
    s.store
        .projects()
        .get(&id)
        .await?
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("project `{id}` not found")))
}

async fn patch_project(
    State(s): State<ServerState>,
    Path(id): Path<String>,
    Json(req): Json<PatchProjectRequest>,
) -> Result<Json<ProjectRecord>, ApiError> {
    // Serialise the optional env_config value once so the patch borrow
    // can reference an owned String that outlives the call.
    let owned_env_json: Option<String> = match &req.env_config {
        TriStateJson::Value(v) => Some(serde_json::to_string(v).map_err(|e| {
            ApiError::BadRequest(format!("env_config must serialize to JSON: {e}"))
        })?),
        _ => None,
    };
    let env_config_patch: ProjectPatchOption<Option<String>> = match &req.env_config {
        TriStateJson::Unset => ProjectPatchOption::Unset,
        TriStateJson::Null => ProjectPatchOption::Set(None),
        TriStateJson::Value(_) => ProjectPatchOption::Set(owned_env_json),
    };
    let mut target_base_url_patch = project_patch_for(&req.target_base_url);
    let mut launch_profile_from_runtime: Option<ProjectLaunchProfileInput> = None;
    let runtime_profile_patch: ProjectPatchOption<Option<String>> = match req.runtime_profile {
        TriStateProjectRuntimeProfile::Unset => ProjectPatchOption::Unset,
        TriStateProjectRuntimeProfile::Null => ProjectPatchOption::Set(None),
        TriStateProjectRuntimeProfile::Value(mut profile) => {
            match &req.target_base_url {
                Some(Some(target)) => {
                    let target = normalize_optional_string(Some(target.as_str()));
                    if let (Some(profile_target), Some(top_level_target)) = (
                        normalize_optional_string(profile.target_base_url.as_deref()),
                        target.as_deref(),
                    ) {
                        if profile_target != top_level_target {
                            return Err(ApiError::BadRequest(
                                "runtime_profile.target_base_url must match target_base_url"
                                    .to_string(),
                            ));
                        }
                    }
                    profile.target_base_url = target;
                }
                Some(None) => {
                    profile.target_base_url = None;
                }
                None => {
                    if let Some(profile_target) =
                        normalize_optional_string(profile.target_base_url.as_deref())
                    {
                        target_base_url_patch = ProjectPatchOption::Set(Some(profile_target));
                    }
                }
            }
            let runtime_profile_json = serde_json::to_string(&profile).map_err(|e| {
                ApiError::BadRequest(format!("runtime_profile must serialize to JSON: {e}"))
            })?;
            let target = match &req.target_base_url {
                Some(Some(value)) => Some(value.as_str()),
                _ => profile.target_base_url.as_deref(),
            };
            launch_profile_from_runtime = Some(launch_profile_input_from_runtime(&profile, target));
            ProjectPatchOption::Set(Some(runtime_profile_json))
        }
    };
    let now = now_epoch_ms();
    let patch = ProjectPatch {
        description: project_patch_for(&req.description),
        target_base_url: target_base_url_patch,
        env_config_json: env_config_patch,
        runtime_profile_json: runtime_profile_patch,
        updated_at: now,
    };
    if !s.store.projects().update(&id, &patch).await? {
        return Err(ApiError::NotFound(format!("project `{id}` not found")));
    }
    if let Some(input) = launch_profile_from_runtime.as_ref() {
        s.store.launch_profiles().upsert_default(&id, input, now).await?;
    }
    let row = s
        .store
        .projects()
        .get(&id)
        .await?
        .ok_or_else(|| ApiError::Internal("project vanished after update".to_string()))?;
    Ok(Json(row))
}

async fn start_auth_auto_setup_project(
    State(s): State<ServerState>,
    Path(id): Path<String>,
    Json(req): Json<AuthSetupRequest>,
) -> Result<Json<AuthSetupStartResponse>, ApiError> {
    let project = s
        .store
        .projects()
        .get(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("project `{id}` not found")))?;
    let target_base_url = auth_setup_target_base_url(&project, req.target_base_url.as_deref());
    if let Some(url) = target_base_url.as_deref() {
        if !is_local_http_url(url) {
            return Err(ApiError::BadRequest(format!("target URL `{url}` must be local")));
        }
    }

    let job = s.auth_setup_jobs.create(&id, now_epoch_ms()).await;
    let job_id = job.id.clone();
    let state = s.clone();
    tokio::spawn(async move {
        let panic_state = state.clone();
        let panic_job_id = job_id.clone();
        let result =
            AssertUnwindSafe(run_auth_auto_setup_job(state, id, req, job_id)).catch_unwind().await;
        if let Err(payload) = result {
            let detail = panic_payload_message(payload.as_ref());
            tracing::error!(job_id = %panic_job_id, %detail, "auth setup job panicked");
            panic_state
                .auth_setup_jobs
                .fail(
                    &panic_job_id,
                    auth_setup_internal_error(format!(
                        "auth setup background task panicked: {detail}"
                    )),
                )
                .await;
        }
    });

    Ok(Json(AuthSetupStartResponse { job }))
}

async fn get_auth_auto_setup_job(
    State(s): State<ServerState>,
    Path((project_id, job_id)): Path<(String, String)>,
) -> Result<Json<AuthSetupJobRecord>, ApiError> {
    let job = s
        .auth_setup_jobs
        .get(&job_id)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("auth setup job `{job_id}` not found")))?;
    if job.project_id != project_id {
        return Err(ApiError::NotFound(format!("auth setup job `{job_id}` not found")));
    }
    Ok(Json(job))
}

async fn start_ai_project_setup(
    State(s): State<ServerState>,
    Path(id): Path<String>,
    Json(req): Json<ProjectSetupRequest>,
) -> Result<Json<ProjectSetupStartResponse>, ApiError> {
    let project = s
        .store
        .projects()
        .get(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("project `{id}` not found")))?;
    let target_base_url = auth_setup_target_base_url(&project, req.target_base_url.as_deref());
    if let Some(url) = target_base_url.as_deref() {
        if !is_local_http_url(url) {
            return Err(ApiError::BadRequest(format!("target URL `{url}` must be local")));
        }
    }

    let job = s.project_setup_jobs.create(&id, now_epoch_ms()).await;
    let job_id = job.id.clone();
    let state = s.clone();
    tokio::spawn(async move {
        let panic_state = state.clone();
        let panic_job_id = job_id.clone();
        let result =
            AssertUnwindSafe(run_ai_project_setup_job(state, id, req, job_id)).catch_unwind().await;
        if let Err(payload) = result {
            let detail = panic_payload_message(payload.as_ref());
            tracing::error!(job_id = %panic_job_id, %detail, "project setup job panicked");
            panic_state
                .project_setup_jobs
                .fail(
                    &panic_job_id,
                    project_setup_internal_error(format!(
                        "project setup background task panicked: {detail}"
                    )),
                )
                .await;
        }
    });

    Ok(Json(ProjectSetupStartResponse { job }))
}

async fn list_ai_project_setup_jobs(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
) -> Result<Json<ProjectSetupJobListResponse>, ApiError> {
    s.store
        .projects()
        .get(&project_id)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("project `{project_id}` not found")))?;
    let jobs = s.project_setup_jobs.list_by_project(&project_id).await;
    Ok(Json(ProjectSetupJobListResponse { jobs }))
}

async fn get_ai_project_setup_job(
    State(s): State<ServerState>,
    Path((project_id, job_id)): Path<(String, String)>,
) -> Result<Json<ProjectSetupJobRecord>, ApiError> {
    let job = s
        .project_setup_jobs
        .get(&job_id)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("project setup job `{job_id}` not found")))?;
    if job.project_id != project_id {
        return Err(ApiError::NotFound(format!("project setup job `{job_id}` not found")));
    }
    Ok(Json(job))
}

async fn run_ai_project_setup_job(
    s: ServerState,
    id: String,
    req: ProjectSetupRequest,
    job_id: String,
) {
    tracing::info!(
        project_id = %id,
        job_id = %job_id,
        project_setup = req.project_setup,
        seed_setup = req.seed_setup,
        auth_setup = req.auth_setup,
        "AI project setup job started"
    );
    let result = run_ai_project_setup_once(s.clone(), &id, req, &job_id).await;
    match result {
        Ok(response) => {
            tracing::info!(
                project_id = %id,
                job_id = %job_id,
                profile_id = %response.profile.id,
                auth_profiles = response
                    .project
                    .runtime_profile
                    .as_ref()
                    .map(|profile| profile.auth_profiles.len())
                    .unwrap_or(0),
                seed_setup = response.seed_setup.is_some(),
                auth_setup = response.auth_setup.is_some(),
                "AI project setup job finished"
            );
            s.project_setup_jobs.complete(&job_id, response).await;
        }
        Err(error) => {
            tracing::error!(
                project_id = %id,
                job_id = %job_id,
                code = %error.code,
                detail = %error.detail,
                "AI project setup job failed"
            );
            s.project_setup_jobs.fail(&job_id, error).await;
        }
    }
}

async fn run_ai_project_setup_once(
    s: ServerState,
    id: &str,
    req: ProjectSetupRequest,
    job_id: &str,
) -> Result<ProjectSetupResponse, ProjectSetupError> {
    if !req.project_setup && !req.seed_setup && !req.auth_setup {
        return Err(project_setup_no_features_error());
    }

    s.project_setup_jobs
        .push_phase(job_id, ProjectSetupPhase::CollectingRepos, "Collecting project repositories.")
        .await;
    let mut project = s
        .store
        .projects()
        .get(id)
        .await
        .map_err(project_setup_store_error)?
        .ok_or_else(|| project_setup_not_found_error(format!("project `{id}` not found")))?;
    let repos = s.store.repos().list_by_project(id).await.map_err(project_setup_store_error)?;
    let workspace_roots = auth_setup_workspace_roots(&repos, s.state_repos_dir.as_deref());
    if workspace_roots.is_empty() && (req.project_setup || req.seed_setup) {
        return Err(ProjectSetupError {
            code: "no_local_workspace".to_string(),
            title: "Project setup needs a local repository".to_string(),
            detail: "No local repo workspace was available for the agent to inspect.".to_string(),
            hint: Some(
                "Add or ingest at least one local project repository, then retry.".to_string(),
            ),
            retryable: true,
        });
    }
    let target_base_url = auth_setup_target_base_url(&project, req.target_base_url.as_deref());
    if let Some(url) = target_base_url.as_deref() {
        if !is_local_http_url(url) {
            return Err(ProjectSetupError {
                code: "target_not_local".to_string(),
                title: "Project setup target is not local".to_string(),
                detail: format!("target URL `{url}` must be local"),
                hint: Some("Use a localhost or loopback app URL for AI project setup.".to_string()),
                retryable: false,
            });
        }
    }

    let mut launch_profile = project.default_launch_profile.clone();
    let mut overall_checks = Vec::new();
    let mut overall_warnings = Vec::new();
    let mut messages = Vec::new();
    let mut seed_setup = None;
    let mut auth_setup = None;
    let mut agent_used = false;
    let mut seed_roles = Vec::new();
    let mut seeded_objects = Vec::new();

    if req.project_setup {
        let Some(agent) = s.project_setup_agent.as_ref() else {
            return Err(ProjectSetupError {
                code: "agent_runtime_unavailable".to_string(),
                title: "No AI project setup agent is configured".to_string(),
                detail: "AI project setup requires a CLI-backed agent runtime.".to_string(),
                hint: Some("Choose Codex or Claude Code in AI setup and make sure the CLI is installed and logged in.".to_string()),
                retryable: true,
            });
        };

        s.project_setup_jobs
            .push_phase(
                job_id,
                ProjectSetupPhase::StartingAgent,
                "Starting the repository setup agent.",
            )
            .await;
        let agent_req = ProjectSetupAgentRequest {
            project_id: id.to_string(),
            project_name: project.name.clone(),
            target_base_url: target_base_url.clone(),
            workspace_roots: workspace_roots.clone(),
            existing_launch_profile: launch_profile.clone(),
        };
        s.project_setup_jobs
            .push_phase(
                job_id,
                ProjectSetupPhase::InspectingProject,
                "Agent is inspecting scripts, env files, migrations, and local dev workflow.",
            )
            .await;
        let mut output = agent.explore(agent_req).await.map_err(project_setup_agent_error)?;
        agent_used = true;
        validate_project_setup_profile(&mut output.profile)?;

        s.project_setup_jobs
            .push_phase(job_id, ProjectSetupPhase::ApplyingProfile, "Saving launch profile.")
            .await;
        let now = now_epoch_ms();
        let profile = s
            .store
            .launch_profiles()
            .upsert_default(id, &output.profile, now)
            .await
            .map_err(project_setup_store_error)?;
        if project.target_base_url.is_none() {
            if let Some(target) = profile.target_urls.first().cloned() {
                let patch = ProjectPatch {
                    description: ProjectPatchOption::Unset,
                    target_base_url: ProjectPatchOption::Set(Some(target)),
                    env_config_json: ProjectPatchOption::Unset,
                    runtime_profile_json: ProjectPatchOption::Unset,
                    updated_at: now,
                };
                s.store.projects().update(id, &patch).await.map_err(project_setup_store_error)?;
            }
        }
        launch_profile = Some(profile);
        overall_checks.extend(output.checks);
        overall_warnings.extend(output.warnings);
        messages.push(output.message);
        if output.verification_status == ProjectSetupVerificationStatus::NeedsReview
            && overall_warnings.is_empty()
        {
            overall_warnings
                .push("Project setup agent marked the launch profile for review.".to_string());
        }
        project = s.store.projects().get(id).await.map_err(project_setup_store_error)?.ok_or_else(
            || project_setup_internal_error("project vanished after AI project setup".to_string()),
        )?;
    }

    if req.seed_setup {
        let Some(agent) = s.seed_setup_agent.as_ref() else {
            return Err(ProjectSetupError {
                code: "agent_runtime_unavailable".to_string(),
                title: "No AI seed setup agent is configured".to_string(),
                detail: "AI seed setup requires a CLI-backed agent runtime.".to_string(),
                hint: Some("Choose Codex or Claude Code in AI setup and make sure the CLI is installed and logged in.".to_string()),
                retryable: true,
            });
        };

        s.project_setup_jobs
            .push_phase(job_id, ProjectSetupPhase::StartingAgent, "Starting the seed setup agent.")
            .await;
        let agent_req = SeedSetupAgentRequest {
            project_id: id.to_string(),
            project_name: project.name.clone(),
            target_base_url: target_base_url.clone(),
            workspace_roots: workspace_roots.clone(),
            launch_profile: launch_profile.clone(),
        };
        s.project_setup_jobs
            .push_phase(
                job_id,
                ProjectSetupPhase::InspectingSeed,
                "Agent is preparing deterministic local fixtures, roles, owned objects, and reset hooks.",
            )
            .await;
        let output = agent.explore(agent_req).await.map_err(seed_setup_agent_error)?;
        agent_used = true;
        validate_seed_setup_plan(&output.plan)?;

        let mut input = launch_profile
            .as_ref()
            .map(project_launch_profile_to_input)
            .unwrap_or_else(|| blank_launch_profile_input(target_base_url.as_deref()));
        apply_seed_plan_to_launch_profile(&mut input, &output.plan);

        s.project_setup_jobs
            .push_phase(job_id, ProjectSetupPhase::ApplyingSeed, "Saving seed and reset setup.")
            .await;
        let now = now_epoch_ms();
        let profile = s
            .store
            .launch_profiles()
            .upsert_default(id, &input, now)
            .await
            .map_err(project_setup_store_error)?;
        launch_profile = Some(profile.clone());

        if apply_seed_env_to_project_runtime_profile(
            &s,
            id,
            &project,
            &output.plan,
            target_base_url.clone(),
            launch_profile.as_ref(),
            now,
        )
        .await?
        {
            project =
                s.store.projects().get(id).await.map_err(project_setup_store_error)?.ok_or_else(
                    || {
                        project_setup_internal_error(
                            "project vanished after seed setup".to_string(),
                        )
                    },
                )?;
        }

        let verification = ProjectSetupVerification {
            status: if output.plan.warnings.is_empty() {
                ProjectSetupVerificationStatus::Verified
            } else {
                ProjectSetupVerificationStatus::NeedsReview
            },
            checks: output.plan.checks.clone(),
            warnings: output.plan.warnings.clone(),
        };
        overall_checks.extend(verification.checks.clone());
        overall_warnings.extend(verification.warnings.clone());
        seed_roles = output.plan.roles.clone();
        seeded_objects = output.plan.seeded_objects.clone();
        messages.push(output.message.clone());
        seed_setup =
            Some(SeedSetupResponse { plan: output.plan, verification, message: output.message });
    }

    if req.auth_setup {
        s.project_setup_jobs
            .push_phase(
                job_id,
                ProjectSetupPhase::InspectingAuth,
                "Running auth setup with seeded roles and owned objects.",
            )
            .await;
        let auth_job = s.auth_setup_jobs.create(id, now_epoch_ms()).await;
        let auth_req = AuthSetupRequest {
            target_base_url: target_base_url.clone(),
            roles: seed_roles.clone(),
            seeded_objects: seeded_objects.clone(),
        };
        let result = run_auth_auto_setup_once(s.clone(), id, auth_req, &auth_job.id).await;
        match result {
            Ok(response) => {
                s.auth_setup_jobs.complete(&auth_job.id, response.clone()).await;
                overall_checks.extend(response.verification.checks.clone());
                overall_warnings.extend(response.verification.warnings.clone());
                if response.verification.status != AuthSetupVerificationStatus::Verified {
                    overall_warnings.push("Auth setup needs review.".to_string());
                }
                messages.push(response.message.clone());
                agent_used |= response.agent_used;
                project = response.project.clone();
                auth_setup = Some(response);
            }
            Err(error) => {
                s.auth_setup_jobs.fail(&auth_job.id, error.clone()).await;
                return Err(project_setup_from_auth_error(error));
            }
        }
    }

    let profile = ensure_project_setup_launch_profile(
        &s,
        id,
        &mut project,
        launch_profile,
        target_base_url.as_deref(),
    )
    .await?;
    let project =
        s.store.projects().get(id).await.map_err(project_setup_store_error)?.ok_or_else(|| {
            project_setup_internal_error("project vanished after setup".to_string())
        })?;
    let verification = ProjectSetupVerification {
        status: if overall_warnings.is_empty() {
            ProjectSetupVerificationStatus::Verified
        } else {
            ProjectSetupVerificationStatus::NeedsReview
        },
        checks: overall_checks,
        warnings: overall_warnings,
    };
    let mut message =
        if messages.is_empty() { "AI setup finished.".to_string() } else { messages.join(" ") };
    if !verification.warnings.is_empty() {
        message.push_str(&format!(" Review {} warning(s).", verification.warnings.len()));
    }
    let response = ProjectSetupResponse {
        project,
        profile,
        agent_used,
        verification,
        seed_setup,
        auth_setup,
        message,
    };
    validate_project_setup_postconditions(&req, &response)?;
    Ok(response)
}

async fn run_auth_auto_setup_job(
    s: ServerState,
    id: String,
    req: AuthSetupRequest,
    job_id: String,
) {
    tracing::info!(project_id = %id, job_id = %job_id, "auth setup job started");
    let result = run_auth_auto_setup_once(s.clone(), &id, req, &job_id).await;
    match result {
        Ok(response) => {
            tracing::info!(
                project_id = %id,
                job_id = %job_id,
                profiles = response.profiles_added + response.profiles_updated,
                "auth setup job finished"
            );
            s.auth_setup_jobs.complete(&job_id, response).await;
        }
        Err(error) => {
            tracing::error!(
                project_id = %id,
                job_id = %job_id,
                code = %error.code,
                detail = %error.detail,
                "auth setup job failed"
            );
            s.auth_setup_jobs.fail(&job_id, error).await;
        }
    }
}

async fn run_auth_auto_setup_once(
    s: ServerState,
    id: &str,
    req: AuthSetupRequest,
    job_id: &str,
) -> Result<AuthSetupResponse, AuthSetupError> {
    s.auth_setup_jobs
        .push_phase(job_id, AuthSetupPhase::CollectingRepos, "Collecting project repositories.")
        .await;
    let project = s
        .store
        .projects()
        .get(id)
        .await
        .map_err(auth_setup_store_error)?
        .ok_or_else(|| auth_setup_not_found_error(format!("project `{id}` not found")))?;
    let target_base_url = auth_setup_target_base_url(&project, req.target_base_url.as_deref());
    if let Some(url) = target_base_url.as_deref() {
        if !is_local_http_url(url) {
            return Err(AuthSetupError {
                code: "target_not_local".to_string(),
                title: "Auth setup target is not local".to_string(),
                detail: format!("target URL `{url}` must be local"),
                hint: Some("Use a localhost or loopback app URL for auth setup.".to_string()),
                retryable: false,
            });
        }
    }

    let repos = s.store.repos().list_by_project(id).await.map_err(auth_setup_store_error)?;
    let workspace_roots = auth_setup_workspace_roots(&repos, s.state_repos_dir.as_deref());
    let discovery = discover_auth_setup(&workspace_roots);
    s.auth_setup_jobs
        .push_phase(
            job_id,
            AuthSetupPhase::StartingAgent,
            if s.auth_setup_agent.is_some() {
                "Starting repository exploration agent."
            } else {
                "No exploration agent is configured; using static repository scan."
            },
        )
        .await;
    let agent_output = if let Some(agent) = s.auth_setup_agent.as_ref() {
        let agent_req = AuthSetupAgentRequest {
            project_id: id.to_string(),
            project_name: project.name.clone(),
            target_base_url: target_base_url.clone(),
            workspace_roots: workspace_roots.clone(),
            requested_roles: req.roles.clone(),
            seeded_objects: req.seeded_objects.clone(),
            existing_profiles: project
                .runtime_profile
                .as_ref()
                .map(|profile| profile.auth_profiles.clone())
                .unwrap_or_default(),
            static_login_paths: discovery.login_paths.clone(),
            static_object_routes: discovery.object_routes.clone(),
            files_inspected: discovery.files_inspected,
        };
        s.auth_setup_jobs
            .push_phase(
                job_id,
                AuthSetupPhase::InspectingAuthRoutes,
                "Agent is inspecting auth routes, sessions, roles, and ownership hints.",
            )
            .await;
        match agent.explore(agent_req).await {
            Ok(output) if output.profiles.is_empty() => return Err(auth_setup_no_profiles_error()),
            Ok(output) => Some(output),
            Err(err) => return Err(auth_setup_agent_error(err)),
        }
    } else {
        None
    };
    let mut runtime_profile = project.runtime_profile.clone().unwrap_or_else(|| {
        empty_runtime_profile_for_auth_setup(
            target_base_url.clone(),
            project.default_launch_profile.as_ref(),
        )
    });
    if runtime_profile.target_base_url.is_none() {
        runtime_profile.target_base_url = target_base_url.clone();
    }
    if runtime_profile.health_check_url.is_none() {
        runtime_profile.health_check_url = target_base_url.clone();
    }

    let agent_used = agent_output.is_some();
    let (
        roles,
        login_paths,
        object_routes,
        mut verification,
        agent_message,
        profiles_added,
        profiles_updated,
    ) = if let Some(output) = agent_output {
        s.auth_setup_jobs
            .push_phase(
                job_id,
                AuthSetupPhase::DraftingProfiles,
                "Normalizing agent-generated auth profiles.",
            )
            .await;
        apply_agent_auth_setup_output(
            &mut runtime_profile.auth_profiles,
            output,
            discovery.login_paths.first().cloned(),
            &req.seeded_objects,
        )
    } else {
        s.auth_setup_jobs
            .push_phase(
                job_id,
                AuthSetupPhase::DraftingProfiles,
                "Drafting auth profiles from static repository hints.",
            )
            .await;
        let roles = auth_setup_roles(&req.roles, &discovery);
        let (profiles_added, profiles_updated) = merge_auth_setup_profiles(
            &mut runtime_profile.auth_profiles,
            &roles,
            discovery.login_paths.first().cloned(),
            &req.seeded_objects,
        );
        let verification = static_auth_setup_verification(&discovery, None);
        (
            roles,
            discovery.login_paths.clone(),
            discovery.object_routes.clone(),
            verification,
            None,
            profiles_added,
            profiles_updated,
        )
    };
    apply_discovered_otp_hints(&mut runtime_profile, target_base_url.as_deref(), &discovery);
    let auth_env_resolution =
        apply_discovered_auth_env_values(&mut runtime_profile, &discovery.credentials);
    apply_auth_env_resolution_to_verification(&mut verification, &auth_env_resolution);
    s.auth_setup_jobs
        .push_phase(
            job_id,
            AuthSetupPhase::VerifyingProfiles,
            "Reviewing generated profiles against discovered auth evidence.",
        )
        .await;
    let runtime_profile_json = serde_json::to_string(&runtime_profile).map_err(|e| {
        auth_setup_internal_error(format!("runtime_profile must serialize to JSON: {e}"))
    })?;
    s.auth_setup_jobs
        .push_phase(job_id, AuthSetupPhase::SavingProfiles, "Saving auth profiles.")
        .await;
    let now = now_epoch_ms();
    let patch = ProjectPatch {
        description: ProjectPatchOption::Unset,
        target_base_url: target_base_url
            .clone()
            .map(|url| ProjectPatchOption::Set(Some(url)))
            .unwrap_or(ProjectPatchOption::Unset),
        env_config_json: ProjectPatchOption::Unset,
        runtime_profile_json: ProjectPatchOption::Set(Some(runtime_profile_json)),
        updated_at: now,
    };
    if !s.store.projects().update(id, &patch).await.map_err(auth_setup_store_error)? {
        return Err(auth_setup_not_found_error(format!("project `{id}` not found")));
    }
    let project =
        s.store.projects().get(id).await.map_err(auth_setup_store_error)?.ok_or_else(|| {
            auth_setup_internal_error("project vanished after auth setup".to_string())
        })?;
    let message = auth_setup_response_message(
        agent_used,
        profiles_added,
        profiles_updated,
        discovery.files_inspected,
        &verification,
        agent_message,
        auth_env_resolution_message(&auth_env_resolution),
    );
    Ok(AuthSetupResponse {
        project,
        roles,
        login_paths,
        object_routes,
        agent_used,
        verification,
        profiles_added,
        profiles_updated,
        message,
    })
}

fn auth_setup_store_error(err: nyx_agent_core::store::StoreError) -> AuthSetupError {
    AuthSetupError {
        code: "store_error".to_string(),
        title: "Auth setup could not read or save project data".to_string(),
        detail: err.to_string(),
        hint: Some("Retry the setup. If this repeats, restart the Nyx Agent daemon.".to_string()),
        retryable: true,
    }
}

fn auth_setup_not_found_error(detail: String) -> AuthSetupError {
    AuthSetupError {
        code: "project_not_found".to_string(),
        title: "Project was not found".to_string(),
        detail,
        hint: Some("Refresh the project list and try again.".to_string()),
        retryable: false,
    }
}

fn auth_setup_internal_error(detail: String) -> AuthSetupError {
    AuthSetupError {
        code: "internal_error".to_string(),
        title: "Auth setup hit an internal error".to_string(),
        detail,
        hint: Some("Retry the setup. If this repeats, check the daemon logs.".to_string()),
        retryable: true,
    }
}

fn auth_setup_no_profiles_error() -> AuthSetupError {
    AuthSetupError {
        code: "agent_returned_no_profiles".to_string(),
        title: "The auth setup agent did not return any profiles".to_string(),
        detail: "The exploration agent completed but did not record a usable auth profile."
            .to_string(),
        hint: Some(
            "Check that the repository contains login/session code or add a role manually."
                .to_string(),
        ),
        retryable: true,
    }
}

fn auth_setup_agent_error(err: AuthSetupAgentError) -> AuthSetupError {
    let raw = err.to_string();
    let lower = raw.to_ascii_lowercase();
    let network_like = lower.contains("network")
        || lower.contains("dns")
        || lower.contains("could not resolve")
        || lower.contains("connection")
        || lower.contains("timeout")
        || lower.contains("timed out")
        || lower.contains("transport");
    let unavailable = matches!(err, AuthSetupAgentError::Unavailable(_));
    let (code, title, hint, retryable) = if network_like {
        (
            "agent_upstream_network",
            "The auth setup agent could not reach its AI runtime",
            "Check your network connection and the configured AI CLI login, then retry.",
            true,
        )
    } else if unavailable {
        (
            "agent_runtime_unavailable",
            "The configured auth setup agent is unavailable",
            "Choose Codex or Claude Code in AI setup and make sure the CLI is installed and logged in.",
            true,
        )
    } else {
        (
            "agent_failed",
            "The auth setup agent failed",
            "Retry the job. If this repeats, inspect the daemon logs for the underlying CLI error.",
            true,
        )
    };
    AuthSetupError {
        code: code.to_string(),
        title: title.to_string(),
        detail: raw,
        hint: Some(hint.to_string()),
        retryable,
    }
}

fn project_setup_store_error(err: nyx_agent_core::store::StoreError) -> ProjectSetupError {
    ProjectSetupError {
        code: "store_error".to_string(),
        title: "Project setup could not read or save project data".to_string(),
        detail: err.to_string(),
        hint: Some("Retry the setup. If this repeats, restart the Nyx Agent daemon.".to_string()),
        retryable: true,
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_string()
}

fn project_setup_not_found_error(detail: String) -> ProjectSetupError {
    ProjectSetupError {
        code: "project_not_found".to_string(),
        title: "Project was not found".to_string(),
        detail,
        hint: Some("Refresh the project list and try again.".to_string()),
        retryable: false,
    }
}

fn project_setup_internal_error(detail: String) -> ProjectSetupError {
    ProjectSetupError {
        code: "internal_error".to_string(),
        title: "Project setup hit an internal error".to_string(),
        detail,
        hint: Some("Retry the setup. If this repeats, check the daemon logs.".to_string()),
        retryable: true,
    }
}

fn project_setup_agent_error(err: ProjectSetupAgentError) -> ProjectSetupError {
    let raw = err.to_string();
    let unavailable = matches!(err, ProjectSetupAgentError::Unavailable(_));
    ProjectSetupError {
        code: if unavailable { "agent_runtime_unavailable" } else { "agent_failed" }.to_string(),
        title: if unavailable {
            "The configured project setup agent is unavailable"
        } else {
            "The project setup agent failed"
        }
        .to_string(),
        detail: raw,
        hint: Some(if unavailable {
            "Choose Codex or Claude Code in AI setup and make sure the CLI is installed and logged in."
        } else {
            "Retry the job. If this repeats, inspect the daemon logs for the underlying CLI error."
        }
        .to_string()),
        retryable: true,
    }
}

fn seed_setup_agent_error(err: SeedSetupAgentError) -> ProjectSetupError {
    let raw = err.to_string();
    let unavailable = matches!(err, SeedSetupAgentError::Unavailable(_));
    ProjectSetupError {
        code: if unavailable { "agent_runtime_unavailable" } else { "seed_agent_failed" }
            .to_string(),
        title: if unavailable {
            "The configured seed setup agent is unavailable"
        } else {
            "The seed setup agent failed"
        }
        .to_string(),
        detail: raw,
        hint: Some(if unavailable {
            "Choose Codex or Claude Code in AI setup and make sure the CLI is installed and logged in."
        } else {
            "Retry the job. If this repeats, inspect the daemon logs for the underlying CLI error."
        }
        .to_string()),
        retryable: true,
    }
}

fn project_setup_from_auth_error(err: AuthSetupError) -> ProjectSetupError {
    ProjectSetupError {
        code: format!("auth_{}", err.code),
        title: format!("Auth setup failed: {}", err.title),
        detail: err.detail,
        hint: err.hint,
        retryable: err.retryable,
    }
}

fn project_setup_no_features_error() -> ProjectSetupError {
    ProjectSetupError {
        code: "no_setup_features_selected".to_string(),
        title: "No setup features were selected".to_string(),
        detail: "Select project setup, seed setup, auth setup, or any combination of them."
            .to_string(),
        hint: Some("Choose at least one AI setup feature and retry.".to_string()),
        retryable: false,
    }
}

fn validate_project_setup_profile(
    profile: &mut ProjectLaunchProfileInput,
) -> Result<(), ProjectSetupError> {
    for url in &profile.target_urls {
        if !is_local_http_url(url) {
            return Err(ProjectSetupError {
                code: "target_not_local".to_string(),
                title: "AI project setup proposed a non-local target".to_string(),
                detail: format!("target URL `{url}` must be local"),
                hint: Some(
                    "Ask the setup agent to use a localhost or loopback dev URL.".to_string(),
                ),
                retryable: true,
            });
        }
    }
    for check in &profile.health_checks {
        if let Some(url) = check.url.as_deref() {
            if !is_local_http_url(url) {
                return Err(ProjectSetupError {
                    code: "health_target_not_local".to_string(),
                    title: "AI project setup proposed a non-local health check".to_string(),
                    detail: format!("health check URL `{url}` must be local"),
                    hint: Some(
                        "Ask the setup agent to use a localhost or loopback health URL."
                            .to_string(),
                    ),
                    retryable: true,
                });
            }
        }
    }
    if profile.target_urls.is_empty()
        && profile.start_steps.is_empty()
        && profile.health_checks.is_empty()
    {
        return Err(ProjectSetupError {
            code: "empty_profile".to_string(),
            title: "AI project setup returned an empty launch profile".to_string(),
            detail: "The agent did not provide a target URL, start command, or health check."
                .to_string(),
            hint: Some("Retry after adding local setup docs or a package script.".to_string()),
            retryable: true,
        });
    }
    Ok(())
}

fn validate_seed_setup_plan(plan: &SeedSetupPlan) -> Result<(), ProjectSetupError> {
    let empty = plan.seed_steps.is_empty()
        && plan.reset_steps.is_empty()
        && plan.env_vars.is_empty()
        && plan.roles.is_empty()
        && plan.seeded_objects.is_empty();
    if empty {
        return Err(ProjectSetupError {
            code: "empty_seed_plan".to_string(),
            title: "AI seed setup returned an empty plan".to_string(),
            detail: "The seed setup agent did not provide seed commands, reset commands, env vars, roles, or seeded objects.".to_string(),
            hint: Some("Retry after adding local seed docs or fixture scripts to the repository.".to_string()),
            retryable: true,
        });
    }
    for var in &plan.env_vars {
        if var.name.trim().is_empty() {
            return Err(ProjectSetupError {
                code: "empty_seed_env_name".to_string(),
                title: "AI seed setup proposed an invalid environment variable".to_string(),
                detail: "A seed environment variable had an empty name.".to_string(),
                hint: Some("Retry seed setup or add the fixture env vars manually.".to_string()),
                retryable: true,
            });
        }
    }
    Ok(())
}

fn validate_project_setup_postconditions(
    req: &ProjectSetupRequest,
    response: &ProjectSetupResponse,
) -> Result<(), ProjectSetupError> {
    if req.project_setup {
        let mut profile = project_launch_profile_to_input(&response.profile);
        validate_project_setup_profile(&mut profile).map_err(|_| ProjectSetupError {
            code: "launch_profile_not_persisted".to_string(),
            title: "AI setup did not save a usable launch profile".to_string(),
            detail: "The setup job finished without a target URL, start command, or health check in the saved launch profile.".to_string(),
            hint: Some("Retry AI setup. If this repeats, add local setup docs or commands manually in the environment profile.".to_string()),
            retryable: true,
        })?;
    }
    if req.seed_setup && response.seed_setup.is_none() {
        return Err(ProjectSetupError {
            code: "seed_setup_not_persisted".to_string(),
            title: "AI setup did not save seed setup".to_string(),
            detail: "The setup job finished without a seed setup result.".to_string(),
            hint: Some("Retry AI setup with Seed setup selected.".to_string()),
            retryable: true,
        });
    }
    if req.auth_setup {
        if response.auth_setup.is_none() {
            return Err(ProjectSetupError {
                code: "auth_setup_not_persisted".to_string(),
                title: "AI setup did not save auth setup".to_string(),
                detail: "The setup job finished without an auth setup result.".to_string(),
                hint: Some("Retry AI setup with Auth setup selected.".to_string()),
                retryable: true,
            });
        }
        let auth_profiles = response
            .project
            .runtime_profile
            .as_ref()
            .map(|profile| profile.auth_profiles.len())
            .unwrap_or(0);
        if auth_profiles == 0 {
            return Err(ProjectSetupError {
                code: "auth_profiles_not_persisted".to_string(),
                title: "AI setup did not save auth profiles".to_string(),
                detail: "The setup job finished but the project still has zero runtime auth profiles.".to_string(),
                hint: Some("Retry auth setup after confirming the local app has deterministic test users or fixture credentials.".to_string()),
                retryable: true,
            });
        }
    }
    Ok(())
}

fn project_launch_profile_to_input(profile: &ProjectLaunchProfile) -> ProjectLaunchProfileInput {
    ProjectLaunchProfileInput {
        name: Some(profile.name.clone()),
        mode: Some(profile.mode.clone()),
        build_steps: profile.build_steps.clone(),
        start_steps: profile.start_steps.clone(),
        seed_steps: profile.seed_steps.clone(),
        reset_steps: profile.reset_steps.clone(),
        login_steps: profile.login_steps.clone(),
        stop_steps: profile.stop_steps.clone(),
        health_checks: profile.health_checks.clone(),
        target_urls: profile.target_urls.clone(),
        env_refs: profile.env_refs.clone(),
        working_dirs: profile.working_dirs.clone(),
    }
}

fn blank_launch_profile_input(target_base_url: Option<&str>) -> ProjectLaunchProfileInput {
    ProjectLaunchProfileInput {
        name: Some("AI local setup".to_string()),
        mode: Some("already-running".to_string()),
        build_steps: Vec::new(),
        start_steps: Vec::new(),
        seed_steps: Vec::new(),
        reset_steps: Vec::new(),
        login_steps: Vec::new(),
        stop_steps: Vec::new(),
        health_checks: Vec::new(),
        target_urls: target_base_url.map(str::to_string).into_iter().collect(),
        env_refs: Vec::new(),
        working_dirs: Vec::new(),
    }
}

fn apply_seed_plan_to_launch_profile(input: &mut ProjectLaunchProfileInput, plan: &SeedSetupPlan) {
    if !plan.seed_steps.is_empty() {
        input.seed_steps = plan.seed_steps.clone();
    }
    if !plan.reset_steps.is_empty() {
        input.reset_steps = plan.reset_steps.clone();
    }
    if !plan.seed_steps.is_empty() || !plan.reset_steps.is_empty() {
        input.mode = Some("custom-commands".to_string());
    }
    for var in &plan.env_vars {
        let name = var.name.trim();
        if name.is_empty() {
            continue;
        }
        if !input.env_refs.iter().any(|entry| entry.kind == "env-var" && entry.value == name) {
            input.env_refs.push(nyx_agent_types::product::LaunchEnvRef {
                kind: "env-var".to_string(),
                value: name.to_string(),
                secret: var.secret,
            });
        }
    }
}

async fn apply_seed_env_to_project_runtime_profile(
    s: &ServerState,
    id: &str,
    project: &ProjectRecord,
    plan: &SeedSetupPlan,
    target_base_url: Option<String>,
    launch_profile: Option<&ProjectLaunchProfile>,
    now: i64,
) -> Result<bool, ProjectSetupError> {
    if plan.env_vars.is_empty() {
        return Ok(false);
    }

    let mut runtime_profile = project.runtime_profile.clone().unwrap_or_else(|| {
        empty_runtime_profile_for_auth_setup(target_base_url.clone(), launch_profile)
    });
    if runtime_profile.target_base_url.is_none() {
        runtime_profile.target_base_url = target_base_url.clone();
    }
    if runtime_profile.health_check_url.is_none() {
        runtime_profile.health_check_url = target_base_url.clone();
    }
    let changed = merge_runtime_env_vars(&mut runtime_profile.env_vars, &plan.env_vars);
    if !changed {
        return Ok(false);
    }

    let runtime_profile_json = serde_json::to_string(&runtime_profile).map_err(|e| {
        project_setup_internal_error(format!("runtime_profile must serialize to JSON: {e}"))
    })?;
    let patch = ProjectPatch {
        description: ProjectPatchOption::Unset,
        target_base_url: target_base_url
            .map(|url| ProjectPatchOption::Set(Some(url)))
            .unwrap_or(ProjectPatchOption::Unset),
        env_config_json: ProjectPatchOption::Unset,
        runtime_profile_json: ProjectPatchOption::Set(Some(runtime_profile_json)),
        updated_at: now,
    };
    if !s.store.projects().update(id, &patch).await.map_err(project_setup_store_error)? {
        return Err(project_setup_not_found_error(format!("project `{id}` not found")));
    }
    Ok(true)
}

fn merge_runtime_env_vars(
    existing: &mut Vec<ProjectRuntimeEnvVar>,
    incoming: &[ProjectRuntimeEnvVar],
) -> bool {
    let mut changed = false;
    for var in incoming {
        let name = var.name.trim();
        if name.is_empty() {
            continue;
        }
        if let Some(current) = existing.iter_mut().find(|current| current.name == name) {
            if current.value != var.value || current.secret != var.secret || current.name != name {
                current.name = name.to_string();
                current.value = var.value.clone();
                current.secret = var.secret;
                changed = true;
            }
        } else {
            existing.push(ProjectRuntimeEnvVar {
                name: name.to_string(),
                value: var.value.clone(),
                secret: var.secret,
            });
            changed = true;
        }
    }
    changed
}

async fn ensure_project_setup_launch_profile(
    s: &ServerState,
    id: &str,
    project: &mut ProjectRecord,
    launch_profile: Option<ProjectLaunchProfile>,
    target_base_url: Option<&str>,
) -> Result<ProjectLaunchProfile, ProjectSetupError> {
    if let Some(profile) = launch_profile {
        return Ok(profile);
    }

    let input = project
        .runtime_profile
        .as_ref()
        .map(|profile| launch_profile_input_from_runtime(profile, target_base_url))
        .unwrap_or_else(|| blank_launch_profile_input(target_base_url));
    let now = now_epoch_ms();
    let profile = s
        .store
        .launch_profiles()
        .upsert_default(id, &input, now)
        .await
        .map_err(project_setup_store_error)?;
    if project.target_base_url.is_none() {
        if let Some(target) = profile.target_urls.first().cloned() {
            let patch = ProjectPatch {
                description: ProjectPatchOption::Unset,
                target_base_url: ProjectPatchOption::Set(Some(target)),
                env_config_json: ProjectPatchOption::Unset,
                runtime_profile_json: ProjectPatchOption::Unset,
                updated_at: now,
            };
            s.store.projects().update(id, &patch).await.map_err(project_setup_store_error)?;
            *project =
                s.store.projects().get(id).await.map_err(project_setup_store_error)?.ok_or_else(
                    || project_setup_internal_error("project vanished after setup".to_string()),
                )?;
        }
    }
    Ok(profile)
}

fn project_patch_for(opt: &Option<Option<String>>) -> ProjectPatchOption<Option<String>> {
    match opt {
        None => ProjectPatchOption::Unset,
        Some(None) => ProjectPatchOption::Set(None),
        Some(Some(v)) => ProjectPatchOption::Set(Some(v.clone())),
    }
}

fn normalize_create_target_base_url(
    target_base_url: Option<String>,
    runtime_profile: &mut Option<ProjectRuntimeProfile>,
) -> Result<Option<String>, ApiError> {
    let target_base_url = normalize_optional_string(target_base_url.as_deref());
    let profile_target = runtime_profile
        .as_ref()
        .and_then(|profile| normalize_optional_string(profile.target_base_url.as_deref()));

    if let (Some(top_level), Some(profile_target)) = (&target_base_url, &profile_target) {
        if top_level != profile_target {
            return Err(ApiError::BadRequest(
                "runtime_profile.target_base_url must match target_base_url".to_string(),
            ));
        }
    }

    let resolved = target_base_url.or(profile_target);
    if let Some(profile) = runtime_profile.as_mut() {
        profile.target_base_url = resolved.clone();
    }
    Ok(resolved)
}

fn normalize_optional_string(value: Option<&str>) -> Option<String> {
    value.map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

fn auth_setup_target_base_url(project: &ProjectRecord, requested: Option<&str>) -> Option<String> {
    normalize_optional_string(requested)
        .or_else(|| {
            project
                .runtime_profile
                .as_ref()
                .and_then(|profile| normalize_optional_string(profile.target_base_url.as_deref()))
        })
        .or_else(|| normalize_optional_string(project.target_base_url.as_deref()))
        .or_else(|| {
            project.default_launch_profile.as_ref().and_then(|profile| {
                profile
                    .target_urls
                    .first()
                    .and_then(|url| normalize_optional_string(Some(url.as_str())))
            })
        })
}

fn empty_runtime_profile_for_auth_setup(
    target_base_url: Option<String>,
    launch: Option<&nyx_agent_types::product::ProjectLaunchProfile>,
) -> ProjectRuntimeProfile {
    let launch_target = launch
        .and_then(|profile| profile.target_urls.first())
        .and_then(|url| normalize_optional_string(Some(url.as_str())));
    let target = target_base_url.or(launch_target);
    ProjectRuntimeProfile {
        build_commands: Vec::new(),
        start_commands: Vec::new(),
        health_check_url: target.clone(),
        health_check_command: None,
        target_base_url: target,
        allowed_hosts: Vec::new(),
        env_vars: Vec::new(),
        auth_profiles: Vec::new(),
        env_file: None,
        timeout_seconds: None,
    }
}

#[derive(Debug, Default)]
struct AuthSetupDiscovery {
    login_paths: Vec<String>,
    object_routes: Vec<String>,
    dev_mail_paths: Vec<String>,
    credentials: AuthSetupCredentialDiscovery,
    files_inspected: usize,
    admin_signal: bool,
    otp_signal: bool,
}

#[derive(Debug, Clone, Default)]
struct AuthSetupCredentialDiscovery {
    exact_env: HashMap<String, String>,
    by_role: HashMap<String, AuthSetupRoleCredentials>,
}

#[derive(Debug, Clone, Default)]
struct AuthSetupRoleCredentials {
    email: Option<String>,
    username: Option<String>,
    password: Option<String>,
    bearer_token: Option<String>,
    cookie: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct AuthSetupEnvResolution {
    values_added: usize,
    values_filled: usize,
    refs_resolved: Vec<String>,
    refs_missing: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthSetupCredentialKind {
    Email,
    Username,
    Password,
    BearerToken,
    Cookie,
    ExactOnly,
}

fn auth_setup_workspace_roots(
    repos: &[RepoRecord],
    state_repos_dir: Option<&FsPath>,
) -> Vec<PathBuf> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for repo in repos {
        if matches!(repo.source_kind.as_str(), "local" | "local-path") {
            let path = PathBuf::from(&repo.source_url_or_path);
            if path.is_dir() && seen.insert(path.clone()) {
                out.push(path);
            }
        }
        if let Some(root) = state_repos_dir {
            let path = root.join(&repo.name);
            if path.is_dir() && seen.insert(path.clone()) {
                out.push(path);
            }
        }
    }
    out
}

fn discover_auth_setup(workspace_paths: &[PathBuf]) -> AuthSetupDiscovery {
    let mut discovery = AuthSetupDiscovery::default();
    let path_re =
        Regex::new(r#"(?i)["'`](/[^"'`\s]*?(?:login|signin|sign-in|session|auth)[^"'`\s]*)["'`]"#)
            .expect("auth setup path regex");
    let object_re = Regex::new(
        r#"(?i)["'`](/[^"'`\s]*(?:projects|invoices|accounts|documents|orders|users|tenants|orgs)[^"'`\s]*/(?::[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\}|[0-9A-Fa-f-]{4,})[^"'`\s]*)["'`]"#,
    )
    .expect("auth setup object-route regex");
    let dev_mail_re =
        Regex::new(r#"(?i)["'`](/[^"'`\s]*(?:dev[-_]mail|mailpit|mailhog|mailbox)[^"'`\s]*)["'`]"#)
            .expect("auth setup dev-mail path regex");
    for root in workspace_paths {
        discover_auth_setup_in_root(root, &path_re, &object_re, &dev_mail_re, &mut discovery);
    }
    discovery.login_paths = dedupe_setup_paths(discovery.login_paths);
    discovery.object_routes = dedupe_setup_paths(discovery.object_routes);
    discovery.dev_mail_paths = dedupe_setup_paths(discovery.dev_mail_paths);
    discovery
}

fn discover_auth_setup_in_root(
    root: &FsPath,
    path_re: &Regex,
    object_re: &Regex,
    dev_mail_re: &Regex,
    discovery: &mut AuthSetupDiscovery,
) {
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((path, depth)) = stack.pop() {
        if discovery.files_inspected >= 1_000 || depth > 8 {
            break;
        }
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_dir() {
            if should_skip_auth_setup_dir(&path) {
                continue;
            }
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries.flatten() {
                    stack.push((entry.path(), depth + 1));
                }
            }
            continue;
        }
        if !meta.is_file() || meta.len() > 256 * 1024 || !is_auth_setup_scannable_file(&path) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        discovery.files_inspected += 1;
        let lower = text.to_ascii_lowercase();
        if lower.contains("/admin") || lower.contains("requireadmin") || lower.contains("is_admin")
        {
            discovery.admin_signal = true;
        }
        if lower.contains("otp")
            || lower.contains("one-time")
            || lower.contains("one time")
            || lower.contains("login code")
            || lower.contains("magic code")
            || lower.contains("verification code")
            || lower.contains("dev-mail")
            || lower.contains("dev_mail")
            || lower.contains("mailpit")
            || lower.contains("mailhog")
        {
            discovery.otp_signal = true;
        }
        for cap in path_re.captures_iter(&text) {
            if let Some(path) = cap.get(1).map(|m| m.as_str()) {
                if auth_setup_path_is_login_candidate(path) {
                    discovery.login_paths.push(path.to_string());
                }
            }
        }
        for cap in dev_mail_re.captures_iter(&text) {
            if let Some(path) = cap.get(1).map(|m| m.as_str()) {
                discovery.dev_mail_paths.push(path.to_string());
            }
        }
        for cap in object_re.captures_iter(&text) {
            if let Some(path) = cap.get(1).map(|m| m.as_str()) {
                discovery.object_routes.push(path.to_string());
            }
        }
        discover_auth_setup_credentials_in_text(&text, &mut discovery.credentials);
    }
}

fn should_skip_auth_setup_dir(path: &FsPath) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    matches!(
        name,
        ".git" | "node_modules" | "target" | "dist" | "build" | ".next" | "coverage" | "vendor"
    )
}

fn is_auth_setup_extension(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "js" | "jsx"
            | "ts"
            | "tsx"
            | "mjs"
            | "cjs"
            | "rs"
            | "py"
            | "rb"
            | "go"
            | "php"
            | "java"
            | "kt"
            | "cs"
            | "html"
            | "vue"
            | "svelte"
            | "json"
            | "jsonl"
            | "toml"
            | "yaml"
            | "yml"
            | "env"
    )
}

fn is_auth_setup_scannable_file(path: &FsPath) -> bool {
    if path.extension().and_then(|e| e.to_str()).is_some_and(is_auth_setup_extension) {
        return true;
    }
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower == ".env"
        || lower.starts_with(".env.")
        || lower.ends_with(".env")
        || matches!(lower.as_str(), "seed" | "seeds" | "fixtures")
}

fn discover_auth_setup_credentials_in_text(
    text: &str,
    credentials: &mut AuthSetupCredentialDiscovery,
) {
    let env_re = Regex::new(
        r#"(?m)(?:^|[\s,{])["']?([A-Z][A-Z0-9_]*(?:EMAIL|USERNAME|PASSWORD|TOKEN|COOKIE)[A-Z0-9_]*)["']?\s*[:=]\s*["']?([^"'\r\n#;,]+)["']?"#,
    )
    .expect("auth setup credential env regex");
    for cap in env_re.captures_iter(text) {
        let Some(name) = cap.get(1).map(|m| m.as_str().trim()) else {
            continue;
        };
        let Some(raw_value) = cap.get(2).map(|m| m.as_str()) else {
            continue;
        };
        let Some(kind) = credential_kind_for_env_name(name) else {
            continue;
        };
        let Some(value) = normalize_credential_literal(raw_value, kind) else {
            continue;
        };
        credentials.exact_env.entry(name.to_string()).or_insert_with(|| value.clone());
        if let Some(role_slug) = role_slug_from_env_name(name) {
            insert_role_credential(credentials, &role_slug, kind, value);
        }
    }

    let keyed_object_re =
        Regex::new(r#"(?is)([A-Za-z][A-Za-z0-9_-]{1,48})\s*:\s*\{([^{}]{0,1600})\}"#)
            .expect("auth setup keyed credential object regex");
    for cap in keyed_object_re.captures_iter(text) {
        let Some(key) = cap.get(1).map(|m| m.as_str()) else {
            continue;
        };
        let Some(body) = cap.get(2).map(|m| m.as_str()) else {
            continue;
        };
        discover_auth_setup_credentials_in_object(Some(key), body, credentials);
    }

    let object_re =
        Regex::new(r#"(?is)\{([^{}]{0,1600})\}"#).expect("auth setup credential object regex");
    for cap in object_re.captures_iter(text) {
        let Some(body) = cap.get(1).map(|m| m.as_str()) else {
            continue;
        };
        discover_auth_setup_credentials_in_object(None, body, credentials);
    }
}

fn discover_auth_setup_credentials_in_object(
    parent_key: Option<&str>,
    body: &str,
    credentials: &mut AuthSetupCredentialDiscovery,
) {
    let email = extract_literal_field(body, &["email", "email_address", "emailAddress"])
        .and_then(|v| normalize_credential_literal(&v, AuthSetupCredentialKind::Email));
    let username = extract_literal_field(body, &["username", "user_name", "login"])
        .and_then(|v| normalize_credential_literal(&v, AuthSetupCredentialKind::Username));
    let password = extract_literal_field(body, &["password", "pass", "plainPassword"])
        .and_then(|v| normalize_credential_literal(&v, AuthSetupCredentialKind::Password));
    if password.is_none() && email.is_none() && username.is_none() {
        return;
    }
    let role = extract_literal_field(body, &["role", "type", "kind"]);
    let role_slug = role
        .as_deref()
        .and_then(credential_role_slug)
        .or_else(|| parent_key.and_then(credential_role_slug))
        .or_else(|| email.as_deref().and_then(role_slug_from_email))
        .or_else(|| username.as_deref().and_then(credential_role_slug));
    let Some(role_slug) = role_slug else {
        return;
    };
    if let Some(value) = email {
        insert_role_credential(credentials, &role_slug, AuthSetupCredentialKind::Email, value);
    }
    if let Some(value) = username {
        insert_role_credential(credentials, &role_slug, AuthSetupCredentialKind::Username, value);
    }
    if let Some(value) = password {
        insert_role_credential(credentials, &role_slug, AuthSetupCredentialKind::Password, value);
    }
}

fn extract_literal_field(body: &str, fields: &[&str]) -> Option<String> {
    for field in fields {
        let field_re = Regex::new(&format!(
            r#"(?i)["']?{}["']?\s*[:=]\s*["']([^"'\r\n]+)["']"#,
            regex::escape(field)
        ))
        .ok()?;
        if let Some(value) =
            field_re.captures(body).and_then(|cap| cap.get(1).map(|m| m.as_str().trim()))
        {
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn normalize_credential_literal(value: &str, kind: AuthSetupCredentialKind) -> Option<String> {
    let value = value.trim().trim_matches(',').trim();
    if value.is_empty() || value.len() > 512 {
        return None;
    }
    let lower = value.to_ascii_lowercase();
    if lower.contains("process.env")
        || lower.contains("import.meta.env")
        || lower.contains("dotenv")
        || value.contains("${")
        || value.contains("{{")
        || value.contains('<')
        || value.contains('>')
        || lower.contains("replace_me")
        || lower.contains("changeme")
        || lower.contains("todo")
    {
        return None;
    }
    if kind == AuthSetupCredentialKind::Email && !value.contains('@') {
        return None;
    }
    if kind == AuthSetupCredentialKind::Password
        && (lower.contains("bcrypt") || lower.contains("argon2") || value.starts_with("$2"))
    {
        return None;
    }
    Some(value.to_string())
}

fn credential_kind_for_env_name(name: &str) -> Option<AuthSetupCredentialKind> {
    let upper = name.to_ascii_uppercase();
    if upper.ends_with("_EMAIL") {
        Some(AuthSetupCredentialKind::Email)
    } else if upper.ends_with("_USERNAME") || upper.ends_with("_USER") || upper.ends_with("_LOGIN")
    {
        Some(AuthSetupCredentialKind::Username)
    } else if upper.ends_with("_PASSWORD") || upper.ends_with("_PASS") {
        Some(AuthSetupCredentialKind::Password)
    } else if upper.ends_with("_TOKEN") || upper.ends_with("_BEARER_TOKEN") {
        Some(AuthSetupCredentialKind::BearerToken)
    } else if upper.ends_with("_COOKIE") || upper.ends_with("_SESSION_COOKIE") {
        Some(AuthSetupCredentialKind::Cookie)
    } else {
        None
    }
}

fn role_slug_from_env_name(name: &str) -> Option<String> {
    let mut stem = name.trim().trim_start_matches("NYX_AGENT_").to_string();
    for suffix in [
        "_SESSION_COOKIE",
        "_BEARER_TOKEN",
        "_PASSWORD",
        "_USERNAME",
        "_COOKIE",
        "_EMAIL",
        "_LOGIN",
        "_TOKEN",
        "_PASS",
        "_USER",
    ] {
        if stem.to_ascii_uppercase().ends_with(suffix) {
            let new_len = stem.len().saturating_sub(suffix.len());
            stem.truncate(new_len);
            break;
        }
    }
    credential_role_slug(&stem)
}

fn role_slug_from_email(email: &str) -> Option<String> {
    let local = email.split('@').next()?.split('+').next().unwrap_or_default();
    credential_role_slug(local)
}

fn credential_role_slug(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || credential_role_slug_is_generic(value) {
        return None;
    }
    let mut out = String::new();
    let mut prev_lower_or_digit = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            if ch.is_ascii_uppercase() && prev_lower_or_digit && !out.ends_with('_') {
                out.push('_');
            }
            out.push(ch.to_ascii_uppercase());
            prev_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        } else {
            if !out.ends_with('_') {
                out.push('_');
            }
            prev_lower_or_digit = false;
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() || credential_role_slug_is_generic(&out) {
        None
    } else {
        Some(out)
    }
}

fn credential_role_slug_is_generic(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "user"
            | "users"
            | "account"
            | "accounts"
            | "profile"
            | "profiles"
            | "credential"
            | "credentials"
            | "auth"
            | "login"
            | "data"
            | "test"
            | "tests"
            | "test_user"
            | "test_users"
    )
}

fn insert_role_credential(
    credentials: &mut AuthSetupCredentialDiscovery,
    role_slug: &str,
    kind: AuthSetupCredentialKind,
    value: String,
) {
    let entry = credentials.by_role.entry(role_slug.to_string()).or_default();
    let slot = match kind {
        AuthSetupCredentialKind::Email => &mut entry.email,
        AuthSetupCredentialKind::Username => &mut entry.username,
        AuthSetupCredentialKind::Password => &mut entry.password,
        AuthSetupCredentialKind::BearerToken => &mut entry.bearer_token,
        AuthSetupCredentialKind::Cookie => &mut entry.cookie,
        AuthSetupCredentialKind::ExactOnly => return,
    };
    if slot.as_deref().is_none_or(str::is_empty) {
        *slot = Some(value);
    }
}

fn auth_setup_path_is_login_candidate(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.contains("login")
        || lower.contains("signin")
        || lower.contains("sign-in")
        || lower.contains("/session")
        || lower.contains("/auth")
}

fn dedupe_setup_paths(paths: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for path in paths {
        let trimmed = path.trim();
        if trimmed.is_empty() || trimmed.contains("..") {
            continue;
        }
        let normalized = trimmed.trim_end_matches('/').to_string();
        if seen.insert(normalized.clone()) {
            out.push(normalized);
        }
    }
    out.sort_by_key(|p| {
        let lower = p.to_ascii_lowercase();
        (!lower.contains("login") && !lower.contains("signin"), !lower.contains("/api/"), p.len())
    });
    out
}

fn auth_setup_roles(requested: &[String], discovery: &AuthSetupDiscovery) -> Vec<String> {
    let mut roles =
        requested.iter().filter_map(|role| normalize_role_name(role)).collect::<Vec<_>>();
    if roles.is_empty() {
        roles.extend(["user_a".to_string(), "user_b".to_string()]);
        if discovery.admin_signal {
            roles.push("admin".to_string());
        }
    }
    let mut seen = BTreeSet::new();
    roles.retain(|role| seen.insert(role.clone()));
    roles
}

fn normalize_role_name(role: &str) -> Option<String> {
    let role = role.trim();
    if role.is_empty() || role.eq_ignore_ascii_case("anonymous") {
        return None;
    }
    Some(role.to_string())
}

#[allow(clippy::type_complexity)]
fn apply_agent_auth_setup_output(
    profiles: &mut Vec<ProjectAuthProfile>,
    output: AuthSetupAgentOutput,
    fallback_login_path: Option<String>,
    seeded_objects: &[ProjectAuthOwnedObject],
) -> (Vec<String>, Vec<String>, Vec<String>, AuthSetupVerification, Option<String>, usize, usize) {
    let roles = if output.roles.is_empty() {
        output
            .profiles
            .iter()
            .filter_map(|profile| normalize_role_name(&profile.role))
            .collect::<Vec<_>>()
    } else {
        output.roles.clone()
    };
    let login_paths = output.login_paths.clone();
    let object_routes = output.object_routes.clone();
    let verification = output.verification.clone();
    let message = Some(output.message);
    let (profiles_added, profiles_updated) = merge_auth_setup_profile_records(
        profiles,
        output.profiles,
        fallback_login_path,
        seeded_objects,
    );
    (roles, login_paths, object_routes, verification, message, profiles_added, profiles_updated)
}

fn merge_auth_setup_profile_records(
    profiles: &mut Vec<ProjectAuthProfile>,
    candidates: Vec<ProjectAuthProfile>,
    fallback_login_path: Option<String>,
    seeded_objects: &[ProjectAuthOwnedObject],
) -> (usize, usize) {
    let mut added = 0usize;
    let mut updated = 0usize;
    for candidate in candidates {
        let Some(candidate) = finalize_auth_setup_candidate(
            candidate,
            fallback_login_path.as_deref(),
            seeded_objects,
        ) else {
            continue;
        };
        if let Some(existing) = profiles.iter_mut().find(|profile| profile.role == candidate.role) {
            if merge_auth_setup_candidate(existing, candidate) {
                updated += 1;
            }
        } else {
            profiles.push(candidate);
            added += 1;
        }
    }
    (added, updated)
}

fn finalize_auth_setup_candidate(
    mut profile: ProjectAuthProfile,
    fallback_login_path: Option<&str>,
    seeded_objects: &[ProjectAuthOwnedObject],
) -> Option<ProjectAuthProfile> {
    profile.role = normalize_role_name(&profile.role)?;
    normalize_auth_setup_identity_refs(&mut profile);
    normalize_auth_setup_otp_mode(&mut profile);
    if profile.mode == ProjectAuthMode::Anonymous {
        profile.mode = ProjectAuthMode::AiAuto;
    }
    if profile.label.as_deref().is_none_or(|label| label.trim().is_empty()) {
        profile.label = Some(format!("AI setup {}", profile.role));
    }
    if profile.login_url.as_deref().is_none_or(|url| url.trim().is_empty()) {
        profile.login_url = fallback_login_path.map(str::to_string);
    }
    if !auth_setup_profile_has_secret_ref(&profile) {
        let role_env = env_role_slug(&profile.role);
        profile.username_env = Some(format!("NYX_AGENT_{role_env}_USERNAME"));
        profile.password_env = Some(format!("NYX_AGENT_{role_env}_PASSWORD"));
    }
    if profile.owned_objects.is_empty() {
        profile.owned_objects = seeded_objects.to_vec();
    }
    Some(profile)
}

fn normalize_auth_setup_identity_refs(profile: &mut ProjectAuthProfile) {
    let Some(username_env) = profile.username_env.as_deref().map(str::trim) else {
        return;
    };
    if !profile.login_email_env.as_deref().is_none_or(|v| v.trim().is_empty()) {
        return;
    }
    if credential_kind_for_env_name(username_env) == Some(AuthSetupCredentialKind::Email) {
        profile.login_email_env = Some(username_env.to_string());
        profile.username_env = None;
    }
}

fn normalize_auth_setup_otp_mode(profile: &mut ProjectAuthProfile) {
    if profile
        .otp_source
        .as_ref()
        .is_some_and(|source| source.kind == ProjectOtpSourceKind::Mailbox)
    {
        profile.mode = ProjectAuthMode::OtpEmailMailbox;
    }
}

fn merge_auth_setup_candidate(
    existing: &mut ProjectAuthProfile,
    candidate: ProjectAuthProfile,
) -> bool {
    let before = existing.clone();
    existing.mode = candidate.mode;
    merge_option(&mut existing.label, candidate.label);
    merge_option(&mut existing.session_cache_ttl_seconds, candidate.session_cache_ttl_seconds);
    merge_option(&mut existing.session_import_path, candidate.session_import_path);
    merge_option(&mut existing.login_url, candidate.login_url);
    merge_option(&mut existing.username, candidate.username);
    merge_option(&mut existing.username_env, candidate.username_env);
    merge_option(&mut existing.login_email_env, candidate.login_email_env);
    merge_option(&mut existing.password_env, candidate.password_env);
    merge_option(&mut existing.password_secret_ref, candidate.password_secret_ref);
    merge_option(&mut existing.cookie_env, candidate.cookie_env);
    merge_option(&mut existing.bearer_token_env, candidate.bearer_token_env);
    if !candidate.headers.is_empty() {
        existing.headers = candidate.headers;
    }
    merge_option(&mut existing.otp_source, candidate.otp_source);
    if !candidate.post_login_assertions.is_empty() {
        existing.post_login_assertions = candidate.post_login_assertions;
    }
    merge_option(&mut existing.post_login_assertion, candidate.post_login_assertion);
    merge_option(&mut existing.custom_command, candidate.custom_command);
    if !candidate.owned_objects.is_empty() {
        existing.owned_objects = candidate.owned_objects;
    }
    *existing != before
}

fn merge_option<T>(slot: &mut Option<T>, candidate: Option<T>) {
    if candidate.is_some() {
        *slot = candidate;
    }
}

fn auth_setup_profile_has_secret_ref(profile: &ProjectAuthProfile) -> bool {
    profile.session_import_path.is_some()
        || profile.username_env.is_some()
        || profile.login_email_env.is_some()
        || profile.password_env.is_some()
        || profile.password_secret_ref.is_some()
        || profile.cookie_env.is_some()
        || profile.bearer_token_env.is_some()
        || !profile.headers.is_empty()
        || profile.custom_command.is_some()
}

fn apply_discovered_otp_hints(
    runtime_profile: &mut ProjectRuntimeProfile,
    target_base_url: Option<&str>,
    discovery: &AuthSetupDiscovery,
) {
    if !discovery.otp_signal && discovery.dev_mail_paths.is_empty() {
        return;
    }
    let mailbox_url =
        discovery.dev_mail_paths.first().and_then(|path| absolute_local_url(target_base_url, path));
    for profile in &mut runtime_profile.auth_profiles {
        if profile.mode != ProjectAuthMode::AiAuto
            && profile.mode != ProjectAuthMode::OtpEmailMailbox
        {
            continue;
        }
        if mailbox_url.is_some() {
            profile.mode = ProjectAuthMode::OtpEmailMailbox;
            let email_env = profile
                .login_email_env
                .clone()
                .or_else(|| profile.username_env.clone())
                .or_else(|| Some(format!("NYX_AGENT_{}_EMAIL", env_role_slug(&profile.role))));
            let source = profile.otp_source.get_or_insert_with(|| ProjectOtpSourceConfig {
                kind: ProjectOtpSourceKind::Mailbox,
                mailbox_url: None,
                email_env: None,
                subject_contains: Some("code".to_string()),
                body_regex: Some(r"\b(\d{4,8})\b".to_string()),
                imap_url_env: None,
                imap_username_env: None,
                imap_password_env: None,
            });
            source.kind = ProjectOtpSourceKind::Mailbox;
            if source.mailbox_url.as_deref().is_none_or(|url| url.trim().is_empty()) {
                source.mailbox_url = mailbox_url.clone();
            }
            if source.email_env.as_deref().is_none_or(|env| env.trim().is_empty()) {
                source.email_env = email_env;
            }
            if source.subject_contains.as_deref().is_none_or(|value| value.trim().is_empty()) {
                source.subject_contains = Some("code".to_string());
            }
            if source.body_regex.as_deref().is_none_or(|value| value.trim().is_empty()) {
                source.body_regex = Some(r"\b(\d{4,8})\b".to_string());
            }
        }
    }
}

fn absolute_local_url(target_base_url: Option<&str>, path: &str) -> Option<String> {
    let path = path.trim();
    if path.starts_with("http://") || path.starts_with("https://") {
        return Some(path.to_string());
    }
    let target = reqwest::Url::parse(target_base_url?).ok()?;
    let mut url = target.join(path).ok()?;
    if !url.path().ends_with('/') {
        let next = format!("{}/", url.path());
        url.set_path(&next);
    }
    Some(url.to_string())
}

fn apply_discovered_auth_env_values(
    runtime_profile: &mut ProjectRuntimeProfile,
    credentials: &AuthSetupCredentialDiscovery,
) -> AuthSetupEnvResolution {
    let mut report = AuthSetupEnvResolution::default();
    let auth_profiles = runtime_profile.auth_profiles.clone();
    for profile in &auth_profiles {
        let role_slug = env_role_slug(&profile.role);
        maybe_apply_auth_env_value(
            &mut runtime_profile.env_vars,
            profile.username_env.as_deref(),
            &role_slug,
            AuthSetupCredentialKind::Username,
            credentials,
            &mut report,
        );
        maybe_apply_auth_env_value(
            &mut runtime_profile.env_vars,
            profile.login_email_env.as_deref(),
            &role_slug,
            AuthSetupCredentialKind::Email,
            credentials,
            &mut report,
        );
        maybe_apply_auth_env_value(
            &mut runtime_profile.env_vars,
            profile.password_env.as_deref(),
            &role_slug,
            AuthSetupCredentialKind::Password,
            credentials,
            &mut report,
        );
        maybe_apply_auth_env_value(
            &mut runtime_profile.env_vars,
            profile.bearer_token_env.as_deref(),
            &role_slug,
            AuthSetupCredentialKind::BearerToken,
            credentials,
            &mut report,
        );
        maybe_apply_auth_env_value(
            &mut runtime_profile.env_vars,
            profile.cookie_env.as_deref(),
            &role_slug,
            AuthSetupCredentialKind::Cookie,
            credentials,
            &mut report,
        );
        for header in &profile.headers {
            maybe_apply_auth_env_value(
                &mut runtime_profile.env_vars,
                header.value_env.as_deref(),
                &role_slug,
                AuthSetupCredentialKind::ExactOnly,
                credentials,
                &mut report,
            );
        }
        if let Some(source) = &profile.otp_source {
            maybe_apply_auth_env_value(
                &mut runtime_profile.env_vars,
                source.email_env.as_deref(),
                &role_slug,
                AuthSetupCredentialKind::Email,
                credentials,
                &mut report,
            );
        }
    }

    let resolved_env = runtime_env_values(&runtime_profile.env_vars);
    let mut seen = BTreeSet::new();
    for profile in &runtime_profile.auth_profiles {
        for env in auth_setup_env_refs(profile) {
            if !seen.insert(env.clone()) {
                continue;
            }
            if resolved_env.get(&env).is_some_and(|value| !value.is_empty())
                || std::env::var_os(&env).is_some()
            {
                report.refs_resolved.push(env);
            } else {
                report.refs_missing.push(env);
            }
        }
    }
    report.refs_resolved.sort();
    report.refs_missing.sort();
    report
}

fn maybe_apply_auth_env_value(
    env_vars: &mut Vec<ProjectRuntimeEnvVar>,
    env_name: Option<&str>,
    role_slug: &str,
    kind: AuthSetupCredentialKind,
    credentials: &AuthSetupCredentialDiscovery,
    report: &mut AuthSetupEnvResolution,
) {
    let Some(env_name) = env_name.map(str::trim).filter(|name| !name.is_empty()) else {
        return;
    };
    let Some(value) = credential_value_for_env(env_name, role_slug, kind, credentials) else {
        return;
    };
    let secret = matches!(
        kind,
        AuthSetupCredentialKind::Password
            | AuthSetupCredentialKind::BearerToken
            | AuthSetupCredentialKind::Cookie
            | AuthSetupCredentialKind::ExactOnly
    );
    upsert_runtime_env_value(env_vars, env_name, &value, secret, report);
}

fn credential_value_for_env(
    env_name: &str,
    role_slug: &str,
    kind: AuthSetupCredentialKind,
    credentials: &AuthSetupCredentialDiscovery,
) -> Option<String> {
    if let Some(value) = credentials.exact_env.get(env_name).filter(|value| !value.is_empty()) {
        return Some(value.clone());
    }
    let role_credentials = credentials.by_role.get(role_slug)?;
    match kind {
        AuthSetupCredentialKind::Email => role_credentials.email.clone(),
        AuthSetupCredentialKind::Username => {
            role_credentials.username.clone().or_else(|| role_credentials.email.clone())
        }
        AuthSetupCredentialKind::Password => role_credentials.password.clone(),
        AuthSetupCredentialKind::BearerToken => role_credentials.bearer_token.clone(),
        AuthSetupCredentialKind::Cookie => role_credentials.cookie.clone(),
        AuthSetupCredentialKind::ExactOnly => None,
    }
}

fn upsert_runtime_env_value(
    env_vars: &mut Vec<ProjectRuntimeEnvVar>,
    name: &str,
    value: &str,
    secret: bool,
    report: &mut AuthSetupEnvResolution,
) {
    if let Some(existing) = env_vars.iter_mut().find(|var| var.name.trim() == name) {
        if existing.value.is_empty() {
            existing.value = value.to_string();
            existing.secret = existing.secret || secret;
            report.values_filled += 1;
        } else if secret && !existing.secret {
            existing.secret = true;
        }
        return;
    }
    env_vars.push(ProjectRuntimeEnvVar {
        name: name.to_string(),
        value: value.to_string(),
        secret,
    });
    report.values_added += 1;
}

fn runtime_env_values(env_vars: &[ProjectRuntimeEnvVar]) -> HashMap<String, String> {
    env_vars
        .iter()
        .filter_map(|var| {
            let name = var.name.trim();
            if name.is_empty() {
                None
            } else {
                Some((name.to_string(), var.value.clone()))
            }
        })
        .collect()
}

fn auth_setup_env_refs(profile: &ProjectAuthProfile) -> Vec<String> {
    let mut refs = Vec::new();
    refs.extend(profile.username_env.iter().cloned());
    refs.extend(profile.login_email_env.iter().cloned());
    refs.extend(profile.password_env.iter().cloned());
    refs.extend(profile.cookie_env.iter().cloned());
    refs.extend(profile.bearer_token_env.iter().cloned());
    refs.extend(profile.headers.iter().filter_map(|header| header.value_env.clone()));
    if let Some(source) = &profile.otp_source {
        refs.extend(source.email_env.iter().cloned());
        refs.extend(source.imap_url_env.iter().cloned());
        refs.extend(source.imap_username_env.iter().cloned());
        refs.extend(source.imap_password_env.iter().cloned());
    }
    refs.into_iter().map(|env| env.trim().to_string()).filter(|env| !env.is_empty()).collect()
}

fn apply_auth_env_resolution_to_verification(
    verification: &mut AuthSetupVerification,
    report: &AuthSetupEnvResolution,
) {
    let saved = report.values_added + report.values_filled;
    if saved > 0 {
        verification
            .checks
            .push(format!("Saved {saved} auth credential env value(s) from repo-local hints."));
    }
    if !report.refs_resolved.is_empty() {
        verification.checks.push(format!(
            "Resolved {} auth env ref(s) for generated profiles.",
            report.refs_resolved.len()
        ));
    }
    if !report.refs_missing.is_empty() {
        verification
            .warnings
            .push(format!("Missing auth env value(s): {}.", report.refs_missing.join(", ")));
        verification.status = AuthSetupVerificationStatus::NeedsReview;
    }
}

fn auth_env_resolution_message(report: &AuthSetupEnvResolution) -> Option<String> {
    if report.refs_missing.is_empty() {
        return None;
    }
    Some(format!("Auth setup still needs value(s) for {}.", report.refs_missing.join(", ")))
}

fn static_auth_setup_verification(
    discovery: &AuthSetupDiscovery,
    fallback_warning: Option<String>,
) -> AuthSetupVerification {
    let mut checks = Vec::new();
    let mut warnings = Vec::new();
    if discovery.files_inspected > 0 {
        checks.push(format!(
            "Static repo scan inspected {} source file(s).",
            discovery.files_inspected
        ));
    } else {
        warnings.push("No local repo files were available for auth setup.".to_string());
    }
    if discovery.login_paths.is_empty() {
        warnings.push("No login or session route was discovered.".to_string());
    } else {
        checks.push(format!("Discovered login/session path {}.", discovery.login_paths[0]));
    }
    if discovery.object_routes.is_empty() {
        warnings.push("No object ownership routes were discovered.".to_string());
    } else {
        checks.push(format!(
            "Discovered {} object ownership route hint(s).",
            discovery.object_routes.len()
        ));
    }
    if !discovery.dev_mail_paths.is_empty() {
        checks.push(format!("Discovered dev-mail route {}.", discovery.dev_mail_paths[0]));
        warnings.push(
            "Detected OTP/dev-mail auth; profile setup recorded a mailbox OTP source, but live OTP browser capture is not implemented yet."
                .to_string(),
        );
    } else if discovery.otp_signal {
        warnings.push(
            "Detected OTP-like auth code hints, but no local dev-mail mailbox route was discovered."
                .to_string(),
        );
    }
    if let Some(warning) = fallback_warning {
        warnings.push(warning);
    }
    AuthSetupVerification {
        status: if warnings.is_empty() {
            AuthSetupVerificationStatus::Verified
        } else if discovery.files_inspected == 0 {
            AuthSetupVerificationStatus::Skipped
        } else {
            AuthSetupVerificationStatus::NeedsReview
        },
        checks,
        warnings,
    }
}

fn auth_setup_response_message(
    agent_used: bool,
    profiles_added: usize,
    profiles_updated: usize,
    files_inspected: usize,
    verification: &AuthSetupVerification,
    agent_message: Option<String>,
    fallback_warning: Option<String>,
) -> String {
    if let Some(message) = agent_message.filter(|message| !message.trim().is_empty()) {
        if let Some(warning) = fallback_warning {
            return format!("{message} {warning}");
        }
        return message;
    }
    let changed = profiles_added + profiles_updated;
    let verification_phrase = match verification.status {
        AuthSetupVerificationStatus::Verified => "verification passed",
        AuthSetupVerificationStatus::NeedsReview => "verification needs review",
        AuthSetupVerificationStatus::Skipped => "verification skipped",
    };
    let mut message = if agent_used {
        if changed == 0 {
            format!("Auth exploration agent kept the existing role profiles unchanged; {verification_phrase}.")
        } else {
            format!(
                "Auth exploration agent saved {changed} repo-specific role profile(s); {verification_phrase}."
            )
        }
    } else if changed == 0 {
        format!("Auth setup kept the existing role profiles unchanged; {verification_phrase}.")
    } else {
        format!(
            "Auth setup saved {changed} role profile(s) from {files_inspected} inspected source file(s); {verification_phrase}."
        )
    };
    if let Some(warning) = fallback_warning {
        message.push(' ');
        message.push_str(&warning);
    }
    message
}

fn merge_auth_setup_profiles(
    profiles: &mut Vec<ProjectAuthProfile>,
    roles: &[String],
    login_path: Option<String>,
    seeded_objects: &[ProjectAuthOwnedObject],
) -> (usize, usize) {
    let mut added = 0usize;
    let mut updated = 0usize;
    for role in roles {
        if let Some(existing) = profiles.iter_mut().find(|profile| profile.role == *role) {
            if fill_auth_setup_profile(existing, login_path.as_deref(), seeded_objects) {
                updated += 1;
            }
        } else {
            profiles.push(auth_setup_profile(role, login_path.as_deref(), seeded_objects));
            added += 1;
        }
    }
    (added, updated)
}

fn auth_setup_profile(
    role: &str,
    login_path: Option<&str>,
    seeded_objects: &[ProjectAuthOwnedObject],
) -> ProjectAuthProfile {
    let role_env = env_role_slug(role);
    ProjectAuthProfile {
        role: role.to_string(),
        role_aliases: Vec::new(),
        mode: ProjectAuthMode::AiAuto,
        label: Some(format!("AI setup {role}")),
        tenant: None,
        session_cache_ttl_seconds: None,
        session_import_path: None,
        login_url: login_path.map(str::to_string),
        username: None,
        username_env: Some(format!("NYX_AGENT_{role_env}_USERNAME")),
        login_email_env: None,
        password_env: Some(format!("NYX_AGENT_{role_env}_PASSWORD")),
        password_secret_ref: None,
        cookie_env: None,
        bearer_token_env: None,
        headers: Vec::new(),
        otp_source: None,
        post_login_assertions: Vec::new(),
        post_login_assertion: None,
        custom_command: None,
        owned_objects: seeded_objects.to_vec(),
    }
}

fn fill_auth_setup_profile(
    profile: &mut ProjectAuthProfile,
    login_path: Option<&str>,
    seeded_objects: &[ProjectAuthOwnedObject],
) -> bool {
    let mut changed = false;
    if profile.login_url.as_deref().is_none_or(|v| v.trim().is_empty()) {
        if let Some(login_path) = login_path {
            profile.login_url = Some(login_path.to_string());
            changed = true;
        }
    }
    let role_env = env_role_slug(&profile.role);
    if profile.username_env.as_deref().is_none_or(|v| v.trim().is_empty())
        && profile.username.as_deref().is_none_or(|v| v.trim().is_empty())
        && profile.login_email_env.as_deref().is_none_or(|v| v.trim().is_empty())
    {
        profile.username_env = Some(format!("NYX_AGENT_{role_env}_USERNAME"));
        changed = true;
    }
    if profile.password_env.as_deref().is_none_or(|v| v.trim().is_empty()) {
        profile.password_env = Some(format!("NYX_AGENT_{role_env}_PASSWORD"));
        changed = true;
    }
    if profile.owned_objects.is_empty() && !seeded_objects.is_empty() {
        profile.owned_objects = seeded_objects.to_vec();
        changed = true;
    }
    changed
}

fn env_role_slug(role: &str) -> String {
    let mut out = String::new();
    for ch in role.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_uppercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "ROLE".to_string()
    } else {
        out
    }
}

fn launch_profile_input_from_runtime(
    profile: &ProjectRuntimeProfile,
    fallback_target: Option<&str>,
) -> ProjectLaunchProfileInput {
    let build_steps: Vec<nyx_agent_types::product::LaunchStep> =
        profile.build_commands.iter().map(runtime_command_to_launch_step).collect();
    let start_steps: Vec<nyx_agent_types::product::LaunchStep> =
        profile.start_commands.iter().map(runtime_command_to_launch_step).collect();
    let mut health_checks = Vec::new();
    if let Some(url) = normalize_optional_string(profile.health_check_url.as_deref()) {
        health_checks.push(nyx_agent_types::product::LaunchHealthCheck {
            kind: "http".to_string(),
            url: Some(url),
            host: None,
            port: None,
            command: None,
            timeout_seconds: profile.timeout_seconds,
        });
    }
    if let Some(cmd) = &profile.health_check_command {
        health_checks.push(nyx_agent_types::product::LaunchHealthCheck {
            kind: "command".to_string(),
            url: None,
            host: None,
            port: None,
            command: Some(runtime_command_to_launch_step(cmd)),
            timeout_seconds: cmd.timeout_seconds.or(profile.timeout_seconds),
        });
    }
    let mut target_urls = Vec::new();
    if let Some(target) = normalize_optional_string(profile.target_base_url.as_deref())
        .or_else(|| normalize_optional_string(fallback_target))
    {
        target_urls.push(target);
    }
    let mut env_refs = Vec::new();
    if let Some(env_file) = normalize_optional_string(profile.env_file.as_deref()) {
        env_refs.push(nyx_agent_types::product::LaunchEnvRef {
            kind: "env-file".to_string(),
            value: env_file,
            secret: true,
        });
    }
    for var in &profile.env_vars {
        if var.name.trim().is_empty() {
            continue;
        }
        env_refs.push(nyx_agent_types::product::LaunchEnvRef {
            kind: "env-var".to_string(),
            value: var.name.trim().to_string(),
            secret: var.secret,
        });
    }
    let mode = if build_steps.is_empty() && start_steps.is_empty() {
        "already-running"
    } else {
        "custom-commands"
    };
    ProjectLaunchProfileInput {
        name: Some("local dev".to_string()),
        mode: Some(mode.to_string()),
        build_steps,
        start_steps,
        seed_steps: Vec::new(),
        reset_steps: Vec::new(),
        login_steps: Vec::new(),
        stop_steps: Vec::new(),
        health_checks,
        target_urls,
        env_refs,
        working_dirs: Vec::new(),
    }
}

fn runtime_command_to_launch_step(
    cmd: &nyx_agent_types::project::ProjectRuntimeCommand,
) -> nyx_agent_types::product::LaunchStep {
    nyx_agent_types::product::LaunchStep {
        command: cmd.command.clone(),
        repo_id: None,
        repo_name: cmd.repo_name.clone(),
        working_directory: cmd.working_directory.clone(),
        timeout_seconds: cmd.timeout_seconds,
        stdin: None,
    }
}

async fn delete_project(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<StatusBody, ApiError> {
    let affected = s.store.projects().delete(&id).await?;
    if affected == 0 {
        return Err(ApiError::NotFound(format!("project `{id}` not found")));
    }
    Ok(StatusBody::ok(format!("deleted {affected} project row(s); repos cascaded")))
}

/// Lightweight stable id helper. Concatenates a slug of `name` with the
/// supplied epoch ms so collisions across rapid creates are vanishingly
/// rare without pulling in a UUID crate dependency.
fn uuid_like(name: &str, now_ms: i64) -> String {
    let slug: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    let trimmed: String = slug
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .take(32)
        .collect();
    format!("{trimmed}-{now_ms:x}")
}

async fn require_project(s: &ServerState, project_id: &str) -> Result<ProjectRecord, ApiError> {
    s.store
        .projects()
        .get(project_id)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("project `{project_id}` not found")))
}

async fn require_project_integration(
    s: &ServerState,
    project_id: &str,
    integration_id: &str,
) -> Result<ProjectIntegrationRecord, ApiError> {
    let row =
        s.store.integrations().get(integration_id).await?.ok_or_else(|| {
            ApiError::NotFound(format!("integration `{integration_id}` not found"))
        })?;
    if row.project_id != project_id {
        return Err(ApiError::NotFound(format!(
            "integration `{integration_id}` not found in project `{project_id}`"
        )));
    }
    Ok(row)
}

fn validate_integration_name(raw: &str) -> Result<String, ApiError> {
    let name = raw.trim();
    if name.is_empty() {
        return Err(ApiError::BadRequest("integration name is required".to_string()));
    }
    if name.len() > 80 {
        return Err(ApiError::BadRequest(
            "integration name must be 80 characters or less".to_string(),
        ));
    }
    Ok(name.to_string())
}

fn validate_integration_events(
    events: &[nyx_agent_types::integration::ProjectIntegrationEvent],
) -> Result<(), ApiError> {
    if events.is_empty() {
        return Err(ApiError::BadRequest("select at least one integration event".to_string()));
    }
    Ok(())
}

async fn get_default_launch_profile(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
) -> Result<Json<nyx_agent_types::product::ProjectLaunchProfile>, ApiError> {
    require_project(&s, &project_id).await?;
    s.store.launch_profiles().get_default(&project_id).await?.map(Json).ok_or_else(|| {
        ApiError::NotFound(format!("default launch profile for project `{project_id}` not found"))
    })
}

async fn patch_default_launch_profile(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
    Json(input): Json<ProjectLaunchProfileInput>,
) -> Result<Json<nyx_agent_types::product::ProjectLaunchProfile>, ApiError> {
    require_project(&s, &project_id).await?;
    validate_launch_profile_input(&input)?;
    let row = s.store.launch_profiles().upsert_default(&project_id, &input, now_epoch_ms()).await?;
    Ok(Json(row))
}

fn validate_launch_profile_input(input: &ProjectLaunchProfileInput) -> Result<(), ApiError> {
    let mode = input.mode.as_deref().unwrap_or("custom-commands");
    if !matches!(mode, "already-running" | "custom-commands" | "docker-compose" | "devcontainer") {
        return Err(ApiError::BadRequest(format!("unknown launch profile mode `{mode}`")));
    }
    for url in &input.target_urls {
        if !is_local_http_url(url) {
            return Err(ApiError::BadRequest(format!(
                "target URL `{url}` must be a local http:// or https:// URL"
            )));
        }
    }
    for check in &input.health_checks {
        if let Some(url) = check.url.as_deref() {
            if !is_local_http_url(url) {
                return Err(ApiError::BadRequest(format!(
                    "health check URL `{url}` must be local"
                )));
            }
        }
    }
    Ok(())
}

fn is_local_http_url(raw: &str) -> bool {
    local_http_url(raw).is_some()
}

fn local_http_url(raw: &str) -> Option<reqwest::Url> {
    let url = reqwest::Url::parse(raw.trim()).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    let host = url.host_str()?;
    let allowed = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::Ipv4Addr>()
            .is_ok_and(|addr| addr.is_loopback() || addr.is_unspecified())
        || host.parse::<std::net::Ipv6Addr>().is_ok_and(|addr| addr.is_loopback());
    allowed.then_some(url)
}

// ---- /projects/:project_id/repos --------------------------------------------

async fn list_project_repos(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<RepoRecord>>, ApiError> {
    require_project(&s, &project_id).await?;
    let rows = s.store.repos().list_by_project(&project_id).await?;
    Ok(Json(rows))
}

async fn create_project_repo(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
    Json(req): Json<CreateRepoRequest>,
) -> Result<Json<RepoRecord>, ApiError> {
    require_project(&s, &project_id).await?;
    if req.name.trim().is_empty() {
        return Err(ApiError::BadRequest("name is required".to_string()));
    }
    if !matches!(req.source_kind.as_str(), "git" | "local-path" | "github" | "gitlab" | "local") {
        return Err(ApiError::BadRequest(format!("unknown source_kind `{}`", req.source_kind)));
    }
    if !req.i_own_this {
        return Err(ApiError::BadRequest(
            "i_own_this must be set to true before the daemon will accept a repo".to_string(),
        ));
    }
    validate_git_auth_ref(&req.source_kind, req.auth_ref.as_deref())?;
    let now = now_epoch_ms();
    let existing = s.store.repos().get_by_project_and_name(&project_id, &req.name).await?;
    // Refuse re-POST against a different project so an operator cannot
    // silently re-home an existing repo via a same-name create call.
    if let Some(row) = &existing {
        if row.project_id != project_id {
            return Err(ApiError::BadRequest(format!(
                "repo `{}` already belongs to project `{}`",
                row.name, row.project_id
            )));
        }
    }
    let rec = RepoRecord {
        id: existing.as_ref().map(|r| r.id.clone()).unwrap_or_else(|| {
            format!("repo-{}", uuid_like(&format!("{project_id}-{}", req.name), now))
        }),
        name: req.name,
        project_id: project_id.clone(),
        source_kind: req.source_kind,
        source_url_or_path: req.source_url_or_path,
        branch: req.branch,
        auth_ref: req.auth_ref,
        i_own_this: req.i_own_this,
        last_scan_run_id: existing.as_ref().and_then(|r| r.last_scan_run_id.clone()),
        last_scan_finished_at: existing.as_ref().and_then(|r| r.last_scan_finished_at),
        created_at: existing.as_ref().map(|r| r.created_at).unwrap_or(now),
        updated_at: now,
    };
    s.store.repos().upsert(&rec).await?;
    Ok(Json(rec))
}

async fn get_project_repo(
    State(s): State<ServerState>,
    Path((project_id, name)): Path<(String, String)>,
) -> Result<Json<RepoRecord>, ApiError> {
    require_project(&s, &project_id).await?;
    let row = s
        .store
        .repos()
        .get_by_project_and_name(&project_id, &name)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("repo `{name}` not found")))?;
    if row.project_id != project_id {
        return Err(ApiError::NotFound(format!(
            "repo `{name}` not found in project `{project_id}`"
        )));
    }
    Ok(Json(row))
}

async fn patch_project_repo(
    State(s): State<ServerState>,
    Path((project_id, name)): Path<(String, String)>,
    Json(req): Json<PatchRepoRequest>,
) -> Result<Json<RepoRecord>, ApiError> {
    require_project(&s, &project_id).await?;
    let existing = s
        .store
        .repos()
        .get_by_project_and_name(&project_id, &name)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("repo `{name}` not found")))?;
    if existing.project_id != project_id {
        return Err(ApiError::NotFound(format!(
            "repo `{name}` not found in project `{project_id}`"
        )));
    }
    if let Some(kind) = req.source_kind.as_deref() {
        if !matches!(kind, "git" | "local-path" | "github" | "gitlab" | "local") {
            return Err(ApiError::BadRequest(format!("unknown source_kind `{kind}`")));
        }
    }
    if let Some(false) = req.i_own_this {
        return Err(ApiError::BadRequest(
            "i_own_this cannot be cleared via PATCH; remove the repo instead".to_string(),
        ));
    }
    let effective_source_kind = req.source_kind.as_deref().unwrap_or(existing.source_kind.as_str());
    let effective_auth_ref: Option<&str> = match &req.auth_ref {
        None => existing.auth_ref.as_deref(),
        Some(None) => None,
        Some(Some(v)) => Some(v.as_str()),
    };
    validate_git_auth_ref(effective_source_kind, effective_auth_ref)?;
    let now = now_epoch_ms();
    let rec = RepoRecord {
        id: existing.id,
        name: existing.name,
        project_id: existing.project_id,
        source_kind: req.source_kind.unwrap_or(existing.source_kind),
        source_url_or_path: req.source_url_or_path.unwrap_or(existing.source_url_or_path),
        branch: match req.branch {
            None => existing.branch,
            Some(next) => next,
        },
        auth_ref: match req.auth_ref {
            None => existing.auth_ref,
            Some(next) => next,
        },
        i_own_this: req.i_own_this.unwrap_or(existing.i_own_this),
        last_scan_run_id: existing.last_scan_run_id,
        last_scan_finished_at: existing.last_scan_finished_at,
        created_at: existing.created_at,
        updated_at: now,
    };
    s.store.repos().upsert(&rec).await?;
    let row = s
        .store
        .repos()
        .get_by_project_and_name(&project_id, &name)
        .await?
        .ok_or_else(|| ApiError::Internal("repo vanished after update".to_string()))?;
    Ok(Json(row))
}

/// Refuse a repo create/patch whose `auth_ref` would fail the same grammar
/// the ingestion crate expects (`ssh-key:<path>` / `token-env:<VAR>` /
/// `gh-app:<id>`). Validation runs only when the effective `source_kind`
/// is `git` / `github` / `gitlab`; other source kinds ignore `auth_ref`,
/// so we do not block on a stale value left over from a kind switch.
fn validate_git_auth_ref(source_kind: &str, auth_ref: Option<&str>) -> Result<(), ApiError> {
    if !matches!(source_kind, "git" | "github" | "gitlab") {
        return Ok(());
    }
    let Some(raw) = auth_ref else {
        return Ok(());
    };
    parse_git_auth(raw).map_err(|err| match err {
        IngestError::AuthMalformed { raw } => ApiError::BadRequest(format!(
            "auth_ref `{raw}` is malformed; expected `ssh-key:<path>`, `token-env:<VAR>`, or \
             `gh-app:<id>`"
        )),
        IngestError::AuthUnknownScheme { scheme } => ApiError::BadRequest(format!(
            "auth_ref scheme `{scheme}` is not supported; use `ssh-key`, `token-env`, or `gh-app`"
        )),
        other => ApiError::BadRequest(format!("auth_ref failed validation: {other}")),
    })?;
    Ok(())
}

async fn delete_project_repo(
    State(s): State<ServerState>,
    Path((project_id, name)): Path<(String, String)>,
) -> Result<StatusBody, ApiError> {
    require_project(&s, &project_id).await?;
    let existing = s
        .store
        .repos()
        .get_by_project_and_name(&project_id, &name)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("repo `{name}` not found")))?;
    if existing.project_id != project_id {
        return Err(ApiError::NotFound(format!(
            "repo `{name}` not found in project `{project_id}`"
        )));
    }
    let affected = s.store.repos().delete_by_project_and_name(&project_id, &name).await?;
    if affected == 0 {
        return Err(ApiError::NotFound(format!("repo `{name}` not found")));
    }
    let mut workspace_msg = String::new();
    if let Some(root) = &s.state_repos_dir {
        let target = root.join(&name);
        if target.is_dir() {
            match std::fs::remove_dir_all(&target) {
                Ok(()) => {
                    workspace_msg = format!(" (workspace {} removed)", target.display());
                }
                Err(err) => {
                    tracing::warn!(
                        repo = %name,
                        path = %target.display(),
                        error = %err,
                        "failed to remove repo workspace; row was still deleted",
                    );
                    workspace_msg =
                        format!(" (workspace {} could not be removed: {err})", target.display());
                }
            }
        }
    }
    Ok(StatusBody::ok(format!("deleted {affected} row(s){workspace_msg}")))
}

// ---- /repos/test ------------------------------------------------------------

/// Lightweight probe wired to the wizard's "test connectivity" button.
/// Performs only a read-only side effect (`git ls-remote` for git
/// sources, `stat` + read of `.git/config` for local-path sources). The
/// `project_id` from the route is validated to exist but otherwise does
/// not affect the probe (the call is stateless).
async fn test_repo_connectivity(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
    Json(req): Json<TestRepoRequest>,
) -> Result<Json<TestRepoResponse>, ApiError> {
    require_project(&s, &project_id).await?;
    match req.source_kind.as_str() {
        "git" | "github" | "gitlab" => {
            let url = req.source_url_or_path.trim();
            if url.is_empty() {
                return Err(ApiError::BadRequest("source_url_or_path is required".to_string()));
            }
            let branch = req.branch.as_deref().map(str::trim).filter(|s| !s.is_empty());
            let (ok, message) = git_ls_remote_probe(url, branch).await;
            Ok(Json(TestRepoResponse { ok, message, on_disk_git_remote: None }))
        }
        "local-path" | "local" => {
            let path = std::path::Path::new(&req.source_url_or_path);
            if !path.exists() {
                return Ok(Json(TestRepoResponse {
                    ok: false,
                    message: format!("path `{}` does not exist", path.display()),
                    on_disk_git_remote: None,
                }));
            }
            if !path.is_dir() {
                return Ok(Json(TestRepoResponse {
                    ok: false,
                    message: format!("path `{}` is not a directory", path.display()),
                    on_disk_git_remote: None,
                }));
            }
            let remote = read_local_git_remote(path);
            let message = match &remote {
                Some(url) => format!(
                    "path readable; on-disk `.git/config` remote = `{url}`. Confirm before adding.",
                ),
                None => "path readable; no `.git/config` remote on disk (untracked directory)."
                    .to_string(),
            };
            Ok(Json(TestRepoResponse { ok: true, message, on_disk_git_remote: remote }))
        }
        other => Err(ApiError::BadRequest(format!("unknown source_kind `{other}`"))),
    }
}

const GIT_PROBE_TIMEOUT: Duration = Duration::from_secs(15);

async fn git_ls_remote_probe(url: &str, branch: Option<&str>) -> (bool, String) {
    let mut cmd = tokio::process::Command::new("git");
    cmd.arg("-c").arg("credential.helper=").arg("ls-remote").arg("--exit-code").arg(url);
    if let Some(b) = branch {
        cmd.arg(format!("refs/heads/{b}"));
    }
    // Match ingestion-path env hardening: no terminal prompts, no user
    // git config bleed.
    cmd.env("GIT_TERMINAL_PROMPT", "0");
    cmd.env("GIT_CONFIG_GLOBAL", "/dev/null");
    cmd.env("GIT_CONFIG_SYSTEM", "/dev/null");
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.stdin(std::process::Stdio::null());
    // Otherwise a timed-out probe leaks the underlying `git ls-remote`
    // process: `tokio::time::timeout` drops the future that owns the
    // `Child`, but `tokio::process::Child` does not kill on drop by
    // default.
    cmd.kill_on_drop(true);

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(err) => return (false, format!("could not spawn git: {err}")),
    };
    let wait = child.wait_with_output();
    match tokio::time::timeout(GIT_PROBE_TIMEOUT, wait).await {
        Ok(Ok(output)) => {
            if output.status.success() {
                let line_count = output.stdout.iter().filter(|b| **b == b'\n').count();
                (
                    true,
                    match branch {
                        Some(b) => format!("ls-remote reached upstream; branch `{b}` exists"),
                        None => format!("ls-remote reached upstream ({line_count} refs visible)"),
                    },
                )
            } else if output.status.code() == Some(2) {
                (
                    false,
                    match branch {
                        Some(b) => format!("upstream reachable but branch `{b}` does not exist"),
                        None => "upstream reachable but has no refs".to_string(),
                    },
                )
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let trimmed = stderr.trim();
                (
                    false,
                    if trimmed.is_empty() {
                        format!("git ls-remote exited with status {}", output.status)
                    } else {
                        format!("git ls-remote failed: {trimmed}")
                    },
                )
            }
        }
        Ok(Err(err)) => (false, format!("git wait failed: {err}")),
        Err(_) => {
            (false, format!("git ls-remote timed out after {}s", GIT_PROBE_TIMEOUT.as_secs()))
        }
    }
}

fn read_local_git_remote(path: &std::path::Path) -> Option<String> {
    let cfg = path.join(".git").join("config");
    let raw = std::fs::read_to_string(&cfg).ok()?;
    parse_git_config_remote(&raw)
}

/// Tiny line-oriented parser for the `[remote "origin"]` block's `url =`
/// key. Sufficient for the inspection use case; falls back gracefully on
/// exotic `include = path` configs (those return `None`).
fn parse_git_config_remote(raw: &str) -> Option<String> {
    let mut in_origin = false;
    for line in raw.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with(';') || line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            in_origin = line == "[remote \"origin\"]";
            continue;
        }
        if in_origin {
            if let Some(rest) = line.strip_prefix("url") {
                if let Some(eq) = rest.find('=') {
                    return Some(rest[eq + 1..].trim().to_string());
                }
            }
        }
    }
    None
}

// ---- /projects/:project_id/scan --------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ScanQuery {
    #[serde(default)]
    pub repo: Option<String>,
}

#[derive(Debug, Serialize)]
struct ScanResponse {
    run_id: String,
}

async fn scan_project(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
    Query(q): Query<ScanQuery>,
) -> Result<Json<ScanResponse>, ApiError> {
    require_project(&s, &project_id).await?;
    // A `?repo=...` filter scopes the trigger to a single repo; the
    // dispatcher / config-resolver downstream is responsible for
    // rejecting unknown names so this handler stays a thin pass-through.
    let run_id = s.scan.trigger(ScanTriggerSource::Manual, Some(project_id), q.repo, None).await?;
    Ok(Json(ScanResponse { run_id }))
}

async fn start_pentest_project(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
    body: Option<Json<StartPentestRequest>>,
) -> Result<Json<StartPentestResponse>, ApiError> {
    let project = require_project(&s, &project_id).await?;
    let profile = project.default_launch_profile.ok_or_else(|| {
        ApiError::BadRequest(
            "configure a default launch profile before starting a pentest".to_string(),
        )
    })?;
    if profile.readiness != "Ready" {
        return Err(ApiError::BadRequest(format!(
            "default launch profile is not ready ({})",
            profile.readiness
        )));
    }
    let request = body.map(|Json(body)| body).unwrap_or_default();
    if request.allow_state_changing_live_probes && !request.exploit_mode_enabled {
        return Err(ApiError::BadRequest(
            "state-changing live probes require exploit mode to be enabled".to_string(),
        ));
    }
    for template_id in &request.business_logic_template_ids {
        if business_logic_template_by_id(template_id).is_none() {
            return Err(ApiError::BadRequest(format!(
                "unknown business-logic template id `{template_id}`"
            )));
        }
    }
    let run_id = s
        .scan
        .trigger(
            ScanTriggerSource::Manual,
            Some(project_id),
            None,
            Some(ScanRunOverrides {
                exploit_mode_enabled: request.exploit_mode_enabled,
                allow_state_changing_live_probes: request.allow_state_changing_live_probes,
                exploit_dry_run: request.exploit_dry_run,
                browser_checks_enabled: request.browser_checks_enabled,
                business_logic_templates_enabled: request.business_logic_templates_enabled,
                research_mode_enabled: request.research_mode_enabled,
                unsafe_attack_agent_enabled: request.unsafe_attack_agent_enabled,
                business_logic_template_ids: if request.business_logic_template_ids.is_empty() {
                    None
                } else {
                    Some(request.business_logic_template_ids)
                },
            }),
        )
        .await?;
    Ok(Json(StartPentestResponse { run_id }))
}

// ---- /projects/:project_id/integrations -----------------------------------

async fn list_project_integrations(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<ProjectIntegrationRecord>>, ApiError> {
    require_project(&s, &project_id).await?;
    Ok(Json(s.store.integrations().list_by_project(&project_id).await?))
}

async fn create_project_integration(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
    Json(req): Json<CreateProjectIntegrationRequest>,
) -> Result<Json<ProjectIntegrationRecord>, ApiError> {
    require_project(&s, &project_id).await?;
    let name = validate_integration_name(&req.name)?;
    validate_integration_events(&req.events)?;
    crate::integrations::validate_min_severity(req.min_severity.as_deref())
        .map_err(ApiError::BadRequest)?;
    let prepared =
        crate::integrations::prepare_config(&req.config).map_err(ApiError::BadRequest)?;
    let now = now_epoch_ms();
    let id = format!("int-{}", uuid_like(&format!("{project_id}-{name}"), now));
    let row = s
        .store
        .integrations()
        .create(ProjectIntegrationInsert {
            id,
            project_id,
            kind: prepared.kind,
            name,
            enabled: req.enabled,
            events: req.events,
            min_severity: req.min_severity,
            config_json: prepared.config_json,
            target: prepared.target,
            now_ms: now,
        })
        .await?;
    Ok(Json(row))
}

async fn get_project_integration(
    State(s): State<ServerState>,
    Path((project_id, integration_id)): Path<(String, String)>,
) -> Result<Json<ProjectIntegrationRecord>, ApiError> {
    require_project(&s, &project_id).await?;
    let row = require_project_integration(&s, &project_id, &integration_id).await?;
    Ok(Json(row))
}

async fn patch_project_integration(
    State(s): State<ServerState>,
    Path((project_id, integration_id)): Path<(String, String)>,
    Json(req): Json<PatchProjectIntegrationRequest>,
) -> Result<Json<ProjectIntegrationRecord>, ApiError> {
    require_project(&s, &project_id).await?;
    require_project_integration(&s, &project_id, &integration_id).await?;
    if let Some(events) = &req.events {
        validate_integration_events(events)?;
    }
    if let Some(min) = &req.min_severity {
        crate::integrations::validate_min_severity(min.as_deref()).map_err(ApiError::BadRequest)?;
    }
    let (config_json, target) = if let Some(config) = &req.config {
        let prepared = crate::integrations::prepare_config(config).map_err(ApiError::BadRequest)?;
        (Some(prepared.config_json), Some(prepared.target))
    } else {
        (None, None)
    };
    let name = req.name.as_deref().map(validate_integration_name).transpose()?;
    let row = s
        .store
        .integrations()
        .update(
            &integration_id,
            ProjectIntegrationPatch {
                name,
                enabled: req.enabled,
                events: req.events,
                min_severity: req.min_severity,
                config_json,
                target,
                updated_at: now_epoch_ms(),
            },
        )
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("integration `{integration_id}` not found")))?;
    if row.project_id != project_id {
        return Err(ApiError::NotFound(format!(
            "integration `{integration_id}` not found in project `{project_id}`"
        )));
    }
    Ok(Json(row))
}

async fn delete_project_integration(
    State(s): State<ServerState>,
    Path((project_id, integration_id)): Path<(String, String)>,
) -> Result<StatusBody, ApiError> {
    require_project(&s, &project_id).await?;
    require_project_integration(&s, &project_id, &integration_id).await?;
    let affected = s.store.integrations().delete(&integration_id).await?;
    Ok(StatusBody::ok(format!("deleted {affected} integration row(s)")))
}

async fn test_project_integration(
    State(s): State<ServerState>,
    Path((project_id, integration_id)): Path<(String, String)>,
) -> Result<Json<TestProjectIntegrationResponse>, ApiError> {
    require_project(&s, &project_id).await?;
    let row =
        s.store.integrations().get_stored(&integration_id).await?.ok_or_else(|| {
            ApiError::NotFound(format!("integration `{integration_id}` not found"))
        })?;
    if row.public.project_id != project_id {
        return Err(ApiError::NotFound(format!(
            "integration `{integration_id}` not found in project `{project_id}`"
        )));
    }
    match crate::integrations::IntegrationDispatcher::new().send_test(&s.store, &row).await {
        Ok(()) => {
            let _ = s
                .store
                .integrations()
                .record_delivery(&integration_id, now_epoch_ms(), "ok", None)
                .await;
            Ok(Json(TestProjectIntegrationResponse {
                ok: true,
                message: "test delivery sent".to_string(),
            }))
        }
        Err(err) => {
            let _ = s
                .store
                .integrations()
                .record_delivery(&integration_id, now_epoch_ms(), "error", Some(&err))
                .await;
            Err(ApiError::BadRequest(format!("test delivery failed: {err}")))
        }
    }
}

// ---- /runs ------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct RunsQuery {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
}

async fn list_runs(
    State(s): State<ServerState>,
    Query(q): Query<RunsQuery>,
) -> Result<Json<Vec<RunRecord>>, ApiError> {
    let status = q.status.as_deref().unwrap_or("Running");
    let rows = if let Some(project_id) = q.project_id.as_deref() {
        require_project(&s, project_id).await?;
        s.store.runs().list_by_status_for_project(status, project_id).await?
    } else {
        s.store.runs().list_by_status(status).await?
    };
    Ok(Json(rows))
}

async fn get_run(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<RunRecord>, ApiError> {
    s.store
        .runs()
        .get(&id)
        .await?
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("run `{id}` not found")))
}

async fn environment_runs_for_run(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<nyx_agent_types::product::EnvironmentRunRecord>>, ApiError> {
    require_run(&s, &id).await?;
    Ok(Json(s.store.environment_runs().list_by_run(&id).await?))
}

async fn run_business_logic(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<BusinessLogicRunSummary>, ApiError> {
    let run = require_run(&s, &id).await?;
    let rows = s.store.business_logic_template_runs().list_by_run(&id).await?;
    let candidates_generated = rows.iter().map(|row| row.generated_count).sum();
    let templates_skipped = rows.iter().filter(|row| row.skipped_count > 0).count() as u32;
    let dry_run = rows.iter().any(|row| row.dry_run);
    Ok(Json(BusinessLogicRunSummary {
        run_id: run.id,
        templates_considered: rows.len() as u32,
        candidates_generated,
        templates_skipped,
        dry_run,
        templates: rows,
    }))
}

async fn run_event_log(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    require_run(&s, &id).await?;
    let logs_dir = s
        .state_logs_dir
        .as_ref()
        .ok_or_else(|| ApiError::Internal("logs directory is not configured".to_string()))?;
    let path = run_event_log_path(logs_dir, &id);
    let file = match tokio::fs::File::open(&path).await {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(ApiError::NotFound(format!("event log for run `{id}` not found")));
        }
        Err(err) => {
            return Err(ApiError::Internal(format!(
                "open run event log `{}`: {err}",
                path.display()
            )));
        }
    };

    let stream = async_stream::stream! {
        let mut file = file;
        let mut buf = vec![0_u8; 16 * 1024];
        loop {
            match file.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => yield Ok::<Bytes, std::io::Error>(Bytes::copy_from_slice(&buf[..n])),
                Err(err) => {
                    yield Err(err);
                    break;
                }
            }
        }
    };
    let filename = format!("{}.events.jsonl", safe_run_log_segment(&id));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{filename}\""))
        .body(Body::from_stream(stream))
        .map_err(|err| ApiError::Internal(format!("build run event log response: {err}")))
}

async fn verification_attempts_for_run(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<nyx_agent_types::product::VerificationAttemptRecord>>, ApiError> {
    require_run(&s, &id).await?;
    Ok(Json(s.store.verification_attempts().list_by_run(&id).await?))
}

async fn authz_matrix_for_run(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<nyx_agent_types::product::AuthzMatrixEntryRecord>>, ApiError> {
    require_run(&s, &id).await?;
    Ok(Json(s.store.authz_matrix().list_by_run(&id).await?))
}

async fn exploration_memory_for_run(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<nyx_agent_types::product::ExplorationMemoryRecord>>, ApiError> {
    require_run(&s, &id).await?;
    Ok(Json(s.store.exploration_memory().list_by_run(&id).await?))
}

#[derive(Debug, Deserialize)]
struct SignalsQuery {
    #[serde(default)]
    meaningful_only: bool,
}

async fn signals_for_run(
    State(s): State<ServerState>,
    Path(id): Path<String>,
    Query(q): Query<SignalsQuery>,
) -> Result<Json<Vec<nyx_agent_types::product::NyxSignalRecord>>, ApiError> {
    require_run(&s, &id).await?;
    Ok(Json(s.store.nyx_signals().list_by_run(&id, q.meaningful_only).await?))
}

async fn candidates_for_run(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<nyx_agent_types::product::PentestCandidateRecord>>, ApiError> {
    require_run(&s, &id).await?;
    Ok(Json(s.store.pentest_candidates().list_by_run(&id).await?))
}

async fn route_model_for_run(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<nyx_agent_types::product::RouteModelRecord>, ApiError> {
    require_run(&s, &id).await?;
    s.store
        .route_models()
        .get_by_run(&id)
        .await?
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("route model for run `{id}` not found")))
}

async fn run_vulnerabilities(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<nyx_agent_types::product::VerifiedVulnerabilityRecord>>, ApiError> {
    require_run(&s, &id).await?;
    Ok(Json(s.store.verified_vulnerabilities().list_by_run_including_triaged(&id).await?))
}

async fn project_vulnerabilities(
    State(s): State<ServerState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<nyx_agent_types::product::VerifiedVulnerabilityRecord>>, ApiError> {
    require_project(&s, &project_id).await?;
    Ok(Json(
        s.store.verified_vulnerabilities().list_by_project_including_triaged(&project_id).await?,
    ))
}

async fn list_vulnerabilities(
    State(s): State<ServerState>,
) -> Result<Json<Vec<nyx_agent_types::product::VerifiedVulnerabilityRecord>>, ApiError> {
    Ok(Json(s.store.verified_vulnerabilities().list_all_including_triaged().await?))
}

async fn get_vulnerability(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<nyx_agent_types::product::VerifiedVulnerabilityRecord>, ApiError> {
    s.store
        .verified_vulnerabilities()
        .get(&id)
        .await?
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("vulnerability `{id}` not found")))
}

#[derive(Debug, Serialize)]
struct RemediationStartResponse {
    job: crate::state::RemediationJobRecord,
}

async fn start_vulnerability_fix(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<RemediationStartResponse>, ApiError> {
    let vulnerability = s
        .store
        .verified_vulnerabilities()
        .get(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("vulnerability `{id}` not found")))?;
    let agent = s.remediation_agent.clone().ok_or_else(|| {
        ApiError::BadRequest(
            "no remediation agent is configured; select Codex or Claude Code as the AI runtime"
                .to_string(),
        )
    })?;
    let repos = s.store.repos().list_by_project(&vulnerability.project_id).await?;
    let workspace_roots = remediation_workspace_roots(&repos, s.state_repos_dir.as_deref());
    if workspace_roots.is_empty() {
        return Err(ApiError::BadRequest(
            "no writable local repository workspace is available for this project".to_string(),
        ));
    }

    let job = s
        .remediation_jobs
        .create(&vulnerability.id, &vulnerability.project_id, now_epoch_ms())
        .await;
    let job_id = job.id.clone();
    let jobs = s.remediation_jobs.clone();
    tokio::spawn(async move {
        jobs.push_phase(&job_id, "preparing", "Preparing vulnerability context.").await;
        let request = RemediationAgentRequest { vulnerability, workspace_roots };
        jobs.push_phase(&job_id, "editing", "Fix agent is editing the local repository.").await;
        match agent.fix(request).await {
            Ok(output) => jobs.complete(&job_id, output).await,
            Err(err) => jobs.fail(&job_id, remediation_error_to_job_error(err)).await,
        }
    });

    Ok(Json(RemediationStartResponse { job }))
}

async fn get_vulnerability_fix_job(
    State(s): State<ServerState>,
    Path((id, job_id)): Path<(String, String)>,
) -> Result<Json<crate::state::RemediationJobRecord>, ApiError> {
    let job = s
        .remediation_jobs
        .get(&job_id)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("fix job `{job_id}` not found")))?;
    if job.vulnerability_id != id {
        return Err(ApiError::NotFound(format!(
            "fix job `{job_id}` not found for vulnerability `{id}`"
        )));
    }
    Ok(Json(job))
}

fn remediation_error_to_job_error(err: crate::state::RemediationAgentError) -> RemediationJobError {
    match err {
        crate::state::RemediationAgentError::Unavailable(detail) => {
            RemediationJobError { title: "Fix agent unavailable".to_string(), detail }
        }
        crate::state::RemediationAgentError::Failed(detail) => {
            RemediationJobError { title: "Fix agent failed".to_string(), detail }
        }
    }
}

fn remediation_workspace_roots(
    repos: &[RepoRecord],
    state_repos_dir: Option<&FsPath>,
) -> Vec<PathBuf> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for repo in repos {
        if matches!(repo.source_kind.as_str(), "local" | "local-path") {
            push_workspace_root(&mut out, &mut seen, PathBuf::from(&repo.source_url_or_path));
        }
        if let Some(root) = state_repos_dir {
            let legacy = root.join(&repo.name);
            push_workspace_root(&mut out, &mut seen, legacy.join("checkout"));
            push_workspace_root(&mut out, &mut seen, legacy);
            if let Some(state_root) = root.parent() {
                let project_scoped = state_root
                    .join("projects")
                    .join(&repo.project_id)
                    .join("repos")
                    .join(&repo.name);
                push_workspace_root(&mut out, &mut seen, project_scoped.join("checkout"));
                push_workspace_root(&mut out, &mut seen, project_scoped);
            }
        }
    }
    out
}

fn push_workspace_root(out: &mut Vec<PathBuf>, seen: &mut BTreeSet<PathBuf>, path: PathBuf) {
    if path.is_dir() && seen.insert(path.clone()) {
        out.push(path);
    }
}

async fn require_run(s: &ServerState, id: &str) -> Result<RunRecord, ApiError> {
    s.store.runs().get(id).await?.ok_or_else(|| ApiError::NotFound(format!("run `{id}` not found")))
}

#[derive(Debug, Deserialize)]
struct VulnerabilityStatusPatch {
    status: String,
}

#[derive(Debug, Deserialize)]
struct BulkVulnerabilityStatusPatch {
    ids: Vec<String>,
    status: String,
}

async fn update_vulnerability_status(
    State(s): State<ServerState>,
    Path(id): Path<String>,
    Json(req): Json<VulnerabilityStatusPatch>,
) -> Result<Json<nyx_agent_types::product::VerifiedVulnerabilityRecord>, ApiError> {
    let status = normalize_vulnerability_status(&req.status)?;
    let row = s
        .store
        .verified_vulnerabilities()
        .set_status(&id, status)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("vulnerability `{id}` not found")))?;
    Ok(Json(row))
}

async fn bulk_update_vulnerability_status(
    State(s): State<ServerState>,
    Json(req): Json<BulkVulnerabilityStatusPatch>,
) -> Result<Json<Vec<nyx_agent_types::product::VerifiedVulnerabilityRecord>>, ApiError> {
    if req.ids.is_empty() {
        return Err(ApiError::BadRequest(
            "ids must contain at least one vulnerability".to_string(),
        ));
    }
    let status = normalize_vulnerability_status(&req.status)?;
    let mut ids = Vec::new();
    let mut seen = HashSet::new();
    for raw in req.ids {
        let id = raw.trim();
        if id.is_empty() {
            continue;
        }
        if seen.insert(id.to_string()) {
            ids.push(id.to_string());
        }
    }
    if ids.is_empty() {
        return Err(ApiError::BadRequest(
            "ids must contain at least one vulnerability".to_string(),
        ));
    }
    for id in &ids {
        if s.store.verified_vulnerabilities().get(id).await?.is_none() {
            return Err(ApiError::NotFound(format!("vulnerability `{id}` not found")));
        }
    }
    let mut updated = Vec::with_capacity(ids.len());
    for id in ids {
        let Some(row) = s.store.verified_vulnerabilities().set_status(&id, status).await? else {
            return Err(ApiError::NotFound(format!("vulnerability `{id}` not found")));
        };
        updated.push(row);
    }
    Ok(Json(updated))
}

fn normalize_vulnerability_status(raw: &str) -> Result<&'static str, ApiError> {
    let mut normalized = raw.trim().to_ascii_lowercase();
    normalized.retain(|ch| !matches!(ch, ' ' | '-' | '_'));
    match normalized.as_str() {
        "open" => Ok("Open"),
        "inprogress" | "investigating" => Ok("InProgress"),
        "fixed" | "resolved" => Ok("Fixed"),
        "falsepositive" => Ok("FalsePositive"),
        "acceptedrisk" | "accepted" => Ok("AcceptedRisk"),
        _ => Err(ApiError::BadRequest(format!("unknown vulnerability status `{raw}`"))),
    }
}

// ---- /findings --------------------------------------------------------------

/// Composite filter for `GET /api/v1/findings`. Every field is
/// optional; combining them ANDs server-side. Quarantined rows are
/// hidden by default; the Quarantine view passes
/// `include_quarantine=true`.
#[derive(Debug, Deserialize)]
pub struct FindingsQuery {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub cap: Option<String>,
    #[serde(default)]
    pub origin: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub severity: Option<String>,
    #[serde(default)]
    pub triage_state: Option<String>,
    #[serde(default)]
    pub chain_id: Option<String>,
    #[serde(default)]
    pub include_quarantine: bool,
}

async fn list_findings(
    State(s): State<ServerState>,
    Query(q): Query<FindingsQuery>,
) -> Result<Json<Vec<FindingRecord>>, ApiError> {
    if let Some(project_id) = q.project_id.as_deref() {
        require_project(&s, project_id).await?;
    }
    let filter = FindingFilter {
        project_id: q.project_id.as_deref(),
        repo: q.repo.as_deref(),
        run_id: q.run_id.as_deref(),
        cap: q.cap.as_deref(),
        origin: q.origin.as_deref(),
        status: q.status.as_deref(),
        severity: q.severity.as_deref(),
        triage_state: q.triage_state.as_deref(),
        chain_id: q.chain_id.as_deref(),
        include_quarantine: q.include_quarantine,
        limit: None,
    };
    let rows = s.store.findings().list_filtered(&filter).await?;
    Ok(Json(rows))
}

async fn get_finding(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<FindingRecord>, ApiError> {
    s.store
        .findings()
        .get(&id)
        .await?
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("finding `{id}` not found")))
}

// `FindingDiffStatus`, `FindingWithDiff`, and `RunFindingsResponse`
// live in `nyx_agent_types::api`; re-imported at the top of this file.

/// Composite filter for `GET /api/v1/runs/:id/findings`. Mirrors the
/// `FindingsQuery` shape minus `run_id` (taken from the path) and
/// `include_quarantine` (the run-scoped view always excludes
/// quarantined rows; the dedicated `/quarantine` endpoint covers that
/// surface).
#[derive(Debug, Deserialize, Default)]
pub struct RunFindingsQuery {
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub cap: Option<String>,
    #[serde(default)]
    pub origin: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub severity: Option<String>,
    #[serde(default)]
    pub triage_state: Option<String>,
    #[serde(default)]
    pub chain_id: Option<String>,
}

async fn findings_for_run(
    State(s): State<ServerState>,
    Path(run_id): Path<String>,
    Query(q): Query<RunFindingsQuery>,
) -> Result<Json<RunFindingsResponse>, ApiError> {
    let run = s
        .store
        .runs()
        .get(&run_id)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("run `{run_id}` not found")))?;
    let started_at = run.started_at;
    let prior_run_id = s.store.runs().prior_run_id(&run_id, started_at).await?;

    let filter = FindingFilter {
        project_id: None,
        run_id: Some(&run_id),
        repo: q.repo.as_deref(),
        cap: q.cap.as_deref(),
        origin: q.origin.as_deref(),
        status: q.status.as_deref(),
        severity: q.severity.as_deref(),
        triage_state: q.triage_state.as_deref(),
        chain_id: q.chain_id.as_deref(),
        include_quarantine: false,
        limit: None,
    };
    let current_rows = s.store.findings().list_filtered(&filter).await?;

    let prior_membership: HashMap<String, String> = match prior_run_id.as_deref() {
        Some(prior_id) => {
            s.store.findings().list_run_membership(prior_id).await?.into_iter().collect()
        }
        None => HashMap::new(),
    };
    let prior_known = !prior_membership.is_empty();
    let current_ids: HashSet<String> = current_rows.iter().map(|r| r.id.clone()).collect();

    let mut items: Vec<FindingWithDiff> = current_rows
        .into_iter()
        .map(|record| {
            let diff_status =
                classify_current_row(&record, &prior_membership, prior_known, started_at);
            FindingWithDiff { record, diff_status }
        })
        .collect();

    // Findings observed in the prior run but absent from the current
    // run: surface as `Closed`. Their row body is the latest-known
    // shape (fetched by id from `findings`), filtered by the same
    // user-supplied facets so `?repo=X` does not bleed Closed rows
    // from other repos.
    if !prior_membership.is_empty() {
        let closed_ids: Vec<&String> = prior_membership
            .iter()
            .filter_map(|(fid, prior_status)| {
                if current_ids.contains(fid) {
                    None
                } else if prior_status.eq_ignore_ascii_case("Closed") {
                    // Already closed in the prior run; not a regression.
                    None
                } else {
                    Some(fid)
                }
            })
            .collect();
        for fid in closed_ids {
            let Some(record) = s.store.findings().get(fid).await? else {
                continue;
            };
            if !row_passes_filter(&record, &q) {
                continue;
            }
            items.push(FindingWithDiff { record, diff_status: FindingDiffStatus::Closed });
        }
    }

    Ok(Json(RunFindingsResponse { run_id, prior_run_id, items }))
}

fn classify_current_row(
    record: &FindingRecord,
    prior_membership: &HashMap<String, String>,
    prior_known: bool,
    run_started_at: i64,
) -> FindingDiffStatus {
    if let Some(prior_status) = prior_membership.get(&record.id) {
        if prior_status.eq_ignore_ascii_case(&record.status) {
            return FindingDiffStatus::Unchanged;
        }
        return FindingDiffStatus::Regressed;
    }
    // No prior membership row for this finding.
    if prior_known {
        // The prior run produced `run_findings` rows; the absence is
        // authoritative; this finding is new in the current run.
        return FindingDiffStatus::New;
    }
    // Pre-migration prior run (or no prior run at all). Fall back to
    // the legacy first-seen heuristic so freshly-observed rows still
    // surface as `New` and older rows wallpaper as `Unchanged`.
    if record.first_seen >= run_started_at {
        FindingDiffStatus::New
    } else {
        FindingDiffStatus::Unchanged
    }
}

/// Same predicate `FindingStore::list_filtered` runs at the DB level,
/// applied in-memory to a row fetched by id. Used for the `Closed`
/// path where the row does not live in the current run's filtered
/// projection. Mirrors every facet `RunFindingsQuery` accepts.
fn row_passes_filter(record: &FindingRecord, q: &RunFindingsQuery) -> bool {
    if record.status.eq_ignore_ascii_case("Quarantine") {
        return false;
    }
    if let Some(repo) = q.repo.as_deref() {
        if record.repo != repo {
            return false;
        }
    }
    if let Some(cap) = q.cap.as_deref() {
        if record.cap != cap {
            return false;
        }
    }
    if let Some(origin) = q.origin.as_deref() {
        if record.finding_origin != origin {
            return false;
        }
    }
    if let Some(status) = q.status.as_deref() {
        if record.status != status {
            return false;
        }
    }
    if let Some(severity) = q.severity.as_deref() {
        if record.severity != severity {
            return false;
        }
    }
    if let Some(triage) = q.triage_state.as_deref() {
        if record.triage_state != triage {
            return false;
        }
    }
    if let Some(chain_id) = q.chain_id.as_deref() {
        if record.chain_id.as_deref() != Some(chain_id) {
            return false;
        }
    }
    true
}

// ---- /chains ----------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ChainListQuery {
    run_id: Option<String>,
    #[serde(default)]
    include_proposed: bool,
}

async fn list_chains(
    State(s): State<ServerState>,
    Query(q): Query<ChainListQuery>,
) -> Result<Json<Vec<ChainRecord>>, ApiError> {
    let run_id = q
        .run_id
        .ok_or_else(|| ApiError::BadRequest("missing `run_id` query parameter".to_string()))?;
    let mut rows = s.store.chains().list_by_run(&run_id).await?;
    if !q.include_proposed {
        rows.retain(|row| row.status == "Verified");
    }
    Ok(Json(rows))
}

async fn get_chain(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<ChainRecord>, ApiError> {
    s.store
        .chains()
        .get(&id)
        .await?
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("chain `{id}` not found")))
}

// ---- /quarantine ------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
struct QuarantineQuery {
    #[serde(default)]
    project_id: Option<String>,
}

async fn list_quarantine(
    State(s): State<ServerState>,
    Query(q): Query<QuarantineQuery>,
) -> Result<Json<Vec<QuarantineItem>>, ApiError> {
    if let Some(project_id) = q.project_id.as_deref() {
        require_project(&s, project_id).await?;
    }
    let mut out: Vec<QuarantineItem> = Vec::new();
    let filter = FindingFilter {
        project_id: q.project_id.as_deref(),
        status: Some("Quarantine"),
        include_quarantine: true,
        ..Default::default()
    };
    let findings = s.store.findings().list_filtered(&filter).await?;
    for f in findings {
        out.push(QuarantineItem {
            kind: QuarantineKind::Finding,
            id: f.id,
            run_id: f.run_id,
            repo: f.repo,
            path: f.path,
            line: f.line,
            cap: f.cap,
            rule: Some(f.rule),
            severity: Some(f.severity),
            finding_origin: Some(f.finding_origin),
            prompt_version: f.prompt_version,
            attack_provenance: f.attack_provenance,
            rationale: None,
            verdict_blob: f.verdict_blob,
            last_seen: Some(f.last_seen),
        });
    }
    let pending = if let Some(project_id) = q.project_id.as_deref() {
        s.store.candidate_findings().list_pending_by_project(project_id).await?
    } else {
        s.store.candidate_findings().list_pending().await?
    };
    for c in pending {
        out.push(QuarantineItem {
            kind: QuarantineKind::Candidate,
            id: c.id,
            run_id: c.run_id,
            repo: c.repo,
            path: c.path,
            line: c.line,
            cap: c.cap,
            rule: c.rule_hint,
            severity: None,
            finding_origin: Some("AiExploration".to_string()),
            prompt_version: c.prompt_version,
            attack_provenance: None,
            rationale: c.rationale,
            verdict_blob: None,
            last_seen: None,
        });
    }
    // Most-recently-stamped findings first; candidates fall in after
    // (no `last_seen`).
    out.sort_by_key(|b| std::cmp::Reverse(b.last_seen.unwrap_or(0)));
    Ok(Json(out))
}

async fn promote_quarantine(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<QuarantineItem>, ApiError> {
    if id.starts_with("cand-") {
        let cand = s
            .store
            .candidate_findings()
            .get(&id)
            .await?
            .ok_or_else(|| ApiError::NotFound(format!("candidate `{id}` not found")))?;
        if cand.status != CandidateStatus::Pending.as_str() {
            return Err(ApiError::BadRequest(format!(
                "candidate `{id}` is not pending (status = `{}`)",
                cand.status
            )));
        }
        promote_candidate_to_finding(&s, &cand).await?;
        Ok(Json(candidate_to_quarantine_item(&cand)))
    } else {
        // Findings-table quarantine: flip status to 'Open' so the row
        // reappears in the Findings browser. The operator's manual
        // promote skips the dynamic-confirm gate by design (acceptance:
        // "Manually promoting it moves it to Findings.").
        let row = manual_promote_finding_row(&s, &id).await?;
        Ok(Json(finding_to_quarantine_item(&row)))
    }
}

async fn dismiss_quarantine(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<QuarantineItem>, ApiError> {
    if id.starts_with("cand-") {
        let cand = s
            .store
            .candidate_findings()
            .get(&id)
            .await?
            .ok_or_else(|| ApiError::NotFound(format!("candidate `{id}` not found")))?;
        if cand.status != CandidateStatus::Pending.as_str() {
            return Err(ApiError::BadRequest(format!(
                "candidate `{id}` is not pending (status = `{}`)",
                cand.status
            )));
        }
        s.store.candidate_findings().set_status(&id, CandidateStatus::Dismissed.as_str()).await?;
        Ok(Json(candidate_to_quarantine_item(&cand)))
    } else {
        let row = manual_dismiss_finding_row(&s, &id).await?;
        Ok(Json(finding_to_quarantine_item(&row)))
    }
}

async fn manual_promote_finding_row(s: &ServerState, id: &str) -> Result<FindingRecord, ApiError> {
    let existing = require_quarantined_finding(s, id).await?;
    let blob = serde_json::to_string(&json!({
        "kind": "ManualPromote",
        "from": "quarantine",
        "prev_provenance": existing.attack_provenance,
        "prev_verdict_blob": existing.verdict_blob,
    }))
    .map_err(|e| ApiError::Internal(format!("serialize manual-promote blob: {e}")))?;
    s.store.findings().manual_promote(id, "Open", &blob).await?;
    s.store
        .findings()
        .get(id)
        .await?
        .ok_or_else(|| ApiError::Internal("finding vanished after promote".to_string()))
}

async fn manual_dismiss_finding_row(s: &ServerState, id: &str) -> Result<FindingRecord, ApiError> {
    let existing = require_quarantined_finding(s, id).await?;
    let blob = serde_json::to_string(&json!({
        "kind": "ManualDismiss",
        "from": "quarantine",
        "prev_provenance": existing.attack_provenance,
        "prev_verdict_blob": existing.verdict_blob,
    }))
    .map_err(|e| ApiError::Internal(format!("serialize manual-dismiss blob: {e}")))?;
    s.store.findings().manual_dismiss(id, &blob).await?;
    s.store
        .findings()
        .get(id)
        .await?
        .ok_or_else(|| ApiError::Internal("finding vanished after dismiss".to_string()))
}

async fn require_quarantined_finding(s: &ServerState, id: &str) -> Result<FindingRecord, ApiError> {
    let existing = s
        .store
        .findings()
        .get(id)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("finding `{id}` not found")))?;
    if existing.status != "Quarantine" {
        return Err(ApiError::BadRequest(format!(
            "finding `{id}` is not in Quarantine (status = `{}`)",
            existing.status
        )));
    }
    Ok(existing)
}

async fn promote_candidate_to_finding(
    s: &ServerState,
    cand: &CandidateFindingRecord,
) -> Result<(), ApiError> {
    let line = cand.line.unwrap_or(-1);
    let rule = cand.rule_hint.clone().unwrap_or_else(|| format!("ai-exploration:{}", cand.cap));
    let id = nyx_agent_core::store::finding_id_hash(
        &cand.repo,
        &cand.path,
        Some(line),
        &cand.cap,
        &rule,
    );
    let now = now_epoch_ms();
    let verdict_blob = serde_json::to_string(&json!({
        "kind": "ManualPromote",
        "from": "candidate",
        "candidate_id": cand.id,
        "rationale": cand.rationale,
    }))
    .map_err(|e| ApiError::Internal(format!("serialize verdict blob: {e}")))?;
    let rec = FindingRecord {
        id,
        run_id: cand.run_id.clone(),
        repo: cand.repo.clone(),
        path: cand.path.clone(),
        line: cand.line,
        cap: cand.cap.clone(),
        rule,
        severity: "High".to_string(),
        // Manual promote skips the dynamic-verifier gate; mark Open
        // (not Verified) so the operator's intent is preserved
        // without claiming the row has been confirmed by the
        // sandbox-replayed differential.
        status: "Open".to_string(),
        finding_origin: "AiExploration".to_string(),
        first_seen: now,
        last_seen: now,
        superseded_by: None,
        triage_state: "Open".to_string(),
        triage_assigned_to: None,
        verdict_blob: Some(verdict_blob),
        repro_path: None,
        attack_provenance: Some("ManualPromote".to_string()),
        prompt_version: cand.prompt_version.clone(),
        chain_id: None,
        spec_id: None,
    };
    s.store.findings().upsert(&rec).await?;
    s.store.candidate_findings().set_status(&cand.id, CandidateStatus::Promoted.as_str()).await?;
    Ok(())
}

fn finding_to_quarantine_item(f: &FindingRecord) -> QuarantineItem {
    QuarantineItem {
        kind: QuarantineKind::Finding,
        id: f.id.clone(),
        run_id: f.run_id.clone(),
        repo: f.repo.clone(),
        path: f.path.clone(),
        line: f.line,
        cap: f.cap.clone(),
        rule: Some(f.rule.clone()),
        severity: Some(f.severity.clone()),
        finding_origin: Some(f.finding_origin.clone()),
        prompt_version: f.prompt_version.clone(),
        attack_provenance: f.attack_provenance.clone(),
        rationale: None,
        verdict_blob: f.verdict_blob.clone(),
        last_seen: Some(f.last_seen),
    }
}

fn candidate_to_quarantine_item(c: &CandidateFindingRecord) -> QuarantineItem {
    QuarantineItem {
        kind: QuarantineKind::Candidate,
        id: c.id.clone(),
        run_id: c.run_id.clone(),
        repo: c.repo.clone(),
        path: c.path.clone(),
        line: c.line,
        cap: c.cap.clone(),
        rule: c.rule_hint.clone(),
        severity: None,
        finding_origin: Some("AiExploration".to_string()),
        prompt_version: c.prompt_version.clone(),
        attack_provenance: None,
        rationale: c.rationale.clone(),
        verdict_blob: None,
        last_seen: None,
    }
}

// ---- /traces ----------------------------------------------------------------
//
// `AgentTraceRow` lives in `nyx_agent_types::api`; the `From<AgentTraceRecord>`
// projection it carries drops the persistence-only `verifier_blob` field
// so the FE shape stays minimal. Lift `verifier_blob` onto the wire here
// when the trace viewer (Phase 24) starts rendering Verifier-row
// inputs/outputs without joining `findings.verdict_blob`.

async fn traces_for_finding(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<AgentTraceRow>>, ApiError> {
    // Candidate ids carry a `cand-` prefix (see
    // `nyx_agent::ai_pipeline::candidate_id`); route those through the
    // `candidate_findings.trace_id` back-link so the trace viewer can
    // render the proposing AI call for a Pending candidate. Finding ids
    // hit the direct `agent_traces.finding_id` index as before.
    let rows = if id.starts_with("cand-") {
        s.store.agent_traces().list_for_candidate(&id).await?
    } else {
        s.store.agent_traces().list_for_finding(&id).await?
    };
    Ok(Json(rows.into_iter().map(AgentTraceRow::from).collect()))
}

async fn get_trace(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<AgentTraceRow>, ApiError> {
    s.store
        .agent_traces()
        .get(&id)
        .await?
        .map(AgentTraceRow::from)
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("trace `{id}` not found")))
}

// ---- /events ----------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct EventsQuery {
    #[serde(default)]
    pub run_id: Option<String>,
}

async fn events_ws(
    State(s): State<ServerState>,
    Query(q): Query<EventsQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    // Subscribe *before* reading the replay so events that fire between
    // snapshot and join still hit this receiver. The snapshot is sent
    // first; duplicate frames are idempotent client-side because
    // applyEvent in repoStatus.ts treats per-repo state as a fold over
    // the latest event per key.
    let rx = s.events.subscribe();
    let filter = q.run_id.clone();
    let replay = if let Some(run_id) = filter.as_deref() {
        s.replay.snapshot(run_id).await
    } else {
        Vec::new()
    };
    ws.on_upgrade(move |socket| handle_events_ws(socket, rx, filter, replay))
}

async fn handle_events_ws(
    socket: WebSocket,
    mut rx: tokio::sync::broadcast::Receiver<AgentEvent>,
    run_filter: Option<String>,
    replay: Vec<AgentEvent>,
) {
    let (mut tx, mut rx_socket) = socket.split();
    for ev in replay {
        match serde_json::to_string(&ev) {
            Ok(payload) => {
                if tx.send(Message::Text(payload.into())).await.is_err() {
                    return;
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, "failed to serialize replay AgentEvent");
            }
        }
    }
    loop {
        tokio::select! {
            biased;
            client_msg = rx_socket.next() => {
                match client_msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(payload))) => {
                        if tx.send(Message::Pong(payload)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(_)) => {
                        // Ignore client-initiated frames; this stream is
                        // server-push only.
                    }
                    Some(Err(_)) => break,
                }
            }
            event = rx.recv() => {
                match event {
                    Ok(ev) => {
                        if !run_matches(&ev, run_filter.as_deref()) {
                            continue;
                        }
                        match serde_json::to_string(&ev) {
                            Ok(payload) => {
                                if tx.send(Message::Text(payload.into())).await.is_err() {
                                    break;
                                }
                            }
                            Err(err) => {
                                tracing::warn!(error = %err, "failed to serialize AgentEvent");
                            }
                        }
                    }
                    Err(RecvError::Lagged(skipped)) => {
                        let warning = json!({
                            "kind": "Lagged",
                            "skipped": skipped,
                        });
                        if tx.send(Message::Text(warning.to_string().into())).await.is_err() {
                            break;
                        }
                    }
                    Err(RecvError::Closed) => break,
                }
            }
        }
    }
}

fn run_matches(ev: &AgentEvent, run_filter: Option<&str>) -> bool {
    let Some(want) = run_filter else { return true };
    match ev {
        AgentEvent::Run { data } => {
            let id = match data {
                RunEvent::Heartbeat { .. } => return true,
                RunEvent::RunStarted { run_id, .. }
                | RunEvent::ProjectStarted { run_id, .. }
                | RunEvent::PhaseStarted { run_id, .. }
                | RunEvent::PhaseFinished { run_id, .. }
                | RunEvent::EnvironmentStatus { run_id, .. }
                | RunEvent::AuthSessionStatus { run_id, .. }
                | RunEvent::LiveVerificationCapabilities { run_id, .. }
                | RunEvent::RepoStarted { run_id, .. }
                | RunEvent::RepoStaticDone { run_id, .. }
                | RunEvent::RepoDynamicDone { run_id, .. }
                | RunEvent::RepoFailed { run_id, .. }
                | RunEvent::RepoIngestFailed { run_id, .. }
                | RunEvent::RepoFinished { run_id, .. }
                | RunEvent::ProjectFinished { run_id, .. }
                | RunEvent::RunFinished { run_id, .. } => run_id.as_str(),
            };
            id == want
        }
        AgentEvent::Ai { data: AiEvent::BudgetTick { run_id, .. } } => run_id == want,
        AgentEvent::Sandbox { data } => {
            let run_id = match data {
                SandboxEvent::VerifierStarted { run_id, .. }
                | SandboxEvent::VerifierFinished { run_id, .. } => run_id.as_str(),
            };
            run_id == want
        }
        _ => true,
    }
}

// ---- /runs/:id/summary ------------------------------------------------------

async fn run_summary(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<RunCard>, ApiError> {
    let card = build_run_card(s.store.pool(), &id).await.map_err(run_card_to_api)?;
    Ok(Json(card))
}

async fn run_summary_markdown(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let card = build_run_card(s.store.pool(), &id).await.map_err(run_card_to_api)?;
    let body = render_run_card_markdown(&card);
    Ok((StatusCode::OK, [(header::CONTENT_TYPE, "text/markdown; charset=utf-8")], body)
        .into_response())
}

async fn run_summary_html(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let card = build_run_card(s.store.pool(), &id).await.map_err(run_card_to_api)?;
    let body = render_run_card_html(&card);
    Ok((StatusCode::OK, [(header::CONTENT_TYPE, "text/html; charset=utf-8")], body).into_response())
}

fn run_card_to_api(err: RunCardError) -> ApiError {
    match err {
        RunCardError::NotFound(id) => ApiError::NotFound(format!("run `{id}` not found")),
        RunCardError::Store(e) => ApiError::Store(e),
        RunCardError::Sqlx(e) => ApiError::Internal(format!("sqlx: {e}")),
    }
}

// ---- /findings/:id/repro-bundle ---------------------------------------------

async fn create_repro_bundle(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<BundleManifest>, ApiError> {
    let out_dir = s
        .state_bundles_dir
        .as_ref()
        .cloned()
        .ok_or_else(|| ApiError::Internal("bundle output dir is not configured".to_string()))?;
    let manifest =
        build_bundle(&s.store, &id, &out_dir, now_epoch_ms()).await.map_err(bundle_to_api)?;
    Ok(Json(manifest))
}

async fn download_repro_bundle(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let bundles = s.store.repro_bundles().list_for_finding(&id).await?;
    // Most-recently-built bundle wins. If none, build one inline so the
    // operator can hit the download URL directly without first calling
    // POST /repro-bundle from a script.
    let row = if let Some(latest) = bundles.last().cloned() {
        latest
    } else {
        let out_dir =
            s.state_bundles_dir.as_ref().cloned().ok_or_else(|| {
                ApiError::Internal("bundle output dir is not configured".to_string())
            })?;
        let manifest =
            build_bundle(&s.store, &id, &out_dir, now_epoch_ms()).await.map_err(bundle_to_api)?;
        s.store
            .repro_bundles()
            .list_for_finding(&id)
            .await?
            .into_iter()
            .find(|r| r.path == manifest.bundle_path.display().to_string())
            .ok_or_else(|| ApiError::Internal("bundle row vanished after build".to_string()))?
    };

    let safe_path = ensure_bundle_path_inside_root(&row.path, s.state_bundles_dir.as_deref())?;
    let bytes = std::fs::read(&safe_path)
        .map_err(|e| ApiError::Internal(format!("read {}: {e}", safe_path.display())))?;
    let filename = format!("{id}.tar");
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/x-tar".to_string()),
            (header::CONTENT_DISPOSITION, format!("attachment; filename=\"{filename}\"")),
            ("X-Nyx-Agent-Bundle-Sha256".parse().unwrap(), row.sha256),
        ],
        Body::from(bytes),
    )
        .into_response())
}

/// Defense-in-depth: refuse to read a `repro_bundles.path` value that
/// canonicalises outside the configured bundles root. `build_bundle` is the
/// only writer today, but a future migration / import / handler that takes a
/// path from JSON could otherwise turn the download endpoint into an
/// authenticated arbitrary-file-read.
fn ensure_bundle_path_inside_root(
    path: &str,
    bundles_dir: Option<&std::path::Path>,
) -> Result<std::path::PathBuf, ApiError> {
    let root = bundles_dir
        .ok_or_else(|| ApiError::Internal("bundle output dir is not configured".to_string()))?;
    let canonical_root = root
        .canonicalize()
        .map_err(|e| ApiError::Internal(format!("canonicalize bundles root: {e}")))?;
    let canonical_path = std::path::Path::new(path)
        .canonicalize()
        .map_err(|e| ApiError::Internal(format!("canonicalize bundle path `{path}`: {e}")))?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(ApiError::Internal("bundle path escapes configured root".to_string()));
    }
    Ok(canonical_path)
}

fn bundle_to_api(err: BundleError) -> ApiError {
    match err {
        BundleError::FindingNotFound(id) => ApiError::NotFound(format!("finding `{id}` not found")),
        BundleError::Tar(e) => ApiError::Internal(format!("bundle tar write: {e}")),
        BundleError::Store(e) => ApiError::Store(e),
        BundleError::Io { path, source } => {
            ApiError::Internal(format!("bundle io at {}: {source}", path.display()))
        }
    }
}

// ---- /findings/:id/replay ---------------------------------------------------

/// Hard wall-clock ceiling on a single replay invocation. A runaway
/// `repro.sh` cannot keep a daemon worker pinned indefinitely.
const REPLAY_WALL_CLOCK_TIMEOUT_SECS: u64 = 120;
/// Grace window after SIGKILL for the kernel to reap the child.
const REPLAY_REAP_GRACE_SECS: u64 = 5;

async fn replay_repro_bundle(
    State(s): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<SseEvent, std::convert::Infallible>>>, ApiError> {
    // Resolve (or build) the most recent bundle on disk.
    let bundles = s.store.repro_bundles().list_for_finding(&id).await?;
    let bundle_path: std::path::PathBuf = match bundles.last() {
        Some(row) => ensure_bundle_path_inside_root(&row.path, s.state_bundles_dir.as_deref())?,
        None => {
            let out_dir = s.state_bundles_dir.as_ref().cloned().ok_or_else(|| {
                ApiError::Internal("bundle output dir is not configured".to_string())
            })?;
            let manifest = build_bundle(&s.store, &id, &out_dir, now_epoch_ms())
                .await
                .map_err(bundle_to_api)?;
            ensure_bundle_path_inside_root(
                &manifest.bundle_path.display().to_string(),
                s.state_bundles_dir.as_deref(),
            )?
        }
    };

    let extract_root = match tempfile::tempdir() {
        Ok(t) => t,
        Err(e) => return Err(ApiError::Internal(format!("tempdir: {e}"))),
    };
    let extract_path = extract_root.path().to_path_buf();
    let tar_bytes = std::fs::read(&bundle_path)
        .map_err(|e| ApiError::Internal(format!("read {}: {e}", bundle_path.display())))?;
    // Guard against on-disk substitution between build_bundle (which
    // stamps repro_bundles.sha256) and this exec. If the row exists
    // and the digest disagrees, refuse to extract.
    if let Some(expected) = bundles.last().map(|r| r.sha256.as_str()) {
        if !verify_bundle_sha256(&tar_bytes, expected) {
            return Err(ApiError::Internal(format!(
                "bundle integrity check failed for {}: stored sha256 does not match on-disk bytes",
                bundle_path.display()
            )));
        }
    }
    extract_ustar(&tar_bytes, &extract_path)
        .map_err(|e| ApiError::Internal(format!("extract bundle: {e}")))?;
    let repro_sh = extract_path.join(&id).join("repro.sh");
    if !repro_sh.exists() {
        return Err(ApiError::Internal(format!(
            "bundle did not contain repro.sh at {}",
            repro_sh.display()
        )));
    }

    let started_at = now_epoch_ms();
    let store = s.store.clone();
    let bundle_id_for_status = bundles.last().map(|r| r.id.clone());
    let finding_id = id.clone();
    let events = s.events.clone();
    let bundle_path_str = bundle_path.display().to_string();
    let stream = async_stream::stream! {
        let _ = events.send(AgentEvent::Repro {
            data: ReproEvent::ReplayStarted {
                finding_id: finding_id.clone(),
                bundle_path: bundle_path_str.clone(),
                started_at_ms: started_at,
            },
        });
        yield Ok(SseEvent::default()
            .event("start")
            .data(serde_json::json!({
                "finding_id": finding_id,
                "bundle_path": bundle_path_str,
                "started_at_ms": started_at,
            }).to_string()));

        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg(&repro_sh);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        cmd.stdin(std::process::Stdio::null());
        cmd.kill_on_drop(true);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let msg = format!("spawn bash: {e}");
                let _ = events.send(AgentEvent::Repro {
                    data: ReproEvent::ReplayError {
                        finding_id: finding_id.clone(),
                        message: msg.clone(),
                    },
                });
                yield Ok(SseEvent::default().event("error").data(msg));
                yield Ok(SseEvent::default().event("end").data("error"));
                return;
            }
        };
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let mut stdout_lines = tokio::io::AsyncBufReadExt::lines(
            tokio::io::BufReader::new(stdout),
        );
        let mut stderr_lines = tokio::io::AsyncBufReadExt::lines(
            tokio::io::BufReader::new(stderr),
        );
        // Deadline keeps a runaway repro.sh (infinite loop, `sleep
        // infinity`, etc.) from pinning a daemon worker. On expiry we
        // SIGKILL the child; kill_on_drop also fires if the SSE client
        // disconnects.
        let deadline = tokio::time::Instant::now()
            + std::time::Duration::from_secs(REPLAY_WALL_CLOCK_TIMEOUT_SECS);
        let mut stdout_done = false;
        let mut stderr_done = false;
        let mut timed_out = false;
        while (!stdout_done || !stderr_done) && !timed_out {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {
                    let _ = child.start_kill();
                    let msg = format!(
                        "replay exceeded {REPLAY_WALL_CLOCK_TIMEOUT_SECS}s wall-clock timeout; killed"
                    );
                    let _ = events.send(AgentEvent::Repro {
                        data: ReproEvent::ReplayError {
                            finding_id: finding_id.clone(),
                            message: msg.clone(),
                        },
                    });
                    yield Ok(SseEvent::default().event("error").data(msg));
                    timed_out = true;
                }
                line = stdout_lines.next_line(), if !stdout_done => {
                    match line {
                        Ok(Some(text)) => {
                            let _ = events.send(AgentEvent::Repro {
                                data: ReproEvent::ReplayStdout {
                                    finding_id: finding_id.clone(),
                                    line: text.clone(),
                                },
                            });
                            yield Ok(SseEvent::default().event("stdout").data(text));
                        }
                        Ok(None) => stdout_done = true,
                        Err(e) => {
                            let msg = format!("stdout read: {e}");
                            let _ = events.send(AgentEvent::Repro {
                                data: ReproEvent::ReplayError {
                                    finding_id: finding_id.clone(),
                                    message: msg.clone(),
                                },
                            });
                            yield Ok(SseEvent::default().event("error").data(msg));
                            stdout_done = true;
                        }
                    }
                }
                line = stderr_lines.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(text)) => {
                            let _ = events.send(AgentEvent::Repro {
                                data: ReproEvent::ReplayStderr {
                                    finding_id: finding_id.clone(),
                                    line: text.clone(),
                                },
                            });
                            yield Ok(SseEvent::default().event("stderr").data(text));
                        }
                        Ok(None) => stderr_done = true,
                        Err(e) => {
                            let msg = format!("stderr read: {e}");
                            let _ = events.send(AgentEvent::Repro {
                                data: ReproEvent::ReplayError {
                                    finding_id: finding_id.clone(),
                                    message: msg.clone(),
                                },
                            });
                            yield Ok(SseEvent::default().event("error").data(msg));
                            stderr_done = true;
                        }
                    }
                }
            }
        }
        // Bound the wait so a child that ignores SIGKILL (or a kernel
        // that is slow to reap) cannot pin the task forever either.
        let status = match tokio::time::timeout(
            std::time::Duration::from_secs(REPLAY_REAP_GRACE_SECS),
            child.wait(),
        )
        .await
        {
            Ok(Ok(status)) => status,
            Ok(Err(e)) => {
                let msg = format!("wait: {e}");
                let _ = events.send(AgentEvent::Repro {
                    data: ReproEvent::ReplayError {
                        finding_id: finding_id.clone(),
                        message: msg.clone(),
                    },
                });
                yield Ok(SseEvent::default().event("error").data(msg));
                yield Ok(SseEvent::default().event("end").data("error"));
                return;
            }
            Err(_) => {
                let msg = format!(
                    "child not reaped within {REPLAY_REAP_GRACE_SECS}s after kill"
                );
                let _ = events.send(AgentEvent::Repro {
                    data: ReproEvent::ReplayError {
                        finding_id: finding_id.clone(),
                        message: msg.clone(),
                    },
                });
                yield Ok(SseEvent::default().event("error").data(msg));
                yield Ok(SseEvent::default().event("end").data("error"));
                return;
            }
        };
        let exit_code = status.code().unwrap_or(-1);
        let finished_at = now_epoch_ms();
        let verdict = if exit_code == 0 { "Pass" } else { "Fail" };
        if let Some(bid) = bundle_id_for_status.as_deref() {
            if let Err(e) = store
                .repro_bundles()
                .record_replay(bid, finished_at, verdict)
                .await
            {
                tracing::warn!(error = %e, "failed to record replay status");
            }
        }
        // Keep the extracted tempdir alive until after the child exits.
        drop(extract_root);
        let _ = events.send(AgentEvent::Repro {
            data: ReproEvent::ReplayFinished {
                finding_id: finding_id.clone(),
                status: verdict.to_string(),
                exit_code,
                started_at_ms: started_at,
                finished_at_ms: finished_at,
                duration_ms: finished_at - started_at,
            },
        });
        yield Ok(SseEvent::default()
            .event("end")
            .data(serde_json::json!({
                "exit_code": exit_code,
                "status": verdict,
                "started_at_ms": started_at,
                "finished_at_ms": finished_at,
                "duration_ms": finished_at - started_at,
            }).to_string()));
    };
    Ok(Sse::new(stream).keep_alive(Default::default()))
}

/// Extract the USTAR/PAX tarball produced by
/// `nyx_agent_core::report::repro_bundle::build_ustar`. Rejects entries
/// whose path escapes the destination via `..` or absolute components
/// so a substituted bundle cannot write outside the tempdir.
fn extract_ustar(bytes: &[u8], dest: &std::path::Path) -> std::io::Result<()> {
    let mut archive = tar::Archive::new(std::io::Cursor::new(bytes));
    // Honor on-record perms but refuse path traversal. The `tar` crate
    // calls this "Overwrite" + "PreservePermissions" + "Unpack"; the
    // default unpack already rejects `..`, but we sanitise explicitly
    // so a malformed PAX `path` extension cannot smuggle one in.
    archive.set_preserve_permissions(true);
    archive.set_overwrite(true);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let safe = sanitise_tar_path(&path).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "unsafe tar path")
        })?;
        let target = dest.join(safe);
        entry.unpack(&target)?;
    }
    Ok(())
}

/// Reject tar entries containing `..` components or absolute paths so
/// extraction stays inside the destination tempdir.
fn sanitise_tar_path(name: &std::path::Path) -> Option<std::path::PathBuf> {
    if name.is_absolute() {
        return None;
    }
    let mut out = std::path::PathBuf::new();
    for component in name.components() {
        match component {
            std::path::Component::Normal(s) => out.push(s),
            std::path::Component::CurDir => {}
            _ => return None,
        }
    }
    Some(out)
}

// ---- helpers ----------------------------------------------------------------

#[derive(Debug, Serialize)]
struct StatusBody {
    ok: bool,
    message: String,
}

impl StatusBody {
    fn ok(message: impl Into<String>) -> Self {
        Self { ok: true, message: message.into() }
    }
}

impl IntoResponse for StatusBody {
    fn into_response(self) -> Response {
        Json(self).into_response()
    }
}
