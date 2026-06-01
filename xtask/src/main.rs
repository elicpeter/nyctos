// xtask binary. Subcommands:
//   - `gen-ts`           regenerates the frontend TypeScript bindings
//                        driven by `#[derive(TS)]` on every type in
//                        `nyx-agent-types`. CI invokes this binary and
//                        then runs `git diff --exit-code
//                        frontend/src/api/types.gen.ts` to catch drift
//                        between the Rust schema and the committed
//                        bindings.
//   - `lint-instrument`  warn-only lint for public functions that
//                        forgot `#[tracing::instrument]`. Replaces the
//                        prior `.ci/missing-instrument.sh` awk script;
//                        the shell wrapper now shells out here.

mod lint_instrument;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use nyx_agent_types::{
    agent::{
        AgentResult, AgentTask, Budget, BudgetKind, CacheStats, CostEstimate, ExtractedAgentResult,
        HaltReason, Prompt, Response, TokenUsage,
    },
    api::{
        AgentTraceRow, BundleManifest, DoctorCheck, DoctorRequest, DoctorResponse,
        FindingDiffStatus, FindingWithDiff, HealthResponse, QuarantineItem, QuarantineKind,
        ReplayEvent, ReplayEventKind, RunFindingsResponse, SetupRequest, SetupStatusResponse,
    },
    attack_graph::{AttackGraphEdgeRecord, AttackGraphEvidenceTrail, AttackGraphNodeRecord},
    chain::ChainRecord,
    event::{
        AgentEvent, AiEvent, BudgetEvent, FindingEvent, QuarantineEvent, RepoOutcomeTag,
        ReproEvent, RunEvent, SandboxEvent,
    },
    finding::FindingRecord,
    integration::{
        CreateProjectIntegrationRequest, PatchProjectIntegrationRequest,
        ProjectIntegrationConfigInput, ProjectIntegrationEvent, ProjectIntegrationKind,
        ProjectIntegrationRecord, SmtpSecurity, TestProjectIntegrationResponse,
    },
    product::{
        ApiClientCallModel, AuthzMatrixEntryRecord, EnvironmentRunRecord, ExplorationMemoryRecord,
        FormModel, FrontendRouteModel, LaunchEnvRef, LaunchHealthCheck, LaunchStep,
        LaunchWorkingDir, NyxSignalRecord, PentestCandidateRecord, ProjectLaunchProfile,
        ProjectLaunchProfileInput, ProjectSetupError, ProjectSetupJobEvent,
        ProjectSetupJobListResponse, ProjectSetupJobRecord, ProjectSetupJobStatus,
        ProjectSetupPhase, ProjectSetupRequest, ProjectSetupResponse, ProjectSetupStartResponse,
        ProjectSetupVerification, ProjectSetupVerificationStatus, RouteEvidence, RouteModel,
        RouteModelEndpoint, RouteModelRecord, SeedSetupPlan, SeedSetupResponse,
        StartPentestRequest, StartPentestResponse, TestLaunchTargetRequest,
        TestLaunchTargetResponse, VerificationAttemptRecord, VerifiedVulnerabilityRecord,
    },
    project::{
        AuthSetupError, AuthSetupJobEvent, AuthSetupJobRecord, AuthSetupJobStatus, AuthSetupPhase,
        AuthSetupRequest, AuthSetupResponse, AuthSetupStartResponse, AuthSetupVerification,
        AuthSetupVerificationStatus, CreateProjectRequest, PatchProjectRequest,
        ProjectAuthAssertion, ProjectAuthAssertionKind, ProjectAuthHeaderRef, ProjectAuthMode,
        ProjectAuthOwnedObject, ProjectAuthProfile, ProjectOtpSourceConfig, ProjectOtpSourceKind,
        ProjectRecord, ProjectRuntimeCommand, ProjectRuntimeEnvVar, ProjectRuntimeProfile,
    },
    repo::{
        CreateRepoRequest, GitAuth, PatchRepoRequest, Repo, RepoRecord, RepoSource,
        TestRepoRequest, TestRepoResponse,
    },
    run::RunRecord,
    target::{LocalBinaryTarget, PentestTarget},
    trace::AgentTraceRecord,
};
use ts_rs::{Config, TS};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let cmd = args.next();
    match cmd.as_deref() {
        Some("gen-ts") => match gen_ts() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("xtask gen-ts failed: {err}");
                ExitCode::from(1)
            }
        },
        Some("lint-instrument") => match lint_instrument::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("xtask lint-instrument failed: {err}");
                ExitCode::from(1)
            }
        },
        Some(other) => {
            eprintln!("xtask: unknown subcommand `{other}` (try `gen-ts` or `lint-instrument`)");
            ExitCode::from(2)
        }
        None => {
            eprintln!("xtask: missing subcommand (try `gen-ts` or `lint-instrument`)");
            ExitCode::from(2)
        }
    }
}

fn gen_ts() -> Result<(), String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .ok_or_else(|| "xtask Cargo.toml is not in a workspace".to_string())?;
    let out_path = workspace_root.join("frontend").join("src").join("api").join("types.gen.ts");

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }

    let body = render();
    fs::write(&out_path, body).map_err(|e| format!("write {}: {e}", out_path.display()))?;
    Ok(())
}

fn render() -> String {
    let mut out = String::new();
    out.push_str("// AUTO-GENERATED FILE. DO NOT EDIT.\n");
    out.push_str("// Regenerated by `cargo run -p xtask -- gen-ts`.\n");
    out.push_str("// Source of truth: #[derive(TS)] types in crates/nyx-agent-types/src/.\n");
    out.push('\n');

    // Top-level types in dependency-friendly order. ts-rs `decl()`
    // produces a self-contained declaration per type; cross-type
    // references resolve at the TS level by name, so the order only
    // affects the file's readability, not correctness.
    let decls: Vec<String> = vec![
        decl_of::<RepoOutcomeTag>(),
        decl_of::<RunEvent>(),
        decl_of::<HaltReason>(),
        decl_of::<AiEvent>(),
        decl_of::<BudgetKind>(),
        decl_of::<TokenUsage>(),
        decl_of::<CacheStats>(),
        decl_of::<Budget>(),
        decl_of::<CostEstimate>(),
        decl_of::<Prompt>(),
        decl_of::<Response>(),
        decl_of::<AgentTask>(),
        decl_of::<ExtractedAgentResult>(),
        decl_of::<AgentResult>(),
        decl_of::<SandboxEvent>(),
        decl_of::<FindingEvent>(),
        decl_of::<BudgetEvent>(),
        decl_of::<QuarantineEvent>(),
        decl_of::<ReproEvent>(),
        decl_of::<AgentEvent>(),
        decl_of::<GitAuth>(),
        decl_of::<RepoSource>(),
        decl_of::<Repo>(),
        decl_of::<RepoRecord>(),
        decl_of::<CreateRepoRequest>(),
        decl_of::<PatchRepoRequest>(),
        decl_of::<TestRepoRequest>(),
        decl_of::<TestRepoResponse>(),
        decl_of::<RunRecord>(),
        decl_of::<ChainRecord>(),
        decl_of::<FindingRecord>(),
        decl_of::<ProjectIntegrationKind>(),
        decl_of::<ProjectIntegrationEvent>(),
        decl_of::<SmtpSecurity>(),
        decl_of::<ProjectIntegrationRecord>(),
        decl_of::<ProjectIntegrationConfigInput>(),
        decl_of::<CreateProjectIntegrationRequest>(),
        decl_of::<PatchProjectIntegrationRequest>(),
        decl_of::<TestProjectIntegrationResponse>(),
        decl_of::<AttackGraphNodeRecord>(),
        decl_of::<AttackGraphEdgeRecord>(),
        decl_of::<AttackGraphEvidenceTrail>(),
        decl_of::<ProjectRuntimeCommand>(),
        decl_of::<ProjectRuntimeEnvVar>(),
        decl_of::<ProjectAuthHeaderRef>(),
        decl_of::<ProjectAuthMode>(),
        decl_of::<ProjectOtpSourceKind>(),
        decl_of::<ProjectOtpSourceConfig>(),
        decl_of::<ProjectAuthAssertionKind>(),
        decl_of::<ProjectAuthAssertion>(),
        decl_of::<ProjectAuthOwnedObject>(),
        decl_of::<ProjectAuthProfile>(),
        decl_of::<ProjectRuntimeProfile>(),
        decl_of::<AuthSetupRequest>(),
        decl_of::<AuthSetupVerificationStatus>(),
        decl_of::<AuthSetupVerification>(),
        decl_of::<AuthSetupResponse>(),
        decl_of::<AuthSetupJobStatus>(),
        decl_of::<AuthSetupPhase>(),
        decl_of::<AuthSetupJobEvent>(),
        decl_of::<AuthSetupError>(),
        decl_of::<AuthSetupJobRecord>(),
        decl_of::<AuthSetupStartResponse>(),
        decl_of::<LaunchStep>(),
        decl_of::<LaunchHealthCheck>(),
        decl_of::<LaunchEnvRef>(),
        decl_of::<LaunchWorkingDir>(),
        decl_of::<ProjectLaunchProfile>(),
        decl_of::<ProjectLaunchProfileInput>(),
        decl_of::<ProjectSetupRequest>(),
        decl_of::<ProjectSetupVerificationStatus>(),
        decl_of::<ProjectSetupVerification>(),
        decl_of::<SeedSetupPlan>(),
        decl_of::<SeedSetupResponse>(),
        decl_of::<ProjectSetupResponse>(),
        decl_of::<ProjectSetupJobStatus>(),
        decl_of::<ProjectSetupPhase>(),
        decl_of::<ProjectSetupJobEvent>(),
        decl_of::<ProjectSetupError>(),
        decl_of::<ProjectSetupJobRecord>(),
        decl_of::<ProjectSetupStartResponse>(),
        decl_of::<ProjectSetupJobListResponse>(),
        decl_of::<EnvironmentRunRecord>(),
        decl_of::<NyxSignalRecord>(),
        decl_of::<PentestCandidateRecord>(),
        decl_of::<VerificationAttemptRecord>(),
        decl_of::<AuthzMatrixEntryRecord>(),
        decl_of::<VerifiedVulnerabilityRecord>(),
        decl_of::<ExplorationMemoryRecord>(),
        decl_of::<RouteEvidence>(),
        decl_of::<RouteModelEndpoint>(),
        decl_of::<FrontendRouteModel>(),
        decl_of::<ApiClientCallModel>(),
        decl_of::<FormModel>(),
        decl_of::<RouteModel>(),
        decl_of::<RouteModelRecord>(),
        decl_of::<StartPentestRequest>(),
        decl_of::<StartPentestResponse>(),
        decl_of::<TestLaunchTargetRequest>(),
        decl_of::<TestLaunchTargetResponse>(),
        decl_of::<ProjectRecord>(),
        decl_of::<CreateProjectRequest>(),
        decl_of::<PatchProjectRequest>(),
        decl_of::<HealthResponse>(),
        decl_of::<SetupStatusResponse>(),
        decl_of::<SetupRequest>(),
        decl_of::<BundleManifest>(),
        decl_of::<DoctorCheck>(),
        decl_of::<DoctorResponse>(),
        decl_of::<DoctorRequest>(),
        decl_of::<FindingDiffStatus>(),
        decl_of::<FindingWithDiff>(),
        decl_of::<RunFindingsResponse>(),
        decl_of::<QuarantineKind>(),
        decl_of::<QuarantineItem>(),
        decl_of::<AgentTraceRecord>(),
        decl_of::<AgentTraceRow>(),
        decl_of::<ReplayEventKind>(),
        decl_of::<ReplayEvent>(),
        decl_of::<LocalBinaryTarget>(),
        decl_of::<PentestTarget>(),
    ];

    let decl_count = decls.len();
    for (idx, decl) in decls.into_iter().enumerate() {
        out.push_str("export ");
        out.push_str(&trim_trailing_line_whitespace(&decl));
        if !decl.ends_with('\n') {
            out.push('\n');
        }
        if idx + 1 < decl_count {
            out.push('\n');
        }
    }
    out
}

fn decl_of<T: TS>() -> String {
    <T as TS>::decl(&Config::default())
}

fn trim_trailing_line_whitespace(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for segment in input.split_inclusive('\n') {
        let (line, newline) =
            segment.strip_suffix('\n').map(|line| (line, "\n")).unwrap_or((segment, ""));
        out.push_str(line.trim_end_matches([' ', '\t']));
        out.push_str(newline);
    }
    out
}
