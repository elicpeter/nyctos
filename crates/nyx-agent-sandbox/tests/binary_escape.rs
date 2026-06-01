//! Binary-lane sandbox-escape regression suite.
//!
//! `docs/binary-target-pentest.md` §8.7 requires the binary-target exec
//! path — [`BinaryRunner`] — to inherit the same containment guarantees
//! the raw [`crate::Sandbox`] backends provide. The binary pass feeds
//! attacker-influenced input to native code and *expects* crashes, so
//! the runner that wraps the sandbox must not widen the blast radius.
//!
//! These cases drive the `escape-attempt` probe (a stand-in hostile
//! "target") through `BinaryRunner` under the [`BirdcageSandbox`]
//! backend and assert each escape attempt is contained as observed via
//! the redacted [`BinaryExecResult`] the agent would receive — write
//! outside the workspace, read a secret outside it, open a socket, and a
//! forked child retrying the write. The raw-`SandboxOpts` versions of
//! these live in `escape.rs`; this suite is the guard that the new exec
//! wrapper does not regress them.

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::path::{Path, PathBuf};
use std::time::Duration;

use nyx_agent_sandbox::{
    BackendKind, BinaryExecRequest, BinaryExecResult, BinaryRunner, BirdcageSandbox, Sandbox,
    SandboxOpts, SandboxStatus,
};
use tempfile::tempdir;
use tokio::sync::OnceCell;

const SHIM: &str = env!("CARGO_BIN_EXE_nyx-sandbox-shim");
const PROBE: &str = env!("CARGO_BIN_EXE_escape-attempt");
static BIRDCAGE_RUNTIME: OnceCell<Result<(), String>> = OnceCell::const_new();

fn runner() -> BinaryRunner {
    BinaryRunner {
        backend: BackendKind::Birdcage,
        shim_path: Some(PathBuf::from(SHIM)),
        ..BinaryRunner::default()
    }
}

/// Build an exec request that launches the probe with `args`, granting a
/// read+execute exception for the probe binary (it lives in the cargo
/// target dir, outside the workspace).
fn probe_request(args: Vec<String>) -> BinaryExecRequest {
    let mut argv = vec![PROBE.to_string()];
    argv.extend(args);
    let mut req = BinaryExecRequest::new(argv);
    req.timeout = Duration::from_secs(5);
    req.allow_read.push(PathBuf::from(PROBE));
    req
}

async fn exec(workspace: &Path, req: BinaryExecRequest) -> BinaryExecResult {
    runner().exec(workspace, req).await.expect("binary runner exec")
}

async fn probe_birdcage_runtime() -> Result<(), String> {
    let scratch = tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let workspace = scratch.path().join("ws");
    std::fs::create_dir(&workspace).map_err(|e| format!("create workspace: {e}"))?;
    // The raw sandbox is the cheapest readiness probe; mirror escape.rs.
    let argv = vec![PROBE.to_string(), "noop".to_string()];
    let mut opts = SandboxOpts::new(workspace.clone(), argv);
    opts.timeout = Duration::from_secs(5);
    opts.allow_read.push(PathBuf::from(PROBE));
    let mut sb = BirdcageSandbox::with_shim_path(PathBuf::from(SHIM));
    sb.run(opts).await.map_err(|e| format!("sandbox run failed: {e}"))?;
    let outcome = sb.wait().await.map_err(|e| format!("sandbox wait failed: {e}"))?;
    if matches!(outcome.status, SandboxStatus::Exited(0)) {
        Ok(())
    } else {
        Err(format!("noop returned {:?}", outcome.status))
    }
}

async fn require_birdcage_runtime() -> bool {
    let probe = BIRDCAGE_RUNTIME.get_or_init(probe_birdcage_runtime).await;
    match probe {
        Ok(()) => true,
        Err(reason) => {
            if std::env::var("NYX_AGENT_REQUIRE_BIRDCAGE").ok().as_deref() == Some("1") {
                panic!("NYX_AGENT_REQUIRE_BIRDCAGE=1 but birdcage runtime unavailable: {reason}");
            }
            eprintln!("SKIP: birdcage runtime unavailable; binary escape suite bypassed: {reason}");
            false
        }
    }
}

#[tokio::test]
async fn binary_runner_contains_write_outside_workspace() {
    if !require_birdcage_runtime().await {
        return;
    }
    let scratch = tempdir().unwrap();
    let workspace = scratch.path().join("ws");
    std::fs::create_dir(&workspace).unwrap();
    let secret_dir = tempdir().unwrap();
    let target = secret_dir.path().join("escaped.txt");

    let res =
        exec(&workspace, probe_request(vec!["write-outside".into(), target.display().to_string()]))
            .await;
    assert!(res.contained(), "write-outside escaped through BinaryRunner: {res:?}");
    assert!(!target.exists(), "escape file created at {}", target.display());
}

#[tokio::test]
async fn binary_runner_contains_read_secret_outside_workspace() {
    if !require_birdcage_runtime().await {
        return;
    }
    let scratch = tempdir().unwrap();
    let workspace = scratch.path().join("ws");
    std::fs::create_dir(&workspace).unwrap();
    let secret_dir = tempdir().unwrap();
    let secret = secret_dir.path().join("secret.txt");
    std::fs::write(&secret, b"top-secret-do-not-leak").unwrap();

    let res =
        exec(&workspace, probe_request(vec!["read-outside".into(), secret.display().to_string()]))
            .await;
    assert!(res.contained(), "read-outside escaped through BinaryRunner: {res:?}");
    assert!(
        !res.stdout_preview.contains("top-"),
        "secret leaked to stdout preview: {}",
        res.stdout_preview
    );
}

#[tokio::test]
async fn binary_runner_contains_tcp_egress() {
    if !require_birdcage_runtime().await {
        return;
    }
    let scratch = tempdir().unwrap();
    let workspace = scratch.path().join("ws");
    std::fs::create_dir(&workspace).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let res =
        exec(&workspace, probe_request(vec!["connect-tcp".into(), addr.to_string()])).await;
    assert!(res.contained(), "tcp connect escaped through BinaryRunner: {res:?}");
    let accepted = tokio::time::timeout(Duration::from_millis(100), listener.accept()).await;
    assert!(accepted.is_err(), "loopback connect was accepted: {accepted:?}");
}

#[tokio::test]
async fn binary_runner_contains_forked_child_escape() {
    if !require_birdcage_runtime().await {
        return;
    }
    let scratch = tempdir().unwrap();
    let workspace = scratch.path().join("ws");
    std::fs::create_dir(&workspace).unwrap();
    let secret_dir = tempdir().unwrap();
    let target = secret_dir.path().join("forked-escape.txt");

    let res = exec(
        &workspace,
        probe_request(vec!["fork-write-outside".into(), target.display().to_string()]),
    )
    .await;
    assert!(res.contained(), "forked write escaped through BinaryRunner: {res:?}");
    assert!(!target.exists(), "forked-escape file created at {}", target.display());
}
