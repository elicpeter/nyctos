//! Sandboxed binary-target runner.
//!
//! The sibling of [`crate::payload_runner::PayloadRunner`] for the
//! binary / CLI target kind. Where the payload runner drives a *known*
//! payload through a synthesised harness and renders a differential
//! verdict, this runner drives an *agent-crafted* input set against an
//! operator-pinned native binary and returns the raw crash evidence.
//!
//! One [`BinaryRunner::exec`] call:
//!
//! 1. stages the agent's input files into the run workspace,
//! 2. builds [`SandboxOpts`] (timeout, allow_read, allow_loopback,
//!    capture_files, max_output_bytes, `lane = Chain`, cwd = workspace),
//! 3. runs the target once under the configured backend, and
//! 4. maps the [`SandboxOutcome`] into a redacted, size-capped
//!    [`BinaryExecResult`] with a derived [`CrashSignal`].
//!
//! The full stdout/stderr/artifact bytes stay host-side as evidence; the
//! agent only ever sees the capped previews and the artifact name list
//! (see §8.6 of `docs/binary-target-pentest.md`).
//!
//! Backend policy is enforced by the caller (the binary-target pass
//! refuses to run on [`BackendKind::Process`]); this runner faithfully
//! drives whichever backend it is handed so the unit tests can exercise
//! the unhardened `Process` backend without a shim.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use thiserror::Error;

use crate::{
    BackendKind, BirdcageSandbox, Lane, ProcessSandbox, Sandbox, SandboxError, SandboxOpts,
    SandboxOutcome, SandboxStatus,
};

/// Default cap on stdout/stderr previews fed back to the model, in
/// bytes each. Keeps prompt cost bounded and avoids round-tripping huge
/// crash dumps through the LLM.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 16 * 1024;

/// One sandboxed execution of the target binary, requested by the agent.
#[derive(Debug, Clone)]
pub struct BinaryExecRequest {
    /// Fully-resolved argv (template slots already substituted
    /// host-side). `argv[0]` is the pinned program path.
    pub argv: Vec<String>,
    /// Files to stage into the workspace before exec, keyed by the
    /// workspace-relative path. Bytes already decoded from the agent's
    /// (base64) tool input.
    pub staged_files: Vec<(PathBuf, Vec<u8>)>,
    /// Optional stdin fed to the process.
    pub stdin: Option<Vec<u8>>,
    /// Workspace-relative paths to capture after the run (artifacts the
    /// target may have written). Maps to [`SandboxOpts::capture_files`].
    pub capture_files: Vec<PathBuf>,
    pub timeout: Duration,
    pub allow_loopback: bool,
    pub allow_read: Vec<PathBuf>,
}

impl BinaryExecRequest {
    /// New request with the given argv and sane defaults (10s timeout,
    /// no staged files, no loopback).
    pub fn new(argv: Vec<String>) -> Self {
        Self {
            argv,
            staged_files: Vec::new(),
            stdin: None,
            capture_files: Vec::new(),
            timeout: Duration::from_secs(10),
            allow_loopback: false,
            allow_read: Vec::new(),
        }
    }
}

/// What the agent gets back. A redacted, size-capped view of
/// [`SandboxOutcome`].
#[derive(Debug, Clone, Serialize)]
pub struct BinaryExecResult {
    /// [`BackendKind::as_str`] of the backend that ran the target.
    pub backend: String,
    pub status: BinaryExecStatus,
    /// Capped, lossy-utf8 stdout preview.
    pub stdout_preview: String,
    /// Capped, lossy-utf8 stderr preview.
    pub stderr_preview: String,
    pub duration_ms: u64,
    /// Heuristic crash classification derived from status + stderr.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crash_signal: Option<CrashSignal>,
    /// Captured artifact paths that existed after the run (names only;
    /// full bytes stay host-side as evidence, not fed back to the model).
    pub artifacts_present: Vec<String>,
    /// Sandbox exception refusals collected during setup.
    pub refusals: Vec<String>,
}

impl BinaryExecResult {
    /// Did the sandbox contain the run (anything but a clean exit(0))?
    /// Mirrors [`SandboxStatus::contained`].
    pub fn contained(&self) -> bool {
        !matches!(self.status, BinaryExecStatus::Exited { code: 0 })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BinaryExecStatus {
    Exited { code: i32 },
    Signaled { signal: i32 },
    TimedOut,
    Killed,
}

impl From<SandboxStatus> for BinaryExecStatus {
    fn from(status: SandboxStatus) -> Self {
        match status {
            SandboxStatus::Exited(code) => BinaryExecStatus::Exited { code },
            SandboxStatus::Signaled(signal) => BinaryExecStatus::Signaled { signal },
            SandboxStatus::TimedOut => BinaryExecStatus::TimedOut,
            SandboxStatus::Killed => BinaryExecStatus::Killed,
        }
    }
}

/// Coarse crash taxonomy. Derived, not authoritative — the host attaches
/// the raw [`SandboxOutcome`] as evidence regardless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CrashSignal {
    /// SIGSEGV / SIGBUS.
    Segfault,
    /// SIGABRT (assert / `__stack_chk_fail` / glibc abort).
    Abort,
    /// stderr matches ASAN/UBSAN/MSAN/LSAN markers.
    Sanitizer,
    /// Hang — torn down by the timeout.
    Timeout,
    /// Exited non-zero with no clearer signal.
    NonZeroExit,
}

/// Derive the coarse [`CrashSignal`] from the final status and stderr.
///
/// A sanitizer report takes precedence over the signal mapping: ASAN
/// aborts the process (SIGABRT), but the sanitizer report is the real
/// signal. A clean `exit(0)` yields `None` (only interesting in a
/// differential vs an oracle, handled by the verdict layer not here).
pub fn derive_crash_signal(status: SandboxStatus, stderr: &[u8]) -> Option<CrashSignal> {
    if stderr_has_sanitizer_marker(stderr) {
        return Some(CrashSignal::Sanitizer);
    }
    match status {
        // SIGSEGV (11), SIGBUS (10 on macOS / 7 on Linux), SIGBUS (7).
        SandboxStatus::Signaled(11 | 7 | 10) => Some(CrashSignal::Segfault),
        SandboxStatus::Signaled(6) => Some(CrashSignal::Abort),
        // Any other fatal signal is still a crash; classify by abort as
        // the closest coarse bucket rather than dropping it.
        SandboxStatus::Signaled(_) => Some(CrashSignal::Abort),
        SandboxStatus::TimedOut => Some(CrashSignal::Timeout),
        SandboxStatus::Killed => Some(CrashSignal::Timeout),
        SandboxStatus::Exited(0) => None,
        SandboxStatus::Exited(_) => Some(CrashSignal::NonZeroExit),
    }
}

/// Sanitizer marker scan over stderr. Substring match against the
/// well-known runtime banners.
fn stderr_has_sanitizer_marker(stderr: &[u8]) -> bool {
    const MARKERS: &[&[u8]] = &[
        b"AddressSanitizer",
        b"runtime error:", // UBSAN
        b"ERROR: LeakSanitizer",
        b"MemorySanitizer",
        b"ThreadSanitizer",
        b"SUMMARY: UndefinedBehaviorSanitizer",
    ];
    MARKERS.iter().any(|m| bytes_contains(stderr, m))
}

fn bytes_contains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[derive(Debug, Error)]
pub enum BinaryRunnerError {
    #[error("empty argv: a binary exec needs at least the program path")]
    EmptyArgv,
    #[error("workspace setup failed: {0}")]
    Workspace(#[source] std::io::Error),
    #[error("staged file path escapes the workspace: {0}")]
    StagedPathEscapesWorkspace(String),
    #[error("sandbox error: {0}")]
    Sandbox(#[from] SandboxError),
}

/// Configuration shared across every [`BinaryRunner::exec`] call.
pub struct BinaryRunner {
    pub backend: BackendKind,
    /// Override path to `nyx-sandbox-shim` for [`BackendKind::Birdcage`].
    /// `None` defers to [`BirdcageSandbox::new`]'s default resolution.
    pub shim_path: Option<PathBuf>,
    pub max_output_bytes: usize,
}

impl Default for BinaryRunner {
    fn default() -> Self {
        Self {
            backend: BackendKind::Process,
            shim_path: None,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        }
    }
}

impl BinaryRunner {
    /// Stage files, build [`SandboxOpts`], run one contained exec, and
    /// classify the outcome into a [`BinaryExecResult`].
    pub async fn exec(
        &self,
        workspace: &Path,
        req: BinaryExecRequest,
    ) -> Result<BinaryExecResult, BinaryRunnerError> {
        if req.argv.is_empty() {
            return Err(BinaryRunnerError::EmptyArgv);
        }

        // 1. Stage the agent's input files into the workspace. Reject any
        //    path that escapes the workspace (absolute or `..` traversal)
        //    before touching the filesystem.
        for (rel, bytes) in &req.staged_files {
            let abs = self.safe_join(workspace, rel)?;
            if let Some(parent) = abs.parent() {
                std::fs::create_dir_all(parent).map_err(BinaryRunnerError::Workspace)?;
            }
            std::fs::write(&abs, bytes).map_err(BinaryRunnerError::Workspace)?;
        }

        // 2. Build SandboxOpts on the Chain lane (binary fuzzing is the
        //    microVM-grade isolation lane).
        let mut opts = SandboxOpts::new(workspace.to_path_buf(), req.argv);
        opts.cwd = Some(workspace.to_path_buf());
        opts.timeout = req.timeout;
        opts.lane = Some(Lane::Chain);
        opts.allow_loopback = req.allow_loopback;
        opts.allow_read = req.allow_read;
        opts.max_output_bytes = self.max_output_bytes;
        opts.capture_files = req.capture_files.clone();

        // stdin staging: the Sandbox trait has no stdin channel, so write
        // it to a workspace file the target can read and surface its path
        // via env. Targets that consume stdin proper are out of v1 scope;
        // file/arg fuzzing is the milestone (see doc §12).
        if let Some(stdin) = &req.stdin {
            let stdin_path = workspace.join("nyx_stdin.bin");
            std::fs::write(&stdin_path, stdin).map_err(BinaryRunnerError::Workspace)?;
            opts.env.push(("NYX_STDIN_PATH".to_string(), "nyx_stdin.bin".to_string()));
        }

        // 3. Run + wait.
        let outcome = self.run_sandbox(opts).await?;

        // 4. Map to the redacted result.
        Ok(self.classify(outcome, &req.capture_files))
    }

    fn classify(
        &self,
        outcome: SandboxOutcome,
        declared_captures: &[PathBuf],
    ) -> BinaryExecResult {
        let crash_signal = derive_crash_signal(outcome.status, &outcome.stderr);
        let artifacts_present = declared_captures
            .iter()
            .filter(|rel| {
                matches!(outcome.captured_files.get(*rel), Some(Some(_)))
            })
            .map(|rel| rel.to_string_lossy().to_string())
            .collect();
        BinaryExecResult {
            backend: outcome.backend.as_str().to_string(),
            status: outcome.status.into(),
            stdout_preview: preview(&outcome.stdout, self.max_output_bytes),
            stderr_preview: preview(&outcome.stderr, self.max_output_bytes),
            duration_ms: outcome.duration.as_millis() as u64,
            crash_signal,
            artifacts_present,
            refusals: outcome.refusals,
        }
    }

    /// Join `rel` under `workspace`, refusing absolute paths and any
    /// `..` component that would escape the workspace.
    fn safe_join(&self, workspace: &Path, rel: &Path) -> Result<PathBuf, BinaryRunnerError> {
        use std::path::Component;
        if rel.is_absolute() {
            return Err(BinaryRunnerError::StagedPathEscapesWorkspace(
                rel.to_string_lossy().to_string(),
            ));
        }
        let mut depth: i32 = 0;
        for comp in rel.components() {
            match comp {
                Component::Normal(_) => depth += 1,
                Component::CurDir => {}
                Component::ParentDir => {
                    depth -= 1;
                    if depth < 0 {
                        return Err(BinaryRunnerError::StagedPathEscapesWorkspace(
                            rel.to_string_lossy().to_string(),
                        ));
                    }
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(BinaryRunnerError::StagedPathEscapesWorkspace(
                        rel.to_string_lossy().to_string(),
                    ));
                }
            }
        }
        Ok(workspace.join(rel))
    }

    async fn run_sandbox(&self, opts: SandboxOpts) -> Result<SandboxOutcome, SandboxError> {
        match self.backend {
            BackendKind::Process => {
                let mut sb = ProcessSandbox::new();
                sb.run(opts).await?;
                sb.wait().await
            }
            BackendKind::Birdcage => {
                let mut sb = match &self.shim_path {
                    Some(p) => BirdcageSandbox::with_shim_path(p.clone()),
                    None => BirdcageSandbox::new()?,
                };
                sb.run(opts).await?;
                sb.wait().await
            }
            // The microVM chain-lane backends are the production targets
            // for this runner but are not wired into the unit-test path
            // yet; the binary-target pass selects the strongest available
            // backend and hands it here. Surfacing BackendUnavailable
            // keeps the error path uniform until the chain-lane wiring
            // lands.
            BackendKind::Libkrun => Err(SandboxError::BackendUnavailable {
                backend: "libkrun",
                reason: "binary runner libkrun wiring not yet implemented".into(),
            }),
            BackendKind::Firecracker => Err(SandboxError::BackendUnavailable {
                backend: "firecracker",
                reason: "binary runner firecracker wiring not yet implemented".into(),
            }),
            BackendKind::Docker => Err(SandboxError::BackendUnavailable {
                backend: "docker",
                reason: "binary runner docker wiring not yet implemented".into(),
            }),
        }
    }
}

/// Render a size-capped, lossy-utf8 preview of `bytes`. Truncates on a
/// byte boundary and appends a `…[truncated N bytes]` marker so the
/// model knows output was dropped.
fn preview(bytes: &[u8], max: usize) -> String {
    if bytes.len() <= max {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    let head = String::from_utf8_lossy(&bytes[..max]).into_owned();
    let dropped = bytes.len() - max;
    format!("{head}…[truncated {dropped} bytes]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn crash_signal_segfault_from_sigsegv() {
        assert_eq!(
            derive_crash_signal(SandboxStatus::Signaled(11), b""),
            Some(CrashSignal::Segfault)
        );
    }

    #[test]
    fn crash_signal_abort_from_sigabrt() {
        assert_eq!(
            derive_crash_signal(SandboxStatus::Signaled(6), b""),
            Some(CrashSignal::Abort)
        );
    }

    #[test]
    fn crash_signal_sanitizer_takes_precedence_over_abort() {
        // ASAN aborts (SIGABRT) but the sanitizer report is the real
        // signal.
        let stderr = b"==1234==ERROR: AddressSanitizer: heap-buffer-overflow";
        assert_eq!(
            derive_crash_signal(SandboxStatus::Signaled(6), stderr),
            Some(CrashSignal::Sanitizer)
        );
    }

    #[test]
    fn crash_signal_ubsan_runtime_error() {
        let stderr = b"foo.c:10:5: runtime error: signed integer overflow";
        assert_eq!(
            derive_crash_signal(SandboxStatus::Exited(0), stderr),
            Some(CrashSignal::Sanitizer)
        );
    }

    #[test]
    fn crash_signal_timeout_and_nonzero_and_clean() {
        assert_eq!(derive_crash_signal(SandboxStatus::TimedOut, b""), Some(CrashSignal::Timeout));
        assert_eq!(
            derive_crash_signal(SandboxStatus::Exited(2), b""),
            Some(CrashSignal::NonZeroExit)
        );
        assert_eq!(derive_crash_signal(SandboxStatus::Exited(0), b""), None);
    }

    #[test]
    fn preview_truncates_with_marker() {
        let out = preview(b"abcdefghij", 4);
        assert_eq!(out, "abcd…[truncated 6 bytes]");
        assert_eq!(preview(b"abc", 8), "abc");
    }

    #[test]
    fn safe_join_rejects_escapes() {
        let runner = BinaryRunner::default();
        let ws = Path::new("/tmp/ws");
        assert!(runner.safe_join(ws, Path::new("../etc/passwd")).is_err());
        assert!(runner.safe_join(ws, Path::new("/etc/passwd")).is_err());
        assert!(runner.safe_join(ws, Path::new("a/../b")).is_ok());
        assert!(runner.safe_join(ws, Path::new("sub/input.bin")).is_ok());
    }

    // ---- integration against ProcessSandbox (no shim required) ----

    #[tokio::test]
    async fn clean_exit_yields_no_crash_signal() {
        let dir = tempdir().unwrap();
        let runner = BinaryRunner::default();
        let res = runner
            .exec(dir.path(), BinaryExecRequest::new(vec!["/bin/sh".into(), "-c".into(), "exit 0".into()]))
            .await
            .expect("exec");
        assert_eq!(res.status, BinaryExecStatus::Exited { code: 0 });
        assert!(res.crash_signal.is_none());
        assert!(!res.contained());
    }

    #[tokio::test]
    async fn nonzero_exit_is_classified() {
        let dir = tempdir().unwrap();
        let runner = BinaryRunner::default();
        let res = runner
            .exec(
                dir.path(),
                BinaryExecRequest::new(vec!["/bin/sh".into(), "-c".into(), "exit 3".into()]),
            )
            .await
            .expect("exec");
        assert_eq!(res.status, BinaryExecStatus::Exited { code: 3 });
        assert_eq!(res.crash_signal, Some(CrashSignal::NonZeroExit));
        assert!(res.contained());
    }

    #[tokio::test]
    async fn segfault_is_classified_as_crash() {
        // `kill -SEGV $$` makes the shell signal itself; the sandbox
        // reports Signaled(11) on Unix.
        let dir = tempdir().unwrap();
        let runner = BinaryRunner::default();
        let res = runner
            .exec(
                dir.path(),
                BinaryExecRequest::new(vec![
                    "/bin/sh".into(),
                    "-c".into(),
                    "kill -SEGV $$".into(),
                ]),
            )
            .await
            .expect("exec");
        assert_eq!(res.status, BinaryExecStatus::Signaled { signal: 11 });
        assert_eq!(res.crash_signal, Some(CrashSignal::Segfault));
    }

    #[tokio::test]
    async fn staged_file_is_read_by_target() {
        let dir = tempdir().unwrap();
        let runner = BinaryRunner::default();
        let mut req = BinaryExecRequest::new(vec![
            "/bin/sh".into(),
            "-c".into(),
            "cat input.txt".into(),
        ]);
        req.staged_files.push((PathBuf::from("input.txt"), b"hello-staged".to_vec()));
        let res = runner.exec(dir.path(), req).await.expect("exec");
        assert_eq!(res.status, BinaryExecStatus::Exited { code: 0 });
        assert!(res.stdout_preview.contains("hello-staged"));
    }

    #[tokio::test]
    async fn captured_artifact_is_listed_when_present() {
        let dir = tempdir().unwrap();
        let runner = BinaryRunner::default();
        let mut req = BinaryExecRequest::new(vec![
            "/bin/sh".into(),
            "-c".into(),
            "printf out > artifact.bin".into(),
        ]);
        req.capture_files.push(PathBuf::from("artifact.bin"));
        let res = runner.exec(dir.path(), req).await.expect("exec");
        assert_eq!(res.artifacts_present, vec!["artifact.bin".to_string()]);
    }

    #[tokio::test]
    async fn timeout_is_classified() {
        let dir = tempdir().unwrap();
        let runner = BinaryRunner::default();
        let mut req =
            BinaryExecRequest::new(vec!["/bin/sh".into(), "-c".into(), "sleep 5".into()]);
        req.timeout = Duration::from_millis(300);
        let res = runner.exec(dir.path(), req).await.expect("exec");
        assert_eq!(res.status, BinaryExecStatus::TimedOut);
        assert_eq!(res.crash_signal, Some(CrashSignal::Timeout));
    }

    #[tokio::test]
    async fn empty_argv_is_rejected() {
        let dir = tempdir().unwrap();
        let runner = BinaryRunner::default();
        let err = runner.exec(dir.path(), BinaryExecRequest::new(vec![])).await.expect_err("reject");
        assert!(matches!(err, BinaryRunnerError::EmptyArgv));
    }
}
