//! Sandbox layer. Each backend isolates a single short-lived child process
//! that runs an agent task (a dynamic verify, a payload-runner, an ai-tool
//! call). The trait stays independent of every other nyx-agent crate so a
//! future VM backend can ship without dragging core/api/ai changes along.
//!
//! This crate is published so the `nyx-agent` binary can be installed
//! from crates.io with versioned internal dependencies. It is an
//! implementation detail of Nyx Agent, not a stable public API.
//!
//! Backends shipped today:
//!
//! * `process`: fork+exec with no isolation upgrade. The unhardened
//!   default used when an operator picks the `process` backend, or when
//!   no stronger backend is available on this host.
//! * `birdcage`: wraps the `birdcage` crate, which compiles to Linux
//!   landlock + seccomp or macOS Seatbelt. FS deny-by-default plus a
//!   single workspace-write exception; network deny unless
//!   [`SandboxOpts::allow_loopback`] is set.
//! * `libkrun`: macOS-first microVM via HVF (Linux+KVM also
//!   supported). Routed through a `libkrun-runner` helper binary so
//!   FFI symbol drift cannot crash the daemon.
//! * `firecracker`: Linux+KVM microVM. Routed through a
//!   `nyx-fc-runner` helper binary.
//! * `docker`: fallback container backend used when no stronger
//!   isolation is available; the chain-lane delegates to the
//!   docker-compose env-builder for the actual spin-up.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;

pub mod backend;
pub mod binary_runner;
pub mod chain_runner;
pub mod env;
pub mod payload_runner;
pub mod shim;
pub mod workspace;

pub use backend::birdcage::BirdcageSandbox;
pub use backend::firecracker::{firecracker_host_supported, FirecrackerSandbox, FirecrackerSpec};
pub use backend::libkrun::{libkrun_host_supported, LibkrunSandbox, LibkrunSpec};
pub use backend::process::ProcessSandbox;
pub use binary_runner::{
    derive_crash_signal, BinaryExecRequest, BinaryExecResult, BinaryExecStatus, BinaryRunner,
    BinaryRunnerError, CrashSignal,
};
pub use chain_runner::{
    ChainResult, ChainRun, ChainRunner, ChainRunnerError, ChainStep, ChainStepCapture,
    ChainVerdict, InconclusiveReason,
};
pub use payload_runner::{
    HarnessSource, HarnessSpecInput, PayloadRun, PayloadRunner, PayloadRunnerError,
};

/// Which backend produced (or is about to produce) a [`SandboxOutcome`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// `fork`/`exec` only; no kernel isolation upgrade.
    Process,
    /// Landlock+seccomp on Linux, Seatbelt on macOS.
    Birdcage,
    /// libkrun microVM via HVF (macOS) or KVM (Linux).
    Libkrun,
    /// Firecracker microVM (Linux+KVM).
    Firecracker,
    /// docker container fallback. Chain-lane spin-up delegates to
    /// the docker-compose env-builder.
    Docker,
}

impl BackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Process => "process",
            BackendKind::Birdcage => "birdcage",
            BackendKind::Libkrun => "libkrun",
            BackendKind::Firecracker => "firecracker",
            BackendKind::Docker => "docker",
        }
    }
}

/// Which scan lane the sandbox runs under. The chain lane spins up the
/// full dev-env replay alongside the AI-driven exploitation, which is
/// expensive; it gets a stricter concurrency cap than the fast lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lane {
    /// Static-pass + lightweight verifier work. Tolerates high
    /// fan-out.
    Fast,
    /// Full env-replay + AI exploitation. RAM-bound.
    Chain,
}

impl Lane {
    pub fn as_str(&self) -> &'static str {
        match self {
            Lane::Fast => "fast",
            Lane::Chain => "chain",
        }
    }
}

/// Per-lane simultaneous-spinup caps. The chain lane defaults to 2 (a
/// full env-replay can easily consume several GB of RAM); the fast
/// lane defaults to 8 (matches the `static_concurrency` ceiling on a
/// typical 8-core dev box).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneConcurrency {
    pub chain: usize,
    pub fast: usize,
}

impl LaneConcurrency {
    pub const DEFAULT_CHAIN: usize = 2;
    pub const DEFAULT_FAST: usize = 8;

    pub const fn defaults() -> Self {
        Self { chain: Self::DEFAULT_CHAIN, fast: Self::DEFAULT_FAST }
    }

    pub fn for_lane(&self, lane: Lane) -> usize {
        match lane {
            Lane::Chain => self.chain,
            Lane::Fast => self.fast,
        }
    }
}

impl Default for LaneConcurrency {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Refuse `(Lane::Fast, BackendKind::Birdcage)` when a caller asks for
/// `allow_loopback`. Birdcage cannot scope loopback any tighter than
/// "all network or none" (the field-level note on [`SandboxOpts::allow_loopback`]
/// covers this), so opening it on the fast lane hands the lightweight
/// verifier child a full egress channel it should never need. Chain-lane
/// tasks legitimately want loopback to reach env-builder services on the
/// host. Every other lane/backend combination passes through unchanged;
/// backends that can scope loopback at the kernel (libkrun, firecracker)
/// are free to allow it on either lane.
pub fn permits_loopback(lane: Lane, backend: BackendKind) -> bool {
    !matches!((lane, backend), (Lane::Fast, BackendKind::Birdcage))
}

/// Which backend the selector chose, plus a human-readable reason the
/// doctor / live-scan UI surfaces verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSelection {
    pub backend: BackendKind,
    pub reason: String,
}

/// Operator-facing backend label. Mirrors
/// `nyx_agent_core::config::SandboxBackend` but lives in this crate so
/// the sandbox layer does not depend on core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    /// Pick the strongest available backend for the lane at runtime.
    Auto,
    /// Pin to a specific backend; fall back if it cannot run here.
    Pinned(BackendKind),
}

/// Pick a backend for `lane` honouring the operator's `choice`. Auto
/// picks the strongest backend that can run on this host:
///
/// * Chain lane on macOS:   libkrun -> docker -> birdcage -> process.
/// * Chain lane on Linux:   firecracker -> libkrun -> docker -> birdcage -> process.
/// * Fast lane on macOS:    birdcage -> process.
/// * Fast lane on Linux:    birdcage -> process.
///
/// A pinned choice that cannot run here is downgraded to the same
/// auto-pick ladder with a reason explaining what failed.
pub fn select_backend(choice: BackendChoice, lane: Lane) -> BackendSelection {
    let auto = || auto_select(lane);
    match choice {
        BackendChoice::Auto => auto(),
        BackendChoice::Pinned(kind) => match probe(kind) {
            Ok(()) => {
                BackendSelection { backend: kind, reason: format!("pinned to {}", kind.as_str()) }
            }
            Err(err) => {
                let auto = auto();
                BackendSelection {
                    backend: auto.backend,
                    reason: format!(
                        "pinned {} unavailable ({err}); fell back to {} ({})",
                        kind.as_str(),
                        auto.backend.as_str(),
                        auto.reason
                    ),
                }
            }
        },
    }
}

fn auto_select(lane: Lane) -> BackendSelection {
    let ladder = auto_ladder(lane);
    for kind in ladder {
        match probe(*kind) {
            Ok(()) => {
                return BackendSelection {
                    backend: *kind,
                    reason: format!("auto-selected for {} lane", lane.as_str()),
                };
            }
            Err(_) => continue,
        }
    }
    // ProcessSandbox always probes Ok; this branch is unreachable in
    // practice but keeps the function total without a panic.
    BackendSelection {
        backend: BackendKind::Process,
        reason: format!("auto-selected fallback for {} lane", lane.as_str()),
    }
}

fn auto_ladder(lane: Lane) -> &'static [BackendKind] {
    match lane {
        Lane::Chain => {
            #[cfg(target_os = "macos")]
            {
                &[
                    BackendKind::Libkrun,
                    BackendKind::Docker,
                    BackendKind::Birdcage,
                    BackendKind::Process,
                ]
            }
            #[cfg(target_os = "linux")]
            {
                &[
                    BackendKind::Firecracker,
                    BackendKind::Libkrun,
                    BackendKind::Docker,
                    BackendKind::Birdcage,
                    BackendKind::Process,
                ]
            }
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            {
                &[BackendKind::Docker, BackendKind::Process]
            }
        }
        Lane::Fast => {
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            {
                &[BackendKind::Birdcage, BackendKind::Process]
            }
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            {
                &[BackendKind::Process]
            }
        }
    }
}

/// Options for a single sandboxed child.
#[derive(Debug, Clone)]
pub struct SandboxOpts {
    /// Path the child can read and write. Must already exist.
    pub workspace: PathBuf,
    /// `argv[0]` plus arguments. `argv[0]` is the program to exec.
    pub argv: Vec<String>,
    /// Working directory. Defaults to `workspace`.
    pub cwd: Option<PathBuf>,
    /// Environment variables passed to the child. The parent's `env` is
    /// not inherited.
    pub env: Vec<(String, String)>,
    /// Wall-clock timeout. Backends that miss it report
    /// [`SandboxStatus::TimedOut`].
    pub timeout: Duration,
    /// Allow loopback network traffic. birdcage cannot scope further than
    /// "all network or none": when set, all egress is allowed.
    pub allow_loopback: bool,
    /// Which scan lane spawned this child. `None` means lane-unannotated
    /// (the caller did not opt in to lane policy enforcement). When set,
    /// [`permits_loopback`] gates `allow_loopback` against the backend
    /// at [`Sandbox::run`] time; the birdcage backend refuses
    /// `(Lane::Fast, allow_loopback = true)` with [`SandboxError::Config`].
    pub lane: Option<Lane>,
    /// Extra read-only paths visible to the sandboxed child (defaults
    /// like `/lib`, `/usr` are added by the backend).
    pub allow_read: Vec<PathBuf>,
    /// Extra read-write paths visible to the sandboxed child (in addition
    /// to `workspace`).
    pub allow_write: Vec<PathBuf>,
    /// Cap captured stdout/stderr at this many bytes each. The child is
    /// not killed when its output exceeds the cap; further bytes are
    /// silently dropped.
    pub max_output_bytes: usize,
    /// Snapshot the workspace from this source directory before
    /// spawning. When set, the backend builds a fresh COW snapshot
    /// of `snapshot_from` via [`workspace::snapshot`] under a private
    /// tempdir and overrides [`SandboxOpts::workspace`] to point at
    /// the new copy; the tempdir is reaped on
    /// [`Sandbox::wait`]/[`Sandbox::kill`]. Lets a caller hand the
    /// sandbox a clean COW view of a source tree without staging the
    /// copy itself. Defaults to `None` (the sandbox uses
    /// `workspace` directly).
    pub snapshot_from: Option<PathBuf>,
    /// Workspace-relative paths the backend reads after the child
    /// exits but before the sandbox tears down (in particular, before
    /// `snapshot_from`'s tempdir drops). Each declared path lands in
    /// [`SandboxOutcome::captured_files`] keyed by its workspace-relative
    /// form: `Some(bytes)` when the file existed at capture time,
    /// `None` when it did not. Lets callers observe per-run side
    /// effects (e.g. a `SinkProbe` sentinel file) without keeping the
    /// workspace alive past `wait()`.
    pub capture_files: Vec<PathBuf>,
}

impl SandboxOpts {
    /// New options with sane defaults for a short-lived agent task.
    pub fn new(workspace: PathBuf, argv: Vec<String>) -> Self {
        Self {
            workspace,
            argv,
            cwd: None,
            env: Vec::new(),
            timeout: Duration::from_secs(30),
            allow_loopback: false,
            lane: None,
            allow_read: Vec::new(),
            allow_write: Vec::new(),
            max_output_bytes: 1 << 20,
            snapshot_from: None,
            capture_files: Vec::new(),
        }
    }

    /// Builder: snapshot `src` into a private COW tempdir at run time
    /// and use it as the child's workspace. The tempdir is owned by
    /// the backend's `RunningChild` and dropped (removed) when the
    /// child is reaped. Calling [`SandboxOpts::workspace`] after this
    /// is irrelevant; the backend overrides it with the snapshot
    /// destination. Returns `self` so it composes with the other
    /// `SandboxOpts` setters.
    pub fn with_snapshot_from(mut self, src: PathBuf) -> Self {
        self.snapshot_from = Some(src);
        self
    }

    /// Builder: declare a workspace-relative path the backend should
    /// read at capture time (after `wait`, before snapshot drop) and
    /// stamp on [`SandboxOutcome::captured_files`]. Multiple calls
    /// accumulate. Composes with the other `SandboxOpts` setters.
    pub fn capture_file(mut self, rel: PathBuf) -> Self {
        self.capture_files.push(rel);
        self
    }
}

/// Final state of a sandboxed child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxStatus {
    /// Child exited with the recorded code.
    Exited(i32),
    /// Child died from a signal (Unix only; on other platforms reported as
    /// `Exited(-1)`).
    Signaled(i32),
    /// Backend tore the child down because [`SandboxOpts::timeout`] fired.
    TimedOut,
    /// Caller invoked [`Sandbox::kill`].
    Killed,
}

impl SandboxStatus {
    /// Did the sandbox successfully contain the child? A `contained`
    /// child either failed to exec, exited non-zero, was killed by the
    /// kernel, or was torn down by the harness: anything except a clean
    /// `exit(0)`. The escape regression suite asserts this.
    pub fn contained(&self) -> bool {
        !matches!(self, SandboxStatus::Exited(0))
    }
}

/// The captured result of a single sandboxed run.
#[derive(Debug, Clone)]
pub struct SandboxOutcome {
    pub backend: BackendKind,
    pub status: SandboxStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub duration: Duration,
    /// Birdcage exception refusals the shim collected during sandbox
    /// setup. A non-empty list means at least one declared exception
    /// (allow_read / allow_write / allow_env / Networking) did not
    /// take effect, which typically explains a follow-on "permission
    /// denied" inside the sandboxee. Always empty on backends that do
    /// not exercise the shim's fd-3 report channel.
    pub refusals: Vec<String>,
    /// Files the backend captured from the post-wait workspace, keyed
    /// by the workspace-relative path the caller declared via
    /// [`SandboxOpts::capture_files`]. `Some(bytes)` when the file
    /// existed; `None` when it did not. Lets callers observe per-run
    /// side effects (e.g. a `SinkProbe` sentinel) without keeping the
    /// workspace alive past `wait()`. Empty when the caller declared
    /// no captures.
    pub captured_files: HashMap<PathBuf, Option<Vec<u8>>>,
}

/// Sandbox error surface. Backend-specific failures are folded into the
/// closest matching variant so callers can program against the trait
/// without reaching for downcasts.
#[derive(Debug, Error)]
pub enum SandboxError {
    /// The backend cannot run on this host (e.g. birdcage on Windows).
    #[error("backend {backend} unavailable: {reason}")]
    BackendUnavailable { backend: &'static str, reason: String },
    /// `fork`/`exec` failed before any sandbox lock was applied.
    #[error("spawn failed: {0}")]
    Spawn(#[source] std::io::Error),
    /// Workspace setup (the COW snapshot) failed.
    #[error("workspace setup failed: {0}")]
    Workspace(#[source] std::io::Error),
    /// Misconfigured opts (empty argv, non-existent workspace, etc.).
    #[error("sandbox config rejected: {0}")]
    Config(String),
    /// Caller invoked `kill`/`wait` in an order the backend cannot honour.
    #[error("invalid sandbox state: {0}")]
    State(&'static str),
    /// Generic I/O failure while running the child.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// The sandbox surface used by every consumer (chain lane, payload runner,
/// dynamic verifier). Implementors own a single child at a time: call
/// [`Sandbox::run`] once, then either [`Sandbox::wait`] or
/// [`Sandbox::kill`] before another `run`.
#[allow(async_fn_in_trait)]
pub trait Sandbox: Send {
    fn backend(&self) -> BackendKind;

    /// Spawn the child described by `opts`. Returns once the kernel has
    /// accepted the new process. The child may still be sandboxing
    /// itself when this returns (birdcage runs its `lock()` in a
    /// `pre_exec` hook, so the sandbox is in place by the time the
    /// target binary's `main` runs).
    async fn run(&mut self, opts: SandboxOpts) -> Result<(), SandboxError>;

    /// SIGKILL the running child. Idempotent: calling on an already-exited
    /// child returns `Ok(())`.
    async fn kill(&mut self) -> Result<(), SandboxError>;

    /// Block until the child exits, honouring the opts.timeout passed to
    /// [`Sandbox::run`]. After this returns, [`Sandbox::logs`] yields the
    /// captured output and the backend is ready for another `run`.
    async fn wait(&mut self) -> Result<SandboxOutcome, SandboxError>;

    /// Stdout, then stderr, captured from the most recent run. Only
    /// meaningful after [`Sandbox::wait`].
    fn logs(&self) -> (&[u8], &[u8]);
}

/// Lightweight readiness probe: returns the static set of backends
/// that the platform *could* run. Construction-time probes
/// ([`probe`]) further check that the kernel surface + helper binaries
/// are present.
pub fn available_backends() -> &'static [BackendKind] {
    #[cfg(target_os = "macos")]
    {
        &[BackendKind::Process, BackendKind::Birdcage, BackendKind::Libkrun, BackendKind::Docker]
    }
    #[cfg(target_os = "linux")]
    {
        &[
            BackendKind::Process,
            BackendKind::Birdcage,
            BackendKind::Libkrun,
            BackendKind::Firecracker,
            BackendKind::Docker,
        ]
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        &[BackendKind::Process, BackendKind::Docker]
    }
}

/// Return `Ok(())` if `backend` can be constructed on this host, else
/// describe why it cannot. Callers use this to short-circuit a doctor
/// check or to fall back to a weaker backend.
pub fn probe(backend: BackendKind) -> Result<(), SandboxError> {
    match backend {
        BackendKind::Process => Ok(()),
        BackendKind::Birdcage => {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            {
                // The kernel surface exists; the shim binary is the
                // second gate. Surfacing its absence here makes the
                // doctor's `select_backend` ladder downgrade to
                // `Process` instead of silently choosing Birdcage and
                // tripping at the first `run()`.
                backend::birdcage::BirdcageSandbox::new().map(|_| ())
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            {
                Err(SandboxError::BackendUnavailable {
                    backend: "birdcage",
                    reason: "requires Linux landlock or macOS Seatbelt".into(),
                })
            }
        }
        BackendKind::Libkrun => {
            if !libkrun_host_supported() {
                return Err(SandboxError::BackendUnavailable {
                    backend: "libkrun",
                    reason: "requires macOS with Hypervisor.framework or Linux with KVM".into(),
                });
            }
            // Helper binary presence is the second gate.
            backend::libkrun::LibkrunSandbox::new().map(|_| ())
        }
        BackendKind::Firecracker => {
            if !firecracker_host_supported() {
                return Err(SandboxError::BackendUnavailable {
                    backend: "firecracker",
                    reason: "requires Linux with /dev/kvm".into(),
                });
            }
            backend::firecracker::FirecrackerSandbox::new().map(|_| ())
        }
        BackendKind::Docker => {
            if backend::which_on_path("docker").is_some() {
                Ok(())
            } else {
                Err(SandboxError::BackendUnavailable {
                    backend: "docker",
                    reason: "docker not found on PATH".into(),
                })
            }
        }
    }
}

/// Shared lock for tests that mutate process-wide env vars (notably
/// `$NYX_LIBKRUN_RUNNER`). Tests in this crate run in the same lib-test
/// binary and the default cargo test runner is multi-threaded, so two
/// env-mutating tests can clobber each other's `set_var`/`remove_var`
/// pairs mid-call. Hold this guard for the duration of any env
/// mutation.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_policy_refuses_fast_birdcage_loopback() {
        assert!(!permits_loopback(Lane::Fast, BackendKind::Birdcage));
    }

    #[test]
    fn lane_policy_allows_chain_birdcage_loopback() {
        assert!(permits_loopback(Lane::Chain, BackendKind::Birdcage));
    }

    #[test]
    fn lane_policy_allows_fast_process_loopback() {
        // Process backend has no egress cage, so the policy waves it
        // through on either lane; the gate only fires for birdcage on
        // the fast lane.
        assert!(permits_loopback(Lane::Fast, BackendKind::Process));
    }

    #[test]
    fn lane_policy_allows_fast_libkrun_and_firecracker_loopback() {
        assert!(permits_loopback(Lane::Fast, BackendKind::Libkrun));
        assert!(permits_loopback(Lane::Fast, BackendKind::Firecracker));
        assert!(permits_loopback(Lane::Fast, BackendKind::Docker));
    }

    #[test]
    fn lane_concurrency_defaults_match_plan() {
        let cap = LaneConcurrency::defaults();
        assert_eq!(cap.chain, 2);
        assert_eq!(cap.fast, 8);
        assert_eq!(cap.for_lane(Lane::Chain), 2);
        assert_eq!(cap.for_lane(Lane::Fast), 8);
    }

    #[test]
    fn select_auto_chain_picks_strongest_for_host() {
        let sel = select_backend(BackendChoice::Auto, Lane::Chain);
        // The ladder is platform-specific; what matters is that some
        // backend always selects, the auto reason is filled in, and
        // the chosen backend probes Ok at the time of the call.
        assert!(probe(sel.backend).is_ok(), "selected backend must probe Ok");
        assert!(sel.reason.contains("chain"));
    }

    #[test]
    fn select_auto_fast_picks_birdcage_or_process() {
        let sel = select_backend(BackendChoice::Auto, Lane::Fast);
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            assert!(matches!(sel.backend, BackendKind::Birdcage | BackendKind::Process));
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            assert_eq!(sel.backend, BackendKind::Process);
        }
    }

    #[test]
    fn select_pinned_falls_back_when_unavailable() {
        // Force libkrun unavailable by pointing the env override at a
        // non-existent helper. The selector should fall back to the
        // auto-pick and stamp a reason explaining the downgrade.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("NYX_LIBKRUN_RUNNER", "/definitely/does/not/exist/libkrun-runner");
        let sel = select_backend(BackendChoice::Pinned(BackendKind::Libkrun), Lane::Chain);
        std::env::remove_var("NYX_LIBKRUN_RUNNER");
        assert_ne!(
            sel.backend,
            BackendKind::Libkrun,
            "pinned libkrun must downgrade when runner is missing"
        );
        assert!(sel.reason.contains("unavailable"));
        assert!(sel.reason.contains("fell back"));
    }

    #[test]
    fn probe_process_is_always_ok() {
        assert!(probe(BackendKind::Process).is_ok());
    }

    #[test]
    fn available_backends_includes_process() {
        let kinds = available_backends();
        assert!(kinds.contains(&BackendKind::Process));
    }

    #[test]
    fn backend_kind_as_str_round_trip() {
        for k in [
            BackendKind::Process,
            BackendKind::Birdcage,
            BackendKind::Libkrun,
            BackendKind::Firecracker,
            BackendKind::Docker,
        ] {
            assert!(!k.as_str().is_empty());
        }
    }
}
