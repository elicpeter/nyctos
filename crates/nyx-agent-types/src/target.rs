//! What a pentest run is pointed at.
//!
//! Historically scope was implicitly "repos + `target_urls`": the whole
//! pipeline assumed a live HTTP dev app. [`PentestTarget`] makes the
//! target shape a first-class, serializable choice so an HTTP app and a
//! local binary are siblings rather than one hard-coded mode.
//!
//! The enum crosses the wire to the dashboard and the generated TS
//! bindings, so it lives here in `nyx-agent-types` alongside the other
//! shared schema types rather than in a runtime crate.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// What a run is pointed at. A live web app (today's only mode) or a
/// local binary exercised against agent-crafted inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PentestTarget {
    /// Live HTTP dev app. `urls` mirrors the legacy `target_urls`.
    HttpApp { urls: Vec<String> },
    /// A local executable exercised against agent-crafted inputs.
    LocalBinary(LocalBinaryTarget),
}

impl PentestTarget {
    /// Borrow the local-binary payload when this target is a binary.
    pub fn as_local_binary(&self) -> Option<&LocalBinaryTarget> {
        match self {
            PentestTarget::LocalBinary(target) => Some(target),
            PentestTarget::HttpApp { .. } => None,
        }
    }

    /// `true` for [`PentestTarget::LocalBinary`].
    pub fn is_local_binary(&self) -> bool {
        matches!(self, PentestTarget::LocalBinary(_))
    }
}

/// A local executable target. The operator pins the program and the
/// fixed argv shape; the agent only fills the declared slots and the
/// staged file contents — it never picks an arbitrary executable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct LocalBinaryTarget {
    /// Path to the executable, OR the program name resolved on the
    /// sandbox `PATH` (e.g. `"curl"`). Host-resolved to an absolute path
    /// before the first exec; the resolved path is what gets exec'd and
    /// is pinned for the rest of the run.
    pub program: String,
    /// Fixed leading args the agent may NOT change (e.g. `["--config"]`).
    /// The agent appends/inserts its crafted args around these per the
    /// `argv_template`. Empty = agent controls all argv.
    #[serde(default)]
    pub base_args: Vec<String>,
    /// argv template with slots the agent fills. Slot syntax:
    ///   `@FILE:<logical-name>` -> replaced with the workspace path of a
    ///                             file the agent staged via `sandbox.write_file`.
    ///   `@ARG`                 -> replaced with a literal arg string.
    /// Example: `["@ARG", "@FILE:input"]` for `curl <arg> <file>`.
    /// `None` = agent supplies the full argv each exec call.
    #[serde(default)]
    pub argv_template: Option<Vec<String>>,
    /// Whether the target legitimately needs loopback network (e.g.
    /// `curl` to a local server the agent also controls). Defaults
    /// false. Gated by lane/backend policy.
    #[serde(default)]
    pub allow_loopback: bool,
    /// Optional path to a known-good oracle build (e.g. an ASAN build or
    /// a reference implementation) for differential checks. `None` =
    /// single build, crash-only oracle.
    #[serde(default)]
    pub oracle_program: Option<String>,
    /// Per-exec wall-clock cap in seconds. Defaults to 10s when absent.
    #[serde(default)]
    pub per_exec_timeout_secs: Option<u64>,
    /// Extra read-only paths the target needs (libs, data files). Maps
    /// to `SandboxOpts.allow_read`.
    #[serde(default)]
    pub allow_read: Vec<String>,
}

impl LocalBinaryTarget {
    /// Default per-exec wall-clock cap when the target leaves it unset.
    pub const DEFAULT_TIMEOUT_SECS: u64 = 10;

    /// New target pinned to `program` with all other fields defaulted.
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            base_args: Vec::new(),
            argv_template: None,
            allow_loopback: false,
            oracle_program: None,
            per_exec_timeout_secs: None,
            allow_read: Vec::new(),
        }
    }

    /// Resolve the per-exec timeout, falling back to the default.
    pub fn timeout_secs(&self) -> u64 {
        self.per_exec_timeout_secs.unwrap_or(Self::DEFAULT_TIMEOUT_SECS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_app_round_trips_with_kind_tag() {
        let t = PentestTarget::HttpApp { urls: vec!["http://localhost:3000".into()] };
        let json = serde_json::to_value(&t).unwrap();
        assert_eq!(json["kind"], "http_app");
        let back: PentestTarget = serde_json::from_value(json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn local_binary_round_trips_and_defaults() {
        let json = serde_json::json!({
            "kind": "local_binary",
            "program": "curl",
            "argv_template": ["@ARG", "@FILE:input"],
        });
        let t: PentestTarget = serde_json::from_value(json).unwrap();
        let bin = t.as_local_binary().expect("local binary");
        assert_eq!(bin.program, "curl");
        assert_eq!(bin.argv_template.as_deref().unwrap(), &["@ARG", "@FILE:input"]);
        assert!(!bin.allow_loopback);
        assert!(bin.base_args.is_empty());
        assert_eq!(bin.timeout_secs(), LocalBinaryTarget::DEFAULT_TIMEOUT_SECS);
    }

    #[test]
    fn timeout_override_is_honoured() {
        let mut bin = LocalBinaryTarget::new("/bin/ls");
        assert_eq!(bin.timeout_secs(), 10);
        bin.per_exec_timeout_secs = Some(30);
        assert_eq!(bin.timeout_secs(), 30);
    }
}
