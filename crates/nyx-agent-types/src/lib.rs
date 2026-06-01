//! Shared serde structs, enums, and event-bus types used by every other
//! nyx-agent crate.
//!
//! This crate is published so the `nyx-agent` binary can be installed
//! from crates.io with versioned internal dependencies. It is an
//! implementation detail of Nyx Agent, not a stable public API.

pub mod agent;
pub mod api;
pub mod attack_graph;
pub mod budget;
pub mod business_logic;
pub mod chain;
pub mod event;
pub mod finding;
pub mod integration;
pub mod live_plan;
pub mod novel;
pub mod payload;
pub mod product;
pub mod project;
pub mod repo;
pub mod run;
pub mod spec;
pub mod target;
pub mod trace;
pub mod verify;

pub use event::{
    AgentEvent, AiEvent, BudgetEvent, EventSink, EventStream, FindingEvent, QuarantineEvent,
    RepoOutcomeTag, ReproEvent, RunEvent, SandboxEvent,
};
pub use live_plan::{
    AuthzBrowserRoleComparisonPlan, AuthzObjectOwnershipPlan, AuthzOracle, AuthzOwnedObject,
    AuthzRoleComparisonPlan, BrowserOracle, BrowserStep, BrowserWorkflowPlan, DifferentialHttpPlan,
    DifferentialOracle, HttpOracle, HttpWorkflowPlan, LiveHttpRequest, LivePlanValidationError,
    LiveTestPlan, NoPlanReason, NoPlanReasonCode, SingleHttpPlan,
};
pub use target::{LocalBinaryTarget, PentestTarget};
pub use verify::{Oracle, VerifyResult, VerifyRun, VerifyVerdict};
