//! ChainReasoning agent task.
//!
//! Given the full run's finding graph (nodes for static-pass findings,
//! `Reaches` edges, cross-repo edges), drive an `AiRuntime::one_shot`
//! call against a strict JSON contract, parse the response, validate
//! the chains, and produce either a
//! [`ChainReasoningOutcome::Ranked`] envelope for the binary to persist
//! or [`ChainReasoningOutcome::NoChains`] when the model could not
//! identify a single chain (or two attempts both produced malformed
//! output).
//!
//! The task crate stays vendor-neutral. It does not depend on
//! `nyx-agent-core::store`; the binary turns a `Ranked` outcome into
//! `ChainRecord` rows (one per chain) at persistence time and stamps
//! every member finding's `chain_id` back-link.

use std::collections::HashSet;

use nyx_agent_types::agent::{
    AgentResult, AgentTask, AgentTraceMetrics, AiError, Budget, BudgetKind, Prompt, Response,
};
use nyx_agent_types::chain::{
    ChainCandidate, ChainReasoningInput, ChainReasoningOutput, CHAIN_REASONING_PROMPT_VERSION,
};
use nyx_agent_types::event::EventSink;

use crate::runtime::AiRuntime;

/// First-attempt system prompt. Plain JSON contract; describes the
/// `ChainReasoningOutput` shape and the cross-repo emphasis. Source
/// lives at `crates/nyx-agent-ai/src/prompts/chain_reasoning.v1.md`.
const SYSTEM_PROMPT_V1: &str = include_str!("../prompts/chain_reasoning.v1.md");

/// Retry system prompt. Identical contract with the explicit "your
/// previous reply did not validate" framing.
const SYSTEM_PROMPT_V1_STRICTER: &str = include_str!("../prompts/chain_reasoning.v1_stricter.md");

/// Outcome of one `run` invocation. The binary turns `Ranked` into one
/// `ChainRecord` per chain and stamps every member finding's `chain_id`
/// back-link; `NoChains` is recorded only in the agent trace.
#[derive(Debug, Clone)]
pub enum ChainReasoningOutcome {
    /// Both deserialise + validate gates passed. `output.chains` is
    /// non-empty.
    Ranked {
        run_id: String,
        output: ChainReasoningOutput,
        prompt_version: String,
        spent_usd_micros: i64,
        attempts: u32,
        metrics: AgentTraceMetrics,
    },
    /// Both attempts produced malformed or empty output. The binary
    /// surfaces `reason` in the agent-trace store; nothing is persisted
    /// to the `chains` table.
    NoChains {
        run_id: String,
        reason: String,
        spent_usd_micros: i64,
        attempts: u32,
        metrics: AgentTraceMetrics,
    },
}

/// Repository workspace the agentic ChainReasoning mode may inspect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainReasoningWorkspace {
    pub repo: String,
    pub root: String,
}

/// Drive one ChainReasoning call for `input`.
///
/// `cap_usd_micros` is the per-call budget. Both attempts share the
/// same `(run_id, kind)` budget bucket via the `BudgetTracker` host
/// port.
pub async fn run<R: AiRuntime + ?Sized>(
    runtime: &R,
    input: &ChainReasoningInput,
    seeds: &[Vec<String>],
    sink: EventSink,
    cap_usd_micros: i64,
) -> Result<ChainReasoningOutcome, AiError> {
    let task_id = format!("chain-{}", input.run_id);
    let budget =
        || Budget { run_id: input.run_id.clone(), kind: BudgetKind::OneShot, cap_usd_micros };

    let node_ids: HashSet<String> = input.nodes.iter().map(|n| n.id.clone()).collect();
    let edge_pairs: HashSet<(String, String)> =
        input.edges.iter().map(|e| (e.from.clone(), e.to.clone())).collect();

    let prompt = build_prompt(SYSTEM_PROMPT_V1, &task_id, input, seeds);
    let resp1: Response = runtime.one_shot(prompt, budget(), sink.clone()).await?;
    let cost1 = resp1.cost_usd_micros;
    let metrics1 = AgentTraceMetrics::from_response(&resp1);
    let first_err = match parse_and_validate(&resp1.content, &node_ids, &edge_pairs) {
        Ok(output) => {
            return Ok(ChainReasoningOutcome::Ranked {
                run_id: input.run_id.clone(),
                output,
                prompt_version: resp1.prompt_version,
                spent_usd_micros: cost1,
                attempts: 1,
                metrics: metrics1,
            });
        }
        Err(msg) => msg,
    };

    let prompt2 = build_prompt(SYSTEM_PROMPT_V1_STRICTER, &task_id, input, seeds);
    let resp2: Response = runtime.one_shot(prompt2, budget(), sink).await?;
    let total_cost = cost1 + resp2.cost_usd_micros;
    let metrics_total = metrics1.merge(AgentTraceMetrics::from_response(&resp2));
    match parse_and_validate(&resp2.content, &node_ids, &edge_pairs) {
        Ok(output) => Ok(ChainReasoningOutcome::Ranked {
            run_id: input.run_id.clone(),
            output,
            prompt_version: resp2.prompt_version,
            spent_usd_micros: total_cost,
            attempts: 2,
            metrics: metrics_total,
        }),
        Err(second_err) => Ok(ChainReasoningOutcome::NoChains {
            run_id: input.run_id.clone(),
            reason: format!(
                "chain reasoning failed twice (attempt 1: {first_err}; attempt 2: {second_err})"
            ),
            spent_usd_micros: total_cost,
            attempts: 2,
            metrics: metrics_total,
        }),
    }
}

/// Stronger ChainReasoning mode for CLI-backed runtimes.
///
/// The agent receives the same graph contract as the deterministic
/// one-shot worker, plus workspace roots it can inspect with native
/// file/search tools before returning the strict JSON object. The same
/// validation gate is applied afterward so the agent can roam through
/// code, but it cannot persist a chain unless every adjacent member is
/// still backed by the input graph.
pub async fn run_agentic<R: AiRuntime + ?Sized>(
    runtime: &R,
    input: &ChainReasoningInput,
    seeds: &[Vec<String>],
    workspaces: &[ChainReasoningWorkspace],
    sink: EventSink,
    cap_usd_micros: i64,
) -> Result<ChainReasoningOutcome, AiError> {
    if !runtime.supports_agent_loop() {
        return run(runtime, input, seeds, sink, cap_usd_micros).await;
    }

    let task_id = format!("chain-agentic-{}", input.run_id);
    let node_ids: HashSet<String> = input.nodes.iter().map(|n| n.id.clone()).collect();
    let edge_pairs: HashSet<(String, String)> =
        input.edges.iter().map(|e| (e.from.clone(), e.to.clone())).collect();
    let task = build_agent_task(&task_id, input, seeds, workspaces);
    let budget =
        Budget { run_id: input.run_id.clone(), kind: BudgetKind::AgentLoop, cap_usd_micros };
    let result: AgentResult = runtime.agent_loop(task, budget, sink).await?;
    let spent_usd_micros = result.cost_usd_micros;
    let metrics = AgentTraceMetrics::from_agent_result(&result);
    match parse_and_validate(&result.final_message, &node_ids, &edge_pairs) {
        Ok(output) => Ok(ChainReasoningOutcome::Ranked {
            run_id: input.run_id.clone(),
            output,
            prompt_version: result.prompt_version,
            spent_usd_micros,
            attempts: 1,
            metrics,
        }),
        Err(err) => Ok(ChainReasoningOutcome::NoChains {
            run_id: input.run_id.clone(),
            reason: format!("agentic chain reasoning did not return valid chains: {err}"),
            spent_usd_micros,
            attempts: 1,
            metrics,
        }),
    }
}

fn build_prompt(
    system: &str,
    task_id: &str,
    input: &ChainReasoningInput,
    seeds: &[Vec<String>],
) -> Prompt {
    let user = render_user_message(input, seeds);
    Prompt {
        prompt_version: CHAIN_REASONING_PROMPT_VERSION.to_string(),
        task_id: task_id.to_string(),
        model: None,
        system: system.to_string(),
        user,
        // Chain output stays small (ids + a short paragraph each). Cap
        // is generous for max_chains = 10 with multi-sentence rationales.
        max_output_tokens: 2048,
        temperature: 0.0,
        seed: None,
    }
}

/// Serialise the input graph in a compact, model-friendly layout.
/// `serde_json::to_string_pretty` is intentionally avoided; the typed
/// nodes / edges section is easier for the model to consume than a
/// pretty-printed object.
fn render_user_message(input: &ChainReasoningInput, seeds: &[Vec<String>]) -> String {
    let mut out = String::new();
    out.push_str(&format!("run_id     = {}\n", input.run_id));
    out.push_str(&format!("repos      = [{}]\n", input.repos.join(", ")));
    out.push_str(&format!("max_chains = {}\n", input.max_chains));
    out.push('\n');

    out.push_str(&render_seed_section(seeds));

    out.push_str("nodes:\n");
    for n in &input.nodes {
        let line_str = n.line.map(|l| format!(" L{l}")).unwrap_or_default();
        out.push_str(&format!(
            "- id={} graph_kind={} repo={} kind={} cap={} rule={} sev={} label={} path={}{} routes=[{}] roles=[{}] objects=[{}] evidence_refs=[{}]\n",
            n.id,
            n.graph_kind.as_deref().unwrap_or("finding"),
            n.repo,
            n.kind,
            n.cap,
            n.rule,
            n.severity,
            n.label.as_deref().unwrap_or(""),
            n.path,
            line_str,
            n.routes.join(","),
            n.roles.join(","),
            n.objects.join(","),
            n.evidence_refs.join(","),
        ));
    }
    out.push('\n');

    out.push_str("edges:\n");
    if input.edges.is_empty() {
        out.push_str("- (none)\n");
    } else {
        for e in &input.edges {
            let cross = if e.cross_repo { " cross_repo" } else { "" };
            let edge_id = e.edge_id.as_deref().unwrap_or("");
            let evidence_ref = e.evidence_ref.as_deref().unwrap_or("");
            let source = e.source.as_deref().unwrap_or("");
            out.push_str(&format!(
                "- {} --[{} id={} evidence_ref={} source={}]--> {}{}\n",
                e.from, e.label, edge_id, evidence_ref, source, e.to, cross
            ));
        }
    }
    out
}

/// Render the pre-extracted candidate paths the graph engine found.
/// These are graph-backed entry->impact skeletons; the model is asked
/// to treat them as the primary leads to verify, extend, and rank,
/// rather than rediscovering paths in the adjacency list. Empty when no
/// path was extracted (the model then reasons from the graph alone).
fn render_seed_section(seeds: &[Vec<String>]) -> String {
    if seeds.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "candidate_chains (graph-extracted entry->impact paths; verify, extend, and rank these first):\n",
    );
    for (i, path) in seeds.iter().enumerate() {
        out.push_str(&format!("- seed{i}: {}\n", path.join(" -> ")));
    }
    out.push('\n');
    out
}

fn build_agent_task(
    task_id: &str,
    input: &ChainReasoningInput,
    seeds: &[Vec<String>],
    workspaces: &[ChainReasoningWorkspace],
) -> AgentTask {
    let objective = format!(
        "{}\n\n{}\n\n{}",
        agentic_objective_header(input),
        render_workspace_section(workspaces),
        render_user_message(input, seeds)
    );
    AgentTask {
        prompt_version: format!("{CHAIN_REASONING_PROMPT_VERSION}.agentic"),
        task_id: task_id.to_string(),
        system: include_str!("../prompts/chain_reasoning.v1.md").to_string(),
        objective,
        tools: vec!["Read".to_string(), "Grep".to_string(), "Bash".to_string()],
        working_directory: workspaces.first().map(|w| w.root.clone()),
        max_turns: 40,
    }
}

fn agentic_objective_header(input: &ChainReasoningInput) -> String {
    format!(
        r#"You may inspect the repository workspaces before proposing chains.

A `candidate_chains` section lists graph-extracted entry->impact paths.
Treat those as your primary leads: read the code along each one to
confirm it is genuinely exploitable, strengthen or extend it, merge
overlapping seeds, and discard seeds the code refutes. You may also
propose other chains the graph supports.

Use the graph as the source of truth for valid member ids and valid adjacent links, but use code reading/search to find the strongest exploit story:
- inspect source around important node paths/lines;
- search handlers, auth checks, route definitions, model/service calls, and tests;
- look for preconditions that turn low/medium leads into a serious terminal impact;
- prefer chains that end in verification_attempt or verified_vulnerability nodes;
- keep unverified chains explicit by filling missing_verification_steps.

You MUST finish with exactly one JSON object matching the ChainReasoning schema. No markdown, no commentary, no code fences. Emit at most {} chains."#,
        input.max_chains
    )
}

fn render_workspace_section(workspaces: &[ChainReasoningWorkspace]) -> String {
    if workspaces.is_empty() {
        return "workspaces:\n- (none supplied; reason only from graph context)".to_string();
    }
    let mut out = String::from("workspaces:\n");
    for ws in workspaces {
        out.push_str(&format!("- repo={} root={}\n", ws.repo, ws.root));
    }
    out
}

fn parse_and_validate(
    raw: &str,
    node_ids: &HashSet<String>,
    edge_pairs: &HashSet<(String, String)>,
) -> Result<ChainReasoningOutput, String> {
    let body = strip_code_fence(raw.trim());
    let out: ChainReasoningOutput =
        serde_json::from_str(body).map_err(|e| format!("malformed json: {e}"))?;
    if out.chains.is_empty() {
        return Err("chains array was empty".into());
    }
    validate_chains(&out.chains, node_ids, edge_pairs)?;
    Ok(out)
}

fn validate_chains(
    chains: &[ChainCandidate],
    node_ids: &HashSet<String>,
    edge_pairs: &HashSet<(String, String)>,
) -> Result<(), String> {
    for (i, c) in chains.iter().enumerate() {
        if c.member_ids.len() < 2 {
            return Err(format!("chain {i}: member_ids must contain at least 2 entries"));
        }
        if c.rationale.trim().is_empty() {
            return Err(format!("chain {i}: rationale was empty"));
        }
        for id in &c.member_ids {
            if !node_ids.contains(id) {
                return Err(format!("chain {i}: member id {id:?} not present in the input graph"));
            }
        }
        // Real exploit chains never visit the same node twice in
        // succession. A model that copies a node id N times produces
        // a "1-step loop" with no analytic value.
        for w in c.member_ids.windows(2) {
            if w[0] == w[1] {
                return Err(format!("chain {i}: member_ids has consecutive duplicate {:?}", w[0]));
            }
            if !edge_pairs.contains(&(w[0].clone(), w[1].clone())) {
                return Err(format!(
                    "chain {i}: adjacent members {:?} -> {:?} are not connected by an input graph edge",
                    w[0], w[1]
                ));
            }
        }
        if c.confidence > 100 {
            return Err(format!("chain {i}: confidence must be between 0 and 100"));
        }
    }
    Ok(())
}

fn strip_code_fence(s: &str) -> &str {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("```json") {
        return rest.trim().trim_end_matches("```").trim();
    }
    if let Some(rest) = s.strip_prefix("```") {
        return rest.trim().trim_end_matches("```").trim();
    }
    s
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use nyx_agent_types::agent::{
        AgentResult, AgentTask, AiError, CacheStats, CostEstimate, TokenUsage,
    };
    use nyx_agent_types::chain::{
        ChainReasoningEdge, ChainReasoningNode, NODE_KIND_ENTRY, NODE_KIND_SINK,
    };
    use nyx_agent_types::event::AgentEvent;
    use tokio::sync::broadcast;

    use super::*;
    use crate::runtime::{AiRuntime, BudgetTracker, InMemoryBudgetTracker};
    use nyx_agent_types::agent::BudgetKind;

    /// Scripted runtime that replays a fixed sequence of `one_shot`
    /// responses. Same shape as the payload + spec test fixtures.
    struct ScriptedRuntime {
        responses: Mutex<Vec<Result<String, AiError>>>,
        prompts_seen: Mutex<Vec<String>>,
        user_messages_seen: Mutex<Vec<String>>,
        tracker: Arc<dyn BudgetTracker>,
        cost_per_call: i64,
    }

    impl ScriptedRuntime {
        fn new(
            responses: Vec<Result<String, AiError>>,
            tracker: Arc<dyn BudgetTracker>,
            cost_per_call: i64,
        ) -> Self {
            Self {
                responses: Mutex::new(responses),
                prompts_seen: Mutex::new(Vec::new()),
                user_messages_seen: Mutex::new(Vec::new()),
                tracker,
                cost_per_call,
            }
        }

        fn prompts(&self) -> Vec<String> {
            self.prompts_seen.lock().unwrap().clone()
        }

        fn user_messages(&self) -> Vec<String> {
            self.user_messages_seen.lock().unwrap().clone()
        }
    }

    struct ScriptedAgentRuntime {
        final_message: String,
        task_seen: Mutex<Option<AgentTask>>,
        tracker: Arc<dyn BudgetTracker>,
        cost_per_call: i64,
    }

    impl ScriptedAgentRuntime {
        fn new(final_message: String, tracker: Arc<dyn BudgetTracker>, cost_per_call: i64) -> Self {
            Self { final_message, task_seen: Mutex::new(None), tracker, cost_per_call }
        }

        fn task(&self) -> AgentTask {
            self.task_seen.lock().unwrap().clone().expect("agent task captured")
        }
    }

    #[async_trait]
    impl AiRuntime for ScriptedRuntime {
        fn name(&self) -> &'static str {
            "scripted"
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
            _sink: EventSink,
        ) -> Result<Response, AiError> {
            self.prompts_seen.lock().unwrap().push(prompt.system.clone());
            self.user_messages_seen.lock().unwrap().push(prompt.user.clone());
            let next =
                self.responses.lock().unwrap().pop().expect("scripted runtime: no more responses");
            let content = next?;
            let cost = self.cost_per_call;
            self.tracker.add_spend(&budget.run_id, budget.kind, cost).await?;
            Ok(Response {
                prompt_version: prompt.prompt_version,
                task_id: prompt.task_id,
                model: "scripted-model".to_string(),
                content,
                usage: TokenUsage { input_tokens: 400, output_tokens: 200 },
                cache: Some(CacheStats::default()),
                cost_usd_micros: cost,
            })
        }

        async fn agent_loop(
            &self,
            _task: AgentTask,
            _budget: Budget,
            _sink: EventSink,
        ) -> Result<AgentResult, AiError> {
            Err(AiError::UnsupportedMode("agent_loop"))
        }

        fn cost_estimate(&self, _prompt: &Prompt) -> Option<CostEstimate> {
            Some(CostEstimate { min_usd_micros: 0, max_usd_micros: self.cost_per_call })
        }
    }

    #[async_trait]
    impl AiRuntime for ScriptedAgentRuntime {
        fn name(&self) -> &'static str {
            "scripted-agent"
        }
        fn default_model(&self) -> &str {
            "scripted-agent-model"
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
            _sink: EventSink,
        ) -> Result<Response, AiError> {
            Err(AiError::UnsupportedMode("one_shot"))
        }

        async fn agent_loop(
            &self,
            task: AgentTask,
            budget: Budget,
            _sink: EventSink,
        ) -> Result<AgentResult, AiError> {
            *self.task_seen.lock().unwrap() = Some(task.clone());
            self.tracker.add_spend(&budget.run_id, budget.kind, self.cost_per_call).await?;
            Ok(AgentResult {
                prompt_version: task.prompt_version,
                task_id: task.task_id,
                model: "scripted-agent-model".to_string(),
                final_message: self.final_message.clone(),
                turns: 3,
                usage: TokenUsage { input_tokens: 800, output_tokens: 120 },
                cache: Some(CacheStats::default()),
                cost_usd_micros: self.cost_per_call,
                extracted: Vec::new(),
            })
        }

        fn cost_estimate(&self, _prompt: &Prompt) -> Option<CostEstimate> {
            Some(CostEstimate { min_usd_micros: 0, max_usd_micros: self.cost_per_call })
        }
    }

    fn two_repo_input() -> ChainReasoningInput {
        ChainReasoningInput {
            run_id: "run-1".to_string(),
            repos: vec!["repo-A".to_string(), "repo-B".to_string()],
            nodes: vec![
                ChainReasoningNode {
                    id: "a-entry".to_string(),
                    graph_kind: None,
                    label: None,
                    ref_id: None,
                    repo: "repo-A".to_string(),
                    path: "controller.py".to_string(),
                    line: Some(5),
                    cap: "SQL_QUERY".to_string(),
                    rule: "py.taint.flow".to_string(),
                    severity: "High".to_string(),
                    kind: NODE_KIND_ENTRY.to_string(),
                    routes: Vec::new(),
                    roles: Vec::new(),
                    objects: Vec::new(),
                    evidence_refs: Vec::new(),
                },
                ChainReasoningNode {
                    id: "b-sink".to_string(),
                    graph_kind: None,
                    label: None,
                    ref_id: None,
                    repo: "repo-B".to_string(),
                    path: "db.py".to_string(),
                    line: Some(42),
                    cap: "SQL_QUERY".to_string(),
                    rule: "py.sql.exec".to_string(),
                    severity: "Critical".to_string(),
                    kind: NODE_KIND_SINK.to_string(),
                    routes: Vec::new(),
                    roles: Vec::new(),
                    objects: Vec::new(),
                    evidence_refs: Vec::new(),
                },
            ],
            edges: vec![ChainReasoningEdge {
                from: "a-entry".to_string(),
                to: "b-sink".to_string(),
                label: "Reaches".to_string(),
                cross_repo: true,
                edge_id: None,
                evidence_ref: None,
                source: None,
            }],
            max_chains: 10,
        }
    }

    fn ok_body(member_ids: &[&str], rationale: &str) -> String {
        serde_json::json!({
            "chains": [
                {
                    "member_ids": member_ids,
                    "rationale": rationale,
                }
            ]
        })
        .to_string()
    }

    #[tokio::test]
    async fn two_repo_input_produces_cross_repo_chain() {
        // Acceptance: a controller-in-repo-A reaches-sink-in-repo-B
        // fixture produces at least one chain whose members span both
        // repos.
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-1", BudgetKind::OneShot, 5_000_000);
        let rt = ScriptedRuntime::new(
            vec![Ok(ok_body(
                &["a-entry", "b-sink"],
                "controller in repo-A reaches SQL sink in repo-B via shared dispatch",
            ))],
            tracker.clone(),
            7_500,
        );

        let (tx, _rx) = broadcast::channel::<AgentEvent>(16);
        let outcome = run(&rt, &two_repo_input(), &[], tx, 5_000_000).await.expect("ok");
        match outcome {
            ChainReasoningOutcome::Ranked {
                run_id,
                output,
                prompt_version,
                spent_usd_micros,
                attempts,
                metrics,
            } => {
                assert_eq!(run_id, "run-1");
                assert_eq!(prompt_version, CHAIN_REASONING_PROMPT_VERSION);
                assert_eq!(spent_usd_micros, 7_500);
                assert_eq!(attempts, 1);
                assert_eq!(output.chains.len(), 1);
                let c = &output.chains[0];
                assert_eq!(c.member_ids, vec!["a-entry".to_string(), "b-sink".to_string()]);
                assert!(c.rationale.contains("repo-A"));
                assert!(c.rationale.contains("repo-B"));
                assert_eq!(metrics.usage.input_tokens, 400);
                assert_eq!(metrics.usage.output_tokens, 200);
                assert_eq!(metrics.model.as_deref(), Some("scripted-model"));
            }
            other => panic!("expected Ranked, got {other:?}"),
        }
        assert_eq!(tracker.spent("run-1", BudgetKind::OneShot), 7_500);

        // Cross-repo edge surfaces in the rendered user message so the
        // model can reason about it.
        let user = rt.user_messages().into_iter().next().expect("user msg");
        assert!(user.contains("cross_repo"), "user message must surface cross_repo edges: {user}");
        assert!(user.contains("repo-A"));
        assert!(user.contains("repo-B"));
    }

    #[tokio::test]
    async fn agentic_chain_reasoning_gets_workspace_roots_and_tools() {
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-1", BudgetKind::AgentLoop, 1_000_000);
        let body = serde_json::json!({
            "chains": [{
                "member_ids": ["a-entry", "b-sink"],
                "rationale": "reading controller.py and db.py showed the entry flows to the sink",
                "prerequisites": ["authenticated user"],
                "evidence": ["repo-A/controller.py:5", "repo-B/db.py:42"],
                "blast_radius": ["repo-B SQL sink"],
                "confidence": 88,
                "missing_verification_steps": [],
                "edge_provenance": ["a-entry->b-sink"]
            }]
        })
        .to_string();
        let rt = ScriptedAgentRuntime::new(body, tracker, 2_000);
        let (tx, _rx) = broadcast::channel::<AgentEvent>(4);
        let workspaces = vec![
            ChainReasoningWorkspace { repo: "repo-A".to_string(), root: "/tmp/repo-A".to_string() },
            ChainReasoningWorkspace { repo: "repo-B".to_string(), root: "/tmp/repo-B".to_string() },
        ];

        let out = run_agentic(&rt, &two_repo_input(), &[], &workspaces, tx, 1_000_000)
            .await
            .expect("agentic run ok");

        let ChainReasoningOutcome::Ranked { output, attempts, spent_usd_micros, .. } = out else {
            panic!("expected ranked output");
        };
        assert_eq!(attempts, 1);
        assert_eq!(spent_usd_micros, 2_000);
        assert_eq!(output.chains[0].member_ids, vec!["a-entry", "b-sink"]);

        let task = rt.task();
        assert!(task.tools.contains(&"Read".to_string()));
        assert!(task.tools.contains(&"Grep".to_string()));
        assert!(task.tools.contains(&"Bash".to_string()));
        assert_eq!(task.working_directory.as_deref(), Some("/tmp/repo-A"));
        assert!(task.objective.contains("repo=repo-A root=/tmp/repo-A"));
        assert!(task.objective.contains("repo=repo-B root=/tmp/repo-B"));
    }

    #[tokio::test]
    async fn malformed_first_attempt_retries_with_stricter_prompt() {
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-1", BudgetKind::OneShot, 5_000_000);
        let good = ok_body(&["a-entry", "b-sink"], "ok rationale");
        // Queue popped from back: first call -> garbage, second -> good.
        let rt = ScriptedRuntime::new(
            vec![Ok(good), Ok("not json at all".to_string())],
            tracker.clone(),
            2_000,
        );

        let (tx, _rx) = broadcast::channel::<AgentEvent>(16);
        let outcome = run(&rt, &two_repo_input(), &[], tx, 5_000_000).await.expect("ok");
        match outcome {
            ChainReasoningOutcome::Ranked { attempts, spent_usd_micros, .. } => {
                assert_eq!(attempts, 2);
                assert_eq!(spent_usd_micros, 4_000);
            }
            other => panic!("expected Ranked after retry, got {other:?}"),
        }
        let seen = rt.prompts();
        assert_eq!(seen.len(), 2);
        assert!(seen[0].contains("Reply with exactly one JSON object"));
        assert!(seen[1].contains("previous reply did not validate"));
    }

    #[tokio::test]
    async fn two_malformed_attempts_yield_no_chains() {
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-1", BudgetKind::OneShot, 5_000_000);
        let rt = ScriptedRuntime::new(
            vec![Ok("still nope".to_string()), Ok("nope".to_string())],
            tracker.clone(),
            1_000,
        );
        let (tx, _rx) = broadcast::channel::<AgentEvent>(8);
        let outcome = run(&rt, &two_repo_input(), &[], tx, 5_000_000).await.expect("ok");
        match outcome {
            ChainReasoningOutcome::NoChains {
                run_id,
                reason,
                spent_usd_micros,
                attempts,
                metrics,
            } => {
                assert_eq!(run_id, "run-1");
                assert_eq!(attempts, 2);
                assert_eq!(spent_usd_micros, 2_000);
                assert!(reason.contains("failed twice"), "reason: {reason}");
                assert_eq!(metrics.usage.input_tokens, 800);
                assert_eq!(metrics.usage.output_tokens, 400);
            }
            other => panic!("expected NoChains, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_chains_array_is_rejected() {
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-1", BudgetKind::OneShot, 5_000_000);
        let good = ok_body(&["a-entry", "b-sink"], "ok");
        let empty = serde_json::json!({"chains": []}).to_string();
        // Retry path: first response empty, second good.
        let rt = ScriptedRuntime::new(vec![Ok(good), Ok(empty)], tracker.clone(), 1_000);
        let (tx, _rx) = broadcast::channel::<AgentEvent>(8);
        let outcome = run(&rt, &two_repo_input(), &[], tx, 5_000_000).await.expect("ok");
        assert!(matches!(outcome, ChainReasoningOutcome::Ranked { attempts: 2, .. }));
    }

    #[tokio::test]
    async fn unknown_member_id_is_rejected() {
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-1", BudgetKind::OneShot, 5_000_000);
        let bad = ok_body(&["a-entry", "ghost"], "made up node");
        let good = ok_body(&["a-entry", "b-sink"], "real chain");
        // First reply references a node that does not exist; retry
        // returns a valid chain.
        let rt = ScriptedRuntime::new(vec![Ok(good), Ok(bad)], tracker.clone(), 800);
        let (tx, _rx) = broadcast::channel::<AgentEvent>(8);
        let outcome = run(&rt, &two_repo_input(), &[], tx, 5_000_000).await.expect("ok");
        assert!(matches!(outcome, ChainReasoningOutcome::Ranked { attempts: 2, .. }));
    }

    #[tokio::test]
    async fn single_member_chain_is_rejected() {
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-1", BudgetKind::OneShot, 5_000_000);
        let bad = ok_body(&["a-entry"], "single step");
        let good = ok_body(&["a-entry", "b-sink"], "two step");
        let rt = ScriptedRuntime::new(vec![Ok(good), Ok(bad)], tracker.clone(), 800);
        let (tx, _rx) = broadcast::channel::<AgentEvent>(8);
        let outcome = run(&rt, &two_repo_input(), &[], tx, 5_000_000).await.expect("ok");
        assert!(matches!(outcome, ChainReasoningOutcome::Ranked { attempts: 2, .. }));
    }

    #[tokio::test]
    async fn consecutive_duplicate_member_ids_are_rejected() {
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-1", BudgetKind::OneShot, 5_000_000);
        // A model that copies the entry id N times produces a "1-step
        // loop"; the validator must reject the chain so the retry path
        // gets a chance to produce something analytic.
        let bad = ok_body(&["a-entry", "a-entry", "b-sink"], "stuttered chain");
        let good = ok_body(&["a-entry", "b-sink"], "clean chain");
        let rt = ScriptedRuntime::new(vec![Ok(good), Ok(bad)], tracker.clone(), 600);
        let (tx, _rx) = broadcast::channel::<AgentEvent>(8);
        let outcome = run(&rt, &two_repo_input(), &[], tx, 5_000_000).await.expect("ok");
        assert!(matches!(outcome, ChainReasoningOutcome::Ranked { attempts: 2, .. }));
    }

    #[tokio::test]
    async fn code_fence_wrapper_is_tolerated() {
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-1", BudgetKind::OneShot, 5_000_000);
        let inner = ok_body(&["a-entry", "b-sink"], "ok");
        let wrapped = format!("```json\n{inner}\n```");
        let rt = ScriptedRuntime::new(vec![Ok(wrapped)], tracker.clone(), 500);
        let (tx, _rx) = broadcast::channel::<AgentEvent>(8);
        let outcome = run(&rt, &two_repo_input(), &[], tx, 5_000_000).await.expect("ok");
        assert!(matches!(outcome, ChainReasoningOutcome::Ranked { attempts: 1, .. }));
    }

    #[tokio::test]
    async fn upstream_error_surfaces_through() {
        let tracker = Arc::new(InMemoryBudgetTracker::new());
        tracker.set_cap("run-1", BudgetKind::OneShot, 5_000_000);
        let rt = ScriptedRuntime::new(
            vec![Err(AiError::UpstreamRefused("429 rate limit".to_string()))],
            tracker.clone(),
            1_000,
        );
        let (tx, _rx) = broadcast::channel::<AgentEvent>(8);
        let err = run(&rt, &two_repo_input(), &[], tx, 5_000_000).await.expect_err("upstream");
        assert!(matches!(err, AiError::UpstreamRefused(_)));
    }
}
