//! Deterministic candidate-path extraction for ChainReasoning.
//!
//! The run's finding graph can be enormous (thousands of nodes, tens of
//! thousands of edges). Serialising it whole into a single prompt blows
//! the model's context window — the call is refused before any
//! reasoning happens — and even when it fits, an LLM cannot reliably
//! traverse a giant adjacency list in its head, so it invents paths the
//! validation gate then rejects.
//!
//! This module moves the graph traversal where it belongs: a bounded,
//! deterministic engine enumerates the highest-value entry->impact
//! paths and hands the model
//!
//! 1. a small, high-signal **subgraph** (only the nodes/edges on those
//!    paths), and
//! 2. the enumerated paths themselves as **seed chains** the model is
//!    asked to verify (with code reading), extend, and rank.
//!
//! The model spends its tokens on what it is good at — judging
//! exploitability and reading code — instead of parsing an adjacency
//! list it will only hallucinate over. The reduction is pure and
//! testable; it operates entirely on `nyx-agent-types` shapes so the
//! task crate stays vendor-neutral.

use std::collections::{HashMap, HashSet};

use nyx_agent_types::chain::{ChainReasoningEdge, ChainReasoningInput, ChainReasoningNode};

/// Native graph kinds (mirrors the store's `attack_graph` node kinds).
/// Only the handful the ranking heuristics care about are named.
const GK_VERIFIED_VULN: &str = "verified_vulnerability";
const GK_VERIFICATION_ATTEMPT: &str = "verification_attempt";
const GK_CANDIDATE: &str = "candidate";
const GK_SIGNAL: &str = "signal";
const GK_ROUTE: &str = "route";
const GK_ENDPOINT: &str = "endpoint";
const GK_FORM: &str = "form";

/// Coarse role tags carried on `ChainReasoningNode::kind`.
const KIND_ENTRY: &str = "entry";
const KIND_SINK: &str = "sink";

/// Depth bound on enumerated paths (number of nodes). Real exploit
/// chains are short; the bound is the primary guard against exponential
/// blowup on a densely connected graph.
const MAX_DEPTH: usize = 8;
/// Cap on how many terminal nodes we expand backward from.
const TOP_TERMINALS: usize = 96;
/// Cap on distinct paths recorded per terminal.
const MAX_PATHS_PER_TERMINAL: usize = 6;
/// Global ceiling on reverse-edge expansions across the whole
/// enumeration. Keeps the pass fast even on a 30k-edge graph.
const GLOBAL_STEP_BUDGET: usize = 300_000;
/// Hard ceiling on the number of seed chains surfaced to the model.
const MAX_SEEDS: usize = 48;

/// Token budget for the graph portion of the prompt, expressed as node
/// and edge ceilings. The reduction only prunes when the input exceeds
/// one of these; small graphs pass through untouched (but still get
/// seed chains).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphBudget {
    pub max_nodes: usize,
    pub max_edges: usize,
}

impl Default for GraphBudget {
    fn default() -> Self {
        Self { max_nodes: 300, max_edges: 1200 }
    }
}

impl GraphBudget {
    /// Scale the budget to the configured model context window (in
    /// tokens). Roughly a third of the window is reserved for the graph
    /// blob; the rest is left for the system prompt, the seed section,
    /// agent tool traffic, and the model's reply. Falls back to
    /// [`GraphBudget::default`] when no context window is configured.
    pub fn for_context_window(context_window_tokens: Option<usize>) -> Self {
        match context_window_tokens {
            Some(tokens) if tokens > 0 => {
                let graph_tokens = tokens / 3;
                // ~60 tokens per serialised node line.
                let max_nodes = (graph_tokens / 60).clamp(120, 800);
                let max_edges = (max_nodes * 3).clamp(360, 2400);
                Self { max_nodes, max_edges }
            }
            _ => Self::default(),
        }
    }
}

/// Result of reducing a `ChainReasoningInput` for prompt dispatch.
#[derive(Debug, Clone, PartialEq)]
pub struct ReducedGraph {
    /// The graph to actually send to the model. Identical to the input
    /// when it already fit the budget; a pruned subgraph otherwise.
    pub input: ChainReasoningInput,
    /// Pre-extracted entry->impact paths, each a list of node ids in
    /// entry-to-sink order. Every adjacent pair is backed by an edge in
    /// `input.edges`, so a seed is itself a valid chain skeleton.
    pub seeds: Vec<Vec<String>>,
    /// Nodes dropped from the original input by the budget prune.
    pub dropped_nodes: usize,
    /// Edges dropped from the original input by the budget prune.
    pub dropped_edges: usize,
    /// True when no entry->impact path was found and the reduction fell
    /// back to a severity-ranked node truncation.
    pub fell_back: bool,
}

/// Reduce `input` to a budget-bounded subgraph plus seed chains.
///
/// Always enumerates seed paths (bounded work, cheap on small graphs);
/// only prunes the node/edge set when the input exceeds `budget`.
pub fn reduce(input: &ChainReasoningInput, budget: GraphBudget) -> ReducedGraph {
    let node_by_id: HashMap<&str, &ChainReasoningNode> =
        input.nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    // Forward / reverse adjacency + degree counts over the input edges.
    let mut rev: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut in_deg: HashMap<&str, usize> = HashMap::new();
    let mut out_deg: HashMap<&str, usize> = HashMap::new();
    for e in &input.edges {
        if !node_by_id.contains_key(e.from.as_str()) || !node_by_id.contains_key(e.to.as_str()) {
            continue;
        }
        rev.entry(e.to.as_str()).or_default().push(e.from.as_str());
        *out_deg.entry(e.from.as_str()).or_default() += 1;
        *in_deg.entry(e.to.as_str()).or_default() += 1;
    }

    let is_entry = |n: &ChainReasoningNode| -> bool {
        n.kind == KIND_ENTRY
            || in_deg.get(n.id.as_str()).copied().unwrap_or(0) == 0
            || matches!(
                n.graph_kind.as_deref(),
                Some(GK_CANDIDATE | GK_SIGNAL | GK_ROUTE | GK_ENDPOINT | GK_FORM)
            )
    };
    let is_terminal = |n: &ChainReasoningNode| -> bool {
        n.kind == KIND_SINK
            || matches!(n.graph_kind.as_deref(), Some(GK_VERIFIED_VULN | GK_VERIFICATION_ATTEMPT))
            || (out_deg.get(n.id.as_str()).copied().unwrap_or(0) == 0
                && severity_rank(&n.severity) >= 2)
    };

    // Rank terminals by value and expand the strongest ones backward.
    let mut terminals: Vec<&ChainReasoningNode> =
        input.nodes.iter().filter(|n| is_terminal(n)).collect();
    terminals
        .sort_by(|a, b| terminal_value(b).cmp(&terminal_value(a)).then_with(|| a.id.cmp(&b.id)));
    terminals.truncate(TOP_TERMINALS);

    // Backward bounded DFS from each terminal to any entry, recording
    // simple (acyclic) paths. `steps` is the shared global budget.
    let mut steps = 0usize;
    let mut raw_paths: Vec<Vec<String>> = Vec::new();
    let mut seen_paths: HashSet<String> = HashSet::new();
    for term in &terminals {
        if steps >= GLOBAL_STEP_BUDGET {
            break;
        }
        let mut found_here = 0usize;
        let mut stack: Vec<Vec<&str>> = vec![vec![term.id.as_str()]];
        while let Some(path) = stack.pop() {
            if found_here >= MAX_PATHS_PER_TERMINAL || steps >= GLOBAL_STEP_BUDGET {
                break;
            }
            let tail = *path.last().unwrap();
            let tail_node = node_by_id[tail];
            // A path of >=2 nodes whose tail (the upstream end) is an
            // entry is a complete entry->impact candidate.
            if path.len() >= 2 && is_entry(tail_node) {
                let ordered: Vec<String> = path.iter().rev().map(|s| s.to_string()).collect();
                let key = ordered.join(">");
                if seen_paths.insert(key) {
                    raw_paths.push(ordered);
                    found_here += 1;
                    if found_here >= MAX_PATHS_PER_TERMINAL {
                        break;
                    }
                }
                // Keep exploring past an entry too — a longer chain
                // through an even earlier entry can outrank it.
            }
            if path.len() >= MAX_DEPTH {
                continue;
            }
            if let Some(preds) = rev.get(tail) {
                // Deterministic expansion order.
                let mut preds_sorted: Vec<&str> = preds.clone();
                preds_sorted.sort_unstable();
                preds_sorted.dedup();
                for &p in &preds_sorted {
                    steps += 1;
                    if path.contains(&p) {
                        continue; // simple paths only
                    }
                    let mut next = path.clone();
                    next.push(p);
                    stack.push(next);
                }
            }
        }
    }

    // Score and order the candidate paths, strongest first.
    let cross_repo_pairs = input.edges_cross_repo_set();
    raw_paths.sort_by(|a, b| {
        path_score(b, &node_by_id, &cross_repo_pairs)
            .cmp(&path_score(a, &node_by_id, &cross_repo_pairs))
            .then_with(|| a.len().cmp(&b.len()))
            .then_with(|| a.join(">").cmp(&b.join(">")))
    });

    // Greedily keep paths whose union of nodes stays within budget.
    let mut kept_nodes: HashSet<String> = HashSet::new();
    let mut seeds: Vec<Vec<String>> = Vec::new();
    let node_cap = budget.max_nodes;
    for path in &raw_paths {
        if seeds.len() >= MAX_SEEDS {
            break;
        }
        let mut union = kept_nodes.clone();
        for id in path {
            union.insert(id.clone());
        }
        if union.len() > node_cap && !kept_nodes.is_empty() {
            continue; // adding this path would overflow; try the next
        }
        kept_nodes = union;
        seeds.push(path.clone());
    }

    let within_budget =
        input.nodes.len() <= budget.max_nodes && input.edges.len() <= budget.max_edges;
    if within_budget {
        // Small graph: keep it whole, but still surface seed chains.
        return ReducedGraph {
            input: input.clone(),
            seeds,
            dropped_nodes: 0,
            dropped_edges: 0,
            fell_back: false,
        };
    }

    // Over budget: build the pruned subgraph from the kept paths.
    if kept_nodes.len() >= 2 {
        let reduced = build_subgraph(input, &kept_nodes, &seeds, budget);
        let dropped_nodes = input.nodes.len().saturating_sub(reduced.nodes.len());
        let dropped_edges = input.edges.len().saturating_sub(reduced.edges.len());
        return ReducedGraph {
            input: reduced,
            seeds,
            dropped_nodes,
            dropped_edges,
            fell_back: false,
        };
    }

    // Fallback: no usable paths. Keep the highest-value nodes and the
    // edges among them so the model at least sees the strongest leads.
    let reduced = fallback_subgraph(input, budget);
    let dropped_nodes = input.nodes.len().saturating_sub(reduced.nodes.len());
    let dropped_edges = input.edges.len().saturating_sub(reduced.edges.len());
    ReducedGraph {
        input: reduced,
        seeds: Vec::new(),
        dropped_nodes,
        dropped_edges,
        fell_back: true,
    }
}

/// Build the pruned subgraph: the kept nodes, plus every input edge
/// whose endpoints are both kept. Edges that lie on a seed path are
/// always retained (so seeds stay valid); remaining edges fill up to
/// `budget.max_edges`.
fn build_subgraph(
    input: &ChainReasoningInput,
    kept_nodes: &HashSet<String>,
    seeds: &[Vec<String>],
    budget: GraphBudget,
) -> ChainReasoningInput {
    let nodes: Vec<ChainReasoningNode> =
        input.nodes.iter().filter(|n| kept_nodes.contains(&n.id)).cloned().collect();

    // Seed-supporting pairs must never be dropped.
    let mut seed_pairs: HashSet<(String, String)> = HashSet::new();
    for path in seeds {
        for w in path.windows(2) {
            seed_pairs.insert((w[0].clone(), w[1].clone()));
        }
    }

    let mut on_seed: Vec<ChainReasoningEdge> = Vec::new();
    let mut other: Vec<ChainReasoningEdge> = Vec::new();
    for e in &input.edges {
        if !kept_nodes.contains(&e.from) || !kept_nodes.contains(&e.to) {
            continue;
        }
        if seed_pairs.contains(&(e.from.clone(), e.to.clone())) {
            on_seed.push(e.clone());
        } else {
            other.push(e.clone());
        }
    }
    let sort_edges = |v: &mut Vec<ChainReasoningEdge>| {
        v.sort_by(|a, b| (&a.from, &a.to, &a.label).cmp(&(&b.from, &b.to, &b.label)));
    };
    sort_edges(&mut on_seed);
    sort_edges(&mut other);

    // Always keep seed edges; fill the rest up to the budget.
    let remaining = budget.max_edges.saturating_sub(on_seed.len());
    other.truncate(remaining);
    let mut edges = on_seed;
    edges.append(&mut other);
    sort_edges(&mut edges);

    let mut repos: Vec<String> = nodes.iter().map(|n| n.repo.clone()).collect();
    repos.sort();
    repos.dedup();

    ChainReasoningInput {
        run_id: input.run_id.clone(),
        repos,
        nodes,
        edges,
        max_chains: input.max_chains,
    }
}

/// No path was found. Rank nodes by standalone value and keep the top
/// `budget.max_nodes`, plus edges among them up to `budget.max_edges`.
fn fallback_subgraph(input: &ChainReasoningInput, budget: GraphBudget) -> ChainReasoningInput {
    let mut ranked: Vec<&ChainReasoningNode> = input.nodes.iter().collect();
    ranked.sort_by(|a, b| node_value(b).cmp(&node_value(a)).then_with(|| a.id.cmp(&b.id)));
    let kept: HashSet<String> =
        ranked.iter().take(budget.max_nodes).map(|n| n.id.clone()).collect();
    build_subgraph(input, &kept, &[], budget)
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

/// Standalone value of a terminal node (drives expansion order).
fn terminal_value(n: &ChainReasoningNode) -> u32 {
    let mut v = u32::from(severity_rank(&n.severity)) * 10;
    match n.graph_kind.as_deref() {
        Some(GK_VERIFIED_VULN) => v += 100,
        Some(GK_VERIFICATION_ATTEMPT) => v += 60,
        _ => {}
    }
    if n.kind == KIND_SINK {
        v += 20;
    }
    v
}

/// Standalone value of any node (drives fallback truncation).
fn node_value(n: &ChainReasoningNode) -> u32 {
    let mut v = u32::from(severity_rank(&n.severity)) * 10;
    match n.graph_kind.as_deref() {
        Some(GK_VERIFIED_VULN) => v += 100,
        Some(GK_VERIFICATION_ATTEMPT) => v += 60,
        Some(GK_CANDIDATE | GK_SIGNAL) => v += 8,
        _ => {}
    }
    match n.kind.as_str() {
        KIND_SINK => v += 20,
        KIND_ENTRY => v += 5,
        _ => {}
    }
    v
}

/// Exploitability score for a complete entry->impact path.
fn path_score(
    path: &[String],
    node_by_id: &HashMap<&str, &ChainReasoningNode>,
    cross_repo_pairs: &HashSet<(String, String)>,
) -> u32 {
    let Some(terminal) = path.last().and_then(|id| node_by_id.get(id.as_str())) else {
        return 0;
    };
    let entry = path.first().and_then(|id| node_by_id.get(id.as_str()));
    let mut score = terminal_value(terminal);
    if let Some(e) = entry {
        score += u32::from(severity_rank(&e.severity)) * 2;
    }
    // Cross-repo / cross-service chains are the highest-value finds.
    let crosses = path.windows(2).any(|w| cross_repo_pairs.contains(&(w[0].clone(), w[1].clone())))
        || path.windows(2).any(|w| {
            match (node_by_id.get(w[0].as_str()), node_by_id.get(w[1].as_str())) {
                (Some(a), Some(b)) => a.repo != b.repo,
                _ => false,
            }
        });
    if crosses {
        score += 40;
    }
    // Mild preference for shorter chains among equals.
    score = score.saturating_sub(3 * (path.len().saturating_sub(2)) as u32);
    score
}

/// Helper extension: the set of `(from, to)` pairs flagged `cross_repo`.
trait CrossRepoEdges {
    fn edges_cross_repo_set(&self) -> HashSet<(String, String)>;
}
impl CrossRepoEdges for ChainReasoningInput {
    fn edges_cross_repo_set(&self) -> HashSet<(String, String)> {
        self.edges.iter().filter(|e| e.cross_repo).map(|e| (e.from.clone(), e.to.clone())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, repo: &str, kind: &str, gk: Option<&str>, sev: &str) -> ChainReasoningNode {
        ChainReasoningNode {
            id: id.to_string(),
            graph_kind: gk.map(str::to_string),
            label: Some(id.to_string()),
            ref_id: None,
            repo: repo.to_string(),
            path: format!("{id}.py"),
            line: Some(1),
            cap: "CAP".to_string(),
            rule: "rule".to_string(),
            severity: sev.to_string(),
            kind: kind.to_string(),
            routes: Vec::new(),
            roles: Vec::new(),
            objects: Vec::new(),
            evidence_refs: Vec::new(),
        }
    }

    fn edge(from: &str, to: &str, cross: bool) -> ChainReasoningEdge {
        ChainReasoningEdge {
            from: from.to_string(),
            to: to.to_string(),
            label: "Reaches".to_string(),
            cross_repo: cross,
            edge_id: Some(format!("{from}->{to}")),
            evidence_ref: None,
            source: None,
        }
    }

    fn input(
        nodes: Vec<ChainReasoningNode>,
        edges: Vec<ChainReasoningEdge>,
    ) -> ChainReasoningInput {
        ChainReasoningInput {
            run_id: "run-1".to_string(),
            repos: vec!["A".to_string(), "B".to_string()],
            nodes,
            edges,
            max_chains: 10,
        }
    }

    #[test]
    fn small_graph_passes_through_with_seed() {
        let inp = input(
            vec![
                node("a", "A", "entry", Some(GK_ROUTE), "High"),
                node("b", "B", "sink", Some(GK_VERIFIED_VULN), "Critical"),
            ],
            vec![edge("a", "b", true)],
        );
        let out = reduce(&inp, GraphBudget::default());
        assert!(!out.fell_back);
        assert_eq!(out.dropped_nodes, 0);
        assert_eq!(out.dropped_edges, 0);
        assert_eq!(out.input.nodes.len(), 2);
        assert_eq!(out.seeds, vec![vec!["a".to_string(), "b".to_string()]]);
    }

    #[test]
    fn seeds_are_entry_to_sink_ordered_and_edge_backed() {
        // a -> m -> b ; entry=a, sink=b
        let inp = input(
            vec![
                node("a", "A", "entry", Some(GK_CANDIDATE), "Medium"),
                node("m", "A", "other", None, "Low"),
                node("b", "B", "sink", Some(GK_VERIFIED_VULN), "Critical"),
            ],
            vec![edge("a", "m", false), edge("m", "b", true)],
        );
        let out = reduce(&inp, GraphBudget::default());
        assert_eq!(out.seeds.len(), 1);
        assert_eq!(out.seeds[0], vec!["a", "m", "b"]);
    }

    #[test]
    fn over_budget_prunes_to_path_subgraph() {
        // One real chain a->b plus 50 isolated noise nodes. A tiny budget
        // must drop the noise and keep the chain + its edge.
        let mut nodes = vec![
            node("a", "A", "entry", Some(GK_ROUTE), "High"),
            node("b", "B", "sink", Some(GK_VERIFIED_VULN), "Critical"),
        ];
        for i in 0..50 {
            nodes.push(node(&format!("noise{i}"), "A", "other", None, "Info"));
        }
        let inp = input(nodes, vec![edge("a", "b", true)]);
        let budget = GraphBudget { max_nodes: 5, max_edges: 5 };
        let out = reduce(&inp, budget);
        assert!(!out.fell_back);
        assert!(out.dropped_nodes >= 48, "expected noise dropped, got {}", out.dropped_nodes);
        let ids: HashSet<&str> = out.input.nodes.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains("a") && ids.contains("b"));
        assert_eq!(out.seeds[0], vec!["a", "b"]);
        // Seed edge survived the prune.
        assert!(out.input.edges.iter().any(|e| e.from == "a" && e.to == "b"));
    }

    #[test]
    fn no_path_falls_back_to_ranked_truncation() {
        // All isolated nodes, no edges -> no path possible.
        let nodes = (0..20)
            .map(|i| {
                let sev = if i == 0 { "Critical" } else { "Info" };
                node(&format!("n{i}"), "A", "other", None, sev)
            })
            .collect();
        let inp = input(nodes, vec![]);
        let budget = GraphBudget { max_nodes: 3, max_edges: 3 };
        let out = reduce(&inp, budget);
        assert!(out.fell_back);
        assert!(out.seeds.is_empty());
        assert!(out.input.nodes.len() <= 3);
        // Highest-severity node is retained.
        assert!(out.input.nodes.iter().any(|n| n.id == "n0"));
    }

    #[test]
    fn cross_repo_path_outranks_same_repo_path() {
        // Two terminals: cross-repo critical vs same-repo critical.
        let inp = input(
            vec![
                node("e1", "A", "entry", Some(GK_ROUTE), "High"),
                node("x", "B", "sink", Some(GK_VERIFIED_VULN), "Critical"),
                node("e2", "A", "entry", Some(GK_ROUTE), "High"),
                node("y", "A", "sink", Some(GK_VERIFIED_VULN), "Critical"),
            ],
            vec![edge("e1", "x", true), edge("e2", "y", false)],
        );
        let out = reduce(&inp, GraphBudget::default());
        // Cross-repo seed ranked first.
        assert_eq!(out.seeds[0], vec!["e1", "x"]);
    }

    #[test]
    fn dense_graph_stays_bounded() {
        // Fully meshed 60-node graph: enumeration must terminate via the
        // step/depth budgets without exploding.
        let nodes: Vec<_> = (0..60)
            .map(|i| {
                let kind = if i == 0 {
                    "entry"
                } else if i == 59 {
                    "sink"
                } else {
                    "other"
                };
                node(&format!("n{i}"), "A", kind, None, "Medium")
            })
            .collect();
        let mut edges = Vec::new();
        for i in 0..60 {
            for j in 0..60 {
                if i != j {
                    edges.push(edge(&format!("n{i}"), &format!("n{j}"), false));
                }
            }
        }
        let inp = input(nodes, edges);
        let out = reduce(&inp, GraphBudget { max_nodes: 20, max_edges: 40 });
        assert!(out.input.nodes.len() <= 20);
        assert!(out.input.edges.len() <= 40);
        assert!(!out.seeds.is_empty());
    }
}
