//! `agent_traces` table - per-AI-task observability rows.

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use crate::store::StoreError;

pub use nyx_agent_types::trace::AgentTraceRecord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskKind {
    PayloadSynthesis,
    SpecDerivation,
    LiveTestPlan,
    ChainReasoning,
    NovelFindings,
    Exploration,
    AttackPlanning,
    /// Unsafe local dev-app attacker loop. This is a pre-MVP final
    /// pentest phase that records directly observed exploit proof.
    AttackAgent,
    /// Binary / CLI target pentest loop. Drives an operator-pinned local
    /// executable against agent-crafted malformed inputs inside the
    /// sandbox, recording sandbox-reproduced crashes / anomalies.
    BinaryTarget,
    /// AI critique pass that reviews a live candidate verification
    /// attempt before the product pipeline creates a user-facing
    /// verified vulnerability.
    LiveEvidenceReview,
    /// Deterministic payload-runner verifier call. Inputs are a
    /// `(harness_spec, payload)` pair already persisted by upstream
    /// tasks; the trace row points back to those rows via
    /// `prompt_version` (copied from the source payload).
    Verifier,
}

impl TaskKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskKind::PayloadSynthesis => "PayloadSynthesis",
            TaskKind::SpecDerivation => "SpecDerivation",
            TaskKind::LiveTestPlan => "LiveTestPlan",
            TaskKind::ChainReasoning => "ChainReasoning",
            TaskKind::NovelFindings => "NovelFindings",
            TaskKind::Exploration => "Exploration",
            TaskKind::AttackPlanning => "AttackPlanning",
            TaskKind::AttackAgent => "AttackAgent",
            TaskKind::BinaryTarget => "BinaryTarget",
            TaskKind::LiveEvidenceReview => "LiveEvidenceReview",
            TaskKind::Verifier => "Verifier",
        }
    }
}

pub struct AgentTraceStore<'a> {
    pool: &'a SqlitePool,
}

impl<'a> AgentTraceStore<'a> {
    pub fn new(pool: &'a SqlitePool) -> Self {
        Self { pool }
    }

    pub async fn insert(&self, t: &AgentTraceRecord) -> Result<(), StoreError> {
        sqlx::query!(
            r#"
            INSERT INTO agent_traces (
                id, finding_id, task_kind, runtime_name, model, prompt_version,
                conversation_jsonl_path, tokens_in, tokens_out, cost_usd_micros,
                cache_hits, cache_misses, duration_ms, started_at, finished_at,
                verifier_blob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            t.id,
            t.finding_id,
            t.task_kind,
            t.runtime_name,
            t.model,
            t.prompt_version,
            t.conversation_jsonl_path,
            t.tokens_in,
            t.tokens_out,
            t.cost_usd_micros,
            t.cache_hits,
            t.cache_misses,
            t.duration_ms,
            t.started_at,
            t.finished_at,
            t.verifier_blob,
        )
        .execute(self.pool)
        .await?;
        Ok(())
    }

    pub async fn get(&self, id: &str) -> Result<Option<AgentTraceRecord>, StoreError> {
        let row = sqlx::query_as!(
            AgentTraceRecord,
            r#"
            SELECT id AS "id!", finding_id,
                   task_kind AS "task_kind!",
                   runtime_name AS "runtime_name!",
                   model AS "model!", prompt_version, conversation_jsonl_path,
                   tokens_in       AS "tokens_in!: i64",
                   tokens_out      AS "tokens_out!: i64",
                   cost_usd_micros AS "cost_usd_micros!: i64",
                   cache_hits      AS "cache_hits!: i64",
                   cache_misses    AS "cache_misses!: i64",
                   duration_ms,
                   started_at      AS "started_at!: i64",
                   finished_at,
                   verifier_blob
            FROM agent_traces WHERE id = ?
            "#,
            id
        )
        .fetch_optional(self.pool)
        .await?;
        Ok(row)
    }

    pub async fn list_for_finding(
        &self,
        finding_id: &str,
    ) -> Result<Vec<AgentTraceRecord>, StoreError> {
        let rows = sqlx::query_as!(
            AgentTraceRecord,
            r#"
            SELECT id AS "id!", finding_id,
                   task_kind AS "task_kind!",
                   runtime_name AS "runtime_name!",
                   model AS "model!", prompt_version, conversation_jsonl_path,
                   tokens_in       AS "tokens_in!: i64",
                   tokens_out      AS "tokens_out!: i64",
                   cost_usd_micros AS "cost_usd_micros!: i64",
                   cache_hits      AS "cache_hits!: i64",
                   cache_misses    AS "cache_misses!: i64",
                   duration_ms,
                   started_at      AS "started_at!: i64",
                   finished_at,
                   verifier_blob
            FROM agent_traces WHERE finding_id = ? ORDER BY started_at
            "#,
            finding_id
        )
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

    /// Return every trace bound to the candidate finding by the
    /// `candidate_findings.trace_id` back-link. Empty when the candidate
    /// has no trace bound, such as candidates synthesised outside the AI
    /// exploration path.
    pub async fn list_for_candidate(
        &self,
        candidate_id: &str,
    ) -> Result<Vec<AgentTraceRecord>, StoreError> {
        let rows = sqlx::query_as!(
            AgentTraceRecord,
            r#"
            SELECT t.id AS "id!", t.finding_id,
                   t.task_kind AS "task_kind!",
                   t.runtime_name AS "runtime_name!",
                   t.model AS "model!", t.prompt_version, t.conversation_jsonl_path,
                   t.tokens_in       AS "tokens_in!: i64",
                   t.tokens_out      AS "tokens_out!: i64",
                   t.cost_usd_micros AS "cost_usd_micros!: i64",
                   t.cache_hits      AS "cache_hits!: i64",
                   t.cache_misses    AS "cache_misses!: i64",
                   t.duration_ms,
                   t.started_at      AS "started_at!: i64",
                   t.finished_at,
                   t.verifier_blob
            FROM agent_traces t
            JOIN candidate_findings c ON c.trace_id = t.id
            WHERE c.id = ?
            ORDER BY t.started_at
            "#,
            candidate_id
        )
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

    pub async fn list_by_task_kind(&self, kind: &str) -> Result<Vec<AgentTraceRecord>, StoreError> {
        let rows = sqlx::query_as!(
            AgentTraceRecord,
            r#"
            SELECT id AS "id!", finding_id,
                   task_kind AS "task_kind!",
                   runtime_name AS "runtime_name!",
                   model AS "model!", prompt_version, conversation_jsonl_path,
                   tokens_in       AS "tokens_in!: i64",
                   tokens_out      AS "tokens_out!: i64",
                   cost_usd_micros AS "cost_usd_micros!: i64",
                   cache_hits      AS "cache_hits!: i64",
                   cache_misses    AS "cache_misses!: i64",
                   duration_ms,
                   started_at      AS "started_at!: i64",
                   finished_at,
                   verifier_blob
            FROM agent_traces WHERE task_kind = ? ORDER BY started_at
            "#,
            kind
        )
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::testutil::{fresh_store, sample_finding, sample_repo, sample_run};

    fn sample_trace(id: &str, finding_id: Option<&str>) -> AgentTraceRecord {
        AgentTraceRecord {
            id: id.to_string(),
            finding_id: finding_id.map(str::to_string),
            task_kind: "PayloadSynthesis".to_string(),
            runtime_name: "anthropic".to_string(),
            model: "claude-opus-4-7".to_string(),
            prompt_version: Some("payload/v1".to_string()),
            conversation_jsonl_path: Some("/var/log/conv-1.jsonl".to_string()),
            tokens_in: 1_000,
            tokens_out: 200,
            cost_usd_micros: 12_345,
            cache_hits: 3,
            cache_misses: 1,
            duration_ms: Some(7_500),
            started_at: 5_000,
            finished_at: Some(12_500),
            verifier_blob: None,
        }
    }

    async fn seed_finding(s: &crate::store::Store) -> String {
        s.repos().upsert(&sample_repo("repo")).await.expect("repo");
        s.runs().insert(&sample_run("run")).await.expect("run");
        let f = sample_finding("run", "repo", "src/a.rs", "rule");
        let fid = f.id.clone();
        s.findings().upsert(&f).await.expect("finding");
        fid
    }

    #[tokio::test]
    async fn insert_then_get_roundtrips() {
        let (_tmp, s) = fresh_store().await;
        let fid = seed_finding(&s).await;
        let t = sample_trace("t-1", Some(&fid));
        s.agent_traces().insert(&t).await.expect("insert");
        let got = s.agent_traces().get("t-1").await.expect("get").expect("row");
        assert_eq!(got, t);
    }

    #[tokio::test]
    async fn insert_with_null_finding_id() {
        let (_tmp, s) = fresh_store().await;
        let t = sample_trace("t-orphan", None);
        s.agent_traces().insert(&t).await.expect("insert");
        let got = s.agent_traces().get("t-orphan").await.expect("get").expect("row");
        assert!(got.finding_id.is_none());
    }

    #[tokio::test]
    async fn list_for_finding_returns_only_matching() {
        let (_tmp, s) = fresh_store().await;
        let fid = seed_finding(&s).await;
        s.agent_traces().insert(&sample_trace("a", Some(&fid))).await.expect("a");
        s.agent_traces().insert(&sample_trace("b", None)).await.expect("b");
        let got = s.agent_traces().list_for_finding(&fid).await.expect("list");
        let ids: Vec<_> = got.into_iter().map(|t| t.id).collect();
        assert_eq!(ids, vec!["a".to_string()]);
    }

    #[tokio::test]
    async fn list_by_task_kind_filters() {
        let (_tmp, s) = fresh_store().await;
        let mut a = sample_trace("a", None);
        a.task_kind = "ChainReasoning".to_string();
        let b = sample_trace("b", None);
        s.agent_traces().insert(&a).await.expect("a");
        s.agent_traces().insert(&b).await.expect("b");
        let got = s.agent_traces().list_by_task_kind("PayloadSynthesis").await.expect("list");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, "b");
    }

    #[tokio::test]
    async fn list_for_candidate_returns_joined_row() {
        use crate::store::testutil::sample_candidate;
        let (_tmp, s) = fresh_store().await;
        s.repos().upsert(&sample_repo("repo")).await.expect("repo");
        s.runs().insert(&sample_run("run")).await.expect("run");
        // Persist the trace first so the candidate FK target exists.
        let trace = sample_trace("trace-novel-1", None);
        s.agent_traces().insert(&trace).await.expect("trace");
        let mut cand = sample_candidate("cand-listfor", "run", "repo");
        cand.trace_id = Some("trace-novel-1".to_string());
        s.candidate_findings().insert(&cand).await.expect("cand");
        // Unrelated trace + candidate that should not appear in the
        // joined result.
        let other = sample_trace("trace-other", None);
        s.agent_traces().insert(&other).await.expect("other trace");
        let mut other_cand = sample_candidate("cand-other", "run", "repo");
        other_cand.trace_id = Some("trace-other".to_string());
        s.candidate_findings().insert(&other_cand).await.expect("other cand");
        let rows = s.agent_traces().list_for_candidate("cand-listfor").await.expect("list");
        let ids: Vec<_> = rows.into_iter().map(|t| t.id).collect();
        assert_eq!(ids, vec!["trace-novel-1".to_string()]);
    }

    #[tokio::test]
    async fn list_for_candidate_without_trace_id_returns_empty() {
        use crate::store::testutil::sample_candidate;
        let (_tmp, s) = fresh_store().await;
        s.repos().upsert(&sample_repo("repo")).await.expect("repo");
        s.runs().insert(&sample_run("run")).await.expect("run");
        let cand = sample_candidate("cand-untraced", "run", "repo");
        s.candidate_findings().insert(&cand).await.expect("cand");
        let rows = s.agent_traces().list_for_candidate("cand-untraced").await.expect("list");
        assert!(rows.is_empty(), "candidate with NULL trace_id must yield no joined rows");
    }

    #[tokio::test]
    async fn finding_delete_sets_finding_id_null() {
        let (_tmp, s) = fresh_store().await;
        let fid = seed_finding(&s).await;
        s.agent_traces().insert(&sample_trace("t", Some(&fid))).await.expect("insert");
        s.findings().delete(&fid).await.expect("del");
        let got = s.agent_traces().get("t").await.expect("get").expect("row");
        assert!(got.finding_id.is_none(), "expected SET NULL after finding deleted");
    }
}
