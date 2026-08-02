from __future__ import annotations

import math

import pandas as pd

from rusearchrank.evaluation import (
    evaluate_bm25_metrics,
    paired_ranking_comparison,
    sparse_judgment_diagnostics,
    stratified_delta_summary,
)


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = pd.DataFrame(
        [
            ("q1", "u1", 1, pd.NA, "unjudged"),
            ("q1", "n1", 2, 0, "judged_non_relevant"),
            ("q1", "r1", 3, 1, "relevant"),
            ("q2", "u2", 1, pd.NA, "unjudged"),
            ("q2", "n2", 2, 0, "judged_non_relevant"),
            ("q3", "r3", 1, 1, "relevant"),
            ("q3", "u3", 2, pd.NA, "unjudged"),
        ],
        columns=["query_id", "docid", "bm25_rank", "relevance_grade", "judgment"],
    )
    qrels = pd.DataFrame(
        [
            ("q1", "r1", 1),
            ("q1", "n1", 0),
            ("q2", "outside", 1),
            ("q2", "n2", 0),
            ("q3", "r3", 1),
        ],
        columns=["query_id", "docid", "relevance_grade"],
    )
    ranking = candidates[["query_id", "docid", "bm25_rank"]].copy()
    return candidates, qrels, ranking


def test_standard_ndcg_matches_manual_example() -> None:
    _, qrels, ranking = fixture()
    report = evaluate_bm25_metrics(
        ranking, qrels, query_ids=["q1", "q2", "q3"]
    )
    by_query = {row["query_id"]: row for row in report["per_query"]}
    assert math.isclose(by_query["q1"]["ndcg_at_10"], 0.5, abs_tol=1e-12)
    assert by_query["q2"]["ndcg_at_10"] == 0.0
    assert by_query["q3"]["ndcg_at_10"] == 1.0
    assert math.isclose(
        report["aggregate"]["ndcg_at_10"], 0.5, abs_tol=1e-12
    )


def test_condensed_ndcg_judged_and_unjudged_counts() -> None:
    candidates, qrels, ranking = fixture()
    diagnostics = sparse_judgment_diagnostics(
        candidates=candidates,
        qrels=qrels,
        ranking=ranking,
        bm25_ranking=ranking,
        query_ids=["q1", "q2", "q3"],
    )
    assert diagnostics["condensed_ndcg_at_10"] >= 0.5
    assert math.isclose(diagnostics["judged_at_10"], 4 / 30, abs_tol=1e-12)
    assert math.isclose(diagnostics["mean_unjudged_in_top10"], 1.0, abs_tol=1e-12)
    assert diagnostics["queries_without_judged_candidate"] == 0
    assert diagnostics["queries_without_relevant_candidate"] == 1


def test_unjudged_relevant_inversions_have_exact_manual_count() -> None:
    candidates, qrels, ranking = fixture()
    diagnostics = sparse_judgment_diagnostics(
        candidates=candidates,
        qrels=qrels,
        ranking=ranking,
        bm25_ranking=ranking,
        query_ids=["q1", "q2", "q3"],
    )
    assert diagnostics["queries_with_unjudged_above_relevant"] == 1
    assert diagnostics["pairwise_unjudged_relevant_inversions"] == 1


def test_oracle_and_strata_keep_no_relevant_separate() -> None:
    candidates, qrels, ranking = fixture()
    diagnostics = sparse_judgment_diagnostics(
        candidates=candidates,
        qrels=qrels,
        ranking=ranking,
        bm25_ranking=ranking,
        query_ids=["q1", "q2", "q3"],
    )
    assert math.isclose(
        diagnostics["oracle_ndcg_at_10_over_candidates"], 2 / 3, abs_tol=1e-12
    )
    assert diagnostics["queries_at_oracle_under_bm25"] == 2
    baseline = {"q1": 0.5, "q2": 0.0, "q3": 1.0}
    system = {"q1": 1.0, "q2": 0.0, "q3": 0.5}
    strata = stratified_delta_summary(
        candidates=candidates,
        baseline_per_query=baseline,
        system_per_query=system,
        oracle_per_query=diagnostics["oracle_per_query"],
    )
    assert strata["no_relevant_in_candidates"] == {
        "query_count": 1,
        "mean_delta": 0.0,
    }
    assert strata["already_at_oracle"] == {
        "query_count": 1,
        "mean_delta": -0.5,
    }
    assert strata["improvable"] == {"query_count": 1, "mean_delta": 0.5}


def test_unjudged_is_never_added_to_ideal_dcg() -> None:
    candidates, qrels, ranking = fixture()
    q1 = evaluate_bm25_metrics(
        ranking.loc[ranking["query_id"].eq("q1")],
        qrels.loc[qrels["query_id"].eq("q1")],
        query_ids=["q1"],
    )
    # Only r1 is relevant in qrels, so IDCG is exactly the rank-1 gain. Treating
    # unjudged u1 as an explicit grade would change this manually verified 0.5.
    assert q1["aggregate"]["ndcg_at_10"] == 0.5
    assert candidates.loc[candidates["docid"].eq("u1"), "relevance_grade"].isna().all()


def test_every_depth_comparison_uses_the_same_baseline() -> None:
    baseline = {"q1": 0.2, "q2": 0.4, "q3": 0.6}
    depth10 = paired_ranking_comparison(
        baseline, {"q1": 0.3, "q2": 0.4, "q3": 0.5}, resamples=100
    )
    depth100 = paired_ranking_comparison(
        baseline, {"q1": 0.1, "q2": 0.5, "q3": 0.6}, resamples=100
    )
    assert (depth10["improved"], depth10["degraded"], depth10["tie"]) == (1, 1, 1)
    assert (depth100["improved"], depth100["degraded"], depth100["tie"]) == (1, 1, 1)
