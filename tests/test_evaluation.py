from __future__ import annotations

import math

import pandas as pd
import pytest

from rusearchrank.evaluation import (
    evaluate_bm25_metrics,
    parse_trec_eval_metric,
    reproduction_rows,
)


def ranking(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["query_id", "docid", "bm25_rank"])


def qrels(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["query_id", "docid", "relevance_grade"])


def test_manual_ndcg_recall_hit_judged_and_condensed() -> None:
    candidates = ranking([("q", "d1", 1), ("q", "u", 2), ("q", "d2", 3)])
    labels = qrels([("q", "d1", 1), ("q", "d2", 1)])
    metrics = evaluate_bm25_metrics(candidates, labels)["aggregate"]
    ideal = 1.0 + 1.0 / math.log2(3)
    expected_standard = (1.0 + 1.0 / math.log2(4)) / ideal
    assert metrics["ndcg_at_10"] == pytest.approx(expected_standard)
    assert metrics["recall_at_100"] == 1.0
    assert metrics["hit_at_100"] == 1.0
    assert metrics["judged_at_10"] == 0.2
    assert metrics["condensed_ndcg_at_10"] == 1.0


def test_multiple_positives_and_retrieval_failure_are_averaged_over_queries() -> None:
    candidates = ranking([("q1", "p1", 1), ("q1", "p2", 2)])
    labels = qrels(
        [("q1", "p1", 1), ("q1", "p2", 1), ("q2", "missing", 1)]
    )
    report = evaluate_bm25_metrics(candidates, labels, query_ids=["q1", "q2"])
    assert report["query_count"] == 2
    assert report["aggregate"]["recall_at_100"] == 0.5
    assert report["aggregate"]["hit_at_100"] == 0.5
    q2 = next(row for row in report["per_query"] if row["query_id"] == "q2")
    assert q2["retrieved"] == 0
    assert q2["ndcg_at_10"] == 0.0


def test_all_unjudged_ranking_has_zero_effectiveness_and_condensed_ndcg() -> None:
    candidates = ranking([("q", "u1", 1), ("q", "u2", 2)])
    labels = qrels([("q", "positive-not-retrieved", 1)])
    metrics = evaluate_bm25_metrics(candidates, labels)["aggregate"]
    assert metrics["ndcg_at_10"] == 0.0
    assert metrics["recall_at_100"] == 0.0
    assert metrics["hit_at_100"] == 0.0
    assert metrics["judged_at_10"] == 0.0
    assert metrics["condensed_ndcg_at_10"] == 0.0


def test_fully_judged_top10_makes_standard_and_condensed_equal() -> None:
    candidates = ranking([("q", f"d{i}", i) for i in range(1, 11)])
    labels = qrels([("q", f"d{i}", int(i in {2, 7})) for i in range(1, 11)])
    metrics = evaluate_bm25_metrics(candidates, labels)["aggregate"]
    assert metrics["judged_at_10"] == 1.0
    assert metrics["condensed_ndcg_at_10"] == pytest.approx(metrics["ndcg_at_10"])


def test_condensed_uses_the_full_candidate_list_before_cutoff() -> None:
    candidates = ranking(
        [("q", f"u{i}", i) for i in range(1, 11)] + [("q", "positive", 11)]
    )
    labels = qrels([("q", "positive", 1)])
    metrics = evaluate_bm25_metrics(candidates, labels)["aggregate"]
    assert metrics["ndcg_at_10"] == 0.0
    assert metrics["condensed_ndcg_at_10"] == 1.0


def test_parse_official_trec_eval_output() -> None:
    output = "ndcg_cut_10\tall\t0.3340\n"
    assert parse_trec_eval_metric(output, "ndcg_cut_10") == 0.334
    with pytest.raises(ValueError, match="expected one"):
        parse_trec_eval_metric("", "ndcg_cut_10")


def test_reproduction_rows_apply_absolute_tolerance() -> None:
    rows = reproduction_rows(
        official={"ndcg_at_10": 0.334, "recall_at_100": 0.661},
        local={"ndcg_at_10": 0.335, "recall_at_100": 0.670},
        tolerances={"ndcg_at_10": 0.002, "recall_at_100": 0.005},
    )
    assert [row["status"] for row in rows] == ["PASS", "FAIL"]


def test_duplicate_candidates_and_qrels_fail_loudly() -> None:
    candidates = ranking([("q", "d", 1), ("q", "d", 2)])
    labels = qrels([("q", "d", 1)])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_bm25_metrics(candidates, labels)
    duplicate_labels = qrels([("q", "d", 1), ("q", "d", 1)])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_bm25_metrics(ranking([("q", "d", 1)]), duplicate_labels)
