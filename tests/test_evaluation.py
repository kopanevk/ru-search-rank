from __future__ import annotations

import math
from pathlib import Path
import subprocess

import pandas as pd
import pytest

import rusearchrank.cli as cli_module
from rusearchrank.evaluation import (
    build_qrels_split_audit,
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


def test_parse_tab_separated_trec_eval_output() -> None:
    output = "ndcg_cut_10\tall\t0.3340\n"
    assert parse_trec_eval_metric(output, "ndcg_cut_10") == 0.334


def test_parse_space_separated_trec_eval_output() -> None:
    output = "recall_100    all    0.6610\n"
    assert parse_trec_eval_metric(output, "recall_100") == 0.661


def test_parse_trec_eval_ignores_warning_and_blank_lines() -> None:
    output = "\nwarning: ignored diagnostic text\nndcg_cut_10 all 0.3342\n"
    assert parse_trec_eval_metric(output, "ndcg_cut_10") == 0.3342


def test_parse_trec_eval_rejects_missing_metric() -> None:
    with pytest.raises(ValueError, match="expected one"):
        parse_trec_eval_metric("map all 0.2\n", "ndcg_cut_10")


def test_parse_trec_eval_rejects_empty_output() -> None:
    with pytest.raises(ValueError, match="ndcg_cut_10"):
        parse_trec_eval_metric("", "ndcg_cut_10")


def test_direct_trec_eval_nonzero_return_code_is_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(
            2,
            args[0],
            output="partial stdout",
            stderr="fatal stderr",
        )

    monkeypatch.setattr(cli_module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="exit code 2"):
        cli_module._run_trec_eval_binary(
            Path("/usr/local/bin/trec_eval"), ["-m", "recall.100", "qrels", "run"]
        )


def test_direct_trec_eval_preserves_stdout_stderr_and_uses_check_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def succeed(command: list[str], **kwargs: object) -> object:
        assert kwargs["check"] is True
        return type(
            "CompletedProcess",
            (),
            {"returncode": 0, "stdout": "recall_100 all 0.661\n", "stderr": "note\n"},
        )()

    monkeypatch.setattr(cli_module.subprocess, "run", succeed)
    result = cli_module._run_trec_eval_binary(
        Path("/usr/local/bin/trec_eval"), ["-m", "recall.100", "qrels", "run"]
    )
    assert result["stdout"] == "recall_100 all 0.661\n"
    assert result["stderr"] == "note\n"


def test_both_official_metrics_pass_reproduction_gate() -> None:
    rows = reproduction_rows(
        official={"ndcg_at_10": 0.334, "recall_at_100": 0.661},
        local={"ndcg_at_10": 0.3342, "recall_at_100": 0.661},
        tolerances={"ndcg_at_10": 0.002, "recall_at_100": 0.005},
    )
    assert all(row["status"] == "PASS" for row in rows)


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


def test_qrels_audit_reports_three_states_per_query_and_zero_judged() -> None:
    queries = pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "query_text": ["one", "two"],
        }
    )
    labels = qrels([("q1", "positive", 1), ("q1", "negative", 0)])
    candidates = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q2"],
            "docid": ["positive", "negative", "unknown", "only-unjudged"],
            "bm25_rank": [1, 2, 3, 1],
            "judgment": [
                "relevant",
                "judged_non_relevant",
                "unjudged",
                "unjudged",
            ],
        }
    )
    report = build_qrels_split_audit(
        queries=queries,
        qrels=labels,
        candidates=candidates,
    )
    assert report["unique_qrels_query_doc_pairs"] == 2
    assert report["candidate_judgment_counts"] == {
        "relevant": 1,
        "judged_non_relevant": 1,
        "unjudged": 2,
    }
    assert report["zero_judged_query_count"] == 1
    assert report["zero_judged_queries"] == [
        {"query_id": "q2", "query_text": "two"}
    ]
