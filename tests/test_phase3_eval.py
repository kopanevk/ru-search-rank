from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from rusearchrank.evaluation import (
    assert_candidate_set_invariant,
    evaluate_ranked_ndcg_at_10,
    sparse_judgment_diagnostics,
)
import rusearchrank.phase3_eval as phase3_eval_module
from rusearchrank.phase3_eval import (
    append_dev_access_ledger,
    build_three_way_comparison,
    classify_control_result,
    classify_ml_outcome,
    phase3_score_tie_statistics,
    select_checkpoint,
)
from rusearchrank.training import StageError
from rusearchrank.training_data import load_finetune_config


@pytest.mark.parametrize(
    ("mean", "low", "high", "expected"),
    [
        (0.02, 0.01, 0.03, "improvement_confirmed"),
        (0.005, 0.001, 0.009, "positive_below_practical_threshold"),
        (0.01, -0.01, 0.03, "inconclusive_positive"),
        (1e-13, -0.01, 0.01, "no_detectable_change"),
        (-0.01, -0.03, 0.01, "inconclusive_negative"),
        (-0.02, -0.03, -0.01, "degradation_confirmed"),
    ],
)
def test_all_six_ml_outcomes_are_reachable(
    mean: float, low: float, high: float, expected: str
) -> None:
    report = classify_ml_outcome(mean_delta=mean, ci_lower=low, ci_upper=high)
    assert report["label"] == expected
    assert report["improvement_confirmed"] is (expected == "improvement_confirmed")
    assert report["degradation_confirmed"] is (expected == "degradation_confirmed")


def test_bootstrap_interval_must_contain_mean() -> None:
    with pytest.raises(ValueError, match="ci_lower"):
        classify_ml_outcome(mean_delta=0.5, ci_lower=-0.1, ci_upper=0.1)


def test_three_way_comparison_is_reproducible_and_pipeline_passes_negative_outcome() -> None:
    bm25 = {"q1": 0.1, "q2": 0.2, "q3": 0.3}
    zero = {"q1": 0.8, "q2": 0.7, "q3": 0.6}
    fine = {"q1": 0.7, "q2": 0.6, "q3": 0.5}
    first = build_three_way_comparison(bm25, zero, fine, resamples=200)
    second = build_three_way_comparison(bm25, zero, fine, resamples=200)
    assert json.dumps(first, sort_keys=True).encode() == json.dumps(second, sort_keys=True).encode()
    assert first["pipeline_status"] == "PASS"
    assert first["ml_outcome"]["label"] == "degradation_confirmed"
    assert math.isclose(first["primary_finetuned_minus_zero_shot"]["mean_delta"], -0.1)


def test_three_way_requires_one_exact_query_universe() -> None:
    with pytest.raises(ValueError, match="universes differ"):
        build_three_way_comparison({"q": 0.0}, {"q": 0.0}, {"other": 0.0})


def test_manual_ndcg_and_three_run_candidate_set_invariant() -> None:
    qrels = pd.DataFrame(
        {
            "query_id": ["q", "q"],
            "docid": ["relevant", "other"],
            "relevance_grade": [1, 0],
        }
    )
    best = pd.DataFrame(
        {"query_id": ["q", "q"], "docid": ["relevant", "other"], "rank": [1, 2]}
    )
    worse = pd.DataFrame(
        {"query_id": ["q", "q"], "docid": ["other", "relevant"], "rank": [1, 2]}
    )
    best_value = evaluate_ranked_ndcg_at_10(best, qrels, query_ids=["q"])["ndcg_at_10"]
    worse_value = evaluate_ranked_ndcg_at_10(worse, qrels, query_ids=["q"])["ndcg_at_10"]
    assert best_value == pytest.approx(1.0)
    assert worse_value == pytest.approx(1.0 / math.log2(3.0))
    assert_candidate_set_invariant(best, worse)
    changed = worse.copy()
    changed.loc[0, "docid"] = "outside"
    with pytest.raises(ValueError, match="candidate-set invariant"):
        assert_candidate_set_invariant(best, changed)


@pytest.mark.parametrize(
    ("mean", "low", "structural", "expected"),
    [
        (0.1, 0.01, True, "BLOCKED_FOR_REVIEW"),
        (0.1, -0.01, True, "WARN"),
        (0.0, -0.01, True, "PASS"),
        (-0.1, -0.2, False, "FAIL"),
    ],
)
def test_control_verdict_is_separate_from_pipeline_failure(
    mean: float, low: float, structural: bool, expected: str
) -> None:
    assert (
        classify_control_result(
            mean_delta=mean,
            ci_lower=low,
            structural_checks_passed=structural,
        )
        == expected
    )


def _scores(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["q"] * len(values),
            "docid": [f"d{index:02d}" for index in range(len(values))],
            "score": np.asarray(values, dtype=np.float32),
        }
    )


def test_phase3_tie_diagnostic_counts_top10_and_boundary_groups() -> None:
    top10 = phase3_score_tie_statistics(
        _scores([12, 11, 10, 9, 9, 7, 6, 5, 4, 3, 2, 1])
    )
    assert top10 == {
        "exact_raw_float32_tie_groups": 1,
        "queries_with_any_tie": 1,
        "queries_with_top10_tie": 1,
        "queries_with_boundary_tie": 0,
    }
    boundary = phase3_score_tie_statistics(
        _scores([12, 11, 10, 9, 8, 7, 6, 5, 4, 1, 1, 0])
    )
    assert boundary["queries_with_boundary_tie"] == 1
    assert boundary["queries_with_top10_tie"] == 0


def test_sparse_inversions_count_relevant_below_top10_exactly() -> None:
    rows = []
    for rank in range(1, 13):
        docid = "r" if rank == 11 else f"u{rank}"
        rows.append(
            {
                "query_id": "q",
                "docid": docid,
                "bm25_rank": rank,
                "relevance_grade": 1 if docid == "r" else pd.NA,
                "judgment": "relevant" if docid == "r" else "unjudged",
            }
        )
    candidates = pd.DataFrame(rows)
    qrels = pd.DataFrame(
        {"query_id": ["q"], "docid": ["r"], "relevance_grade": [1]}
    )
    ranking = candidates[["query_id", "docid", "bm25_rank"]]
    report = sparse_judgment_diagnostics(
        candidates=candidates,
        qrels=qrels,
        ranking=ranking,
        bm25_ranking=ranking,
        query_ids=["q"],
    )
    assert report["queries_with_unjudged_above_relevant"] == 1
    assert report["pairwise_unjudged_relevant_inversions"] == 10


def test_dev_ledger_is_append_only_and_marks_repeat_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "configs/finetune.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("fixture: true\n", encoding="utf-8")
    config = {
        "_config_path": str(config_path),
        "paths": {"repository_root": ".."},
        "audits": {
            "dev_access_ledger": "reports/audit/ledger.jsonl",
            "checkpoint_selection": "reports/audit/selection.json",
        },
    }
    selection = {
        "schema_version": 2,
        "selection_written_before_dev_access": True,
        "selected_at": "2020-01-01T00:00:00+00:00",
        "best_finetuned_checkpoint": {"sha256": "a" * 64},
    }
    selection["selection_sha256"] = phase3_eval_module._selection_payload_sha256(
        selection
    )
    selection_path = tmp_path / "reports/audit/selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps(selection) + "\n", encoding="utf-8")
    inputs = {"qrels": "c" * 64}
    first = append_dev_access_ledger(
        config,
        command="prepare-dev-evaluation",
        checkpoint_sha256="a" * 64,
        input_hashes=inputs,
    )
    second = append_dev_access_ledger(
        config,
        command="score-finetuned",
        checkpoint_sha256="a" * 64,
        input_hashes=inputs,
    )
    assert first["repeat_access"] is False
    assert second["repeat_access"] is True
    ledger = tmp_path / "reports/audit/ledger.jsonl"
    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["sequence"] == 1
    assert lines[0]["previous_event_sha256"] == "0" * 64
    assert lines[1]["sequence"] == 2
    assert lines[1]["previous_event_sha256"] == lines[0]["event_sha256"]

    with pytest.raises(StageError, match="different from the selection"):
        append_dev_access_ledger(
            config,
            command="score-finetuned",
            checkpoint_sha256="b" * 64,
            input_hashes=inputs,
        )
    with pytest.raises(StageError, match="input declarations changed"):
        append_dev_access_ledger(
            config,
            command="score-finetuned",
            checkpoint_sha256="a" * 64,
            input_hashes={"qrels": "d" * 64},
        )
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2


def test_dev_ledger_refuses_access_before_selection(tmp_path: Path) -> None:
    config_path = tmp_path / "configs/finetune.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("fixture: true\n", encoding="utf-8")
    config = {
        "_config_path": str(config_path),
        "paths": {"repository_root": ".."},
        "audits": {
            "dev_access_ledger": "reports/audit/ledger.jsonl",
            "checkpoint_selection": "reports/audit/selection.json",
        },
    }
    with pytest.raises(StageError, match="selection must exist"):
        append_dev_access_ledger(
            config,
            command="prepare-dev-evaluation",
            checkpoint_sha256="a" * 64,
            input_hashes={"qrels": "c" * 64},
        )


def _temporary_config(tmp_path: Path) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "configs/finetune.yaml"
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(repository / "configs/finetune.yaml", config_path)
    return load_finetune_config(config_path)


def test_selection_separates_best_finetuned_from_zero_shot_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _temporary_config(tmp_path)
    metrics_path = tmp_path / "reports/metrics/validation_checkpoint_metrics.json"
    metrics_path.parent.mkdir(parents=True)
    per_query = {"q1": 0.5, "q2": 0.6}
    metrics = {
        "S0": {
            "epoch": 0,
            "ndcg_at_10": 0.55,
            "checkpoint_sha256": "0" * 64,
            "per_query": per_query,
        },
        "runs": {},
    }
    values = {
        "A1": [0.55, 0.55, 0.54],
        "A2": [0.55, 0.53, 0.52],
        "B1": [0.54, 0.53, 0.52],
    }
    for run_id, ndcgs in values.items():
        metrics["runs"][run_id] = {
            f"epoch_{epoch}": {
                "epoch": epoch,
                "ndcg_at_10": ndcg,
                "checkpoint_sha256": str(epoch) * 64,
                "model_generation_sha256": chr(ord("a") + epoch - 1) * 64,
                "per_query": per_query,
                "sparse_diagnostics": {},
            }
            for epoch, ndcg in enumerate(ndcgs, start=1)
        }
        manifest = tmp_path / f"artifacts/models/{run_id}/run_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "finalized": True,
                    "resume_available": False,
                    "learning_rate": 7.0e-6,
                    "training_fingerprint": "e" * 64,
                    "training_fingerprint_components": {"fixture": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    def fake_publish(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "checkpoint_payload_sha256": "f" * 64,
            "source_model_generation_sha256": kwargs[
                "expected_generation_sha256"
            ],
            "files": {
                "model.safetensors": kwargs["expected_weight_sha256"]
            },
        }

    monkeypatch.setattr(phase3_eval_module, "_publish_best_finetuned", fake_publish)
    monkeypatch.setattr(
        phase3_eval_module,
        "training_fingerprint_components",
        lambda *args, **kwargs: {"fixture": True},
    )
    monkeypatch.setattr(
        phase3_eval_module,
        "build_training_fingerprint",
        lambda *args, **kwargs: "e" * 64,
    )
    monkeypatch.setattr(
        phase3_eval_module,
        "_validation_ab_report",
        lambda *args, **kwargs: {"analysis_role": "exploratory_post_selection"},
    )
    monkeypatch.setattr(phase3_eval_module, "_write_model_card", lambda *args, **kwargs: None)
    report = select_checkpoint(config)
    assert len(report["candidates"]) == 10
    assert report["best_finetuned_checkpoint"]["run_id"] == "A1"
    assert report["best_finetuned_checkpoint"]["epoch"] == 1
    assert report["production_system"]["kind"] == "zero_shot"
    assert report["zero_shot_won"] is True
    assert report["schema_version"] == 2
    assert len(report["selection_sha256"]) == 64
    assert "production_dir" not in config["artifacts"]


def test_selection_cannot_be_written_after_dev_ledger(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    ledger = tmp_path / "reports/audit/dev_access_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"checkpoint_sha256":"x"}\n', encoding="utf-8")
    with pytest.raises(StageError, match="after evaluation access"):
        select_checkpoint(config)


def test_failed_zip_candidate_keeps_previous_valid_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "payload.txt"
    archive = tmp_path / "result.zip"
    source.write_text("first\n", encoding="utf-8")
    phase3_eval_module._publish_validated_zip(
        archive, [("payload.txt", source)]
    )
    previous = archive.read_bytes()
    source.write_text("second\n", encoding="utf-8")

    def reject_candidate(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("injected candidate validation failure")

    monkeypatch.setattr(phase3_eval_module, "_validate_zip", reject_candidate)
    with pytest.raises(ValueError, match="injected"):
        phase3_eval_module._publish_validated_zip(
            archive, [("payload.txt", source)]
        )
    assert archive.read_bytes() == previous
    assert not archive.with_name(
        f".{archive.name}.candidate.{os.getpid()}"
    ).exists()
