from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
import pytest
import yaml

from rusearchrank.cli import main
from rusearchrank import phase3_eval as phase3_eval_module
from rusearchrank import rerank as rerank_module
from rusearchrank import training as training_module
from rusearchrank.training_data import (
    atomic_write_json,
    load_finetune_config,
    resolve_path,
    sha256_file,
)


def _fixture_candidates() -> pd.DataFrame:
    rows = []
    for query_number in range(40):
        query_id = f"q{query_number:03d}"
        for rank in range(1, 101):
            if rank == 1:
                judgment, grade, relevance, judged = "relevant", 1, 1, True
            elif query_number % 2 == 0 and rank in {2, 3, 4}:
                judgment, grade, relevance, judged = (
                    "judged_non_relevant",
                    0,
                    0,
                    True,
                )
            else:
                judgment, grade, relevance, judged = "unjudged", pd.NA, pd.NA, False
            rows.append(
                {
                    "split": "train",
                    "query_id": query_id,
                    "docid": f"{query_id}-d{rank:03d}",
                    "bm25_rank": rank,
                    "bm25_score": float(101 - rank),
                    "relevance_grade": grade,
                    "judgment": judgment,
                    "relevance": relevance,
                    "is_judged": judged,
                }
            )
    frame = pd.DataFrame(rows)
    frame["relevance_grade"] = frame["relevance_grade"].astype("Int64")
    frame["relevance"] = frame["relevance"].astype("Int64")
    return frame


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(tmp_path: Path) -> Path:
    repository = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "artifacts/candidates").mkdir(parents=True)
    (tmp_path / "reports/audit").mkdir(parents=True)
    config_path = tmp_path / "configs/finetune.yaml"
    shutil.copyfile(repository / "configs/finetune.yaml", config_path)
    _fixture_candidates().to_parquet(
        tmp_path / "artifacts/candidates/train_top100.parquet", index=False
    )
    # The split command treats a missing external manifest hash as unavailable;
    # a tiny fixture manifest remains immutable across every local stage.
    (tmp_path / "reports/audit/candidate_cache_manifest.json").write_text(
        '{"status":"fixture","artifacts":[]}\n', encoding="utf-8"
    )
    return config_path


def test_offline_data_pipeline_is_idempotent_and_keeps_prior_phases_immutable(
    tmp_path: Path,
) -> None:
    config = build_fixture(tmp_path)
    candidate_path = tmp_path / "artifacts/candidates/train_top100.parquet"
    phase1_manifest = tmp_path / "reports/audit/candidate_cache_manifest.json"
    immutable = {candidate_path: candidate_path.read_bytes(), phase1_manifest: phase1_manifest.read_bytes()}
    assert main(["build-training-split", "--config", str(config)]) == 0
    for regime in ("judged_only", "weak_negatives", "control_c1"):
        assert (
            main(
                [
                    "build-training-pairs",
                    "--config",
                    str(config),
                    "--regime",
                    regime,
                ]
            )
            == 0
        )
    artifacts = [
        tmp_path / "artifacts/training/query_split.parquet",
        tmp_path / "artifacts/training/validation_groups.parquet",
        tmp_path / "artifacts/training/pairs_judged_only.parquet",
        tmp_path / "artifacts/training/pairs_weak_negatives.parquet",
        tmp_path / "artifacts/training/pairs_control_c1.parquet",
        tmp_path / "reports/audit/query_split_manifest.json",
        tmp_path / "reports/audit/pairs_manifest.json",
    ]
    before = {path: path.read_bytes() for path in artifacts}
    assert main(["build-training-split", "--config", str(config)]) == 0
    for regime in ("judged_only", "weak_negatives", "control_c1"):
        assert main(["build-training-pairs", "--config", str(config), "--regime", regime]) == 0
    assert {path: path.read_bytes() for path in artifacts} == before
    assert {path: path.read_bytes() for path in immutable} == immutable
    pairs_manifest = yaml.safe_load((tmp_path / "reports/audit/pairs_manifest.json").read_text())
    assert set(pairs_manifest["regimes"]) == {
        "judged_only",
        "weak_negatives",
        "control_c1",
    }
    assert pairs_manifest["regimes"]["weak_negatives"]["usable_query_count"] > pairs_manifest[
        "regimes"
    ]["judged_only"]["usable_query_count"]


def test_evaluation_commands_are_blocked_before_selection(tmp_path: Path) -> None:
    config = build_fixture(tmp_path)
    assert main(["prepare-dev-evaluation", "--config", str(config)]) == 1
    assert main(["score-finetuned", "--config", str(config)]) == 1
    assert not (tmp_path / "reports/audit/dev_access_ledger.jsonl").exists()


def test_changed_sampling_config_is_rejected_exactly(tmp_path: Path) -> None:
    config_path = build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["negatives"]["weak_rank_min"] = 25
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    assert main(["build-training-split", "--config", str(config_path)]) == 1


def test_clean_process_static_guards_and_notebook_validator() -> None:
    repository = Path(__file__).resolve().parents[1]
    cli = subprocess.run(
        [sys.executable, "-m", "rusearchrank.cli", "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stderr
    assert "package-phase3" in cli.stdout
    result = subprocess.run(
        [sys.executable, "scripts/validate_phase3_notebook.py"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for module in ("training.py", "training_data.py"):
        tree = ast.parse((repository / "src/rusearchrank" / module).read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "datasets" not in imports
    for module in (repository / "src").rglob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        assert all(
            not (
                isinstance(node, ast.Call)
                and any(keyword.arg == "trust_remote_code" for keyword in node.keywords)
            )
            for node in ast.walk(tree)
        )


def _write_run(path: Path, candidates: pd.DataFrame, *, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in candidates.sort_values(
        ["query_id", "bm25_rank"], kind="mergesort"
    ).itertuples(index=False):
        lines.append(
            f"{row.query_id} Q0 {row.docid} {int(row.bm25_rank)} "
            f"{1_000_000 - int(row.bm25_rank):.4f} {tag}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _full_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], pd.DataFrame]:
    config_path = build_fixture(tmp_path)
    repository = Path(__file__).resolve().parents[1]
    for relative in (
        "configs/rerank.yaml",
        "src/rusearchrank/training.py",
        "src/rusearchrank/training_data.py",
        "src/rusearchrank/pair_encoding.py",
        "src/rusearchrank/rerank.py",
        "src/rusearchrank/evaluation.py",
        "src/rusearchrank/phase3_eval.py",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)

    config = load_finetune_config(config_path)
    train = _fixture_candidates()
    train_query_ids = sorted(train["query_id"].unique())
    dev_query_ids = ["dq1", "dq2", "dq3", "dq4"]
    queries = pd.DataFrame(
        {
            "query_id": train_query_ids + dev_query_ids,
            "query_text": [f"query {value}" for value in train_query_ids + dev_query_ids],
        }
    )
    queries_path = resolve_path(config, config["inputs"]["queries"])
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    queries.to_parquet(queries_path, index=False)

    passages = train[["docid"]].copy()
    passages["title"] = "title"
    passages["text"] = passages["docid"].map(lambda value: f"passage {value}")

    dev_rows = []
    for query_id in dev_query_ids:
        for rank in range(1, 13):
            if rank == 1:
                judgment, grade, relevance, judged = "relevant", 1, 1, True
            elif rank == 2:
                judgment, grade, relevance, judged = (
                    "judged_non_relevant",
                    0,
                    0,
                    True,
                )
            else:
                judgment, grade, relevance, judged = (
                    "unjudged",
                    pd.NA,
                    pd.NA,
                    False,
                )
            docid = f"{query_id}-d{rank:03d}"
            dev_rows.append(
                {
                    "split": "dev",
                    "query_id": query_id,
                    "docid": docid,
                    "bm25_rank": rank,
                    "bm25_score": float(13 - rank),
                    "relevance_grade": grade,
                    "judgment": judgment,
                    "relevance": relevance,
                    "is_judged": judged,
                }
            )
            passages.loc[len(passages)] = [docid, "title", f"passage {docid}"]
    dev = pd.DataFrame(dev_rows)
    dev["relevance_grade"] = dev["relevance_grade"].astype("Int64")
    dev["relevance"] = dev["relevance"].astype("Int64")
    passages_path = resolve_path(config, config["inputs"]["passages"])
    passages.to_parquet(passages_path, index=False)
    dev_candidates = resolve_path(config, config["dev_inputs"]["dev_candidates"])
    dev_candidates.parent.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(dev_candidates, index=False)

    train_qrels = resolve_path(config, config["inputs"]["train_qrels"])
    train_qrels.parent.mkdir(parents=True, exist_ok=True)
    train_qrels.write_text(
        "".join(
            f"{query_id}\t0\t{query_id}-d001\t1\n" for query_id in train_query_ids
        ),
        encoding="utf-8",
    )
    dev_qrels = resolve_path(config, config["dev_inputs"]["dev_qrels"])
    dev_qrels.write_text(
        "".join(
            f"{query_id}\t0\t{query_id}-d001\t1\n"
            f"{query_id}\t0\t{query_id}-d002\t0\n"
            for query_id in dev_query_ids
        ),
        encoding="utf-8",
    )

    _write_run(
        resolve_path(config, config["dev_inputs"]["bm25_run"]), dev, tag="bm25"
    )
    _write_run(
        resolve_path(config, config["dev_inputs"]["zeroshot_run"]),
        dev,
        tag="zero",
    )
    zero_scores = dev[["query_id", "docid"]].copy()
    zero_scores["score"] = (13 - dev["bm25_rank"]).astype("float32")
    zero_scores["pair_tokens_before_truncation"] = 12
    zero_scores["pair_tokens_after_truncation"] = 12
    zero_scores["truncated"] = False
    zero_scores_path = resolve_path(config, config["dev_inputs"]["zeroshot_scores"])
    zero_scores_path.parent.mkdir(parents=True, exist_ok=True)
    zero_scores.to_parquet(zero_scores_path, index=False)
    for key, payload in (
        ("zeroshot_metrics", {"pipeline_status": "PASS", "ndcg_at_10": 0.5365}),
        ("bm25_metrics", {"pipeline_status": "PASS", "ndcg_at_10": 0.3342}),
    ):
        path = resolve_path(config, config["dev_inputs"][key])
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload)

    declared = [
        {
            "path": str(relative),
            "sha256": sha256_file(resolve_path(config, relative)),
        }
        for relative in config["dev_inputs"].values()
    ]
    phase1_manifest = resolve_path(config, config["inputs"]["phase1_manifest"])
    atomic_write_json(phase1_manifest, {"schema_version": 1, "artifacts": declared})
    phase2_manifest = resolve_path(config, config["inputs"]["phase2_manifest"])
    phase2_manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(phase2_manifest, {"schema_version": 1, "artifacts": []})
    return config_path, config, dev


def _stub_checkpoint_payload(
    config: dict[str, object], *, run_id: str, epoch: int
) -> dict[str, bytes]:
    return {
        "model.safetensors": f"fixture weights {run_id} epoch {epoch}\n".encode(),
        "config.json": b'{"architectures":["FixtureSequenceClassifier"],"num_labels":1}\n',
        "tokenizer.json": b'{"fixture":true}\n',
        "tokenizer_config.json": b'{"model_max_length":320}\n',
        "special_tokens_map.json": b'{"unk_token":"<unk>"}\n',
        "sentencepiece.bpe.model": b"fixture sentencepiece payload\n",
    }


def test_full_phase3_stub_pipeline_packages_exact_payloads_and_preserves_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config, dev = _full_fixture(tmp_path)
    immutable_paths = {
        resolve_path(config, value)
        for value in (
            *config["inputs"].values(),
            *config["dev_inputs"].values(),
        )
    }
    immutable_before = {path: path.read_bytes() for path in immutable_paths}

    def assert_immutable() -> None:
        assert {path: path.read_bytes() for path in immutable_paths} == immutable_before

    def run_cli(*arguments: str) -> None:
        assert main([*arguments, "--config", str(config_path)]) == 0
        assert_immutable()

    run_cli("build-training-split")
    for regime in ("judged_only", "weak_negatives", "control_c1"):
        run_cli("build-training-pairs", "--regime", regime)

    dev_qrels_reads = 0
    original_load_qrels = phase3_eval_module.load_qrels

    def counted_load_qrels(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal dev_qrels_reads
        dev_qrels_reads += 1
        return original_load_qrels(*args, **kwargs)

    monkeypatch.setattr(phase3_eval_module, "load_qrels", counted_load_qrels)

    def stub_validate_checkpoint(
        stage_config, *, checkpoint="base", device=None  # type: ignore[no-untyped-def]
    ):
        assert checkpoint == "base"
        validation = pd.read_parquet(
            resolve_path(stage_config, stage_config["artifacts"]["validation_groups"])
        )
        per_query = {
            str(query_id): 0.60
            for query_id in sorted(validation["query_id"].astype("string").unique())
        }
        entry = {
            "kind": "S0",
            "epoch": 0,
            "ndcg_at_10": 0.60,
            "per_query": per_query,
            "checkpoint_sha256": "0" * 64,
            "sparse_diagnostics": {},
        }
        atomic_write_json(
            resolve_path(
                stage_config,
                stage_config["metrics"]["validation_checkpoint_metrics"],
            ),
            {"schema_version": 1, "S0": entry, "runs": {}},
        )
        return {"status": "PASS", **entry}

    def stub_smoke(stage_config, *, limit_pairs=64):  # type: ignore[no-untyped-def]
        report = {
            "status": "PASS",
            "fixture_only": True,
            "limit_pairs": int(limit_pairs),
            "note": "deterministic fixture smoke; production gate remains Colab-only",
        }
        atomic_write_json(
            resolve_path(stage_config, stage_config["audits"]["finetune_smoke"]),
            report,
        )
        atomic_write_json(
            resolve_path(stage_config, stage_config["audits"]["resource_report"]),
            {
                "fixture_only": True,
                "token_cache_bytes": 0,
                "training_seconds_range": [0.0, 0.0],
            },
        )
        return report

    validation_values = {
        "A1": (0.55, 0.54, 0.53),
        "A2": (0.54, 0.53, 0.52),
        "B1": (0.53, 0.52, 0.51),
    }

    def stub_finetune(
        stage_config, *, run_id, resume=False, overwrite=False  # type: ignore[no-untyped-def]
    ):
        metrics_path = resolve_path(
            stage_config,
            stage_config["metrics"]["validation_checkpoint_metrics"],
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        learning_rate = training_module.run_learning_rate(
            stage_config, run_id, metrics=metrics
        )
        components = training_module.training_fingerprint_components(
            stage_config, run_id=run_id, learning_rate=learning_rate
        )
        fingerprint = training_module.build_training_fingerprint(components)
        epochs = int(stage_config["runs"][run_id]["epochs"])
        history_epochs = []
        if run_id in validation_values:
            validation = pd.read_parquet(
                resolve_path(
                    stage_config,
                    stage_config["artifacts"]["validation_groups"],
                )
            )
            query_ids = sorted(validation["query_id"].astype("string").unique())
            entries = {}
            for epoch, ndcg in enumerate(validation_values[run_id], start=1):
                files = _stub_checkpoint_payload(
                    stage_config, run_id=run_id, epoch=epoch
                )
                weight_sha = hashlib.sha256(files["model.safetensors"]).hexdigest()
                generation_sha = hashlib.sha256(
                    f"fixture generation {run_id} {epoch}".encode()
                ).hexdigest()
                epoch_entry = {
                    "epoch": epoch,
                    "ndcg_at_10": ndcg,
                    "per_query": {str(query_id): ndcg for query_id in query_ids},
                    "sparse_diagnostics": {},
                    "checkpoint_sha256": weight_sha,
                    "model_generation_sha256": generation_sha,
                    "optimizer_steps": epoch,
                    "last_window_query_count": 1,
                    "throughput": {"queries_per_second": 1.0},
                    "step_logs": [],
                }
                entries[f"epoch_{epoch}"] = epoch_entry
                history_epochs.append(epoch_entry)
            metrics.setdefault("runs", {})[run_id] = entries
            atomic_write_json(metrics_path, metrics)
        else:
            atomic_write_json(
                resolve_path(stage_config, stage_config["audits"]["control_report"]),
                {
                    "status": "PASS",
                    "verdict": "PASS",
                    "fixture_only": True,
                    "pair_file_sha256": components["pair_file_sha256"],
                },
            )
        history_path = resolve_path(
            stage_config,
            str(stage_config["metrics"]["training_history_template"]).format(
                run_id=run_id
            ),
        )
        atomic_write_json(
            history_path,
            {
                "run_id": run_id,
                "fixture_only": True,
                "epochs": history_epochs,
            },
        )
        run_dir = (
            resolve_path(stage_config, stage_config["artifacts"]["models_dir"])
            / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            run_dir / "run_manifest.json",
            {
                "run_id": run_id,
                "regime": stage_config["runs"][run_id]["regime"],
                "learning_rate": learning_rate,
                "epochs": epochs,
                "training_fingerprint": fingerprint,
                "training_fingerprint_components": components,
                "finalized": True,
                "resume_available": False,
                "optimizer_steps": max(epochs, 1),
                "last_window_query_count": 1,
                "token_cache_bytes": 0,
                "token_cache_representation": "fixture_stub",
                "throughput": {"queries_per_second": 1.0},
                "validation_history": history_epochs,
                "peak_rss_bytes": 0,
                "peak_gpu_memory_bytes": 0,
                "started_at": "fixture",
                "completed_at": "fixture",
            },
        )
        return {
            "status": "PASS",
            "run_id": run_id,
            "pair_file_sha256": components["pair_file_sha256"],
        }

    monkeypatch.setattr(training_module, "validate_checkpoint", stub_validate_checkpoint)
    monkeypatch.setattr(training_module, "smoke_finetune", stub_smoke)
    monkeypatch.setattr(training_module, "run_finetune", stub_finetune)

    run_cli("validate-checkpoint", "--checkpoint", "base")
    run_cli("smoke-finetune", "--limit-pairs", "64")
    for run_id in ("C1", "A1", "A2", "B1"):
        run_cli("finetune", "--run-id", run_id)
    assert dev_qrels_reads == 0

    def stub_publish(
        stage_config,
        *,
        run_id,
        epoch,
        expected_generation_sha256,
        expected_weight_sha256,
        expected_training_fingerprint,
        overwrite,
    ):
        destination = resolve_path(
            stage_config, stage_config["artifacts"]["best_finetuned_dir"]
        )
        sidecar_path = destination / "checkpoint_sha256.json"
        if sidecar_path.is_file() and not overwrite:
            return json.loads(sidecar_path.read_text(encoding="utf-8"))
        payloads = _stub_checkpoint_payload(
            stage_config, run_id=run_id, epoch=int(epoch)
        )
        assert hashlib.sha256(payloads["model.safetensors"]).hexdigest() == (
            expected_weight_sha256
        )
        destination.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            (destination / name).write_bytes(payload)
        files = {name: sha256_file(destination / name) for name in payloads}
        sidecar = {
            "source_run_id": run_id,
            "source_epoch": int(epoch),
            "source_path": f"artifacts/models/{run_id}/epoch_{epoch}",
            "source_model_generation_sha256": expected_generation_sha256,
            "files": files,
            "checkpoint_payload_sha256": phase3_eval_module._checkpoint_payload_hash(
                files
            ),
            "training_fingerprint": expected_training_fingerprint,
            "published_at": "fixture",
        }
        atomic_write_json(sidecar_path, sidecar)
        return sidecar

    monkeypatch.setattr(phase3_eval_module, "_publish_best_finetuned", stub_publish)
    run_cli("select-checkpoint")
    selection_path = resolve_path(
        config, config["audits"]["checkpoint_selection"]
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["best_finetuned_checkpoint"]["run_id"] == "A1"
    assert selection["production_system"]["kind"] == "zero_shot"
    assert selection["zero_shot_won"] is True
    assert dev_qrels_reads == 0

    run_cli("prepare-dev-evaluation")
    assert dev_qrels_reads == 1

    def stub_score(stage_config, **kwargs):  # type: ignore[no-untyped-def]
        scores = dev[["query_id", "docid"]].copy()
        scores["score"] = (13 - dev["bm25_rank"]).astype("float32") + 0.25
        scores["pair_tokens_before_truncation"] = 12
        scores["pair_tokens_after_truncation"] = 12
        scores["truncated"] = False
        destination = tmp_path / stage_config["artifacts"]["scores"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(destination, index=False)
        checkpoint = resolve_path(
            config, config["artifacts"]["best_finetuned_dir"]
        )
        components = {
            name: f"fixture:{name}"
            for name in rerank_module.SCORING_FINGERPRINT_FIELDS
        }
        components["checkpoint_sha256"] = (
            phase3_eval_module._checkpoint_scoring_hash(checkpoint)
        )
        atomic_write_json(
            destination.with_name(f"{destination.name}.json"),
            {
                "status": "PASS",
                "scores_sha256": sha256_file(destination),
                "fingerprint_components": components,
                "input_fingerprint": rerank_module.build_input_fingerprint(
                    components
                ),
            },
        )
        return {"status": "PASS", "fixture_only": True, "rows": len(scores)}

    def stub_run(stage_config, **kwargs):  # type: ignore[no-untyped-def]
        destination = tmp_path / stage_config["artifacts"]["rerank_run"]
        _write_run(destination, dev, tag="fine")
        return {"status": "PASS", "fixture_only": True, "rows": len(dev)}

    monkeypatch.setattr(rerank_module, "run_rerank_scoring", stub_score)
    monkeypatch.setattr(rerank_module, "build_rerank_run", stub_run)
    monkeypatch.setattr(
        rerank_module,
        "validate_current_score_sidecar",
        lambda stage_config, *, score_path: json.loads(
            Path(score_path)
            .with_name(f"{Path(score_path).name}.json")
            .read_text(encoding="utf-8")
        ),
    )
    run_cli("score-finetuned")

    query_ids = sorted(dev["query_id"].unique())
    metric_values = {"bm25": 0.3342, "zero": 0.5365, "fine": 0.55}

    def system_name_from_path(path: Path) -> str:
        name = path.name
        if "bm25" in name:
            return "bm25"
        if "zeroshot" in name:
            return "zero"
        return "fine"

    def stub_official(stage_config, path):  # type: ignore[no-untyped-def]
        system = system_name_from_path(Path(path))
        value = metric_values[system]
        return {
            "ndcg_at_10": value,
            "recall_at_100": 1.0,
            "mrr_at_10": value,
            "per_query_ndcg_at_10": {query_id: value for query_id in query_ids},
            "commands": {"fixture": True},
        }

    def stub_python_metric(ranking, qrels, *, query_ids):  # type: ignore[no-untyped-def]
        tag = str(ranking["tag"].iloc[0])
        value = metric_values[tag]
        ids = [str(value) for value in query_ids]
        return {
            "query_count": len(ids),
            "ndcg_at_10": value,
            "per_query": {query_id: value for query_id in ids},
        }

    monkeypatch.setattr(phase3_eval_module, "_official_metrics", stub_official)
    monkeypatch.setattr(
        phase3_eval_module, "evaluate_ranked_ndcg_at_10", stub_python_metric
    )
    monkeypatch.setattr(rerank_module, "resolve_trec_eval", lambda config: Path("fixture"))
    monkeypatch.setattr(
        rerank_module,
        "validate_trec_eval_build_provenance",
        lambda *args, **kwargs: {"status": "PASS", "fixture_only": True},
    )
    run_cli("evaluate-phase3")
    run_cli("evaluate-phase3")
    assert dev_qrels_reads == 1

    model_forward_checks = 0

    def stub_model_forward(archive_path: Path) -> None:
        nonlocal model_forward_checks
        model_forward_checks += 1
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.namelist() == list(phase3_eval_module.MODEL_ZIP_MEMBERS)

    monkeypatch.setattr(
        phase3_eval_module, "_validate_model_zip_forward", stub_model_forward
    )
    run_cli("package-phase3")
    result_zip = resolve_path(config, config["archive"]["results_zip"])
    model_zip = resolve_path(
        config,
        str(config["archive"]["model_zip_template"]).format(run_id="A1"),
    )
    assert model_forward_checks == 1
    assert result_zip.is_file() and model_zip.is_file()

    with zipfile.ZipFile(result_zip) as archive:
        assert archive.namelist() == list(phase3_eval_module.RESULT_ZIP_MEMBERS)
        manifest = json.loads(
            archive.read("reports/audit/training_manifest.json").decode("utf-8")
        )
        manifest_paths = [entry["path"] for entry in manifest["entries"]]
        assert "reports/audit/training_manifest.json" not in manifest_paths
        assert manifest["self_reference"] is False
        for name in archive.namelist():
            assert archive.read(name) == (tmp_path / name).read_bytes()
    with zipfile.ZipFile(model_zip) as archive:
        assert archive.namelist() == list(phase3_eval_module.MODEL_ZIP_MEMBERS)
        for name in archive.namelist():
            source = (
                resolve_path(config, config["audits"]["model_card"])
                if name == "model_card.md"
                else resolve_path(
                    config, config["artifacts"]["best_finetuned_dir"]
                )
                / name
            )
            assert archive.read(name) == source.read_bytes()
    assert resolve_path(
        config, config["audits"]["protocol_snapshot"]
    ).read_bytes() == config_path.read_bytes()

    ledger = [
        json.loads(line)
        for line in resolve_path(
            config, config["audits"]["dev_access_ledger"]
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [event["command"] for event in ledger] == [
        "prepare-dev-evaluation",
        "score-finetuned",
        "evaluate-phase3",
        "evaluate-phase3",
    ]
    assert len({event["checkpoint_sha256"] for event in ledger}) == 1
    assert [event["repeat_access"] for event in ledger] == [
        False,
        True,
        True,
        True,
    ]

    packaged_before = {path: path.read_bytes() for path in (result_zip, model_zip)}
    run_cli("package-phase3")
    assert {path: path.read_bytes() for path in packaged_before} == packaged_before
    assert model_forward_checks == 2
