from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

import rusearchrank.cli as cli_module
import rusearchrank.rerank as rerank
from rusearchrank.cli import main


class FixtureTokenizer:
    @staticmethod
    def _tokens(text: str) -> list[int]:
        return [10 + ord(character) % 211 for character in text]

    def __call__(
        self,
        query: str,
        document: str,
        *,
        truncation: bool | str,
        add_special_tokens: bool,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        query_tokens = self._tokens(query)
        document_tokens = self._tokens(document)
        if truncation == "only_second" and max_length is not None:
            document_tokens = document_tokens[: max_length - len(query_tokens) - 4]
        ids = [0, *query_tokens, 2, 2, *document_tokens, 2]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(
        self,
        encoded: list[dict[str, list[int]]],
        *,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        length = max(len(row["input_ids"]) for row in encoded)
        return {
            name: torch.tensor(
                [row[name] + [0] * (length - len(row[name])) for row in encoded],
                dtype=torch.int64,
            )
            for name in ("input_ids", "attention_mask")
        }


class StubScorer:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.tokenizer = FixtureTokenizer()
        self.calls = 0
        self.fail_on_call = fail_on_call

    def score_batch(
        self,
        encoded_pairs: list[dict[str, list[int]]],
        *,
        device: str,
        dtype: str,
    ) -> tuple[np.ndarray, int]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("intentional fixture interruption")
        padded = max(len(row["input_ids"]) for row in encoded_pairs)
        scores = np.asarray(
            [
                (sum(row["input_ids"]) % 1009) / 100.0
                + len(row["input_ids"]) / 10000.0
                for row in encoded_pairs
            ],
            dtype=np.float32,
        )
        return scores, len(encoded_pairs) * padded


def _write_fake_trec_eval(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import math, sys
args = sys.argv[1:]
if '-v' in args:
    print('trec_eval version 9.0.7')
    raise SystemExit(0)
metric = args[args.index('-m') + 1]
per_query = '-q' in args
qrels_path, run_path = args[-2:]
qrels = {}
with open(qrels_path, encoding='utf-8') as stream:
    for line in stream:
        qid, _, docid, grade = line.split()
        qrels.setdefault(qid, {})[docid] = int(grade)
runs = {}
with open(run_path, encoding='utf-8') as stream:
    for line in stream:
        qid, _, docid, rank, score, tag = line.split()
        runs.setdefault(qid, []).append((float(score), docid))
for qid in runs:
    runs[qid].sort(key=lambda row: (row[0], row[1]), reverse=True)
def dcg(grades):
    return sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades[:10], 1))
values = {}
for qid in qrels:
    docs = [docid for _, docid in runs.get(qid, [])]
    grades = [qrels[qid].get(docid, 0) for docid in docs]
    if metric == 'ndcg_cut.10':
        ideal = dcg(sorted(qrels[qid].values(), reverse=True))
        value = dcg(grades) / ideal if ideal else 0.0
        label = 'ndcg_cut_10'
    elif metric == 'recall.100':
        relevant = sum(grade > 0 for grade in qrels[qid].values())
        found = sum(grade > 0 for grade in grades[:100])
        value = found / relevant if relevant else 0.0
        label = 'recall_100'
    elif metric == 'recip_rank':
        positions = [index for index, grade in enumerate(grades[:10], 1) if grade > 0]
        value = 1.0 / positions[0] if positions else 0.0
        label = 'recip_rank'
    else:
        raise SystemExit('unsupported metric ' + metric)
    values[qid] = value
    if per_query:
        print(label, qid, f'{value:.4f}')
print(label, 'all', f'{sum(values.values()) / len(values):.4f}')
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _metadata(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": None,
    }


def build_fixture(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    for directory in (
        "configs",
        "src/rusearchrank",
        "artifacts/candidates",
        "artifacts/runs",
        "artifacts/raw/miracl-ru",
        "reports/audit",
        "reports/metrics",
    ):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    for relative in (
        "src/rusearchrank/rerank.py",
        "src/rusearchrank/cli.py",
        "src/rusearchrank/evaluation.py",
        "pyproject.toml",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)

    rows: list[dict[str, object]] = []
    passages: dict[str, tuple[str, str]] = {}
    for query_number, query_id in enumerate(("q1", "q2"), start=1):
        for rank in range(1, 13):
            docid = f"d{query_number}{rank:02d}"
            grade: int | None = 1 if rank == 3 else 0 if rank in {2, 7} else None
            rows.append(
                {
                    "split": "dev",
                    "query_id": query_id,
                    "docid": docid,
                    "bm25_rank": rank,
                    "bm25_score": float(100 - rank),
                    "relevance_grade": grade,
                    "judgment": (
                        "relevant"
                        if grade == 1
                        else "judged_non_relevant"
                        if grade == 0
                        else "unjudged"
                    ),
                    "relevance": grade,
                    "is_judged": grade is not None,
                }
            )
            passages[docid] = (
                "" if rank == 4 else f"Заголовок {docid}",
                f"Русский текст документа {docid} с рангом {rank}",
            )
    candidates = pd.DataFrame(rows)
    candidate_path = tmp_path / "artifacts/candidates/dev_top100.parquet"
    candidates.to_parquet(candidate_path, index=False)
    queries = pd.DataFrame(
        {
            "split": ["dev", "dev"],
            "query_id": ["q1", "q2"],
            "query_text": ["первый запрос", "второй\xa0запрос"],
        }
    )
    queries.to_parquet(tmp_path / "artifacts/candidates/queries.parquet", index=False)
    passage_rows = [
        {"docid": docid, "title": title, "text": text}
        for docid, (title, text) in sorted(passages.items())
    ] + [
        {"docid": f"noise-{index}", "title": "noise", "text": "never materialized"}
        for index in range(20)
    ]
    pd.DataFrame(passage_rows).to_parquet(
        tmp_path / "artifacts/candidates/passages.parquet", index=False
    )
    run_lines = [
        f"{row['query_id']} Q0 {row['docid']} {row['bm25_rank']} "
        f"{row['bm25_score']:.8f} fixture"
        for row in rows
    ]
    for name in ("dev_bm25_top100.trec", "dev_bm25_top1000.trec"):
        (tmp_path / f"artifacts/runs/{name}").write_text(
            "\n".join(run_lines) + "\n", encoding="utf-8"
        )
    qrels_path = tmp_path / "artifacts/raw/miracl-ru/qrels.miracl-v1.0-ru-dev.tsv"
    qrels_path.write_text(
        "q1\t0\td103\t1\nq1\t0\td102\t0\nq1\t0\td107\t0\n"
        "q2\t0\td203\t1\nq2\t0\td202\t0\nq2\t0\td207\t0\n",
        encoding="utf-8",
    )
    manifest_relatives = [
        "artifacts/candidates/dev_top100.parquet",
        "artifacts/candidates/queries.parquet",
        "artifacts/candidates/passages.parquet",
        "artifacts/runs/dev_bm25_top100.trec",
        "artifacts/runs/dev_bm25_top1000.trec",
    ]
    manifest = {
        "status": "PASS",
        "artifacts": [_metadata(tmp_path, relative) for relative in manifest_relatives],
    }
    (tmp_path / "reports/audit/candidate_cache_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    fake_trec = tmp_path / "trec_eval"
    _write_fake_trec_eval(fake_trec)

    config = {
        "implementation": {
            "version": "2.0.0",
            "score_schema_version": 1,
            "source_files": [
                "src/rusearchrank/rerank.py",
                "src/rusearchrank/cli.py",
                "src/rusearchrank/evaluation.py",
                "configs/rerank.yaml",
                "pyproject.toml",
            ],
        },
        "model": {
            "id": rerank.MODEL_ID,
            "revision": rerank.MODEL_REVISION,
            "tokenizer_revision": rerank.MODEL_REVISION,
            "tag": rerank.MODEL_TAG,
            "num_labels": 1,
            "score_activation": "identity",
            "score_direction": "higher_is_more_relevant",
        },
        "input": {
            "max_length": 64,
            "truncation": "only_second",
            "title_separator": "\n",
            "pair_order": "query_document",
        },
        "inference": {
            "batch_size": 16,
            "cpu_batch_size": 16,
            "device": "auto",
            "cuda_dtype": "float16",
            "fallback_dtype": "float32",
            "score_dtype": "float32",
            "shard_queries": 1,
            "seed": 20260802,
            "minimum_available_memory_gib": 1,
        },
        "protocol": {
            "split": "dev",
            "official_depth": 100,
            "diagnostic_depths": [10, 20, 50],
            "internal_tie_break": "raw_score_desc_then_docid_asc",
            "trec_score_encoding": "rank_preserving",
            "trec_score_base": 1000000,
            "trec_score_format": "%.4f",
        },
        "inputs": {
            "candidates": "artifacts/candidates/dev_top100.parquet",
            "queries": "artifacts/candidates/queries.parquet",
            "passages": "artifacts/candidates/passages.parquet",
            "bm25_run": "artifacts/runs/dev_bm25_top100.trec",
            "bm25_top1000_run": "artifacts/runs/dev_bm25_top1000.trec",
            "qrels": "artifacts/raw/miracl-ru/qrels.miracl-v1.0-ru-dev.tsv",
            "phase1_manifest": "reports/audit/candidate_cache_manifest.json",
        },
        "artifacts": {
            "scores": "artifacts/scores/dev_zeroshot_mminilmv2l12.parquet",
            "partial_dir": "artifacts/scores/_partial",
            "rerank_run": "artifacts/runs/dev_rerank_zeroshot_k100.trec",
            "diagnostic_run_template": "artifacts/runs/dev_rerank_zeroshot_k{depth}.trec",
        },
        "metrics": {
            "baseline": "reports/metrics/dev_bm25_baseline.json",
            "system": "reports/metrics/dev_zeroshot_mminilmv2l12.json",
            "comparison": "reports/metrics/dev_zeroshot_vs_bm25.json",
            "depth_profile": "reports/metrics/dev_zeroshot_depth_profile.json",
        },
        "audits": {
            "manifest": "reports/audit/rerank_manifest.json",
            "protocol_snapshot": "reports/audit/rerank_protocol.yaml",
            "smoke": "reports/audit/rerank_smoke.json",
        },
        "evaluation": {
            "trec_eval_executable": str(fake_trec),
            "trec_eval_expected_release": "9.0.8",
            "trec_eval_expected_source_tag": "v9.0.8",
            "trec_eval_expected_source_commit": "1" * 40,
            "trec_eval_expected_reported_version": "9.0.7",
            "trec_eval_known_version_string_mismatch": True,
            "trec_eval_provenance_path": (
                "artifacts/work/phase2/trec_eval_build_provenance.json"
            ),
            "ndcg_command": ["-c", "-M", "100", "-m", "ndcg_cut.10"],
            "recall_command": ["-c", "-m", "recall.100"],
            "per_query_command": ["-c", "-M", "100", "-q", "-m", "ndcg_cut.10"],
            "mrr_command": ["-c", "-M", "10", "-m", "recip_rank"],
            "python_vs_trec_eval_tolerance": 0.0001,
            "bootstrap_resamples": 200,
            "bootstrap_seed": 20260802,
            "bootstrap_confidence": 0.95,
            "expected_bm25_ndcg_at_10": 0.3342,
            "expected_bm25_recall_at_100": 0.6614,
            "expected_recall_invariant": "strict_set_equality",
        },
        "archive": {"path": "artifacts/rusearchrank_phase2_results.zip"},
        "paths": {"repository_root": "..", "work_dir": "artifacts/work/phase2"},
    }
    config_path = tmp_path / "configs/rerank.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    provenance_path = (
        tmp_path / "artifacts/work/phase2/trec_eval_build_provenance.json"
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(
            {
                "source_repository": rerank.TREC_EVAL_SOURCE_REPOSITORY,
                "source_tag": "v9.0.8",
                "source_commit": "1" * 40,
                "source_tree_clean": True,
                "fresh_checkout": True,
                "makefile_sha256": "3" * 64,
                "binary_path": str(fake_trec.resolve()),
                "binary_sha256": hashlib.sha256(fake_trec.read_bytes()).hexdigest(),
                "binary_reported_version": "9.0.7",
                "expected_release_version": "9.0.8",
                "known_upstream_version_string_mismatch": True,
                "build_command": "make -j2",
                "compiler": "fixture cc 1.0",
                "built_at": "2026-08-02T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = rerank.load_rerank_config(config_path)
    smoke = {
        "status": "PASS",
        "real_model_forward": True,
        "fixture_only": False,
        **rerank.smoke_expected_fields(loaded),
    }
    rerank.atomic_write_json(tmp_path / "reports/audit/rerank_smoke.json", smoke)
    phase1_paths = [
        *(tmp_path / relative for relative in manifest_relatives),
        qrels_path,
        tmp_path / "reports/audit/candidate_cache_manifest.json",
    ]
    return config_path, {str(path): path.read_bytes() for path in phase1_paths}


def _score_args(config_path: Path, *, overwrite: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config_path),
        split="dev",
        device="cpu",
        batch_size=None,
        overwrite=overwrite,
    )


def _evaluation_args(
    config_path: Path, *, overwrite: bool = False
) -> argparse.Namespace:
    return argparse.Namespace(config=str(config_path), split="dev", overwrite=overwrite)


def _prepare_fixture_evaluation(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], dict[str, Path], dict[str, bytes]]:
    config_path, _ = build_fixture(tmp_path)
    config = rerank.load_rerank_config(config_path)
    rerank.run_rerank_scoring(
        config, split="dev", requested_device="cpu", scorer=StubScorer()
    )
    for depth in (100, 10, 20, 50):
        rerank.build_rerank_run(config, split="dev", depth=depth)
    assert cli_module._evaluate_rerank(_evaluation_args(config_path)) == 0
    metric_paths = {
        name: rerank.resolve_path(config, config["metrics"][name])
        for name in ("baseline", "system", "comparison", "depth_profile")
    }
    protected_paths = [
        rerank.resolve_path(config, config["artifacts"]["scores"]),
        *(rerank.rerank_run_path(config, depth) for depth in (10, 20, 50, 100)),
    ]
    protected_bytes = {str(path): path.read_bytes() for path in protected_paths}
    return config_path, config, metric_paths, protected_bytes


def test_full_fixture_pipeline_resume_evaluation_package_and_idempotence(
    tmp_path: Path,
) -> None:
    config_path, phase1_before = build_fixture(tmp_path)
    config = rerank.load_rerank_config(config_path)

    with pytest.raises(RuntimeError, match="intentional fixture interruption"):
        rerank.run_rerank_scoring(
            config,
            split="dev",
            requested_device="cpu",
            scorer=StubScorer(fail_on_call=2),
        )
    partial = tmp_path / "artifacts/scores/_partial"
    assert (partial / "shard_00000.parquet").is_file()
    report = rerank.run_rerank_scoring(
        config,
        split="dev",
        requested_device="cpu",
        scorer=StubScorer(),
    )
    assert report["reused_shards"] == 1
    assert report["scored_shards"] == 1
    assert report["token_accounting"]["processed_pair_count"] == 24
    assert report["token_accounting"]["processed_tokens"] <= 24 * 64
    assert not partial.exists()

    for depth in (100, 10, 20, 50):
        assert (
            main(
                [
                    "build-rerank-run",
                    "--config",
                    str(config_path),
                    "--split",
                    "dev",
                    "--depth",
                    str(depth),
                ]
            )
            == 0
        )
    assert (
        main(
            [
                "evaluate-rerank",
                "--config",
                str(config_path),
                "--split",
                "dev",
            ]
        )
        == 0
    )
    assert main(["package-phase2", "--config", str(config_path)]) == 0

    depth_profile = json.loads(
        (tmp_path / "reports/metrics/dev_zeroshot_depth_profile.json").read_text()
    )
    assert [entry["depth"] for entry in depth_profile["depths"]] == [10, 20, 50, 100]
    assert all(entry["candidate_set_invariant"] for entry in depth_profile["depths"])
    assert depth_profile["depths"][-1]["official"] is True
    system_metrics = json.loads(
        (tmp_path / "reports/metrics/dev_zeroshot_mminilmv2l12.json").read_text()
    )
    for field in (
        "raw_score_tie_definition",
        "raw_score_tie_groups",
        "rows_in_raw_score_ties",
        "queries_with_any_raw_score_tie",
        "queries_with_top10_raw_score_tie",
        "ties_crossing_rank10_boundary",
        "largest_raw_score_tie_group",
        "scoring_source_sha256",
        "evaluation_source_sha256",
    ):
        assert field in system_metrics
    version_probe = system_metrics["trec_eval"]["version_probe"]
    assert version_probe["command"][-1] == "-v"
    assert version_probe["binary_reported_version"] == "9.0.7"
    assert version_probe["binary_reported_version_matches_expected"] is True
    assert system_metrics["trec_eval_provenance"][
        "expected_release_version"
    ] == "9.0.8"
    baseline_metrics = json.loads(
        (tmp_path / "reports/metrics/dev_bm25_baseline.json").read_text()
    )
    assert baseline_metrics["python_cross_check"]["tie_break"]["conclusion"] == (
        "inconclusive_metric_equivalent"
    )
    strata = json.loads(
        (tmp_path / "reports/metrics/dev_zeroshot_vs_bm25.json").read_text()
    )["stratified_mean_delta"]
    assert strata["invariants"]["each_query_in_exactly_one_stratum"] is True
    assert strata["invariants"]["stratum_query_count_sum"] == 2
    manifest_path = tmp_path / "reports/audit/rerank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "PASS"
    assert all(entry["path"] != "reports/audit/rerank_manifest.json" for entry in manifest["files"])
    assert "manifest_sha256" not in manifest
    assert all(
        entry["scoring_source_sha256"]
        and entry["evaluation_source_sha256"]
        and "raw_score_tie_definition" in entry
        and "score_producer_commit" in entry
        and "evaluation_commit" in entry
        and "package_commit" in entry
        for entry in manifest["files"]
    )
    protocol = tmp_path / "reports/audit/rerank_protocol.yaml"
    assert protocol.read_bytes() == config_path.read_bytes()
    archive_path = tmp_path / "artifacts/rusearchrank_phase2_results.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert archive.namelist()[-1] == "reports/audit/rerank_manifest.json"
        extraction = tmp_path / "extracted"
        archive.extractall(extraction)
    for entry in manifest["files"]:
        extracted = extraction / entry["path"]
        assert hashlib.sha256(extracted.read_bytes()).hexdigest() == entry["sha256"]

    outputs = [
        tmp_path / "artifacts/scores/dev_zeroshot_mminilmv2l12.parquet",
        *(tmp_path / f"artifacts/runs/dev_rerank_zeroshot_k{depth}.trec" for depth in (10, 20, 50, 100)),
        *(tmp_path / f"reports/metrics/{name}" for name in (
            "dev_bm25_baseline.json",
            "dev_zeroshot_mminilmv2l12.json",
            "dev_zeroshot_vs_bm25.json",
            "dev_zeroshot_depth_profile.json",
        )),
        protocol,
        manifest_path,
        archive_path,
    ]
    before = {str(path): path.read_bytes() for path in outputs}
    reused = rerank.run_rerank_scoring(
        config,
        split="dev",
        requested_device="cpu",
        scorer=StubScorer(),
    )
    assert reused["action"] == "reused_valid_scores"
    for depth in (10, 20, 50, 100):
        assert main([
            "build-rerank-run", "--config", str(config_path), "--split", "dev", "--depth", str(depth)
        ]) == 0
    assert main(["evaluate-rerank", "--config", str(config_path), "--split", "dev"]) == 0
    assert main(["package-phase2", "--config", str(config_path)]) == 0
    assert all(path.read_bytes() == before[str(path)] for path in outputs)
    assert all(Path(path).read_bytes() == content for path, content in phase1_before.items())


def test_trec_eval_provenance_preflight_failure_preserves_metrics(
    tmp_path: Path,
) -> None:
    config_path, config, metric_paths, protected = _prepare_fixture_evaluation(tmp_path)
    before = {name: path.read_bytes() for name, path in metric_paths.items()}
    provenance_path = rerank.resolve_path(
        config, config["evaluation"]["trec_eval_provenance_path"]
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_commit"] = "2" * 40
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")
    with pytest.raises(cli_module.StageError, match="evaluate-rerank/preflight"):
        cli_module._evaluate_rerank(_evaluation_args(config_path, overwrite=True))
    assert all(path.read_bytes() == before[name] for name, path in metric_paths.items())
    assert not list((tmp_path / "reports/metrics").glob("*.stale.*"))
    assert all(Path(path).read_bytes() == value for path, value in protected.items())


def test_qrels_preflight_failure_preserves_metrics(tmp_path: Path) -> None:
    config_path, config, metric_paths, protected = _prepare_fixture_evaluation(tmp_path)
    before = {name: path.read_bytes() for name, path in metric_paths.items()}
    qrels_path = rerank.resolve_path(config, config["inputs"]["qrels"])
    qrels_path.write_text(qrels_path.read_text() + "broken\n", encoding="utf-8")
    with pytest.raises(cli_module.StageError, match="evaluate-rerank/preflight"):
        cli_module._evaluate_rerank(_evaluation_args(config_path, overwrite=True))
    assert all(path.read_bytes() == before[name] for name, path in metric_paths.items())
    assert all(Path(path).read_bytes() == value for path, value in protected.items())


@pytest.mark.parametrize("failed_report", [2, 3, 4])
def test_temporary_report_failure_preserves_production_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_report: int,
) -> None:
    config_path, _, metric_paths, protected = _prepare_fixture_evaluation(tmp_path)
    before = {name: path.read_bytes() for name, path in metric_paths.items()}
    original_write = cli_module._write_json
    calls = 0

    def failing_write(path: Path, payload: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == failed_report:
            raise RuntimeError(f"fixture report {failed_report} failure")
        original_write(path, payload)

    monkeypatch.setattr(cli_module, "_write_json", failing_write)
    with pytest.raises(
        cli_module.StageError, match="temporary-calculation-validation"
    ):
        cli_module._evaluate_rerank(_evaluation_args(config_path, overwrite=True))
    assert all(path.read_bytes() == before[name] for name, path in metric_paths.items())
    assert all(Path(path).read_bytes() == value for path, value in protected.items())


def test_publication_failure_rolls_back_complete_metrics_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _, metric_paths, protected = _prepare_fixture_evaluation(tmp_path)
    before = {name: path.read_bytes() for name, path in metric_paths.items()}
    real_replace = os.replace
    published = 0
    failed = False

    def failing_replace(source: object, destination: object) -> None:
        nonlocal published, failed
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name in {"baseline.json", "system.json", "comparison.json", "depth_profile.json"}:
            published += 1
            if published == 2 and not failed:
                failed = True
                raise OSError("fixture failure after first published report")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(cli_module.os, "replace", failing_replace)
    with pytest.raises(cli_module.StageError, match='"rollback_status": "PASS"'):
        cli_module._evaluate_rerank(_evaluation_args(config_path, overwrite=True))
    assert all(path.read_bytes() == before[name] for name, path in metric_paths.items())
    assert all(Path(path).read_bytes() == value for path, value in protected.items())


def test_successful_overwrite_publishes_one_consistent_generation(
    tmp_path: Path,
) -> None:
    config_path, config, metric_paths, _ = _prepare_fixture_evaluation(tmp_path)
    before = {name: path.read_bytes() for name, path in metric_paths.items()}
    assert cli_module._evaluate_rerank(
        _evaluation_args(config_path, overwrite=True)
    ) == 0
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in metric_paths.items()
    }
    assert len({report["evaluation_fingerprint"] for report in reports.values()}) == 1
    assert len({json.dumps(report["trec_eval_provenance"], sort_keys=True) for report in reports.values()}) == 1
    assert any(path.read_bytes() != before[name] for name, path in metric_paths.items())
    backup_dirs = list(
        rerank.resolve_path(config, config["paths"]["work_dir"]).glob("metrics-backup-*")
    )
    assert len(backup_dirs) == 1
    assert {
        path.name: path.read_bytes() for path in backup_dirs[0].glob("*.json")
    } == {
        metric_paths[name].name: value for name, value in before.items()
    }


def test_evaluation_recreates_missing_active_metrics_without_using_stale_backups(
    tmp_path: Path,
) -> None:
    config_path, _, metric_paths, protected = _prepare_fixture_evaluation(tmp_path)
    stale_paths: dict[str, Path] = {}
    for name, active in metric_paths.items():
        stale = active.with_name(f"{active.name}.stale.fixture")
        active.replace(stale)
        stale.write_bytes(f"invalid stale backup for {name}\n".encode())
        stale_paths[name] = stale
    assert not any(path.exists() for path in metric_paths.values())

    assert cli_module._evaluate_rerank(
        _evaluation_args(config_path, overwrite=True)
    ) == 0

    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in metric_paths.items()
    }
    assert all(report["status"] == "PASS" for report in reports.values())
    assert len({report["evaluation_fingerprint"] for report in reports.values()}) == 1
    assert all(
        path.read_bytes() == f"invalid stale backup for {name}\n".encode()
        for name, path in stale_paths.items()
    )
    assert all(Path(path).read_bytes() == value for path, value in protected.items())


@pytest.mark.parametrize("as_stale", [False, True])
def test_package_rejects_incomplete_metrics_without_touching_outputs(
    tmp_path: Path, as_stale: bool
) -> None:
    _, config, metric_paths, _ = _prepare_fixture_evaluation(tmp_path)
    missing = metric_paths["baseline"]
    if as_stale:
        missing.replace(missing.with_name(f"{missing.name}.stale.fixture"))
    else:
        missing.unlink()
    manifest = rerank.resolve_path(config, config["audits"]["manifest"])
    archive = rerank.resolve_path(config, config["archive"]["path"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(b"old-manifest")
    archive.write_bytes(b"old-zip")
    with pytest.raises(ValueError, match="evaluation outputs are incomplete") as error:
        rerank.package_phase2(config, overwrite=True)
    assert "*.stale.* files are backups" in str(error.value)
    assert manifest.read_bytes() == b"old-manifest"
    assert archive.read_bytes() == b"old-zip"


@pytest.mark.parametrize("damage", ["version", "byte", "row", "null"])
def test_invalid_partial_shard_is_preserved_and_recomputed(
    tmp_path: Path, damage: str, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, _ = build_fixture(tmp_path)
    config = rerank.load_rerank_config(config_path)
    with pytest.raises(RuntimeError):
        rerank.run_rerank_scoring(
            config,
            split="dev",
            requested_device="cpu",
            scorer=StubScorer(fail_on_call=2),
        )
    shard = tmp_path / "artifacts/scores/_partial/shard_00000.parquet"
    sidecar = tmp_path / "artifacts/scores/_partial/shard_00000.json"
    if damage == "version":
        value = json.loads(sidecar.read_text())
        value["implementation_version"] = "old"
        sidecar.write_text(json.dumps(value), encoding="utf-8")
    elif damage == "byte":
        data = bytearray(shard.read_bytes())
        data[-8] ^= 1
        shard.write_bytes(data)
    else:
        table = pq.read_table(shard)
        if damage == "row":
            table = table.slice(0, table.num_rows - 1)
        else:
            nullable_schema = pa.schema(
                [pa.field(field.name, field.type, nullable=True) for field in rerank.SCORE_SCHEMA]
            )
            rows = table.to_pylist()
            rows[0]["score"] = None
            table = pa.Table.from_pylist(rows, schema=nullable_schema)
        pq.write_table(table, shard)
        value = json.loads(sidecar.read_text())
        value["shard_sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(value), encoding="utf-8")
    result = rerank.run_rerank_scoring(
        config,
        split="dev",
        requested_device="cpu",
        scorer=StubScorer(),
    )
    assert result["scored_shards"] >= 1
    assert "preserved incompatible artifact" in capsys.readouterr().out
    # The completed partial directory is removed only after final validation.
    assert result["status"] == "PASS"


def test_stale_final_fingerprint_requires_overwrite_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    config_path, _ = build_fixture(tmp_path)
    config = rerank.load_rerank_config(config_path)
    rerank.run_rerank_scoring(
        config, split="dev", requested_device="cpu", scorer=StubScorer()
    )
    scores = tmp_path / "artifacts/scores/dev_zeroshot_mminilmv2l12.parquet"
    before = scores.read_bytes()
    source = tmp_path / "src/rusearchrank/rerank.py"
    original_source = source.read_text(encoding="utf-8")
    changed_source = original_source.replace(
        'return f"{title}{separator}{text}" if title and title.strip() else text',
        'return (f"{title}{separator}{text}" if title and title.strip() else text)',
        1,
    )
    assert changed_source != original_source
    source.write_text(changed_source, encoding="utf-8")
    with pytest.raises(ValueError, match="stale or invalid"):
        rerank.run_rerank_scoring(
            config, split="dev", requested_device="cpu", scorer=StubScorer()
        )
    assert scores.read_bytes() == before
    result = rerank.run_rerank_scoring(
        config,
        split="dev",
        requested_device="cpu",
        overwrite=True,
        scorer=StubScorer(),
    )
    assert result["status"] == "PASS"
    assert list(scores.parent.glob(f"{scores.name}.stale.*"))


def test_passages_are_materialized_only_via_shard_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _ = build_fixture(tmp_path)
    config = rerank.load_rerank_config(config_path)
    passage_path = tmp_path / "artifacts/candidates/passages.parquet"
    real_read = rerank.pq.read_table
    counts = {"takes": 0, "max_rows": 0}

    class PassageProxy:
        def __init__(self, table: pa.Table) -> None:
            self.table = table
            self.schema = table.schema

        def column(self, name: str) -> pa.ChunkedArray:
            if name in {"title", "text"}:
                raise AssertionError("full passage text column was materialized")
            return self.table.column(name)

        def take(self, indices: pa.Array) -> pa.Table:
            counts["takes"] += 1
            counts["max_rows"] = max(counts["max_rows"], len(indices))
            return self.table.take(indices)

    def instrumented_read(path: object, *args: object, **kwargs: object) -> object:
        table = real_read(path, *args, **kwargs)
        raw_path = path if isinstance(path, (str, os.PathLike)) else getattr(path, "name", path)
        return (
            PassageProxy(table)
            if isinstance(raw_path, (str, os.PathLike))
            and Path(raw_path).resolve() == passage_path.resolve()
            else table
        )

    monkeypatch.setattr(rerank.pq, "read_table", instrumented_read)
    rerank.run_rerank_scoring(
        config, split="dev", requested_device="cpu", scorer=StubScorer()
    )
    assert counts == {"takes": 2, "max_rows": 12}


def test_cli_smoke_gate_rejects_missing_fixture_and_stale_reports(
    tmp_path: Path,
) -> None:
    config_path, _ = build_fixture(tmp_path)
    smoke_path = tmp_path / "reports/audit/rerank_smoke.json"
    smoke_path.unlink()
    with pytest.raises(cli_module.StageError, match="smoke report is missing"):
        cli_module._rerank_score(_score_args(config_path), scorer=StubScorer())
    config = rerank.load_rerank_config(config_path)
    report = {
        "status": "PASS",
        "real_model_forward": True,
        "fixture_only": True,
        **rerank.smoke_expected_fields(config),
    }
    rerank.atomic_write_json(smoke_path, report)
    with pytest.raises(cli_module.StageError, match="fixture-only"):
        cli_module._rerank_score(_score_args(config_path), scorer=StubScorer())
    report["fixture_only"] = False
    report["scoring_source_sha256"] = "0" * 64
    rerank.atomic_write_json(smoke_path, report)
    with pytest.raises(cli_module.StageError, match="stale or incompatible"):
        cli_module._rerank_score(_score_args(config_path), scorer=StubScorer())


@pytest.mark.parametrize(
    "field",
    [
        "scoring_source_sha256",
        "scoring_config_sha256",
        "candidates_sha256",
        "model_revision",
    ],
)
def test_cli_smoke_gate_rejects_each_load_bearing_mismatch(
    tmp_path: Path, field: str
) -> None:
    config_path, _ = build_fixture(tmp_path)
    smoke_path = tmp_path / "reports/audit/rerank_smoke.json"
    report = json.loads(smoke_path.read_text())
    report[field] = "0" * len(str(report[field]))
    rerank.atomic_write_json(smoke_path, report)
    with pytest.raises(cli_module.StageError, match="stale or incompatible"):
        cli_module._rerank_score(_score_args(config_path), scorer=StubScorer())


def test_cli_smoke_gate_allows_evaluation_only_hash_change(tmp_path: Path) -> None:
    config_path, _ = build_fixture(tmp_path)
    smoke_path = tmp_path / "reports/audit/rerank_smoke.json"
    report = json.loads(smoke_path.read_text())
    report["evaluation_source_sha256"] = "0" * 64
    report["source_tree_sha256"] = "0" * 64
    report["config_sha256"] = "0" * 64
    rerank.atomic_write_json(smoke_path, report)
    assert cli_module._rerank_score(
        _score_args(config_path), scorer=StubScorer()
    ) == 0


def test_valid_real_pass_smoke_unlocks_cli_scoring(tmp_path: Path) -> None:
    config_path, _ = build_fixture(tmp_path)
    assert cli_module._rerank_score(_score_args(config_path), scorer=StubScorer()) == 0
    assert (
        tmp_path / "artifacts/scores/dev_zeroshot_mminilmv2l12.parquet"
    ).is_file()


def test_evaluation_source_change_preserves_existing_score(tmp_path: Path) -> None:
    config_path, _ = build_fixture(tmp_path)
    config = rerank.load_rerank_config(config_path)
    rerank.run_rerank_scoring(
        config, split="dev", requested_device="cpu", scorer=StubScorer()
    )
    score_path = tmp_path / "artifacts/scores/dev_zeroshot_mminilmv2l12.parquet"
    before = score_path.read_bytes()
    old_sidecar = json.loads(
        score_path.with_name(f"{score_path.name}.json").read_text()
    )
    source = tmp_path / "src/rusearchrank/evaluation.py"
    source.write_bytes(source.read_bytes() + b"\n# evaluation-only audit patch\n")
    validated = rerank.validate_current_score_sidecar(config, score_path=score_path)
    assert (
        validated["scoring_source_sha256"]
        == old_sidecar["fingerprint_components"]["scoring_source_sha256"]
    )
    assert (
        validated["evaluation_source_sha256"]
        != old_sidecar["fingerprint_components"]["evaluation_source_sha256"]
    )
    reused = rerank.run_rerank_scoring(
        config, split="dev", requested_device="cpu", scorer=StubScorer()
    )
    assert reused["action"] == "reused_valid_scores"
    assert score_path.read_bytes() == before


def test_legacy_production_sidecar_is_reused_after_evaluation_only_patch(
    tmp_path: Path,
) -> None:
    config_path, _ = build_fixture(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"], cwd=tmp_path, check=True
    )
    tracked = [
        "configs/rerank.yaml",
        "pyproject.toml",
        "src/rusearchrank/rerank.py",
        "src/rusearchrank/cli.py",
        "src/rusearchrank/evaluation.py",
    ]
    subprocess.run(["git", "add", *tracked], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture producer"],
        cwd=tmp_path,
        check=True,
    )
    config = rerank.load_rerank_config(config_path)
    rerank.run_rerank_scoring(
        config, split="dev", requested_device="cpu", scorer=StubScorer()
    )
    score_path = tmp_path / "artifacts/scores/dev_zeroshot_mminilmv2l12.parquet"
    sidecar_path = score_path.with_name(f"{score_path.name}.json")
    sidecar = json.loads(sidecar_path.read_text())
    components = dict(sidecar["fingerprint_components"])
    components.pop("scoring_source_sha256")
    components.pop("scoring_config_sha256")
    components.pop("evaluation_source_sha256")
    sidecar["fingerprint_components"] = components
    sidecar["input_fingerprint"] = rerank.build_legacy_input_fingerprint(
        components
    )
    rerank.atomic_write_json(sidecar_path, sidecar)

    evaluation_source = tmp_path / "src/rusearchrank/evaluation.py"
    evaluation_source.write_bytes(
        evaluation_source.read_bytes() + b"\n# post-production evaluation patch\n"
    )
    validated = rerank.validate_current_score_sidecar(config, score_path=score_path)
    assert validated["provenance_migration"]["mode"] == (
        "verified_legacy_git_source"
    )
    assert (
        validated["scoring_source_sha256"]
        == validated["provenance_migration"][
            "producer_scoring_source_sha256"
        ]
    )


@pytest.mark.parametrize("damage", ["file_hash", "schema", "key_set"])
def test_downstream_rejects_damaged_existing_score(
    tmp_path: Path, damage: str
) -> None:
    config_path, _ = build_fixture(tmp_path)
    config = rerank.load_rerank_config(config_path)
    rerank.run_rerank_scoring(
        config, split="dev", requested_device="cpu", scorer=StubScorer()
    )
    score_path = tmp_path / "artifacts/scores/dev_zeroshot_mminilmv2l12.parquet"
    sidecar_path = score_path.with_name(f"{score_path.name}.json")
    if damage == "file_hash":
        data = bytearray(score_path.read_bytes())
        data[-12] ^= 1
        score_path.write_bytes(data)
    else:
        table = pq.read_table(score_path)
        if damage == "schema":
            columns = [
                table.column(name).cast(pa.float64())
                if name == "score"
                else table.column(name)
                for name in table.column_names
            ]
            table = pa.Table.from_arrays(columns, names=table.column_names)
        else:
            rows = table.to_pylist()
            rows[0]["docid"] = "unexpected-docid"
            table = pa.Table.from_pylist(rows, schema=rerank.SCORE_SCHEMA)
        pq.write_table(table, score_path)
        sidecar = json.loads(sidecar_path.read_text())
        digest = hashlib.sha256(score_path.read_bytes()).hexdigest()
        sidecar["scores_sha256"] = digest
        sidecar["shard_sha256"] = digest
        rerank.atomic_write_json(sidecar_path, sidecar)
    with pytest.raises((ValueError, OSError, pa.ArrowInvalid)):
        rerank.validate_current_score_sidecar(config, score_path=score_path)


def test_scoring_source_change_rejects_old_partial_shard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, _ = build_fixture(tmp_path)
    config = rerank.load_rerank_config(config_path)
    with pytest.raises(RuntimeError):
        rerank.run_rerank_scoring(
            config,
            split="dev",
            requested_device="cpu",
            scorer=StubScorer(fail_on_call=2),
        )
    source = tmp_path / "src/rusearchrank/rerank.py"
    original_source = source.read_text(encoding="utf-8")
    changed_source = original_source.replace(
        'return f"{title}{separator}{text}" if title and title.strip() else text',
        'return (f"{title}{separator}{text}" if title and title.strip() else text)',
        1,
    )
    assert changed_source != original_source
    source.write_text(changed_source, encoding="utf-8")
    result = rerank.run_rerank_scoring(
        config,
        split="dev",
        requested_device="cpu",
        scorer=StubScorer(),
    )
    assert result["reused_shards"] == 0
    assert result["scored_shards"] == 2
    assert "input_fingerprint is stale" in capsys.readouterr().out


def test_scoring_config_change_rejects_existing_score(tmp_path: Path) -> None:
    config_path, _ = build_fixture(tmp_path)
    config = rerank.load_rerank_config(config_path)
    rerank.run_rerank_scoring(
        config, split="dev", requested_device="cpu", scorer=StubScorer()
    )
    changed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    changed["inference"]["seed"] += 1
    config_path.write_text(
        yaml.safe_dump(changed, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    changed_config = rerank.load_rerank_config(config_path)
    with pytest.raises(ValueError, match="scoring_config_sha256"):
        rerank.validate_current_score_sidecar(changed_config)


def test_preflight_reports_unavailable_pinned_model_clearly(tmp_path: Path) -> None:
    config_path, _ = build_fixture(tmp_path)
    config = rerank.load_rerank_config(config_path)

    class UnavailableApi:
        def model_info(self, model_id: str, *, revision: str) -> object:
            raise RuntimeError("fixture hub unavailable")

    with pytest.raises(ValueError, match="pinned model revision is unavailable") as error:
        rerank.preflight_rerank(config, model_api=UnavailableApi())
    assert "fixture hub unavailable" in str(error.value)


def test_clean_scoring_process_never_imports_datasets(tmp_path: Path) -> None:
    config_path, _ = build_fixture(tmp_path)
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "sitecustomize.py").write_text(
        "import builtins\n"
        "_original = builtins.__import__\n"
        "def _guard(name, *args, **kwargs):\n"
        "    if name.split('.')[0] == 'datasets':\n"
        "        raise RuntimeError('datasets import is forbidden on scoring path')\n"
        "    return _original(name, *args, **kwargs)\n"
        "builtins.__import__ = _guard\n",
        encoding="utf-8",
    )
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(poison), str(repository / "src"), str(repository / "tests")]
    )
    script = (
        "import sys\n"
        "from rusearchrank.rerank import load_rerank_config, run_rerank_scoring\n"
        "from test_rerank_fixture_e2e import StubScorer\n"
        f"config = load_rerank_config({str(config_path)!r})\n"
        "run_rerank_scoring(config, split='dev', requested_device='cpu', scorer=StubScorer())\n"
        "assert 'datasets' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_clean_cli_and_phase2_notebook_validator() -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    cli = subprocess.run(
        [sys.executable, "-m", "rusearchrank.cli", "--help"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stderr
    assert "rerank-score" in cli.stdout and "package-phase2" in cli.stdout
    validator = subprocess.run(
        [sys.executable, "scripts/validate_phase2_notebook.py"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validator.returncode == 0, validator.stderr
    assert '"status": "PASS"' in validator.stdout
