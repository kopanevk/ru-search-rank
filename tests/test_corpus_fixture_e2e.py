"""Offline end-to-end Phase 1 fixture driven by real local ``.jsonl.gz`` shards.

This exercises the same CLI commands Cells 12-14 run, with real gzip corpus
shards, real Parquet writes, a real qrels audit, a real manifest, a real ZIP,
real extraction and real hash revalidation. Only the two things that genuinely
cannot exist on a developer machine are replaced: the Linux/Java/Pyserini
environment preflight and the network download of official annotations.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pandas as pd
import pytest
import yaml

import rusearchrank.cli as cli_module
import shard_fixtures
from rusearchrank.cli import main

TRAIN_QUERIES = 3
DEV_QUERIES = 2
CANDIDATE_DEPTH = 4
DEV_HITS = 8


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    raw_dir = tmp_path / "artifacts/raw/miracl-ru"
    run_dir = tmp_path / "artifacts/runs"
    corpus_dir = tmp_path / "artifacts/work/local-corpus"
    for directory in (raw_dir, run_dir):
        directory.mkdir(parents=True, exist_ok=True)

    topics = {
        "train": [(f"t{index}", f"train запрос {index}") for index in range(TRAIN_QUERIES)],
        "dev": [(f"d{index}", f"dev запрос {index}") for index in range(DEV_QUERIES)],
    }
    qrels_rows = {
        # relevant, judged non-relevant, and deliberately unjudged candidates.
        "train": [("t0", "p0", 1), ("t0", "p1", 0), ("t1", "p2", 1)],
        "dev": [("d0", "p3", 1), ("d0", "p4", 0)],
    }
    for split, rows in topics.items():
        (raw_dir / f"topics.miracl-v1.0-ru-{split}.tsv").write_text(
            "".join(f"{query_id}\t{text}\n" for query_id, text in rows),
            encoding="utf-8",
        )
        (raw_dir / f"qrels.miracl-v1.0-ru-{split}.tsv").write_text(
            "".join(
                f"{query_id}\t0\t{docid}\t{grade}\n"
                for query_id, docid, grade in qrels_rows[split]
            ),
            encoding="utf-8",
        )

    # Train run: exactly candidate depth for two queries, deliberately short for one.
    train_lines: list[str] = []
    depths = {"t0": CANDIDATE_DEPTH, "t1": CANDIDATE_DEPTH, "t2": 2}
    for query_id, depth in depths.items():
        for rank in range(1, depth + 1):
            train_lines.append(
                f"{query_id} Q0 p{rank - 1} {rank} {float(100 - rank):.6f} official"
            )
    train_run = run_dir / "train_bm25_top100.trec"
    train_run.write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    dev_lines: list[str] = []
    for query_id in ("d0", "d1"):
        for rank in range(1, DEV_HITS + 1):
            dev_lines.append(
                f"{query_id} Q0 p{rank + 2} {rank} {float(50 - rank):.6f} official"
            )
    dev_run = run_dir / "dev_bm25_top1000.trec"
    dev_run.write_text("\n".join(dev_lines) + "\n", encoding="utf-8")

    # Real gzip shards spread over several files, with non-candidate noise.
    candidate_docids = sorted({f"p{index}" for index in range(11)})
    shard_fixtures.write_corpus(
        corpus_dir,
        shards=[
            [
                *shard_fixtures.filler("noise-a", 4),
                *[shard_fixtures.passage(docid) for docid in candidate_docids[:4]],
            ],
            [
                *[shard_fixtures.passage(docid) for docid in candidate_docids[4:8]],
                *shard_fixtures.filler("noise-b", 3),
            ],
            [
                *shard_fixtures.filler("noise-c", 2),
                *[shard_fixtures.passage(docid) for docid in candidate_docids[8:]],
            ],
            shard_fixtures.filler("never-read", 5),
        ],
    )

    audit_dir = tmp_path / "reports/audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "bm25_reproduction.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "gate_passed": True,
                "overall_pass": True,
                "source_run": "artifacts/runs/dev_bm25_top1000.trec",
                "source_run_sha256": hashlib.sha256(dev_run.read_bytes()).hexdigest(),
                "source_run_size_bytes": dev_run.stat().st_size,
                "official_tool_values": {"ndcg_at_10": 0.3342, "recall_at_100": 0.6612},
            }
        ),
        encoding="utf-8",
    )

    config = {
        "dataset": {
            "name": "miracl",
            "language": "ru",
            "corpus_source": "miracl/miracl-corpus",
            "corpus_revision": "d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
            "corpus_repo_type": "dataset",
            "corpus_shard_count": 4,
            "corpus_shard_files": [
                f"miracl-corpus-v1.0-ru/docs-{index}.jsonl.gz" for index in range(4)
            ],
            "corpus_cache_dir": "artifacts/work/hf-corpus",
            "corpus_local_dir": "artifacts/work/local-corpus",
            "corpus_passage_batch_rows": 3,
            "annotations_revision": "5be20db9509754dadad47689368639fcec739c00",
            "train_topics_path": "artifacts/raw/miracl-ru/topics.miracl-v1.0-ru-train.tsv",
            "topics": {"train": "https://example.test/train", "dev": "https://example.test/dev"},
            "qrels": {"train": "https://example.test/train", "dev": "https://example.test/dev"},
            "expected_rows": {
                "train_queries": TRAIN_QUERIES,
                "dev_queries": DEV_QUERIES,
                "train_qrels": len(qrels_rows["train"]),
                "dev_qrels": len(qrels_rows["dev"]),
            },
        },
        "environment": {
            "platform": "linux",
            "python": "3.12",
            "java": "21",
            "pyserini": "2.3.0",
            "minimum_free_disk_gib": 1,
        },
        "retrieval": {
            "engine": "pyserini_lucene",
            "index_name": "miracl-v1.0-ru",
            "analyzer": "RussianAnalyzer",
            "candidate_depth": CANDIDATE_DEPTH,
            "retrieval_hits": {"train": CANDIDATE_DEPTH, "dev": DEV_HITS},
            "threads": 2,
            "batch_size": 2,
            "bm25_k1": 0.9,
            "bm25_b": 0.4,
        },
        "splits": ["train", "dev"],
        "artifacts": {
            "train_run": "artifacts/runs/train_bm25_top100.trec",
            "dev_run": "artifacts/runs/dev_bm25_top1000.trec",
            "dev_top100_run": "artifacts/runs/dev_bm25_top100.trec",
            "train_candidates": "artifacts/candidates/train_top100.parquet",
            "dev_candidates": "artifacts/candidates/dev_top100.parquet",
            "queries": "artifacts/candidates/queries.parquet",
            "passages": "artifacts/candidates/passages.parquet",
        },
        "audits": {
            "reproduction": "reports/audit/bm25_reproduction.json",
            "qrels": "reports/audit/qrels_audit.json",
            "manifest": "reports/audit/candidate_cache_manifest.json",
        },
        "archive": {
            "path": "artifacts/rusearchrank_phase1_results.zip",
            "include_runs": True,
        },
        "reproduction_gate": {
            "official_topic": "miracl-v1.0-ru-dev",
            "official_ndcg_at_10": 0.334,
            "official_recall_at_100": 0.661,
            "official_retrieval_command": "official command",
            "trec_eval_executable": "trec_eval",
            "trec_eval_version": "9.0.8",
            "ndcg_at_10_tolerance": 0.002,
            "recall_at_100_tolerance": 0.005,
        },
        "paths": {
            "repository_root": ".",
            "raw_dir": "artifacts/raw/miracl-ru",
            "work_dir": "artifacts/work/phase1",
        },
    }
    config_path = tmp_path / "retrieval.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return {
        "config": config_path,
        "raw_dir": raw_dir,
        "run_dir": run_dir,
        "corpus_dir": corpus_dir,
        "audit_dir": audit_dir,
        "train_run": train_run,
        "dev_run": dev_run,
    }


def test_local_jsonl_gz_fixture_runs_cells_12_to_14_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    config_path = fixture["config"]
    annotations = {
        split: {
            "topics": fixture["raw_dir"] / f"topics.miracl-v1.0-ru-{split}.tsv",
            "qrels": fixture["raw_dir"] / f"qrels.miracl-v1.0-ru-{split}.tsv",
        }
        for split in ("train", "dev")
    }
    monkeypatch.setattr(cli_module, "_preflight_candidate_cache", lambda config: {})
    monkeypatch.setattr(cli_module, "_preflight_retrieval", lambda config, check_index: {})
    monkeypatch.setattr(cli_module, "_annotation_paths", lambda config: annotations)

    dev_before = fixture["dev_run"].read_bytes()
    train_before = fixture["train_run"].read_bytes()

    # Cell 8/9 equivalent: existing valid raw runs are revalidated and reused.
    assert main(["run-bm25", "--config", str(config_path), "--split", "train"]) == 0
    assert main(["run-bm25", "--config", str(config_path), "--split", "dev"]) == 0
    work_dir = tmp_path / "artifacts/work/phase1"
    for split in ("train", "dev"):
        report = json.loads((work_dir / f"retrieval_{split}.json").read_text())
        assert report["action"] == "reused_valid_run"
    assert fixture["train_run"].read_bytes() == train_before

    # Cell 12 equivalent: stable dev top-100, three-state cache, real passages.
    assert main(["build-candidate-cache", "--config", str(config_path)]) == 0
    assert fixture["dev_run"].read_bytes() == dev_before  # raw top-1000 untouched
    cache = json.loads((work_dir / "candidate_cache.json").read_text())
    extraction = cache["passage_extraction"]
    assert extraction["status"] == "PASS"
    assert extraction["missing_docids"] == 0
    assert extraction["batches_written"] >= 2
    assert extraction["early_stop"] is True
    # All seven candidate passages live in the first two shards, so the last two
    # shards are never opened at all.
    assert extraction["shards_visited"] == 2
    assert extraction["shards_skipped_by_early_stop"] == 2
    assert extraction["found_docids"] == 7
    assert extraction["source"]["kind"] == "local_directory"

    candidates = pd.read_parquet(tmp_path / "artifacts/candidates/train_top100.parquet")
    assert set(candidates["judgment"]) == {
        "relevant",
        "judged_non_relevant",
        "unjudged",
    }
    unjudged = candidates.loc[candidates["judgment"].eq("unjudged")]
    assert unjudged["relevance"].isna().all()  # never an implicit negative
    assert unjudged["relevance_grade"].isna().all()
    assert (~unjudged["is_judged"]).all()
    assert candidates["query_id"].map(type).eq(str).all()
    assert candidates["docid"].map(type).eq(str).all()
    # Variable depth is preserved: the short query keeps its two candidates.
    assert candidates.groupby("query_id").size().to_dict() == {
        "t0": CANDIDATE_DEPTH,
        "t1": CANDIDATE_DEPTH,
        "t2": 2,
    }

    passages = pd.read_parquet(tmp_path / "artifacts/candidates/passages.parquet")
    assert passages["docid"].is_unique
    assert not passages["text"].str.strip().eq("").any()

    # Cell 12 equivalent (continued): the qrels audit.
    assert main(["audit-qrels", "--config", str(config_path), "--overwrite"]) == 0
    qrels_audit = json.loads(
        (fixture["audit_dir"] / "qrels_audit.json").read_text(encoding="utf-8")
    )
    assert qrels_audit["status"] == "PASS"
    train_audit = qrels_audit["splits"]["train"]
    assert train_audit["candidate_judgment_counts"]["relevant"] >= 1
    assert train_audit["candidate_judgment_counts"]["judged_non_relevant"] >= 1
    assert train_audit["candidate_judgment_counts"]["unjudged"] >= 1

    # Cell 13 equivalent: candidate validation.
    for split in ("train", "dev"):
        assert (
            main(
                [
                    "validate-candidates",
                    str(tmp_path / f"artifacts/candidates/{split}_top100.parquet"),
                    "--config",
                    str(config_path),
                ]
            )
            == 0
        )
    assert main(["preflight", "--config", str(config_path), "--stage", "package"]) == 0

    # Cell 14 equivalent: manifest, ZIP, extraction, revalidation.
    assert main(["package-phase1", "--config", str(config_path), "--overwrite"]) == 0
    archive_path = tmp_path / "artifacts/rusearchrank_phase1_results.zip"
    manifest_path = fixture["audit_dir"] / "candidate_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert archive.testzip() is None
    assert names == [
        "artifacts/runs/dev_bm25_top1000.trec",
        "artifacts/runs/dev_bm25_top100.trec",
        "artifacts/runs/train_bm25_top100.trec",
        "artifacts/candidates/train_top100.parquet",
        "artifacts/candidates/dev_top100.parquet",
        "artifacts/candidates/queries.parquet",
        "artifacts/candidates/passages.parquet",
        "reports/audit/bm25_reproduction.json",
        "reports/audit/qrels_audit.json",
        "reports/audit/candidate_cache_manifest.json",
    ]
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)

    extraction_root = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extraction_root)
    for entry in manifest["artifacts"]:
        extracted = extraction_root / entry["path"]
        assert extracted.is_file()
        assert extracted.stat().st_size == entry["size_bytes"]
        assert (
            hashlib.sha256(extracted.read_bytes()).hexdigest() == entry["sha256"]
        )
        assert entry["producer_command"]
        assert entry["input_hashes"]
        if entry["path"].endswith(".parquet"):
            assert pd.read_parquet(extracted).shape[0] == entry["row_count"]
            assert entry["schema"]

    # Reruns are idempotent: nothing is rebuilt and nothing is rewritten.
    passages_before = (tmp_path / "artifacts/candidates/passages.parquet").read_bytes()
    assert main(["build-candidate-cache", "--config", str(config_path)]) == 0
    assert (
        tmp_path / "artifacts/candidates/passages.parquet"
    ).read_bytes() == passages_before
    assert main(["package-phase1", "--config", str(config_path)]) == 0


def test_fresh_clone_placeholder_manifest_does_not_block_packaging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A freshly cloned repository ships a placeholder manifest and no ZIP."""

    fixture = _build_fixture(tmp_path)
    config_path = fixture["config"]
    annotations = {
        split: {
            "topics": fixture["raw_dir"] / f"topics.miracl-v1.0-ru-{split}.tsv",
            "qrels": fixture["raw_dir"] / f"qrels.miracl-v1.0-ru-{split}.tsv",
        }
        for split in ("train", "dev")
    }
    monkeypatch.setattr(cli_module, "_preflight_candidate_cache", lambda config: {})
    monkeypatch.setattr(cli_module, "_preflight_retrieval", lambda config, check_index: {})
    monkeypatch.setattr(cli_module, "_annotation_paths", lambda config: annotations)
    assert main(["run-bm25", "--config", str(config_path), "--split", "train"]) == 0
    assert main(["build-candidate-cache", "--config", str(config_path)]) == 0
    assert main(["audit-qrels", "--config", str(config_path), "--overwrite"]) == 0

    # Exactly what git ships on phase-0 before any Colab run.
    manifest_path = fixture["audit_dir"] / "candidate_cache_manifest.json"
    manifest_path.write_text(
        json.dumps({"status": "BLOCKED_EXTERNAL_RUN", "artifacts": []}), encoding="utf-8"
    )
    archive_path = tmp_path / "artifacts/rusearchrank_phase1_results.zip"
    assert not archive_path.exists()

    assert main(["package-phase1", "--config", str(config_path), "--overwrite"]) == 0
    assert archive_path.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "PASS"


def test_partial_package_without_overwrite_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _build_fixture(tmp_path)
    config_path = fixture["config"]
    annotations = {
        split: {
            "topics": fixture["raw_dir"] / f"topics.miracl-v1.0-ru-{split}.tsv",
            "qrels": fixture["raw_dir"] / f"qrels.miracl-v1.0-ru-{split}.tsv",
        }
        for split in ("train", "dev")
    }
    monkeypatch.setattr(cli_module, "_preflight_candidate_cache", lambda config: {})
    monkeypatch.setattr(cli_module, "_preflight_retrieval", lambda config, check_index: {})
    monkeypatch.setattr(cli_module, "_annotation_paths", lambda config: annotations)
    assert main(["run-bm25", "--config", str(config_path), "--split", "train"]) == 0
    assert main(["build-candidate-cache", "--config", str(config_path)]) == 0
    assert main(["audit-qrels", "--config", str(config_path), "--overwrite"]) == 0
    manifest_path = fixture["audit_dir"] / "candidate_cache_manifest.json"
    manifest_path.write_text(json.dumps({"status": "STALE"}), encoding="utf-8")
    before = manifest_path.read_bytes()

    assert main(["package-phase1", "--config", str(config_path)]) == 1
    assert "partial package output detected" in capsys.readouterr().err
    assert manifest_path.read_bytes() == before  # preserved, never silently replaced


def test_real_smoke_runs_the_whole_round_trip_on_local_shards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Cell 11 smoke, offline: gzip, JSON, Parquet, manifest, ZIP, hashes."""

    fixture = _build_fixture(tmp_path)
    assert (
        main(
            [
                "smoke-corpus-access",
                "--config",
                str(fixture["config"]),
                "--shards-dir",
                "artifacts/work/local-corpus",
                "--max-rows",
                "8",
                "--min-passages",
                "4",
                "--output",
                "reports/audit/corpus_smoke.json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    report = json.loads(
        (tmp_path / "reports/audit/corpus_smoke.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS"
    assert report["full_corpus_downloaded"] is False
    assert report["production_artifacts_written"] == []
    performed = [check["check"] for check in report["checks"]]
    assert performed == [
        "pinned_revision_available",
        "no_dataset_script_loader",
        "numeric_shard_order",
        "shard_available",
        "gzip_json_fields_and_string_docid",
        "real_passage_filtering",
        "parquet_write_and_read_back",
        "temporary_manifest",
        "zip_created_and_extracted",
        "hash_revalidated_after_extraction",
    ]
    assert all(check["status"] == "PASS" for check in report["checks"])
    assert report["source"]["dataset_script_used"] is False
    assert report["source"]["trust_remote_code"] is False
    # No production artifact exists and the temporary directory is cleaned up.
    assert not (tmp_path / "artifacts/candidates").exists()
    assert not (tmp_path / "artifacts/rusearchrank_phase1_results.zip").exists()
    leftovers = list((tmp_path / "artifacts/work/phase1").glob("*corpus-smoke*"))
    assert leftovers == []


def test_smoke_failure_preserves_its_diagnostic_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _build_fixture(tmp_path)
    # A shard that cannot satisfy the minimum passage count fails the smoke.
    assert (
        main(
            [
                "smoke-corpus-access",
                "--config",
                str(fixture["config"]),
                "--shards-dir",
                "artifacts/work/local-corpus",
                "--max-rows",
                "3",
                "--min-passages",
                "50",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "diagnostic files preserved at" in captured.err
    assert not (tmp_path / "reports/audit/corpus_smoke.json").exists()


def test_corpus_config_contract_rejects_unsafe_settings(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    config = cli_module._load_retrieval_config(fixture["config"])
    assert cli_module._corpus_settings(config)["shard_count"] == 4

    lexicographic = dict(config)
    lexicographic["dataset"] = {
        **config["dataset"],
        "corpus_shard_files": sorted(config["dataset"]["corpus_shard_files"]),
        "corpus_shard_count": 4,
    }
    # docs-0, docs-1, docs-2, docs-3 happen to sort correctly; use 12 shards.
    lexicographic["dataset"]["corpus_shard_count"] = 12
    lexicographic["dataset"]["corpus_shard_files"] = sorted(
        f"miracl-corpus-v1.0-ru/docs-{index}.jsonl.gz" for index in range(12)
    )
    with pytest.raises(ValueError, match="numeric order"):
        cli_module._corpus_settings(lexicographic)

    mutable_revision = dict(config)
    mutable_revision["dataset"] = {**config["dataset"], "corpus_revision": "main"}
    with pytest.raises(ValueError, match="immutable 40-character commit SHA"):
        cli_module._corpus_settings(mutable_revision)

    wrong_repo_type = dict(config)
    wrong_repo_type["dataset"] = {**config["dataset"], "corpus_repo_type": "model"}
    with pytest.raises(ValueError, match="corpus_repo_type must be 'dataset'"):
        cli_module._corpus_settings(wrong_repo_type)


def test_production_extraction_never_imports_datasets(tmp_path: Path) -> None:
    """A clean process with a poisoned ``datasets`` module still extracts.

    The previous implementation called ``load_dataset`` and would fail here with
    the exact Colab error; the static-shard reader never touches the package.
    """

    poison_dir = tmp_path / "poison"
    poison_dir.mkdir()
    (poison_dir / "datasets.py").write_text(
        "raise RuntimeError('Dataset scripts are no longer supported, "
        "but found miracl-corpus.py')\n",
        encoding="utf-8",
    )
    corpus_dir = tmp_path / "corpus"
    shard_fixtures.write_corpus(
        corpus_dir,
        shards=[[shard_fixtures.passage("p0"), shard_fixtures.passage("p1")]],
    )
    script = tmp_path / "run_extraction.py"
    script.write_text(
        "import datasets_guard  # noqa: F401\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "from rusearchrank.corpus import LocalShardSource, extract_candidate_passages\n"
        "report = extract_candidate_passages(\n"
        "    {'p0', 'p1'},\n"
        "    Path(sys.argv[2]),\n"
        "    source=LocalShardSource(language='ru', shard_count=1, directory=Path(sys.argv[1])),\n"
        ")\n"
        "assert 'datasets' not in sys.modules, 'datasets was imported'\n"
        "print(json.dumps(report))\n",
        encoding="utf-8",
    )
    (tmp_path / "datasets_guard.py").write_text(
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _guard(name, *args, **kwargs):\n"
        "    if name.split('.')[0] == 'datasets':\n"
        "        raise RuntimeError('Dataset scripts are no longer supported, "
        "but found miracl-corpus.py')\n"
        "    return _real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _guard\n",
        encoding="utf-8",
    )
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(poison_dir), str(repository / "src")]
    )
    result = subprocess.run(
        [sys.executable, str(script), str(corpus_dir), str(tmp_path / "out.parquet")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["status"] == "PASS"
    assert report["found_docids"] == 2
    assert (tmp_path / "out.parquet").is_file()


def test_installed_dependency_versions_are_reported(tmp_path: Path) -> None:
    """Clean-process probe of the versions the compatibility matrix depends on."""

    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, pandas, pyarrow, huggingface_hub;"
            "print(json.dumps({'pandas': pandas.__version__,"
            " 'pyarrow': pyarrow.__version__,"
            " 'huggingface_hub': huggingface_hub.__version__}))",
        ],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    versions = json.loads(result.stdout)
    assert int(versions["pandas"].split(".")[0]) >= 2
    assert int(versions["pyarrow"].split(".")[0]) >= 17
    assert int(versions["huggingface_hub"].split(".")[0]) >= 0
