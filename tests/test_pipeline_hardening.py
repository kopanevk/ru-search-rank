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
import rusearchrank.trec_eval as trec_eval_module
import shard_fixtures
from rusearchrank.cli import main
from rusearchrank.data import extract_passages_from_rows
from rusearchrank.retrieval import normalize_bm25_run, read_trec_run, write_trec_run


@pytest.mark.parametrize(
    "line",
    [
        "q Q0 d 1 1.0\n",
        "q Q0 d 1 1.0 tag extra\n",
    ],
)
def test_trec_parser_requires_exactly_six_fields(tmp_path: Path, line: str) -> None:
    path = tmp_path / "bad.trec"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 6 fields"):
        read_trec_run(str(path), split="dev")


def test_trec_parser_preserves_string_ids_and_rejects_nonfinite_score(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ids.trec"
    path.write_text("0007 Q0 0009 1 1.25 tag\n", encoding="utf-8")
    run = read_trec_run(str(path), split="dev")
    assert run.loc[0, "query_id"] == "0007"
    assert run.loc[0, "docid"] == "0009"
    path.write_text("q Q0 d 1 nan tag\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        read_trec_run(str(path), split="dev")


def test_existing_valid_stable_run_is_reused_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "dev_top1000.trec"
    target = tmp_path / "dev_top100.trec"
    source_frame = normalize_bm25_run(
        pd.DataFrame(
            {
                "split": ["dev"] * 5,
                "query_id": ["q"] * 5,
                "docid": ["e", "d", "c", "b", "a"],
                "bm25_score": [1.0] * 5,
            }
        )
    )
    write_trec_run(source_frame, str(source), tag="official")
    cli_module._stable_truncate_trec_run(
        source, target, split="dev", top_k=3, source_top_k=5
    )
    before_source = source.read_bytes()
    before_target = target.read_bytes()

    reused, action = cli_module._load_or_create_stable_trec_run(
        source,
        target,
        split="dev",
        top_k=3,
        source_top_k=5,
        overwrite=False,
    )

    assert action == "reused_valid"
    assert source.read_bytes() == before_source
    assert target.read_bytes() == before_target
    assert reused["bm25_rank"].tolist() == [1, 2, 3]


def test_existing_invalid_stable_run_is_preserved_without_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dev_top1000.trec"
    target = tmp_path / "dev_top100.trec"
    source.write_text(
        "q Q0 a 1 2.0 official\nq Q0 b 2 1.0 official\n",
        encoding="utf-8",
    )
    target.write_text("q Q0 wrong 1 9.0 stale\n", encoding="utf-8")
    before = target.read_bytes()
    with pytest.raises(ValueError, match="invalid or stale"):
        cli_module._load_or_create_stable_trec_run(
            source,
            target,
            split="dev",
            top_k=1,
            source_top_k=2,
            overwrite=False,
        )
    assert target.read_bytes() == before


def test_dev_stable_run_is_not_a_conflicting_candidate_output(tmp_path: Path) -> None:
    config = {
        "_config_path": str(tmp_path / "configs/retrieval.yaml"),
        "paths": {"repository_root": ".."},
        "artifacts": {
            "train_candidates": "artifacts/candidates/train.parquet",
            "dev_candidates": "artifacts/candidates/dev.parquet",
            "queries": "artifacts/candidates/queries.parquet",
            "passages": "artifacts/candidates/passages.parquet",
            "dev_top100_run": "artifacts/runs/dev_top100.trec",
        },
    }
    paths = cli_module._candidate_output_paths(config)
    assert "dev_top100_run" not in paths
    assert len(paths) == 4


def test_artifact_and_audit_paths_are_resolved_from_config_root_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = {
        "_config_path": str(config_dir / "retrieval.yaml"),
        "paths": {"repository_root": ".."},
        "artifacts": {"run": "artifacts/runs/run.trec"},
        "audits": {"audit": "reports/audit/audit.json"},
    }
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    assert cli_module._artifact_path(config, "run") == (
        tmp_path / "artifacts/runs/run.trec"
    )
    assert cli_module._audit_path(config, "audit") == (
        tmp_path / "reports/audit/audit.json"
    )


def test_missing_passage_error_includes_count_docid_split_and_query() -> None:
    context = {"d1": [{"split": "train", "query_id": "375"}]}
    with pytest.raises(ValueError) as error:
        extract_passages_from_rows([], {"d1"}, candidate_context=context)
    message = str(error.value)
    assert "1 candidate passages missing" in message
    assert "docid=d1" in message
    assert "split=train" in message
    assert "query_id=375" in message


def _cell11_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_dir = tmp_path / "artifacts/raw/miracl-ru"
    run_dir = tmp_path / "artifacts/runs"
    audit_dir = tmp_path / "reports/audit"
    for directory in (raw_dir, run_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)
    annotation_paths: dict[str, dict[str, Path]] = {}
    for split, query_id in (("train", "qt"), ("dev", "qd")):
        topics = raw_dir / f"topics.miracl-v1.0-ru-{split}.tsv"
        qrels = raw_dir / f"qrels.miracl-v1.0-ru-{split}.tsv"
        topics.write_text(f"{query_id}\tquery {query_id}\n", encoding="utf-8")
        qrels.write_text(f"{query_id}\t0\td1\t1\n", encoding="utf-8")
        annotation_paths[split] = {"topics": topics, "qrels": qrels}

    train_run = run_dir / "train_bm25_top100.trec"
    dev_raw = run_dir / "dev_bm25_top1000.trec"
    dev_stable = run_dir / "dev_bm25_top100.trec"
    train_run.write_text(
        "qt Q0 d1 1 2.0 official\nqt Q0 d2 2 1.0 official\n",
        encoding="utf-8",
    )
    dev_raw.write_text(
        "qd Q0 d1 1 3.0 official\nqd Q0 d2 2 2.0 official\n"
        "qd Q0 d3 3 1.0 official\n",
        encoding="utf-8",
    )
    cli_module._stable_truncate_trec_run(
        dev_raw, dev_stable, split="dev", top_k=2, source_top_k=3
    )
    audit = {
        "status": "PASS",
        "gate_passed": True,
        "overall_pass": True,
        "source_run": "artifacts/runs/dev_bm25_top1000.trec",
        "source_run_sha256": hashlib.sha256(dev_raw.read_bytes()).hexdigest(),
        "source_run_size_bytes": dev_raw.stat().st_size,
    }
    (audit_dir / "reproduction.json").write_text(json.dumps(audit), encoding="utf-8")
    # Real gzip shards, so the candidate cache streams genuine JSONL.GZ bytes.
    corpus_dir = tmp_path / "artifacts/work/local-corpus"
    shard_fixtures.write_corpus(
        corpus_dir,
        language="ru",
        shards=[
            [*shard_fixtures.filler("noise", 5), shard_fixtures.passage("d1")],
            [shard_fixtures.passage("d2"), *shard_fixtures.filler("other", 3)],
            [shard_fixtures.passage("d3"), *shard_fixtures.filler("tail", 2)],
        ],
    )
    config = {
        "dataset": {
            "language": "ru",
            "corpus_source": "miracl/miracl-corpus",
            "corpus_revision": "d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
            "corpus_repo_type": "dataset",
            "corpus_shard_count": 3,
            "corpus_shard_files": [
                f"miracl-corpus-v1.0-ru/docs-{index}.jsonl.gz" for index in range(3)
            ],
            "corpus_cache_dir": "artifacts/work/hf-corpus",
            "corpus_local_dir": "artifacts/work/local-corpus",
            "corpus_passage_batch_rows": 1,
            "annotations_revision": "fixture-annotations",
            "train_topics_path": str(annotation_paths["train"]["topics"].relative_to(tmp_path)),
            "topics": {"train": "fixture", "dev": "fixture"},
            "qrels": {"train": "fixture", "dev": "fixture"},
            "expected_rows": {
                "train_queries": 1,
                "dev_queries": 1,
                "train_qrels": 1,
                "dev_qrels": 1,
            },
        },
        "environment": {},
        "retrieval": {
            "candidate_depth": 2,
            "retrieval_hits": {"train": 2, "dev": 3},
            "threads": 1,
            "batch_size": 2,
            "index_name": "fixture-index",
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
            "reproduction": "reports/audit/reproduction.json",
            "qrels": "reports/audit/qrels.json",
            "manifest": "reports/audit/manifest.json",
        },
        "archive": {"path": "artifacts/results.zip"},
        "reproduction_gate": {},
        "paths": {
            "repository_root": ".",
            "raw_dir": "artifacts/raw/miracl-ru",
            "work_dir": "artifacts/work/phase1",
        },
    }
    config_path = tmp_path / "retrieval.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, dev_stable, raw_dir


def test_cell11_continues_with_preexisting_valid_dev_top100(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, dev_stable, raw_dir = _cell11_fixture(tmp_path)
    before = dev_stable.read_bytes()
    annotations = {
        split: {
            "topics": raw_dir / f"topics.miracl-v1.0-ru-{split}.tsv",
            "qrels": raw_dir / f"qrels.miracl-v1.0-ru-{split}.tsv",
        }
        for split in ("train", "dev")
    }
    monkeypatch.setattr(cli_module, "_preflight_candidate_cache", lambda config: {})
    monkeypatch.setattr(cli_module, "_annotation_paths", lambda config: annotations)

    assert main(["build-candidate-cache", "--config", str(config_path)]) == 0
    assert dev_stable.read_bytes() == before
    assert (tmp_path / "artifacts/candidates/train_top100.parquet").is_file()
    passages = pd.read_parquet(tmp_path / "artifacts/candidates/passages.parquet")
    assert sorted(passages["docid"].tolist()) == ["d1", "d2"]
    assert passages["text"].str.contains("Русский").all()
    work = json.loads(
        (tmp_path / "artifacts/work/phase1/candidate_cache.json").read_text(
            encoding="utf-8"
        )
    )
    assert work["dev_top100_action"] == "reused_valid"
    extraction = work["passage_extraction"]
    assert extraction["source"]["kind"] == "local_directory"
    assert extraction["source"]["dataset_script_used"] is False
    assert extraction["found_docids"] == 2
    assert extraction["missing_docids"] == 0
    assert extraction["batches_written"] == 2  # one batch per row, really batched
    assert extraction["early_stop"] is True  # shard 2 is never opened
    assert extraction["shards_visited"] == 2
    assert extraction["lines_visited"] > 2  # non-candidate rows really streamed past
    assert not (
        tmp_path / "artifacts/work/phase1/passages.staging.parquet"
    ).exists()


def test_cell11_rejects_partial_candidate_parquets_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, _, raw_dir = _cell11_fixture(tmp_path)
    partial = tmp_path / "artifacts/candidates/train_top100.parquet"
    partial.parent.mkdir(parents=True)
    pd.DataFrame({"diagnostic": ["preserve me"]}).to_parquet(partial, index=False)
    before = partial.read_bytes()
    annotations = {
        split: {
            "topics": raw_dir / f"topics.miracl-v1.0-ru-{split}.tsv",
            "qrels": raw_dir / f"qrels.miracl-v1.0-ru-{split}.tsv",
        }
        for split in ("train", "dev")
    }
    monkeypatch.setattr(cli_module, "_preflight_candidate_cache", lambda config: {})
    monkeypatch.setattr(cli_module, "_annotation_paths", lambda config: annotations)
    assert main(["build-candidate-cache", "--config", str(config_path)]) == 1
    assert "partial candidate cache detected" in capsys.readouterr().err
    assert partial.read_bytes() == before


def test_run_bm25_reuses_existing_valid_raw_run_without_pyserini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _, _ = _cell11_fixture(tmp_path)
    monkeypatch.setattr(
        cli_module, "_preflight_retrieval", lambda config, check_index: {}
    )

    def forbidden(*_: object, **__: object) -> dict[str, object]:
        raise AssertionError("Pyserini must not run for a valid existing raw run")

    monkeypatch.setattr(cli_module, "_run_streaming_process", forbidden)
    assert main(["run-bm25", "--config", str(config_path), "--split", "train"]) == 0
    report = json.loads(
        (tmp_path / "artifacts/work/phase1/retrieval_train.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["action"] == "reused_valid_run"


def test_depth_audit_rejects_unknown_query_id() -> None:
    run = pd.DataFrame(
        {
            "split": ["train"],
            "query_id": ["unknown"],
            "docid": ["d"],
            "bm25_rank": [1],
            "bm25_score": [1.0],
        }
    )
    topics = pd.DataFrame(
        {"split": ["train"], "query_id": ["expected"], "query_text": ["query"]}
    )
    from rusearchrank.retrieval import build_retrieval_depth_audit

    with pytest.raises(ValueError, match="unknown query_id"):
        build_retrieval_depth_audit(run, topics, requested_top_k=100)


def test_run_validation_returns_zero_hit_audit_before_coverage_gate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.trec"
    path.write_text("hit Q0 d1 1 1.0 fixture\n", encoding="utf-8")
    topics = pd.DataFrame(
        {
            "split": ["train", "train"],
            "query_id": ["hit", "zero"],
            "query_text": ["has result", "no result"],
        }
    )
    _, _, audit = cli_module._validate_retrieval_run(
        path,
        split="train",
        queries=topics,
        requested_top_k=100,
    )
    assert audit["zero_hit_query_count"] == 1
    assert audit["zero_hit_queries"][0]["query_id"] == "zero"


def test_missing_trec_eval_is_a_clear_preflight_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "_config_path": str(tmp_path / "retrieval.yaml"),
        "paths": {"repository_root": "."},
        "reproduction_gate": {"trec_eval_executable": "trec_eval"},
    }
    monkeypatch.setattr(trec_eval_module.shutil, "which", lambda executable: None)
    with pytest.raises(ValueError, match="trec_eval.*не найден"):
        cli_module._resolve_trec_eval_executable(config)


def test_missing_qrels_is_a_clear_annotation_preflight_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "artifacts/raw/miracl-ru"
    raw_dir.mkdir(parents=True)
    train_topics = raw_dir / "topics.miracl-v1.0-ru-train.tsv"
    train_topics.write_text("q\tquery\n", encoding="utf-8")
    config_path = tmp_path / "retrieval.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    config = {
        "_config_path": str(config_path),
        "paths": {"repository_root": ".", "raw_dir": "artifacts/raw/miracl-ru"},
        "dataset": {
            "train_topics_path": str(train_topics.relative_to(tmp_path)),
            "expected_rows": {
                "train_queries": 1,
                "dev_queries": 1,
                "train_qrels": 1,
                "dev_qrels": 1,
            },
            "topics": {"train": "fixture"},
            "annotations_revision": "fixture",
        },
        "reproduction_gate": {"official_topic": "miracl-v1.0-ru-dev"},
    }
    with pytest.raises(ValueError, match="official train qrels does not exist"):
        cli_module._validate_official_annotations(config)


def test_streaming_process_persists_both_pipes_and_return_code(tmp_path: Path) -> None:
    log_path = tmp_path / "command.json"
    result = cli_module._run_streaming_process(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(7)",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=log_path,
    )
    assert result["returncode"] == 7
    logged = json.loads(log_path.read_text(encoding="utf-8"))
    assert logged["stdout"] == "out\n"
    assert logged["stderr"] == "err\n"
    assert logged["returncode"] == 7


def test_clean_process_cli_and_notebook_validation() -> None:
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
    assert "build-candidate-cache" in cli.stdout
    notebook = subprocess.run(
        [sys.executable, "scripts/validate_phase1_notebook.py"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert notebook.returncode == 0, notebook.stderr
    assert '"status": "PASS"' in notebook.stdout


def test_archive_validator_rejects_stale_manifest_and_extra_member(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs/retrieval.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("fixture: true\n", encoding="utf-8")
    payload = tmp_path / "artifacts/value.json"
    manifest_path = tmp_path / "reports/audit/manifest.json"
    archive_path = tmp_path / "artifacts/results.zip"
    payload.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    payload.write_text('{"value": 1}\n', encoding="utf-8")
    config = {"_config_path": str(config_path), "paths": {"repository_root": ".."}}
    entry = cli_module._artifact_manifest_entry(
        config,
        payload,
        producer_command="fixture",
        input_hashes={"fixture": "0" * 64},
    )
    manifest = {"status": "PASS", "artifacts": [entry]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(payload, "artifacts/value.json")
        archive.write(manifest_path, "reports/audit/manifest.json")
    cli_module._validate_phase1_archive(
        config, archive_path, manifest_path, payload_paths=[payload]
    )

    stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale["artifacts"][0]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="stale manifest SHA-256"):
        cli_module._validate_phase1_archive(
            config, archive_path, manifest_path, payload_paths=[payload]
        )

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("unexpected.bin", b"no")
    with pytest.raises(ValueError, match="exact ordered allowlist"):
        cli_module._validate_phase1_archive(
            config, archive_path, manifest_path, payload_paths=[payload]
        )
