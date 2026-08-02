from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest
import yaml

import rusearchrank.cli as cli_module
from rusearchrank.cli import _stable_truncate_trec_run, main
from rusearchrank.retrieval import (
    join_qrels,
    normalize_bm25_run,
    read_trec_run,
    stable_sort_bm25,
    validate_query_coverage,
    validate_stable_order,
    validate_top_k,
    write_trec_run,
)


def test_stable_sorting_preserves_query_order_and_assigns_ranks() -> None:
    run = pd.DataFrame(
        {
            "split": ["dev"] * 4,
            "query_id": ["q2", "q1", "q2", "q1"],
            "docid": ["b", "z", "a", "a"],
            "bm25_score": [2.0, 1.0, 2.0, 3.0],
        }
    )
    result = stable_sort_bm25(run)
    assert result[["query_id", "docid"]].values.tolist() == [
        ["q2", "a"],
        ["q2", "b"],
        ["q1", "a"],
        ["q1", "z"],
    ]
    assert result["bm25_rank"].tolist() == [1, 2, 1, 2]


def test_equal_scores_are_tied_by_docid_ascending_and_idempotent() -> None:
    run = pd.DataFrame(
        {
            "split": ["dev"] * 3,
            "query_id": ["q"] * 3,
            "docid": ["10", "2", "1"],
            "bm25_score": [1, 1, 1],
        }
    )
    result = normalize_bm25_run(run)
    repeated = normalize_bm25_run(result)
    assert result["docid"].tolist() == ["1", "10", "2"]
    pd.testing.assert_frame_equal(result, repeated)
    validate_stable_order(result)


def test_normalize_can_cut_to_top_k() -> None:
    run = pd.DataFrame(
        {
            "split": ["train"] * 3,
            "query_id": ["q"] * 3,
            "docid": ["a", "b", "c"],
            "bm25_score": [3.0, 2.0, 1.0],
        }
    )
    result = normalize_bm25_run(run, top_k=2)
    assert result["docid"].tolist() == ["a", "b"]
    validate_top_k(result, 2)


def test_top_k_exact_validation_rejects_short_query() -> None:
    run = pd.DataFrame(
        {
            "split": ["dev"],
            "query_id": ["q"],
            "docid": ["d"],
            "bm25_rank": [1],
            "bm25_score": [1.0],
        }
    )
    with pytest.raises(ValueError, match="exactly top_k=2"):
        validate_top_k(run, 2)


def test_top_k_validation_rejects_rank_gap_and_duplicate_rank() -> None:
    gap = pd.DataFrame(
        {
            "split": ["dev", "dev"],
            "query_id": ["q", "q"],
            "docid": ["a", "b"],
            "bm25_rank": [1, 3],
            "bm25_score": [2.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="non-contiguous"):
        validate_top_k(gap, 3, require_exact=False)
    duplicate = gap.copy()
    duplicate["bm25_rank"] = [1, 1]
    with pytest.raises(ValueError, match="duplicate bm25_rank"):
        validate_top_k(duplicate, 2)


def test_query_coverage_rejects_missing_query() -> None:
    run = pd.DataFrame(
        {"split": ["dev"], "query_id": ["q1"], "docid": ["d"], "bm25_rank": [1]}
    )
    queries = pd.DataFrame({"split": ["dev", "dev"], "query_id": ["q1", "q2"]})
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_query_coverage(run, queries)


def test_qrels_join_keeps_unjudged_distinct_and_has_no_row_multiplication() -> None:
    run = pd.DataFrame(
        {
            "split": ["dev"] * 3,
            "query_id": ["q"] * 3,
            "docid": ["positive", "negative", "missing"],
            "bm25_rank": [1, 2, 3],
            "bm25_score": [3.0, 2.0, 1.0],
        }
    )
    qrels = pd.DataFrame(
        {
            "query_id": ["q", "q"],
            "docid": ["positive", "negative"],
            "relevance": [1, 0],
        }
    )
    result = join_qrels(run, qrels).set_index("docid")
    assert len(result) == len(run)
    assert result.loc["positive", "judgment"] == "relevant"
    assert result.loc["negative", "judgment"] == "non_relevant"
    assert result.loc["missing", "judgment"] == "unjudged"
    assert bool(result.loc["missing", "is_judged"]) is False
    assert pd.isna(result.loc["missing", "relevance"])


def test_duplicate_documents_are_rejected() -> None:
    run = pd.DataFrame(
        {
            "split": ["dev", "dev"],
            "query_id": ["q", "q"],
            "docid": ["d", "d"],
            "bm25_score": [2.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match=r"duplicate \(query_id, docid\)"):
        normalize_bm25_run(run)


def test_trec_round_trip_preserves_deterministic_run(tmp_path) -> None:
    run = normalize_bm25_run(
        pd.DataFrame(
            {
                "split": ["dev", "dev"],
                "query_id": ["q", "q"],
                "docid": ["b", "a"],
                "bm25_score": [1.0, 2.0],
            }
        )
    )
    path = tmp_path / "run.trec"
    write_trec_run(run, str(path))
    loaded = normalize_bm25_run(
        read_trec_run(str(path), split="dev")[[
            "split", "query_id", "docid", "bm25_score"
        ]]
    )
    pd.testing.assert_frame_equal(run, loaded)


def test_stable_top100_is_separate_and_does_not_modify_top1000(tmp_path: Path) -> None:
    source = tmp_path / "dev_bm25_top1000.trec"
    target = tmp_path / "dev_bm25_top100.trec"
    source.write_text(
        "".join(
            f"q Q0 d{docid:04d} {rank} 1.0 official\n"
            for rank, docid in enumerate(range(1000, 0, -1), start=1)
        ),
        encoding="utf-8",
    )
    before = source.read_bytes()

    top100 = _stable_truncate_trec_run(
        source, target, split="dev", top_k=100
    )

    assert source.read_bytes() == before
    assert target.is_file()
    assert top100["docid"].tolist() == [f"d{rank:04d}" for rank in range(1, 101)]
    assert top100["bm25_rank"].tolist() == list(range(1, 101))
    validate_stable_order(top100)


def _write_cache_gate_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_path = tmp_path / "artifacts/runs/dev_bm25_top1000.trec"
    audit_path = tmp_path / "reports/audit/bm25_reproduction.json"
    run_path.parent.mkdir(parents=True)
    audit_path.parent.mkdir(parents=True)
    run_path.write_text("q Q0 d 1 1.0 official\n", encoding="utf-8")
    config = {
        "dataset": {},
        "environment": {},
        "retrieval": {},
        "artifacts": {
            "dev_run": "artifacts/runs/dev_bm25_top1000.trec",
        },
        "audits": {
            "reproduction": "reports/audit/bm25_reproduction.json",
        },
        "archive": {},
        "reproduction_gate": {},
    }
    config_path = tmp_path / "retrieval.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, run_path, audit_path


@pytest.mark.parametrize(
    "audit",
    [
        None,
        {"status": "FAIL", "gate_passed": False},
    ],
)
def test_candidate_cache_rejects_missing_or_failed_reproduction_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audit: dict[str, object] | None,
) -> None:
    config_path, _, audit_path = _write_cache_gate_config(tmp_path)
    if audit is not None:
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "_require_external_environment", lambda config: {})
    assert main(["build-candidate-cache", "--config", str(config_path)]) == 1


def test_candidate_cache_rejects_hash_from_another_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _, audit_path = _write_cache_gate_config(tmp_path)
    audit_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "gate_passed": True,
                "source_run": "artifacts/runs/dev_bm25_top1000.trec",
                "source_run_sha256": hashlib.sha256(b"another run").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "_require_external_environment", lambda config: {})
    assert main(["build-candidate-cache", "--config", str(config_path)]) == 1


def test_modifying_top1000_after_evaluation_blocks_candidate_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, run_path, audit_path = _write_cache_gate_config(tmp_path)
    evaluated_hash = hashlib.sha256(run_path.read_bytes()).hexdigest()
    audit_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "gate_passed": True,
                "source_run": "artifacts/runs/dev_bm25_top1000.trec",
                "source_run_sha256": evaluated_hash,
            }
        ),
        encoding="utf-8",
    )
    run_path.write_text(
        run_path.read_text(encoding="utf-8") + "q Q0 changed 2 0.5 altered\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "_require_external_environment", lambda config: {})
    assert main(["build-candidate-cache", "--config", str(config_path)]) == 1


def test_package_phase1_uses_an_explicit_portable_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    candidate_dir = Path("artifacts/candidates")
    run_dir = Path("artifacts/runs")
    audit_dir = Path("reports/audit")
    candidate_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)

    frames: dict[str, pd.DataFrame] = {}
    for split, query_id in (("train", "qt"), ("dev", "qd")):
        frame = pd.DataFrame(
            {
                "split": [split] * 100,
                "query_id": [query_id] * 100,
                "docid": [f"d{rank:03d}" for rank in range(1, 101)],
                "bm25_rank": list(range(1, 101)),
                "bm25_score": [float(1001 - rank) for rank in range(1, 101)],
                "relevance_grade": pd.Series([pd.NA] * 100, dtype="Int64"),
                "judgment": ["unjudged"] * 100,
                "relevance": pd.Series([pd.NA] * 100, dtype="Int64"),
                "is_judged": [False] * 100,
            }
        )
        frame.to_parquet(candidate_dir / f"{split}_top100.parquet", index=False)
        frames[split] = frame

    train_run_path = run_dir / "train_bm25_top100.trec"
    dev_top1000_path = run_dir / "dev_bm25_top1000.trec"
    dev_top100_path = run_dir / "dev_bm25_top100.trec"
    write_trec_run(frames["train"], str(train_run_path))
    dev_top1000 = normalize_bm25_run(
        pd.DataFrame(
            {
                "split": ["dev"] * 1000,
                "query_id": ["qd"] * 1000,
                "docid": [f"d{rank:03d}" for rank in range(1, 1001)],
                "bm25_score": [float(1001 - rank) for rank in range(1, 1001)],
            }
        )
    )
    write_trec_run(dev_top1000, str(dev_top1000_path), tag="official")
    _stable_truncate_trec_run(
        dev_top1000_path, dev_top100_path, split="dev", top_k=100
    )

    pd.DataFrame(
        {
            "split": ["train", "dev"],
            "query_id": ["qt", "qd"],
            "query_text": ["train query", "dev query"],
        }
    ).to_parquet(candidate_dir / "queries.parquet", index=False)
    pd.DataFrame(
        {
            "docid": [f"d{rank:03d}" for rank in range(1, 101)],
            "title": [""] * 100,
            "text": [f"text {rank}" for rank in range(1, 101)],
        }
    ).to_parquet(candidate_dir / "passages.parquet", index=False)
    (audit_dir / "bm25.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "gate_passed": True,
                "source_run": str(dev_top1000_path),
                "source_run_sha256": hashlib.sha256(
                    dev_top1000_path.read_bytes()
                ).hexdigest(),
                "official_tool_values": {},
            }
        ),
        encoding="utf-8",
    )
    (audit_dir / "qrels.json").write_text(
        json.dumps({"status": "COMPLETED"}), encoding="utf-8"
    )
    Path("artifacts/cache").mkdir()
    Path("artifacts/cache/secret-index.bin").write_bytes(b"must not be packaged")

    config = {
        "dataset": {
            "name": "miracl",
            "language": "ru",
            "corpus_revision": "corpus-revision",
            "annotations_revision": "annotations-revision",
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
            "candidate_depth": 100,
            "retrieval_hits": {"train": 100, "dev": 1000},
            "threads": 16,
            "batch_size": 128,
            "bm25_k1": 0.9,
            "bm25_b": 0.4,
        },
        "artifacts": {
            "train_candidates": str(candidate_dir / "train_top100.parquet"),
            "dev_candidates": str(candidate_dir / "dev_top100.parquet"),
            "queries": str(candidate_dir / "queries.parquet"),
            "passages": str(candidate_dir / "passages.parquet"),
            "train_run": str(train_run_path),
            "dev_run": str(dev_top1000_path),
            "dev_top100_run": str(dev_top100_path),
        },
        "audits": {
            "reproduction": str(audit_dir / "bm25.json"),
            "qrels": str(audit_dir / "qrels.json"),
            "manifest": str(audit_dir / "manifest.json"),
        },
        "archive": {"path": "artifacts/results.zip", "include_runs": True},
        "reproduction_gate": {
            "official_ndcg_at_10": 0.334,
            "official_recall_at_100": 0.661,
            "official_retrieval_command": "official command",
        },
        "paths": {"work_dir": "artifacts/work"},
    }
    config_path = Path("retrieval.yaml")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["package-phase1", "--config", str(config_path), "--overwrite"]) == 0
    with zipfile.ZipFile("artifacts/results.zip") as archive:
        names = set(archive.namelist())
    assert "artifacts/cache/secret-index.bin" not in names
    assert names == {
        "artifacts/candidates/train_top100.parquet",
        "artifacts/candidates/dev_top100.parquet",
        "artifacts/candidates/queries.parquet",
        "artifacts/candidates/passages.parquet",
        "artifacts/runs/train_bm25_top100.trec",
        "artifacts/runs/dev_bm25_top1000.trec",
        "artifacts/runs/dev_bm25_top100.trec",
        "reports/audit/bm25.json",
        "reports/audit/qrels.json",
        "reports/audit/manifest.json",
    }
    manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
    hashed_paths = {entry["path"] for entry in manifest["artifacts"]}
    assert hashed_paths == names.difference({"reports/audit/manifest.json"})
    assert all("sha256" in entry and "size_bytes" in entry for entry in manifest["artifacts"])
    for entry in manifest["artifacts"]:
        artifact = Path(entry["path"])
        assert entry["size_bytes"] == artifact.stat().st_size
        assert entry["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert manifest["provenance_chain"]["flow"] == [
        "dev top-1000 run",
        "official evaluation PASS",
        "stable truncation (bm25_score DESC, docid ASC; ranks 1..100)",
        "dev top-100 run",
        "three-state candidate cache",
    ]


def test_colab_runner_gates_top100_cache_after_official_dev_top1000() -> None:
    repository = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repository / "scripts/run_full_bm25_retrieval.ipynb").read_text(
            encoding="utf-8"
        )
    )
    cells = ["".join(cell["source"]) for cell in notebook["cells"]]
    assert len(cells) == 14
    assert "'-m', 'pytest', '-q'" in cells[5]
    assert "'--split', 'train'" in cells[7]
    assert "'--split', 'dev'" in cells[8]
    assert "evaluate-bm25" in cells[9]
    assert "build-candidate-cache" in cells[10]

    config = yaml.safe_load((repository / "configs/retrieval.yaml").read_text())
    assert config["retrieval"]["retrieval_hits"] == {"train": 100, "dev": 1000}
    assert config["retrieval"]["candidate_depth"] == 100
    assert config["artifacts"]["dev_run"].endswith("dev_bm25_top1000.trec")
    assert config["artifacts"]["dev_top100_run"].endswith("dev_bm25_top100.trec")
