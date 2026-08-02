from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pandas as pd

import rusearchrank.cli as cli_module
from rusearchrank.data import CANDIDATE_COLUMNS, extract_passages_from_rows
from rusearchrank.evaluation import build_qrels_split_audit
from rusearchrank.retrieval import (
    build_retrieval_depth_audit,
    join_qrels,
    normalize_bm25_run,
    read_trec_run,
    validate_query_coverage,
    validate_top_k,
    write_trec_run,
)


def test_small_phase1_fixture_end_to_end(tmp_path: Path) -> None:
    """Run the portable Phase 1 transformations without Java, network, or Pyserini."""

    config_path = tmp_path / "configs/retrieval.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("fixture: true\n", encoding="utf-8")
    config = {"_config_path": str(config_path), "paths": {"repository_root": ".."}}
    run_dir = tmp_path / "artifacts/runs"
    candidate_dir = tmp_path / "artifacts/candidates"
    audit_dir = tmp_path / "reports/audit"
    for directory in (run_dir, candidate_dir, audit_dir):
        directory.mkdir(parents=True)

    topics = pd.DataFrame(
        {
            "split": ["dev", "dev", "dev"],
            "query_id": ["q1", "q2", "q3"],
            "query_text": ["один", "два", "три"],
        }
    )
    raw_rows = []
    for query_id, count in (("q1", 3), ("q2", 2), ("q3", 3)):
        for rank in range(1, count + 1):
            raw_rows.append(
                {
                    "split": "dev",
                    "query_id": query_id,
                    "docid": f"{query_id}-d{rank}",
                    "bm25_score": float(10 - rank),
                }
            )
    raw = normalize_bm25_run(pd.DataFrame(raw_rows))
    dev_top1000 = run_dir / "dev_bm25_top1000.trec"
    dev_top100 = run_dir / "dev_bm25_top100.trec"
    train_top100 = run_dir / "train_bm25_top100.trec"
    write_trec_run(raw, str(dev_top1000), tag="fixture-official")
    parsed = read_trec_run(str(dev_top1000), split="dev")
    parsed_ranked = parsed.rename(columns={"source_rank": "bm25_rank"})
    validate_top_k(parsed_ranked, 3, require_exact=False)
    depth = build_retrieval_depth_audit(parsed_ranked, topics, requested_top_k=3)
    assert depth["short_query_count"] == 1
    assert depth["total_shortfall"] == 1
    validate_query_coverage(parsed_ranked, topics)

    stable = cli_module._stable_truncate_trec_run(
        dev_top1000,
        dev_top100,
        split="dev",
        top_k=2,
        source_top_k=3,
    )
    train_run = normalize_bm25_run(
        pd.DataFrame(
            {
                "split": ["train", "train"],
                "query_id": ["qt", "qt"],
                "docid": ["q1-d1", "qt-unjudged"],
                "bm25_score": [2.0, 1.0],
            }
        )
    )
    write_trec_run(train_run, str(train_top100), tag="fixture-train")

    dev_qrels = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "docid": ["q1-d1", "q1-d2", "q2-d1"],
            "relevance_grade": [1, 0, 1],
        }
    )
    train_qrels = pd.DataFrame(
        {
            "query_id": ["qt"],
            "docid": ["q1-d1"],
            "relevance_grade": [0],
        }
    )
    dev_candidates = join_qrels(stable, dev_qrels).loc[:, list(CANDIDATE_COLUMNS)]
    train_candidates = join_qrels(train_run, train_qrels).loc[
        :, list(CANDIDATE_COLUMNS)
    ]
    assert set(dev_candidates["judgment"]) == {
        "relevant",
        "judged_non_relevant",
        "unjudged",
    }
    all_queries = pd.concat(
        [
            pd.DataFrame(
                {"split": ["train"], "query_id": ["qt"], "query_text": ["train"]}
            ),
            topics,
        ],
        ignore_index=True,
    )
    candidate_docids = set(
        pd.concat([train_candidates["docid"], dev_candidates["docid"]]).astype(
            "string"
        )
    )
    corpus_rows = [
        {"docid": docid, "title": "", "text": f"текст {docid}"}
        for docid in sorted(candidate_docids)
    ]
    passages = extract_passages_from_rows(corpus_rows, candidate_docids)

    paths = {
        "train_candidates": candidate_dir / "train_top100.parquet",
        "dev_candidates": candidate_dir / "dev_top100.parquet",
        "queries": candidate_dir / "queries.parquet",
        "passages": candidate_dir / "passages.parquet",
    }
    train_candidates.to_parquet(paths["train_candidates"], index=False)
    dev_candidates.to_parquet(paths["dev_candidates"], index=False)
    all_queries.to_parquet(paths["queries"], index=False)
    passages.to_parquet(paths["passages"], index=False)

    qrels_audit = {
        "status": "PASS",
        "splits": {
            "train": build_qrels_split_audit(
                queries=all_queries.loc[all_queries["split"].eq("train")],
                qrels=train_qrels,
                candidates=train_candidates,
            ),
            "dev": build_qrels_split_audit(
                queries=all_queries.loc[all_queries["split"].eq("dev")],
                qrels=dev_qrels,
                candidates=dev_candidates,
            ),
        },
    }
    qrels_path = audit_dir / "qrels_audit.json"
    reproduction_path = audit_dir / "bm25_reproduction.json"
    qrels_path.write_text(json.dumps(qrels_audit), encoding="utf-8")
    reproduction_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "overall_pass": True,
                "source_run_sha256": cli_module._sha256(dev_top1000),
            }
        ),
        encoding="utf-8",
    )

    payload_paths = [
        dev_top1000,
        dev_top100,
        train_top100,
        paths["train_candidates"],
        paths["dev_candidates"],
        paths["queries"],
        paths["passages"],
        reproduction_path,
        qrels_path,
    ]
    input_hashes = {"fixture_raw": cli_module._sha256(dev_top1000)}
    manifest = {
        "status": "PASS",
        "retrieval_depth": {"dev_raw": depth},
        "artifacts": [
            cli_module._artifact_manifest_entry(
                config,
                path,
                producer_command="fixture pipeline",
                input_hashes=input_hashes,
            )
            for path in payload_paths
        ],
    }
    manifest_path = audit_dir / "candidate_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    archive_path = tmp_path / "artifacts/rusearchrank_phase1_results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*payload_paths, manifest_path]:
            archive.write(path, cli_module._portable_repository_path(config, path))

    report = cli_module._validate_phase1_archive(
        config,
        archive_path,
        manifest_path,
        payload_paths=payload_paths,
    )
    assert report["contents"][-1] == "reports/audit/candidate_cache_manifest.json"
    assert report["size_bytes"] > 0
