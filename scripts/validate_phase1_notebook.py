#!/usr/bin/env python3
"""Static, network-free validation for the Phase 1 Colab runner."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "scripts/run_full_bm25_retrieval.ipynb"
CONFIG = REPOSITORY / "configs/retrieval.yaml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


EXPECTED_CELLS = 15
SOURCE_TREE = REPOSITORY / "src"


def _cell(sources: list[str], number: int) -> str:
    """Return the source of a 1-based notebook cell number."""

    return sources[number - 1]


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    require(
        isinstance(cells, list) and len(cells) == EXPECTED_CELLS,
        f"notebook must have {EXPECTED_CELLS} cells",
    )
    require(cells[0].get("cell_type") == "markdown", "Cell 1 must be markdown")
    sources = ["".join(cell.get("source", [])) for cell in cells]
    for number, cell in enumerate(cells[1:], start=2):
        require(cell.get("cell_type") == "code", f"Cell {number} must be code")
        require(cell.get("execution_count") is None, f"Cell {number} has saved execution state")
        require(not cell.get("outputs"), f"Cell {number} has saved outputs")
        ast.parse(_cell(sources, number), filename=f"Cell {number}")

    full_source = "\n".join(sources)
    code_source = "\n".join(sources[1:])
    required_by_cell = {
        2: ["platform.system() != 'Linux'", "MIN_FREE_GIB = 30"],
        3: [
            "def run_checked(",
            "https://github.com/kopanevk/ru-search-rank.git",
            "BRANCH = 'phase-0'",
            "return code",
            "stdout",
            "stderr",
            "complete log",
            "cwd:",
            "env:",
        ],
        4: [
            "openjdk-21-jdk-headless",
            "TREC_EVAL_TAG = 'v9.0.8'",
            "'rev-list', '-n', '1', TREC_EVAL_TAG",
            "trec_eval', '-h'",
        ],
        5: ["python3.12-venv", "UV_VERSION = '0.8.13'", "RUN_PYTHON"],
        6: ["'-m', 'pytest', '-q'", "[retrieval]", "datasets", "huggingface-hub"],
        7: ["prepare-annotations", "'preflight'", "'retrieval'", "'--check-index'"],
        8: ["'run-bm25'", "'--split', 'train'", "stream=True"],
        9: ["'run-bm25'", "'--split', 'dev'", "stream=True"],
        10: ["evaluate-bm25"],
        11: [
            "REAL COLAB SMOKE",
            "smoke-corpus-access",
            "CORPUS_SMOKE_PASSED",
            "docs-0.jsonl.gz",
        ],
        12: [
            "'candidate-cache'",
            "build-candidate-cache",
            "audit-qrels",
            "CORPUS_SMOKE_PASSED",
        ],
        13: ["validate-candidates", "'package'"],
        14: ["package-phase1", "--overwrite"],
        15: ["rusearchrank_phase1_results.zip", "files.download"],
    }
    for number, fragments in required_by_cell.items():
        for fragment in fragments:
            require(
                fragment in _cell(sources, number),
                f"Cell {number} is missing {fragment!r}",
            )

    # Executable cells only: the markdown intro deliberately names what is banned.
    require("pyserini.eval" not in code_source, "evaluation must use the NIST binary")
    require("OWNER/REPOSITORY" not in full_source, "manual repository placeholder found")
    require("load_dataset" not in code_source, "notebook must not use load_dataset")
    require("trust_remote_code" not in code_source, "trust_remote_code is forbidden")
    require("miracl-corpus.py" not in code_source, "dataset script must never be used")

    # Ordering contract: gate before cache, smoke before cache, validation before ZIP.
    require(
        "evaluate-bm25" in _cell(sources, 10)
        and "smoke-corpus-access" in _cell(sources, 11)
        and "build-candidate-cache" in _cell(sources, 12),
        "official evaluation and the real smoke must precede the candidate cache",
    )
    require(
        "validate-candidates" in _cell(sources, 13)
        and "package-phase1" in _cell(sources, 14),
        "candidate validation must precede packaging",
    )
    require(
        "build-candidate-cache" not in _cell(sources, 11),
        "the smoke cell must not build the full candidate cache",
    )
    for number in (8, 9, 12):
        require(
            "ALLOW_OVERWRITE_RUNS_AND_CACHE" in _cell(sources, number),
            f"Cell {number} must gate overwrite behind the explicit safety flag",
        )

    # No production module may reach for the removed script-backed loader. The
    # check is syntactic, not textual, so comments may still name what is banned.
    for module in sorted(SOURCE_TREE.rglob("*.py")):
        relative = module.relative_to(REPOSITORY)
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                name = (
                    target.attr
                    if isinstance(target, ast.Attribute)
                    else getattr(target, "id", "")
                )
                require(
                    name != "load_dataset",
                    f"{relative}:{node.lineno} still calls load_dataset",
                )
                require(
                    all(
                        keyword.arg != "trust_remote_code" for keyword in node.keywords
                    ),
                    f"{relative}:{node.lineno} passes trust_remote_code",
                )
            if isinstance(node, ast.ImportFrom) and node.module == "datasets":
                require(
                    False,
                    f"{relative}:{node.lineno} imports from datasets",
                )
            if isinstance(node, ast.Import):
                require(
                    all(alias.name.split(".")[0] != "datasets" for alias in node.names),
                    f"{relative}:{node.lineno} imports datasets",
                )

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require(
        config["dataset"]["train_topics_path"]
        == "artifacts/raw/miracl-ru/topics.miracl-v1.0-ru-train.tsv",
        "train topics must use the local official TSV",
    )
    require(
        config["reproduction_gate"]["official_topic"] == "miracl-v1.0-ru-dev",
        "dev must use the registered official topic ID",
    )
    require(
        config["retrieval"]["retrieval_hits"] == {"train": 100, "dev": 1000},
        "retrieval depths changed",
    )
    require(config["retrieval"]["candidate_depth"] == 100, "candidate depth changed")

    shards = config["dataset"]["corpus_shard_files"]
    expected_shards = [
        f"miracl-corpus-v1.0-{config['dataset']['language']}/docs-{index}.jsonl.gz"
        for index in range(int(config["dataset"]["corpus_shard_count"]))
    ]
    require(shards == expected_shards, "corpus shards must be listed in numeric order")
    require(len(shards) == 20, "the Russian corpus has exactly 20 official shards")
    require(
        re.fullmatch(r"[0-9a-f]{40}", str(config["dataset"]["corpus_revision"])) is not None,
        "corpus_revision must be an immutable 40-character commit SHA",
    )
    require(
        str(config["dataset"]["corpus_repo_type"]) == "dataset",
        "corpus_repo_type must be dataset",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "notebook": str(NOTEBOOK.relative_to(REPOSITORY)),
                "cells": len(cells),
                "branch": "phase-0",
                "heavy_stage_cells": [8, 9, 10, 12, 14],
                "real_smoke_cell": 11,
                "corpus_shards": len(shards),
                "corpus_revision": config["dataset"]["corpus_revision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"notebook validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
