#!/usr/bin/env python3
"""Static, network-free validation for the Phase 1 Colab runner."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "scripts/run_full_bm25_retrieval.ipynb"
CONFIG = REPOSITORY / "configs/retrieval.yaml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    require(isinstance(cells, list) and len(cells) == 14, "notebook must have 14 cells")
    require(cells[0].get("cell_type") == "markdown", "Cell 1 must be markdown")
    sources = ["".join(cell.get("source", [])) for cell in cells]
    for number, cell in enumerate(cells[1:], start=2):
        require(cell.get("cell_type") == "code", f"Cell {number} must be code")
        require(cell.get("execution_count") is None, f"Cell {number} has saved execution state")
        require(not cell.get("outputs"), f"Cell {number} has saved outputs")
        ast.parse(sources[number - 1], filename=f"Cell {number}")

    full_source = "\n".join(sources)
    required_by_cell = {
        2: ["platform.system() != 'Linux'", "MIN_FREE_GIB = 30"],
        3: [
            "def run_checked(",
            "https://github.com/kopanevk/ru-search-rank.git",
            "BRANCH = 'phase-0'",
            "return code",
            "stdout",
            "stderr",
        ],
        4: [
            "openjdk-21-jdk-headless",
            "TREC_EVAL_TAG = 'v9.0.8'",
            "'rev-list', '-n', '1', TREC_EVAL_TAG",
            "trec_eval', '-h'",
        ],
        5: ["python3.12-venv", "UV_VERSION = '0.8.13'", "RUN_PYTHON"],
        6: ["'-m', 'pytest', '-q'", "[retrieval]"],
        7: ["prepare-annotations", "'preflight'", "'retrieval'", "'--check-index'"],
        8: ["'run-bm25'", "'--split', 'train'", "stream=True"],
        9: ["'run-bm25'", "'--split', 'dev'", "stream=True"],
        10: ["evaluate-bm25"],
        11: ["'candidate-cache'", "build-candidate-cache", "audit-qrels"],
        12: ["validate-candidates", "'package'"],
        13: ["package-phase1"],
        14: ["rusearchrank_phase1_results.zip", "files.download"],
    }
    for number, fragments in required_by_cell.items():
        for fragment in fragments:
            require(fragment in sources[number - 1], f"Cell {number} is missing {fragment!r}")

    require("python -m pyserini.eval.trec_eval" not in full_source, "deprecated evaluator found")
    require("OWNER/REPOSITORY" not in full_source, "manual repository placeholder found")
    require(
        sources[9].find("evaluate-bm25") >= 0
        and sources[10].find("build-candidate-cache") >= 0,
        "official evaluation must precede candidate cache",
    )
    require(
        "ALLOW_OVERWRITE_RUNS_AND_CACHE" in sources[7],
        "Cell 8 must gate raw-run overwrite behind the explicit safety flag",
    )
    require(
        "ALLOW_OVERWRITE_RUNS_AND_CACHE" in sources[8],
        "Cell 9 must gate raw-run overwrite behind the explicit safety flag",
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
    print(
        json.dumps(
            {
                "status": "PASS",
                "notebook": str(NOTEBOOK.relative_to(REPOSITORY)),
                "cells": len(cells),
                "branch": "phase-0",
                "heavy_stage_cells": [8, 9, 10, 11, 13],
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
