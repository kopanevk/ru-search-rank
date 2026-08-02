#!/usr/bin/env python3
"""Static, offline validation for the exact Phase 2 Colab protocol."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "scripts/run_zeroshot_rerank.ipynb"
CONFIG = REPOSITORY / "configs/rerank.yaml"
SOURCE_TREE = REPOSITORY / "src"
EXPECTED_CELLS = 14


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    require(
        isinstance(cells, list) and len(cells) == EXPECTED_CELLS,
        f"Phase 2 notebook must contain exactly {EXPECTED_CELLS} cells",
    )
    require(cells[0].get("cell_type") == "markdown", "Cell 1 must be markdown")
    sources = ["".join(cell.get("source", [])) for cell in cells]
    for number, cell in enumerate(cells[1:], start=2):
        require(cell.get("cell_type") == "code", f"Cell {number} must be code")
        require(
            cell.get("execution_count") is None,
            f"Cell {number} contains a saved execution count",
        )
        require(not cell.get("outputs"), f"Cell {number} contains saved outputs")
        ast.parse(sources[number - 1], filename=f"Cell {number}")

    required_by_cell = {
        1: [
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "1427fd652930e4ba29e8149678df786c240d8825",
            "125,200",
        ],
        2: ["platform.system() != 'Linux'", "nvidia-smi", "MIN_FREE_GIB", "MIN_RAM_GIB"],
        3: [
            "def run_checked(",
            "stream=False",
            "return code",
            "stdout",
            "stderr",
            "complete log",
            "BRANCH = 'phase-2'",
            "ALLOW_OVERWRITE_PHASE2 = False",
        ],
        4: [
            "openjdk-21-jdk-headless",
            "TREC_EVAL_TAG = 'v9.0.8'",
            "TREC_EVAL_COMMIT = 'd95ca64e14a47d763ae349fb65e6d8cde4141dbd'",
            "shutil.rmtree(TREC_EVAL_DIR)",
            "'clone', '--depth', '1', '--branch', TREC_EVAL_TAG",
            "rev-list",
            "'diff', '--quiet', 'HEAD'",
            "'diff', '--cached', '--quiet'",
            "makefile_sha256_before",
            "makefile_sha256_after",
            "'/opt/bin/trec_eval', '-v'",
            "9\\.0\\.7",
            "binary_sha256",
            "trec_eval_build_provenance.json",
            "known_upstream_version_string_mismatch",
            "'fresh_checkout': True",
            "'makefile_sha256': makefile_sha256_after",
            "shutil.which('trec_eval')",
        ],
        5: ["3.12", "RUN_PYTHON", "pip_install_project"],
        6: [
            "'-m', 'pytest', '-q'",
            "validate_phase1_notebook.py",
            "validate_phase2_notebook.py",
            "'torch'",
            "'transformers'",
            "'tokenizers'",
            "'huggingface-hub'",
            "'pyarrow'",
            "'pandas'",
        ],
        7: ["PHASE1_ZIP", "candidate_cache_manifest.json", "archive.extractall", "prepare-annotations"],
        8: ["'preflight'", "'rerank'", "CONFIG"],
        9: ["'smoke-rerank'", "'64'", "RERANK_SMOKE_PASSED", "real_model_forward", "fixture_only"],
        10: ["RERANK_SMOKE_PASSED", "'rerank-score'", "'--split', 'dev'", "stream=True"],
        11: ["'build-rerank-run'", "(100, 10, 20, 50)"],
        12: ["'evaluate-rerank'", "stream=True"],
        13: ["'package-phase2'", "stream=True"],
        14: ["rusearchrank_phase2_results.zip", "hashlib.sha256", "archive.namelist", "files.download"],
    }
    for number, fragments in required_by_cell.items():
        for fragment in fragments:
            require(
                fragment in sources[number - 1],
                f"Cell {number} is missing {fragment!r}",
            )
    require(
        "trec_eval', '-h'" not in sources[3],
        "Cell 4 must probe the actual trec_eval version with -v, not -h help.",
    )
    require(
        not re.search(r"(?:sed|patch).*Makefile|Makefile.*(?:sed|patch)", sources[3]),
        "Cell 4 must not patch or rewrite the upstream Makefile.",
    )
    require(
        "'fetch'" not in sources[3]
        and "trec_eval_fetch" not in sources[3]
        and "if not (TREC_EVAL_DIR / '.git').is_dir()" not in sources[3],
        "Cell 4 must always discard the old checkout and perform a fresh clone.",
    )
    require(
        "['git', 'checkout'" not in sources[3],
        "Cell 4 fresh tag clone must not reuse or switch an existing checkout.",
    )

    code_source = "\n".join(sources[1:])
    for banned in ("load_dataset", "trust_remote_code", "pyserini.eval"):
        require(banned not in code_source, f"Phase 2 notebook code contains banned {banned!r}")
    for banned_command in (
        "run-bm25",
        "build-candidate-cache",
        "evaluate-bm25",
        "package-phase1",
    ):
        require(
            banned_command not in code_source,
            f"Phase 2 notebook attempts forbidden Phase 1 rebuild command {banned_command!r}",
        )
    require(
        "smoke-rerank" in sources[8] and "rerank-score" in sources[9],
        "real smoke must immediately precede full rerank scoring",
    )
    require(
        "evaluate-rerank" in sources[11] and "package-phase2" in sources[12],
        "evaluation must precede Phase 2 packaging",
    )
    for number in (10, 11, 12, 13):
        require(
            "ALLOW_OVERWRITE_PHASE2" in sources[number - 1],
            f"Cell {number} must gate overwrite behind ALLOW_OVERWRITE_PHASE2",
        )

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require(config["model"]["id"] in sources[0], "notebook model id differs from config")
    require(
        config["model"]["revision"] in sources[0],
        "notebook pinned model revision differs from config",
    )
    require(
        config["model"]["revision"] == config["model"]["tokenizer_revision"],
        "model/tokenizer revisions must be identical",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", str(config["model"]["revision"])) is not None,
        "model revision must be an immutable SHA",
    )
    require(config["input"]["max_length"] == 320, "max_length changed")
    require(config["input"]["truncation"] == "only_second", "truncation changed")
    require(config["protocol"]["official_depth"] == 100, "official depth changed")
    require(
        config["protocol"]["diagnostic_depths"] == [10, 20, 50],
        "diagnostic depths changed",
    )
    evaluation = config["evaluation"]
    require(
        evaluation["trec_eval_executable"] == "/opt/bin/trec_eval",
        "Phase 2 must use the provenance-bound /opt/bin/trec_eval",
    )
    require(
        evaluation["trec_eval_expected_release"] == "9.0.8"
        and evaluation["trec_eval_expected_source_tag"] == "v9.0.8"
        and evaluation["trec_eval_expected_source_commit"]
        == "d95ca64e14a47d763ae349fb65e6d8cde4141dbd"
        and evaluation["trec_eval_expected_reported_version"] == "9.0.7"
        and evaluation["trec_eval_known_version_string_mismatch"] is True,
        "trec_eval release/source/reported-version contract changed",
    )

    # Production modules are inspected syntactically: comments and audit keys may
    # name a forbidden mechanism, but executable imports/calls/keywords may not.
    for module in sorted(SOURCE_TREE.rglob("*.py")):
        relative = module.relative_to(REPOSITORY)
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
                require(name != "load_dataset", f"{relative}:{node.lineno} calls load_dataset")
                require(
                    all(keyword.arg != "trust_remote_code" for keyword in node.keywords),
                    f"{relative}:{node.lineno} enables remote model code",
                )
            if isinstance(node, ast.ImportFrom):
                require(node.module != "datasets", f"{relative}:{node.lineno} imports datasets")
            if isinstance(node, ast.Import):
                require(
                    all(alias.name.split(".")[0] != "datasets" for alias in node.names),
                    f"{relative}:{node.lineno} imports datasets",
                )

    print(
        json.dumps(
            {
                "status": "PASS",
                "notebook": str(NOTEBOOK.relative_to(REPOSITORY)),
                "cells": len(cells),
                "branch": "phase-2",
                "preflight_cell": 8,
                "real_smoke_cell": 9,
                "heavy_scoring_cell": 10,
                "evaluation_cell": 12,
                "package_cell": 13,
                "model_id": config["model"]["id"],
                "revision": config["model"]["revision"],
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
        print(f"Phase 2 notebook validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
