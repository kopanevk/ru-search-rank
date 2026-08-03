#!/usr/bin/env python3
"""Static, offline validation for the frozen Phase 3 Colab protocol."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "scripts/run_finetune.ipynb"
CONFIG = REPOSITORY / "configs/finetune.yaml"
SOURCE_TREE = REPOSITORY / "src"
EXPECTED_CELLS = 19


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    require(
        isinstance(cells, list) and len(cells) == EXPECTED_CELLS,
        f"Phase 3 notebook must contain exactly {EXPECTED_CELLS} cells",
    )
    require(cells[0].get("cell_type") == "markdown", "Cell 1 must be markdown")
    sources = ["".join(cell.get("source", [])) for cell in cells]
    for number, cell in enumerate(cells[1:], start=2):
        require(cell.get("cell_type") == "code", f"Cell {number} must be code")
        require(cell.get("execution_count") is None, f"Cell {number} has execution_count")
        require(not cell.get("outputs"), f"Cell {number} contains saved outputs")
        ast.parse(sources[number - 1], filename=f"Cell {number}")

    required_by_cell = {
        1: [
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "1427fd652930e4ba29e8149678df786c240d8825",
            "Colab-GPU-only",
        ],
        2: [
            "platform.system() != 'Linux'",
            "nvidia-smi",
            "MIN_FREE_GIB",
            "MIN_RAM_GIB",
            "available_ram",
        ],
        3: [
            "BRANCH = 'phase-3'",
            "ALLOW_OVERWRITE_PHASE3 = False",
            "RESUME_TRAINING = True",
            "def run_checked(",
            "def training_options(",
            "return code",
            "complete log",
            "git', 'pull', '--ff-only'",
        ],
        4: [
            "openjdk-21-jdk-headless",
            "TREC_EVAL_TAG = 'v9.0.8'",
            "TREC_EVAL_COMMIT = 'd95ca64e14a47d763ae349fb65e6d8cde4141dbd'",
            "shutil.rmtree(TREC_EVAL_DIR)",
            "'/opt/bin/trec_eval', '-v'",
            "makefile_sha256_before",
            "makefile_sha256_after",
            "trec_eval_build_provenance.json",
            "known_upstream_version_string_mismatch",
            "'fresh_checkout': True",
            "binary_sha256",
        ],
        5: [
            "sys.version_info[:2] != (3, 12)",
            "venv.EnvBuilder",
            "pip', 'install', '-e', '.'",
            "sentencepiece.bpe.model",
            "torch.cuda.is_available()",
        ],
        6: [
            "pytest', '-q'",
            "validate_phase3_notebook.py",
            "transformers.__version__",
            "tokenizers.__version__",
        ],
        7: [
            "PHASE1_ZIP",
            "PHASE2_ZIP",
            "archive.testzip()",
            "archive.extractall",
            "entry['sha256']",
            "prepare-annotations",
            "'--split', 'train'",
        ],
        8: [
            "phase12_immutable_snapshot",
            "verify_phase12_immutable",
            "build-training-split",
            "build-training-pairs",
            "usable_query_count",
            "population_disclosure",
            "weight_disclosure",
            "heuristic_disclosure",
        ],
        9: [
            "validate-checkpoint",
            "'base'",
        ],
        10: [
            "smoke-finetune",
            "real_model_forward",
            "fixture_only",
            "FINETUNE_SMOKE_PASSED = True",
            "estimated_training_time_range_seconds",
        ],
        11: ["'finetune'", "'C1'", "BLOCKED_FOR_REVIEW"],
        12: ["'finetune'", "'A1'"],
        13: ["'finetune'", "'A2'"],
        14: ["'finetune'", "'B1'"],
        15: [
            "select-checkpoint",
            "checkpoint_selection.json",
            "validation_ab_comparison.json",
            "exploratory_post_selection_ab",
        ],
        16: ["prepare-dev-evaluation"],
        17: ["score-finetuned"],
        18: ["evaluate-phase3", "pipeline_status", "ml_outcome"],
        19: [
            "package-phase3",
            "rusearchrank_phase3_results.zip",
            "streaming_sha256",
            "shutil.copy2",
            "files.download",
        ],
    }
    for number, fragments in required_by_cell.items():
        for fragment in fragments:
            require(fragment in sources[number - 1], f"Cell {number} is missing {fragment!r}")

    code = "\n".join(sources[1:])
    for banned in ("load_dataset", "trust_remote_code", "pyserini.eval"):
        require(banned not in code, f"notebook contains banned {banned!r}")
    for banned_command in (
        "run-bm25",
        "build-candidate-cache",
        "rerank-score",
        "evaluate-rerank",
    ):
        require(banned_command not in code, f"notebook attempts forbidden rebuild {banned_command}")

    ordered_commands = (
        "smoke-finetune",
        "'C1'",
        "'A1'",
        "'A2'",
        "'B1'",
        "select-checkpoint",
        "prepare-dev-evaluation",
        "score-finetuned",
        "evaluate-phase3",
        "package-phase3",
    )
    positions = [code.index(fragment) for fragment in ordered_commands]
    require(positions == sorted(positions), "Phase 3 command ordering changed")

    # No source-evaluation annotation or scoring path may appear before selection.
    preselection = "\n".join(sources[:14])
    for forbidden in (
        "qrels.miracl-v1.0-ru-dev",
        "dev_top100",
        "dev_bm25",
        "dev_rerank",
        "dev_zeroshot",
        "score-finetuned",
        "evaluate-phase3",
    ):
        require(forbidden not in preselection, f"pre-selection cell exposes {forbidden!r}")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    exact = {
        ("implementation", "version"): "3.0.0",
        ("input", "max_length"): 320,
        ("input", "truncation"): "only_second",
        ("split", "seed"): 20260803,
        ("split", "min_stratum_size"): 20,
        ("training", "epochs"): 3,
        ("training", "micro_batch_queries"): 1,
        ("training", "grad_accumulation"): 16,
        ("training", "precision"): "fp32",
        ("training", "device"): "cuda",
        ("training", "token_cache"): "pretokenized_flat_int32",
    }
    for (section, key), expected in exact.items():
        require(config[section][key] == expected, f"{section}.{key} changed")
    for key in ("revision", "tokenizer_revision"):
        require(
            re.fullmatch(r"[0-9a-f]{40}", str(config["base_model"][key])) is not None,
            f"base_model.{key} must be an immutable SHA",
        )
    require(
        {config["runs"]["A1"]["learning_rate"], config["runs"]["A2"]["learning_rate"]}
        == {7.0e-6, 2.0e-5},
        "judged-only LR set changed",
    )
    golden = json.loads(
        (REPOSITORY / "tests/fixtures/pair_encoding_golden.json").read_text(encoding="utf-8")
    )
    require(len(golden.get("cases", [])) == 64, "encoding golden must contain 64 pairs")
    require(
        golden.get("model_id") == config["base_model"]["id"]
        and golden.get("tokenizer_revision")
        == config["base_model"]["tokenizer_revision"],
        "config model/tokenizer revision differs from the encoding golden",
    )
    require(
        set(golden.get("tokenizer_payload_sha256", {}))
        == {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        }
        and all(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is not None
            for value in golden["tokenizer_payload_sha256"].values()
        ),
        "encoding golden must bind the complete pinned tokenizer payload",
    )

    forbidden_training_literals = {
        "dev_top100",
        "dev_bm25",
        "dev_rerank",
        "miracl-v1.0-ru-dev",
    }
    for relative in (
        Path("src/rusearchrank/training.py"),
        Path("src/rusearchrank/training_data.py"),
    ):
        tree = ast.parse((REPOSITORY / relative).read_text(encoding="utf-8"), filename=str(relative))
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        require(
            not constants.intersection(forbidden_training_literals),
            f"{relative} contains a forbidden evaluation-path literal",
        )

    for module in sorted(SOURCE_TREE.rglob("*.py")):
        relative = module.relative_to(REPOSITORY)
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                require(name != "load_dataset", f"{relative}:{node.lineno} calls load_dataset")
                require(
                    all(keyword.arg != "trust_remote_code" for keyword in node.keywords),
                    f"{relative}:{node.lineno} enables remote code",
                )

    print(
        json.dumps(
            {
                "status": "PASS",
                "notebook": str(NOTEBOOK.relative_to(REPOSITORY)),
                "cells": len(cells),
                "branch": "phase-3",
                "smoke_cell": 10,
                "control_cell": 11,
                "selection_cell": 15,
                "first_evaluation_access_cell": 16,
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
        print(f"Phase 3 notebook validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
