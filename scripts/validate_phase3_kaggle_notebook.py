#!/usr/bin/env python3
"""Static, offline validation for the Phase 3 Kaggle production runner."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "scripts/run_finetune_kaggle.ipynb"
SOURCE_NOTEBOOK = REPOSITORY / "scripts/run_finetune.ipynb"
CONFIG = REPOSITORY / "configs/finetune.yaml"
GOLDEN = REPOSITORY / "tests/fixtures/pair_encoding_golden.json"
EXPECTED_SOURCE_NOTEBOOK_SHA256 = (
    "c36ec1861221f711e4275c514e52696ee8f4a83bde7574c178c9f47cc6afc4d3"
)
EXPECTED_CELLS = 21


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    require(NOTEBOOK.is_file(), f"missing Kaggle notebook: {NOTEBOOK}")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    require(
        isinstance(cells, list) and len(cells) == EXPECTED_CELLS,
        f"Kaggle notebook must contain exactly {EXPECTED_CELLS} cells",
    )
    require(cells[0].get("cell_type") == "markdown", "cell 1 must be markdown")
    sources = ["".join(cell.get("source", [])) for cell in cells]
    for number, cell in enumerate(cells[1:], start=2):
        require(cell.get("cell_type") == "code", f"cell {number} must be code")
        require(cell.get("execution_count") is None, f"cell {number} has execution_count")
        require(not cell.get("outputs"), f"cell {number} contains saved outputs")
        ast.parse(sources[number - 1], filename=f"cell {number}")

    source_notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    source_sources = ["".join(cell.get("source", [])) for cell in source_notebook["cells"]]
    require(
        sha256_file(SOURCE_NOTEBOOK) == EXPECTED_SOURCE_NOTEBOOK_SHA256,
        "the frozen scripts/run_finetune.ipynb changed",
    )
    require(
        sources[8:19] == source_sources[7:18],
        "training split through evaluate-phase3 must be copied exactly from the frozen notebook",
    )

    required_by_cell = {
        1: [
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "1427fd652930e4ba29e8149678df786c240d8825",
            "cuda:0",
            "Save & Run All",
            "RESUME_TRAINING = True",
        ],
        2: [
            "platform.system() != 'Linux'",
            "Path('/kaggle/working')",
            "Path('/kaggle/input')",
            "nvidia-smi",
            "torch.__version__",
            "torch.version.cuda",
            "torch.cuda.is_available()",
            "torch.device('cuda:0')",
            "compute_capability",
            "vram_total_bytes",
            "vram_free_bytes",
            "cpu_ram_total_bytes",
            "cpu_ram_available_bytes",
            "MIN_TOTAL_RAM_GIB = 12",
            "MIN_FREE_DISK_GIB = 25",
            "WARNING",
        ],
        3: [
            "BRANCH = 'phase-3-kaggle'",
            "Path('/kaggle/working/ru-search-rank')",
            "ALLOW_OVERWRITE_PHASE3 = False",
            "RESUME_TRAINING = True",
            "def run_checked(",
            "def training_options(",
            "['--resume']",
            "['--overwrite']",
            "git', 'pull', '--ff-only'",
        ],
        4: [
            "openjdk-21-jdk-headless",
            "TREC_EVAL_TAG = 'v9.0.8'",
            "TREC_EVAL_COMMIT = 'd95ca64e14a47d763ae349fb65e6d8cde4141dbd'",
            "trec_eval_build_provenance.json",
            "known_upstream_version_string_mismatch",
        ],
        5: [
            "Path('/kaggle/working/rusearchrank-phase3-venv')",
            "sys.version_info[:2] == (3, 12)",
            "'uv', 'venv', '--python', '3.12'",
            "pip', 'install', '-U', 'pip'",
            "pip', 'install', '-e', '.'",
            "RUN_PYTHON",
        ],
        6: [
            "tests/fixtures/pair_encoding_golden.json",
            "tokenizer_payload_sha256",
            "Path('/kaggle/working/rusearchrank-pinned-tokenizer')",
            "hf_hub_download",
            "RUSEARCHRANK_PINNED_TOKENIZER_DIR",
            "local_files_only=True",
            "pytest', '-q'",
            "transformers.__version__",
            "tokenizers.__version__",
        ],
        7: [
            "validate_phase1_notebook.py",
            "validate_phase2_notebook.py",
            "validate_phase3_notebook.py",
        ],
        8: [
            "KAGGLE_INPUT.rglob('*.zip')",
            "candidate_cache_manifest.json",
            "rerank_manifest.json",
            "archive.testzip()",
            "Path('/kaggle/working/phase12-inputs')",
            "streaming_sha256",
            "archive.extractall(REPOSITORY)",
            "entry['sha256']",
            "prepare-annotations",
            "'--split', 'train'",
        ],
        9: ["build-training-split", "build-training-pairs", "phase12_immutable_snapshot"],
        10: ["validate-checkpoint", "'base'"],
        11: ["smoke-finetune", "FINETUNE_SMOKE_PASSED = True"],
        12: ["'finetune'", "'C1'", "BLOCKED_FOR_REVIEW"],
        13: ["'finetune'", "'A1'"],
        14: ["'finetune'", "'A2'"],
        15: ["'finetune'", "'B1'"],
        16: ["select-checkpoint", "checkpoint_selection.json"],
        17: ["prepare-dev-evaluation"],
        18: ["score-finetuned"],
        19: ["evaluate-phase3", "ml_outcome"],
        20: ["package-phase3"],
        21: [
            "Path('/kaggle/working/phase3-final')",
            "rusearchrank_phase3_results.zip",
            "rusearchrank_phase3_model_*.zip",
            "len(model_zips) != 1",
            "size_bytes",
            "size_mib",
            "sha256",
        ],
    }
    for number, fragments in required_by_cell.items():
        for fragment in fragments:
            require(fragment in sources[number - 1], f"cell {number} is missing {fragment!r}")

    code_text = "\n".join(sources[1:])
    lower_code = code_text.lower()
    for banned in (
        "google.colab",
        "google drive",
        "/content/drive",
        "torch_xla",
        "data_parallel",
        "dataparallel",
        "distributeddata",
        "cuda:1",
    ):
        require(banned not in lower_code, f"notebook contains banned infrastructure {banned!r}")
    require(re.search(r"\bddp\b", lower_code) is None, "notebook contains banned DDP")
    require(re.search(r"\btpu\b", lower_code) is None, "notebook contains banned TPU")
    require(re.search(r"\bxla\b", lower_code) is None, "notebook contains banned XLA")
    for banned_command in (
        "run-bm25",
        "build-candidate-cache",
        "rerank-score",
        "evaluate-rerank",
        "package-phase1",
        "package-phase2",
    ):
        require(banned_command not in code_text, f"forbidden Phase 1/2 rebuild: {banned_command}")

    ordered_fragments = (
        "build-training-split",
        "build-training-pairs",
        "validate-checkpoint",
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
        "FINAL_OUTPUT = Path('/kaggle/working/phase3-final')",
    )
    positions = [code_text.index(fragment) for fragment in ordered_fragments]
    require(positions == sorted(positions), "Kaggle production stage ordering changed")

    selection_cell = 16
    preselection = "\n".join(sources[1 : selection_cell - 1])
    for forbidden in (
        "qrels.miracl-v1.0-ru-dev",
        "artifacts/scores/dev_finetuned.parquet",
        "score-finetuned",
        "evaluate-phase3",
    ):
        require(forbidden not in preselection, f"pre-selection cells expose {forbidden!r}")
    require("prepare-dev-evaluation" not in preselection, "dev access occurs before selection")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    model_id = config["base_model"]["id"]
    model_revision = config["base_model"]["revision"]
    tokenizer_revision = config["base_model"]["tokenizer_revision"]
    require(model_id == golden["model_id"], "golden model id differs from config")
    require(model_revision == golden["model_revision"], "golden model revision differs from config")
    require(
        tokenizer_revision == golden["tokenizer_revision"],
        "golden tokenizer revision differs from config",
    )
    require(model_id in sources[0], "notebook description has the wrong model id")
    require(model_revision in sources[0], "notebook description has the wrong pinned revision")
    require(
        set(golden["tokenizer_payload_sha256"])
        == {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        },
        "tokenizer golden payload changed",
    )
    for field, expected in (
        (("input", "max_length"), 320),
        (("input", "truncation"), "only_second"),
        (("training", "precision"), "fp32"),
        (("training", "device"), "cuda"),
        (("training", "micro_batch_queries"), 1),
        (("training", "grad_accumulation"), 16),
        (("training", "seed"), 20260803),
    ):
        section, key = field
        require(config[section][key] == expected, f"frozen {section}.{key} changed")

    print(
        json.dumps(
            {
                "status": "PASS",
                "notebook": str(NOTEBOOK.relative_to(REPOSITORY)),
                "cells": len(cells),
                "source_notebook_sha256": EXPECTED_SOURCE_NOTEBOOK_SHA256,
                "copied_source_cells": "8-18",
                "selection_cell": selection_cell,
                "first_dev_access_cell": 17,
                "final_copy_cell": 21,
                "device": "cuda:0",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"Phase 3 Kaggle notebook validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
