#!/usr/bin/env python3
"""Статическая проверка канонического production notebook этапа 3."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "scripts/run_finetune_kaggle.ipynb"
CONFIG = REPOSITORY / "configs/finetune.yaml"
LOCK = REPOSITORY / "requirements/kaggle.lock"
EXPECTED_STAGES = 24
EXPECTED_CELLS = 47
TRAIN_QRELS_SHA256 = (
    "bf1f737cda0d66bc38fef5f9d91843f7a89428c5c5a8a3dce4764f527ec344ef"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    require(isinstance(source, list), "cell source must be a list")
    return "".join(str(line) for line in source)


def main() -> int:
    require(NOTEBOOK.is_file(), f"не найден notebook: {NOTEBOOK}")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    require(
        isinstance(cells, list) and len(cells) == EXPECTED_CELLS,
        f"ожидалось {EXPECTED_CELLS} ячеек",
    )

    sources = [cell_source(cell) for cell in cells]
    stage_numbers: list[int] = []
    for index, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type")
        require(cell_type in {"markdown", "code"}, f"ячейка {index}: неверный тип")
        if cell_type == "markdown":
            match = re.match(r"^## Этап (\d+)\.", sources[index - 1])
            require(match is not None, f"ячейка {index}: нет номера этапа")
            stage_numbers.append(int(match.group(1)))
            continue
        require(
            cell.get("execution_count") is None,
            f"ячейка {index}: сохранён execution_count",
        )
        outputs = cell.get("outputs", [])
        require(not outputs, f"ячейка {index}: сохранены outputs")
        ast.parse(sources[index - 1], filename=f"ячейка {index}")

    require(
        stage_numbers == list(range(1, EXPECTED_STAGES + 1)),
        "номера 24 этапов нарушены",
    )
    require(cells[0].get("cell_type") == "markdown", "этап 1 должен быть markdown")
    for stage in range(2, EXPECTED_STAGES + 1):
        markdown_index = 1 + (stage - 2) * 2
        require(
            cells[markdown_index].get("cell_type") == "markdown"
            and cells[markdown_index + 1].get("cell_type") == "code",
            f"этап {stage} должен состоять из markdown и code",
        )

    all_text = "\n".join(sources)
    code_text = "\n".join(
        source
        for cell, source in zip(cells, sources, strict=True)
        if cell.get("cell_type") == "code"
    )
    lower_text = all_text.lower()
    require("traceback (most recent call last)" not in lower_text, "найден traceback")
    for banned in (
        "/opt/bin",
        "/kaggle/working",
        "git pull",
        "openjdk",
        "apt-get",
        "filelink",
        "google.colab",
        "trust_remote_code",
        "pip install -u",
    ):
        require(banned not in lower_text, f"найден запрещённый фрагмент {banned!r}")
    require(re.search(r"/(?:Users|home)/", all_text) is None, "найден локальный путь пользователя")
    require(
        re.search(r"[A-Za-z]:\\\\", all_text) is None,
        "найден абсолютный путь Windows",
    )
    require(re.search(r"\bBRANCH\s*=", code_text) is None, "используется плавающая ветка")
    require(
        re.search(
            r"(?:api[_-]?key|access[_-]?token|hf_token|github_token)\s*=",
            code_text,
            flags=re.IGNORECASE,
        )
        is None,
        "в notebook похожая на секрет переменная",
    )

    require(
        re.search(r"RELEASE_REF\s*=\s*[\"']phase3-v\d+\.\d+\.\d+[\"']", code_text)
        is not None,
        "не указан неизменяемый тег выпуска",
    )
    require(code_text.count("def run_checked(") == 1, "run_checked определён не один раз")
    require(
        code_text.count('"select-checkpoint"') == 1,
        "select-checkpoint должен встречаться ровно один раз",
    )
    require(
        code_text.count('"package-phase3"') == 1,
        "package-phase3 должен встречаться ровно один раз",
    )
    for forbidden_command in (
        "run-bm25",
        "build-candidate-cache",
        "rerank-score",
        "evaluate-rerank",
        "package-phase1",
        "package-phase2",
    ):
        require(
            forbidden_command not in code_text,
            f"запрещён повторный запуск предыдущей фазы: {forbidden_command}",
        )

    ordered = (
        '"build-training-split"',
        '"build-training-pairs"',
        '"validate-checkpoint"',
        '"smoke-finetune"',
        '"C1"',
        '"A1"',
        '"A2"',
        '"B1"',
        '"select-checkpoint"',
        '"prepare-dev-evaluation"',
        '"score-finetuned"',
        '"evaluate-phase3"',
        '"package-phase3"',
    )
    positions = [code_text.index(fragment) for fragment in ordered]
    require(positions == sorted(positions), "порядок production-команд нарушен")

    stage_code: dict[int, str] = {}
    for stage in range(2, EXPECTED_STAGES + 1):
        stage_code[stage] = sources[2 + (stage - 2) * 2]
    preselection = "\n".join(stage_code[stage] for stage in range(2, 19))
    for forbidden in (
        "prepare-dev-evaluation",
        "score-finetuned",
        "evaluate-phase3",
        "qrels.miracl-v1.0-ru-dev",
        "prepared_dev_qrels",
    ):
        require(
            forbidden not in preselection,
            f"до выбора обнаружен доступ к dev: {forbidden}",
        )

    smoke_source = stage_code[14]
    for fragment in (
        '"status": "PASS"',
        '"real_model_forward": True',
        '"real_optimizer_step": True',
        '"fixture_only": False',
        '"device": "cuda"',
        '"checkpoint_save_load_roundtrip": True',
        '"resume_state_roundtrip": True',
        '"zip_hash_roundtrip": True',
        '"dtype"',
        '"float32"',
    ):
        require(fragment in smoke_source, f"smoke gate не проверяет {fragment}")

    preflight_source = stage_code[3]
    require("MIN_FREE_DISK_GIB = 25" in preflight_source, "не объявлен порог диска")
    require(
        'disk.free < MIN_FREE_DISK_GIB * 1024**3' in preflight_source,
        "порог диска не используется",
    )
    require("TREC_EVAL_PATH" in stage_code[5], "trec_eval не передаётся через окружение")
    require(TRAIN_QRELS_SHA256 in code_text, "нет точной SHA-256 train qrels")
    require("CUDA_VISIBLE_DEVICES" in stage_code[8], "нет отдельного CUDA-negative теста")
    require(
        "not test_unknown_run_and_non_cuda_production_are_blocked" in stage_code[8],
        "основной pytest не исключает environment-specific CUDA-тест",
    )
    for version in ("3.12.13", "2.13.0", "5.14.1", "0.22.2"):
        require(version in stage_code[6], f"окружение не фиксирует {version}")
    require('"--no-deps"' in stage_code[6], "установка окружения не ограничена")
    require('"--requirement"' in stage_code[6], "lock-файл не используется")
    require("wrapt" not in lower_text, "найдена дублирующая установка wrapt")

    mutation_patterns = (
        r"(?:src|configs)/[^\n]*(?:write_text|write_bytes)",
        r"(?:write_text|write_bytes)[^\n]*(?:src|configs)/",
        r"sed\s+-i[^\n]*(?:src|configs)",
        r"apply_patch",
    )
    for pattern in mutation_patterns:
        require(
            re.search(pattern, code_text, flags=re.IGNORECASE) is None,
            "notebook изменяет src/ или configs/",
        )

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require(config["implementation"]["version"] == "3.1.0", "неверная версия реализации")
    require(config["release"]["version"] == "1.0.1", "неверная версия выпуска")
    require(config["release"]["ref"] == "phase3-v1.0.1", "неверный тег выпуска")
    require(
        config["archive"]["results_zip"].endswith("_v1.0.1.zip"),
        "имя архива результатов не версионировано",
    )
    require(
        config["evaluation"]["reference_zero_shot_ndcg_at_10"] == 0.5365
        and config["evaluation"]["reference_bm25_ndcg_at_10"] == 0.3342,
        "изменены официальные опорные метрики",
    )
    rerank_config = yaml.safe_load(
        (REPOSITORY / "configs/rerank.yaml").read_text(encoding="utf-8")
    )
    retrieval_config = yaml.safe_load(
        (REPOSITORY / "configs/retrieval.yaml").read_text(encoding="utf-8")
    )
    require(
        rerank_config["evaluation"]["trec_eval_executable"] is None,
        "configs/rerank.yaml содержит жёсткий путь trec_eval",
    )
    require(
        retrieval_config["reproduction_gate"]["trec_eval_executable"] is None,
        "configs/retrieval.yaml содержит жёсткий путь trec_eval",
    )

    lock_lines = {
        line.strip()
        for line in LOCK.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for requirement in (
        "torch==2.13.0",
        "transformers==5.14.1",
        "tokenizers==0.22.2",
    ):
        require(requirement in lock_lines, f"lock-файл не содержит {requirement}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "notebook": NOTEBOOK.relative_to(REPOSITORY).as_posix(),
                "cells": len(cells),
                "stages": EXPECTED_STAGES,
                "release_ref": config["release"]["ref"],
                "selection_stage": 19,
                "first_dev_access_stage": 20,
                "package_stage": 23,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"Проверка production notebook не пройдена: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
