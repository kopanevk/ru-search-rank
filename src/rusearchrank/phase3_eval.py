"""Checkpoint selection, isolated final evaluation, diagnostics, and packaging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Прямой импорт модуля также обязан сохранять контракт процесса обучения.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import yaml

from .data import load_qrels
from .evaluation import (
    assert_candidate_set_invariant,
    evaluate_ranked_ndcg_at_10,
    mean_reciprocal_rank_at_10,
    paired_ranking_comparison,
    parse_trec_eval_metric,
    parse_trec_eval_per_query,
    rank_candidates_by_score,
    raw_score_tie_statistics,
    sparse_judgment_diagnostics,
    stratified_delta_summary,
)
from .pair_encoding import build_document, encode_pair
from .retrieval import read_trec_run
from . import rerank
from .training import (
    StageError,
    build_training_fingerprint,
    config_hashes,
    git_provenance,
    package_version,
    source_hashes,
    training_fingerprint_components,
    utc_now,
    validate_epoch_generation,
)
from .training_data import (
    atomic_write_json,
    canonical_json_sha256,
    manifest_declared_hashes,
    phase12_immutable_snapshot,
    portable_path,
    read_json,
    repository_root,
    resolve_path,
    sha256_file,
    validate_finetune_config,
    verify_phase12_immutable,
)


RESULT_ZIP_MEMBERS = (
    "LICENSE",
    "NOTICE",
    "requirements/kaggle.lock",
    "artifacts/training/query_split.parquet",
    "artifacts/training/pairs_judged_only.parquet",
    "artifacts/training/pairs_weak_negatives.parquet",
    "artifacts/training/pairs_control_c1.parquet",
    "artifacts/scores/dev_finetuned.parquet",
    "artifacts/runs/dev_rerank_finetuned_k100.trec",
    "reports/metrics/validation_checkpoint_metrics.json",
    "reports/metrics/validation_ab_comparison.json",
    "reports/metrics/dev_finetuned.json",
    "reports/metrics/dev_three_way_comparison.json",
    "reports/metrics/dev_score_tie_diagnostic.json",
    "reports/training/A1_history.json",
    "reports/training/A2_history.json",
    "reports/training/B1_history.json",
    "reports/audit/query_split_manifest.json",
    "reports/audit/pairs_manifest.json",
    "reports/audit/control_c1.json",
    "reports/audit/finetune_smoke.json",
    "reports/audit/resource_report.json",
    "reports/audit/environment_freeze.txt",
    "reports/audit/dev_access_ledger.jsonl",
    "reports/audit/checkpoint_selection.json",
    "reports/audit/model_card.md",
    "reports/audit/finetune_protocol.yaml",
    "reports/audit/training_manifest.json",
)
MODEL_ZIP_MEMBERS = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "model_card.md",
    "checkpoint_sha256.json",
    "LICENSE",
    "NOTICE",
)
BEST_FINETUNED_FILES = MODEL_ZIP_MEMBERS[:6]


def _metrics(config: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_path(config, config["metrics"]["validation_checkpoint_metrics"])
    if not path.is_file():
        raise StageError("select-checkpoint", f"validation metrics are missing: {path}")
    return read_json(path, training_path=False)


def _run_best(entries: Mapping[str, Any]) -> dict[str, Any]:
    values = [dict(value) for value in entries.values()]
    if not values:
        raise ValueError("run has no validation epochs")
    return max(values, key=lambda value: (float(value["ndcg_at_10"]), -int(value["epoch"])))


def _validate_validation_entry(
    value: Mapping[str, Any],
    *,
    label: str,
    expected_epoch: int,
    expected_query_ids: set[str] | None = None,
) -> set[str]:
    if int(value.get("epoch", -1)) != expected_epoch:
        raise StageError("select-checkpoint", f"{label} epoch metadata is invalid")
    metric = value.get("ndcg_at_10")
    if (
        not isinstance(metric, (int, float))
        or isinstance(metric, bool)
        or not math.isfinite(float(metric))
        or not 0.0 <= float(metric) <= 1.0
    ):
        raise StageError("select-checkpoint", f"{label} nDCG@10 is invalid")
    checkpoint_hash = value.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash) is None
    ):
        raise StageError("select-checkpoint", f"{label} checkpoint hash is invalid")
    if expected_epoch > 0:
        generation_hash = value.get("model_generation_sha256")
        if (
            not isinstance(generation_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", generation_hash) is None
        ):
            raise StageError(
                "select-checkpoint", f"{label} generation hash is invalid"
            )
    per_query = value.get("per_query")
    if not isinstance(per_query, Mapping) or not per_query:
        raise StageError("select-checkpoint", f"{label} per-query vector is missing")
    query_ids = {str(query_id) for query_id in per_query}
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        for item in per_query.values()
    ):
        raise StageError("select-checkpoint", f"{label} per-query vector is invalid")
    if expected_query_ids is not None and query_ids != expected_query_ids:
        raise StageError(
            "select-checkpoint", f"{label} validation query universe changed"
        )
    return query_ids


def _current_run_training_contract(
    config: Mapping[str, Any], run_id: str
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest_path = (
        resolve_path(config, config["artifacts"]["models_dir"])
        / run_id
        / "run_manifest.json"
    )
    manifest = read_json(manifest_path, training_path=False)
    if manifest.get("finalized") is not True or manifest.get(
        "resume_available"
    ) is not False:
        raise StageError("checkpoint-contract", f"{run_id} is not finalized")
    try:
        learning_rate = float(manifest["learning_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StageError(
            "checkpoint-contract", f"{run_id} learning rate is invalid"
        ) from exc
    components = training_fingerprint_components(
        config, run_id=run_id, learning_rate=learning_rate
    )
    fingerprint = build_training_fingerprint(components)
    if (
        manifest.get("training_fingerprint") != fingerprint
        or manifest.get("training_fingerprint_components") != components
    ):
        raise StageError(
            "checkpoint-contract",
            f"{run_id} was trained with stale source, config, or inputs",
        )
    return manifest, components, fingerprint


def _validate_selected_training_contract(
    config: Mapping[str, Any], selection: Mapping[str, Any]
) -> str:
    _validate_checkpoint_selection_integrity(config, selection)
    best = selection.get("best_finetuned_checkpoint")
    if not isinstance(best, Mapping):
        raise StageError("checkpoint-contract", "best fine-tuned selection is invalid")
    run_id = str(best.get("run_id"))
    _, _, fingerprint = _current_run_training_contract(config, run_id)
    if best.get("training_fingerprint") != fingerprint:
        raise StageError(
            "checkpoint-contract", "selected checkpoint training fingerprint is stale"
        )
    return fingerprint


def _selection_payload_sha256(selection: Mapping[str, Any]) -> str:
    body = {
        str(name): value
        for name, value in selection.items()
        if name != "selection_sha256"
    }
    return canonical_json_sha256(body)


def _validate_selection_payload_integrity(
    selection: Mapping[str, Any],
) -> str:
    if int(selection.get("schema_version", -1)) != 2:
        raise StageError(
            "checkpoint-contract",
            "checkpoint selection must use schema version 2",
        )
    recorded = selection.get("selection_sha256")
    computed = _selection_payload_sha256(selection)
    if (
        not isinstance(recorded, str)
        or re.fullmatch(r"[0-9a-f]{64}", recorded) is None
        or recorded != computed
    ):
        raise StageError(
            "checkpoint-contract", "checkpoint selection integrity hash is invalid"
        )
    if selection.get("selection_written_before_dev_access") is not True:
        raise StageError(
            "checkpoint-contract", "selection-before-dev declaration is missing"
        )
    selected_at = selection.get("selected_at")
    try:
        parsed = datetime.fromisoformat(str(selected_at))
    except ValueError as exc:
        raise StageError(
            "checkpoint-contract", "checkpoint selection timestamp is invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise StageError(
            "checkpoint-contract", "checkpoint selection timestamp has no timezone"
        )
    best = selection.get("best_finetuned_checkpoint")
    if not isinstance(best, Mapping) or re.fullmatch(
        r"[0-9a-f]{64}", str(best.get("sha256", ""))
    ) is None:
        raise StageError(
            "checkpoint-contract", "selected checkpoint SHA-256 is invalid"
        )
    return recorded


def _validate_checkpoint_selection_integrity(
    config: Mapping[str, Any], selection: Mapping[str, Any]
) -> str:
    recorded = _validate_selection_payload_integrity(selection)
    metrics_path = resolve_path(
        config, config["metrics"]["validation_checkpoint_metrics"]
    )
    if not metrics_path.is_file() or selection.get(
        "validation_metrics_sha256"
    ) != sha256_file(metrics_path):
        raise StageError(
            "checkpoint-contract",
            "validation metrics changed after checkpoint selection",
        )
    candidate_material = {
        "candidates": selection.get("candidates"),
        "best_epoch_by_run": selection.get("best_epoch_by_run"),
    }
    if selection.get("candidate_set_sha256") != canonical_json_sha256(
        candidate_material
    ):
        raise StageError(
            "checkpoint-contract", "checkpoint candidate set integrity hash is invalid"
        )
    return recorded


def _checkpoint_payload_hash(files: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(files[relative]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_best_finetuned_payload(
    destination: Path, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    files = checkpoint.get("files")
    if not isinstance(files, Mapping) or set(files) != set(BEST_FINETUNED_FILES):
        raise ValueError("best_finetuned file allowlist is invalid")
    actual_files = {
        path.name
        for path in destination.iterdir()
        if path.name != "checkpoint_sha256.json"
    }
    if actual_files != set(BEST_FINETUNED_FILES):
        raise ValueError("best_finetuned directory contains missing or extra files")
    normalized: dict[str, str] = {}
    for name in BEST_FINETUNED_FILES:
        expected = files.get(name)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"best_finetuned has an invalid file hash: {name}")
        actual = sha256_file(destination / name)
        if actual != expected:
            raise ValueError(f"best_finetuned payload is corrupted: {name}")
        normalized[name] = actual
    payload_hash = _checkpoint_payload_hash(normalized)
    if checkpoint.get("checkpoint_payload_sha256") != payload_hash:
        raise ValueError("best_finetuned aggregate hash is invalid")
    return dict(checkpoint)


def _publish_best_finetuned(
    config: Mapping[str, Any],
    *,
    run_id: str,
    epoch: int,
    expected_generation_sha256: str,
    expected_weight_sha256: str,
    expected_training_fingerprint: str,
    overwrite: bool,
) -> dict[str, Any]:
    source = resolve_path(config, config["artifacts"]["models_dir"]) / run_id / f"epoch_{epoch}"
    sidecar = validate_epoch_generation(
        source, expected_fingerprint=expected_training_fingerprint
    )
    if sidecar.get("model_generation_sha256") != expected_generation_sha256:
        raise StageError(
            "select-checkpoint/publication",
            "validation metrics refer to another epoch generation",
            expected_sha256=expected_generation_sha256,
            actual_sha256=sidecar.get("model_generation_sha256"),
        )
    if sidecar.get("files", {}).get("model.safetensors") != expected_weight_sha256:
        raise StageError(
            "select-checkpoint/publication",
            "validation metrics refer to another model weight file",
            expected_sha256=expected_weight_sha256,
            actual_sha256=sidecar.get("files", {}).get("model.safetensors"),
        )
    destination = resolve_path(config, config["artifacts"]["best_finetuned_dir"])
    required = BEST_FINETUNED_FILES
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise StageError("select-checkpoint/publication", f"generation is missing files: {missing}")
    if destination.is_dir() and not overwrite:
        checkpoint = read_json(destination / "checkpoint_sha256.json", training_path=False)
        if (
            checkpoint.get("source_run_id") == run_id
            and int(checkpoint.get("source_epoch", -1)) == epoch
            and checkpoint.get("source_model_generation_sha256")
            == sidecar["model_generation_sha256"]
        ):
            return _validate_best_finetuned_payload(destination, checkpoint)
        raise ValueError("best_finetuned already exists for another selection; use --overwrite")
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise StageError("select-checkpoint/publication", f"temporary directory exists: {temporary}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for name in required:
            shutil.copyfile(source / name, temporary / name)
        files = {name: sha256_file(temporary / name) for name in required}
        payload = {
            "source_run_id": run_id,
            "source_epoch": int(epoch),
            "source_path": portable_path(config, source),
            "source_model_generation_sha256": sidecar["model_generation_sha256"],
            "files": files,
            "checkpoint_payload_sha256": _checkpoint_payload_hash(files),
            "published_at": utc_now(),
        }
        atomic_write_json(temporary / "checkpoint_sha256.json", payload)
        # До публикации убеждаемся, что копии модели и токенизатора открываются локально.
        rerank.TransformersPairScorer(
            _phase3_rerank_config(config), device="cpu", checkpoint=temporary
        )
        backup: Path | None = None
        if destination.exists():
            backup = destination.with_name(
                f"{destination.name}.stale.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
            destination.replace(backup)
        try:
            temporary.replace(destination)
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
        return payload
    except Exception as exc:
        raise StageError(
            "select-checkpoint/publication",
            str(exc),
            preserved_temporary_path=str(temporary),
        ) from exc


def _validation_ab_report(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    best_by_run: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    judged_run = max(
        ("A1", "A2"),
        key=lambda run_id: (
            float(best_by_run[run_id]["ndcg_at_10"]),
            -float(config["runs"][run_id]["learning_rate"]),
        ),
    )
    judged = best_by_run[judged_run]
    weak = best_by_run["B1"]
    comparison = paired_ranking_comparison(
        judged["per_query"],
        weak["per_query"],
        resamples=int(config["selection"]["ab_bootstrap_resamples"]),
        seed=int(config["selection"]["ab_bootstrap_seed"]),
        confidence=0.95,
        tie_tolerance=1e-12,
    )
    pairs_manifest = read_json(
        resolve_path(config, config["audits"]["pairs_manifest"]), training_path=False
    )
    judged_count = int(
        pairs_manifest["regimes"]["judged_only"]["usable_query_count"]
    )
    weak_count = int(
        pairs_manifest["regimes"]["weak_negatives"]["usable_query_count"]
    )
    report = {
        "analysis_role": "exploratory_post_selection",
        "confirmatory_inference": False,
        "reuse_disclosure": (
            "Одна контрольная выборка использована для выбора скорости обучения, "
            "эпохи, лучшего запуска judged_only и контрольной точки после fine-tuning. "
            "Это описательное сравнение после выбора не подтверждает причинный эффект."
        ),
        "judged_only_run": judged_run,
        "judged_only_epoch": int(judged["epoch"]),
        "weak_regime_run": "B1",
        "weak_regime_epoch": int(weak["epoch"]),
        "judged_only_usable_query_count": judged_count,
        "weak_negatives_usable_query_count": weak_count,
        "comparison": comparison,
        "sparse_diagnostics": {
            "judged_only": judged.get("sparse_diagnostics"),
            "weak_negatives": weak.get("sparse_diagnostics"),
        },
        "population_disclosure": (
            "Режимы охватывают разные множества запросов: B1 включает положительные "
            "запросы без экспертно оценённых отрицательных документов. Поэтому разница "
            "смешивает увеличение числа отрицательных примеров и охвата запросов."
        ),
    }
    path = resolve_path(config, config["metrics"]["validation_ab_comparison"])
    atomic_write_json(path, report)
    return report


def _write_model_card(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    final_evaluation: Mapping[str, Any] | None = None,
) -> None:
    pairs_manifest = read_json(
        resolve_path(config, config["audits"]["pairs_manifest"]), training_path=False
    )
    judged = pairs_manifest["regimes"]["judged_only"]
    weak = pairs_manifest["regimes"]["weak_negatives"]
    production = selection["production_system"]
    outcome = (
        "Итоговое оценивание ещё не выполнено."
        if final_evaluation is None
        else (
            "Основной итог ML-эксперимента: "
            f"`{final_evaluation['ml_outcome']['label']}`."
        )
    )
    production_label = {
        "zero_shot": "исходная zero-shot модель",
        "finetuned": "модель после fine-tuning",
    }.get(str(production["kind"]), f"`{production['kind']}`")
    text = f"""# Модельная карточка RuSearchRank, этап 3

Cross-encoder — модель, которая совместно обрабатывает запрос и документ и
оценивает их релевантность. Основа: `{config['base_model']['id']}` в неизменяемой
ревизии `{config['base_model']['revision']}`. На вход подаётся `query`, затем
документ из `title`, перевода строки и `text`; документ обрезается по правилу `only_second` до 320
токенов.

Выбранная контрольная точка после fine-tuning:
`{selection['best_finetuned_checkpoint']['run_id']}`, эпоха
{selection['best_finetuned_checkpoint']['epoch']}. Отдельно выбранная рабочая
система — {production_label}. Эти решения не объединяются: архив модели всегда
содержит выбранную контрольную точку после fine-tuning. {outcome}

## Область обучающих данных и ограничения

Использовались только положительные документы из top-100 BM25. Документы без
экспертной оценки не переобозначались как нерелевантные. Режимы только с
экспертными оценками и с weak negatives — документами без оценки, используемыми
как слабые отрицательные примеры, — охватывают разные множества запросов:
{judged['usable_query_count']} и {weak['usable_query_count']} соответственно.
Поэтому описательная разница между режимами одновременно отражает доступность
отрицательных примеров и охват запросов.

Вес слабой пары 0,5 уменьшает влияние отдельной пары, но не фиксирует общую долю
weak negatives для запроса. Показательные доли: 33% для 8 экспертно оценённых и
8 слабых пар, 67% для 2 экспертно оценённых и 8 слабых пар и 100%, если
экспертно оценённых пар нет. Полное распределение по запросам записано в
`pairs_manifest.json`.

Диапазон рангов 26–100, не более 8 слабых документов, вес 0,5 и ограничения
8/8/16 — заранее заданные консервативные эвристики, зафиксированные до итогового
оценивания. Они не считаются найденными оптимумами. Сравнение режимов на
контрольной выборке является исследовательским анализом после выбора: одна и та
же выборка использована для выбора скорости обучения, эпохи, запуска и
контрольной точки. Поэтому сравнение не подтверждает причинный эффект.

## Воспроизводимость

Модель обучается на CUDA в `fp32` с детерминированными алгоритмами. Один запрос
образует один микропакет; параметры обновляются по точному среднему в окне не
более чем из 16 функций потерь по запросам. Последнее неполное окно нормируется
по фактическому размеру. Так градиенты запросов получают одинаковый вес внутри
одного шага оптимизатора, но не обязательно одинаковый вклад во все обновления
Adam за эпоху.
"""
    path = resolve_path(config, config["audits"]["model_card"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def select_checkpoint(
    config: Mapping[str, Any], *, overwrite: bool = False
) -> dict[str, Any]:
    validate_finetune_config(config)
    selection_path = resolve_path(config, config["audits"]["checkpoint_selection"])
    if selection_path.is_file() and not overwrite:
        existing = read_json(selection_path, training_path=False)
        _validate_selected_training_contract(config, existing)
        for required_path in (
            resolve_path(config, config["metrics"]["validation_ab_comparison"]),
            resolve_path(config, config["audits"]["model_card"]),
        ):
            if not required_path.is_file():
                raise ValueError(
                    "checkpoint selection exists but its post-selection report is missing"
                )
        best_dir = resolve_path(config, config["artifacts"]["best_finetuned_dir"])
        if not (best_dir / "checkpoint_sha256.json").is_file():
            raise ValueError("checkpoint selection exists but best_finetuned is missing")
        best = existing["best_finetuned_checkpoint"]
        published = _publish_best_finetuned(
            config,
            run_id=str(best["run_id"]),
            epoch=int(best["epoch"]),
            expected_generation_sha256=str(best["model_generation_sha256"]),
            expected_weight_sha256=str(best["sha256"]),
            expected_training_fingerprint=str(best["training_fingerprint"]),
            overwrite=False,
        )
        if (
            published["checkpoint_payload_sha256"]
            != best["checkpoint_payload_sha256"]
            or published["files"]["model.safetensors"] != best["sha256"]
        ):
            raise ValueError("checkpoint selection differs from best_finetuned payload")
        return {"status": "PASS", "action": "reused", **existing}
    ledger_path = resolve_path(config, config["audits"]["dev_access_ledger"])
    if ledger_path.is_file() and ledger_path.stat().st_size > 0:
        raise StageError(
            "select-checkpoint",
            "selection cannot be created or overwritten after evaluation access",
            ledger_path=portable_path(config, ledger_path),
        )
    metrics = _metrics(config)
    base = metrics.get("S0")
    if not isinstance(base, Mapping):
        raise StageError("select-checkpoint", "S0 validation result is missing")
    validation_query_ids = _validate_validation_entry(
        base, label="S0", expected_epoch=0
    )
    best_by_run: dict[str, dict[str, Any]] = {}
    run_fingerprints: dict[str, str] = {}
    candidates: list[dict[str, Any]] = [
        {
            "kind": "zero_shot",
            "candidate_id": "S0",
            "validation_ndcg_at_10": float(base["ndcg_at_10"]),
            "sha256": str(base["checkpoint_sha256"]),
        }
    ]
    for run_id in ("A1", "A2", "B1"):
        manifest, _, current_fingerprint = (
            _current_run_training_contract(config, run_id)
        )
        run_fingerprints[run_id] = current_fingerprint
        entries = metrics.get("runs", {}).get(run_id)
        expected_epoch_keys = {"epoch_1", "epoch_2", "epoch_3"}
        if not isinstance(entries, Mapping) or set(entries) != expected_epoch_keys:
            raise StageError("select-checkpoint", f"{run_id} must have exactly three epochs")
        for epoch in (1, 2, 3):
            value = entries[f"epoch_{epoch}"]
            if not isinstance(value, Mapping):
                raise StageError(
                    "select-checkpoint", f"{run_id}/epoch_{epoch} is invalid"
                )
            _validate_validation_entry(
                value,
                label=f"{run_id}/epoch_{epoch}",
                expected_epoch=epoch,
                expected_query_ids=validation_query_ids,
            )
        best_by_run[run_id] = _run_best(entries)
        for epoch_name, value in sorted(entries.items(), key=lambda item: int(item[1]["epoch"])):
            candidates.append(
                {
                    "kind": "finetuned",
                    "candidate_id": f"{run_id}/{epoch_name}",
                    "run_id": run_id,
                    "epoch": int(value["epoch"]),
                    "validation_ndcg_at_10": float(value["ndcg_at_10"]),
                    "sha256": str(value["checkpoint_sha256"]),
                    "model_generation_sha256": str(
                        value["model_generation_sha256"]
                    ),
                }
            )
    tie_order = list(config["selection"]["run_tie_break_order"])
    best_run_id = max(
        tie_order,
        key=lambda run_id: (
            float(best_by_run[run_id]["ndcg_at_10"]),
            -tie_order.index(run_id),
        ),
    )
    best = best_by_run[best_run_id]
    published = _publish_best_finetuned(
        config,
        run_id=best_run_id,
        epoch=int(best["epoch"]),
        expected_generation_sha256=str(best["model_generation_sha256"]),
        expected_weight_sha256=str(best["checkpoint_sha256"]),
        expected_training_fingerprint=run_fingerprints[best_run_id],
        overwrite=overwrite,
    )
    best_payload = {
        "run_id": best_run_id,
        "epoch": int(best["epoch"]),
        "path": str(config["artifacts"]["best_finetuned_dir"]),
        "sha256": published["files"]["model.safetensors"],
        "checkpoint_payload_sha256": published["checkpoint_payload_sha256"],
        "model_generation_sha256": published["source_model_generation_sha256"],
        "training_fingerprint": run_fingerprints[best_run_id],
        "validation_ndcg_at_10": float(best["ndcg_at_10"]),
    }
    if float(base["ndcg_at_10"]) >= float(best["ndcg_at_10"]):
        production = {
            "kind": "zero_shot",
            "candidate_id": "S0",
            "validation_ndcg_at_10": float(base["ndcg_at_10"]),
            "sha256": str(base["checkpoint_sha256"]),
        }
    else:
        production = {"kind": "finetuned", **best_payload}
    payload = {
        "schema_version": 2,
        "selection_written_before_dev_access": True,
        "candidates": candidates,
        "best_epoch_by_run": {
            run_id: {
                "epoch": int(value["epoch"]),
                "validation_ndcg_at_10": float(value["ndcg_at_10"]),
                "sha256": str(value["checkpoint_sha256"]),
                "model_generation_sha256": str(value["model_generation_sha256"]),
            }
            for run_id, value in best_by_run.items()
        },
        "best_finetuned_checkpoint": best_payload,
        "production_system": production,
        "zero_shot_won": production["kind"] == "zero_shot",
        "selected_at": utc_now(),
        "validation_metrics_sha256": sha256_file(
            resolve_path(
                config, config["metrics"]["validation_checkpoint_metrics"]
            )
        ),
    }
    payload["candidate_set_sha256"] = canonical_json_sha256(
        {
            "candidates": payload["candidates"],
            "best_epoch_by_run": payload["best_epoch_by_run"],
        }
    )
    payload["selection_sha256"] = _selection_payload_sha256(payload)
    _validation_ab_report(config, metrics, best_by_run=best_by_run)
    _write_model_card(config, payload)
    atomic_write_json(selection_path, payload)
    return {"status": "PASS", "action": "selected", **payload}


def append_dev_access_ledger(
    config: Mapping[str, Any],
    *,
    command: str,
    checkpoint_sha256: str,
    input_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    """Append a hash-chained audit event before any isolated data read."""

    path = resolve_path(config, config["audits"]["dev_access_ledger"])
    selection_path = resolve_path(
        config, config["audits"]["checkpoint_selection"]
    )
    if not selection_path.is_file():
        raise StageError(
            "dev-access-ledger", "checkpoint selection must exist before dev access"
        )
    selection = read_json(selection_path, training_path=False)
    selection_sha256 = _validate_selection_payload_integrity(selection)
    selected_at = datetime.fromisoformat(str(selection["selected_at"]))
    expected_checkpoint = str(selection["best_finetuned_checkpoint"]["sha256"])
    if checkpoint_sha256 != expected_checkpoint:
        raise StageError(
            "dev-access-ledger",
            "dev access attempted with a checkpoint different from the selection",
            expected_sha256=expected_checkpoint,
            actual_sha256=checkpoint_sha256,
        )
    normalized_inputs = {
        str(name): str(value) for name, value in sorted(input_hashes.items())
    }
    invalid_inputs = sorted(
        name
        for name, digest in normalized_inputs.items()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None
    )
    if not normalized_inputs or invalid_inputs:
        raise StageError(
            "dev-access-ledger",
            "dev input declarations must contain SHA-256 values",
            invalid_inputs=invalid_inputs,
        )
    input_set_sha256 = canonical_json_sha256(normalized_inputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        existing_bytes = b""
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            existing_bytes += chunk
        existing: list[dict[str, Any]] = []
        previous_hash = "0" * 64
        previous_timestamp: datetime | None = None
        for line in existing_bytes.decode("utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("ledger line is not a JSON object")
                sequence = len(existing) + 1
                if int(value.get("schema_version", -1)) != 2:
                    raise ValueError("dev-access ledger schema version is invalid")
                if int(value.get("sequence", -1)) != sequence:
                    raise ValueError("dev-access ledger sequence is invalid")
                if value.get("previous_event_sha256") != previous_hash:
                    raise ValueError("dev-access ledger hash chain is broken")
                body = {
                    str(name): field
                    for name, field in value.items()
                    if name != "event_sha256"
                }
                computed_hash = canonical_json_sha256(body)
                if value.get("event_sha256") != computed_hash:
                    raise ValueError("dev-access ledger event hash is invalid")
                if value.get("selection_sha256") != selection_sha256:
                    raise StageError(
                        "dev-access-ledger",
                        "checkpoint selection changed after the first dev access",
                    )
                if value.get("checkpoint_sha256") != expected_checkpoint:
                    raise StageError(
                        "dev-access-ledger",
                        "checkpoint changed after the first dev access",
                    )
                if value.get("input_set_sha256") != input_set_sha256 or value.get(
                    "input_hashes"
                ) != normalized_inputs:
                    raise StageError(
                        "dev-access-ledger",
                        "dev input declarations changed after the first access",
                    )
                try:
                    timestamp = datetime.fromisoformat(str(value.get("timestamp")))
                except ValueError as exc:
                    raise ValueError(
                        "dev-access ledger timestamp is invalid"
                    ) from exc
                if timestamp.tzinfo is None or timestamp < selected_at or (
                    previous_timestamp is not None
                    and timestamp < previous_timestamp
                ):
                    raise ValueError("dev-access ledger timestamps are invalid")
                if value.get("repeat_access") != (sequence > 1):
                    raise ValueError("dev-access ledger repeat marker is invalid")
                existing.append(value)
                previous_hash = computed_hash
                previous_timestamp = timestamp
        repeat = bool(existing)
        event = {
            "schema_version": 2,
            "sequence": len(existing) + 1,
            "previous_event_sha256": previous_hash,
            "timestamp": utc_now(),
            "command": command,
            "checkpoint_sha256": checkpoint_sha256,
            "selection_sha256": selection_sha256,
            "input_hashes": normalized_inputs,
            "input_set_sha256": input_set_sha256,
            "repeat_access": repeat,
        }
        event["event_sha256"] = canonical_json_sha256(event)
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("dev-access ledger append made no progress")
            written += count
        os.fsync(descriptor)
        return event
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _declared_dev_hashes(config: Mapping[str, Any]) -> dict[str, Any]:
    phase1 = resolve_path(config, config["inputs"]["phase1_manifest"])
    phase2 = resolve_path(config, config["inputs"]["phase2_manifest"])
    declared: dict[str, str] = {}
    for path in (phase1, phase2):
        if path.is_file():
            for identifier, digest in manifest_declared_hashes(path).items():
                previous = declared.get(identifier)
                if previous is not None and previous != digest:
                    raise ValueError(
                        "Phase 1/2 manifests disagree on evaluation SHA-256 for "
                        f"{identifier}"
                    )
                declared[identifier] = digest
    return {
        relative: declared.get(
            str(value), declared.get(relative, "not_declared_in_source_manifest")
        )
        for relative, value in config["dev_inputs"].items()
    }


def _prepared_qrels_path(config: Mapping[str, Any]) -> Path:
    return resolve_path(config, config["paths"]["work_dir"]) / "prepared_dev_qrels.trec"


def _prepared_qrels_parquet_path(config: Mapping[str, Any]) -> Path:
    return resolve_path(config, config["paths"]["work_dir"]) / "prepared_dev_qrels.parquet"


def _materialize_guarded_dev_qrels_source(
    config: Mapping[str, Any], source: Path
) -> None:
    """Download the pinned source only after the ledger event when not restored."""

    if source.is_file():
        return
    retrieval_config = repository_root(config) / "configs/retrieval.yaml"
    payload = yaml.safe_load(retrieval_config.read_text(encoding="utf-8"))
    url = str(payload["dataset"]["qrels"]["dev"])
    request = Request(url, headers={"User-Agent": "RuSearchRank/3.0"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    if not data:
        raise StageError("prepare-dev-evaluation", "downloaded qrels payload is empty")
    source.parent.mkdir(parents=True, exist_ok=True)
    temporary = source.with_name(f"{source.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(source)


def prepare_dev_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_finetune_config(config)
    selection_path = resolve_path(config, config["audits"]["checkpoint_selection"])
    if not selection_path.is_file():
        raise StageError(
            "prepare-dev-evaluation",
            "checkpoint_selection.json must exist before evaluation access",
        )
    selection = read_json(selection_path, training_path=False)
    _validate_selected_training_contract(config, selection)
    selected_checkpoint = selection["best_finetuned_checkpoint"]
    checkpoint = resolve_path(config, config["artifacts"]["best_finetuned_dir"])
    checkpoint_metadata = read_json(
        checkpoint / "checkpoint_sha256.json", training_path=False
    )
    _validate_best_finetuned_payload(checkpoint, checkpoint_metadata)
    if (
        checkpoint_metadata["files"]["model.safetensors"]
        != selected_checkpoint["sha256"]
        or checkpoint_metadata["checkpoint_payload_sha256"]
        != selected_checkpoint["checkpoint_payload_sha256"]
        or checkpoint_metadata["source_model_generation_sha256"]
        != selected_checkpoint["model_generation_sha256"]
    ):
        raise StageError(
            "prepare-dev-evaluation",
            "best_finetuned differs from checkpoint selection",
        )
    checkpoint_hash = str(selected_checkpoint["sha256"])
    declared_hashes = _declared_dev_hashes(config)
    invalid_bindings = sorted(
        name
        for name, digest in declared_hashes.items()
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    )
    if invalid_bindings:
        raise StageError(
            "prepare-dev-evaluation",
            "Phase 1/2 manifests do not bind every evaluation input",
            unbound_inputs=invalid_bindings,
        )
    event = append_dev_access_ledger(
        config,
        command="prepare-dev-evaluation",
        checkpoint_sha256=checkpoint_hash,
        input_hashes=declared_hashes,
    )
    # Только эта функция открывает исходные qrels для итогового оценивания.
    source = resolve_path(config, config["dev_inputs"]["dev_qrels"])
    _materialize_guarded_dev_qrels_source(config, source)
    declared = declared_hashes["dev_qrels"]
    actual_source_hash = sha256_file(source)
    if actual_source_hash != declared:
        raise StageError(
            "prepare-dev-evaluation",
            "evaluation qrels differ from the Phase 1/2 declared SHA-256",
            expected_sha256=declared,
            actual_sha256=actual_source_hash,
        )
    immutable = phase12_immutable_snapshot(config, require_all=True)
    qrels = load_qrels(source, split="dev")
    destination = _prepared_qrels_path(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    grade_column = "relevance_grade" if "relevance_grade" in qrels.columns else "relevance"
    ordered = qrels.sort_values(["query_id", "docid"], kind="mergesort")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in ordered.itertuples(index=False):
            stream.write(
                f"{row.query_id} 0 {row.docid} {int(getattr(row, grade_column))}\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    parquet_destination = _prepared_qrels_parquet_path(config)
    parquet_temporary = parquet_destination.with_name(
        f"{parquet_destination.name}.tmp.{os.getpid()}"
    )
    normalized = ordered[["query_id", "docid", grade_column]].copy()
    normalized.columns = ["query_id", "docid", "relevance_grade"]
    normalized[["query_id", "docid"]] = normalized[["query_id", "docid"]].astype(
        "string"
    )
    normalized["relevance_grade"] = normalized["relevance_grade"].astype("int64")
    normalized.to_parquet(parquet_temporary, index=False)
    parquet_temporary.replace(parquet_destination)
    sidecar = {
        "status": "PASS",
        "source_sha256": actual_source_hash,
        "prepared_sha256": sha256_file(destination),
        "prepared_parquet_sha256": sha256_file(parquet_destination),
        "row_count": int(len(qrels)),
        "query_count": int(qrels["query_id"].nunique()),
        "checkpoint_sha256": checkpoint_hash,
        "ledger_event": event,
    }
    atomic_write_json(destination.with_suffix(".json"), sidecar)
    verify_phase12_immutable(config, immutable)
    return sidecar


def _phase3_rerank_config(config: Mapping[str, Any]) -> dict[str, Any]:
    phase2_path = repository_root(config) / "configs/rerank.yaml"
    phase2 = rerank.load_rerank_config(phase2_path)
    adapted = copy.deepcopy(phase2)
    adapted["_config_path"] = str(phase2_path)
    adapted["implementation"]["version"] = str(config["implementation"]["version"])
    adapted["inference"].update(
        {
            "batch_size": int(config["inference"]["batch_size"]),
            "cpu_batch_size": int(config["inference"]["cpu_batch_size"]),
            "device": str(config["inference"]["device"]),
            "cuda_dtype": str(config["inference"]["cuda_dtype"]),
            "fallback_dtype": "float32",
            "score_dtype": str(config["inference"]["score_dtype"]),
            "shard_queries": int(config["inference"]["shard_queries"]),
            "seed": int(config["training"]["seed"]),
        }
    )
    adapted["inputs"].update(
        {
            "candidates": str(config["dev_inputs"]["dev_candidates"]),
            "queries": str(config["inputs"]["queries"]),
            "passages": str(config["inputs"]["passages"]),
            "bm25_run": str(config["dev_inputs"]["bm25_run"]),
            "qrels": portable_path(config, _prepared_qrels_path(config)),
            "phase1_manifest": str(config["inputs"]["phase1_manifest"]),
        }
    )
    work = str(config["paths"]["work_dir"])
    adapted["artifacts"].update(
        {
            "scores": str(config["artifacts"]["finetuned_scores"]),
            "partial_dir": f"{work}/score_partial",
            "rerank_run": str(config["artifacts"]["finetuned_run"]),
            "diagnostic_run_template": f"{work}/finetuned_k{{depth}}.trec",
        }
    )
    adapted["paths"]["work_dir"] = work
    adapted["archive"]["path"] = str(config["archive"]["results_zip"])
    rerank.validate_rerank_config(adapted)
    return adapted


def _checkpoint_scoring_hash(directory: Path) -> str:
    files = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in directory.rglob("*")
        if path.is_file()
    }
    if not files:
        raise ValueError("fine-tuned checkpoint directory is empty")
    return _checkpoint_payload_hash(files)


def _validate_finetuned_score_binding(
    config: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind persisted dev scores to the exact selected checkpoint directory."""

    checkpoint = resolve_path(config, config["artifacts"]["best_finetuned_dir"])
    metadata = read_json(checkpoint / "checkpoint_sha256.json", training_path=False)
    _validate_best_finetuned_payload(checkpoint, metadata)
    expected = selection["best_finetuned_checkpoint"]
    if (
        metadata["files"]["model.safetensors"] != expected["sha256"]
        or metadata["checkpoint_payload_sha256"]
        != expected["checkpoint_payload_sha256"]
        or metadata["source_model_generation_sha256"]
        != expected["model_generation_sha256"]
    ):
        raise StageError("fine-tuned-score-binding", "checkpoint differs from selection")
    score_path = resolve_path(config, config["artifacts"]["finetuned_scores"])
    sidecar_path = score_path.with_name(f"{score_path.name}.json")
    if not score_path.is_file() or not sidecar_path.is_file():
        raise StageError("fine-tuned-score-binding", "score payload or sidecar is missing")
    sidecar = read_json(sidecar_path, training_path=False)
    components = sidecar.get("fingerprint_components")
    if (
        sidecar.get("status") != "PASS"
        or sidecar.get("scores_sha256") != sha256_file(score_path)
        or not isinstance(components, Mapping)
        or components.get("checkpoint_sha256")
        != _checkpoint_scoring_hash(checkpoint)
    ):
        raise StageError(
            "fine-tuned-score-binding",
            "score sidecar is stale, corrupted, or refers to another checkpoint",
        )
    try:
        computed_fingerprint = rerank.build_input_fingerprint(components)
    except ValueError as exc:
        raise StageError("fine-tuned-score-binding", str(exc)) from exc
    if sidecar.get("input_fingerprint") != computed_fingerprint:
        raise StageError(
            "fine-tuned-score-binding", "score input fingerprint is inconsistent"
        )
    try:
        rerank.validate_current_score_sidecar(
            _phase3_rerank_config(config), score_path=score_path
        )
    except (ValueError, OSError) as exc:
        raise StageError(
            "fine-tuned-score-binding",
            f"score payload violates the current scoring contract: {exc}",
        ) from exc
    return sidecar


def score_finetuned(
    config: Mapping[str, Any], *, overwrite: bool = False
) -> dict[str, Any]:
    validate_finetune_config(config)
    selection_path = resolve_path(config, config["audits"]["checkpoint_selection"])
    if not selection_path.is_file():
        raise StageError("score-finetuned", "checkpoint selection must exist first")
    selection = read_json(selection_path, training_path=False)
    _validate_selected_training_contract(config, selection)
    checkpoint = resolve_path(config, config["artifacts"]["best_finetuned_dir"])
    checkpoint_metadata = read_json(
        checkpoint / "checkpoint_sha256.json", training_path=False
    )
    _validate_best_finetuned_payload(checkpoint, checkpoint_metadata)
    expected = selection["best_finetuned_checkpoint"]
    if (
        checkpoint_metadata["checkpoint_payload_sha256"]
        != expected["checkpoint_payload_sha256"]
        or checkpoint_metadata["files"]["model.safetensors"]
        != expected["sha256"]
        or checkpoint_metadata["source_model_generation_sha256"]
        != expected["model_generation_sha256"]
    ):
        raise StageError("score-finetuned", "best_finetuned differs from selection")
    append_dev_access_ledger(
        config,
        command="score-finetuned",
        checkpoint_sha256=str(expected["sha256"]),
        input_hashes=_declared_dev_hashes(config),
    )
    immutable = phase12_immutable_snapshot(config, require_all=True)
    adapted = _phase3_rerank_config(config)
    report = rerank.run_rerank_scoring(
        adapted,
        split="dev",
        requested_device=str(config["inference"]["device"]),
        overwrite=overwrite,
        checkpoint=checkpoint,
    )
    run_report = rerank.build_rerank_run(
        adapted,
        split="dev",
        depth=int(config["protocol"]["official_depth"]),
        overwrite=overwrite,
    )
    _validate_finetuned_score_binding(config, selection)
    verify_phase12_immutable(config, immutable)
    return {"status": "PASS", "scoring": report, "run": run_report}


def _run_frame(path: Path) -> pd.DataFrame:
    run = read_trec_run(str(path), split="dev")
    return run.rename(columns={"source_rank": "rank"})[
        ["query_id", "docid", "rank", "bm25_score", "tag"]
    ]


def _trec_eval(
    config: Mapping[str, Any], run_path: Path, arguments: Sequence[str]
) -> str:
    prepared = _prepared_qrels_path(config)
    if not prepared.is_file():
        raise StageError(
            "evaluate-phase3",
            "prepare-dev-evaluation must run before official evaluation",
        )
    phase2 = rerank.load_rerank_config(repository_root(config) / "configs/rerank.yaml")
    if config.get("_trec_eval_cli_path") is not None:
        phase2["_trec_eval_cli_path"] = config["_trec_eval_cli_path"]
    executable = rerank.resolve_trec_eval(phase2)
    rerank.validate_trec_eval_build_provenance(phase2, executable=executable)
    command = [str(executable), *map(str, arguments), str(prepared), str(run_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
    if result.returncode != 0:
        raise StageError(
            "evaluate-phase3/trec-eval",
            "trec_eval failed",
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout


def classify_ml_outcome(
    *,
    mean_delta: float,
    ci_lower: float,
    ci_upper: float,
    practical_threshold: float = 0.010,
    zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    if not ci_lower <= mean_delta <= ci_upper:
        raise ValueError("bootstrap invariant ci_lower <= mean_delta <= ci_upper failed")
    if ci_upper < 0:
        label = "degradation_confirmed"
    elif ci_lower > 0 and mean_delta >= practical_threshold:
        label = "improvement_confirmed"
    elif ci_lower > 0:
        label = "positive_below_practical_threshold"
    elif abs(mean_delta) <= zero_tolerance:
        label = "no_detectable_change"
    elif mean_delta > 0:
        label = "inconclusive_positive"
    else:
        label = "inconclusive_negative"
    statistical_direction = (
        "positive" if ci_lower > 0 else "negative" if ci_upper < 0 else "inconclusive"
    )
    return {
        "label": label,
        "mean_delta": float(mean_delta),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "statistical_direction": statistical_direction,
        "practical_magnitude": (
            "above_threshold"
            if abs(mean_delta) >= practical_threshold
            else "below_threshold"
        ),
        "improvement_confirmed": label == "improvement_confirmed",
        "degradation_confirmed": label == "degradation_confirmed",
        "practical_threshold": float(practical_threshold),
        "zero_tolerance": float(zero_tolerance),
    }


def build_three_way_comparison(
    bm25_per_query: Mapping[str, float],
    zero_shot_per_query: Mapping[str, float],
    finetuned_per_query: Mapping[str, float],
    *,
    resamples: int = 10_000,
    seed: int = 20260802,
    confidence: float = 0.95,
    practical_threshold: float = 0.010,
    zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Pure three-system paired comparison used by tests and final reporting."""

    if set(bm25_per_query) != set(zero_shot_per_query) or set(bm25_per_query) != set(
        finetuned_per_query
    ):
        raise ValueError("three-way query universes differ")
    primary = paired_ranking_comparison(
        zero_shot_per_query,
        finetuned_per_query,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        tie_tolerance=zero_tolerance,
    )
    contextual = paired_ranking_comparison(
        bm25_per_query,
        finetuned_per_query,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        tie_tolerance=zero_tolerance,
    )
    interval = primary["paired_bootstrap"]
    outcome = classify_ml_outcome(
        mean_delta=float(primary["mean_delta"]),
        ci_lower=float(interval["ci_low"]),
        ci_upper=float(interval["ci_high"]),
        practical_threshold=practical_threshold,
        zero_tolerance=zero_tolerance,
    )
    return {
        "pipeline_status": "PASS",
        "primary_finetuned_minus_zero_shot": primary,
        "context_finetuned_minus_bm25": contextual,
        "ml_outcome": outcome,
    }


def classify_control_result(
    *,
    mean_delta: float,
    ci_lower: float,
    structural_checks_passed: bool = True,
) -> str:
    if not structural_checks_passed:
        return "FAIL"
    if ci_lower > 0:
        return "BLOCKED_FOR_REVIEW"
    if mean_delta > 0:
        return "WARN"
    return "PASS"


def phase3_score_tie_statistics(scores: pd.DataFrame) -> dict[str, int]:
    raw = raw_score_tie_statistics(scores)
    return {
        "exact_raw_float32_tie_groups": int(raw["raw_score_tie_groups"]),
        "queries_with_any_tie": int(raw["queries_with_any_raw_score_tie"]),
        "queries_with_top10_tie": int(raw["queries_with_top10_raw_score_tie"]),
        "queries_with_boundary_tie": int(raw["ties_crossing_rank10_boundary"]),
    }


def _official_metrics(
    config: Mapping[str, Any], run_path: Path
) -> dict[str, Any]:
    ndcg_output = _trec_eval(config, run_path, ["-c", "-M", "100", "-m", "ndcg_cut.10"])
    recall_output = _trec_eval(config, run_path, ["-c", "-m", "recall.100"])
    mrr_output = _trec_eval(config, run_path, ["-c", "-M", "10", "-m", "recip_rank"])
    query_output = _trec_eval(
        config, run_path, ["-c", "-M", "100", "-q", "-m", "ndcg_cut.10"]
    )
    return {
        "ndcg_at_10": parse_trec_eval_metric(ndcg_output, "ndcg_cut_10"),
        "recall_at_100": parse_trec_eval_metric(recall_output, "recall_100"),
        "mrr_at_10": parse_trec_eval_metric(mrr_output, "recip_rank"),
        "per_query_ndcg_at_10": parse_trec_eval_per_query(query_output, "ndcg_cut_10"),
        "commands": {
            "ndcg": ["-c", "-M", "100", "-m", "ndcg_cut.10"],
            "recall": ["-c", "-m", "recall.100"],
            "mrr": ["-c", "-M", "10", "-m", "recip_rank"],
            "per_query": ["-c", "-M", "100", "-q", "-m", "ndcg_cut.10"],
        },
    }


def _tie_diagnostic(config: Mapping[str, Any]) -> dict[str, Any]:
    systems = {}
    for name, value in (
        ("zero_shot", config["dev_inputs"]["zeroshot_scores"]),
        ("finetuned", config["artifacts"]["finetuned_scores"]),
    ):
        path = resolve_path(config, value)
        scores = pd.read_parquet(path, columns=["query_id", "docid", "score"])
        systems[name] = phase3_score_tie_statistics(scores)
    report = {
        "status": "PASS",
        "scope": "stored_zero_shot_and_finetuned_float32_scores_only",
        "additional_inference": False,
        "systems": systems,
    }
    atomic_write_json(resolve_path(config, config["metrics"]["tie_diagnostic"]), report)
    return report


def _judged_only_quality(
    ranking: pd.DataFrame,
    candidates: pd.DataFrame,
    qrels: pd.DataFrame,
    query_ids: Sequence[str],
) -> dict[str, Any]:
    judged_keys = candidates.loc[
        candidates["judgment"].astype("string").ne("unjudged"), ["query_id", "docid"]
    ]
    judged = ranking.merge(judged_keys, on=["query_id", "docid"], how="inner")
    judged = judged.sort_values(["query_id", "rank"], kind="mergesort")
    judged["rank"] = judged.groupby("query_id", sort=False).cumcount().add(1)
    return evaluate_ranked_ndcg_at_10(judged, qrels, query_ids=query_ids)


def evaluate_phase3(
    config: Mapping[str, Any], *, overwrite: bool = False
) -> dict[str, Any]:
    validate_finetune_config(config)
    selection_path = resolve_path(config, config["audits"]["checkpoint_selection"])
    if not selection_path.is_file():
        raise StageError("evaluate-phase3", "checkpoint selection is missing")
    selection = read_json(selection_path, training_path=False)
    _validate_selected_training_contract(config, selection)
    immutable = phase12_immutable_snapshot(config, require_all=True)
    three_way_path = resolve_path(config, config["metrics"]["three_way"])
    finetuned_path = resolve_path(config, config["metrics"]["finetuned"])
    tie_path = resolve_path(config, config["metrics"]["tie_diagnostic"])
    existing_states = {
        "three_way": three_way_path.is_file(),
        "finetuned": finetuned_path.is_file(),
        "tie_diagnostic": tie_path.is_file(),
    }
    if all(existing_states.values()) and not overwrite:
        append_dev_access_ledger(
            config,
            command="evaluate-phase3",
            checkpoint_sha256=str(
                selection["best_finetuned_checkpoint"]["sha256"]
            ),
            input_hashes=_declared_dev_hashes(config),
        )
        _validate_finetuned_score_binding(config, selection)
        existing_three_way = read_json(three_way_path, training_path=False)
        existing_tie = read_json(tie_path, training_path=False)
        existing_finetuned = read_json(finetuned_path, training_path=False)
        if (
            existing_three_way.get("pipeline_status") != "PASS"
            or existing_finetuned.get("pipeline_status") != "PASS"
            or existing_tie.get("status") != "PASS"
            or existing_three_way.get("selection") != selection
        ):
            raise ValueError("existing Phase 3 evaluation outputs are stale")
        verify_phase12_immutable(config, immutable)
        return {
            "status": "PASS",
            "action": "reused",
            "three_way": existing_three_way,
            "tie_diagnostic": existing_tie,
        }
    if any(existing_states.values()) and not overwrite:
        raise ValueError(
            f"partial Phase 3 evaluation outputs exist: {existing_states}; use --overwrite"
        )
    prepared = _prepared_qrels_path(config)
    prepared_sidecar = prepared.with_suffix(".json")
    if not prepared.is_file() or not prepared_sidecar.is_file():
        raise StageError("evaluate-phase3", "prepared evaluation qrels are missing")
    prepared_parquet = _prepared_qrels_parquet_path(config)
    if not prepared_parquet.is_file():
        raise StageError("evaluate-phase3", "prepared qrels Parquet is missing")
    prepared_contract = read_json(prepared_sidecar, training_path=False)
    if (
        prepared_contract.get("prepared_sha256") != sha256_file(prepared)
        or prepared_contract.get("prepared_parquet_sha256")
        != sha256_file(prepared_parquet)
        or prepared_contract.get("checkpoint_sha256")
        != selection["best_finetuned_checkpoint"]["sha256"]
    ):
        raise StageError(
            "evaluate-phase3/prepared-qrels",
            "prepared evaluation qrels are stale or corrupted",
        )
    append_dev_access_ledger(
        config,
        command="evaluate-phase3",
        checkpoint_sha256=str(selection["best_finetuned_checkpoint"]["sha256"]),
        input_hashes=_declared_dev_hashes(config),
    )
    _validate_finetuned_score_binding(config, selection)
    qrels = pd.read_parquet(prepared_parquet)
    required_qrel_columns = {"query_id", "docid", "relevance_grade"}
    if not required_qrel_columns.issubset(qrels.columns):
        raise StageError(
            "evaluate-phase3/prepared-qrels", "prepared qrels schema is invalid"
        )
    if qrels.duplicated(["query_id", "docid"], keep=False).any():
        raise StageError(
            "evaluate-phase3/prepared-qrels", "prepared qrels contain duplicates"
        )
    if len(qrels) != int(prepared_contract["row_count"]):
        raise StageError(
            "evaluate-phase3/prepared-qrels", "prepared qrels row count changed"
        )
    candidate_path = resolve_path(config, config["dev_inputs"]["dev_candidates"])
    candidates = pd.read_parquet(candidate_path)
    if not candidates["split"].astype("string").eq("dev").all():
        raise StageError(
            "evaluate-phase3/candidates",
            "evaluation candidate artifact contains another split",
        )
    candidates = candidates.loc[candidates["split"].astype("string").eq("dev")].copy()
    query_ids = sorted(candidates["query_id"].astype("string").map(str).unique())
    run_paths = {
        "bm25": resolve_path(config, config["dev_inputs"]["bm25_run"]),
        "zero_shot": resolve_path(config, config["dev_inputs"]["zeroshot_run"]),
        "finetuned": resolve_path(config, config["artifacts"]["finetuned_run"]),
    }
    rankings = {name: _run_frame(path) for name, path in run_paths.items()}
    invariant_bm25_zero = assert_candidate_set_invariant(rankings["bm25"], rankings["zero_shot"])
    invariant_bm25_fine = assert_candidate_set_invariant(rankings["bm25"], rankings["finetuned"])
    score_frame = pd.read_parquet(
        resolve_path(config, config["artifacts"]["finetuned_scores"]),
        columns=["query_id", "docid", "score"],
    )
    expected_finetuned_ranking = rank_candidates_by_score(candidates, score_frame)
    observed_finetuned_ranking = rankings["finetuned"].sort_values(
        ["query_id", "rank"], kind="mergesort"
    )
    expected_finetuned_ranking = expected_finetuned_ranking.sort_values(
        ["query_id", "rank"], kind="mergesort"
    )
    expected_keys_in_order = list(
        map(
            tuple,
            expected_finetuned_ranking[["query_id", "docid", "rank"]].to_numpy(),
        )
    )
    observed_keys_in_order = list(
        map(
            tuple,
            observed_finetuned_ranking[["query_id", "docid", "rank"]].to_numpy(),
        )
    )
    if observed_keys_in_order != expected_keys_in_order:
        raise StageError(
            "evaluate-phase3/score-run-binding",
            "fine-tuned run is not the deterministic ranking of persisted scores",
        )
    expected_trec_scores = (
        int(config["protocol"]["trec_score_base"])
        - observed_finetuned_ranking["rank"].astype("int64")
    ).astype("float64")
    if not np.array_equal(
        observed_finetuned_ranking["bm25_score"].astype("float64").to_numpy(),
        expected_trec_scores.to_numpy(),
    ):
        raise StageError(
            "evaluate-phase3/score-run-binding",
            "fine-tuned run does not use the frozen rank-preserving TREC scores",
        )
    official = {name: _official_metrics(config, path) for name, path in run_paths.items()}
    query_universe = set(query_ids)
    for name, metrics_for_system in official.items():
        observed_queries = set(metrics_for_system["per_query_ndcg_at_10"])
        if observed_queries != query_universe:
            raise StageError(
                "evaluate-phase3/query-universe",
                f"{name} trec_eval per-query universe differs from candidates",
                missing=sorted(query_universe - observed_queries)[:10],
                extra=sorted(observed_queries - query_universe)[:10],
            )
    phase2_config = rerank.load_rerank_config(
        repository_root(config) / "configs/rerank.yaml"
    )
    if config.get("_trec_eval_cli_path") is not None:
        phase2_config["_trec_eval_cli_path"] = config["_trec_eval_cli_path"]
    trec_executable = rerank.resolve_trec_eval(phase2_config)
    trec_provenance = rerank.validate_trec_eval_build_provenance(
        phase2_config, executable=trec_executable
    )
    python_metrics = {
        name: evaluate_ranked_ndcg_at_10(ranking, qrels, query_ids=query_ids)
        for name, ranking in rankings.items()
    }
    tolerance = float(config["validation"]["tolerance"])
    for name in official:
        difference = abs(
            float(official[name]["ndcg_at_10"])
            - float(python_metrics[name]["ndcg_at_10"])
        )
        if difference > tolerance:
            raise StageError(
                "evaluate-phase3/python-cross-check",
                f"{name} Python nDCG differs from NIST trec_eval",
                absolute_difference=difference,
                tolerance=tolerance,
            )
    for name, reference_key in (
        ("bm25", "reference_bm25_ndcg_at_10"),
        ("zero_shot", "reference_zero_shot_ndcg_at_10"),
    ):
        if abs(
            float(official[name]["ndcg_at_10"])
            - float(config["evaluation"][reference_key])
        ) > tolerance:
            raise StageError(
                "evaluate-phase3/reference",
                f"{name} nDCG@10 differs from the frozen Phase 1/2 reference",
            )
    recalls = [official[name]["recall_at_100"] for name in run_paths]
    if not recalls[0] == recalls[1] == recalls[2]:
        raise StageError("evaluate-phase3/recall", "Recall@100 differs despite equal key sets")
    sparse = {
        name: sparse_judgment_diagnostics(
            candidates=candidates,
            qrels=qrels,
            ranking=ranking,
            bm25_ranking=rankings["bm25"],
            query_ids=query_ids,
        )
        for name, ranking in rankings.items()
    }
    for name in sparse:
        sparse[name]["standard_ndcg_at_10"] = float(
            python_metrics[name]["ndcg_at_10"]
        )
    primary = paired_ranking_comparison(
        official["zero_shot"]["per_query_ndcg_at_10"],
        official["finetuned"]["per_query_ndcg_at_10"],
        resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
        confidence=float(config["evaluation"]["bootstrap_confidence"]),
        tie_tolerance=float(config["evaluation"]["zero_tolerance"]),
    )
    contextual = paired_ranking_comparison(
        official["bm25"]["per_query_ndcg_at_10"],
        official["finetuned"]["per_query_ndcg_at_10"],
        resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
        confidence=float(config["evaluation"]["bootstrap_confidence"]),
        tie_tolerance=float(config["evaluation"]["zero_tolerance"]),
    )
    bootstrap = primary["paired_bootstrap"]
    outcome = classify_ml_outcome(
        mean_delta=float(primary["mean_delta"]),
        ci_lower=float(bootstrap["ci_low"]),
        ci_upper=float(bootstrap["ci_high"]),
        practical_threshold=float(config["evaluation"]["minimum_practically_relevant_delta"]),
        zero_tolerance=float(config["evaluation"]["zero_tolerance"]),
    )
    strata = stratified_delta_summary(
        candidates=candidates,
        baseline_per_query=python_metrics["zero_shot"]["per_query"],
        system_per_query=python_metrics["finetuned"]["per_query"],
        oracle_per_query=sparse["bm25"]["oracle_per_query"],
    )
    oracle_mean = float(sparse["bm25"]["oracle_ndcg_at_10_over_candidates"])
    fine_judged = float(sparse["finetuned"]["judged_at_10"])
    zero_judged = float(sparse["zero_shot"]["judged_at_10"])
    fine_condensed = float(sparse["finetuned"]["condensed_ndcg_at_10"])
    zero_condensed = float(sparse["zero_shot"]["condensed_ndcg_at_10"])
    weak_pairs = pd.read_parquet(
        resolve_path(config, config["artifacts"]["pairs_weak_negatives"])
    )
    weak_docids = set(
        weak_pairs.loc[
            weak_pairs["negative_source"].astype("string").eq("weak_unjudged"),
            "negative_docid",
        ].map(str)
    )
    fine_top10 = set(
        rankings["finetuned"].loc[rankings["finetuned"]["rank"].le(10), "docid"].map(str)
    )
    fine_extras = {
        "judged_only_candidate_quality": _judged_only_quality(
            rankings["finetuned"], candidates, qrels, query_ids
        ),
        "change_vs_zero_shot": {
            "judged_at_10": fine_judged - zero_judged,
            "pairwise_unjudged_relevant_inversions": int(
                sparse["finetuned"]["pairwise_unjudged_relevant_inversions"]
                - sparse["zero_shot"]["pairwise_unjudged_relevant_inversions"]
            ),
        },
        "training_weak_documents_promoted_to_dev_top10_fraction": (
            len(weak_docids & fine_top10) / len(weak_docids)
            if weak_docids
            else 0.0
        ),
    }
    flags = {
        "pooling_bias_suspected": (
            primary["mean_delta"] > 0 and zero_judged - fine_judged > 0.05
        ),
        "pool_overfit_suspected": (
            fine_judged > zero_judged and fine_condensed < zero_condensed
        ),
    }
    three_way = {
        "pipeline_status": "PASS",
        "systems": official,
        "trec_eval_provenance": trec_provenance,
        "python_cross_check": {
            name: {
                "ndcg_at_10": value["ndcg_at_10"],
                "absolute_difference": abs(value["ndcg_at_10"] - official[name]["ndcg_at_10"]),
            }
            for name, value in python_metrics.items()
        },
        "candidate_set_invariants": {
            "bm25_vs_zero_shot": invariant_bm25_zero,
            "bm25_vs_finetuned": invariant_bm25_fine,
        },
        "primary_finetuned_minus_zero_shot": primary,
        "context_finetuned_minus_bm25": contextual,
        "ml_outcome": outcome,
        "strata": strata,
        "candidate_oracle_ndcg_at_10": oracle_mean,
        "candidate_oracle_gap": {
            name: oracle_mean - float(official[name]["ndcg_at_10"])
            for name in official
        },
        "sparse_judgment_diagnostics": sparse,
        "finetuned_additional_diagnostics": fine_extras,
        "flags": flags,
        "selection": selection,
        "prepared_qrels": prepared_contract,
    }
    atomic_write_json(three_way_path, three_way)
    atomic_write_json(
        finetuned_path,
        {
            "pipeline_status": "PASS",
            "official": official["finetuned"],
            "python": python_metrics["finetuned"],
            "sparse_judgment_diagnostics": sparse["finetuned"],
            "additional_diagnostics": fine_extras,
            "flags": flags,
            "checkpoint": selection["best_finetuned_checkpoint"],
        },
    )
    tie = _tie_diagnostic(config)
    _write_model_card(config, selection, final_evaluation=three_way)
    verify_phase12_immutable(config, immutable)
    return {"status": "PASS", "three_way": three_way, "tie_diagnostic": tie}


def _schema_json(path: Path) -> dict[str, Any] | None:
    if path.suffix != ".parquet":
        return None
    schema = pq.read_schema(path)
    return {
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ]
    }


def _row_count(path: Path) -> int | None:
    if path.suffix == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    if path.suffix in {".trec", ".jsonl"}:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return None


def _run_provenance(
    config: Mapping[str, Any],
    run_id: str,
    pairs_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    run_manifest, components, fingerprint = _current_run_training_contract(
        config, run_id
    )
    regime = str(run_manifest["regime"])
    section = pairs_manifest.get("regimes", {}).get(regime)
    if not isinstance(section, Mapping):
        raise StageError(
            "package-phase3", f"pairs manifest is missing regime {regime}"
        )
    return {
        "run_id": run_id,
        "regime_id": regime,
        "learning_rate": float(run_manifest["learning_rate"]),
        "epochs": int(run_manifest["epochs"]),
        "training_fingerprint_sha256": fingerprint,
        "training_fingerprint_components": components,
        "pair_file_sha256": section["pair_file_sha256"],
        "pair_manifest_section_sha256": section[
            "pair_manifest_section_sha256"
        ],
        "pair_count": int(section["pair_count"]),
        "usable_query_count": int(section["usable_query_count"]),
        "optimizer": {
            "id": "AdamW",
            "learning_rate": float(run_manifest["learning_rate"]),
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": config["training"]["weight_decay"],
            "no_decay_patterns": config["training"]["no_decay_patterns"],
            "max_grad_norm": config["training"]["max_grad_norm"],
        },
        "scheduler": {
            "id": "linear_with_warmup",
            "warmup_ratio": config["training"]["warmup_ratio"],
            "optimizer_steps": run_manifest.get("optimizer_steps"),
        },
        "optimizer_steps": run_manifest.get("optimizer_steps"),
        "last_window_query_count": run_manifest.get(
            "last_window_query_count"
        ),
        "token_cache_bytes": run_manifest.get("token_cache_bytes"),
        "token_cache_representation": run_manifest.get(
            "token_cache_representation"
        ),
        "throughput": run_manifest.get("throughput"),
        "validation_history": run_manifest.get("validation_history"),
        "peak_rss_bytes": run_manifest.get("peak_rss_bytes"),
        "peak_gpu_memory_bytes": run_manifest.get("peak_gpu_memory_bytes"),
        "timestamps": {
            "run_started_at": run_manifest.get("started_at"),
            "run_completed_at": run_manifest.get("completed_at"),
        },
    }


def _artifact_producer(
    relative: str, *, selected_run_id: str
) -> dict[str, Any]:
    exact: dict[str, tuple[str, str, list[str]]] = {
        "LICENSE": ("release-license", "package-phase3", []),
        "NOTICE": ("third-party-notice", "package-phase3", []),
        "requirements/kaggle.lock": (
            "pinned-environment",
            "package-phase3",
            [],
        ),
        "artifacts/training/query_split.parquet": (
            "query-split",
            "build-training-split",
            [],
        ),
        "artifacts/training/pairs_judged_only.parquet": (
            "training-pairs",
            "build-training-pairs --regime judged_only",
            [],
        ),
        "artifacts/training/pairs_weak_negatives.parquet": (
            "training-pairs",
            "build-training-pairs --regime weak_negatives",
            [],
        ),
        "artifacts/training/pairs_control_c1.parquet": (
            "training-pairs",
            "build-training-pairs --regime control_c1",
            [],
        ),
        "artifacts/scores/dev_finetuned.parquet": (
            "dev-scoring",
            "score-finetuned",
            [selected_run_id],
        ),
        "artifacts/runs/dev_rerank_finetuned_k100.trec": (
            "trec-run",
            "score-finetuned",
            [selected_run_id],
        ),
        "reports/metrics/validation_checkpoint_metrics.json": (
            "validation-metrics",
            "validate-checkpoint and finetune",
            ["C1", "A1", "A2", "B1"],
        ),
        "reports/metrics/validation_ab_comparison.json": (
            "comparative-experiment",
            "select-checkpoint",
            ["A1", "A2", "B1"],
        ),
        "reports/metrics/dev_finetuned.json": (
            "final-metrics",
            "evaluate-phase3",
            [selected_run_id],
        ),
        "reports/metrics/dev_three_way_comparison.json": (
            "three-way-comparison",
            "evaluate-phase3",
            [selected_run_id],
        ),
        "reports/metrics/dev_score_tie_diagnostic.json": (
            "score-tie-diagnostic",
            "evaluate-phase3",
            [selected_run_id],
        ),
        "reports/training/A1_history.json": (
            "training-history",
            "finetune --run-id A1",
            ["A1"],
        ),
        "reports/training/A2_history.json": (
            "training-history",
            "finetune --run-id A2",
            ["A2"],
        ),
        "reports/training/B1_history.json": (
            "training-history",
            "finetune --run-id B1",
            ["B1"],
        ),
        "reports/audit/query_split_manifest.json": (
            "query-split-audit",
            "build-training-split",
            [],
        ),
        "reports/audit/pairs_manifest.json": (
            "pair-construction-audit",
            "build-training-pairs",
            [],
        ),
        "reports/audit/control_c1.json": (
            "control-report",
            "finetune --run-id C1",
            ["C1"],
        ),
        "reports/audit/finetune_smoke.json": (
            "smoke-gate",
            "smoke-finetune",
            [],
        ),
        "reports/audit/resource_report.json": (
            "resource-estimate",
            "smoke-finetune",
            [],
        ),
        "reports/audit/environment_freeze.txt": (
            "normalized-pip-freeze",
            "package-phase3",
            [],
        ),
        "reports/audit/dev_access_ledger.jsonl": (
            "dev-access-ledger",
            "prepare-dev-evaluation, score-finetuned, evaluate-phase3",
            [selected_run_id],
        ),
        "reports/audit/checkpoint_selection.json": (
            "checkpoint-selection",
            "select-checkpoint",
            ["A1", "A2", "B1"],
        ),
        "reports/audit/model_card.md": (
            "model-card",
            "select-checkpoint and evaluate-phase3",
            [selected_run_id],
        ),
        "reports/audit/finetune_protocol.yaml": (
            "protocol-snapshot",
            "package-phase3",
            [],
        ),
    }
    if relative not in exact:
        raise ValueError(f"artifact producer is not registered: {relative}")
    stage, command, run_ids = exact[relative]
    producer: dict[str, Any] = {
        "stage": stage,
        "cli_command": command,
        "run_ids": run_ids,
    }
    regimes = {
        "artifacts/training/pairs_judged_only.parquet": "judged_only",
        "artifacts/training/pairs_weak_negatives.parquet": "weak_negatives",
        "artifacts/training/pairs_control_c1.parquet": "control_c1",
    }
    if relative in regimes:
        producer["regime_id"] = regimes[relative]
    return producer


def _entry_input_hashes(
    config: Mapping[str, Any],
    relative: str,
    immutable_inputs: Mapping[str, str],
) -> dict[str, str]:
    if relative in {
        "LICENSE",
        "NOTICE",
        "requirements/kaggle.lock",
        "reports/audit/environment_freeze.txt",
    }:
        return {}
    dev_outputs = {
        "artifacts/scores/dev_finetuned.parquet",
        "artifacts/runs/dev_rerank_finetuned_k100.trec",
        "reports/metrics/dev_finetuned.json",
        "reports/metrics/dev_three_way_comparison.json",
        "reports/metrics/dev_score_tie_diagnostic.json",
        "reports/audit/dev_access_ledger.jsonl",
        "reports/audit/model_card.md",
    }
    if relative in dev_outputs:
        return dict(immutable_inputs)
    train_paths = {str(value) for value in config["inputs"].values()}
    return {
        path: digest
        for path, digest in immutable_inputs.items()
        if path in train_paths
    }


def build_training_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_path(config, config["audits"]["training_manifest"])
    selection = read_json(
        resolve_path(config, config["audits"]["checkpoint_selection"]),
        training_path=False,
    )
    selection_sha256 = _validate_checkpoint_selection_integrity(config, selection)
    selected_run_id = str(selection["best_finetuned_checkpoint"]["run_id"])
    pairs_manifest = read_json(
        resolve_path(config, config["audits"]["pairs_manifest"]),
        training_path=False,
    )
    run_provenance = {
        run_id: _run_provenance(config, run_id, pairs_manifest)
        for run_id in ("C1", "A1", "A2", "B1")
    }
    if run_provenance[selected_run_id]["training_fingerprint_sha256"] != selection[
        "best_finetuned_checkpoint"
    ]["training_fingerprint"]:
        raise StageError(
            "package-phase3", "selected training fingerprint is stale"
        )
    immutable_inputs = phase12_immutable_snapshot(config, require_all=True)
    sources = source_hashes(config)
    source_sha256 = {
        name: sources[name]
        for name in (
            "training_source_sha256",
            "scoring_source_sha256",
            "evaluation_source_sha256",
        )
    }
    configuration_sha256 = config_hashes(config)
    entries: list[dict[str, Any]] = []
    manifest_relative = str(config["audits"]["training_manifest"])
    for relative in RESULT_ZIP_MEMBERS:
        if relative == manifest_relative:
            continue
        source = repository_root(config) / relative
        if not source.is_file():
            raise StageError("package-phase3", f"result payload is missing: {relative}")
        producer = _artifact_producer(relative, selected_run_id=selected_run_id)
        entries.append(
            {
                "path": relative,
                "size_bytes": source.stat().st_size,
                "file_sha256": sha256_file(source),
                "row_count": _row_count(source),
                "parquet_schema": _schema_json(source),
                "producer": producer,
                "input_file_sha256": _entry_input_hashes(
                    config, relative, immutable_inputs
                ),
                "implementation_source_sha256": source_sha256,
                "configuration_sha256": configuration_sha256,
                "run_provenance": [
                    run_provenance[run_id] for run_id in producer["run_ids"]
                ],
            }
        )
    if any(entry["path"] == manifest_relative for entry in entries):
        raise RuntimeError("training manifest must never contain a self-reference")

    lock_path = resolve_path(config, config["release"]["environment_lock"])
    license_path = resolve_path(config, config["release"]["license_file"])
    notice_path = resolve_path(config, config["release"]["notice_file"])
    for required in (lock_path, license_path, notice_path):
        if not required.is_file():
            raise StageError("package-phase3", f"release file is missing: {required}")
    pinned_requirements = [
        line.strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    freeze_path = resolve_path(config, config["audits"]["environment_freeze"])
    if not freeze_path.is_file():
        raise StageError(
            "package-phase3", f"environment freeze is missing: {freeze_path}"
        )
    normalized_freeze = [
        line.strip()
        for line in freeze_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    commit, dirty = git_provenance(repository_root(config))
    payload = {
        "schema": {
            "name": "rusearchrank.phase3.training_manifest",
            "version": int(config["release"]["manifest_schema_version"]),
        },
        "schema_version": int(config["release"]["manifest_schema_version"]),
        "self_reference": False,
        "release": {
            "version": str(config["release"]["version"]),
            "ref": str(config["release"]["ref"]),
            "archive_schema_version": int(
                config["release"]["archive_schema_version"]
            ),
            "implementation_version": str(config["implementation"]["version"]),
            "selection_timestamp": selection["selected_at"],
            "git_commit": commit,
            "git_dirty": dirty,
        },
        "environment": {
            "target_python_version": str(
                config["release"]["kaggle_python_version"]
            ),
            "runtime_python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "package_versions": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "tokenizers": package_version("tokenizers"),
            },
            "lock_path": portable_path(config, lock_path),
            "lock_file_sha256": sha256_file(lock_path),
            "pinned_requirements": pinned_requirements,
            "normalized_pip_freeze_path": portable_path(config, freeze_path),
            "normalized_pip_freeze_file_sha256": sha256_file(freeze_path),
            "normalized_pip_freeze": normalized_freeze,
        },
        "model": {
            "base_model_id": config["base_model"]["id"],
            "base_model_revision": config["base_model"]["revision"],
            "tokenizer_revision": config["base_model"]["tokenizer_revision"],
            "selected_run_id": selected_run_id,
            "selected_checkpoint_file_sha256": selection[
                "best_finetuned_checkpoint"
            ]["sha256"],
            "selection_sha256": selection_sha256,
        },
        "input_file_sha256": immutable_inputs,
        "implementation_source_sha256": source_sha256,
        "configuration_sha256": configuration_sha256,
        "runs": run_provenance,
        "licensing": {
            "project_license": "MIT",
            "license_path": portable_path(config, license_path),
            "license_file_sha256": sha256_file(license_path),
            "notice_path": portable_path(config, notice_path),
            "notice_file_sha256": sha256_file(notice_path),
            "third_party_terms_preserved": True,
        },
        "limitations": [
            "Итоговые метрики относятся только к русской части MIRACL и top-100 BM25.",
            "Документы без экспертной оценки не являются подтверждёнными отрицательными примерами.",
            "Сравнение режимов на контрольной выборке носит исследовательский характер после выбора.",
            "Запуск с GPU другого типа или иными версиями окружения требует нового отчёта о воспроизводимости.",
        ],
        "hash_algorithm": "SHA-256",
        "entries": entries,
    }
    atomic_write_json(path, payload)
    _validate_training_manifest(config, payload)
    return payload


def _validate_training_manifest(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    expected_schema = int(config["release"]["manifest_schema_version"])
    if (
        manifest.get("schema_version") != expected_schema
        or manifest.get("schema")
        != {
            "name": "rusearchrank.phase3.training_manifest",
            "version": expected_schema,
        }
        or manifest.get("self_reference") is not False
        or manifest.get("hash_algorithm") != "SHA-256"
    ):
        raise ValueError("training manifest header is invalid")
    for section in (
        "release",
        "environment",
        "model",
        "input_file_sha256",
        "implementation_source_sha256",
        "configuration_sha256",
        "runs",
        "licensing",
        "limitations",
    ):
        if section not in manifest:
            raise ValueError(f"training manifest is missing section {section}")
    if manifest["release"].get("version") != config["release"]["version"] or manifest[
        "release"
    ].get("ref") != config["release"]["ref"]:
        raise ValueError("training manifest release identity is invalid")
    if manifest.get("input_file_sha256") != phase12_immutable_snapshot(
        config, require_all=True
    ):
        raise ValueError("training manifest input hashes are stale")
    if not isinstance(manifest.get("limitations"), list) or not manifest[
        "limitations"
    ]:
        raise ValueError("training manifest limitations are missing")

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not all(
        isinstance(entry, Mapping) for entry in entries
    ):
        raise ValueError("training manifest entries are invalid")
    manifest_relative = str(config["audits"]["training_manifest"])
    expected_paths = [
        relative for relative in RESULT_ZIP_MEMBERS if relative != manifest_relative
    ]
    observed_paths = [str(entry.get("path")) for entry in entries]
    if observed_paths != expected_paths or manifest_relative in observed_paths:
        raise ValueError("training manifest payload allowlist or order is invalid")
    required_fields = {
        "path",
        "size_bytes",
        "file_sha256",
        "row_count",
        "parquet_schema",
        "producer",
        "input_file_sha256",
        "implementation_source_sha256",
        "configuration_sha256",
        "run_provenance",
    }
    for entry, relative in zip(entries, expected_paths, strict=True):
        missing = required_fields.difference(entry)
        if missing:
            raise ValueError(
                f"training manifest entry is missing fields: {sorted(missing)}"
            )
        source = repository_root(config) / relative
        if (
            entry["size_bytes"] != source.stat().st_size
            or entry["file_sha256"] != sha256_file(source)
            or re.fullmatch(r"[0-9a-f]{64}", str(entry["file_sha256"])) is None
            or entry["row_count"] != _row_count(source)
            or entry["parquet_schema"] != _schema_json(source)
        ):
            raise ValueError(f"training manifest metadata mismatch: {relative}")
        producer = _artifact_producer(
            relative,
            selected_run_id=str(manifest["model"]["selected_run_id"]),
        )
        if entry["producer"] != producer:
            raise ValueError(f"training manifest producer mismatch: {relative}")
        observed_runs = [
            value.get("run_id") for value in entry["run_provenance"]
        ]
        if observed_runs != producer["run_ids"]:
            raise ValueError(f"training manifest run provenance mismatch: {relative}")

    entries_by_path = {entry["path"]: entry for entry in entries}
    for run_id in ("A1", "A2", "B1"):
        history = entries_by_path[f"reports/training/{run_id}_history.json"]
        if [item["run_id"] for item in history["run_provenance"]] != [run_id]:
            raise ValueError(f"{run_id} history has another run provenance")


def _safe_members(names: Sequence[str]) -> None:
    for name in names:
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts or str(member) != name:
            raise ValueError(f"unsafe ZIP member path: {name}")


def _write_deterministic_zip(
    destination: Path,
    members: Sequence[tuple[str, Path]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    _safe_members([name for name, _ in members])
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, source in members:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source.read_bytes())
        destination_backup = None
        if destination.exists():
            destination_backup = destination.with_name(f"{destination.name}.previous")
            if destination_backup.exists():
                destination_backup.unlink()
            destination.replace(destination_backup)
        try:
            temporary.replace(destination)
        except Exception:
            if destination_backup and destination_backup.exists():
                destination_backup.replace(destination)
            raise
        if destination_backup and destination_backup.exists():
            destination_backup.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _publish_validated_zip(
    destination: Path,
    members: Sequence[tuple[str, Path]],
    *,
    manifest: Mapping[str, Any] | None = None,
    model_forward: bool = False,
) -> dict[str, Any]:
    """Validate a candidate archive completely before atomic replacement."""

    candidate = destination.with_name(f".{destination.name}.candidate.{os.getpid()}")
    if candidate.exists():
        raise ValueError(f"candidate archive already exists: {candidate}")
    try:
        _write_deterministic_zip(candidate, members)
        report = _validate_zip(candidate, members, manifest=manifest)
        if model_forward:
            _validate_model_zip_forward(candidate)
        candidate.replace(destination)
        report["path"] = str(destination)
        report["sha256"] = sha256_file(destination)
        return report
    except Exception:
        if candidate.exists():
            candidate.unlink()
        raise


def _validate_zip(
    archive_path: Path,
    expected: Sequence[tuple[str, Path]],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_names = [name for name, _ in expected]
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        _safe_members(names)
        if names != expected_names:
            raise ValueError(f"ZIP member order mismatch: {names}")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failed: {bad}")
        with tempfile.TemporaryDirectory(prefix="phase3-zip-check-") as directory:
            root = Path(directory)
            archive.extractall(root)
            for name, source in expected:
                extracted = root / name
                if extracted.stat().st_size != source.stat().st_size:
                    raise ValueError(f"ZIP size mismatch: {name}")
                if sha256_file(extracted) != sha256_file(source):
                    raise ValueError(f"ZIP hash mismatch: {name}")
                if source.suffix == ".parquet":
                    if not pq.read_schema(extracted).equals(
                        pq.read_schema(source), check_metadata=False
                    ):
                        raise ValueError(f"ZIP Parquet schema mismatch: {name}")
                    if _row_count(extracted) != _row_count(source):
                        raise ValueError(f"ZIP Parquet row count mismatch: {name}")
    if manifest is not None:
        config_path_entry = next(
            (
                source
                for name, source in expected
                if name == "reports/audit/training_manifest.json"
            ),
            None,
        )
        if config_path_entry is None:
            raise ValueError("result ZIP is missing the training manifest")
        parsed_manifest = json.loads(config_path_entry.read_text(encoding="utf-8"))
        if parsed_manifest != dict(manifest):
            raise ValueError("training manifest source differs from validated object")
        entries = {entry["path"]: entry for entry in manifest["entries"]}
        for name, source in expected:
            if name == "reports/audit/training_manifest.json":
                continue
            entry = entries.get(name)
            if (
                entry is None
                or entry["file_sha256"] != sha256_file(source)
                or entry["size_bytes"] != source.stat().st_size
                or entry["row_count"] != _row_count(source)
                or entry["parquet_schema"] != _schema_json(source)
            ):
                raise ValueError(f"manifest payload mismatch: {name}")
    return {
        "path": str(archive_path),
        "sha256": sha256_file(archive_path),
        "members": expected_names,
    }


def _validate_model_zip_forward(archive_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="phase3-model-check-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(root)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(root, local_files_only=True)
        encoded = [
            encode_pair(
                tokenizer,
                query,
                build_document(title, text),
                max_length=320,
            )
            for query, title, text in (
                ("тест", "Заголовок", "Документ"),
                ("поиск", "", "Текст"),
                ("NBSP\u00a0", None, "значение\u00a0"),
                ("emoji 😀", "T", "D 🚀"),
            )
        ]
        model_inputs = tokenizer.pad(
            [
                {key: value for key, value in row.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
                for row in encoded
            ],
            padding=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**model_inputs).logits
        if list(logits.shape) != [4, 1] or not bool(torch.isfinite(logits).all()):
            raise ValueError("model ZIP four-pair forward validation failed")


def package_phase3(
    config: Mapping[str, Any], *, overwrite: bool = False
) -> dict[str, Any]:
    validate_finetune_config(config)
    selection = read_json(
        resolve_path(config, config["audits"]["checkpoint_selection"]),
        training_path=False,
    )
    _validate_selected_training_contract(config, selection)
    immutable = phase12_immutable_snapshot(config, require_all=True)
    freeze_path = resolve_path(config, config["audits"]["environment_freeze"])
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    installed = sorted(
        {
            f"{str(distribution.metadata.get('Name') or '').strip().lower().replace('_', '-')}"
            f"=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if str(distribution.metadata.get("Name") or "").strip()
        }
    )
    freeze_text = (
        "# Нормализованный pip freeze без локальных путей и editable-ссылок.\n"
        + "\n".join(installed)
        + "\n"
    )
    temporary_freeze = freeze_path.with_name(
        f"{freeze_path.name}.tmp.{os.getpid()}"
    )
    temporary_freeze.write_text(freeze_text, encoding="utf-8")
    temporary_freeze.replace(freeze_path)
    protocol_path = resolve_path(config, config["audits"]["protocol_snapshot"])
    config_path = Path(str(config["_config_path"]))
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    if protocol_path.exists() and not overwrite and protocol_path.read_bytes() != config_path.read_bytes():
        raise ValueError("existing finetune protocol snapshot differs from config")
    temporary_protocol = protocol_path.with_name(f"{protocol_path.name}.tmp.{os.getpid()}")
    temporary_protocol.write_bytes(config_path.read_bytes())
    temporary_protocol.replace(protocol_path)
    if sha256_file(protocol_path) != sha256_file(config_path):
        raise RuntimeError("finetune protocol snapshot is not byte-identical")
    manifest = build_training_manifest(config)
    manifest_path = resolve_path(config, config["audits"]["training_manifest"])
    manifest_sha = sha256_file(manifest_path)
    result_members = [
        (relative, repository_root(config) / relative) for relative in RESULT_ZIP_MEMBERS
    ]
    result_path = resolve_path(config, config["archive"]["results_zip"])
    if result_path.exists() and not overwrite:
        result_report = _validate_zip(result_path, result_members, manifest=manifest)
    else:
        result_report = _publish_validated_zip(
            result_path, result_members, manifest=manifest
        )
    run_id = str(selection["best_finetuned_checkpoint"]["run_id"])
    model_dir = resolve_path(config, config["artifacts"]["best_finetuned_dir"])
    checkpoint_metadata = read_json(
        model_dir / "checkpoint_sha256.json", training_path=False
    )
    _validate_best_finetuned_payload(model_dir, checkpoint_metadata)
    if (
        checkpoint_metadata["files"]["model.safetensors"]
        != selection["best_finetuned_checkpoint"]["sha256"]
        or checkpoint_metadata["checkpoint_payload_sha256"]
        != selection["best_finetuned_checkpoint"]["checkpoint_payload_sha256"]
        or checkpoint_metadata["source_model_generation_sha256"]
        != selection["best_finetuned_checkpoint"]["model_generation_sha256"]
    ):
        raise StageError(
            "package-phase3/model", "best_finetuned differs from selection"
        )
    model_card = resolve_path(config, config["audits"]["model_card"])
    release_sources = {
        "LICENSE": resolve_path(config, config["release"]["license_file"]),
        "NOTICE": resolve_path(config, config["release"]["notice_file"]),
    }
    model_sources = {
        name: (
            model_card
            if name == "model_card.md"
            else release_sources.get(name, model_dir / name)
        )
        for name in MODEL_ZIP_MEMBERS
    }
    missing = [name for name, source in model_sources.items() if not source.is_file()]
    if missing:
        raise StageError("package-phase3/model", f"model payload is missing: {missing}")
    model_members = [(name, model_sources[name]) for name in MODEL_ZIP_MEMBERS]
    model_path = resolve_path(
        config, str(config["archive"]["model_zip_template"]).format(run_id=run_id)
    )
    if model_path.exists() and not overwrite:
        model_report = _validate_zip(model_path, model_members)
        _validate_model_zip_forward(model_path)
    else:
        model_report = _publish_validated_zip(
            model_path, model_members, model_forward=True
        )
    verify_phase12_immutable(config, immutable)
    report = {
        "status": "PASS",
        "training_manifest_sha256": manifest_sha,
        "result_zip": result_report,
        "model_zip": model_report,
        "model_zip_contains_best_finetuned_even_if_zero_shot_won": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


__all__ = [
    "MODEL_ZIP_MEMBERS",
    "RESULT_ZIP_MEMBERS",
    "append_dev_access_ledger",
    "build_three_way_comparison",
    "build_training_manifest",
    "classify_control_result",
    "classify_ml_outcome",
    "evaluate_phase3",
    "package_phase3",
    "prepare_dev_evaluation",
    "phase3_score_tie_statistics",
    "score_finetuned",
    "select_checkpoint",
]
