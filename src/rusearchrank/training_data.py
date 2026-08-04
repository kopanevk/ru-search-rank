"""Deterministic, train-only Phase 3 split and pair materialization."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .data import CANDIDATE_COLUMNS, validate_candidate_schema


DEFAULT_FINETUNE_CONFIG = Path("configs/finetune.yaml")
IMPLEMENTATION_VERSION = "3.1.0"
TRAIN_ROLE = "train_fit"
VALIDATION_ROLE = "train_validation"
PAIR_REGIMES = ("judged_only", "weak_negatives", "control_c1")
CANONICAL_STRATA = (
    "(0,F)",
    "(0,T)",
    "(1,F)",
    "(1,T)",
    "(2,F)",
    "(2,T)",
    "(3+,F)",
    "(3+,T)",
)
_DEV_PATH_MARKERS = (
    "dev" + "_top100",
    "dev" + "_bm25",
    "dev" + "_rerank",
    "miracl-v1.0-ru" + "-dev",
)
_EVALUATION_SPLIT = "d" + "ev"

QUERY_SPLIT_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.large_string(), nullable=False),
        pa.field("split_role", pa.large_string(), nullable=False),
        pa.field("n_relevant_in_candidates", pa.int32(), nullable=False),
        pa.field("n_judged_negatives_in_candidates", pa.int32(), nullable=False),
        pa.field("relevant_bucket", pa.large_string(), nullable=False),
        pa.field("has_judged_negative", pa.bool_(), nullable=False),
        pa.field("stratum", pa.large_string(), nullable=False),
        pa.field("merged_stratum", pa.large_string(), nullable=False),
    ]
)
PAIR_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.large_string(), nullable=False),
        pa.field("positive_docid", pa.large_string(), nullable=False),
        pa.field("negative_docid", pa.large_string(), nullable=False),
        pa.field("negative_source", pa.large_string(), nullable=False),
        pa.field("negative_bm25_rank", pa.int32(), nullable=False),
        pa.field("pair_weight", pa.float32(), nullable=False),
    ]
)
CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("split", pa.large_string(), nullable=False),
        pa.field("query_id", pa.large_string(), nullable=False),
        pa.field("docid", pa.large_string(), nullable=False),
        pa.field("bm25_rank", pa.int64(), nullable=False),
        pa.field("bm25_score", pa.float64(), nullable=False),
        pa.field("relevance_grade", pa.int64(), nullable=True),
        pa.field("judgment", pa.large_string(), nullable=False),
        pa.field("relevance", pa.int64(), nullable=True),
        pa.field("is_judged", pa.bool_(), nullable=False),
    ]
)


def assert_not_dev(path: str | Path) -> None:
    """Reject every forbidden evaluation-path marker in a training I/O path."""

    normalized = str(path).replace("\\", "/").lower()
    matched = [marker for marker in _DEV_PATH_MARKERS if marker in normalized]
    if matched:
        raise ValueError(
            "training path is inside the isolated evaluation boundary: "
            f"{path} (matched {matched[0]!r})"
        )


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary JSON already exists: {temporary}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        loaded = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("temporary JSON is not an object")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def read_json(path: str | Path, *, training_path: bool = True) -> dict[str, Any]:
    source = Path(path)
    if training_path:
        assert_not_dev(source)
    if not source.is_file():
        raise ValueError(f"JSON file does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {source}")
    return payload


def repository_root(config: Mapping[str, Any]) -> Path:
    config_path = Path(str(config["_config_path"]))
    setting = Path(str(config.get("paths", {}).get("repository_root", ".")))
    return (
        setting.resolve()
        if setting.is_absolute()
        else (config_path.parent / setting).resolve()
    )


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root(config) / path).resolve()


def portable_path(config: Mapping[str, Any], path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repository_root(config)).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must stay inside the repository: {resolved}") from exc


def _exact(config: Mapping[str, Any], dotted: str, expected: Any) -> None:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"finetune config is missing {dotted}")
        value = value[part]
    if value != expected:
        raise ValueError(f"{dotted} must equal {expected!r}; got {value!r}")


def validate_finetune_config(config: Mapping[str, Any]) -> None:
    """Validate exact preregistered Phase 3 values, never loose ranges."""

    expected_section_keys = {
        "implementation": {"version", "score_schema_version"},
        "release": {
            "version",
            "manifest_schema_version",
            "archive_schema_version",
            "ref",
            "kaggle_python_version",
            "environment_lock",
            "license_file",
            "notice_file",
        },
        "base_model": {"id", "revision", "tokenizer_revision"},
        "input": {"max_length", "truncation", "title_separator", "pair_order"},
        "split": {
            "seed",
            "validation_fraction",
            "stratify_by",
            "relevant_buckets",
            "min_stratum_size",
        },
        "negatives": {
            "weak_rank_min",
            "weak_rank_max",
            "bucket_boundaries",
            "bucket_targets",
            "max_weak_negatives_per_query",
            "weak_negative_weight",
            "judged_negative_weight",
            "max_judged_pairs_per_query",
            "max_weak_pairs_per_query",
            "max_pairs_per_query",
            "spare_slot_priority",
            "resample_per_epoch",
        },
        "loss": {"id", "aggregation"},
        "training": {
            "epochs",
            "micro_batch_queries",
            "grad_accumulation",
            "max_sequences_per_microbatch",
            "weight_decay",
            "no_decay_patterns",
            "warmup_ratio",
            "max_grad_norm",
            "precision",
            "seed",
            "num_workers",
            "device",
            "token_cache",
            "minimum_free_ram_gib",
        },
        "runs": {"C1", "A1", "A2", "B1"},
        "validation": {
            "metric",
            "include_queries_without_relevant",
            "cross_check_with_trec_eval_once",
            "tolerance",
        },
        "selection": {
            "candidates_include_zero_shot",
            "epoch_tie_break",
            "run_tie_break_order",
            "production_tie_prefers_zero_shot",
            "ab_bootstrap_resamples",
            "ab_bootstrap_seed",
            "ab_analysis_role",
        },
        "control": {
            "c1_bootstrap_resamples",
            "c1_bootstrap_seed",
            "c1_blocking_rule",
        },
        "evaluation": {
            "minimum_practically_relevant_delta",
            "zero_tolerance",
            "bootstrap_resamples",
            "bootstrap_seed",
            "bootstrap_confidence",
            "reference_zero_shot_ndcg_at_10",
            "reference_bm25_ndcg_at_10",
        },
        "inference": {
            "batch_size",
            "cpu_batch_size",
            "device",
            "cuda_dtype",
            "score_dtype",
            "shard_queries",
        },
        "protocol": {
            "official_depth",
            "internal_tie_break",
            "trec_score_encoding",
            "trec_score_base",
            "trec_score_format",
        },
        "inputs": {
            "train_candidates",
            "queries",
            "passages",
            "train_qrels",
            "phase1_manifest",
            "phase2_manifest",
        },
        "dev_inputs": {
            "dev_candidates",
            "dev_qrels",
            "bm25_run",
            "zeroshot_run",
            "zeroshot_scores",
            "zeroshot_metrics",
            "bm25_metrics",
        },
        "artifacts": {
            "query_split",
            "validation_groups",
            "pairs_judged_only",
            "pairs_weak_negatives",
            "pairs_control_c1",
            "models_dir",
            "best_finetuned_dir",
            "finetuned_scores",
            "finetuned_run",
        },
        "audits": {
            "query_split_manifest",
            "pairs_manifest",
            "checkpoint_selection",
            "dev_access_ledger",
            "finetune_smoke",
            "training_manifest",
            "protocol_snapshot",
            "model_card",
            "control_report",
            "resource_report",
            "environment_freeze",
        },
        "metrics": {
            "training_history_template",
            "validation_checkpoint_metrics",
            "validation_ab_comparison",
            "finetuned",
            "three_way",
            "tie_diagnostic",
        },
        "archive": {"results_zip", "model_zip_template"},
        "paths": {"repository_root", "work_dir"},
    }
    actual_top_level = set(config).difference({"_config_path"})
    if actual_top_level != set(expected_section_keys):
        raise ValueError(
            "finetune config top-level schema changed: "
            f"missing={sorted(set(expected_section_keys) - actual_top_level)}, "
            f"extra={sorted(actual_top_level - set(expected_section_keys))}"
        )
    for section, expected_keys in expected_section_keys.items():
        value = config.get(section)
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            actual_keys = set(value) if isinstance(value, Mapping) else set()
            raise ValueError(
                f"finetune config section {section!r} changed: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
    expected_run_keys = {
        "C1": {
            "kind",
            "regime",
            "learning_rate",
            "epochs",
            "target_pairs",
            "control_seed",
            "shuffle_seed",
        },
        "A1": {"kind", "regime", "learning_rate", "epochs"},
        "A2": {"kind", "regime", "learning_rate", "epochs"},
        "B1": {"kind", "regime", "learning_rate_from", "epochs"},
    }
    for run_id, expected_keys in expected_run_keys.items():
        if set(config["runs"][run_id]) != expected_keys:
            raise ValueError(f"finetune config run {run_id} schema changed")

    exact_values = {
        "implementation.version": IMPLEMENTATION_VERSION,
        "implementation.score_schema_version": 1,
        "release.version": "1.0.1",
        "release.manifest_schema_version": 2,
        "release.archive_schema_version": 2,
        "release.ref": "phase3-v1.0.1",
        "release.kaggle_python_version": "3.12.13",
        "release.environment_lock": "requirements/kaggle.lock",
        "release.license_file": "LICENSE",
        "release.notice_file": "NOTICE",
        "base_model.id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "base_model.revision": "1427fd652930e4ba29e8149678df786c240d8825",
        "base_model.tokenizer_revision": "1427fd652930e4ba29e8149678df786c240d8825",
        "input.max_length": 320,
        "input.truncation": "only_second",
        "input.title_separator": "\n",
        "input.pair_order": "query_document",
        "split.seed": 20260803,
        "split.validation_fraction": 0.15,
        "split.stratify_by": "relevant_bucket_x_has_judged_negative",
        "split.relevant_buckets": [0, 1, 2, "3+"],
        "split.min_stratum_size": 20,
        "negatives.weak_rank_min": 26,
        "negatives.weak_rank_max": 100,
        "negatives.bucket_boundaries": [[26, 50], [51, 75], [76, 100]],
        "negatives.bucket_targets": [3, 3, 2],
        "negatives.max_weak_negatives_per_query": 8,
        "negatives.weak_negative_weight": 0.5,
        "negatives.judged_negative_weight": 1.0,
        "negatives.max_judged_pairs_per_query": 8,
        "negatives.max_weak_pairs_per_query": 8,
        "negatives.max_pairs_per_query": 16,
        "negatives.spare_slot_priority": "judged_first",
        "negatives.resample_per_epoch": False,
        "loss.id": "pairwise_logistic",
        "loss.aggregation": "query_weighted_mean_then_window_mean",
        "training.epochs": 3,
        "training.micro_batch_queries": 1,
        "training.grad_accumulation": 16,
        "training.max_sequences_per_microbatch": 40,
        "training.weight_decay": 0.01,
        "training.no_decay_patterns": ["bias", "LayerNorm.weight"],
        "training.warmup_ratio": 0.1,
        "training.max_grad_norm": 1.0,
        "training.precision": "fp32",
        "training.seed": 20260803,
        "training.num_workers": 0,
        "training.device": "cuda",
        "training.token_cache": "pretokenized_flat_int32",
        "training.minimum_free_ram_gib": 4,
        "runs.A1.learning_rate": 7.0e-6,
        "runs.A1.kind": "full",
        "runs.A1.regime": "judged_only",
        "runs.A1.epochs": 3,
        "runs.A2.learning_rate": 2.0e-5,
        "runs.A2.kind": "full",
        "runs.A2.regime": "judged_only",
        "runs.A2.epochs": 3,
        "runs.B1.kind": "full",
        "runs.B1.regime": "weak_negatives",
        "runs.B1.learning_rate_from": "best_judged_run",
        "runs.B1.epochs": 3,
        "runs.C1.kind": "shuffled_control",
        "runs.C1.regime": "control_c1",
        "runs.C1.learning_rate": 2.0e-5,
        "runs.C1.epochs": 1,
        "runs.C1.target_pairs": 2000,
        "runs.C1.control_seed": 20260805,
        "runs.C1.shuffle_seed": 20260804,
        "validation.metric": "ndcg_at_10",
        "validation.include_queries_without_relevant": True,
        "validation.cross_check_with_trec_eval_once": True,
        "validation.tolerance": 0.0001,
        "selection.candidates_include_zero_shot": True,
        "selection.epoch_tie_break": "earliest",
        "selection.run_tie_break_order": ["A1", "A2", "B1"],
        "selection.production_tie_prefers_zero_shot": True,
        "selection.ab_bootstrap_resamples": 10000,
        "selection.ab_bootstrap_seed": 20260803,
        "selection.ab_analysis_role": "exploratory_post_selection",
        "control.c1_bootstrap_resamples": 10000,
        "control.c1_bootstrap_seed": 20260803,
        "control.c1_blocking_rule": "ci_lower_bound_above_zero",
        "evaluation.minimum_practically_relevant_delta": 0.010,
        "evaluation.zero_tolerance": 1.0e-12,
        "evaluation.bootstrap_resamples": 10000,
        "evaluation.bootstrap_seed": 20260802,
        "evaluation.bootstrap_confidence": 0.95,
        "evaluation.reference_zero_shot_ndcg_at_10": 0.5365,
        "evaluation.reference_bm25_ndcg_at_10": 0.3342,
        "inference.batch_size": 64,
        "inference.cpu_batch_size": 16,
        "inference.device": "auto",
        "inference.cuda_dtype": "float16",
        "inference.score_dtype": "float32",
        "inference.shard_queries": 64,
        "protocol.official_depth": 100,
        "protocol.internal_tie_break": "raw_score_desc_then_docid_asc",
        "protocol.trec_score_encoding": "rank_preserving",
        "protocol.trec_score_base": 1_000_000,
        "protocol.trec_score_format": "%.4f",
        "archive.results_zip": "artifacts/rusearchrank_phase3_results_v1.0.1.zip",
        "archive.model_zip_template": (
            "artifacts/rusearchrank_phase3_model_{run_id}_v1.0.1.zip"
        ),
    }
    for dotted, expected in exact_values.items():
        _exact(config, dotted, expected)
    for key in ("revision", "tokenizer_revision"):
        revision = str(config.get("base_model", {}).get(key, ""))
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"base_model.{key} must be a 40-character hex SHA")
    if set(config.get("runs", {})) != {"C1", "A1", "A2", "B1"}:
        raise ValueError("runs registry must contain exactly C1, A1, A2, B1")
    if {
        float(config["runs"]["A1"]["learning_rate"]),
        float(config["runs"]["A2"]["learning_rate"]),
    } != {7.0e-6, 2.0e-5}:
        raise ValueError("judged-only learning-rate set changed")
    if int(config["runs"]["C1"]["epochs"]) != 1:
        raise ValueError("C1 must run exactly one epoch")
    for run_id in ("A1", "A2", "B1"):
        if int(config["runs"][run_id]["epochs"]) != 3:
            raise ValueError(f"{run_id} must run exactly three epochs")
    root = repository_root(config)
    if not root.is_dir():
        raise ValueError(f"configured repository root does not exist: {root}")
    path_sections = ("inputs", "dev_inputs", "artifacts", "audits", "metrics", "archive")
    for section in path_sections:
        values = config.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(f"finetune config section {section!r} must be a mapping")
        for raw in values.values():
            candidate = str(raw).replace("{run_id}", "A1")
            portable_path(config, resolve_path(config, candidate))
    for key in ("environment_lock", "license_file", "notice_file"):
        portable_path(config, resolve_path(config, str(config["release"][key])))


def load_finetune_config(path: str | Path = DEFAULT_FINETUNE_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ValueError(f"finetune config does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("finetune config must contain a mapping")
    payload["_config_path"] = str(config_path)
    validate_finetune_config(payload)
    return payload


def _filter_train(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if "split" not in frame.columns:
        raise ValueError(f"{label} is missing split")
    split_values = frame["split"].astype("string")
    if split_values.eq(_EVALUATION_SPLIT).any():
        raise ValueError(f"{label} contains isolated evaluation rows")
    result = frame.loc[split_values.eq("train")].copy()
    if not result["split"].astype("string").eq("train").all():
        raise RuntimeError(f"{label} train filter failed")
    if result.empty:
        raise ValueError(f"{label} contains no train rows")
    return result


def read_training_parquet(
    path: str | Path,
    *,
    label: str,
    filters: Any | None = None,
) -> pd.DataFrame:
    assert_not_dev(path)
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"{label} Parquet does not exist: {source}")
    return pd.read_parquet(source, filters=filters)


def _bucket(value: int) -> str:
    return str(value) if value in {0, 1, 2} else "3+"


def _stratum_label(bucket: str, has_negative: bool) -> str:
    return f"({bucket},{'T' if has_negative else 'F'})"


def merge_small_strata(
    sizes: Mapping[str, int], *, min_stratum_size: int = 20
) -> tuple[dict[str, str], list[str], dict[str, list[str]]]:
    """Apply the frozen partner/previous deterministic merge to stabilization."""

    if min_stratum_size <= 0:
        raise ValueError("min_stratum_size must be positive")
    unknown = sorted(set(sizes).difference(CANONICAL_STRATA))
    if unknown:
        raise ValueError(f"unknown strata: {unknown}")
    groups: list[set[str]] = [
        {name} for name in CANONICAL_STRATA if int(sizes.get(name, 0)) > 0
    ]
    index = {name: position for position, name in enumerate(CANONICAL_STRATA)}

    def group_size(group: set[str]) -> int:
        return sum(int(sizes.get(name, 0)) for name in group)

    def locate(name: str) -> set[str] | None:
        return next((group for group in groups if name in group), None)

    def merge(left: set[str], right: set[str]) -> set[str]:
        if left is right:
            return left
        combined = left | right
        groups.remove(left)
        groups.remove(right)
        groups.append(combined)
        groups.sort(key=lambda group: min(index[name] for name in group))
        return combined

    changed = True
    while changed and len(groups) > 1:
        changed = False
        for name in CANONICAL_STRATA:
            current = locate(name)
            if current is None or group_size(current) >= min_stratum_size:
                continue
            bucket, flag = name[1:-1].split(",")
            partner = _stratum_label(bucket, flag != "T")
            partner_group = locate(partner)
            if partner_group is not None and partner_group is not current:
                current = merge(current, partner_group)
                changed = True
            if group_size(current) >= min_stratum_size or len(groups) == 1:
                continue
            position = groups.index(current)
            neighbour = groups[position - 1] if position > 0 else groups[position + 1]
            merge(current, neighbour)
            changed = True
        groups.sort(key=lambda group: min(index[name] for name in group))

    labels: dict[int, str] = {}
    for group in groups:
        ordered = sorted(group, key=index.__getitem__)
        labels[id(group)] = "+".join(ordered)
    mapping = {
        name: labels[id(group)]
        for group in groups
        for name in sorted(group, key=index.__getitem__)
    }
    order = [labels[id(group)] for group in groups]
    members = {
        labels[id(group)]: sorted(group, key=index.__getitem__) for group in groups
    }
    return mapping, order, members


def _factor_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    total = len(frame)
    bucket_counts = frame["relevant_bucket"].value_counts().to_dict()
    negative_counts = frame["has_judged_negative"].value_counts().to_dict()
    return {
        "query_count": int(total),
        "relevant_bucket": {
            bucket: {
                "count": int(bucket_counts.get(bucket, 0)),
                "fraction": float(bucket_counts.get(bucket, 0) / total) if total else 0.0,
            }
            for bucket in ("0", "1", "2", "3+")
        },
        "has_judged_negative": {
            key: {
                "count": int(negative_counts.get(value, 0)),
                "fraction": float(negative_counts.get(value, 0) / total) if total else 0.0,
            }
            for key, value in (("false", False), ("true", True))
        },
    }


def build_query_split_frame(
    candidates: pd.DataFrame,
    *,
    seed: int = 20260803,
    validation_fraction: float = 0.15,
    min_stratum_size: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the frozen query-level split and its deterministic audit payload."""

    train = _filter_train(candidates, label="training candidates")
    validate_candidate_schema(train)
    grouped = train.groupby("query_id", sort=True)["judgment"].agg(
        n_relevant_in_candidates=lambda values: int((values == "relevant").sum()),
        n_judged_negatives_in_candidates=lambda values: int(
            (values == "judged_non_relevant").sum()
        ),
    ).reset_index()
    grouped["query_id"] = grouped["query_id"].astype("string")
    grouped["n_relevant_in_candidates"] = grouped[
        "n_relevant_in_candidates"
    ].astype("int32")
    grouped["n_judged_negatives_in_candidates"] = grouped[
        "n_judged_negatives_in_candidates"
    ].astype("int32")
    grouped["relevant_bucket"] = grouped["n_relevant_in_candidates"].map(
        lambda value: _bucket(int(value))
    ).astype("string")
    grouped["has_judged_negative"] = grouped[
        "n_judged_negatives_in_candidates"
    ].gt(0)
    grouped["stratum"] = [
        _stratum_label(str(bucket), bool(has_negative))
        for bucket, has_negative in zip(
            grouped["relevant_bucket"],
            grouped["has_judged_negative"],
            strict=True,
        )
    ]
    sizes = {
        name: int(grouped["stratum"].eq(name).sum()) for name in CANONICAL_STRATA
    }
    merge_map, merged_order, merged_members = merge_small_strata(
        sizes, min_stratum_size=min_stratum_size
    )
    grouped["merged_stratum"] = grouped["stratum"].map(merge_map).astype("string")
    roles: dict[str, str] = {}
    merged_sizes: dict[str, int] = {}
    validation_sizes: dict[str, int] = {}
    for merged in merged_order:
        query_ids = sorted(
            grouped.loc[grouped["merged_stratum"].eq(merged), "query_id"].map(str)
        )
        if not query_ids:
            continue
        permutation = np.random.default_rng(seed).permutation(len(query_ids))
        ordered = [query_ids[int(index)] for index in permutation]
        validation_count = int(math.ceil(validation_fraction * len(query_ids)))
        validation_ids = set(ordered[:validation_count])
        merged_sizes[merged] = len(query_ids)
        validation_sizes[merged] = validation_count
        for query_id in query_ids:
            roles[query_id] = (
                VALIDATION_ROLE if query_id in validation_ids else TRAIN_ROLE
            )
    grouped["split_role"] = grouped["query_id"].map(roles).astype("string")
    output = grouped[
        [
            "query_id",
            "split_role",
            "n_relevant_in_candidates",
            "n_judged_negatives_in_candidates",
            "relevant_bucket",
            "has_judged_negative",
            "stratum",
            "merged_stratum",
        ]
    ].sort_values("query_id", kind="mergesort").reset_index(drop=True)
    fit_ids = set(output.loc[output["split_role"].eq(TRAIN_ROLE), "query_id"].map(str))
    validation_ids = set(
        output.loc[output["split_role"].eq(VALIDATION_ROLE), "query_id"].map(str)
    )
    overlap = sorted(fit_ids & validation_ids)
    if overlap:
        raise RuntimeError(f"query split overlap: {overlap[:10]}")
    if fit_ids | validation_ids != set(output["query_id"].map(str)):
        raise RuntimeError("query split does not cover the full train query universe")
    if any(size < min_stratum_size for size in merged_sizes.values()) and len(merged_sizes) > 1:
        raise RuntimeError("small-stratum merge did not reach the minimum size")
    manifest = {
        "schema_version": 1,
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "min_stratum_size": int(min_stratum_size),
        "algorithm": (
            "lexicographic_query_id_then_fresh_default_rng_seed_per_merged_stratum; "
            "first_ceil(validation_fraction*n)_is_train_validation"
        ),
        "canonical_strata_order": list(CANONICAL_STRATA),
        "original_stratum_sizes": sizes,
        "stratum_merge_map": merge_map,
        "merged_stratum_members": merged_members,
        "merged_strata_canonical_order": merged_order,
        "merged_stratum_sizes": merged_sizes,
        "validation_sizes_by_merged_stratum": validation_sizes,
        "split_role_sizes": {
            TRAIN_ROLE: len(fit_ids),
            VALIDATION_ROLE: len(validation_ids),
        },
        "factor_distributions": {
            role: _factor_distribution(output.loc[output["split_role"].eq(role)])
            for role in (TRAIN_ROLE, VALIDATION_ROLE)
        },
        "train_fit_train_validation_intersection": overlap,
        "empty_intersection_verified": not overlap,
        "dev_query_count": 0,
    }
    return output, manifest


def _table_from_frame(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records")
    return pa.Table.from_pylist(rows, schema=schema)


def atomic_write_parquet(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    schema: pa.Schema,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary Parquet already exists: {temporary}")
    try:
        table = _table_from_frame(frame, schema)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
        )
        observed = pq.read_table(temporary)
        if not observed.schema.equals(schema, check_metadata=False):
            raise ValueError("temporary Parquet schema mismatch")
        if observed.num_rows != len(frame):
            raise ValueError("temporary Parquet row count mismatch")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _candidate_frame_for_schema(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[list(CANDIDATE_COLUMNS)].copy()
    output[["split", "query_id", "docid", "judgment"]] = output[
        ["split", "query_id", "docid", "judgment"]
    ].astype("string")
    output["bm25_rank"] = output["bm25_rank"].astype("int64")
    output["bm25_score"] = output["bm25_score"].astype("float64")
    output["relevance_grade"] = output["relevance_grade"].astype("Int64")
    output["relevance"] = output["relevance"].astype("Int64")
    output["is_judged"] = output["is_judged"].astype("bool")
    return output


def _assert_preselection_mutation_allowed(config: Mapping[str, Any]) -> None:
    selection = resolve_path(config, config["audits"]["checkpoint_selection"])
    ledger = resolve_path(config, config["audits"]["dev_access_ledger"])
    if selection.exists() or (ledger.is_file() and ledger.stat().st_size > 0):
        raise ValueError(
            "training-data mutation is forbidden after checkpoint selection or "
            "evaluation access"
        )


def build_training_split(
    config: Mapping[str, Any], *, overwrite: bool = False
) -> dict[str, Any]:
    validate_finetune_config(config)
    immutable = phase12_immutable_snapshot(config, require_all=False)
    candidate_path = resolve_path(config, config["inputs"]["train_candidates"])
    split_path = resolve_path(config, config["artifacts"]["query_split"])
    validation_path = resolve_path(config, config["artifacts"]["validation_groups"])
    manifest_path = resolve_path(config, config["audits"]["query_split_manifest"])
    for path in (candidate_path, split_path, validation_path, manifest_path):
        assert_not_dev(path)
    outputs = (split_path, validation_path, manifest_path)
    if all(path.is_file() for path in outputs) and not overwrite:
        manifest = read_json(manifest_path)
        if manifest.get("query_split_sha256") != sha256_file(split_path):
            raise ValueError("existing query split differs from its manifest")
        if manifest.get("validation_groups_sha256") != sha256_file(validation_path):
            raise ValueError("existing validation groups differ from their manifest")
        verify_phase12_immutable(config, immutable, require_all=False)
        return {"status": "PASS", "action": "reused", **manifest}
    _assert_preselection_mutation_allowed(config)
    if any(path.exists() for path in outputs) and not overwrite:
        raise ValueError("partial training split outputs exist; use --overwrite after review")

    candidates = read_training_parquet(candidate_path, label="train candidates")
    train = _filter_train(candidates, label="train candidates")
    split, manifest = build_query_split_frame(
        train,
        seed=int(config["split"]["seed"]),
        validation_fraction=float(config["split"]["validation_fraction"]),
        min_stratum_size=int(config["split"]["min_stratum_size"]),
    )
    validation_ids = set(
        split.loc[split["split_role"].eq(VALIDATION_ROLE), "query_id"].map(str)
    )
    validation_groups = train.loc[
        train["query_id"].astype("string").isin(validation_ids)
    ].copy()
    validation_groups = validation_groups.sort_values(
        ["query_id", "bm25_rank", "docid"], kind="mergesort"
    ).reset_index(drop=True)
    if set(validation_groups["query_id"].map(str)) != validation_ids:
        raise RuntimeError("validation candidate groups are incomplete")
    source_counts = train.groupby("query_id").size().to_dict()
    output_counts = validation_groups.groupby("query_id").size().to_dict()
    if any(output_counts.get(query_id) != source_counts.get(query_id) for query_id in validation_ids):
        raise RuntimeError("validation groups do not contain every candidate row")

    atomic_write_parquet(split_path, split, schema=QUERY_SPLIT_SCHEMA)
    atomic_write_parquet(
        validation_path,
        _candidate_frame_for_schema(validation_groups),
        schema=CANDIDATE_SCHEMA,
    )
    manifest.update(
        {
            "inputs": {
                portable_path(config, candidate_path): sha256_file(candidate_path),
                portable_path(
                    config, resolve_path(config, config["inputs"]["phase1_manifest"])
                ): (
                    sha256_file(resolve_path(config, config["inputs"]["phase1_manifest"]))
                    if resolve_path(config, config["inputs"]["phase1_manifest"]).is_file()
                    else None
                ),
            },
            "query_split_path": portable_path(config, split_path),
            "query_split_sha256": sha256_file(split_path),
            "query_split_rows": int(len(split)),
            "validation_groups_path": portable_path(config, validation_path),
            "validation_groups_sha256": sha256_file(validation_path),
            "validation_groups_rows": int(len(validation_groups)),
        }
    )
    atomic_write_json(manifest_path, manifest)
    verify_phase12_immutable(config, immutable, require_all=False)
    return {"status": "PASS", "action": "created", **manifest}


def _weak_bucket_id(rank: int, boundaries: Sequence[Sequence[int]]) -> str:
    for low, high in boundaries:
        if int(low) <= rank <= int(high):
            return f"{int(low)}-{int(high)}"
    raise ValueError(f"weak rank {rank} is outside configured buckets")


def sample_weak_negatives(
    query_id: str,
    candidates: pd.DataFrame,
    *,
    global_seed: int,
    bucket_boundaries: Sequence[Sequence[int]] = ((26, 50), (51, 75), (76, 100)),
    bucket_targets: Sequence[int] = (3, 3, 2),
    maximum: int = 8,
) -> list[dict[str, Any]]:
    """Sample one query's weak documents with frozen bucket fallback."""

    if len(bucket_boundaries) != 3 or len(bucket_targets) != 3:
        raise ValueError("weak sampling requires exactly three buckets")
    pools: list[deque[dict[str, Any]]] = []
    selected: list[dict[str, Any]] = []
    for bucket_index, ((low, high), target) in enumerate(
        zip(bucket_boundaries, bucket_targets, strict=True)
    ):
        bucket = candidates.loc[
            candidates["judgment"].astype("string").eq("unjudged")
            & candidates["bm25_rank"].between(int(low), int(high), inclusive="both")
        ].sort_values(["bm25_rank", "docid"], kind="mergesort")
        rows = [
            {
                "docid": str(row.docid),
                "bm25_rank": int(row.bm25_rank),
                "bucket_id": f"{int(low)}-{int(high)}",
                "bucket_index": bucket_index,
            }
            for row in bucket.itertuples(index=False)
        ]
        seed_material = f"{global_seed}|{query_id}|{bucket_index}".encode("utf-8")
        seed_int = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        order = np.random.default_rng(seed_int).permutation(len(rows))
        shuffled = [rows[int(index)] for index in order]
        take = min(int(target), len(shuffled), maximum - len(selected))
        selected.extend(shuffled[:take])
        pools.append(deque(shuffled[take:]))
    missing = maximum - len(selected)
    while missing > 0 and any(pools):
        progressed = False
        for pool in pools:
            if missing <= 0:
                break
            if pool:
                selected.append(pool.popleft())
                missing -= 1
                progressed = True
        if not progressed:
            break
    if len(selected) > maximum:
        raise RuntimeError("weak sampling exceeded its per-query maximum")
    if len({row["docid"] for row in selected}) != len(selected):
        raise RuntimeError("weak sampling produced duplicate documents")
    return selected


def _source_capacities(
    judged_available: int,
    weak_available: int,
    *,
    max_judged: int,
    max_weak: int,
    max_total: int,
) -> tuple[int, int]:
    judged_take = min(judged_available, max_judged)
    weak_take = min(weak_available, max_weak)
    free = max_total - judged_take - weak_take
    if free > 0:
        extra_judged = min(free, judged_available - judged_take)
        judged_take += extra_judged
        free -= extra_judged
        extra_weak = min(free, weak_available - weak_take)
        weak_take += extra_weak
    return judged_take, weak_take


def _round_robin_pairs(
    query_id: str,
    positives: Sequence[str],
    judged: Sequence[Mapping[str, Any]],
    weak: Sequence[Mapping[str, Any]],
    *,
    judged_take: int,
    weak_take: int,
    judged_weight: float,
    weak_weight: float,
) -> list[dict[str, Any]]:
    positive_order = sorted(str(docid) for docid in positives)
    judged_order = sorted(
        judged, key=lambda row: (int(row["bm25_rank"]), str(row["docid"]))
    )
    weak_order = sorted(
        weak, key=lambda row: (int(row["bm25_rank"]), str(row["docid"]))
    )
    queues = {
        "judged_non_relevant": {
            positive: deque(dict(row) for row in judged_order)
            for positive in positive_order
        },
        "weak_unjudged": {
            positive: deque(dict(row) for row in weak_order)
            for positive in positive_order
        },
    }
    limits = {
        "judged_non_relevant": int(judged_take),
        "weak_unjudged": int(weak_take),
    }
    selected_counts = Counter()
    selected_keys: set[tuple[str, str, str]] = set()
    selected: list[dict[str, Any]] = []

    def take_from(source: str, positive: str) -> bool:
        if selected_counts[source] >= limits[source]:
            return False
        queue = queues[source][positive]
        while queue:
            negative = queue.popleft()
            key = (query_id, positive, str(negative["docid"]))
            if key in selected_keys:
                continue
            if positive == str(negative["docid"]):
                raise ValueError("a document cannot be positive and negative")
            selected_keys.add(key)
            selected_counts[source] += 1
            selected.append(
                {
                    "query_id": query_id,
                    "positive_docid": positive,
                    "negative_docid": str(negative["docid"]),
                    "negative_source": source,
                    "negative_bm25_rank": int(negative["bm25_rank"]),
                    "pair_weight": np.float32(
                        judged_weight if source == "judged_non_relevant" else weak_weight
                    ),
                }
            )
            return True
        return False

    target = judged_take + weak_take
    while len(selected) < target:
        progress = False
        for positive in positive_order:
            if len(selected) >= target:
                break
    # Один общий циклический обход сохраняет приоритет экспертной разметки,
    # а слабые примеры заполняют оставшиеся места в том же круге.
            if take_from("judged_non_relevant", positive):
                progress = True
                continue
            if take_from("weak_unjudged", positive):
                progress = True
        if not progress:
            break
    if selected_counts["judged_non_relevant"] != judged_take:
        raise RuntimeError("global round-robin failed to fill judged capacity")
    if selected_counts["weak_unjudged"] != weak_take:
        raise RuntimeError("global round-robin failed to fill weak capacity")
    if len(selected) != len(selected_keys):
        raise RuntimeError("global round-robin produced duplicate pairs")
    if len(positive_order) <= target:
        allowed = bool(judged_order or weak_order)
        represented = {row["positive_docid"] for row in selected}
        if allowed and not set(positive_order).issubset(represented):
            raise RuntimeError("global positive coverage invariant failed")
    return selected


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(key): int(counts[key]) for key in sorted(counts)}


def _rank_histogram(frame: pd.DataFrame) -> dict[str, int]:
    labels = ("1-25", "26-50", "51-75", "76-100")
    counts = Counter()
    for rank in frame.get("negative_bm25_rank", pd.Series(dtype="int64")):
        value = int(rank)
        label = labels[0] if value <= 25 else labels[1] if value <= 50 else labels[2] if value <= 75 else labels[3]
        counts[label] += 1
    return {label: int(counts[label]) for label in labels}


def _weak_share_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for query_id, group in frame.groupby("query_id", sort=True):
        weights = group["pair_weight"].astype("float64")
        weak_weight = weights.loc[group["negative_source"].eq("weak_unjudged")].sum()
        share = float(weak_weight / weights.sum()) if float(weights.sum()) else 0.0
        values.append({"query_id": str(query_id), "weak_weight_share": share})
    numeric = np.asarray([row["weak_weight_share"] for row in values], dtype=np.float64)
    return {
        "per_query": values,
        "min": float(numeric.min()) if numeric.size else None,
        "median": float(np.median(numeric)) if numeric.size else None,
        "max": float(numeric.max()) if numeric.size else None,
        "queries_at_0": int((numeric == 0.0).sum()),
        "queries_at_1": int((numeric == 1.0).sum()),
    }


def build_pair_frame(
    candidates: pd.DataFrame,
    query_split: pd.DataFrame,
    *,
    regime: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if regime not in {"judged_only", "weak_negatives"}:
        raise ValueError("pair frame regime must be judged_only or weak_negatives")
    train = _filter_train(candidates, label="training candidates")
    validate_candidate_schema(train)
    fit_ids = set(
        query_split.loc[query_split["split_role"].eq(TRAIN_ROLE), "query_id"].map(str)
    )
    validation_ids = set(
        query_split.loc[
            query_split["split_role"].eq(VALIDATION_ROLE), "query_id"
        ].map(str)
    )
    if fit_ids & validation_ids:
        raise ValueError("train_fit and train_validation overlap")
    source = train.loc[train["query_id"].astype("string").isin(fit_ids)].copy()
    if not set(source["query_id"].map(str)).issubset(fit_ids):
        raise RuntimeError("pair source escaped train_fit")

    negatives = config["negatives"]
    seed = int(config["training"]["seed"])
    rows: list[dict[str, Any]] = []
    excluded = Counter()
    judged_before: dict[str, int] = {}
    weak_before: dict[str, int] = {}
    selected_weak_docs: dict[str, list[dict[str, Any]]] = {}
    usable_queries = 0
    for query_id, group in source.groupby("query_id", sort=True):
        query_id = str(query_id)
        positives = sorted(
            group.loc[group["judgment"].eq("relevant"), "docid"].map(str)
        )
        judged = [
            {"docid": str(row.docid), "bm25_rank": int(row.bm25_rank)}
            for row in group.loc[
                group["judgment"].eq("judged_non_relevant")
            ].itertuples(index=False)
        ]
        weak = (
            sample_weak_negatives(
                query_id,
                group,
                global_seed=seed,
                bucket_boundaries=negatives["bucket_boundaries"],
                bucket_targets=negatives["bucket_targets"],
                maximum=int(negatives["max_weak_negatives_per_query"]),
            )
            if regime == "weak_negatives"
            else []
        )
        selected_weak_docs[query_id] = weak
        if not positives:
            excluded["no_positive_in_candidates"] += 1
            continue
        if regime == "judged_only" and not judged:
            excluded["no_judged_negative_in_candidates"] += 1
            continue
        judged_available = len(positives) * len(judged)
        weak_available = len(positives) * len(weak)
        if regime == "weak_negatives" and judged_available + weak_available == 0:
            excluded["no_admissible_negative_in_candidates"] += 1
            continue
        judged_before[query_id] = judged_available
        weak_before[query_id] = weak_available
        judged_take, weak_take = _source_capacities(
            judged_available,
            weak_available,
            max_judged=int(negatives["max_judged_pairs_per_query"]),
            max_weak=(
                int(negatives["max_weak_pairs_per_query"])
                if regime == "weak_negatives"
                else 0
            ),
            max_total=int(negatives["max_pairs_per_query"]),
        )
        query_rows = _round_robin_pairs(
            query_id,
            positives,
            judged,
            weak,
            judged_take=judged_take,
            weak_take=weak_take,
            judged_weight=float(negatives["judged_negative_weight"]),
            weak_weight=float(negatives["weak_negative_weight"]),
        )
        rows.extend(query_rows)
        usable_queries += 1

    output = pd.DataFrame(rows, columns=[field.name for field in PAIR_SCHEMA])
    if output.empty:
        output = pd.DataFrame({field.name: pd.Series(dtype="object") for field in PAIR_SCHEMA})
    else:
        output = output.sort_values(
            ["query_id", "positive_docid", "negative_docid"], kind="mergesort"
        ).reset_index(drop=True)
    for column in ("query_id", "positive_docid", "negative_docid", "negative_source"):
        output[column] = output[column].astype("string")
    output["negative_bm25_rank"] = output["negative_bm25_rank"].astype("int32")
    output["pair_weight"] = output["pair_weight"].astype("float32")
    if output.duplicated(["query_id", "positive_docid", "negative_docid"]).any():
        raise RuntimeError("pair artifact contains duplicate keys")
    if output["positive_docid"].eq(output["negative_docid"]).any():
        raise RuntimeError("pair artifact contains identical positive/negative documents")
    if set(output["query_id"].map(str)) & validation_ids:
        raise RuntimeError("pair artifact leaked train_validation queries")
    allowed_sources = (
        {"judged_non_relevant"}
        if regime == "judged_only"
        else {"judged_non_relevant", "weak_unjudged"}
    )
    if not set(output["negative_source"].dropna().map(str)).issubset(allowed_sources):
        raise RuntimeError("pair artifact contains an invalid negative source")
    weak_rows = output.loc[output["negative_source"].eq("weak_unjudged")]
    if not weak_rows.empty and not weak_rows["negative_bm25_rank"].between(26, 100).all():
        raise RuntimeError("weak pair rank escaped [26,100]")
    expected_weights = output["negative_source"].map(
        {"judged_non_relevant": 1.0, "weak_unjudged": 0.5}
    ).astype("float32")
    if not np.array_equal(output["pair_weight"].to_numpy(), expected_weights.to_numpy()):
        raise RuntimeError("pair weights disagree with negative sources")
    before_totals = [judged_before[qid] + weak_before[qid] for qid in judged_before]
    after_counts = output.groupby("query_id").size()
    weak_bucket_counts = Counter(
        _weak_bucket_id(
            int(rank), config["negatives"]["bucket_boundaries"]
        )
        for rank in weak_rows["negative_bm25_rank"]
    )
    sampled_weak_document_bucket_counts = Counter(
        row["bucket_id"]
        for query_rows in selected_weak_docs.values()
        for row in query_rows
    )
    section = {
        "regime_id": regime,
        "pair_count": int(len(output)),
        "usable_query_count": int(usable_queries),
        "excluded_query_counts": dict(sorted(excluded.items())),
        "judged_pairs_before_cap": int(sum(judged_before.values())),
        "judged_pairs_after_cap": int(
            output["negative_source"].eq("judged_non_relevant").sum()
        ),
        "weak_pairs_before_cap": int(sum(weak_before.values())),
        "weak_pairs_after_cap": int(output["negative_source"].eq("weak_unjudged").sum()),
        "queries_with_weak_pairs_before_cap": int(
            sum(count > 0 for count in weak_before.values())
        ),
        "queries_with_weak_pairs_after_cap": int(
            weak_rows["query_id"].nunique()
        ),
        "negative_source_distribution": {
            source_name: int(output["negative_source"].eq(source_name).sum())
            for source_name in ("judged_non_relevant", "weak_unjudged")
        },
        "negative_bm25_rank_histogram": _rank_histogram(output),
        "weak_bucket_distribution": {
            label: int(weak_bucket_counts[label])
            for label in ("26-50", "51-75", "76-100")
        },
        "sampled_weak_document_bucket_distribution_before_pairing": {
            label: int(sampled_weak_document_bucket_counts[label])
            for label in ("26-50", "51-75", "76-100")
        },
        "pairs_per_query_before_cap": _histogram(before_totals),
        "pairs_per_query_after_cap": _histogram(after_counts.tolist()),
        "weak_weight_share_per_query": _weak_share_distribution(output),
        "label_audit": {
            state: int(source["judgment"].eq(state).sum())
            for state in ("relevant", "judged_non_relevant", "unjudged")
        },
        "leakage_audit": {
            "train_validation_intersection": [],
            "dev_query_count": 0,
            "passed": True,
        },
        "population_disclosure": (
            "judged_only и weak_negatives охватывают разные множества запросов; "
            "режим со слабыми примерами включает положительные запросы без "
            "экспертно оценённых отрицательных документов"
        ),
        "weight_disclosure": (
            "вес 0,5 уменьшает влияние отдельной слабой пары, но не фиксирует их "
            "суммарную долю: для отдельного запроса она может составлять от 0% до 100%"
        ),
        "heuristic_disclosure": (
            "ранги 26–100, целевые числа 3/3/2, вес 0,5 и ограничения 8/8/16 — "
            "заранее заданные консервативные эвристики, а не оптимумы по dev"
        ),
    }
    return output, section


def materialize_control_pairs(
    judged_pairs: pd.DataFrame,
    query_split: pd.DataFrame,
    *,
    target_pairs: int = 2000,
    control_seed: int = 20260805,
    shuffle_seed: int = 20260804,
    merged_strata_order: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target_pairs <= 0:
        raise ValueError("control target_pairs must be positive")
    pair_groups = {
        str(query_id): group.sort_values(
            ["positive_docid", "negative_docid"], kind="mergesort"
        ).copy()
        for query_id, group in judged_pairs.groupby("query_id", sort=True)
    }
    pair_query_ids = set(pair_groups)
    fit_query_ids = set(
        query_split.loc[query_split["split_role"].eq(TRAIN_ROLE), "query_id"].map(str)
    )
    validation_query_ids = set(
        query_split.loc[
            query_split["split_role"].eq(VALIDATION_ROLE), "query_id"
        ].map(str)
    )
    validation_overlap = sorted(pair_query_ids & validation_query_ids)
    if validation_overlap:
        raise ValueError(
            "control input contains train_validation queries: "
            f"{validation_overlap[:10]}"
        )
    outside_fit = sorted(pair_query_ids.difference(fit_query_ids))
    if outside_fit:
        raise ValueError(f"control input contains non-train_fit queries: {outside_fit[:10]}")
    split_lookup = query_split.set_index("query_id")["merged_stratum"].map(str).to_dict()
    unknown = sorted(set(pair_groups).difference(split_lookup))
    if unknown:
        raise ValueError(f"control pairs contain unknown query ids: {unknown[:10]}")
    available: dict[str, list[str]] = defaultdict(list)
    for query_id in sorted(pair_groups):
        available[str(split_lookup[query_id])].append(query_id)
    if merged_strata_order is None:
        merged_strata_order = list(dict.fromkeys(query_split["merged_stratum"].map(str)))
    canonical = [stratum for stratum in merged_strata_order if stratum in available]
    for stratum in sorted(set(available).difference(canonical)):
        canonical.append(stratum)
    queues: dict[str, deque[str]] = {}
    stats: dict[str, dict[str, Any]] = {}
    total_queries = sum(len(query_ids) for query_ids in available.values())
    for stratum in canonical:
        query_ids = sorted(available[stratum])
        seed_material = f"{control_seed}|{stratum}".encode("utf-8")
        seed_int = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        order = np.random.default_rng(seed_int).permutation(len(query_ids))
        queues[stratum] = deque(query_ids[int(index)] for index in order)
        stats[stratum] = {
            "available_query_count": len(query_ids),
            "selected_query_count": 0,
            "available_pair_count": int(sum(len(pair_groups[qid]) for qid in query_ids)),
            "selected_pair_count": 0,
            "target_share": float(len(query_ids) / total_queries),
            "realized_query_share": 0.0,
        }
    selected_queries: list[tuple[str, str]] = []
    selected_pair_count = 0
    t = 0
    while selected_pair_count < target_pairs and any(queues.values()):
        candidates_for_step = [stratum for stratum in canonical if queues[stratum]]
        if not candidates_for_step:
            break
        deficits = {
            stratum: (t + 1) * float(stats[stratum]["target_share"])
            - int(stats[stratum]["selected_query_count"])
            for stratum in candidates_for_step
        }
        best = max(
            candidates_for_step,
            key=lambda stratum: (deficits[stratum], -canonical.index(stratum)),
        )
        query_id = queues[best].popleft()
        pair_count = len(pair_groups[query_id])
        selected_queries.append((best, query_id))
        selected_pair_count += pair_count
        stats[best]["selected_query_count"] += 1
        stats[best]["selected_pair_count"] += pair_count
        t += 1
    for stratum in canonical:
        stats[stratum]["realized_query_share"] = (
            float(stats[stratum]["selected_query_count"] / t) if t else 0.0
        )
    materialized: list[pd.DataFrame] = []
    for stratum, query_id in selected_queries:
        group = pair_groups[query_id].copy()
        group["merged_stratum"] = stratum
        materialized.append(group)
    if not materialized:
        raise ValueError("control selection produced no pairs")
    output = pd.concat(materialized, ignore_index=True).sort_values(
        ["merged_stratum", "query_id", "positive_docid", "negative_docid"],
        kind="mergesort",
    ).reset_index(drop=True)
    rng = np.random.default_rng(shuffle_seed)
    swap = rng.random(len(output)) < 0.5
    positive = output.loc[swap, "positive_docid"].copy()
    output.loc[swap, "positive_docid"] = output.loc[swap, "negative_docid"].to_numpy()
    output.loc[swap, "negative_docid"] = positive.to_numpy()
    output = output.drop(columns="merged_stratum")
    output = output.sort_values(
        ["query_id", "positive_docid", "negative_docid"], kind="mergesort"
    ).reset_index(drop=True)
    if output.duplicated(["query_id", "positive_docid", "negative_docid"]).any():
        raise RuntimeError("control role swapping produced duplicate keys")
    if output["positive_docid"].eq(output["negative_docid"]).any():
        raise RuntimeError("control role swapping produced an identical pair")
    section = {
        "regime_id": "control_c1",
        "target_pairs": int(target_pairs),
        "pair_count": int(len(output)),
        "usable_query_count": int(t),
        "excluded_query_counts": {},
        "judged_pairs_before_cap": int(len(judged_pairs)),
        "judged_pairs_after_cap": int(len(output)),
        "weak_pairs_before_cap": 0,
        "weak_pairs_after_cap": 0,
        "queries_with_weak_pairs_before_cap": 0,
        "queries_with_weak_pairs_after_cap": 0,
        "control_seed": int(control_seed),
        "shuffle_seed": int(shuffle_seed),
        "merged_strata_canonical_order": canonical,
        "strata": stats,
        "selected_query_sequence": [
            {"merged_stratum": stratum, "query_id": query_id}
            for stratum, query_id in selected_queries
        ],
        "swapped_pair_count": int(swap.sum()),
        "negative_source_distribution": {
            source: int(output["negative_source"].eq(source).sum())
            for source in ("judged_non_relevant", "weak_unjudged")
        },
        "negative_bm25_rank_histogram": _rank_histogram(output),
        "weak_bucket_distribution": {
            label: 0 for label in ("26-50", "51-75", "76-100")
        },
        "pairs_per_query_before_cap": _histogram(
            len(group) for group in pair_groups.values()
        ),
        "pairs_per_query_after_cap": _histogram(output.groupby("query_id").size()),
        "weak_weight_share_per_query": _weak_share_distribution(output),
        "label_audit": {
            "relevant": int(len(judged_pairs)),
            "judged_non_relevant": int(len(judged_pairs)),
            "unjudged": 0,
        },
        "leakage_audit": {
            "train_validation_intersection": validation_overlap,
            "dev_query_count": 0,
            "passed": True,
        },
        "population_disclosure": (
            "judged_only и weak_negatives охватывают разные множества запросов; "
            "режим со слабыми примерами включает положительные запросы без "
            "экспертно оценённых отрицательных документов"
        ),
        "weight_disclosure": (
            "вес 0,5 уменьшает влияние отдельной слабой пары, но не фиксирует их "
            "суммарную долю: для отдельного запроса она может составлять от 0% до 100%"
        ),
        "heuristic_disclosure": (
            "ранги 26–100, целевые числа 3/3/2, вес 0,5 и ограничения 8/8/16 — "
            "заранее заданные консервативные эвристики, а не оптимумы по dev"
        ),
    }
    return output, section


def _pair_section_hash(section: Mapping[str, Any]) -> str:
    payload = dict(section)
    payload.pop("pair_manifest_section_sha256", None)
    return canonical_json_sha256(payload)


def _validate_existing_pair(
    path: Path, manifest_path: Path, regime: str
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    section = manifest.get("regimes", {}).get(regime)
    if not isinstance(section, dict):
        raise ValueError(f"pairs manifest has no {regime} section")
    if section.get("pair_file_sha256") != sha256_file(path):
        raise ValueError(f"existing {regime} pair file differs from its manifest")
    if section.get("pair_manifest_section_sha256") != _pair_section_hash(section):
        raise ValueError(f"existing {regime} manifest section hash is invalid")
    table = pq.read_table(path)
    if not table.schema.equals(PAIR_SCHEMA, check_metadata=False):
        raise ValueError(f"existing {regime} pair schema is invalid")
    if table.num_rows != int(section["pair_count"]):
        raise ValueError(f"existing {regime} pair row count is invalid")
    return section


def build_training_pairs(
    config: Mapping[str, Any], *, regime: str, overwrite: bool = False
) -> dict[str, Any]:
    validate_finetune_config(config)
    immutable = phase12_immutable_snapshot(config, require_all=False)
    if regime not in PAIR_REGIMES:
        raise ValueError(f"regime must be one of {list(PAIR_REGIMES)}")
    artifact_key = {
        "judged_only": "pairs_judged_only",
        "weak_negatives": "pairs_weak_negatives",
        "control_c1": "pairs_control_c1",
    }[regime]
    pair_path = resolve_path(config, config["artifacts"][artifact_key])
    manifest_path = resolve_path(config, config["audits"]["pairs_manifest"])
    split_path = resolve_path(config, config["artifacts"]["query_split"])
    split_manifest_path = resolve_path(config, config["audits"]["query_split_manifest"])
    for path in (pair_path, manifest_path, split_path, split_manifest_path):
        assert_not_dev(path)
    if pair_path.is_file() and manifest_path.is_file() and not overwrite:
        section = _validate_existing_pair(pair_path, manifest_path, regime)
        verify_phase12_immutable(config, immutable, require_all=False)
        return {"status": "PASS", "action": "reused", **section}
    _assert_preselection_mutation_allowed(config)
    if pair_path.exists() and not overwrite:
        raise ValueError(f"pair output exists without reusable manifest: {pair_path}")
    if not split_path.is_file() or not split_manifest_path.is_file():
        raise ValueError("build-training-split must complete before pair materialization")
    split = read_training_parquet(split_path, label="query split")
    split_manifest = read_json(split_manifest_path)

    if regime == "control_c1":
        judged_path = resolve_path(config, config["artifacts"]["pairs_judged_only"])
        assert_not_dev(judged_path)
        if not judged_path.is_file():
            raise ValueError("judged_only pairs must exist before control_c1")
        judged = read_training_parquet(judged_path, label="judged-only pairs")
        output, section = materialize_control_pairs(
            judged,
            split,
            target_pairs=int(config["runs"]["C1"]["target_pairs"]),
            control_seed=int(config["runs"]["C1"]["control_seed"]),
            shuffle_seed=int(config["runs"]["C1"]["shuffle_seed"]),
            merged_strata_order=split_manifest["merged_strata_canonical_order"],
        )
        input_hashes = {
            portable_path(config, judged_path): sha256_file(judged_path),
            portable_path(config, split_path): sha256_file(split_path),
        }
    else:
        candidate_path = resolve_path(config, config["inputs"]["train_candidates"])
        assert_not_dev(candidate_path)
        candidates = read_training_parquet(candidate_path, label="train candidates")
        output, section = build_pair_frame(
            candidates, split, regime=regime, config=config
        )
        input_hashes = {
            portable_path(config, candidate_path): sha256_file(candidate_path),
            portable_path(config, split_path): sha256_file(split_path),
        }
    atomic_write_parquet(pair_path, output, schema=PAIR_SCHEMA)
    section.update(
        {
            "pair_path": portable_path(config, pair_path),
            "pair_file_sha256": sha256_file(pair_path),
            "input_hashes": input_hashes,
        }
    )
    section["pair_manifest_section_sha256"] = _pair_section_hash(section)
    manifest = (
        read_json(manifest_path) if manifest_path.is_file() else {"schema_version": 1, "regimes": {}}
    )
    regimes = manifest.setdefault("regimes", {})
    if not isinstance(regimes, dict):
        raise ValueError("pairs manifest regimes must be an object")
    regimes[regime] = section
    manifest["regime_order"] = [name for name in PAIR_REGIMES if name in regimes]
    atomic_write_json(manifest_path, manifest)
    _validate_existing_pair(pair_path, manifest_path, regime)
    verify_phase12_immutable(config, immutable, require_all=False)
    return {"status": "PASS", "action": "created", **section}


def source_set_sha256(
    root: str | Path, relatives: Sequence[str]
) -> tuple[str, dict[str, str]]:
    repository = Path(root).resolve()
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    for relative in sorted(str(value) for value in relatives):
        path = repository / relative
        if not path.is_file():
            raise ValueError(f"source file is missing: {relative}")
        file_hash = sha256_file(path)
        hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), hashes


def manifest_declared_hashes(path: str | Path) -> dict[str, str]:
    """Extract path and named-input SHA bindings from Phase 1/2 manifests."""

    payload = read_json(path, training_path=False)
    found: dict[str, str] = {}

    def remember(identifier: str, digest: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            return
        previous = found.get(identifier)
        if previous is not None and previous != digest:
            raise ValueError(
                f"manifest declares conflicting SHA-256 values for {identifier}"
            )
        found[identifier] = digest

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            candidate_path = value.get("path")
            candidate_hash = value.get("sha256")
            if isinstance(candidate_path, str) and isinstance(candidate_hash, str):
                remember(candidate_path, candidate_hash)
            input_hashes = value.get("input_hashes")
            if isinstance(input_hashes, Mapping):
                for input_name, input_hash in input_hashes.items():
                    if isinstance(input_name, str) and isinstance(input_hash, str):
                        remember(input_name, input_hash)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(payload)
    return found


def phase12_immutable_snapshot(
    config: Mapping[str, Any], *, require_all: bool = True
) -> dict[str, str]:
    """Bind Phase 1/2 inputs by opaque bytes, never by dev semantics.

    Hashing restored payload bytes is explicitly part of the archive contract
    and does not parse evaluation annotations or scores.  The dev-qrels hash is
    late-bound to its Phase 1/2 declaration until the guarded post-selection
    command materializes that file.
    """

    direct_inputs = {
        key: str(config["inputs"][key])
        for key in (
            "train_candidates",
            "queries",
            "passages",
            "train_qrels",
            "phase1_manifest",
            "phase2_manifest",
        )
    }
    direct_relatives = list(direct_inputs.values())
    snapshot: dict[str, str] = {}
    missing: list[str] = []
    for relative in dict.fromkeys(direct_relatives):
        path = resolve_path(config, relative)
        if not path.is_file():
            missing.append(relative)
            continue
        snapshot[relative] = sha256_file(path)
    declared: dict[str, str] = {}
    for manifest_key in ("phase1_manifest", "phase2_manifest"):
        manifest_path = resolve_path(config, config["inputs"][manifest_key])
        if not manifest_path.is_file():
            continue
        entries = manifest_declared_hashes(manifest_path)
        for identifier, digest in entries.items():
            previous = declared.get(identifier)
            if previous is not None and previous != digest:
                raise ValueError(
                    "Phase 1/2 manifests disagree on immutable SHA-256 for "
                    f"{identifier}"
                )
            declared[identifier] = digest
        for input_name, relative in direct_inputs.items():
            expected = entries.get(relative, entries.get(input_name))
            actual_path = resolve_path(config, relative)
            # Обучающие файлы можно сверить побайтно сразу; изолированные данные
            # оценивания остаются закрытыми до защищённой стадии.
            if (
                expected is not None
                and actual_path.is_file()
                and sha256_file(actual_path) != expected
            ):
                raise ValueError(
                    f"immutable artifact differs from {manifest_key}: {relative}"
                )
    for input_name, value in config["dev_inputs"].items():
        relative = str(value)
        path = resolve_path(config, relative)
        expected = declared.get(relative, declared.get(str(input_name)))
        if expected is None:
            missing.append(f"{relative} (SHA not declared by Phase 1/2 manifests)")
            continue
        late_bound_qrels = relative == str(config["dev_inputs"]["dev_qrels"])
        if not path.is_file() and not late_bound_qrels:
            missing.append(relative)
            continue
        if path.is_file():
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"immutable evaluation payload differs from Phase 1/2 "
                    f"manifests: {relative}"
                )
            snapshot[relative] = actual
        else:
            snapshot[relative] = expected
    if missing and require_all:
        raise ValueError(f"Phase 1/2 immutable inputs are missing or unbound: {missing}")
    return dict(sorted(snapshot.items()))


def verify_phase12_immutable(
    config: Mapping[str, Any],
    expected: Mapping[str, str],
    *,
    require_all: bool = True,
) -> None:
    observed = phase12_immutable_snapshot(config, require_all=require_all)
    if dict(expected) != observed:
        changed = sorted(
            relative
            for relative in set(expected) | set(observed)
            if expected.get(relative) != observed.get(relative)
        )
        raise RuntimeError(f"Phase 1/2 immutable artifacts changed: {changed}")


__all__ = [
    "CANDIDATE_SCHEMA",
    "CANONICAL_STRATA",
    "DEFAULT_FINETUNE_CONFIG",
    "PAIR_REGIMES",
    "PAIR_SCHEMA",
    "QUERY_SPLIT_SCHEMA",
    "TRAIN_ROLE",
    "VALIDATION_ROLE",
    "assert_not_dev",
    "atomic_write_json",
    "atomic_write_parquet",
    "build_pair_frame",
    "build_query_split_frame",
    "build_training_pairs",
    "build_training_split",
    "canonical_json_sha256",
    "load_finetune_config",
    "manifest_declared_hashes",
    "materialize_control_pairs",
    "merge_small_strata",
    "phase12_immutable_snapshot",
    "portable_path",
    "read_json",
    "read_training_parquet",
    "repository_root",
    "resolve_path",
    "sample_weak_negatives",
    "sha256_file",
    "source_set_sha256",
    "validate_finetune_config",
    "verify_phase12_immutable",
]
