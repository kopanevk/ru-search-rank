"""CUDA-only supervised fine-tuning and atomic checkpoint lifecycle for Phase 3."""

from __future__ import annotations

import os

# The deterministic CUDA workspace contract must be set before torch import.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import contextlib
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import random
import re
import resource
import shutil
import subprocess
import tempfile
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .data import load_qrels, validate_passages, validate_queries
from .evaluation import (
    evaluate_ranked_ndcg_at_10,
    paired_bootstrap,
    parse_trec_eval_metric,
    rank_candidates_by_score,
    sparse_judgment_diagnostics,
)
from .pair_encoding import build_document, encode_pair
from .training_data import (
    PAIR_REGIMES,
    TRAIN_ROLE,
    VALIDATION_ROLE,
    assert_not_dev,
    atomic_write_json,
    canonical_json_sha256,
    load_finetune_config,
    phase12_immutable_snapshot,
    portable_path,
    read_json,
    read_training_parquet,
    repository_root,
    resolve_path,
    sha256_file,
    source_set_sha256,
    validate_finetune_config,
    verify_phase12_immutable,
)


TRAINING_SOURCE_FILES = (
    "src/rusearchrank/training.py",
    "src/rusearchrank/training_data.py",
    "src/rusearchrank/pair_encoding.py",
)
SCORING_SOURCE_FILES = (
    "src/rusearchrank/rerank.py",
    "src/rusearchrank/pair_encoding.py",
)
EVALUATION_SOURCE_FILES = (
    "src/rusearchrank/evaluation.py",
    "src/rusearchrank/phase3_eval.py",
)
PINNED_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
PINNED_MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
TOKENIZER_PAYLOAD_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)
MODEL_GENERATION_FILES = frozenset(
    ("model.safetensors", "config.json", *TOKENIZER_PAYLOAD_FILES)
)
RESUME_STATE_FILES = frozenset(
    ("optimizer.pt", "scheduler.pt", "rng_state.pt", "resume_sidecar.json")
)
TRAINING_FINGERPRINT_FIELDS = frozenset(
    {
        "training_source_sha256",
        "training_config_sha256",
        "split_manifest_sha256",
        "pair_file_sha256",
        "pair_manifest_section_sha256",
        "validation_groups_sha256",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "run_id",
        "regime",
        "learning_rate",
        "epochs",
        "implementation_version",
        "python_version",
        "torch_version",
        "transformers_version",
        "tokenizers_version",
    }
)


class StageError(RuntimeError):
    """A named, non-recovering Phase 3 stage failure."""

    def __init__(self, stage: str, root_cause: str, **diagnostics: Any) -> None:
        self.stage = stage
        self.root_cause = root_cause
        self.diagnostics = diagnostics
        super().__init__(
            json.dumps(
                {"stage": stage, "root_cause": root_cause, **diagnostics},
                ensure_ascii=False,
                indent=2,
            )
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "UNAVAILABLE"


def git_provenance(root: str | Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE", False


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def available_memory_bytes() -> int | None:
    if platform.system() == "Linux":
        path = Path("/proc/meminfo")
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError):
            return None
    return None


def pairwise_logistic_loss(
    positive_scores: torch.Tensor, negative_scores: torch.Tensor
) -> torch.Tensor:
    """Numerically stable pairwise logistic loss."""

    return F.softplus(-(positive_scores - negative_scores))


def weighted_query_loss(
    pair_losses: torch.Tensor, pair_weights: torch.Tensor
) -> torch.Tensor:
    """Compute sum(weight*loss)/sum(weight), never mean(weight*loss)."""

    if pair_losses.shape != pair_weights.shape:
        raise ValueError("pair losses and weights must have identical shapes")
    if pair_losses.numel() == 0:
        raise ValueError("query loss requires at least one pair")
    if not bool(torch.isfinite(pair_losses).all()) or not bool(
        torch.isfinite(pair_weights).all()
    ):
        raise StageError("query-loss", "pair loss or weight is non-finite")
    if bool((pair_weights < 0).any()):
        raise ValueError("pair weights must be non-negative")
    return (pair_weights * pair_losses).sum() / pair_weights.sum().clamp_min(1e-12)


def query_pairwise_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    pair_weights: torch.Tensor,
) -> torch.Tensor:
    return weighted_query_loss(
        pairwise_logistic_loss(positive_scores, negative_scores), pair_weights
    )


def epoch_query_order(
    query_ids: Sequence[str], *, seed: int, epoch: int
) -> list[str]:
    """Pure epoch permutation: only the sorted IDs, seed, and epoch matter."""

    ordered = sorted(str(query_id) for query_id in query_ids)
    permutation = np.random.default_rng(seed + epoch).permutation(len(ordered))
    return [ordered[int(index)] for index in permutation]


def accumulation_windows(
    query_order: Sequence[str], accumulation: int
) -> list[tuple[str, ...]]:
    if accumulation <= 0:
        raise ValueError("gradient accumulation must be positive")
    return [
        tuple(query_order[offset : offset + accumulation])
        for offset in range(0, len(query_order), accumulation)
    ]


def run_accumulation_epoch(
    query_order: Sequence[str],
    *,
    loss_for_query: Callable[[str], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    parameters: Iterable[torch.nn.Parameter],
    accumulation: int = 16,
    max_grad_norm: float = 1.0,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Apply the exact actual-window mean gradient accumulation contract."""

    windows = accumulation_windows(query_order, accumulation)
    seen: list[str] = []
    logs: list[dict[str, Any]] = []
    parameter_list = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    for step_index, window in enumerate(windows, start=1):
        started = time.perf_counter()
        detached_losses: list[float] = []
        for query_id in window:
            query_loss = loss_for_query(query_id)
            if query_loss.ndim != 0:
                raise ValueError("per-query loss must be a scalar")
            if not bool(torch.isfinite(query_loss)):
                raise StageError(
                    "finetune/non-finite-loss",
                    "non-finite query loss",
                    query_id=query_id,
                    optimizer_step=step_index,
                )
            (query_loss / len(window)).backward()
            detached_losses.append(float(query_loss.detach().cpu()))
            seen.append(query_id)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameter_list, max_norm=max_grad_norm
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise StageError(
                "finetune/non-finite-gradient",
                "non-finite gradient norm before optimizer step",
                optimizer_step=step_index,
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        elapsed = max(time.perf_counter() - started, 1e-12)
        log = {
            "optimizer_step": step_index,
            "optimizer_step_loss": float(np.mean(detached_losses)),
            "window_query_count": len(window),
            "seconds": elapsed,
            "queries_per_second": len(window) / elapsed,
        }
        logs.append(log)
        if on_step is not None:
            on_step(log)
    if seen != list(query_order):
        raise RuntimeError("gradient accumulation lost or duplicated query groups")
    return {
        "optimizer_steps": len(windows),
        "last_window_query_count": len(windows[-1]) if windows else 0,
        "seen_query_ids": seen,
        "logs": logs,
    }


def build_adamw_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float = 0.01,
    no_decay_patterns: Sequence[str] = ("bias", "LayerNorm.weight"),
) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(pattern in name for pattern in no_decay_patterns):
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(learning_rate),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    optimizer._rusearchrank_parameter_names = {  # type: ignore[attr-defined]
        "decay": decay_names,
        "no_decay": no_decay_names,
    }
    return optimizer


@dataclass(frozen=True)
class TokenCache:
    ids_buffer: np.ndarray
    offsets: np.ndarray
    lookup: dict[tuple[str, str], int]
    tokens_before: np.ndarray
    tokens_after: np.ndarray
    truncated: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(
            self.ids_buffer.nbytes
            + self.offsets.nbytes
            + self.tokens_before.nbytes
            + self.tokens_after.nbytes
            + self.truncated.nbytes
        )

    def encoded(self, query_id: str, docid: str) -> dict[str, list[int]]:
        index = self.lookup[(str(query_id), str(docid))]
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        ids = self.ids_buffer[start:end].astype(np.int64, copy=False).tolist()
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def estimate_token_cache_bytes(sequence_count: int, max_length: int) -> int:
    if sequence_count < 0 or max_length <= 0:
        raise ValueError("token-cache dimensions are invalid")
    return int(sequence_count * max_length * 4 + (sequence_count + 1) * 8)


def ensure_token_cache_ram(
    expected_bytes: int,
    *,
    minimum_free_ram_gib: float,
    available_bytes: int | None = None,
) -> dict[str, int]:
    observed = available_memory_bytes() if available_bytes is None else int(available_bytes)
    if observed is None:
        raise StageError("token-cache/preflight", "available RAM could not be measured")
    reserve = int(float(minimum_free_ram_gib) * 1024**3)
    required = reserve + int(expected_bytes)
    if observed < required:
        raise StageError(
            "token-cache/preflight",
            "insufficient RAM for the mandatory pretokenized cache",
            available_bytes=observed,
            expected_cache_bytes=int(expected_bytes),
            minimum_free_ram_bytes=reserve,
            fallback="forbidden",
        )
    return {"available_bytes": observed, "required_bytes": required}


def build_token_cache(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    passages: pd.DataFrame,
    tokenizer: Any,
    *,
    max_length: int = 320,
    minimum_free_ram_gib: float = 4,
    available_bytes: int | None = None,
) -> tuple[TokenCache, dict[str, Any]]:
    required_pair_columns = {"query_id", "positive_docid", "negative_docid"}
    missing = sorted(required_pair_columns.difference(pairs.columns))
    if missing:
        raise ValueError(f"pairs are missing columns: {missing}")
    if "split" in queries.columns:
        if queries["split"].astype("string").eq("dev").any():
            raise ValueError("token-cache queries contain isolated evaluation rows")
        queries = queries.loc[queries["split"].astype("string").eq("train")].copy()
        if not queries["split"].astype("string").eq("train").all():
            raise RuntimeError("query train filter failed")
    validate_queries(queries.assign(split="train") if "split" not in queries else queries)
    validate_passages(passages)
    keys = sorted(
        {
            (str(row.query_id), str(docid))
            for row in pairs.itertuples(index=False)
            for docid in (row.positive_docid, row.negative_docid)
        }
    )
    expected = estimate_token_cache_bytes(len(keys), max_length)
    preflight = ensure_token_cache_ram(
        expected,
        minimum_free_ram_gib=minimum_free_ram_gib,
        available_bytes=available_bytes,
    )
    query_lookup = {
        str(row.query_id): str(row.query_text) for row in queries.itertuples(index=False)
    }
    passage_lookup = {
        str(row.docid): (row.title, str(row.text))
        for row in passages.itertuples(index=False)
    }
    unknown_queries = sorted({qid for qid, _ in keys}.difference(query_lookup))
    unknown_docs = sorted({docid for _, docid in keys}.difference(passage_lookup))
    if unknown_queries or unknown_docs:
        raise ValueError(
            f"token cache keys are missing text: queries={unknown_queries[:10]}, "
            f"documents={unknown_docs[:10]}"
        )
    ids_parts: list[np.ndarray] = []
    offsets = np.zeros(len(keys) + 1, dtype=np.int64)
    before = np.zeros(len(keys), dtype=np.int32)
    after = np.zeros(len(keys), dtype=np.int32)
    truncated = np.zeros(len(keys), dtype=bool)
    lookup: dict[tuple[str, str], int] = {}
    cursor = 0
    for index, (query_id, docid) in enumerate(keys):
        title, text = passage_lookup[docid]
        encoded = encode_pair(
            tokenizer,
            query_lookup[query_id],
            build_document(title, text),
            max_length=max_length,
        )
        values = np.asarray(encoded["input_ids"], dtype=np.int32)
        ids_parts.append(values)
        lookup[(query_id, docid)] = index
        cursor += len(values)
        offsets[index + 1] = cursor
        before[index] = int(encoded["tokens_before"])
        after[index] = int(encoded["tokens_after"])
        truncated[index] = bool(encoded["truncated"])
    ids_buffer = (
        np.concatenate(ids_parts).astype(np.int32, copy=False)
        if ids_parts
        else np.asarray([], dtype=np.int32)
    )
    cache = TokenCache(ids_buffer, offsets, lookup, before, after, truncated)
    return cache, {
        **preflight,
        "sequence_count": len(keys),
        "estimated_bytes_before_allocation": expected,
        "token_cache_bytes": cache.nbytes,
        "token_cache_representation": "pretokenized_flat_int32",
        "tokens_before_sum": int(before.sum()),
        "tokens_after_sum": int(after.sum()),
        "truncated_sequence_count": int(truncated.sum()),
    }


def score_query_group_once(
    model: torch.nn.Module,
    tokenizer: Any,
    cache: TokenCache,
    group: pd.DataFrame,
    *,
    device: str,
    max_sequences_per_microbatch: int = 40,
) -> tuple[torch.Tensor, dict[str, int]]:
    query_ids = set(group["query_id"].map(str))
    if len(query_ids) != 1:
        raise ValueError("one micro-batch must contain exactly one query group")
    query_id = next(iter(query_ids))
    docids = sorted(
        set(group["positive_docid"].map(str)) | set(group["negative_docid"].map(str))
    )
    if len(docids) > max_sequences_per_microbatch:
        raise StageError(
            "finetune/microbatch",
            "max_sequences_per_microbatch exceeded; splitting is forbidden",
            query_id=query_id,
            observed_sequences=len(docids),
            maximum=max_sequences_per_microbatch,
        )
    encoded = [cache.encoded(query_id, docid) for docid in docids]
    batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
    tensors = {name: value.to(device) for name, value in batch.items()}
    logits = model(**tensors).logits
    if list(logits.shape) != [len(docids), 1]:
        raise StageError(
            "finetune/forward",
            "model returned an unexpected logits shape",
            observed=list(logits.shape),
            expected=[len(docids), 1],
        )
    scores = logits.reshape(-1)
    if not bool(torch.isfinite(scores).all()):
        raise StageError("finetune/forward", "model returned non-finite logits")
    return scores, {docid: index for index, docid in enumerate(docids)}


def model_query_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    cache: TokenCache,
    group: pd.DataFrame,
    *,
    device: str,
    max_sequences_per_microbatch: int,
) -> torch.Tensor:
    scores, indices = score_query_group_once(
        model,
        tokenizer,
        cache,
        group,
        device=device,
        max_sequences_per_microbatch=max_sequences_per_microbatch,
    )
    positive_indices = torch.tensor(
        [indices[str(value)] for value in group["positive_docid"]],
        dtype=torch.long,
        device=scores.device,
    )
    negative_indices = torch.tensor(
        [indices[str(value)] for value in group["negative_docid"]],
        dtype=torch.long,
        device=scores.device,
    )
    weights = torch.tensor(
        group["pair_weight"].astype("float32").to_numpy(),
        dtype=torch.float32,
        device=scores.device,
    )
    return query_pairwise_loss(
        scores.index_select(0, positive_indices),
        scores.index_select(0, negative_indices),
        weights,
    )


def _subset(config: Mapping[str, Any], section: str, keys: Sequence[str]) -> dict[str, Any]:
    source = config[section]
    return {key: copy.deepcopy(source[key]) for key in keys}


def training_config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_model": _subset(config, "base_model", ("id", "revision", "tokenizer_revision")),
        "input": _subset(config, "input", ("max_length", "truncation", "title_separator", "pair_order")),
        "split": _subset(
            config,
            "split",
            ("seed", "validation_fraction", "stratify_by", "relevant_buckets", "min_stratum_size"),
        ),
        "negatives": _subset(
            config,
            "negatives",
            (
                "weak_rank_min",
                "weak_rank_max",
                "bucket_boundaries",
                "bucket_targets",
                "max_weak_negatives_per_query",
                "weak_negative_weight",
                "judged_negative_weight",
                "max_pairs_per_query",
                "max_judged_pairs_per_query",
                "max_weak_pairs_per_query",
                "resample_per_epoch",
                "spare_slot_priority",
            ),
        ),
        "loss": _subset(config, "loss", ("id", "aggregation")),
        "training": _subset(
            config,
            "training",
            (
                "epochs",
                "micro_batch_queries",
                "grad_accumulation",
                "max_sequences_per_microbatch",
                "weight_decay",
                "warmup_ratio",
                "max_grad_norm",
                "precision",
                "seed",
                "num_workers",
                "no_decay_patterns",
                "device",
                "token_cache",
                "minimum_free_ram_gib",
            ),
        ),
        "runs": copy.deepcopy(config["runs"]),
    }


def scoring_config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_model": copy.deepcopy(config["base_model"]),
        "input": copy.deepcopy(config["input"]),
        "inference": copy.deepcopy(config["inference"]),
        "protocol": copy.deepcopy(config["protocol"]),
    }


def evaluation_config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "validation": copy.deepcopy(config["validation"]),
        "selection": copy.deepcopy(config["selection"]),
        "control": copy.deepcopy(config["control"]),
        "evaluation": copy.deepcopy(config["evaluation"]),
    }


def packaging_config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "archive": copy.deepcopy(config["archive"]),
        "artifacts": copy.deepcopy(config["artifacts"]),
    }


def config_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        "training_config_sha256": canonical_json_sha256(training_config_contract(config)),
        "scoring_config_sha256": canonical_json_sha256(scoring_config_contract(config)),
        "evaluation_config_sha256": canonical_json_sha256(evaluation_config_contract(config)),
        "packaging_config_sha256": canonical_json_sha256(packaging_config_contract(config)),
    }


def source_hashes(config: Mapping[str, Any]) -> dict[str, Any]:
    root = repository_root(config)
    training_hash, training_files = source_set_sha256(root, TRAINING_SOURCE_FILES)
    scoring_hash, scoring_files = source_set_sha256(root, SCORING_SOURCE_FILES)
    evaluation_hash, evaluation_files = source_set_sha256(root, EVALUATION_SOURCE_FILES)
    return {
        "training_source_sha256": training_hash,
        "training_source_files": training_files,
        "scoring_source_sha256": scoring_hash,
        "scoring_source_files": scoring_files,
        "evaluation_source_sha256": evaluation_hash,
        "evaluation_source_files": evaluation_files,
    }


def build_training_fingerprint(components: Mapping[str, Any]) -> str:
    missing = sorted(TRAINING_FINGERPRINT_FIELDS.difference(components))
    extra = sorted(set(components).difference(TRAINING_FINGERPRINT_FIELDS))
    if missing or extra:
        raise ValueError(
            f"training fingerprint component mismatch: missing={missing}, extra={extra}"
        )
    return canonical_json_sha256(
        {name: components[name] for name in sorted(TRAINING_FINGERPRINT_FIELDS)}
    )


def run_learning_rate(
    config: Mapping[str, Any], run_id: str, metrics: Mapping[str, Any] | None = None
) -> float:
    if run_id not in config["runs"]:
        raise ValueError(
            f"unknown run-id {run_id!r}; allowed: {', '.join(config['runs'])}"
        )
    selection_path = resolve_path(config, config["audits"]["checkpoint_selection"])
    ledger_path = resolve_path(config, config["audits"]["dev_access_ledger"])
    if selection_path.exists() or (
        ledger_path.is_file() and ledger_path.stat().st_size > 0
    ):
        raise StageError(
            "finetune/temporal-isolation",
            "fine-tuning is forbidden after checkpoint selection or evaluation access",
        )
    run = config["runs"][run_id]
    if "learning_rate" in run:
        return float(run["learning_rate"])
    if run_id != "B1" or run.get("learning_rate_from") != "best_judged_run":
        raise ValueError(f"run {run_id} has no resolvable learning rate")
    if metrics is None:
        raise StageError("finetune/B1", "A1 and A2 metrics are required")
    choices: list[tuple[float, float]] = []
    for judged_id in ("A1", "A2"):
        entries = metrics.get("runs", {}).get(judged_id, {})
        if not isinstance(entries, Mapping) or set(entries) != {
            "epoch_1",
            "epoch_2",
            "epoch_3",
        }:
            raise StageError("finetune/B1", f"{judged_id} is not complete")
        best = max(
            (dict(value) for value in entries.values()),
            key=lambda value: (float(value["ndcg_at_10"]), -int(value["epoch"])),
        )
        choices.append((float(best["ndcg_at_10"]), float(config["runs"][judged_id]["learning_rate"])))
    choices.sort(key=lambda value: (-value[0], value[1]))
    return choices[0][1]


def training_fingerprint_components(
    config: Mapping[str, Any], *, run_id: str, learning_rate: float
) -> dict[str, Any]:
    regime = str(config["runs"][run_id]["regime"])
    pair_key = {
        "judged_only": "pairs_judged_only",
        "weak_negatives": "pairs_weak_negatives",
        "control_c1": "pairs_control_c1",
    }[regime]
    pair_path = resolve_path(config, config["artifacts"][pair_key])
    split_manifest_path = resolve_path(config, config["audits"]["query_split_manifest"])
    pairs_manifest_path = resolve_path(config, config["audits"]["pairs_manifest"])
    validation_path = resolve_path(config, config["artifacts"]["validation_groups"])
    for path in (pair_path, split_manifest_path, pairs_manifest_path, validation_path):
        assert_not_dev(path)
        if not path.is_file():
            raise StageError("finetune/fingerprint", f"required artifact is missing: {path}")
    pairs_manifest = read_json(pairs_manifest_path)
    section = pairs_manifest.get("regimes", {}).get(regime)
    if not isinstance(section, Mapping):
        raise StageError("finetune/fingerprint", f"pairs manifest is missing {regime}")
    pair_file_hash = sha256_file(pair_path)
    if section.get("pair_file_sha256") != pair_file_hash:
        raise StageError(
            "finetune/fingerprint", f"{regime} pair file differs from its manifest"
        )
    section_payload = dict(section)
    section_hash = section_payload.pop("pair_manifest_section_sha256", None)
    if (
        not isinstance(section_hash, str)
        or canonical_json_sha256(section_payload) != section_hash
    ):
        raise StageError(
            "finetune/fingerprint", f"{regime} pair manifest section hash is invalid"
        )
    training_source = source_set_sha256(repository_root(config), TRAINING_SOURCE_FILES)[0]
    hashes = config_hashes(config)
    return {
        "training_source_sha256": training_source,
        "training_config_sha256": hashes["training_config_sha256"],
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "pair_file_sha256": pair_file_hash,
        "pair_manifest_section_sha256": section_hash,
        "validation_groups_sha256": sha256_file(validation_path),
        "model_id": str(config["base_model"]["id"]),
        "model_revision": str(config["base_model"]["revision"]),
        "tokenizer_revision": str(config["base_model"]["tokenizer_revision"]),
        "run_id": run_id,
        "regime": regime,
        "learning_rate": float(learning_rate),
        "epochs": int(config["runs"][run_id]["epochs"]),
        "implementation_version": str(config["implementation"]["version"]),
        "python_version": platform.python_version(),
        "torch_version": package_version("torch"),
        "transformers_version": package_version("transformers"),
        "tokenizers_version": package_version("tokenizers"),
    }


def _generation_hash(directory: Path, files: Mapping[str, str] | None = None) -> str:
    file_hashes = dict(files or {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in directory.rglob("*")
        if path.is_file() and path.name != "epoch_sidecar.json"
    })
    digest = hashlib.sha256()
    for relative in sorted(file_hashes):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hashes[relative].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _uses_frozen_phase3_model(components: Mapping[str, Any]) -> bool:
    return (
        components.get("model_id") == PINNED_MODEL_ID
        and components.get("model_revision") == PINNED_MODEL_REVISION
        and components.get("tokenizer_revision") == PINNED_MODEL_REVISION
    )


def _materialize_pinned_tokenizer_files(
    directory: Path, components: Mapping[str, Any]
) -> None:
    """Copy all four immutable tokenizer payloads from the pinned Hub revision."""

    if not _uses_frozen_phase3_model(components):
        return
    from huggingface_hub import hf_hub_download

    for filename in TOKENIZER_PAYLOAD_FILES:
        source = Path(
            hf_hub_download(
                repo_id=PINNED_MODEL_ID,
                filename=filename,
                revision=PINNED_MODEL_REVISION,
            )
        )
        shutil.copyfile(source, directory / filename)


def _stale_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return path.with_name(f"{path.name}.stale.{stamp}")


def _atomic_replace_directory(temporary: Path, destination: Path) -> None:
    backup: Path | None = None
    if destination.exists():
        backup = _stale_path(destination)
        destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _invalidate_latest_pointer(run_directory: Path) -> None:
    """Remove an old pointer before replacing the singleton resume boundary.

    There is no atomic transaction spanning a directory rename and a JSON
    pointer rename.  Absence is therefore the only safe intermediate pointer
    state: an older pointer must never claim that a newly replaced (or already
    deleted) ``resume_state`` belongs to its epoch.
    """

    pointer = run_directory / "latest_checkpoint.json"
    if pointer.exists():
        pointer.unlink()


def save_rng_state(path: Path) -> None:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(state, path)


def restore_rng_state(path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_epoch_generation(
    directory: str | Path,
    *,
    expected_fingerprint: str | None = None,
    forward_batch: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    generation = Path(directory)
    sidecar_path = generation / "epoch_sidecar.json"
    sidecar = read_json(sidecar_path)
    if expected_fingerprint is not None and sidecar.get("training_fingerprint") != expected_fingerprint:
        raise ValueError("epoch generation training fingerprint mismatch")
    hashes = sidecar.get("files")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("epoch sidecar has no generation file hashes")
    for relative, expected in hashes.items():
        member = PurePosixPath(str(relative))
        if member.is_absolute() or ".." in member.parts or str(member) != str(relative):
            raise ValueError("epoch sidecar contains an unsafe generation path")
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise ValueError(f"epoch generation has an invalid SHA-256: {relative}")
        path = generation.joinpath(*member.parts)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"epoch generation file hash mismatch: {relative}")
        if "resume" in member.parts:
            raise ValueError("epoch sidecar refers outside its immutable generation")
    actual_files = {
        path.relative_to(generation).as_posix()
        for path in generation.rglob("*")
        if path.is_file() and path.name != "epoch_sidecar.json"
    }
    if actual_files != set(map(str, hashes)):
        raise ValueError(
            "epoch generation file allowlist mismatch: "
            f"missing={sorted(set(map(str, hashes)) - actual_files)}, "
            f"extra={sorted(actual_files - set(map(str, hashes)))}"
        )
    components = sidecar.get("training_fingerprint_components")
    if isinstance(components, Mapping) and _uses_frozen_phase3_model(components):
        if actual_files != MODEL_GENERATION_FILES:
            raise ValueError(
                "frozen Phase 3 generation payload mismatch: "
                f"missing={sorted(MODEL_GENERATION_FILES - actual_files)}, "
                f"extra={sorted(actual_files - MODEL_GENERATION_FILES)}"
            )
    actual_generation_hash = _generation_hash(generation, dict(hashes))
    if sidecar.get("model_generation_sha256") != actual_generation_hash:
        raise ValueError("epoch model_generation_sha256 mismatch")
    # Loading a Transformers model constructs modules before applying stored
    # weights and may otherwise advance the global CPU RNG.  Checkpoint
    # validation must not perturb the dropout stream of the training process.
    with torch.random.fork_rng(devices=[], enabled=True):
        model = AutoModelForSequenceClassification.from_pretrained(
            generation, local_files_only=True
        )
        if int(getattr(model.config, "num_labels", 0)) != 1:
            raise ValueError("epoch generation does not expose num_labels=1")
        if isinstance(components, Mapping) and _uses_frozen_phase3_model(components):
            AutoTokenizer.from_pretrained(generation, local_files_only=True)
        model.eval()
        if forward_batch is not None:
            with torch.inference_mode():
                logits = model(
                    **{name: value.cpu() for name, value in forward_batch.items()}
                ).logits
            if list(logits.shape) != [8, 1] or not bool(torch.isfinite(logits).all()):
                raise ValueError("epoch generation eight-pair forward validation failed")
    return sidecar


def publish_epoch_generation(
    run_directory: str | Path,
    *,
    run_id: str,
    epoch: int,
    model: torch.nn.Module,
    tokenizer: Any,
    training_fingerprint: str,
    fingerprint_components: Mapping[str, Any],
    validation: Mapping[str, Any],
    optimizer_steps: int,
    last_window_query_count: int,
    throughput: Mapping[str, Any],
    peak_gpu_memory_bytes: int,
    peak_rss: int,
    forward_batch: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_directory)
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / f"epoch_{epoch}"
    if destination.is_dir():
        try:
            existing = validate_epoch_generation(
                destination,
                expected_fingerprint=training_fingerprint,
                forward_batch=forward_batch,
            )
        except (ValueError, OSError):
            stale = _stale_path(destination)
            destination.replace(stale)
            print(f"stale epoch generation preserved at {stale}", flush=True)
        else:
            return existing
    temporary = run_dir / f".epoch_{epoch}.tmp.{os.getpid()}"
    if temporary.exists():
        raise StageError("checkpoint/generation", f"temporary directory exists: {temporary}")
    temporary.mkdir(parents=True)
    started_at = utc_now()
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        _materialize_pinned_tokenizer_files(temporary, fingerprint_components)
        files = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        sidecar = {
            "run_id": run_id,
            "epoch": int(epoch),
            "training_fingerprint": training_fingerprint,
            "training_fingerprint_components": dict(fingerprint_components),
            "validation_ndcg_at_10": float(validation["ndcg_at_10"]),
            "optimizer_steps": int(optimizer_steps),
            "last_window_query_count": int(last_window_query_count),
            "throughput": dict(throughput),
            "peak_gpu_memory_bytes": int(peak_gpu_memory_bytes),
            "peak_rss_bytes": int(peak_rss),
            "files": files,
            "model_generation_sha256": _generation_hash(temporary, files),
            "started_at": started_at,
            "published_at": utc_now(),
        }
        atomic_write_json(temporary / "epoch_sidecar.json", sidecar)
        validate_epoch_generation(
            temporary,
            expected_fingerprint=training_fingerprint,
            forward_batch=forward_batch,
        )
        temporary.replace(destination)
        return validate_epoch_generation(
            destination,
            expected_fingerprint=training_fingerprint,
            forward_batch=forward_batch,
        )
    except Exception:
        raise StageError(
            "checkpoint/generation",
            "epoch generation publication failed",
            temporary_path=str(temporary),
        )


def publish_resume_state(
    run_directory: str | Path,
    *,
    run_id: str,
    completed_epoch: int,
    training_fingerprint: str,
    model_generation_sha256: str,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    run_dir = Path(run_directory)
    destination = run_dir / "resume_state"
    temporary = run_dir / f".resume_state.tmp.{os.getpid()}"
    if temporary.exists():
        raise StageError("checkpoint/resume", f"temporary directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        save_rng_state(temporary / "rng_state.pt")
        hashes = {
            name: sha256_file(temporary / name)
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pt")
        }
        sidecar = {
            "run_id": run_id,
            "completed_epoch": int(completed_epoch),
            "training_fingerprint": training_fingerprint,
            "model_generation_sha256": model_generation_sha256,
            "files": hashes,
            "created_at": utc_now(),
        }
        atomic_write_json(temporary / "resume_sidecar.json", sidecar)
        _validate_resume_state_directory(
            temporary,
            expected_fingerprint=training_fingerprint,
            expected_run_id=run_id,
            expected_epoch=completed_epoch,
            expected_generation_sha256=model_generation_sha256,
        )
        _invalidate_latest_pointer(run_dir)
        _atomic_replace_directory(temporary, destination)
        return _validate_resume_state_directory(
            destination,
            expected_fingerprint=training_fingerprint,
            expected_run_id=run_id,
            expected_epoch=completed_epoch,
            expected_generation_sha256=model_generation_sha256,
        )
    except Exception as exc:
        raise StageError(
            "checkpoint/resume",
            str(exc),
            temporary_path=str(temporary),
        ) from exc


def _validate_resume_state_directory(
    directory: Path,
    *,
    expected_fingerprint: str | None = None,
    expected_run_id: str | None = None,
    expected_epoch: int | None = None,
    expected_generation_sha256: str | None = None,
) -> dict[str, Any]:
    actual_files = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
    if actual_files != RESUME_STATE_FILES:
        raise ValueError(
            "resume-state file allowlist mismatch: "
            f"missing={sorted(RESUME_STATE_FILES - actual_files)}, "
            f"extra={sorted(actual_files - RESUME_STATE_FILES)}"
        )
    sidecar = read_json(directory / "resume_sidecar.json")
    if expected_fingerprint is not None and sidecar.get(
        "training_fingerprint"
    ) != expected_fingerprint:
        raise ValueError("resume training fingerprint mismatch")
    if expected_run_id is not None and sidecar.get("run_id") != expected_run_id:
        raise ValueError("resume run_id mismatch")
    if expected_epoch is not None and int(sidecar.get("completed_epoch", -1)) != int(
        expected_epoch
    ):
        raise ValueError("resume completed_epoch mismatch")
    if expected_generation_sha256 is not None and sidecar.get(
        "model_generation_sha256"
    ) != expected_generation_sha256:
        raise ValueError("resume model_generation_sha256 mismatch")
    hashes = sidecar.get("files")
    expected_names = {"optimizer.pt", "scheduler.pt", "rng_state.pt"}
    if not isinstance(hashes, Mapping) or set(hashes) != expected_names:
        raise ValueError("resume sidecar file allowlist mismatch")
    for name in sorted(expected_names):
        expected = hashes[name]
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or sha256_file(directory / name) != expected
        ):
            raise ValueError(f"resume file hash mismatch: {name}")
    generation_hash = sidecar.get("model_generation_sha256")
    if (
        not isinstance(generation_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", generation_hash) is None
    ):
        raise ValueError("resume model_generation_sha256 is invalid")
    return sidecar


def update_latest_checkpoint(
    run_directory: str | Path,
    *,
    epoch: int,
    model_generation_sha256: str,
    resume_state_present: bool,
) -> None:
    run_dir = Path(run_directory)
    generation = run_dir / f"epoch_{epoch}"
    generation_sidecar = validate_epoch_generation(generation)
    if generation_sidecar.get("model_generation_sha256") != model_generation_sha256:
        raise ValueError("latest checkpoint generation hash mismatch")
    resume_dir = run_dir / "resume_state"
    if resume_state_present:
        _validate_resume_state_directory(
            resume_dir,
            expected_epoch=epoch,
            expected_generation_sha256=model_generation_sha256,
        )
    elif resume_dir.exists():
        raise ValueError("latest checkpoint cannot hide an existing resume_state")
    atomic_write_json(
        run_dir / "latest_checkpoint.json",
        {
            "epoch": int(epoch),
            "generation": f"epoch_{epoch}",
            "model_generation_sha256": model_generation_sha256,
            "resume_state_present": bool(resume_state_present),
            "updated_at": utc_now(),
        },
    )


def load_resume_state(
    run_directory: str | Path,
    *,
    expected_fingerprint: str,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> tuple[int, Path]:
    run_dir = Path(run_directory)
    resume_dir = run_dir / "resume_state"
    sidecar = _validate_resume_state_directory(
        resume_dir, expected_fingerprint=expected_fingerprint
    )
    epoch = int(sidecar["completed_epoch"])
    generation = run_dir / f"epoch_{epoch}"
    generation_sidecar = validate_epoch_generation(
        generation, expected_fingerprint=expected_fingerprint
    )
    if sidecar.get("model_generation_sha256") != generation_sidecar.get(
        "model_generation_sha256"
    ):
        raise ValueError("resume model_generation_sha256 mismatch")
    optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location="cpu", weights_only=False))
    scheduler.load_state_dict(torch.load(resume_dir / "scheduler.pt", map_location="cpu", weights_only=False))
    restore_rng_state(resume_dir / "rng_state.pt")
    return epoch, generation


def finalize_run(
    run_directory: str | Path,
    run_manifest: Mapping[str, Any],
    *,
    final_epoch: int,
    model_generation_sha256: str,
) -> dict[str, Any]:
    run_dir = Path(run_directory)
    payload = dict(run_manifest)
    payload.update(
        {
            "finalized": True,
            "resume_available": False,
            "completed_at": utc_now(),
        }
    )
    atomic_write_json(run_dir / "run_manifest.json", payload)
    resume_dir = run_dir / "resume_state"
    _invalidate_latest_pointer(run_dir)
    if resume_dir.exists():
        shutil.rmtree(resume_dir)
    update_latest_checkpoint(
        run_dir,
        epoch=final_epoch,
        model_generation_sha256=model_generation_sha256,
        resume_state_present=False,
    )
    return payload


def disk_preflight(
    directory: str | Path,
    *,
    weight_size_bytes: int,
    resume_state_size_bytes: int,
    free_bytes: int | None = None,
) -> dict[str, int]:
    observed = shutil.disk_usage(directory).free if free_bytes is None else int(free_bytes)
    required = int(math.ceil((3 * weight_size_bytes + resume_state_size_bytes) * 1.5))
    if observed < required:
        raise StageError(
            "finetune/disk-preflight",
            "insufficient free disk for immutable epochs and resume state",
            available_bytes=observed,
            required_bytes=required,
        )
    return {"available_bytes": observed, "required_bytes": required}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _load_text_inputs(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_path = resolve_path(config, config["inputs"]["queries"])
    passage_path = resolve_path(config, config["inputs"]["passages"])
    assert_not_dev(query_path)
    assert_not_dev(passage_path)
    queries = read_training_parquet(
        query_path, label="queries", filters=[("split", "==", "train")]
    )
    if not queries["split"].astype("string").eq("train").all():
        raise RuntimeError("query text train filter failed")
    passages = read_training_parquet(passage_path, label="passages")
    return queries, passages


def _load_model_and_tokenizer(
    config: Mapping[str, Any], *, checkpoint: str | Path | None = None
) -> tuple[torch.nn.Module, Any]:
    if checkpoint is None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(config["base_model"]["id"]),
            revision=str(config["base_model"]["tokenizer_revision"]),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(config["base_model"]["id"]),
            revision=str(config["base_model"]["revision"]),
        )
    else:
        local = Path(checkpoint).resolve()
        assert_not_dev(local)
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(local, local_files_only=True)
    if int(getattr(model.config, "num_labels", 0)) != 1:
        raise StageError("model-load", "checkpoint does not expose num_labels=1")
    return model, tokenizer


def _assert_fp32_cuda_model(model: torch.nn.Module, *, stage: str) -> None:
    parameters = list(model.parameters())
    if not parameters or {parameter.device.type for parameter in parameters} != {"cuda"}:
        raise StageError(stage, "model parameters are not entirely on CUDA")
    floating_dtypes = {
        parameter.dtype for parameter in parameters if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.float32}:
        raise StageError(
            stage,
            "training precision must be exactly fp32",
            observed_dtypes=sorted(str(value) for value in floating_dtypes),
        )


def _validation_inputs(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    candidates_path = resolve_path(config, config["artifacts"]["validation_groups"])
    qrels_path = resolve_path(config, config["inputs"]["train_qrels"])
    assert_not_dev(candidates_path)
    assert_not_dev(qrels_path)
    candidates = read_training_parquet(candidates_path, label="validation groups")
    if not candidates["split"].astype("string").eq("train").all():
        raise StageError(
            "validation", "validation groups contain isolated evaluation rows"
        )
    if not qrels_path.is_file():
        raise StageError("validation", f"train qrels are missing: {qrels_path}")
    qrels = load_qrels(qrels_path, split="train")
    if "split" in qrels.columns:
        qrels = qrels.loc[qrels["split"].astype("string").eq("train")].copy()
        if not qrels["split"].astype("string").eq("train").all():
            raise RuntimeError("qrels train filter failed")
    query_ids = sorted(candidates["query_id"].astype("string").map(str).unique())
    return candidates, qrels, query_ids


def score_validation(
    config: Mapping[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    device: str,
    limit_queries: int | None = None,
) -> dict[str, Any]:
    candidates, qrels, query_ids = _validation_inputs(config)
    if limit_queries is not None:
        query_ids = query_ids[: int(limit_queries)]
        candidates = candidates.loc[candidates["query_id"].astype("string").isin(query_ids)].copy()
    queries, passages = _load_text_inputs(config)
    synthetic_pairs = candidates[["query_id", "docid"]].rename(
        columns={"docid": "positive_docid"}
    )
    synthetic_pairs["negative_docid"] = synthetic_pairs["positive_docid"]
    cache, cache_report = build_token_cache(
        synthetic_pairs,
        queries,
        passages,
        tokenizer,
        max_length=int(config["input"]["max_length"]),
        minimum_free_ram_gib=float(config["training"]["minimum_free_ram_gib"]),
    )
    keys = sorted(cache.lookup)
    batch_size = int(config["inference"]["batch_size"])
    rows: list[dict[str, Any]] = []
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(keys), batch_size):
            batch_keys = keys[offset : offset + batch_size]
            try:
                encoded = [cache.encoded(*key) for key in batch_keys]
                batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
                tensors = {name: value.to(device) for name, value in batch.items()}
                logits = model(**tensors).logits.reshape(-1)
            except torch.cuda.OutOfMemoryError as exc:
                raise StageError(
                    "validation/oom",
                    "CUDA OOM; automatic batch-size changes are forbidden",
                    batch_size=len(batch_keys),
                ) from exc
            if not bool(torch.isfinite(logits).all()):
                raise StageError("validation/forward", "non-finite validation logits")
            scores = logits.detach().to(dtype=torch.float32).cpu().numpy()
            rows.extend(
                {"query_id": query_id, "docid": docid, "score": np.float32(score)}
                for (query_id, docid), score in zip(batch_keys, scores, strict=True)
            )
    score_frame = pd.DataFrame(rows)
    ranking = rank_candidates_by_score(candidates, score_frame)
    metric = evaluate_ranked_ndcg_at_10(ranking, qrels, query_ids=query_ids)
    bm25 = candidates[["query_id", "docid", "bm25_rank"]]
    sparse = sparse_judgment_diagnostics(
        candidates=candidates,
        qrels=qrels,
        ranking=ranking,
        bm25_ranking=bm25,
        query_ids=query_ids,
    )
    elapsed = max(time.perf_counter() - started, 1e-12)
    return {
        **metric,
        "ranking": ranking,
        "sparse_diagnostics": sparse,
        "validation_pairs_per_second": len(candidates) / elapsed,
        "seconds": elapsed,
        "token_cache": cache_report,
    }


def _base_weight_sha256(config: Mapping[str, Any]) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=str(config["base_model"]["id"]),
        filename="model.safetensors",
        revision=str(config["base_model"]["revision"]),
    )
    return sha256_file(path)


def _metrics_path(config: Mapping[str, Any]) -> Path:
    path = resolve_path(config, config["metrics"]["validation_checkpoint_metrics"])
    assert_not_dev(path)
    return path


def _validation_trec_cross_check(
    config: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the preregistered one-time NIST check for the S0 validation metric."""

    from . import rerank as rerank_module

    ranking = validation.get("ranking")
    if not isinstance(ranking, pd.DataFrame):
        raise StageError("validation/trec-cross-check", "validation ranking is missing")
    _, qrels, query_ids = _validation_inputs(config)
    qrels = qrels.loc[qrels["query_id"].astype("string").isin(query_ids)].copy()
    work = resolve_path(config, config["paths"]["work_dir"])
    assert_not_dev(work)
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s0-trec-check-", dir=work) as directory:
        root = Path(directory)
        qrels_path = root / "validation_qrels.trec"
        run_path = root / "validation_s0.trec"
        with qrels_path.open("w", encoding="utf-8") as stream:
            for row in qrels.sort_values(["query_id", "docid"], kind="mergesort").itertuples(index=False):
                stream.write(
                    f"{row.query_id} 0 {row.docid} {int(row.relevance_grade)}\n"
                )
        with run_path.open("w", encoding="utf-8") as stream:
            for row in ranking.sort_values(["query_id", "rank"], kind="mergesort").itertuples(index=False):
                stream.write(
                    f"{row.query_id} Q0 {row.docid} {int(row.rank)} "
                    f"{1_000_000 - int(row.rank):.4f} validation-s0\n"
                )
        phase2 = rerank_module.load_rerank_config(
            repository_root(config) / "configs/rerank.yaml"
        )
        executable = rerank_module.resolve_trec_eval(phase2)
        rerank_module.validate_trec_eval_build_provenance(
            phase2, executable=executable
        )
        command = [
            str(executable),
            "-c",
            "-M",
            "100",
            "-m",
            "ndcg_cut.10",
            str(qrels_path),
            str(run_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if result.returncode != 0:
            raise StageError(
                "validation/trec-cross-check",
                "trec_eval failed",
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        official = parse_trec_eval_metric(result.stdout, "ndcg_cut_10")
    python_value = float(validation["ndcg_at_10"])
    difference = abs(official - python_value)
    tolerance = float(config["validation"]["tolerance"])
    if difference > tolerance:
        raise StageError(
            "validation/trec-cross-check",
            "Python validation nDCG differs from NIST trec_eval",
            official=official,
            python=python_value,
            absolute_difference=difference,
            tolerance=tolerance,
        )
    return {
        "status": "PASS",
        "performed_once_on": "S0",
        "command": command[:-2] + ["<validation-qrels>", "<validation-run>"],
        "official_ndcg_at_10": official,
        "python_ndcg_at_10": python_value,
        "absolute_difference": difference,
        "tolerance": tolerance,
    }


def _load_metrics(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _metrics_path(config)
    return read_json(path) if path.is_file() else {"schema_version": 1, "runs": {}}


def validate_checkpoint(
    config: Mapping[str, Any], *, checkpoint: str | Path = "base", device: str | None = None
) -> dict[str, Any]:
    validate_finetune_config(config)
    selection_path = resolve_path(config, config["audits"]["checkpoint_selection"])
    ledger_path = resolve_path(config, config["audits"]["dev_access_ledger"])
    if selection_path.exists() or (
        ledger_path.is_file() and ledger_path.stat().st_size > 0
    ):
        raise StageError(
            "validation/temporal-isolation",
            "validation metrics are frozen after checkpoint selection",
        )
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    local = None if str(checkpoint) == "base" else Path(checkpoint)
    model, tokenizer = _load_model_and_tokenizer(config, checkpoint=local)
    model.to(selected_device)
    validation = score_validation(config, model, tokenizer, device=selected_device)
    local_generation = None if local is None else validate_epoch_generation(local)
    checkpoint_hash = (
        _base_weight_sha256(config)
        if local is None
        else local_generation["files"]["model.safetensors"]
    )
    entry = {
        "kind": "S0" if local is None else "checkpoint",
        "epoch": 0 if local is None else int(local_generation["epoch"]),
        "ndcg_at_10": float(validation["ndcg_at_10"]),
        "per_query": validation["per_query"],
        "checkpoint_sha256": checkpoint_hash,
        "sparse_diagnostics": validation["sparse_diagnostics"],
        "evaluated_at": utc_now(),
    }
    if local_generation is not None:
        entry["model_generation_sha256"] = local_generation[
            "model_generation_sha256"
        ]
    if local is None and bool(config["validation"]["cross_check_with_trec_eval_once"]):
        entry["trec_eval_cross_check"] = _validation_trec_cross_check(
            config, validation
        )
    metrics = _load_metrics(config)
    if local is None:
        metrics["S0"] = entry
    atomic_write_json(_metrics_path(config), metrics)
    return {"status": "PASS", **entry}


def smoke_expected_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_finetune_config(config)
    hashes = config_hashes(config)
    source = source_set_sha256(repository_root(config), TRAINING_SOURCE_FILES)[0]
    split_manifest = resolve_path(config, config["audits"]["query_split_manifest"])
    pairs_manifest_path = resolve_path(config, config["audits"]["pairs_manifest"])
    assert_not_dev(split_manifest)
    assert_not_dev(pairs_manifest_path)
    pairs_manifest = read_json(pairs_manifest_path)
    pair_hashes: dict[str, str] = {}
    for regime, key in (
        ("judged_only", "pairs_judged_only"),
        ("weak_negatives", "pairs_weak_negatives"),
        ("control_c1", "pairs_control_c1"),
    ):
        pair_path = resolve_path(config, config["artifacts"][key])
        assert_not_dev(pair_path)
        pair_hashes[regime] = sha256_file(pair_path)
        section = pairs_manifest.get("regimes", {}).get(regime, {})
        if section.get("pair_file_sha256") != pair_hashes[regime]:
            raise ValueError(f"{regime} pair manifest/file mismatch")
    return {
        "model_id": str(config["base_model"]["id"]),
        "model_revision": str(config["base_model"]["revision"]),
        "tokenizer_revision": str(config["base_model"]["tokenizer_revision"]),
        "training_config_sha256": hashes["training_config_sha256"],
        "training_source_sha256": source,
        "split_manifest_sha256": sha256_file(split_manifest),
        "pair_file_sha256": pair_hashes,
        "implementation_version": str(config["implementation"]["version"]),
        "score_schema_version": int(config["implementation"]["score_schema_version"]),
    }


def validate_smoke_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_path(config, config["audits"]["finetune_smoke"])
    assert_not_dev(path)
    report = read_json(path)
    expected = smoke_expected_fields(config)
    if report.get("status") != "PASS":
        raise StageError("finetune/smoke-gate", "smoke status is not PASS")
    if report.get("fixture_only") is not False:
        raise StageError("finetune/smoke-gate", "fixture-only smoke is forbidden")
    for boolean_field in (
        "real_model_forward",
        "real_optimizer_step",
        "checkpoint_save_load_roundtrip",
        "resume_state_roundtrip",
        "zip_hash_roundtrip",
    ):
        if report.get(boolean_field) is not True:
            raise StageError("finetune/smoke-gate", f"{boolean_field} is not true")
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        raise StageError(
            "finetune/smoke-gate",
            "smoke load-bearing fields differ from current inputs",
            mismatches=mismatches,
        )
    if report.get("device") != "cuda" or report.get("dtype") != "float32":
        raise StageError(
            "finetune/smoke-gate", "smoke device/dtype contract is incompatible"
        )
    for measured_field in (
        "pairs_processed",
        "peak_gpu_memory_bytes",
        "peak_rss_bytes",
        "token_cache_bytes",
    ):
        measured = report.get(measured_field)
        if (
            not isinstance(measured, int)
            or isinstance(measured, bool)
            or measured <= 0
        ):
            raise StageError(
                "finetune/smoke-gate",
                f"smoke field {measured_field} must be a positive integer",
            )
    if report.get("token_cache_representation") != config["training"]["token_cache"]:
        raise StageError(
            "finetune/smoke-gate", "smoke token-cache representation changed"
        )
    throughput = report.get("throughput")
    if not isinstance(throughput, Mapping) or any(
        not isinstance(throughput.get(name), (int, float))
        or isinstance(throughput.get(name), bool)
        or not math.isfinite(float(throughput[name]))
        or float(throughput[name]) <= 0
        for name in ("pairs_per_second", "validation_pairs_per_second")
    ):
        raise StageError(
            "finetune/smoke-gate", "smoke throughput is missing or non-positive"
        )
    return report


def _forward_validation_batch(
    pairs: pd.DataFrame, cache: TokenCache, tokenizer: Any
) -> dict[str, torch.Tensor]:
    keys = sorted(cache.lookup)[:8]
    if not keys:
        raise StageError("checkpoint/validation", "no real pairs for checkpoint forward")
    while len(keys) < 8:
        keys.append(keys[len(keys) % len(keys)])
    return tokenizer.pad(
        [cache.encoded(*key) for key in keys], padding=True, return_tensors="pt"
    )


def _write_control_report(
    config: Mapping[str, Any], *, c1_entry: Mapping[str, Any], pair_hash: str, pairs: pd.DataFrame
) -> dict[str, Any]:
    metrics = _load_metrics(config)
    base = metrics.get("S0", {}).get("per_query")
    system = c1_entry.get("per_query")
    if not isinstance(base, Mapping) or not isinstance(system, Mapping) or set(base) != set(system):
        raise StageError("control/C1", "S0 and C1 validation universes differ")
    deltas = [float(system[qid]) - float(base[qid]) for qid in sorted(base)]
    bootstrap = paired_bootstrap(
        deltas,
        resamples=int(config["control"]["c1_bootstrap_resamples"]),
        seed=int(config["control"]["c1_bootstrap_seed"]),
        confidence=0.95,
    )
    if float(bootstrap["ci_low"]) > 0:
        verdict = "BLOCKED_FOR_REVIEW"
        reason = "shuffled-label control has a strictly positive paired-bootstrap lower bound"
    elif float(bootstrap["mean_delta"]) > 0:
        verdict = "WARN"
        reason = "positive control mean with a confidence interval containing zero"
    else:
        verdict = "PASS"
        reason = "control does not show a positive mean validation delta"
    split_path = resolve_path(config, config["artifacts"]["query_split"])
    assert_not_dev(split_path)
    split = read_training_parquet(split_path, label="query split")
    fit_ids = set(
        split.loc[split["split_role"].eq(TRAIN_ROLE), "query_id"].map(str)
    )
    validation_ids = set(
        split.loc[split["split_role"].eq(VALIDATION_ROLE), "query_id"].map(str)
    )
    overlap = sorted(fit_ids & validation_ids)
    structural = {
        "train_fit_train_validation_intersection_empty": not overlap,
        "train_fit_train_validation_intersection": overlap,
        "forbidden_evaluation_path_opened": False,
        "positive_equals_negative_count": int(
            pairs["positive_docid"].eq(pairs["negative_docid"]).sum()
        ),
    }
    if (
        not structural["train_fit_train_validation_intersection_empty"]
        or structural["forbidden_evaluation_path_opened"]
        or structural["positive_equals_negative_count"]
    ):
        verdict = "FAIL"
        reason = "a structural control invariant failed"
    report = {
        "status": verdict,
        "reason": reason,
        "mean_delta": float(bootstrap["mean_delta"]),
        "ci_lower": float(bootstrap["ci_low"]),
        "ci_upper": float(bootstrap["ci_high"]),
        "bootstrap": bootstrap,
        "structural_checks": structural,
        "pair_file_sha256": pair_hash,
        "query_group_count": int(pairs["query_id"].nunique()),
        "pair_count": int(len(pairs)),
        "diagnostic_role": "training_path_control_not_absolute_leakage_detector",
    }
    path = resolve_path(config, config["audits"]["control_report"])
    assert_not_dev(path)
    atomic_write_json(path, report)
    return report


def _require_control_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_path(config, config["audits"]["control_report"])
    assert_not_dev(path)
    report = read_json(path)
    control_pairs = resolve_path(
        config, config["artifacts"]["pairs_control_c1"]
    )
    assert_not_dev(control_pairs)
    if (
        not control_pairs.is_file()
        or report.get("pair_file_sha256") != sha256_file(control_pairs)
    ):
        raise StageError(
            "finetune/control-gate", "C1 report refers to another control pair set"
        )
    if report.get("status") in {"FAIL", "BLOCKED_FOR_REVIEW"}:
        raise StageError(
            "finetune/control-gate",
            f"C1 verdict is {report.get('status')}",
            reason=report.get("reason"),
        )
    if report.get("status") not in {"PASS", "WARN"}:
        raise StageError("finetune/control-gate", "C1 report is missing a valid verdict")
    return report


def run_finetune(
    config: Mapping[str, Any],
    *,
    run_id: str,
    resume: bool = False,
    overwrite: bool = False,
    interrupt_after_epoch: int | None = None,
    fault_after_generation: bool = False,
) -> dict[str, Any]:
    validate_finetune_config(config)
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if run_id not in config["runs"]:
        raise ValueError(
            f"unknown run-id {run_id!r}; allowed: {', '.join(config['runs'])}"
        )
    if str(config["training"]["device"]) != "cuda" or not torch.cuda.is_available():
        raise StageError(
            "finetune/device",
            "production fine-tuning requires CUDA; CPU/MPS is forbidden",
            configured_device=config["training"]["device"],
            cuda_available=torch.cuda.is_available(),
        )
    immutable = phase12_immutable_snapshot(config, require_all=True)
    smoke = validate_smoke_gate(config)
    if run_id != "C1":
        _require_control_gate(config)
    metrics = _load_metrics(config)
    if "S0" not in metrics:
        raise StageError("finetune/validation", "validate-checkpoint base must run first")
    if run_id == "B1":
        for judged_run_id in ("A1", "A2"):
            judged_manifest_path = (
                resolve_path(config, config["artifacts"]["models_dir"])
                / judged_run_id
                / "run_manifest.json"
            )
            assert_not_dev(judged_manifest_path)
            if not judged_manifest_path.is_file():
                raise StageError(
                    "finetune/B1", f"{judged_run_id} finalized run is missing"
                )
            judged_manifest = read_json(judged_manifest_path)
            if (
                judged_manifest.get("finalized") is not True
                or judged_manifest.get("resume_available") is not False
            ):
                raise StageError(
                    "finetune/B1", f"{judged_run_id} is not finalized"
                )
    learning_rate = run_learning_rate(config, run_id, metrics)
    components = training_fingerprint_components(
        config, run_id=run_id, learning_rate=learning_rate
    )
    fingerprint = build_training_fingerprint(components)
    run_config = config["runs"][run_id]
    regime = str(run_config["regime"])
    pair_key = {
        "judged_only": "pairs_judged_only",
        "weak_negatives": "pairs_weak_negatives",
        "control_c1": "pairs_control_c1",
    }[regime]
    pair_path = resolve_path(config, config["artifacts"][pair_key])
    assert_not_dev(pair_path)
    pairs = read_training_parquet(pair_path, label=f"{regime} pairs")
    run_dir = resolve_path(config, config["artifacts"]["models_dir"]) / run_id
    assert_not_dev(run_dir)
    run_manifest_path = run_dir / "run_manifest.json"
    existing_manifest: dict[str, Any] | None = None
    if run_manifest_path.is_file():
        existing_manifest = read_json(run_manifest_path)
        if existing_manifest.get("finalized") is True:
            if resume:
                raise StageError(
                    "finetune/resume",
                    "finalized run cannot be resumed; omit --resume or use --overwrite",
                )
            if (
                existing_manifest.get("training_fingerprint") == fingerprint
                and not overwrite
            ):
                latest = read_json(run_dir / "latest_checkpoint.json")
                expected_final_epoch = int(config["runs"][run_id]["epochs"])
                if (
                    existing_manifest.get("resume_available") is not False
                    or int(existing_manifest.get("completed_epoch", -1))
                    != expected_final_epoch
                    or int(latest.get("epoch", -1)) != expected_final_epoch
                    or latest.get("generation") != f"epoch_{expected_final_epoch}"
                    or latest.get("resume_state_present") is not False
                    or (run_dir / "resume_state").exists()
                ):
                    raise StageError(
                        "finetune/reuse", "finalized run lifecycle is inconsistent"
                    )
                final_generation = validate_epoch_generation(
                    run_dir / str(latest["generation"]),
                    expected_fingerprint=fingerprint,
                )
                if latest.get("model_generation_sha256") != final_generation.get(
                    "model_generation_sha256"
                ):
                    raise StageError(
                        "finetune/reuse", "latest checkpoint hash is inconsistent"
                    )
                verify_phase12_immutable(config, immutable)
                return {
                    "status": "PASS",
                    "action": "reused_finalized",
                    **existing_manifest,
                }
            if not overwrite:
                raise StageError("finetune/reuse", "finalized run exists and is incompatible")
        elif not resume and not overwrite:
            raise StageError("finetune/resume", "unfinished run exists; pass --resume")
    if overwrite and run_dir.exists():
        stale = _stale_path(run_dir)
        run_dir.replace(stale)
        print(f"previous run preserved at {stale}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["training"]["seed"])
    seed_everything(seed)
    resume_generation: Path | None = None
    resume_completed_epoch = 0
    if resume:
        existing_manifest = read_json(run_manifest_path)
        if (
            existing_manifest.get("finalized") is True
            or existing_manifest.get("resume_available") is not True
        ):
            raise StageError("finetune/resume", "run is finalized or resume is unavailable")
        resume_sidecar_path = run_dir / "resume_state" / "resume_sidecar.json"
        resume_sidecar = read_json(resume_sidecar_path)
        resume_completed_epoch = int(resume_sidecar["completed_epoch"])
        resume_generation = run_dir / f"epoch_{resume_completed_epoch}"
        if resume_sidecar.get("training_fingerprint") != fingerprint:
            stale = _stale_path(resume_generation)
            if resume_generation.exists():
                resume_generation.replace(stale)
                print(f"incompatible generation preserved at {stale}", flush=True)
            raise StageError(
                "finetune/resume",
                "resume training fingerprint mismatch",
                stale_generation=str(stale),
            )
        try:
            generation_sidecar = validate_epoch_generation(
                resume_generation, expected_fingerprint=fingerprint
            )
        except (ValueError, OSError) as exc:
            stale = _stale_path(resume_generation)
            if resume_generation.exists():
                resume_generation.replace(stale)
                print(f"invalid generation preserved at {stale}", flush=True)
            raise StageError(
                "finetune/resume",
                str(exc),
                stale_generation=str(stale),
            ) from exc
        if (
            generation_sidecar.get("model_generation_sha256")
            != resume_sidecar.get("model_generation_sha256")
        ):
            stale = _stale_path(resume_generation)
            if resume_generation.exists():
                resume_generation.replace(stale)
                print(f"mismatched generation preserved at {stale}", flush=True)
            raise StageError(
                "finetune/resume",
                "resume model_generation_sha256 mismatch",
                stale_generation=str(stale),
            )
    model, tokenizer = _load_model_and_tokenizer(
        config, checkpoint=resume_generation
    )
    try:
        model.to("cuda")
    except torch.cuda.OutOfMemoryError as exc:
        raise StageError(
            "finetune/oom",
            "CUDA OOM while loading the model; automatic batch-size changes are forbidden",
        ) from exc
    _assert_fp32_cuda_model(model, stage="finetune/device")
    queries, passages = _load_text_inputs(config)
    token_cache, cache_report = build_token_cache(
        pairs,
        queries,
        passages,
        tokenizer,
        max_length=int(config["input"]["max_length"]),
        minimum_free_ram_gib=float(config["training"]["minimum_free_ram_gib"]),
    )
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    disk = disk_preflight(
        run_dir,
        weight_size_bytes=weight_bytes,
        resume_state_size_bytes=3 * weight_bytes,
    )
    query_groups = {
        str(query_id): group.copy()
        for query_id, group in pairs.groupby("query_id", sort=True)
    }
    epochs = int(run_config["epochs"])
    accumulation = int(config["training"]["grad_accumulation"])
    total_steps = epochs * math.ceil(len(query_groups) / accumulation)
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=float(config["training"]["weight_decay"]),
        no_decay_patterns=config["training"]["no_decay_patterns"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config["training"]["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    start_epoch = 1
    if resume:
        completed, generation = load_resume_state(
            run_dir,
            expected_fingerprint=fingerprint,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if completed != resume_completed_epoch or generation != resume_generation:
            raise StageError("finetune/resume", "resume boundary changed during load")
        start_epoch = completed + 1
    torch.cuda.reset_peak_memory_stats()
    history_path = resolve_path(
        config,
        str(config["metrics"]["training_history_template"]).format(run_id=run_id),
    )
    assert_not_dev(history_path)
    history = read_json(history_path) if history_path.is_file() and resume else {
        "run_id": run_id,
        "regime": regime,
        "learning_rate": learning_rate,
        "epochs": [],
    }
    run_manifest: dict[str, Any] = {
        "run_id": run_id,
        "regime": regime,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "training_fingerprint": fingerprint,
        "training_fingerprint_components": components,
        "finalized": False,
        "resume_available": True,
        "token_cache_bytes": cache_report["token_cache_bytes"],
        "token_cache_representation": cache_report["token_cache_representation"],
        "disk_preflight": disk,
        "smoke_gate_sha256": sha256_file(resolve_path(config, config["audits"]["finetune_smoke"])),
        "immutable_phase12": immutable,
        "started_at": utc_now(),
    }
    if resume and existing_manifest is not None:
        run_manifest["started_at"] = existing_manifest.get(
            "started_at", run_manifest["started_at"]
        )
        run_manifest["resumed_at"] = utc_now()
        completed_history = list(history.get("epochs", []))
        if completed_history:
            latest_history = max(
                completed_history, key=lambda item: int(item["epoch"])
            )
            run_manifest.update(
                {
                    "completed_epoch": int(latest_history["epoch"]),
                    "last_window_query_count": int(
                        latest_history["last_window_query_count"]
                    ),
                    "optimizer_steps": sum(
                        int(item["optimizer_steps"]) for item in completed_history
                    ),
                    "validation_history": [
                        {
                            "epoch": int(item["epoch"]),
                            "ndcg_at_10": float(item["ndcg_at_10"]),
                        }
                        for item in completed_history
                    ],
                    "throughput": dict(latest_history["throughput"]),
                    "peak_rss_bytes": existing_manifest.get("peak_rss_bytes"),
                    "peak_gpu_memory_bytes": existing_manifest.get(
                        "peak_gpu_memory_bytes"
                    ),
                }
            )
    atomic_write_json(run_manifest_path, run_manifest)
    forward_batch = _forward_validation_batch(pairs, token_cache, tokenizer)
    final_sidecar: dict[str, Any] | None = (
        validate_epoch_generation(
            resume_generation,
            expected_fingerprint=fingerprint,
            forward_batch=forward_batch,
        )
        if resume_generation is not None and start_epoch > epochs
        else None
    )
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        order = epoch_query_order(list(query_groups), seed=seed, epoch=epoch)
        epoch_started = time.perf_counter()
        pair_counter = 0
        global_step_offset = sum(
            int(item["optimizer_steps"]) for item in history.get("epochs", [])
        )

        def loss_for_query(query_id: str) -> torch.Tensor:
            nonlocal pair_counter
            group = query_groups[query_id]
            pair_counter += len(group)
            try:
                return model_query_loss(
                    model,
                    tokenizer,
                    token_cache,
                    group,
                    device="cuda",
                    max_sequences_per_microbatch=int(
                        config["training"]["max_sequences_per_microbatch"]
                    ),
                )
            except torch.cuda.OutOfMemoryError as exc:
                raise StageError(
                    "finetune/oom",
                    "CUDA OOM; automatic batch-size changes are forbidden",
                    query_id=query_id,
                ) from exc

        def log_step(item: dict[str, Any]) -> None:
            item.update(
                {
                    "step": global_step_offset + int(item["optimizer_step"]),
                    "epoch": epoch,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "pairs_per_second": pair_counter
                    / max(time.perf_counter() - epoch_started, 1e-12),
                }
            )
            print(json.dumps(item, ensure_ascii=False), flush=True)

        try:
            epoch_result = run_accumulation_epoch(
                order,
                loss_for_query=loss_for_query,
                optimizer=optimizer,
                scheduler=scheduler,
                parameters=model.parameters(),
                accumulation=accumulation,
                max_grad_norm=float(config["training"]["max_grad_norm"]),
                on_step=log_step,
            )
        except StageError as exc:
            atomic_write_json(
                run_dir / "failed_step_diagnostic.json",
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "stage": exc.stage,
                    "root_cause": exc.root_cause,
                    "diagnostics": exc.diagnostics,
                    "recorded_at": utc_now(),
                },
            )
            raise
        except torch.cuda.OutOfMemoryError as exc:
            failure = StageError(
                "finetune/oom",
                "CUDA OOM; automatic batch-size changes are forbidden",
                epoch=epoch,
            )
            atomic_write_json(
                run_dir / "failed_step_diagnostic.json",
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "stage": failure.stage,
                    "root_cause": failure.root_cause,
                    "diagnostics": failure.diagnostics,
                    "recorded_at": utc_now(),
                },
            )
            raise failure from exc
        elapsed = max(time.perf_counter() - epoch_started, 1e-12)
        model.eval()
        validation = score_validation(config, model, tokenizer, device="cuda")
        throughput = {
            "seconds": elapsed,
            "queries_per_second": len(order) / elapsed,
            "pairs_per_second": pair_counter / elapsed,
        }
        final_sidecar = publish_epoch_generation(
            run_dir,
            run_id=run_id,
            epoch=epoch,
            model=model,
            tokenizer=tokenizer,
            training_fingerprint=fingerprint,
            fingerprint_components=components,
            validation=validation,
            optimizer_steps=int(epoch_result["optimizer_steps"]),
            last_window_query_count=int(epoch_result["last_window_query_count"]),
            throughput=throughput,
            peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()),
            peak_rss=peak_rss_bytes(),
            forward_batch=forward_batch,
        )
        if fault_after_generation:
            raise StageError(
                "checkpoint/fault-injection",
                "injected failure after generation publication",
            )
        epoch_history = {
            "epoch": epoch,
            "ndcg_at_10": float(validation["ndcg_at_10"]),
            "per_query": validation["per_query"],
            "sparse_diagnostics": validation["sparse_diagnostics"],
            "checkpoint_sha256": final_sidecar["files"]["model.safetensors"],
            "model_generation_sha256": final_sidecar["model_generation_sha256"],
            "optimizer_steps": epoch_result["optimizer_steps"],
            "last_window_query_count": epoch_result["last_window_query_count"],
            "throughput": throughput,
            "step_logs": epoch_result["logs"],
        }
        history["epochs"] = [
            item for item in history.get("epochs", []) if int(item["epoch"]) != epoch
        ] + [epoch_history]
        history["epochs"].sort(key=lambda item: int(item["epoch"]))
        atomic_write_json(history_path, history)
        metrics = _load_metrics(config)
        metrics.setdefault("runs", {}).setdefault(run_id, {})[f"epoch_{epoch}"] = epoch_history
        atomic_write_json(_metrics_path(config), metrics)
        # Persist deterministic validation/history before publishing the resume
        # boundary.  If interrupted here, the previous resume state remains the
        # authority and this epoch is safely replayed/replaced.
        resume_sidecar = publish_resume_state(
            run_dir,
            run_id=run_id,
            completed_epoch=epoch,
            training_fingerprint=fingerprint,
            model_generation_sha256=str(final_sidecar["model_generation_sha256"]),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        run_manifest.update(
            {
                "completed_epoch": epoch,
                "resume_available": True,
                "latest_model_generation_sha256": final_sidecar[
                    "model_generation_sha256"
                ],
                "last_window_query_count": epoch_result["last_window_query_count"],
                "optimizer_steps": sum(
                    int(item["optimizer_steps"]) for item in history["epochs"]
                ),
                "validation_history": [
                    {"epoch": item["epoch"], "ndcg_at_10": item["ndcg_at_10"]}
                    for item in history["epochs"]
                ],
                "throughput": throughput,
                "peak_rss_bytes": max(
                    int(run_manifest.get("peak_rss_bytes") or 0), peak_rss_bytes()
                ),
                "peak_gpu_memory_bytes": max(
                    int(run_manifest.get("peak_gpu_memory_bytes") or 0),
                    int(torch.cuda.max_memory_allocated()),
                ),
                "resume_sidecar": resume_sidecar,
            }
        )
        atomic_write_json(run_manifest_path, run_manifest)
        # Publish the latest pointer only after the generation, resume state,
        # validation/history, and resumable manifest are all durable.  A crash
        # at any earlier boundary therefore leaves the previous pointer valid.
        update_latest_checkpoint(
            run_dir,
            epoch=epoch,
            model_generation_sha256=str(final_sidecar["model_generation_sha256"]),
            resume_state_present=True,
        )
        verify_phase12_immutable(config, immutable)
        if interrupt_after_epoch == epoch:
            return {
                "status": "INTERRUPTED_FOR_TEST",
                "run_id": run_id,
                "completed_epoch": epoch,
                "resume_available": True,
            }
    if final_sidecar is None:
        raise StageError("finetune", "no epoch was executed or resumed")
    finalized = finalize_run(
        run_dir,
        run_manifest,
        final_epoch=epochs,
        model_generation_sha256=str(final_sidecar["model_generation_sha256"]),
    )
    if run_id == "C1":
        metrics = _load_metrics(config)
        c1_entry = metrics["runs"]["C1"]["epoch_1"]
        control = _write_control_report(
            config,
            c1_entry=c1_entry,
            pair_hash=sha256_file(pair_path),
            pairs=pairs,
        )
        finalized["control_verdict"] = control["status"]
    verify_phase12_immutable(config, immutable)
    return {"status": "PASS", "action": "trained", **finalized}


def smoke_finetune(
    config: Mapping[str, Any], *, limit_pairs: int = 64
) -> dict[str, Any]:
    validate_finetune_config(config)
    selection_path = resolve_path(config, config["audits"]["checkpoint_selection"])
    ledger_path = resolve_path(config, config["audits"]["dev_access_ledger"])
    if selection_path.exists() or (
        ledger_path.is_file() and ledger_path.stat().st_size > 0
    ):
        raise StageError(
            "smoke-finetune/temporal-isolation",
            "smoke training is forbidden after checkpoint selection or dev access",
        )
    if not torch.cuda.is_available():
        raise StageError("smoke-finetune/device", "real smoke requires CUDA")
    if limit_pairs <= 0:
        raise ValueError("limit-pairs must be positive")
    expected = smoke_expected_fields(config)
    pair_path = resolve_path(config, config["artifacts"]["pairs_judged_only"])
    assert_not_dev(pair_path)
    pairs = read_training_parquet(pair_path, label="smoke pairs")
    selected_queries: list[str] = []
    selected_count = 0
    for query_id, group in pairs.groupby("query_id", sort=True):
        selected_queries.append(str(query_id))
        selected_count += len(group)
        if selected_count >= limit_pairs and len(selected_queries) >= 17:
            break
    smoke_pairs = pairs.loc[pairs["query_id"].astype("string").isin(selected_queries)].copy()
    model, tokenizer = _load_model_and_tokenizer(config)
    try:
        model.to("cuda")
    except torch.cuda.OutOfMemoryError as exc:
        raise StageError(
            "smoke-finetune/oom",
            "CUDA OOM; automatic batch-size changes are forbidden",
        ) from exc
    _assert_fp32_cuda_model(model, stage="smoke-finetune/device")
    queries, passages = _load_text_inputs(config)
    cache, cache_report = build_token_cache(
        smoke_pairs,
        queries,
        passages,
        tokenizer,
        max_length=int(config["input"]["max_length"]),
        minimum_free_ram_gib=float(config["training"]["minimum_free_ram_gib"]),
    )
    optimizer = build_adamw_optimizer(model, learning_rate=2.0e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=2
    )
    groups = {
        str(query_id): group for query_id, group in smoke_pairs.groupby("query_id", sort=True)
    }
    order = sorted(groups)[:32]
    if len(order) < 17:
        raise StageError("smoke-finetune/data", "smoke needs at least 17 query groups")
    started = time.perf_counter()
    try:
        result = run_accumulation_epoch(
            order,
            loss_for_query=lambda query_id: model_query_loss(
                model,
                tokenizer,
                cache,
                groups[query_id],
                device="cuda",
                max_sequences_per_microbatch=int(
                    config["training"]["max_sequences_per_microbatch"]
                ),
            ),
            optimizer=optimizer,
            scheduler=scheduler,
            parameters=model.parameters(),
            accumulation=16,
            max_grad_norm=1.0,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise StageError(
            "smoke-finetune/oom",
            "CUDA OOM; automatic batch-size changes are forbidden",
        ) from exc
    if result["optimizer_steps"] != 2:
        raise StageError("smoke-finetune/training", "smoke did not execute two optimizer steps")
    validation = score_validation(
        config, model, tokenizer, device="cuda", limit_queries=16
    )
    work_root = resolve_path(config, config["paths"]["work_dir"])
    assert_not_dev(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="smoke-", dir=work_root))
    try:
        run_dir = temporary / "SMOKE"
        fingerprint = canonical_json_sha256({"smoke": expected})
        forward_batch = _forward_validation_batch(smoke_pairs, cache, tokenizer)
        generation = publish_epoch_generation(
            run_dir,
            run_id="SMOKE",
            epoch=1,
            model=model,
            tokenizer=tokenizer,
            training_fingerprint=fingerprint,
            fingerprint_components=expected,
            validation=validation,
            optimizer_steps=2,
            last_window_query_count=int(result["last_window_query_count"]),
            throughput={"pairs_per_second": len(smoke_pairs) / max(time.perf_counter() - started, 1e-12)},
            peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()),
            peak_rss=peak_rss_bytes(),
            forward_batch=forward_batch,
        )
        publish_resume_state(
            run_dir,
            run_id="SMOKE",
            completed_epoch=1,
            training_fingerprint=fingerprint,
            model_generation_sha256=str(generation["model_generation_sha256"]),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        update_latest_checkpoint(
            run_dir,
            epoch=1,
            model_generation_sha256=str(generation["model_generation_sha256"]),
            resume_state_present=True,
        )
        resumed_epoch, resumed_generation = load_resume_state(
            run_dir,
            expected_fingerprint=fingerprint,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if resumed_epoch != 1 or resumed_generation != run_dir / "epoch_1":
            raise ValueError("smoke resume-state round trip returned another generation")
        loaded = validate_epoch_generation(
            run_dir / "epoch_1", expected_fingerprint=fingerprint, forward_batch=forward_batch
        )
        zip_path = temporary / "smoke.zip"
        members = sorted(path for path in (run_dir / "epoch_1").rglob("*") if path.is_file())
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in members:
                archive.write(path, path.relative_to(run_dir / "epoch_1").as_posix())
        extraction = temporary / "extracted"
        extraction.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("smoke ZIP CRC failed")
            archive.extractall(extraction)
        for path in members:
            relative = path.relative_to(run_dir / "epoch_1")
            if sha256_file(path) != sha256_file(extraction / relative):
                raise ValueError(f"smoke ZIP hash mismatch: {relative}")
        elapsed = max(time.perf_counter() - started, 1e-12)
        report = {
            "status": "PASS",
            "real_model_forward": True,
            "real_optimizer_step": True,
            "fixture_only": False,
            **expected,
            "device": "cuda",
            "dtype": "float32",
            "pairs_processed": int(len(smoke_pairs)),
            "checkpoint_save_load_roundtrip": loaded["model_generation_sha256"]
            == generation["model_generation_sha256"],
            "resume_state_roundtrip": True,
            "zip_hash_roundtrip": True,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_rss_bytes": peak_rss_bytes(),
            "token_cache_bytes": cache_report["token_cache_bytes"],
            "token_cache_representation": cache_report[
                "token_cache_representation"
            ],
            "throughput": {
                "pairs_per_second": len(smoke_pairs) / elapsed,
                "validation_pairs_per_second": validation["validation_pairs_per_second"],
            },
        }
        smoke_path = resolve_path(config, config["audits"]["finetune_smoke"])
        resource_path = resolve_path(config, config["audits"]["resource_report"])
        assert_not_dev(smoke_path)
        assert_not_dev(resource_path)
        run_pair_counts: dict[str, int] = {}
        for estimated_run in ("A1", "A2", "B1"):
            estimated_regime = str(config["runs"][estimated_run]["regime"])
            estimated_key = {
                "judged_only": "pairs_judged_only",
                "weak_negatives": "pairs_weak_negatives",
            }[estimated_regime]
            estimated_path = resolve_path(config, config["artifacts"][estimated_key])
            assert_not_dev(estimated_path)
            run_pair_counts[estimated_run] = len(
                pd.read_parquet(estimated_path, columns=["query_id"])
            )
        validation_groups_path = resolve_path(
            config, config["artifacts"]["validation_groups"]
        )
        assert_not_dev(validation_groups_path)
        validation_pair_count = len(
            pd.read_parquet(validation_groups_path, columns=["query_id"])
        )
        def projected_seconds(run_id: str) -> float:
            run_epochs = int(config["runs"][run_id]["epochs"])
            training_seconds = run_pair_counts[run_id] * run_epochs / max(
                float(report["throughput"]["pairs_per_second"]), 1e-12
            )
            validation_seconds = validation_pair_count * run_epochs / max(
                float(report["throughput"]["validation_pairs_per_second"]), 1e-12
            )
            return training_seconds + validation_seconds
        resource = {
            "status": "PASS",
            "peak_gpu_memory_bytes": report["peak_gpu_memory_bytes"],
            "peak_rss_bytes": report["peak_rss_bytes"],
            "pairs_per_second": report["throughput"]["pairs_per_second"],
            "validation_pairs_per_second": report["throughput"]["validation_pairs_per_second"],
            "checkpoint_size_bytes": sum(path.stat().st_size for path in members),
            "token_cache_bytes": cache_report["token_cache_bytes"],
            "token_cache_representation": cache_report[
                "token_cache_representation"
            ],
            "estimated_training_time_range_seconds": {
                run_id: [
                    int(projected_seconds(run_id) * 0.8),
                    int(math.ceil(projected_seconds(run_id) * 1.4)),
                ]
                for run_id in ("A1", "A2", "B1")
            },
        }
        atomic_write_json(resource_path, resource)
        # The PASS gate is the final publication: it cannot exist without the
        # successfully materialized resource report.
        atomic_write_json(smoke_path, report)
    except Exception as exc:
        raise StageError(
            "smoke-finetune",
            str(exc),
            preserved_temporary_path=str(temporary),
        ) from exc
    shutil.rmtree(temporary)
    return report


__all__ = [
    "StageError",
    "TokenCache",
    "TRAINING_FINGERPRINT_FIELDS",
    "accumulation_windows",
    "build_adamw_optimizer",
    "build_token_cache",
    "build_training_fingerprint",
    "config_hashes",
    "disk_preflight",
    "ensure_token_cache_ram",
    "epoch_query_order",
    "estimate_token_cache_bytes",
    "evaluation_config_contract",
    "finalize_run",
    "load_resume_state",
    "model_query_loss",
    "packaging_config_contract",
    "pairwise_logistic_loss",
    "publish_epoch_generation",
    "publish_resume_state",
    "query_pairwise_loss",
    "run_accumulation_epoch",
    "run_finetune",
    "run_learning_rate",
    "score_validation",
    "scoring_config_contract",
    "seed_everything",
    "smoke_expected_fields",
    "smoke_finetune",
    "source_hashes",
    "training_config_contract",
    "training_fingerprint_components",
    "update_latest_checkpoint",
    "validate_checkpoint",
    "validate_epoch_generation",
    "validate_smoke_gate",
    "weighted_query_loss",
]
