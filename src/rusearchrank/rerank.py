"""Deterministic zero-shot cross-encoder reranking for RuSearchRank Phase 2.

The production path is deliberately independent of ``datasets``.  It consumes
the immutable Phase 1 Parquet cache, loads the pinned model only for smoke or
scoring, writes resumable query shards, and publishes an exact float32 score
table only after complete validation.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Mapping, Sequence
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any, Protocol
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .evaluation import raw_score_tie_statistics
from .retrieval import read_trec_run, validate_top_k


DEFAULT_RERANK_CONFIG = Path("configs/rerank.yaml")
MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
MODEL_TAG = "mminilmv2l12"
SCORE_COLUMNS = (
    "query_id",
    "docid",
    "score",
    "pair_tokens_before_truncation",
    "pair_tokens_after_truncation",
    "truncated",
)
SCORE_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.large_string(), nullable=False),
        pa.field("docid", pa.large_string(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
        pa.field("pair_tokens_before_truncation", pa.int32(), nullable=False),
        pa.field("pair_tokens_after_truncation", pa.int32(), nullable=False),
        pa.field("truncated", pa.bool_(), nullable=False),
    ]
)
SCORE_ENCODING = {
    "raw_model_score_field": "score",
    "raw_model_score_dtype": "float32",
    "trec_score_encoding": "rank_preserving",
    "trec_score_formula": (
        "trec_score = 1000000 - rank; rank in 1..N per query_id; "
        "written with format %.4f"
    ),
    "trec_score_is_model_score": False,
    "internal_tie_break": (
        "raw_score_desc_then_docid_asc_on_exact_equality"
    ),
}

# These exact top-level definitions are the scoring implementation contract.
# Reporting, evaluation, diagnostics, and packaging definitions are deliberately
# absent: changing them must not invalidate already-computed model logits.
SCORING_SOURCE_SYMBOLS: dict[str, tuple[str, ...]] = {
    "src/rusearchrank/rerank.py": (
        "MODEL_ID",
        "MODEL_REVISION",
        "SCORE_COLUMNS",
        "SCORE_SCHEMA",
        "PairScorer",
        "PreparedPair",
        "format_document",
        "_plain_encoding",
        "prepare_pair",
        "token_accounting",
        "resolve_device",
        "select_batch_size",
        "TransformersPairScorer",
        "seed_everything",
        "plan_query_shards",
        "_score_table_from_rows",
        "validate_score_table",
        "load_scoring_inputs",
        "_write_score_parquet",
        "_candidate_keys",
        "_sidecar_path",
        "_validate_shard_reuse",
        "_prepare_shard_pairs",
        "_score_prepared_pairs",
        "_final_score_is_valid",
        "run_rerank_scoring",
    ),
    "src/rusearchrank/cli.py": ("_rerank_score",),
}
SCORING_DEPENDENCIES = {
    "huggingface-hub",
    "numpy",
    "pandas",
    "pyarrow",
    "pyyaml",
    "torch",
    "transformers",
}
SCORING_FINGERPRINT_FIELDS = {
    "implementation_version",
    "score_schema_version",
    "scoring_source_sha256",
    "scoring_config_sha256",
    "candidates_sha256",
    "queries_sha256",
    "passages_sha256",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "max_length",
    "truncation",
    "pair_order",
    "title_separator",
    "batch_size",
    "device",
    "dtype",
    "shard_queries",
    "seed",
    "python_version",
    "torch_version",
    "transformers_version",
    "tokenizers_version",
}
NON_SCORING_PROVENANCE_FIELDS = {
    "source_tree_sha256",
    "evaluation_source_sha256",
    "config_sha256",
    "git_commit",
    "git_dirty",
}
LEGACY_FINGERPRINT_FIELDS = {
    "implementation_version",
    "score_schema_version",
    "source_tree_sha256",
    "git_commit",
    "git_dirty",
    "config_sha256",
    "candidates_sha256",
    "queries_sha256",
    "passages_sha256",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "max_length",
    "truncation",
    "pair_order",
    "title_separator",
    "batch_size",
    "device",
    "dtype",
    "shard_queries",
    "seed",
    "python_version",
    "torch_version",
    "transformers_version",
    "tokenizers_version",
}


class PairScorer(Protocol):
    """Injectable scoring contract used by the real model and offline fixtures."""

    tokenizer: Any

    def score_batch(
        self,
        encoded_pairs: Sequence[dict[str, list[int]]],
        *,
        device: str,
        dtype: str,
    ) -> tuple[np.ndarray, int]:
        """Return raw float32 logits and the padded token count."""


@dataclass(frozen=True)
class PreparedPair:
    query_id: str
    docid: str
    encoded: dict[str, list[int]]
    pair_tokens_before_truncation: int
    pair_tokens_after_truncation: int
    truncated: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(
            f"diagnostic temporary JSON already exists and was preserved: {temporary}"
        )
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
    except Exception as exc:
        raise RuntimeError(
            f"failed to write JSON atomically; diagnostic temporary file: {temporary}"
        ) from exc


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"JSON file does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {source}")
    return value


def load_rerank_config(path: str | Path = DEFAULT_RERANK_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ValueError(f"rerank config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("rerank config must contain a mapping")
    payload["_config_path"] = str(config_path)
    validate_rerank_config(payload)
    return payload


def repository_root(config: Mapping[str, Any]) -> Path:
    config_path = Path(str(config["_config_path"]))
    setting = Path(str(config.get("paths", {}).get("repository_root", ".")))
    return setting.resolve() if setting.is_absolute() else (
        config_path.parent / setting
    ).resolve()


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root(config) / path).resolve()


def portable_path(config: Mapping[str, Any], path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repository_root(config)).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must stay inside the repository: {resolved}") from exc


def _require_mapping_keys(
    config: Mapping[str, Any], section: str, required: set[str]
) -> Mapping[str, Any]:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise ValueError(f"rerank config section {section!r} must be a mapping")
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(
            f"rerank config section {section!r} is missing: {', '.join(missing)}"
        )
    return value


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def validate_rerank_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every load-bearing Phase 2 setting and repository path."""

    implementation = _require_mapping_keys(
        config, "implementation", {"version", "score_schema_version", "source_files"}
    )
    model = _require_mapping_keys(
        config,
        "model",
        {
            "id",
            "revision",
            "tokenizer_revision",
            "tag",
            "num_labels",
            "score_activation",
            "score_direction",
        },
    )
    input_config = _require_mapping_keys(
        config, "input", {"max_length", "truncation", "title_separator", "pair_order"}
    )
    inference = _require_mapping_keys(
        config,
        "inference",
        {
            "batch_size",
            "cpu_batch_size",
            "device",
            "cuda_dtype",
            "fallback_dtype",
            "score_dtype",
            "shard_queries",
            "seed",
            "minimum_available_memory_gib",
        },
    )
    protocol = _require_mapping_keys(
        config,
        "protocol",
        {
            "split",
            "official_depth",
            "diagnostic_depths",
            "internal_tie_break",
            "trec_score_encoding",
            "trec_score_base",
            "trec_score_format",
        },
    )
    inputs = _require_mapping_keys(
        config,
        "inputs",
        {
            "candidates",
            "queries",
            "passages",
            "bm25_run",
            "bm25_top1000_run",
            "qrels",
            "phase1_manifest",
        },
    )
    artifacts = _require_mapping_keys(
        config,
        "artifacts",
        {"scores", "partial_dir", "rerank_run", "diagnostic_run_template"},
    )
    metrics = _require_mapping_keys(
        config, "metrics", {"baseline", "system", "comparison", "depth_profile"}
    )
    audits = _require_mapping_keys(
        config, "audits", {"manifest", "protocol_snapshot", "smoke"}
    )
    evaluation = _require_mapping_keys(
        config,
        "evaluation",
        {
            "trec_eval_executable",
            "trec_eval_version",
            "ndcg_command",
            "recall_command",
            "per_query_command",
            "mrr_command",
            "python_vs_trec_eval_tolerance",
            "bootstrap_resamples",
            "bootstrap_seed",
            "bootstrap_confidence",
            "expected_bm25_ndcg_at_10",
            "expected_bm25_recall_at_100",
            "expected_recall_invariant",
        },
    )
    archive = _require_mapping_keys(config, "archive", {"path"})
    paths = _require_mapping_keys(config, "paths", {"repository_root", "work_dir"})

    if not re.fullmatch(r"[0-9a-f]{40}", str(model["revision"])):
        raise ValueError("model.revision must be a 40-character hexadecimal SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", str(model["tokenizer_revision"])):
        raise ValueError(
            "model.tokenizer_revision must be a 40-character hexadecimal SHA"
        )
    if model["id"] != MODEL_ID or model["revision"] != MODEL_REVISION:
        raise ValueError("Phase 2 model id and pinned revision must not change")
    if model["tokenizer_revision"] != MODEL_REVISION or model["tag"] != MODEL_TAG:
        raise ValueError("Phase 2 tokenizer revision and model tag must not change")
    if int(model["num_labels"]) != 1:
        raise ValueError("the pinned cross-encoder must expose exactly one label")
    if model["score_activation"] != "identity":
        raise ValueError("model score activation must be identity")
    if model["score_direction"] != "higher_is_more_relevant":
        raise ValueError("model score direction must be higher_is_more_relevant")

    max_length = int(input_config["max_length"])
    if not 64 <= max_length <= 512:
        raise ValueError("input.max_length must be between 64 and 512")
    if input_config["truncation"] != "only_second":
        raise ValueError("input.truncation must be only_second")
    if input_config["title_separator"] != "\n":
        raise ValueError("input.title_separator must be one newline")
    if input_config["pair_order"] != "query_document":
        raise ValueError("input.pair_order must be query_document")

    if int(protocol["official_depth"]) != 100:
        raise ValueError("protocol.official_depth must be 100")
    if [int(value) for value in protocol["diagnostic_depths"]] != [10, 20, 50]:
        raise ValueError("protocol.diagnostic_depths must be [10, 20, 50]")
    if protocol["trec_score_encoding"] != "rank_preserving":
        raise ValueError("protocol.trec_score_encoding must be rank_preserving")
    if protocol["internal_tie_break"] != "raw_score_desc_then_docid_asc":
        raise ValueError("the fixed raw-score tie-break contract changed")
    if int(protocol["trec_score_base"]) != 1_000_000:
        raise ValueError("protocol.trec_score_base must be 1000000")
    if protocol["trec_score_format"] != "%.4f":
        raise ValueError("protocol.trec_score_format must be %.4f")
    if protocol["split"] != "dev":
        raise ValueError("Phase 2 scores only the dev split")

    positive_integers = (
        "batch_size",
        "cpu_batch_size",
        "shard_queries",
        "minimum_available_memory_gib",
    )
    if any(int(inference[name]) <= 0 for name in positive_integers):
        raise ValueError("inference batch, shard, and memory values must be positive")
    if inference["cuda_dtype"] != "float16":
        raise ValueError("CUDA inference dtype must be float16")
    if inference["fallback_dtype"] != "float32" or inference["score_dtype"] != "float32":
        raise ValueError("fallback and persisted score dtypes must be float32")
    if int(implementation["score_schema_version"]) != 1:
        raise ValueError("implementation.score_schema_version must be 1")
    if not str(implementation["version"]).strip():
        raise ValueError("implementation.version must be non-empty")
    if any("round" in key.lower() for key in _walk_keys(config)):
        raise ValueError("score rounding keys are forbidden in the rerank config")

    root = repository_root(config)
    if not root.is_dir():
        raise ValueError(f"configured repository root does not exist: {root}")
    configured_values: list[str | Path] = [
        *inputs.values(),
        *artifacts.values(),
        *metrics.values(),
        *audits.values(),
        archive["path"],
        paths["work_dir"],
        *implementation["source_files"],
    ]
    for raw in configured_values:
        # The diagnostic run path is a literal portable template.
        candidate = resolve_path(config, str(raw).replace("{depth}", "10"))
        portable_path(config, candidate)

    expected_eval = {
        "ndcg_command": ["-c", "-M", "100", "-m", "ndcg_cut.10"],
        "recall_command": ["-c", "-m", "recall.100"],
        "per_query_command": ["-c", "-M", "100", "-q", "-m", "ndcg_cut.10"],
        "mrr_command": ["-c", "-M", "10", "-m", "recip_rank"],
    }
    for name, expected in expected_eval.items():
        if [str(value) for value in evaluation[name]] != expected:
            raise ValueError(f"evaluation.{name} changed from the fixed command")
    return {
        "repository_root": str(root),
        "config": portable_path(config, Path(str(config["_config_path"]))),
        "implementation_version": str(implementation["version"]),
        "score_schema_version": int(implementation["score_schema_version"]),
    }


def source_tree_sha256(config: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Hash load-bearing files using the specified path/NUL/hash/newline stream."""

    files: list[tuple[str, Path]] = []
    for value in config["implementation"]["source_files"]:
        path = resolve_path(config, str(value))
        files.append((portable_path(config, path), path))
    files.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    for relative, path in files:
        if not path.is_file():
            raise ValueError(f"load-bearing source file is missing: {relative}")
        file_hash = sha256_file(path)
        hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), hashes


def evaluation_source_sha256(
    config: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    """Hash the complete Phase 2 implementation used by evaluation/package."""

    return source_tree_sha256(config)


def scoring_config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only config values capable of changing score production."""

    inference_keys = (
        "batch_size",
        "cpu_batch_size",
        "device",
        "cuda_dtype",
        "fallback_dtype",
        "score_dtype",
        "shard_queries",
        "seed",
    )
    inference = config["inference"]
    return {
        "implementation_version": str(config["implementation"]["version"]),
        "score_schema_version": int(
            config["implementation"]["score_schema_version"]
        ),
        "model": dict(config["model"]),
        "input": dict(config["input"]),
        "inference": {name: inference[name] for name in inference_keys},
        "split": str(config["protocol"]["split"]),
    }


def scoring_config_sha256(config: Mapping[str, Any]) -> str:
    return canonical_json_sha256(scoring_config_contract(config))


def _top_level_source_fragments(
    source: str, *, relative_path: str, symbols: Sequence[str]
) -> dict[str, str]:
    """Extract exact source spans for named top-level definitions."""

    tree = ast.parse(source, filename=relative_path)
    nodes: dict[str, ast.AST] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        for name in names:
            if name in nodes:
                raise ValueError(
                    f"duplicate top-level scoring symbol {name!r} in {relative_path}"
                )
            nodes[name] = node
    missing = sorted(set(symbols).difference(nodes))
    if missing:
        raise ValueError(
            f"scoring source symbols are missing from {relative_path}: {missing}"
        )
    lines = source.splitlines(keepends=True)
    fragments: dict[str, str] = {}
    for name in symbols:
        node = nodes[name]
        start = int(getattr(node, "lineno"))
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, *(int(decorator.lineno) for decorator in decorators))
        end = int(getattr(node, "end_lineno"))
        fragments[name] = "".join(lines[start - 1 : end])
    return fragments


def _dependency_constraints(pyproject_bytes: bytes) -> list[str]:
    project = tomllib.loads(pyproject_bytes.decode("utf-8"))["project"]
    selected: list[str] = []
    for value in project.get("dependencies", []):
        dependency = str(value)
        match = re.match(r"[A-Za-z0-9_.-]+", dependency)
        if match and match.group(0).lower() in SCORING_DEPENDENCIES:
            selected.append(dependency)
    found = {
        re.match(r"[A-Za-z0-9_.-]+", value).group(0).lower()  # type: ignore[union-attr]
        for value in selected
    }
    missing = sorted(SCORING_DEPENDENCIES.difference(found))
    if missing:
        raise ValueError(f"scoring dependency constraints are missing: {missing}")
    return sorted(selected, key=str.lower)


def _current_scoring_source_bytes(config: Mapping[str, Any]) -> dict[str, bytes]:
    root = repository_root(config)
    relatives = {*SCORING_SOURCE_SYMBOLS, "pyproject.toml"}
    bundle: dict[str, bytes] = {}
    for relative in sorted(relatives):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"scoring source file is missing: {relative}")
        bundle[relative] = path.read_bytes()
    return bundle


def scoring_source_sha256(
    config: Mapping[str, Any],
    *,
    source_bytes: Mapping[str, bytes] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Hash exact scoring source fragments, config, and dependency constraints."""

    bundle = dict(source_bytes or _current_scoring_source_bytes(config))
    source_hashes: dict[str, dict[str, str]] = {}
    for relative, symbols in sorted(SCORING_SOURCE_SYMBOLS.items()):
        if relative not in bundle:
            raise ValueError(f"scoring source bundle is missing {relative}")
        fragments = _top_level_source_fragments(
            bundle[relative].decode("utf-8"),
            relative_path=relative,
            symbols=symbols,
        )
        source_hashes[relative] = {
            name: hashlib.sha256(fragment.encode("utf-8")).hexdigest()
            for name, fragment in fragments.items()
        }
    if "pyproject.toml" not in bundle:
        raise ValueError("scoring source bundle is missing pyproject.toml")
    details: dict[str, Any] = {
        "algorithm": "exact_top_level_source_fragments_v1",
        "source_symbols": source_hashes,
        "scoring_config_sha256": scoring_config_sha256(config),
        "dependency_constraints": _dependency_constraints(
            bundle["pyproject.toml"]
        ),
    }
    return canonical_json_sha256(details), details


def _git_show_bytes(root: Path, commit: str, relative: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"legacy producer git commit is invalid: {commit!r}")
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"unsafe git source path: {relative!r}")
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise ValueError(
            f"cannot read legacy producer source {relative} at {commit}: "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout


def legacy_scoring_provenance(
    config: Mapping[str, Any], *, producer_commit: str
) -> dict[str, Any]:
    """Reconstruct a legacy sidecar's scoring contract from its git commit."""

    root = repository_root(config)
    config_relative = portable_path(config, Path(str(config["_config_path"])))
    config_bytes = _git_show_bytes(root, producer_commit, config_relative)
    producer_config = yaml.safe_load(config_bytes.decode("utf-8"))
    if not isinstance(producer_config, dict):
        raise ValueError("legacy producer config is not a mapping")
    relatives = {
        *SCORING_SOURCE_SYMBOLS,
        "pyproject.toml",
        *[str(value) for value in producer_config["implementation"]["source_files"]],
    }
    bundle = {
        relative: _git_show_bytes(root, producer_commit, relative)
        for relative in sorted(relatives)
    }
    producer_scoring_hash, scoring_details = scoring_source_sha256(
        producer_config, source_bytes=bundle
    )
    legacy_digest = hashlib.sha256()
    legacy_files: dict[str, str] = {}
    for relative in sorted(
        str(value) for value in producer_config["implementation"]["source_files"]
    ):
        file_hash = hashlib.sha256(bundle[relative]).hexdigest()
        legacy_files[relative] = file_hash
        legacy_digest.update(relative.encode("utf-8"))
        legacy_digest.update(b"\0")
        legacy_digest.update(file_hash.encode("ascii"))
        legacy_digest.update(b"\n")
    return {
        "producer_commit": producer_commit,
        "producer_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "producer_scoring_source_sha256": producer_scoring_hash,
        "producer_scoring_config_sha256": scoring_details[
            "scoring_config_sha256"
        ],
        "producer_legacy_source_tree_sha256": legacy_digest.hexdigest(),
        "producer_legacy_source_files": legacy_files,
    }


def resolve_legacy_scoring_provenance(
    config: Mapping[str, Any],
    *,
    legacy_source_tree_sha256: str,
    preferred_commit: str | None = None,
) -> dict[str, Any]:
    """Find the recorded legacy source tree in reachable git history."""

    root = repository_root(config)
    candidates: list[str] = []
    if preferred_commit and re.fullmatch(r"[0-9a-f]{40}", preferred_commit):
        candidates.append(preferred_commit)
    result = subprocess.run(
        ["git", "rev-list", "--max-count=200", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise ValueError(
            "cannot inspect git history for legacy scoring provenance: "
            f"{result.stderr}"
        )
    candidates.extend(
        commit
        for commit in result.stdout.splitlines()
        if commit and commit not in candidates
    )
    failures: list[str] = []
    for commit in candidates:
        try:
            provenance = legacy_scoring_provenance(
                config, producer_commit=commit
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{commit[:12]}:{type(exc).__name__}")
            continue
        if (
            provenance["producer_legacy_source_tree_sha256"]
            == legacy_source_tree_sha256
        ):
            return provenance
    raise ValueError(
        "legacy score/smoke source tree was not found in reachable git history; "
        f"sha256={legacy_source_tree_sha256}, checked={len(candidates)}, "
        f"unreadable={failures[:5]}"
    )


def git_provenance(root: str | Path) -> tuple[str, bool]:
    """Return HEAD and dirty state; fixture directories use an explicit sentinel."""

    repository = Path(root)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE", False
    return head, bool(status.strip())


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "UNAVAILABLE"


def build_input_fingerprint(components: Mapping[str, Any]) -> str:
    missing = sorted(SCORING_FINGERPRINT_FIELDS.difference(components))
    extra = sorted(
        set(components).difference(
            SCORING_FINGERPRINT_FIELDS | NON_SCORING_PROVENANCE_FIELDS
        )
    )
    if missing or extra:
        raise ValueError(
            f"input fingerprint component mismatch: missing={missing}, extra={extra}"
        )
    scoring_components = {
        name: components[name] for name in sorted(SCORING_FINGERPRINT_FIELDS)
    }
    return canonical_json_sha256(scoring_components)


def build_legacy_input_fingerprint(components: Mapping[str, Any]) -> str:
    missing = sorted(LEGACY_FINGERPRINT_FIELDS.difference(components))
    if missing:
        raise ValueError(f"legacy input fingerprint is missing fields: {missing}")
    legacy = {name: components[name] for name in sorted(LEGACY_FINGERPRINT_FIELDS)}
    return canonical_json_sha256(legacy)


def fingerprint_components(
    config: Mapping[str, Any], *, device: str, dtype: str, batch_size: int
) -> tuple[str, dict[str, Any]]:
    scoring_hash, scoring_details = scoring_source_sha256(config)
    evaluation_hash, _ = evaluation_source_sha256(config)
    root = repository_root(config)
    git_commit, git_dirty = git_provenance(root)
    values: dict[str, Any] = {
        "implementation_version": str(config["implementation"]["version"]),
        "score_schema_version": int(config["implementation"]["score_schema_version"]),
        "scoring_source_sha256": scoring_hash,
        "scoring_config_sha256": scoring_details["scoring_config_sha256"],
        "evaluation_source_sha256": evaluation_hash,
        # Compatibility alias for pre-audit reports; it is not load-bearing for scores.
        "source_tree_sha256": evaluation_hash,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_sha256": sha256_file(Path(str(config["_config_path"]))),
        "candidates_sha256": sha256_file(resolve_path(config, config["inputs"]["candidates"])),
        "queries_sha256": sha256_file(resolve_path(config, config["inputs"]["queries"])),
        "passages_sha256": sha256_file(resolve_path(config, config["inputs"]["passages"])),
        "model_id": str(config["model"]["id"]),
        "model_revision": str(config["model"]["revision"]),
        "tokenizer_revision": str(config["model"]["tokenizer_revision"]),
        "max_length": int(config["input"]["max_length"]),
        "truncation": str(config["input"]["truncation"]),
        "pair_order": str(config["input"]["pair_order"]),
        "title_separator": str(config["input"]["title_separator"]),
        "batch_size": int(batch_size),
        "device": device,
        "dtype": dtype,
        "shard_queries": int(config["inference"]["shard_queries"]),
        "seed": int(config["inference"]["seed"]),
        "python_version": platform.python_version(),
        "torch_version": package_version("torch"),
        "transformers_version": package_version("transformers"),
        "tokenizers_version": package_version("tokenizers"),
    }
    return build_input_fingerprint(values), values


def key_set_sha256(keys: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    normalized = sorted((str(query_id), str(docid)) for query_id, docid in keys)
    for query_id, docid in normalized:
        if "\t" in query_id or "\n" in query_id or "\t" in docid or "\n" in docid:
            raise ValueError("query_id and docid must not contain tabs or newlines")
        digest.update(f"{query_id}\t{docid}\n".encode("utf-8"))
    return digest.hexdigest()


def score_schema_json(schema: pa.Schema = SCORE_SCHEMA) -> dict[str, Any]:
    return {
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ]
    }


def format_document(title: Any, text: Any, *, separator: str = "\n") -> str:
    if text is None or not isinstance(text, str):
        raise ValueError("passage text must be a string")
    if title is not None and not isinstance(title, str):
        raise ValueError("passage title must be a string or null")
    return f"{title}{separator}{text}" if title and title.strip() else text


def _plain_encoding(value: Mapping[str, Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for name, raw in value.items():
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if raw and isinstance(raw[0], list):
            if len(raw) != 1:
                raise ValueError("pair tokenizer unexpectedly returned a batch")
            raw = raw[0]
        result[str(name)] = [int(token) for token in raw]
    if "input_ids" not in result or not result["input_ids"]:
        raise ValueError("tokenizer returned no input_ids")
    return result


def prepare_pair(
    tokenizer: Any,
    *,
    query_id: str,
    docid: str,
    query_text: str,
    title: str | None,
    text: str,
    max_length: int,
    separator: str = "\n",
) -> PreparedPair:
    """Apply the exact query/document format and runtime token accounting."""

    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError(f"query {query_id!r} has empty text")
    document = format_document(title, text, separator=separator)
    encoded_full = _plain_encoding(
        tokenizer(
            query_text,
            document,
            truncation=False,
            add_special_tokens=True,
        )
    )
    before = len(encoded_full["input_ids"])
    if before <= max_length:
        encoded = encoded_full
        truncated = False
    else:
        encoded = _plain_encoding(
            tokenizer(
                query_text,
                document,
                truncation="only_second",
                max_length=max_length,
                add_special_tokens=True,
            )
        )
        truncated = True
    after = len(encoded["input_ids"])
    if before <= 0 or after <= 0 or after > max_length:
        raise ValueError(
            f"invalid token accounting for {(query_id, docid)!r}: "
            f"before={before}, after={after}, max_length={max_length}"
        )
    if truncated and after >= before:
        raise ValueError("only_second truncation did not shorten an overlength pair")
    return PreparedPair(
        query_id=str(query_id),
        docid=str(docid),
        encoded=encoded,
        pair_tokens_before_truncation=before,
        pair_tokens_after_truncation=after,
        truncated=truncated,
    )


def token_accounting(
    table: pa.Table | pd.DataFrame,
    *,
    processed_tokens: int,
    max_length: int,
) -> dict[str, Any]:
    if isinstance(table, pa.Table):
        before = np.asarray(
            table.column("pair_tokens_before_truncation").to_numpy(), dtype=np.int64
        )
        after = np.asarray(
            table.column("pair_tokens_after_truncation").to_numpy(), dtype=np.int64
        )
        truncated = np.asarray(table.column("truncated").to_numpy(), dtype=bool)
        row_count = table.num_rows
    else:
        before = table["pair_tokens_before_truncation"].to_numpy(dtype=np.int64)
        after = table["pair_tokens_after_truncation"].to_numpy(dtype=np.int64)
        truncated = table["truncated"].to_numpy(dtype=bool)
        row_count = len(table)
    sum_after = int(after.sum())
    upper_bound = int(row_count * max_length)
    if processed_tokens < sum_after:
        raise ValueError("processed token count is below the unpadded token count")
    if processed_tokens > upper_bound:
        raise ValueError(
            f"processed tokens {processed_tokens} exceed upper bound {upper_bound}"
        )
    if row_count == 0:
        raise ValueError("token accounting requires at least one pair")
    return {
        "processed_pair_count": int(row_count),
        "truncated_pair_fraction": float(truncated.mean()),
        "sum_pair_tokens_after_truncation": sum_after,
        "processed_tokens": int(processed_tokens),
        "padding_tokens": int(processed_tokens - sum_after),
        "processed_tokens_upper_bound": upper_bound,
        "processed_tokens_fraction_of_upper_bound": float(
            processed_tokens / upper_bound
        ),
        "pair_tokens_before_truncation": {
            "p50": float(np.quantile(before, 0.50)),
            "p90": float(np.quantile(before, 0.90)),
            "p95": float(np.quantile(before, 0.95)),
            "p99": float(np.quantile(before, 0.99)),
            "max": int(before.max()),
        },
    }


def available_memory_bytes() -> int | None:
    if platform.system() == "Linux":
        path = Path("/proc/meminfo")
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=True, timeout=30
            ).stdout
            page_match = re.search(r"page size of (\d+) bytes", result)
            page_size = int(page_match.group(1)) if page_match else 4096
            values = {
                name: int(value)
                for name, value in re.findall(r"^([^:]+):\s+(\d+)\.", result, re.MULTILINE)
            }
            pages = sum(
                values.get(name, 0)
                for name in (
                    "Pages free",
                    "Pages inactive",
                    "Pages speculative",
                )
            )
            return pages * page_size
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            return None
    return None


def require_available_memory(config: Mapping[str, Any]) -> dict[str, Any]:
    available = available_memory_bytes()
    required = int(config["inference"]["minimum_available_memory_gib"]) * 1024**3
    if available is None:
        raise ValueError("available memory could not be measured safely")
    if available < required:
        raise ValueError(
            f"insufficient available memory before passages read: "
            f"{available / 1024**3:.2f} GiB available, {required / 1024**3:.2f} GiB required"
        )
    return {"available_bytes": available, "required_bytes": required, "passed": True}


def peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if platform.system() == "Darwin" else observed * 1024


def resolve_device(requested: str = "auto") -> tuple[str, str]:
    import torch

    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be one of auto, cuda, mps, cpu")
    if requested == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif bool(torch.backends.mps.is_available()):
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    if device == "mps" and not bool(torch.backends.mps.is_available()):
        raise ValueError("MPS was requested but torch.backends.mps.is_available() is false")
    return device, "float16" if device == "cuda" else "float32"


def select_batch_size(
    config: Mapping[str, Any], *, device: str, override: int | None
) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError("batch size override must be positive")
        return int(override)
    key = "cpu_batch_size" if device == "cpu" else "batch_size"
    return int(config["inference"][key])


class TransformersPairScorer:
    """Pinned XLM-R sequence classifier with dynamic per-batch padding."""

    def __init__(self, config: Mapping[str, Any], *, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_id = str(config["model"]["id"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=str(config["model"]["tokenizer_revision"]),
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=str(config["model"]["revision"]),
        )
        if int(getattr(self.model.config, "num_labels", 0)) != 1:
            raise ValueError("loaded checkpoint does not expose num_labels=1")
        self.device = device
        self.model.to(device)
        self.model.eval()
        self._torch = torch

    def score_batch(
        self,
        encoded_pairs: Sequence[dict[str, list[int]]],
        *,
        device: str,
        dtype: str,
    ) -> tuple[np.ndarray, int]:
        if device != self.device:
            raise ValueError("scorer device changed after model loading")
        batch = self.tokenizer.pad(
            list(encoded_pairs),
            padding=True,
            return_tensors="pt",
        )
        padded_length = int(batch["input_ids"].shape[1])
        tensors = {name: value.to(device) for name, value in batch.items()}
        autocast = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            if device == "cuda"
            else contextlib.nullcontext()
        )
        with self._torch.inference_mode(), autocast:
            logits = self.model(**tensors).logits
        if list(logits.shape) != [len(encoded_pairs), 1]:
            raise ValueError(
                f"unexpected model logits shape {list(logits.shape)}; "
                f"expected {[len(encoded_pairs), 1]}"
            )
        if device == "cuda":
            self._torch.cuda.synchronize()
        elif device == "mps":
            self._torch.mps.synchronize()
        scores = logits.detach().to(dtype=self._torch.float32).cpu().numpy().reshape(-1)
        if scores.dtype != np.float32 or not np.isfinite(scores).all():
            raise ValueError("model scores must be finite float32 raw logits")
        return scores, len(encoded_pairs) * padded_length


def seed_everything(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plan_query_shards(query_ids: Iterable[str], shard_queries: int) -> list[tuple[str, ...]]:
    if shard_queries <= 0:
        raise ValueError("shard_queries must be positive")
    ordered = sorted({str(query_id) for query_id in query_ids})
    if not ordered:
        raise ValueError("cannot plan shards for an empty query set")
    shards = [
        tuple(ordered[offset : offset + shard_queries])
        for offset in range(0, len(ordered), shard_queries)
    ]
    flattened = [query_id for shard in shards for query_id in shard]
    if flattened != ordered or len(flattened) != len(set(flattened)):
        raise RuntimeError("deterministic query shard plan is invalid")
    return shards


def _score_table_from_rows(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist([dict(row) for row in rows], schema=SCORE_SCHEMA)


def validate_score_table(
    table_or_path: pa.Table | str | Path,
    *,
    expected_keys: set[tuple[str, str]] | None = None,
    expected_rows: int | None = None,
    max_length: int = 320,
) -> dict[str, Any]:
    table = (
        pq.read_table(table_or_path)
        if isinstance(table_or_path, (str, Path))
        else table_or_path
    )
    if not table.schema.equals(SCORE_SCHEMA, check_metadata=False):
        raise ValueError(
            "score Parquet schema mismatch: "
            f"actual={score_schema_json(table.schema)}, expected={score_schema_json()}"
        )
    if expected_rows is not None and table.num_rows != expected_rows:
        raise ValueError(
            f"score table has {table.num_rows} rows; expected {expected_rows}"
        )
    if sum(column.null_count for column in table.columns):
        raise ValueError("score table contains null values")
    query_ids = table.column("query_id").to_pylist()
    docids = table.column("docid").to_pylist()
    if any(not isinstance(value, str) or not value.strip() for value in query_ids + docids):
        raise ValueError("score identifiers must be non-empty strings")
    keys = list(zip(query_ids, docids, strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("score table contains duplicate (query_id, docid) pairs")
    actual_keys = set(keys)
    if expected_keys is not None and actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))[:10]
        extra = sorted(actual_keys.difference(expected_keys))[:10]
        raise ValueError(f"score key coverage mismatch: missing={missing}, extra={extra}")
    scores = np.asarray(table.column("score").to_numpy(), dtype=np.float32)
    before = np.asarray(
        table.column("pair_tokens_before_truncation").to_numpy(), dtype=np.int64
    )
    after = np.asarray(
        table.column("pair_tokens_after_truncation").to_numpy(), dtype=np.int64
    )
    if not np.isfinite(scores).all():
        raise ValueError("score table contains NaN or infinite scores")
    if (before <= 0).any() or (after <= 0).any():
        raise ValueError("score table token counts must be positive")
    if (after > max_length).any():
        raise ValueError("score table after-truncation count exceeds max_length")
    return {
        "row_count": table.num_rows,
        "expected_key_set_sha256": key_set_sha256(actual_keys),
        "schema_json": score_schema_json(table.schema),
    }


def _require_regular_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def validate_phase1_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind every immutable Phase 1 input to its passed manifest entry."""

    paths = {name: resolve_path(config, value) for name, value in config["inputs"].items()}
    for name, path in paths.items():
        _require_regular_file(path, f"Phase 1 input {name}")
    manifest = _read_json(paths["phase1_manifest"])
    if manifest.get("status") != "PASS":
        raise ValueError("candidate-cache manifest status must be PASS")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("candidate-cache manifest artifacts must be a list")
    by_path = {
        str(entry.get("path")): entry for entry in entries if isinstance(entry, dict)
    }
    bound_names = ("candidates", "queries", "passages", "bm25_run", "bm25_top1000_run")
    report: dict[str, Any] = {}
    for name in bound_names:
        path = paths[name]
        portable = portable_path(config, path)
        entry = by_path.get(portable)
        if entry is None:
            raise ValueError(
                f"candidate-cache manifest does not bind configured input {portable}"
            )
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if entry.get("sha256") != actual_hash or entry.get("size_bytes") != actual_size:
            raise ValueError(
                f"Phase 1 input differs from candidate-cache manifest: {portable}"
            )
        report[name] = {
            "path": portable,
            "size_bytes": actual_size,
            "sha256": actual_hash,
        }
    report["qrels"] = {
        "path": portable_path(config, paths["qrels"]),
        "size_bytes": paths["qrels"].stat().st_size,
        "sha256": sha256_file(paths["qrels"]),
    }
    report["phase1_manifest"] = {
        "path": portable_path(config, paths["phase1_manifest"]),
        "size_bytes": paths["phase1_manifest"].stat().st_size,
        "sha256": sha256_file(paths["phase1_manifest"]),
    }
    return report


def load_scoring_inputs(
    config: Mapping[str, Any], *, split: str
) -> tuple[pd.DataFrame, dict[str, str], pa.Table, dict[str, int]]:
    """Read candidates/queries and exactly one Arrow passage table plus index."""

    if split != str(config["protocol"]["split"]):
        raise ValueError("Phase 2 only supports the configured dev split")
    candidates = pd.read_parquet(resolve_path(config, config["inputs"]["candidates"]))
    required_candidate_columns = {"split", "query_id", "docid", "bm25_rank"}
    missing = sorted(required_candidate_columns.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidate table is missing columns: {', '.join(missing)}")
    candidates = candidates.loc[candidates["split"].astype("string").eq(split)].copy()
    if candidates.empty:
        raise ValueError(f"candidate table contains no rows for split {split!r}")
    candidates[["query_id", "docid"]] = candidates[["query_id", "docid"]].astype("string")
    if candidates[["query_id", "docid"]].isna().any().any():
        raise ValueError("candidate identifiers contain nulls")
    if candidates.duplicated(["query_id", "docid"], keep=False).any():
        raise ValueError("candidate table contains duplicate query-doc pairs")

    queries = pd.read_parquet(resolve_path(config, config["inputs"]["queries"]))
    required_query_columns = {"split", "query_id", "query_text"}
    missing_queries = sorted(required_query_columns.difference(queries.columns))
    if missing_queries:
        raise ValueError(f"queries table is missing columns: {', '.join(missing_queries)}")
    queries = queries.loc[queries["split"].astype("string").eq(split)]
    if queries["query_id"].astype("string").duplicated().any():
        raise ValueError("queries table contains duplicate query_id values")
    query_lookup = {
        str(row.query_id): str(row.query_text) for row in queries.itertuples(index=False)
    }
    missing_query_ids = sorted(set(candidates["query_id"]).difference(query_lookup))
    if missing_query_ids:
        raise ValueError(f"candidate queries are missing text: {missing_query_ids[:10]}")

    require_available_memory(config)
    passages = pq.read_table(
        resolve_path(config, config["inputs"]["passages"]),
        columns=["docid", "title", "text"],
    )
    if passages.schema.names != ["docid", "title", "text"]:
        raise ValueError("passage table has unexpected columns")
    passage_docids = passages.column("docid").to_pylist()
    if any(not isinstance(docid, str) or not docid.strip() for docid in passage_docids):
        raise ValueError("passage docids must be non-empty strings")
    if len(passage_docids) != len(set(passage_docids)):
        raise ValueError("passage table contains duplicate docids")
    passage_index = {docid: index for index, docid in enumerate(passage_docids)}
    missing_passages = sorted(set(candidates["docid"]).difference(passage_index))
    if missing_passages:
        raise ValueError(f"candidate passages are missing: {missing_passages[:10]}")
    return candidates, query_lookup, passages, passage_index


def _stale_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def preserve_stale(
    path: str | Path, *, destination_dir: str | Path | None = None
) -> Path | None:
    source = Path(path)
    if not source.exists():
        return None
    directory = Path(destination_dir) if destination_dir is not None else source.parent
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{source.name}.stale.{_stale_suffix()}"
    source.replace(destination)
    print(f"preserved incompatible artifact: {destination}", flush=True)
    return destination


def _write_score_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _candidate_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["query_id"].astype("string").map(str),
            frame["docid"].astype("string").map(str),
            strict=True,
        )
    )


def _sidecar_path(parquet_path: Path) -> Path:
    return parquet_path.with_name(f"{parquet_path.name}.json")


def _validate_shard_reuse(
    parquet_path: Path,
    sidecar_path: Path,
    *,
    expected_keys: set[tuple[str, str]],
    input_fingerprint: str,
    config: Mapping[str, Any],
) -> tuple[pa.Table, dict[str, Any]]:
    _require_regular_file(parquet_path, "score shard")
    sidecar = _read_json(sidecar_path)
    if sidecar.get("input_fingerprint") != input_fingerprint:
        raise ValueError("shard input_fingerprint is stale")
    if sidecar.get("implementation_version") != str(
        config["implementation"]["version"]
    ):
        raise ValueError("shard implementation_version is stale")
    if sidecar.get("score_schema_version") != int(
        config["implementation"]["score_schema_version"]
    ):
        raise ValueError("shard score_schema_version is stale")
    if sidecar.get("shard_sha256") != sha256_file(parquet_path):
        raise ValueError("shard SHA-256 differs from its sidecar")
    table = pq.read_table(parquet_path)
    validation = validate_score_table(
        table,
        expected_keys=expected_keys,
        expected_rows=len(expected_keys),
        max_length=int(config["input"]["max_length"]),
    )
    if sidecar.get("row_count") != validation["row_count"]:
        raise ValueError("shard row_count differs from its sidecar")
    if sidecar.get("expected_key_set_sha256") != validation[
        "expected_key_set_sha256"
    ]:
        raise ValueError("shard key-set hash differs from its sidecar")
    if sidecar.get("schema_json") != score_schema_json():
        raise ValueError("shard schema sidecar differs from the exact score schema")
    processed_tokens = sidecar.get("processed_tokens")
    if not isinstance(processed_tokens, int) or processed_tokens <= 0:
        raise ValueError("shard sidecar has no valid processed_tokens")
    token_accounting(
        table,
        processed_tokens=processed_tokens,
        max_length=int(config["input"]["max_length"]),
    )
    return table, sidecar


def _prepare_shard_pairs(
    candidates: pd.DataFrame,
    query_lookup: Mapping[str, str],
    passages: pa.Table,
    passage_index: Mapping[str, int],
    scorer: PairScorer,
    config: Mapping[str, Any],
) -> list[PreparedPair]:
    """Materialize passage strings only through one ``Table.take`` per shard."""

    row_indices = [passage_index[str(docid)] for docid in candidates["docid"]]
    selected = passages.take(pa.array(row_indices, type=pa.int64()))
    selected_docids = selected.column("docid").to_pylist()
    titles = selected.column("title").to_pylist()
    texts = selected.column("text").to_pylist()
    expected_docids = candidates["docid"].astype("string").map(str).tolist()
    if selected_docids != expected_docids:
        raise RuntimeError("Arrow Table.take returned passages in an unexpected order")
    prepared: list[PreparedPair] = []
    max_length = int(config["input"]["max_length"])
    separator = str(config["input"]["title_separator"])
    for row, title, text in zip(
        candidates.itertuples(index=False), titles, texts, strict=True
    ):
        prepared.append(
            prepare_pair(
                scorer.tokenizer,
                query_id=str(row.query_id),
                docid=str(row.docid),
                query_text=query_lookup[str(row.query_id)],
                title=title,
                text=text,
                max_length=max_length,
                separator=separator,
            )
        )
    # Stable sorting supplies a deterministic order even when one document is a
    # candidate for multiple queries. The specified primary/secondary key is
    # exactly (tokens after truncation, docid).
    prepared.sort(
        key=lambda pair: (pair.pair_tokens_after_truncation, pair.docid)
    )
    return prepared


def _score_prepared_pairs(
    prepared: Sequence[PreparedPair],
    *,
    scorer: PairScorer,
    device: str,
    dtype: str,
    batch_size: int,
) -> tuple[pa.Table, int]:
    rows: list[dict[str, Any]] = []
    processed_tokens = 0
    for offset in range(0, len(prepared), batch_size):
        batch = prepared[offset : offset + batch_size]
        scores, batch_tokens = scorer.score_batch(
            [pair.encoded for pair in batch], device=device, dtype=dtype
        )
        values = np.asarray(scores)
        if values.dtype != np.float32:
            raise ValueError(f"scorer returned {values.dtype}; expected float32")
        if values.shape != (len(batch),):
            raise ValueError(
                f"scorer returned shape {values.shape}; expected {(len(batch),)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("scorer returned NaN or infinite raw logits")
        if batch_tokens <= 0:
            raise ValueError("scorer returned a non-positive padded token count")
        processed_tokens += int(batch_tokens)
        for pair, score in zip(batch, values, strict=True):
            rows.append(
                {
                    "query_id": pair.query_id,
                    "docid": pair.docid,
                    "score": np.float32(score),
                    "pair_tokens_before_truncation": np.int32(
                        pair.pair_tokens_before_truncation
                    ),
                    "pair_tokens_after_truncation": np.int32(
                        pair.pair_tokens_after_truncation
                    ),
                    "truncated": pair.truncated,
                }
            )
    table = _score_table_from_rows(rows)
    return table, processed_tokens


def _final_score_is_valid(
    score_path: Path,
    sidecar_path: Path,
    *,
    expected_keys: set[tuple[str, str]],
    input_fingerprint: str,
    config: Mapping[str, Any],
) -> tuple[pa.Table, dict[str, Any]]:
    table, sidecar = _validate_shard_reuse(
        score_path,
        sidecar_path,
        expected_keys=expected_keys,
        input_fingerprint=input_fingerprint,
        config=config,
    )
    if sidecar.get("scores_sha256") != sha256_file(score_path):
        raise ValueError("final score SHA-256 differs from its sidecar")
    if not isinstance(sidecar.get("token_accounting"), dict):
        raise ValueError("final score sidecar is missing token accounting")
    return table, sidecar


def run_rerank_scoring(
    config: Mapping[str, Any],
    *,
    split: str,
    requested_device: str = "auto",
    batch_size_override: int | None = None,
    overwrite: bool = False,
    scorer: PairScorer | None = None,
) -> dict[str, Any]:
    """Score the configured split with deterministic shards and strict resume."""

    validate_rerank_config(config)
    phase1_inputs = validate_phase1_inputs(config)
    device, dtype = resolve_device(requested_device)
    batch_size = select_batch_size(
        config, device=device, override=batch_size_override
    )
    input_fingerprint, components = fingerprint_components(
        config, device=device, dtype=dtype, batch_size=batch_size
    )
    candidates, query_lookup, passages, passage_index = load_scoring_inputs(
        config, split=split
    )
    expected_keys = _candidate_keys(candidates)
    score_path = resolve_path(config, config["artifacts"]["scores"])
    final_sidecar = _sidecar_path(score_path)
    final_exists = score_path.exists() or final_sidecar.exists()
    if score_path.is_file() and final_sidecar.is_file() and not overwrite:
        try:
            _, sidecar = _final_score_is_valid(
                score_path,
                final_sidecar,
                expected_keys=expected_keys,
                input_fingerprint=input_fingerprint,
                config=config,
            )
        except (ValueError, OSError) as exc:
            raise ValueError(
                "existing final scores are stale or invalid and were preserved; "
                f"use --overwrite after review: {score_path}: {exc}"
            ) from exc
        return {
            "status": "PASS",
            "action": "reused_valid_scores",
            "scores": portable_path(config, score_path),
            "scores_sha256": sha256_file(score_path),
            "row_count": len(expected_keys),
            "input_fingerprint": input_fingerprint,
            "fingerprint_components": components,
            "token_accounting": sidecar["token_accounting"],
            "peak_rss_bytes": peak_rss_bytes(),
            "reused_shards": 0,
            "scored_shards": 0,
        }
    if final_exists and not overwrite:
        raise ValueError(
            "partial or invalid final score output was preserved; rerun with "
            f"--overwrite after review: scores={score_path.exists()}, "
            f"sidecar={final_sidecar.exists()}"
        )
    if final_exists and overwrite:
        preserve_stale(score_path)
        preserve_stale(final_sidecar)

    seed_everything(int(config["inference"]["seed"]))
    active_scorer = scorer or TransformersPairScorer(config, device=device)
    partial_dir = resolve_path(config, config["artifacts"]["partial_dir"])
    partial_dir.mkdir(parents=True, exist_ok=True)
    stale_shard_dir = partial_dir.parent / "_stale"
    query_shards = plan_query_shards(
        candidates["query_id"], int(config["inference"]["shard_queries"])
    )
    started = time.perf_counter()
    processed_pairs = 0
    processed_tokens_total = 0
    truncated_total = 0
    reused_shards = 0
    scored_shards = 0
    shard_paths: list[Path] = []
    for shard_index, query_ids in enumerate(query_shards):
        shard_number = shard_index + 1
        shard_path = partial_dir / f"shard_{shard_index:05d}.parquet"
        shard_sidecar = partial_dir / f"shard_{shard_index:05d}.json"
        shard_candidates = candidates.loc[
            candidates["query_id"].astype("string").isin(query_ids)
        ].copy()
        shard_keys = _candidate_keys(shard_candidates)
        table: pa.Table
        sidecar: dict[str, Any]
        if shard_path.exists() or shard_sidecar.exists():
            try:
                table, sidecar = _validate_shard_reuse(
                    shard_path,
                    shard_sidecar,
                    expected_keys=shard_keys,
                    input_fingerprint=input_fingerprint,
                    config=config,
                )
            except (ValueError, OSError) as exc:
                print(
                    f"shard {shard_number}/{len(query_shards)} cannot be reused: {exc}",
                    flush=True,
                )
                preserve_stale(shard_path, destination_dir=stale_shard_dir)
                preserve_stale(shard_sidecar, destination_dir=stale_shard_dir)
            else:
                reused_shards += 1
                processed_tokens_total += int(sidecar["processed_tokens"])
                processed_pairs += table.num_rows
                truncated_total += int(
                    np.asarray(table.column("truncated").to_numpy(), dtype=bool).sum()
                )
                shard_paths.append(shard_path)
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"shard {shard_number}/{len(query_shards)} reused; "
                    f"pairs={processed_pairs}/{len(candidates)}, "
                    f"pairs_per_second={processed_pairs / elapsed:.2f}, "
                    f"remaining_shards={len(query_shards) - shard_number}, "
                    f"truncated_pair_fraction={truncated_total / processed_pairs:.6f}",
                    flush=True,
                )
                continue

        prepared = _prepare_shard_pairs(
            shard_candidates,
            query_lookup,
            passages,
            passage_index,
            active_scorer,
            config,
        )
        table, shard_processed_tokens = _score_prepared_pairs(
            prepared,
            scorer=active_scorer,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        validation = validate_score_table(
            table,
            expected_keys=shard_keys,
            expected_rows=len(shard_keys),
            max_length=int(config["input"]["max_length"]),
        )
        accounting = token_accounting(
            table,
            processed_tokens=shard_processed_tokens,
            max_length=int(config["input"]["max_length"]),
        )
        temporary = shard_path.with_name(f"{shard_path.name}.tmp.{os.getpid()}")
        if temporary.exists():
            preserve_stale(temporary, destination_dir=stale_shard_dir)
        _write_score_parquet(table, temporary)
        validate_score_table(
            temporary,
            expected_keys=shard_keys,
            expected_rows=len(shard_keys),
            max_length=int(config["input"]["max_length"]),
        )
        temporary.replace(shard_path)
        sidecar = {
            "input_fingerprint": input_fingerprint,
            "fingerprint_components": components,
            "source_tree_sha256": components["source_tree_sha256"],
            "row_count": validation["row_count"],
            "expected_key_set_sha256": validation["expected_key_set_sha256"],
            "shard_sha256": sha256_file(shard_path),
            "schema_json": score_schema_json(),
            "created_at": utc_now(),
            "implementation_version": str(config["implementation"]["version"]),
            "score_schema_version": int(
                config["implementation"]["score_schema_version"]
            ),
            "processed_tokens": shard_processed_tokens,
            "token_accounting": accounting,
        }
        atomic_write_json(shard_sidecar, sidecar)
        # Sidecar publication is the last step; now prove the complete pair is reusable.
        table, sidecar = _validate_shard_reuse(
            shard_path,
            shard_sidecar,
            expected_keys=shard_keys,
            input_fingerprint=input_fingerprint,
            config=config,
        )
        scored_shards += 1
        shard_paths.append(shard_path)
        processed_tokens_total += shard_processed_tokens
        processed_pairs += table.num_rows
        truncated_total += int(
            np.asarray(table.column("truncated").to_numpy(), dtype=bool).sum()
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        print(
            f"shard {shard_number}/{len(query_shards)} scored; "
            f"pairs={processed_pairs}/{len(candidates)}, "
            f"pairs_per_second={processed_pairs / elapsed:.2f}, "
            f"remaining_shards={len(query_shards) - shard_number}, "
            f"truncated_pair_fraction={truncated_total / processed_pairs:.6f}",
            flush=True,
        )

    current_phase1 = validate_phase1_inputs(config)
    for name in ("candidates", "queries", "passages"):
        expected_hash = str(components[f"{name}_sha256"])
        if current_phase1[name]["sha256"] != expected_hash:
            raise RuntimeError(
                f"immutable Phase 1 input {name} changed during scoring; "
                "validated partial shards were preserved"
            )
    tables = [pq.read_table(path) for path in shard_paths]
    merged = pa.concat_tables(tables)
    final_validation = validate_score_table(
        merged,
        expected_keys=expected_keys,
        expected_rows=len(candidates),
        max_length=int(config["input"]["max_length"]),
    )
    final_accounting = token_accounting(
        merged,
        processed_tokens=processed_tokens_total,
        max_length=int(config["input"]["max_length"]),
    )
    score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_final = score_path.with_name(f"{score_path.name}.tmp.{os.getpid()}")
    if temporary_final.exists():
        preserve_stale(temporary_final)
    _write_score_parquet(merged, temporary_final)
    validate_score_table(
        temporary_final,
        expected_keys=expected_keys,
        expected_rows=len(candidates),
        max_length=int(config["input"]["max_length"]),
    )
    temporary_final.replace(score_path)
    rss = peak_rss_bytes()
    final_payload = {
        "status": "PASS",
        "input_fingerprint": input_fingerprint,
        "fingerprint_components": components,
        "source_tree_sha256": components["source_tree_sha256"],
        "row_count": final_validation["row_count"],
        "expected_key_set_sha256": final_validation["expected_key_set_sha256"],
        "scores_sha256": sha256_file(score_path),
        # Alias retained so final validation can use the exact shard checks too.
        "shard_sha256": sha256_file(score_path),
        "schema_json": score_schema_json(),
        "created_at": utc_now(),
        "implementation_version": str(config["implementation"]["version"]),
        "score_schema_version": int(config["implementation"]["score_schema_version"]),
        "processed_tokens": processed_tokens_total,
        "token_accounting": final_accounting,
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "peak_rss_bytes": rss,
        "phase1_inputs": phase1_inputs,
    }
    atomic_write_json(final_sidecar, final_payload)
    _final_score_is_valid(
        score_path,
        final_sidecar,
        expected_keys=expected_keys,
        input_fingerprint=input_fingerprint,
        config=config,
    )
    shutil.rmtree(partial_dir)
    return {
        "status": "PASS",
        "action": "scored",
        "scores": portable_path(config, score_path),
        "scores_sha256": sha256_file(score_path),
        "row_count": len(candidates),
        "query_count": len(query_shards) and candidates["query_id"].nunique(),
        "input_fingerprint": input_fingerprint,
        "fingerprint_components": components,
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "reused_shards": reused_shards,
        "scored_shards": scored_shards,
        "token_accounting": final_accounting,
        "peak_rss_bytes": rss,
        "seconds": time.perf_counter() - started,
    }


def derive_rerank_ranking(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    depth: int,
    official_depth: int = 100,
) -> tuple[pd.DataFrame, int]:
    """Rerank one BM25 prefix and retain the tail in exact BM25 order."""

    if depth not in {10, 20, 50, official_depth}:
        raise ValueError(f"unsupported rerank depth: {depth}")
    required_candidates = {"query_id", "docid", "bm25_rank"}
    required_scores = {"query_id", "docid", "score"}
    missing_candidates = sorted(required_candidates.difference(candidates.columns))
    missing_scores = sorted(required_scores.difference(scores.columns))
    if missing_candidates or missing_scores:
        raise ValueError(
            f"ranking columns missing: candidates={missing_candidates}, scores={missing_scores}"
        )
    base = candidates[["query_id", "docid", "bm25_rank"]].copy()
    scored = scores[["query_id", "docid", "score"]].copy()
    for frame in (base, scored):
        frame[["query_id", "docid"]] = frame[["query_id", "docid"]].astype("string")
        if frame.duplicated(["query_id", "docid"], keep=False).any():
            raise ValueError("ranking inputs contain duplicate query-doc keys")
    if _candidate_keys(base) != _candidate_keys(scored):
        raise ValueError("score keys do not exactly equal candidate keys")
    base["bm25_rank"] = pd.to_numeric(base["bm25_rank"], errors="raise").astype("int64")
    scored["score"] = pd.to_numeric(scored["score"], errors="raise").astype("float32")
    if not np.isfinite(scored["score"].to_numpy(dtype=np.float32)).all():
        raise ValueError("raw reranker scores must be finite")
    joined = base.merge(
        scored, on=["query_id", "docid"], how="inner", validate="one_to_one", sort=False
    )
    output_rows: list[dict[str, Any]] = []
    exact_ties = 0
    for query_id, group in joined.groupby("query_id", sort=True):
        if group["bm25_rank"].duplicated().any():
            raise ValueError(f"query {query_id!r} has duplicate BM25 ranks")
        observed = sorted(int(value) for value in group["bm25_rank"])
        if observed != list(range(1, len(observed) + 1)):
            raise ValueError(f"query {query_id!r} has non-contiguous BM25 ranks")
        prefix_rows = list(group.loc[group["bm25_rank"].le(depth)].itertuples(index=False))
        # Python's tuple comparison invokes docid only when the exact float32
        # values compare equal. No rounded representation enters this key.
        prefix_rows.sort(key=lambda row: (-float(np.float32(row.score)), str(row.docid)))
        counts = pd.Series(
            [np.float32(row.score).tobytes() for row in prefix_rows], dtype="object"
        ).value_counts()
        exact_ties += int(sum(int(count) - 1 for count in counts if int(count) > 1))
        tail_rows = list(
            group.loc[group["bm25_rank"].gt(depth)]
            .sort_values("bm25_rank", kind="mergesort")
            .itertuples(index=False)
        )
        for rank, row in enumerate([*prefix_rows, *tail_rows], start=1):
            output_rows.append(
                {
                    "split": "dev",
                    "query_id": str(query_id),
                    "docid": str(row.docid),
                    "rank": rank,
                    "raw_score": np.float32(row.score),
                    "bm25_rank": int(row.bm25_rank),
                }
            )
    ranking = pd.DataFrame(output_rows)
    if _candidate_keys(ranking) != _candidate_keys(base):
        raise RuntimeError("reranking changed the candidate set")
    return ranking, exact_ties


def trec_score_for_rank(rank: int, *, base: int = 1_000_000) -> float:
    if rank <= 0 or rank >= base:
        raise ValueError("rank must be positive and smaller than the TREC score base")
    return float(base - rank)


def render_rank_preserving_trec(
    ranking: pd.DataFrame,
    *,
    tag: str,
    score_base: int = 1_000_000,
    score_format: str = "%.4f",
) -> str:
    required = {"query_id", "docid", "rank"}
    missing = sorted(required.difference(ranking.columns))
    if missing:
        raise ValueError(f"ranking is missing columns: {', '.join(missing)}")
    lines: list[str] = []
    ordered = ranking.sort_values(["query_id", "rank"], kind="mergesort")
    for row in ordered.itertuples(index=False):
        score = trec_score_for_rank(int(row.rank), base=score_base)
        lines.append(
            f"{row.query_id} Q0 {row.docid} {int(row.rank)} "
            f"{score_format % score} {tag}"
        )
    return "\n".join(lines) + "\n"


def validate_rank_preserving_trec(
    path: str | Path,
    *,
    expected_keys: set[tuple[str, str]],
    expected_tag: str,
    official_depth: int = 100,
) -> dict[str, Any]:
    run = read_trec_run(str(path), split="dev")
    ranked = run.rename(columns={"source_rank": "bm25_rank"})
    validate_top_k(ranked, official_depth, require_exact=False)
    keys = _candidate_keys(run)
    if keys != expected_keys:
        missing = sorted(expected_keys.difference(keys))[:10]
        extra = sorted(keys.difference(expected_keys))[:10]
        raise ValueError(f"TREC candidate set mismatch: missing={missing}, extra={extra}")
    if set(run["tag"].astype("string")) != {expected_tag}:
        raise ValueError("TREC run tag differs from the configured tag")
    for query_id, group in run.groupby("query_id", sort=False):
        by_rank = group.sort_values("source_rank", kind="mergesort")
        scores = by_rank["bm25_score"].to_numpy(dtype=np.float64)
        if len(scores) > 1 and not np.all(scores[:-1] > scores[1:]):
            raise ValueError(f"TREC scores are not strictly decreasing for {query_id}")
        if len(set(scores.tolist())) != len(scores):
            raise ValueError(f"TREC scores are not unique for {query_id}")
        ranks = by_rank["source_rank"].astype("int64").tolist()
        expected_scores = [float(1_000_000 - rank) for rank in ranks]
        if scores.tolist() != expected_scores:
            raise ValueError("TREC score does not exactly encode 1000000 - rank")
        rank_docids = by_rank["docid"].astype("string").tolist()
        doc_asc = group.sort_values(
            ["bm25_score", "docid"],
            ascending=[False, True],
            kind="mergesort",
        )["docid"].astype("string").tolist()
        doc_desc = group.sort_values(
            ["bm25_score", "docid"],
            ascending=[False, False],
            kind="mergesort",
        )["docid"].astype("string").tolist()
        if rank_docids != doc_asc or rank_docids != doc_desc:
            raise ValueError(
                "rank-preserving proof failed: score/docno ASC and DESC orders differ"
            )
    return {
        "row_count": int(len(run)),
        "query_count": int(run["query_id"].nunique()),
        "candidate_set_invariant": True,
        "strictly_decreasing_scores": True,
        "unique_scores_per_query": True,
        "docno_asc_desc_tie_break_independent": True,
    }


def rerank_run_path(config: Mapping[str, Any], depth: int) -> Path:
    if depth == int(config["protocol"]["official_depth"]):
        return resolve_path(config, config["artifacts"]["rerank_run"])
    template = str(config["artifacts"]["diagnostic_run_template"])
    return resolve_path(config, template.format(depth=depth))


def build_rerank_run(
    config: Mapping[str, Any],
    *,
    split: str,
    depth: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    validate_rerank_config(config)
    validate_phase1_inputs(config)
    if split != "dev":
        raise ValueError("Phase 2 run construction only supports dev")
    allowed_depths = {
        int(config["protocol"]["official_depth"]),
        *[int(value) for value in config["protocol"]["diagnostic_depths"]],
    }
    if depth not in allowed_depths:
        raise ValueError(f"depth must be one of {sorted(allowed_depths)}")
    candidates = pd.read_parquet(resolve_path(config, config["inputs"]["candidates"]))
    candidates = candidates.loc[candidates["split"].astype("string").eq(split)].copy()
    score_path = resolve_path(config, config["artifacts"]["scores"])
    score_sidecar_path = _sidecar_path(score_path)
    expected_keys = _candidate_keys(candidates)
    _require_regular_file(score_path, "final score Parquet")
    _require_regular_file(score_sidecar_path, "final score sidecar")
    score_table = pq.read_table(score_path)
    validate_score_table(
        score_table,
        expected_keys=expected_keys,
        expected_rows=len(candidates),
        max_length=int(config["input"]["max_length"]),
    )
    score_sidecar = validate_current_score_sidecar(config, score_path=score_path)
    scores = score_table.select(["query_id", "docid", "score"]).to_pandas()
    ranking, tie_count = derive_rerank_ranking(
        candidates,
        scores,
        depth=depth,
        official_depth=int(config["protocol"]["official_depth"]),
    )
    tag = f"rusearchrank-rerank-{config['model']['tag']}-k{depth}"
    rendered = render_rank_preserving_trec(
        ranking,
        tag=tag,
        score_base=int(config["protocol"]["trec_score_base"]),
        score_format=str(config["protocol"]["trec_score_format"]),
    )
    destination = rerank_run_path(config, depth)
    if destination.exists() and not overwrite:
        try:
            validation = validate_rank_preserving_trec(
                destination,
                expected_keys=expected_keys,
                expected_tag=tag,
                official_depth=int(config["protocol"]["official_depth"]),
            )
        except (ValueError, OSError) as exc:
            raise ValueError(
                f"existing rerank run is invalid and was preserved: {destination}: {exc}; "
                "use --overwrite after review"
            ) from exc
        if destination.read_text(encoding="utf-8") != rendered:
            raise ValueError(
                f"existing rerank run is stale and was preserved: {destination}; "
                "use --overwrite after review"
            )
        return {
            "status": "PASS",
            "action": "reused_valid_run",
            "depth": depth,
            "official": depth == int(config["protocol"]["official_depth"]),
            "run": portable_path(config, destination),
            "run_sha256": sha256_file(destination),
            "raw_score_ties": tie_count,
            "validation": validation,
            "score_encoding": SCORE_ENCODING,
        }
    if destination.exists() and overwrite:
        preserve_stale(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        preserve_stale(temporary)
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    validation = validate_rank_preserving_trec(
        temporary,
        expected_keys=expected_keys,
        expected_tag=tag,
        official_depth=int(config["protocol"]["official_depth"]),
    )
    temporary.replace(destination)
    return {
        "status": "PASS",
        "action": "created_run",
        "depth": depth,
        "official": depth == int(config["protocol"]["official_depth"]),
        "run": portable_path(config, destination),
        "run_sha256": sha256_file(destination),
        "raw_score_ties": tie_count,
        "validation": validation,
        "score_encoding": SCORE_ENCODING,
    }


def resolve_trec_eval(config: Mapping[str, Any]) -> Path:
    configured = str(config["evaluation"]["trec_eval_executable"])
    if Path(configured).is_absolute() or "/" in configured:
        executable = resolve_path(config, configured)
    else:
        located = shutil.which(configured)
        if located is None:
            raise ValueError("official NIST trec_eval executable was not found on PATH")
        executable = Path(located).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"trec_eval is not an executable file: {executable}")
    return executable


def parse_trec_eval_version(stdout: str, stderr: str = "") -> str | None:
    combined = "\n".join(part for part in (stdout, stderr) if part)
    patterns = (
        r"\btrec_eval(?:\s+(?:version|v))?\s*[:=]?\s*v?(\d+\.\d+\.\d+)\b",
        r"^\s*v?(\d+\.\d+\.\d+)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)
    return None


def probe_trec_eval(executable: Path, *, expected_version: str) -> dict[str, Any]:
    command = [str(executable), "-v"]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    parsed_version = parse_trec_eval_version(result.stdout, result.stderr)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "parsed_version": parsed_version,
        "expected_version": str(expected_version),
        "version_matches_expected": (
            result.returncode == 0 and parsed_version == str(expected_version)
        ),
    }


def require_trec_eval_version(probe: Mapping[str, Any]) -> None:
    if int(probe.get("returncode", -1)) != 0:
        raise ValueError(
            "trec_eval -v failed before official evaluation: "
            f"returncode={probe.get('returncode')}, stdout={probe.get('stdout')!r}, "
            f"stderr={probe.get('stderr')!r}"
        )
    if probe.get("parsed_version") is None:
        raise ValueError(
            "trec_eval -v output did not contain a recognizable semantic version: "
            f"stdout={probe.get('stdout')!r}, stderr={probe.get('stderr')!r}"
        )
    if probe.get("version_matches_expected") is not True:
        raise ValueError(
            "trec_eval version differs from the production protocol: "
            f"parsed={probe.get('parsed_version')!r}, "
            f"expected={probe.get('expected_version')!r}"
        )


def resolve_model_revision(
    config: Mapping[str, Any], *, api: Any | None = None
) -> dict[str, Any]:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    requested = str(config["model"]["revision"])
    try:
        info = api.model_info(str(config["model"]["id"]), revision=requested)
    except Exception as exc:
        raise ValueError(
            f"pinned model revision is unavailable: {config['model']['id']}@{requested}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    resolved = str(getattr(info, "sha", "") or "")
    if resolved != requested:
        raise ValueError(
            f"pinned model revision {requested} resolved to unexpected SHA {resolved}"
        )
    return {
        "model_id": str(config["model"]["id"]),
        "requested_revision": requested,
        "resolved_revision": resolved,
        "weights_downloaded": False,
    }


def preflight_rerank(
    config: Mapping[str, Any], *, model_api: Any | None = None
) -> dict[str, Any]:
    contract = validate_rerank_config(config)
    phase1 = validate_phase1_inputs(config)
    model = resolve_model_revision(config, api=model_api)
    memory = require_available_memory(config)
    executable = resolve_trec_eval(config)
    expected_trec_version = str(config["evaluation"]["trec_eval_version"])
    trec_probe = probe_trec_eval(
        executable, expected_version=expected_trec_version
    )
    require_trec_eval_version(trec_probe)
    scoring_hash, scoring_details = scoring_source_sha256(config)
    evaluation_hash, source_files = evaluation_source_sha256(config)
    root = repository_root(config)
    disk = shutil.disk_usage(root)
    candidates_path = resolve_path(config, config["inputs"]["candidates"])
    required_disk = max(512 * 1024**2, candidates_path.stat().st_size * 6)
    if disk.free < required_disk:
        raise ValueError(
            f"insufficient free disk for Phase 2 temporaries: {disk.free} bytes "
            f"available, {required_disk} required"
        )
    return {
        "stage": "rerank",
        "config_contract": contract,
        "phase1_inputs": phase1,
        "model": model,
        "qrels_present": True,
        "trec_eval": {
            "path": str(executable),
            "expected_version": expected_trec_version,
            "probe": trec_probe,
        },
        "memory": memory,
        "disk": {
            "free_bytes": disk.free,
            "required_bytes": required_disk,
            "passed": True,
        },
        "scoring_source_sha256": scoring_hash,
        "scoring_config_sha256": scoring_details["scoring_config_sha256"],
        "evaluation_source_sha256": evaluation_hash,
        "source_tree_sha256": evaluation_hash,
        "source_files": source_files,
        "weights_downloaded": False,
    }


def scoring_expected_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    scoring_hash, scoring_details = scoring_source_sha256(config)
    return {
        "model_id": str(config["model"]["id"]),
        "model_revision": str(config["model"]["revision"]),
        "tokenizer_revision": str(config["model"]["tokenizer_revision"]),
        "implementation_version": str(config["implementation"]["version"]),
        "score_schema_version": int(config["implementation"]["score_schema_version"]),
        "scoring_source_sha256": scoring_hash,
        "scoring_config_sha256": scoring_details["scoring_config_sha256"],
        "candidates_sha256": sha256_file(resolve_path(config, config["inputs"]["candidates"])),
        "queries_sha256": sha256_file(resolve_path(config, config["inputs"]["queries"])),
        "passages_sha256": sha256_file(resolve_path(config, config["inputs"]["passages"])),
        "max_length": int(config["input"]["max_length"]),
        "truncation": str(config["input"]["truncation"]),
        "pair_order": str(config["input"]["pair_order"]),
        "title_separator": str(config["input"]["title_separator"]),
    }


def smoke_expected_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation_hash, _ = evaluation_source_sha256(config)
    return {
        **scoring_expected_fields(config),
        "config_sha256": sha256_file(Path(str(config["_config_path"]))),
        "evaluation_source_sha256": evaluation_hash,
        # Backward-compatible full-tree alias; smoke validation does not use it
        # to decide whether logits may be reused.
        "source_tree_sha256": evaluation_hash,
    }


def validate_current_score_sidecar(
    config: Mapping[str, Any],
    *,
    score_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate score bytes/coverage and only scoring-relevant provenance."""

    scores = (
        Path(score_path)
        if score_path is not None
        else resolve_path(config, config["artifacts"]["scores"])
    )
    sidecar_path = _sidecar_path(scores)
    _require_regular_file(scores, "final score Parquet")
    candidates = pd.read_parquet(
        resolve_path(config, config["inputs"]["candidates"]),
        columns=["split", "query_id", "docid"],
    )
    candidates = candidates.loc[
        candidates["split"].astype("string").eq(str(config["protocol"]["split"]))
    ]
    expected_keys = _candidate_keys(candidates)
    validate_score_table(
        scores,
        expected_keys=expected_keys,
        expected_rows=len(candidates),
        max_length=int(config["input"]["max_length"]),
    )
    sidecar = _read_json(sidecar_path)
    if sidecar.get("status") != "PASS":
        raise ValueError("final score sidecar status must be PASS")
    if sidecar.get("scores_sha256") != sha256_file(scores):
        raise ValueError("final score Parquet differs from its sidecar SHA-256")
    components = sidecar.get("fingerprint_components")
    if not isinstance(components, Mapping):
        raise ValueError("final score sidecar has no fingerprint components")
    expected = scoring_expected_fields(config)
    component_names = {
        "model_id": "model_id",
        "model_revision": "model_revision",
        "tokenizer_revision": "tokenizer_revision",
        "implementation_version": "implementation_version",
        "score_schema_version": "score_schema_version",
        "candidates_sha256": "candidates_sha256",
        "queries_sha256": "queries_sha256",
        "passages_sha256": "passages_sha256",
        "max_length": "max_length",
        "truncation": "truncation",
        "pair_order": "pair_order",
        "title_separator": "title_separator",
    }
    mismatches = {
        name: {"expected": expected[expected_name], "actual": components.get(name)}
        for name, expected_name in component_names.items()
        if components.get(name) != expected[expected_name]
    }
    migration: dict[str, Any]
    if "scoring_source_sha256" in components:
        for name in ("scoring_source_sha256", "scoring_config_sha256"):
            if components.get(name) != expected[name]:
                mismatches[name] = {
                    "expected": expected[name],
                    "actual": components.get(name),
                }
        computed_fingerprint = build_input_fingerprint(components)
        if sidecar.get("input_fingerprint") != computed_fingerprint:
            mismatches["input_fingerprint"] = {
                "expected": computed_fingerprint,
                "actual": sidecar.get("input_fingerprint"),
            }
        migration = {"mode": "native_split_provenance"}
    else:
        legacy_fingerprint = build_legacy_input_fingerprint(components)
        if sidecar.get("input_fingerprint") != legacy_fingerprint:
            mismatches["legacy_input_fingerprint"] = {
                "expected": legacy_fingerprint,
                "actual": sidecar.get("input_fingerprint"),
            }
        legacy_tree = str(components.get("source_tree_sha256", ""))
        if sidecar.get("source_tree_sha256") != legacy_tree:
            mismatches["legacy_sidecar.source_tree_sha256"] = {
                "expected": legacy_tree,
                "actual": sidecar.get("source_tree_sha256"),
            }
        provenance = resolve_legacy_scoring_provenance(
            config,
            legacy_source_tree_sha256=legacy_tree,
            preferred_commit=str(components.get("git_commit", "")),
        )
        legacy_checks = {
            "config_sha256": (
                components.get("config_sha256"),
                provenance["producer_config_sha256"],
            ),
            "producer_scoring_source_sha256": (
                provenance["producer_scoring_source_sha256"],
                expected["scoring_source_sha256"],
            ),
            "producer_scoring_config_sha256": (
                provenance["producer_scoring_config_sha256"],
                expected["scoring_config_sha256"],
            ),
        }
        for name, (actual, wanted) in legacy_checks.items():
            if actual != wanted:
                mismatches[name] = {"expected": wanted, "actual": actual}
        migration = {"mode": "verified_legacy_git_source", **provenance}
    if mismatches:
        raise ValueError(
            "final scores are stale for the current protocol: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    evaluation_hash, _ = evaluation_source_sha256(config)
    normalized = dict(sidecar)
    normalized["producer_source_tree_sha256"] = sidecar.get(
        "source_tree_sha256"
    )
    normalized["scoring_source_sha256"] = expected["scoring_source_sha256"]
    normalized["scoring_config_sha256"] = expected["scoring_config_sha256"]
    normalized["evaluation_source_sha256"] = evaluation_hash
    normalized["source_tree_sha256"] = evaluation_hash
    normalized["provenance_migration"] = migration
    return normalized


def validate_smoke_gate(
    config: Mapping[str, Any], *, report_path: str | Path | None = None
) -> dict[str, Any]:
    path = (
        Path(report_path).resolve()
        if report_path is not None
        else resolve_path(config, config["audits"]["smoke"])
    )
    repeat = (
        "python -m rusearchrank.cli smoke-rerank --config configs/rerank.yaml "
        "--limit 64"
    )
    if not path.is_file():
        raise ValueError(f"real rerank smoke report is missing; repeat: {repeat}")
    report = _read_json(path)
    if report.get("status") != "PASS":
        raise ValueError(f"rerank smoke status is not PASS; repeat: {repeat}")
    if report.get("real_model_forward") is not True:
        raise ValueError(f"rerank smoke did not execute a real model forward; repeat: {repeat}")
    if report.get("fixture_only") is not False:
        raise ValueError(f"fixture-only smoke cannot unlock full scoring; repeat: {repeat}")
    expected = scoring_expected_fields(config)
    common_names = set(expected).difference(
        {"scoring_source_sha256", "scoring_config_sha256"}
    )
    mismatches = {
        name: {"expected": expected[name], "actual": report.get(name)}
        for name in sorted(common_names)
        if report.get(name) != expected[name]
    }
    migration: dict[str, Any]
    if "scoring_source_sha256" in report:
        for name in ("scoring_source_sha256", "scoring_config_sha256"):
            if report.get(name) != expected[name]:
                mismatches[name] = {
                    "expected": expected[name],
                    "actual": report.get(name),
                }
        migration = {"mode": "native_split_provenance"}
    else:
        legacy_tree = str(report.get("source_tree_sha256", ""))
        provenance = resolve_legacy_scoring_provenance(
            config,
            legacy_source_tree_sha256=legacy_tree,
        )
        legacy_checks = {
            "config_sha256": (
                report.get("config_sha256"),
                provenance["producer_config_sha256"],
            ),
            "producer_scoring_source_sha256": (
                provenance["producer_scoring_source_sha256"],
                expected["scoring_source_sha256"],
            ),
            "producer_scoring_config_sha256": (
                provenance["producer_scoring_config_sha256"],
                expected["scoring_config_sha256"],
            ),
        }
        for name, (actual, wanted) in legacy_checks.items():
            if actual != wanted:
                mismatches[name] = {"expected": wanted, "actual": actual}
        migration = {"mode": "verified_legacy_git_source", **provenance}
    if mismatches:
        raise ValueError(
            "rerank smoke is stale or incompatible: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            + f"; repeat: {repeat}"
        )
    normalized = dict(report)
    normalized["scoring_source_sha256"] = expected["scoring_source_sha256"]
    normalized["scoring_config_sha256"] = expected["scoring_config_sha256"]
    normalized["provenance_migration"] = migration
    return normalized


def run_smoke_rerank(
    config: Mapping[str, Any],
    *,
    limit: int = 64,
    requested_device: str = "auto",
    output: str | Path | None = None,
    scorer: PairScorer | None = None,
    model_api: Any | None = None,
) -> dict[str, Any]:
    """Run a real, cheap model/Parquet/TREC/ZIP/hash round trip."""

    if limit <= 1:
        raise ValueError("smoke limit must be greater than one")
    validate_rerank_config(config)
    phase1 = validate_phase1_inputs(config)
    model_resolution = resolve_model_revision(config, api=model_api)
    device, dtype = resolve_device(requested_device)
    batch_size = min(
        limit, select_batch_size(config, device=device, override=None)
    )
    work_dir = resolve_path(config, config["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix="rusearchrank-rerank-smoke-", dir=work_dir)
    )
    cleaned = False
    started = utc_now()
    try:
        candidates, query_lookup, passages, passage_index = load_scoring_inputs(
            config, split="dev"
        )
        sample = candidates.iloc[:limit].copy()
        if len(sample) != limit:
            raise ValueError(
                f"smoke requested {limit} real pairs but only {len(sample)} are available"
            )
        seed_everything(int(config["inference"]["seed"]))
        injected_fixture = scorer is not None
        active_scorer = scorer or TransformersPairScorer(config, device=device)
        prepared = _prepare_shard_pairs(
            sample,
            query_lookup,
            passages,
            passage_index,
            active_scorer,
            config,
        )
        table, processed_tokens = _score_prepared_pairs(
            prepared,
            scorer=active_scorer,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        expected_keys = _candidate_keys(sample)
        validate_score_table(
            table,
            expected_keys=expected_keys,
            expected_rows=limit,
            max_length=int(config["input"]["max_length"]),
        )
        score_values = np.asarray(table.column("score").to_numpy(), dtype=np.float32)
        if np.unique(score_values).size <= 1:
            raise ValueError("smoke model scores are all identical")
        accounting = token_accounting(
            table,
            processed_tokens=processed_tokens,
            max_length=int(config["input"]["max_length"]),
        )

        score_path = temporary_root / "smoke_scores.parquet"
        _write_score_parquet(table, score_path)
        parquet_validation = validate_score_table(
            score_path,
            expected_keys=expected_keys,
            expected_rows=limit,
            max_length=int(config["input"]["max_length"]),
        )
        scores = table.select(["query_id", "docid", "score"]).to_pandas()
        ranking, tie_count = derive_rerank_ranking(
            sample,
            scores,
            depth=100,
            official_depth=100,
        )
        tag = f"rusearchrank-rerank-{config['model']['tag']}-k100"
        run_path = temporary_root / "smoke_run.trec"
        run_path.write_text(
            render_rank_preserving_trec(ranking, tag=tag), encoding="utf-8"
        )
        trec_validation = validate_rank_preserving_trec(
            run_path,
            expected_keys=expected_keys,
            expected_tag=tag,
            official_depth=100,
        )

        manifest_path = temporary_root / "smoke_manifest.json"
        manifest = {
            "status": "PASS",
            "created_at": started,
            "files": [
                {
                    "path": score_path.name,
                    "size_bytes": score_path.stat().st_size,
                    "sha256": sha256_file(score_path),
                    "row_count": table.num_rows,
                    "schema": score_schema_json(),
                },
                {
                    "path": run_path.name,
                    "size_bytes": run_path.stat().st_size,
                    "sha256": sha256_file(run_path),
                    "row_count": limit,
                    "schema": "TREC six-column rank-preserving run",
                },
            ],
        }
        atomic_write_json(manifest_path, manifest)
        archive_path = temporary_root / "smoke_results.zip"
        members = [score_path.name, run_path.name, manifest_path.name]
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in (score_path, run_path, manifest_path):
                archive.write(path, arcname=path.name)
        extraction_root = temporary_root / "extracted"
        with zipfile.ZipFile(archive_path) as archive:
            if archive.namelist() != members:
                raise ValueError("smoke ZIP member order differs from its allowlist")
            if archive.testzip() is not None:
                raise ValueError("smoke ZIP failed CRC validation")
            for info in archive.infolist():
                member = Path(info.filename)
                if member.is_absolute() or ".." in member.parts or info.is_dir():
                    raise ValueError(f"unsafe smoke ZIP member: {info.filename}")
            archive.extractall(extraction_root)
        extracted_manifest = _read_json(extraction_root / manifest_path.name)
        if extracted_manifest != manifest:
            raise ValueError("smoke manifest changed after ZIP extraction")
        for entry in manifest["files"]:
            extracted = extraction_root / str(entry["path"])
            if extracted.stat().st_size != entry["size_bytes"]:
                raise ValueError("smoke extracted file size mismatch")
            if sha256_file(extracted) != entry["sha256"]:
                raise ValueError("smoke extracted file hash mismatch")
        validate_score_table(
            extraction_root / score_path.name,
            expected_keys=expected_keys,
            expected_rows=limit,
            max_length=int(config["input"]["max_length"]),
        )
        validate_rank_preserving_trec(
            extraction_root / run_path.name,
            expected_keys=expected_keys,
            expected_tag=tag,
            official_depth=100,
        )

        expected = smoke_expected_fields(config)
        payload = {
            "status": "PASS",
            "real_model_forward": not injected_fixture,
            "fixture_only": injected_fixture,
            **expected,
            "resolved_model_revision": model_resolution["resolved_revision"],
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "processed_pair_count": limit,
            "logits_shape": [limit, 1],
            "scores_finite": True,
            "scores_not_all_identical": True,
            "score_ties": tie_count,
            "token_accounting": accounting,
            "phase1_inputs": phase1,
            "checks": {
                "parquet_round_trip": {"status": "PASS", **parquet_validation},
                "trec_rank_preserving_round_trip": {
                    "status": "PASS",
                    **trec_validation,
                },
                "zip_exact_allowlist": {"status": "PASS", "members": members},
                "zip_crc": {"status": "PASS"},
                "hash_round_trip": {"status": "PASS", "files": len(manifest["files"])},
            },
            "production_artifacts_written": [],
            "temporary_root": str(temporary_root),
            "started_at": started,
            "finished_at": utc_now(),
        }
        output_path = (
            resolve_path(config, output)
            if output is not None
            else resolve_path(config, config["audits"]["smoke"])
        )
        atomic_write_json(output_path, payload)
        shutil.rmtree(temporary_root)
        cleaned = True
        return {**payload, "report": portable_path(config, output_path)}
    finally:
        if not cleaned and temporary_root.is_dir():
            print(
                f"rerank smoke failed; diagnostic files preserved at {temporary_root}",
                file=sys.stderr,
            )


def phase2_payload_paths(config: Mapping[str, Any]) -> list[Path]:
    return [
        resolve_path(config, config["artifacts"]["scores"]),
        resolve_path(config, config["artifacts"]["rerank_run"]),
        resolve_path(config, config["metrics"]["baseline"]),
        resolve_path(config, config["metrics"]["system"]),
        resolve_path(config, config["metrics"]["comparison"]),
        resolve_path(config, config["metrics"]["depth_profile"]),
        resolve_path(config, config["audits"]["protocol_snapshot"]),
    ]


def artifact_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    _require_regular_file(source, "Phase 2 payload")
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        table = pq.read_table(source)
        return {
            "row_count": table.num_rows,
            "schema": score_schema_json(table.schema),
        }
    if suffix == ".trec":
        with source.open("r", encoding="utf-8") as stream:
            rows = sum(bool(line.strip()) for line in stream)
        return {
            "row_count": int(rows),
            "schema": {
                "query_id": "string",
                "q0": "string",
                "docid": "string",
                "rank": "int64",
                "score": "float64",
                "tag": "string",
            },
        }
    if suffix == ".json":
        _read_json(source)
        return {"row_count": None, "schema": "json_object"}
    if suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"protocol snapshot is not a YAML mapping: {source}")
        return {"row_count": None, "schema": "yaml_mapping_byte_snapshot"}
    raise ValueError(f"unsupported Phase 2 payload type: {source}")


def _phase2_input_hashes(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    payloads = phase2_payload_paths(config)
    names = [portable_path(config, path) for path in payloads]
    score, run, baseline, system, comparison, depth_profile, protocol = payloads
    configured_inputs = {
        name: resolve_path(config, value) for name, value in config["inputs"].items()
    }
    diagnostic_runs = {
        depth: rerank_run_path(config, depth)
        for depth in [
            *[int(value) for value in config["protocol"]["diagnostic_depths"]],
            int(config["protocol"]["official_depth"]),
        ]
    }
    return {
        names[0]: {
            "candidates": sha256_file(configured_inputs["candidates"]),
            "queries": sha256_file(configured_inputs["queries"]),
            "passages": sha256_file(configured_inputs["passages"]),
            "config": sha256_file(Path(str(config["_config_path"]))),
        },
        names[1]: {
            "scores": sha256_file(score),
            "candidates": sha256_file(configured_inputs["candidates"]),
        },
        names[2]: {
            "bm25_run": sha256_file(configured_inputs["bm25_run"]),
            "qrels": sha256_file(configured_inputs["qrels"]),
            "candidates": sha256_file(configured_inputs["candidates"]),
        },
        names[3]: {
            "rerank_run": sha256_file(run),
            "qrels": sha256_file(configured_inputs["qrels"]),
            "scores": sha256_file(score),
            "candidates": sha256_file(configured_inputs["candidates"]),
        },
        names[4]: {
            "baseline_metrics": sha256_file(baseline),
            "system_metrics": sha256_file(system),
        },
        names[5]: {
            "baseline_metrics": sha256_file(baseline),
            **{
                f"rerank_run_k{run_depth}": sha256_file(path)
                for run_depth, path in diagnostic_runs.items()
            },
            "candidates": sha256_file(configured_inputs["candidates"]),
            "qrels": sha256_file(configured_inputs["qrels"]),
        },
        names[6]: {"config": sha256_file(Path(str(config["_config_path"])))},
    }


def _manifest_entry(
    config: Mapping[str, Any],
    path: Path,
    *,
    score_sidecar: Mapping[str, Any],
    raw_score_ties: Mapping[str, Any],
    producer_command: str,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    components = score_sidecar.get("fingerprint_components")
    if not isinstance(components, Mapping):
        raise ValueError("score sidecar is missing fingerprint components")
    contract = artifact_contract(path)
    return {
        "path": portable_path(config, path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **contract,
        "producer_command": producer_command,
        "input_hashes": dict(input_hashes),
        "model_id": str(config["model"]["id"]),
        "model_revision": str(config["model"]["revision"]),
        "tokenizer_revision": str(config["model"]["tokenizer_revision"]),
        "max_length": int(config["input"]["max_length"]),
        "truncation": str(config["input"]["truncation"]),
        "device": str(score_sidecar["device"]),
        "dtype": str(score_sidecar["dtype"]),
        "batch_size": int(score_sidecar["batch_size"]),
        "torch_version": str(components["torch_version"]),
        "transformers_version": str(components["transformers_version"]),
        "tokenizers_version": str(components["tokenizers_version"]),
        "python_version": str(components["python_version"]),
        "platform": platform.platform(),
        "implementation_version": str(config["implementation"]["version"]),
        "score_schema_version": int(config["implementation"]["score_schema_version"]),
        "scoring_source_sha256": str(
            score_sidecar["scoring_source_sha256"]
        ),
        "scoring_config_sha256": str(
            score_sidecar["scoring_config_sha256"]
        ),
        "evaluation_source_sha256": str(
            score_sidecar["evaluation_source_sha256"]
        ),
        "source_tree_sha256": str(score_sidecar["evaluation_source_sha256"]),
        "git_commit": str(components["git_commit"]),
        "git_dirty": bool(components["git_dirty"]),
        "producer_config_sha256": str(components["config_sha256"]),
        "config_sha256": sha256_file(Path(str(config["_config_path"]))),
        "input_fingerprint": str(score_sidecar["input_fingerprint"]),
        "peak_rss_bytes": int(score_sidecar["peak_rss_bytes"]),
        "score_encoding": dict(SCORE_ENCODING),
        "token_accounting": dict(score_sidecar["token_accounting"]),
        **dict(raw_score_ties),
    }


def validate_phase2_manifest(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_payloads: Sequence[Path],
) -> None:
    if manifest.get("status") != "PASS":
        raise ValueError("rerank manifest status must be PASS")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("rerank manifest files must be a list")
    manifest_name = portable_path(
        config, resolve_path(config, config["audits"]["manifest"])
    )
    if any(isinstance(entry, dict) and entry.get("path") == manifest_name for entry in files):
        raise ValueError("rerank manifest must not list itself")
    expected_names = [portable_path(config, path) for path in expected_payloads]
    actual_names = [
        str(entry.get("path")) for entry in files if isinstance(entry, dict)
    ]
    if actual_names != expected_names or len(actual_names) != len(files):
        raise ValueError("rerank manifest paths differ from the ordered payload allowlist")
    required = {
        "path",
        "size_bytes",
        "sha256",
        "row_count",
        "schema",
        "producer_command",
        "input_hashes",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "max_length",
        "truncation",
        "device",
        "dtype",
        "batch_size",
        "torch_version",
        "transformers_version",
        "tokenizers_version",
        "python_version",
        "platform",
        "implementation_version",
        "score_schema_version",
        "scoring_source_sha256",
        "scoring_config_sha256",
        "evaluation_source_sha256",
        "source_tree_sha256",
        "git_commit",
        "git_dirty",
        "config_sha256",
        "producer_config_sha256",
        "input_fingerprint",
        "peak_rss_bytes",
        "score_encoding",
        "token_accounting",
        "raw_score_tie_definition",
        "raw_score_tie_groups",
        "rows_in_raw_score_ties",
        "queries_with_any_raw_score_tie",
        "queries_with_top10_raw_score_tie",
        "ties_crossing_rank10_boundary",
        "largest_raw_score_tie_group",
    }
    score_frame = pq.read_table(
        expected_payloads[0], columns=["query_id", "docid", "score"]
    ).to_pandas()
    expected_ties = raw_score_tie_statistics(score_frame)
    for path, entry in zip(expected_payloads, files, strict=True):
        if not isinstance(entry, dict) or required.difference(entry):
            raise ValueError(
                f"manifest entry is incomplete for {path}: "
                f"{sorted(required.difference(entry if isinstance(entry, dict) else {}))}"
            )
        if entry["size_bytes"] != path.stat().st_size or entry["sha256"] != sha256_file(path):
            raise ValueError(f"manifest size/hash is stale for {path}")
        contract = artifact_contract(path)
        if entry["row_count"] != contract["row_count"] or entry["schema"] != contract["schema"]:
            raise ValueError(f"manifest tabular contract is stale for {path}")
        if entry["score_encoding"] != SCORE_ENCODING:
            raise ValueError(f"manifest score encoding changed for {path}")
        if any(entry[name] != value for name, value in expected_ties.items()):
            raise ValueError(f"manifest raw-score tie statistics are stale for {path}")
        hashes = entry["input_hashes"]
        if not isinstance(hashes, dict) or not hashes or not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in hashes.values()
        ):
            raise ValueError(f"manifest input hashes are invalid for {path}")


def validate_phase2_archive(
    config: Mapping[str, Any],
    archive_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    archive_file = Path(archive_path)
    manifest_file = Path(manifest_path)
    _require_regular_file(archive_file, "Phase 2 ZIP")
    _require_regular_file(manifest_file, "rerank manifest")
    payloads = phase2_payload_paths(config)
    manifest = _read_json(manifest_file)
    validate_phase2_manifest(config, manifest, expected_payloads=payloads)
    allowlist = [
        *[portable_path(config, path) for path in payloads],
        portable_path(config, manifest_file),
    ]
    with zipfile.ZipFile(archive_file) as archive:
        if archive.namelist() != allowlist or len(set(archive.namelist())) != len(allowlist):
            raise ValueError("Phase 2 ZIP differs from the exact ordered allowlist")
        if archive.testzip() is not None:
            raise ValueError("Phase 2 ZIP failed CRC validation")
        for info in archive.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts or info.is_dir():
                raise ValueError(f"unsafe Phase 2 ZIP member: {info.filename}")
        with tempfile.TemporaryDirectory(prefix="rusearchrank-phase2-verify-") as directory:
            extraction_root = Path(directory)
            archive.extractall(extraction_root)
            extracted_manifest_path = extraction_root / portable_path(config, manifest_file)
            extracted_manifest = _read_json(extracted_manifest_path)
            if extracted_manifest != manifest:
                raise ValueError("rerank manifest changed after ZIP extraction")
            if any(
                entry.get("path") == portable_path(config, manifest_file)
                for entry in extracted_manifest["files"]
            ):
                raise ValueError("extracted rerank manifest lists itself")
            for entry in extracted_manifest["files"]:
                extracted = extraction_root / str(entry["path"])
                if not extracted.is_file():
                    raise ValueError(f"ZIP extraction is missing {entry['path']}")
                if extracted.stat().st_size != entry["size_bytes"]:
                    raise ValueError(f"extracted size mismatch for {entry['path']}")
                if sha256_file(extracted) != entry["sha256"]:
                    raise ValueError(f"extracted hash mismatch for {entry['path']}")
                contract = artifact_contract(extracted)
                if contract["row_count"] != entry["row_count"] or contract["schema"] != entry["schema"]:
                    raise ValueError(f"extracted contract mismatch for {entry['path']}")
            extracted_protocol = extraction_root / portable_path(
                config, resolve_path(config, config["audits"]["protocol_snapshot"])
            )
            if sha256_file(extracted_protocol) != sha256_file(Path(str(config["_config_path"]))):
                raise ValueError("archived protocol snapshot differs byte-for-byte from config")
    return {
        "path": portable_path(config, archive_file),
        "size_bytes": archive_file.stat().st_size,
        "sha256": sha256_file(archive_file),
        "contents": allowlist,
        "manifest_sha256": sha256_file(manifest_file),
    }


def package_phase2(
    config: Mapping[str, Any], *, overwrite: bool = False
) -> dict[str, Any]:
    validate_rerank_config(config)
    validate_phase1_inputs(config)
    base_payloads = phase2_payload_paths(config)[:-1]
    for path in base_payloads:
        _require_regular_file(path, "Phase 2 package input")
    score_path = resolve_path(config, config["artifacts"]["scores"])
    score_sidecar_path = _sidecar_path(score_path)
    score_sidecar = validate_current_score_sidecar(config, score_path=score_path)
    raw_score_ties = raw_score_tie_statistics(
        pq.read_table(
            score_path, columns=["query_id", "docid", "score"]
        ).to_pandas()
    )
    protocol_path = resolve_path(config, config["audits"]["protocol_snapshot"])
    manifest_path = resolve_path(config, config["audits"]["manifest"])
    archive_path = resolve_path(config, config["archive"]["path"])
    states = {
        "protocol_snapshot": protocol_path.exists(),
        "manifest": manifest_path.exists(),
        "archive": archive_path.exists(),
    }
    if all(states.values()) and not overwrite:
        try:
            report = validate_phase2_archive(config, archive_path, manifest_path)
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            raise ValueError(
                "existing Phase 2 package is invalid or stale and was preserved; "
                f"use --overwrite after review: {exc}"
            ) from exc
        return {"status": "PASS", "action": "reused_valid_package", **report}
    if any(states.values()) and not all(states.values()) and not overwrite:
        raise ValueError(
            f"partial Phase 2 package output detected and preserved: {states}; "
            "use --overwrite after review"
        )
    if overwrite:
        preserve_stale(archive_path)
        preserve_stale(manifest_path)
        preserve_stale(protocol_path)

    config_path = Path(str(config["_config_path"]))
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_protocol = protocol_path.with_name(
        f"{protocol_path.name}.tmp.{os.getpid()}"
    )
    if temporary_protocol.exists():
        preserve_stale(temporary_protocol)
    shutil.copyfile(config_path, temporary_protocol)
    with temporary_protocol.open("rb") as stream:
        os.fsync(stream.fileno())
    if sha256_file(temporary_protocol) != sha256_file(config_path):
        raise ValueError("temporary protocol snapshot differs byte-for-byte from config")
    temporary_protocol.replace(protocol_path)

    payloads = phase2_payload_paths(config)
    input_hashes = _phase2_input_hashes(config)
    producers = {
        portable_path(config, payloads[0]): (
            "python -m rusearchrank.cli rerank-score --config configs/rerank.yaml --split dev"
        ),
        portable_path(config, payloads[1]): (
            "python -m rusearchrank.cli build-rerank-run --config configs/rerank.yaml "
            "--split dev --depth 100"
        ),
        portable_path(config, payloads[2]): (
            "python -m rusearchrank.cli evaluate-rerank --config configs/rerank.yaml --split dev"
        ),
        portable_path(config, payloads[3]): (
            "python -m rusearchrank.cli evaluate-rerank --config configs/rerank.yaml --split dev"
        ),
        portable_path(config, payloads[4]): (
            "python -m rusearchrank.cli evaluate-rerank --config configs/rerank.yaml --split dev"
        ),
        portable_path(config, payloads[5]): (
            "python -m rusearchrank.cli evaluate-rerank --config configs/rerank.yaml --split dev"
        ),
        portable_path(config, payloads[6]): (
            "python -m rusearchrank.cli package-phase2 --config configs/rerank.yaml"
        ),
    }
    manifest = {
        "status": "PASS",
        "created_at": utc_now(),
        "files": [
            _manifest_entry(
                config,
                path,
                score_sidecar=score_sidecar,
                raw_score_ties=raw_score_ties,
                producer_command=producers[portable_path(config, path)],
                input_hashes=input_hashes[portable_path(config, path)],
            )
            for path in payloads
        ],
    }
    # The manifest intentionally has no entry, size, or digest for itself.
    atomic_write_json(manifest_path, manifest)
    validate_phase2_manifest(config, manifest, expected_payloads=payloads)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(f"{archive_path.name}.tmp.{os.getpid()}")
    if temporary_archive.exists():
        preserve_stale(temporary_archive)
    with zipfile.ZipFile(
        temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in [*payloads, manifest_path]:
            archive.write(path, arcname=portable_path(config, path))
    with temporary_archive.open("rb") as stream:
        os.fsync(stream.fileno())
    try:
        validate_phase2_archive(config, temporary_archive, manifest_path)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"temporary Phase 2 ZIP validation failed; preserved at {temporary_archive}: {exc}"
        ) from exc
    temporary_archive.replace(archive_path)
    report = validate_phase2_archive(config, archive_path, manifest_path)
    return {"status": "PASS", "action": "created_package", **report}
