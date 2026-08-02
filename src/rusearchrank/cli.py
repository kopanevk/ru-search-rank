"""Phase 0 checks and the guarded Linux/Colab Phase 1A command line."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence
import zipfile

import pandas as pd
import yaml

from .data import (
    CANDIDATE_COLUMNS,
    download_annotation_files,
    inspect_miracl_ru,
    load_qrels,
    load_topics,
    read_candidate_table,
    stream_candidate_passages,
    validate_candidate_schema,
    validate_passages,
    validate_queries,
)
from .evaluation import (
    build_qrels_split_audit,
    parse_trec_eval_metric,
    reproduction_rows,
)
from .retrieval import (
    build_retrieval_depth_audit,
    join_qrels,
    normalize_bm25_run,
    read_trec_run,
    smoke_prebuilt_index,
    validate_query_coverage,
    validate_stable_order,
    validate_top_k,
    write_trec_run,
)


DEFAULT_CHECKPOINT = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_AUDIT_DIR = Path("reports/audit")
DEFAULT_RETRIEVAL_CONFIG = Path("configs/retrieval.yaml")
FULL_RETRIEVAL_ERROR = (
    "Full retrieval requires Linux, Python 3.12 and Java 21. "
    "Use scripts/run_full_bm25_retrieval.ipynb."
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"JSON file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_retrieval_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ValueError(f"retrieval config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("retrieval config must contain a mapping")
    required = {
        "dataset",
        "environment",
        "retrieval",
        "artifacts",
        "audits",
        "archive",
        "reproduction_gate",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"retrieval config is missing sections: {', '.join(missing)}")
    payload["_config_path"] = str(config_path)
    return payload


def _repository_root(config: dict[str, Any]) -> Path:
    config_path = Path(str(config["_config_path"]))
    root_setting = Path(str(config.get("paths", {}).get("repository_root", ".")))
    if root_setting.is_absolute():
        return root_setting.resolve()
    return (config_path.parent / root_setting).resolve()


def _resolve_repository_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_repository_root(config) / path).resolve()


def _portable_repository_path(config: dict[str, Any], path: Path) -> str:
    try:
        return path.resolve().relative_to(_repository_root(config)).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must stay inside the repository: {path}") from exc


def _java_major_version(output: str) -> int | None:
    match = re.search(r'(?:version\s+"|openjdk\s+)(\d+)', output.lower())
    return int(match.group(1)) if match else None


def _phase1_environment(config: dict[str, Any]) -> dict[str, Any]:
    java = _run_capture(["java", "-version"])
    try:
        pyserini_version = importlib.metadata.version("pyserini")
    except importlib.metadata.PackageNotFoundError:
        pyserini_version = None
    disk = shutil.disk_usage(Path.cwd())
    return {
        "platform": platform.system(),
        "platform_detail": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "java": java,
        "java_major": _java_major_version(str(java.get("output", ""))),
        "pyserini": pyserini_version,
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "disk_free_bytes": disk.free,
        "required": config["environment"],
    }


def _require_external_environment(config: dict[str, Any]) -> dict[str, Any]:
    report = _phase1_environment(config)
    expected = config["environment"]
    python_expected = tuple(int(part) for part in str(expected["python"]).split("."))
    python_ok = sys.version_info[:2] == python_expected
    java_ok = report["java_major"] == int(expected["java"])
    pyserini_ok = report["pyserini"] == str(expected["pyserini"])
    disk_required = int(expected["minimum_free_disk_gib"]) * 1024**3
    disk_ok = int(report["disk_free_bytes"]) >= disk_required
    if not (
        platform.system() == "Linux"
        and python_ok
        and java_ok
        and pyserini_ok
        and disk_ok
    ):
        raise RuntimeError(FULL_RETRIEVAL_ERROR)
    return report


def _artifact_path(config: dict[str, Any], key: str) -> Path:
    return Path(str(config["artifacts"][key]))


def _audit_path(config: dict[str, Any], key: str) -> Path:
    return Path(str(config["audits"][key]))


def _ensure_writable_targets(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "refusing to overwrite existing artifacts without --overwrite: "
            + ", ".join(existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _annotation_paths(config: dict[str, Any]) -> dict[str, dict[str, Path]]:
    dataset = config["dataset"]
    return download_annotation_files(
        topics_urls={str(key): str(value) for key, value in dataset["topics"].items()},
        qrels_urls={str(key): str(value) for key, value in dataset["qrels"].items()},
        raw_dir=_resolve_repository_path(config, config["paths"]["raw_dir"]),
        expected_rows={str(key): int(value) for key, value in dataset["expected_rows"].items()},
    )


def _expected_annotation_paths(config: dict[str, Any]) -> dict[str, dict[str, Path]]:
    raw_dir = _resolve_repository_path(config, config["paths"]["raw_dir"])
    paths = {
        split: {
            "topics": raw_dir / f"topics.miracl-v1.0-ru-{split}.tsv",
            "qrels": raw_dir / f"qrels.miracl-v1.0-ru-{split}.tsv",
        }
        for split in ("train", "dev")
    }
    paths["train"]["topics"] = _resolve_repository_path(
        config, config["dataset"]["train_topics_path"]
    )
    return paths


def _validate_train_topics(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _resolve_repository_path(config, config["dataset"]["train_topics_path"])
    portable_path = _portable_repository_path(config, path)
    download_hint = (
        "Run `python -m rusearchrank.cli prepare-annotations "
        "--config configs/retrieval.yaml` first to download official MIRACL topics."
    )
    if not path.exists():
        raise ValueError(f"official train topics file is missing: {portable_path}. {download_hint}")
    if not path.is_file():
        raise ValueError(f"official train topics path is not a regular file: {portable_path}")
    if path.suffix.lower() != ".tsv":
        raise ValueError(f"official train topics file must have a .tsv suffix: {portable_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"official train topics TSV is empty: {portable_path}. {download_hint}")

    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if "\t" not in line:
                raise ValueError(
                    f"malformed train topics line {line_number}: expected query_id<TAB>query_text"
                )
            query_id, query_text = line.split("\t", 1)
            if not query_id.strip():
                raise ValueError(f"empty query_id in train topics at line {line_number}")
            if not query_text.strip():
                raise ValueError(f"empty query_text in train topics at line {line_number}")
            if query_id in seen:
                raise ValueError(f"duplicate query_id in train topics: {query_id}")
            seen.add(query_id)
            rows.append((query_id, query_text))

    expected = int(config["dataset"]["expected_rows"]["train_queries"])
    if len(rows) != expected:
        raise ValueError(
            f"official train topics contain {len(rows)} unique queries; expected {expected}"
        )
    queries = pd.DataFrame(rows, columns=["query_id", "query_text"], dtype="string")
    queries.insert(0, "split", "train")
    validate_queries(queries)
    metadata = {
        "path": portable_path,
        "sha256": _sha256(path),
        "query_count": len(rows),
        "source": str(config["dataset"]["topics"]["train"]),
        "revision": str(config["dataset"]["annotations_revision"]),
    }
    return queries, metadata


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_capture(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "returncode": None, "output": str(exc)}
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "output": output,
    }


def _memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        sysctl = _run_capture(["sysctl", "-n", "hw.memsize"])
        if sysctl["available"]:
            try:
                return int(sysctl["output"])
            except ValueError:
                pass
        profiler = _run_capture(["system_profiler", "SPHardwareDataType"])
        match = re.search(r"Memory:\s+([\d.]+)\s+(GB|MB)", profiler["output"])
        if match:
            multiplier = 1024**3 if match.group(2) == "GB" else 1024**2
            return int(float(match.group(1)) * multiplier)
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError):
            pass
    return None


def _package_versions() -> dict[str, str | None]:
    packages = (
        "datasets",
        "transformers",
        "sentence-transformers",
        "torch",
        "pandas",
        "pyarrow",
        "numpy",
        "PyYAML",
        "pytest",
        "pyserini",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _collect_environment() -> dict[str, Any]:
    cwd = Path.cwd().resolve()
    disk = shutil.disk_usage(cwd)
    git_root = _run_capture(["git", "rev-parse", "--show-toplevel"], cwd)
    git_branch = _run_capture(["git", "branch", "--show-current"], cwd)
    git_head = _run_capture(["git", "rev-parse", "HEAD"], cwd)
    git_status = _run_capture(["git", "status", "--short"], cwd)
    java = _run_capture(["java", "-version"])

    torch_report: dict[str, Any] = {"installed": importlib.util.find_spec("torch") is not None}
    if torch_report["installed"]:
        import torch

        torch_report.update(
            {
                "version": torch.__version__,
                "mps_built": bool(torch.backends.mps.is_built()),
                "mps_available": bool(torch.backends.mps.is_available()),
            }
        )
    else:
        torch_report.update({"version": None, "mps_built": None, "mps_available": None})

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(cwd),
        "git": {
            "root": git_root["output"] or None,
            "branch": git_branch["output"] or None,
            "head": git_head["output"] or None,
            "status_short": git_status["output"].splitlines() if git_status["output"] else [],
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "python_on_path": shutil.which("python"),
            "python3_on_path": shutil.which("python3"),
        },
        "java": java,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "macos_version": platform.mac_ver()[0] or None,
            "memory_bytes": _memory_bytes(),
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "torch": torch_report,
        "packages": _package_versions(),
    }


def _environment_report(args: argparse.Namespace) -> int:
    output = Path(args.output)
    initial_audit: dict[str, Any] | None = None
    if output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            initial_audit = existing.get("initial_audit")
        except (json.JSONDecodeError, OSError):
            initial_audit = None
    payload = {
        "schema_version": 1,
        "initial_audit": initial_audit,
        "latest_audit": _collect_environment(),
    }
    _write_json(output, payload)
    _print_json({"status": "ok", "report": str(output), "environment": payload["latest_audit"]})
    return 0


def _inspect_data(args: argparse.Namespace) -> int:
    report = inspect_miracl_ru(sample_size=args.sample_size, timeout=args.timeout)
    output = Path(args.output)
    _write_json(output, report)
    _print_json(
        {
            "status": "ok",
            "report": str(output),
            "corpus_sample_size": report["corpus"]["sample_size"],
            "train_queries": report["splits"]["train"]["queries"]["row_count"],
            "dev_queries": report["splits"]["dev"]["queries"]["row_count"],
            "relevance_values": report["splits"]["dev"]["qrels"]["relevance_values"],
            "full_corpus_downloaded": False,
        }
    )
    return 0


def _model_info_total(safetensors_info: Any) -> int | None:
    if safetensors_info is None:
        return None
    if isinstance(safetensors_info, dict):
        value = safetensors_info.get("total")
    else:
        value = getattr(safetensors_info, "total", None)
    return int(value) if value is not None else None


def _inference_on_device(model: Any, encoded: dict[str, Any], device: str) -> tuple[list[int], list[float]]:
    import torch

    model.to(device)
    device_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
    with torch.no_grad():
        logits = model(**device_inputs).logits
    if device == "mps":
        torch.mps.synchronize()
    return list(logits.shape), [float(score) for score in logits.detach().cpu().reshape(-1)]


def _inspect_checkpoint(args: argparse.Namespace) -> int:
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "checkpoint inspection requires the project dependencies; run `python -m pip install -e .`"
        ) from exc

    info = HfApi().model_info(args.checkpoint, revision=args.revision)
    resolved_revision = info.sha
    config = AutoConfig.from_pretrained(args.checkpoint, revision=resolved_revision)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=resolved_revision)
    max_length = min(int(args.max_length), int(tokenizer.model_max_length))

    probe_query = "Какова столица Франции?"
    probe_document = "Париж\nПариж — столица и крупнейший город Франции."
    probe_tokens = tokenizer(
        probe_query,
        probe_document,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    sep_token_id = tokenizer.sep_token_id
    sep_count = int(probe_tokens["input_ids"].eq(sep_token_id).sum().item())

    report: dict[str, Any] = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_name": args.checkpoint,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "model_class": config.architectures[0] if config.architectures else None,
        "tokenizer_class": tokenizer.__class__.__name__,
        "num_labels": int(config.num_labels),
        "sep_token": tokenizer.sep_token,
        "sep_token_id": sep_token_id,
        "model_max_length": int(tokenizer.model_max_length),
        "smoke_max_length": max_length,
        "score_activation": getattr(
            config, "sbert_ce_default_activation_function", "UNVERIFIED"
        ),
        "score_direction": "higher_is_more_relevant",
        "safetensors_element_count": _model_info_total(getattr(info, "safetensors", None)),
        "pair_tokenization": {
            "format": "tokenizer(query, title + '\\n' + text, truncation=True, max_length=...)",
            "literal_sep_inserted": False,
            "input_shape": list(probe_tokens["input_ids"].shape),
            "sep_token_count": sep_count,
        },
        "weights_loaded": bool(args.with_model),
        "parameter_count": None,
        "word_embeddings_parameter_path": None,
        "logits_shape": None,
        "cpu_smoke": {"status": "not_run"},
        "mps_smoke": {
            "available": bool(torch.backends.mps.is_available()),
            "status": "not_run",
        },
    }

    if args.with_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint, revision=resolved_revision
        )
        report["model_class"] = model.__class__.__name__
        report["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        embedding_paths = [
            name
            for name, _ in model.named_parameters()
            if name.endswith("embeddings.word_embeddings.weight")
        ]
        if len(embedding_paths) != 1:
            raise RuntimeError(f"expected one word embedding parameter, found: {embedding_paths}")
        report["word_embeddings_parameter_path"] = embedding_paths[0]

        pairs = [
            {
                "kind": "clearly_relevant",
                "query": "Какова столица Франции?",
                "title": "Париж",
                "text": "Париж — столица и крупнейший город Франции.",
            },
            {
                "kind": "clearly_irrelevant",
                "query": "Какова столица Франции?",
                "title": "Яблочный пирог",
                "text": "Пирог готовят из яблок, муки, масла и сахара.",
            },
            {
                "kind": "neutral",
                "query": "Какова столица Франции?",
                "title": "Франция",
                "text": "Франция — государство в Западной Европе.",
            },
        ]
        encoded = tokenizer(
            [pair["query"] for pair in pairs],
            [f"{pair['title']}\n{pair['text']}" for pair in pairs],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        expected_shape = [len(pairs), int(config.num_labels)]
        cpu_shape, cpu_scores = _inference_on_device(model, encoded, "cpu")
        finite = all(np_value == np_value and abs(np_value) != float("inf") for np_value in cpu_scores)
        distinct = len({round(score, 7) for score in cpu_scores}) > 1
        direction_ok = cpu_scores[0] > cpu_scores[1]
        if cpu_shape != expected_shape:
            raise RuntimeError(f"unexpected logits shape: {cpu_shape}, expected {expected_shape}")
        if not finite or not distinct:
            raise RuntimeError("checkpoint smoke scores must be finite and not all identical")

        report["logits_shape"] = cpu_shape
        report["cpu_smoke"] = {
            "status": "passed",
            "scores": {
                pair["kind"]: score for pair, score in zip(pairs, cpu_scores, strict=True)
            },
            "finite": finite,
            "not_all_identical": distinct,
            "higher_is_more_relevant_check": direction_ok,
        }
        if not direction_ok:
            raise RuntimeError("relevant smoke pair did not score above the irrelevant pair")

        if torch.backends.mps.is_available():
            try:
                mps_shape, mps_scores = _inference_on_device(model, encoded, "mps")
                mps_finite = all(
                    value == value and abs(value) != float("inf") for value in mps_scores
                )
                if mps_shape != expected_shape or not mps_finite:
                    raise RuntimeError("MPS returned an invalid shape or non-finite scores")
                report["mps_smoke"] = {
                    "available": True,
                    "status": "passed",
                    "logits_shape": mps_shape,
                    "scores": {
                        pair["kind"]: score
                        for pair, score in zip(pairs, mps_scores, strict=True)
                    },
                }
            except Exception as exc:  # The failure is preserved in the audit artifact.
                report["mps_smoke"] = {
                    "available": True,
                    "status": "failed",
                    "error": str(exc),
                }
        else:
            report["mps_smoke"] = {
                "available": False,
                "status": "skipped",
                "reason": "torch.backends.mps.is_available() is false",
            }

    output = Path(args.output)
    _write_json(output, report)
    _print_json(
        {
            "status": "ok",
            "report": str(output),
            "resolved_revision": resolved_revision,
            "weights_loaded": report["weights_loaded"],
            "cpu_smoke": report["cpu_smoke"]["status"],
            "mps_smoke": report["mps_smoke"]["status"],
        }
    )
    return 0


def _validate_candidates(args: argparse.Namespace) -> int:
    candidates = read_candidate_table(args.path)
    validate_candidate_schema(candidates)
    config = _load_retrieval_config(args.config)
    top_k = int(config["retrieval"]["candidate_depth"])
    validate_top_k(candidates, top_k, require_exact=False)
    validate_stable_order(candidates)

    queries_path = _artifact_path(config, "queries")
    passages_path = _artifact_path(config, "passages")
    coverage: dict[str, Any] = {"queries": "not_checked", "passages": "not_checked"}
    if queries_path.is_file():
        queries = pd.read_parquet(queries_path)
        validate_queries(queries)
        split_values = candidates["split"].astype("string").drop_duplicates().tolist()
        expected = queries.loc[queries["split"].astype("string").isin(split_values)]
        validate_query_coverage(candidates, expected)
        coverage["queries"] = "complete"
    if passages_path.is_file():
        passages = pd.read_parquet(passages_path)
        validate_passages(passages)
        missing_docids = set(candidates["docid"].astype("string")).difference(
            passages["docid"].astype("string")
        )
        if missing_docids:
            raise ValueError(f"candidate passages are missing: {sorted(missing_docids)[:10]}")
        coverage["passages"] = "complete"

    counts = candidates["judgment"].value_counts().to_dict()
    payload = {
        "status": "valid",
        "path": str(Path(args.path).resolve()),
        "rows": int(len(candidates)),
        "queries": int(candidates["query_id"].nunique()),
        "top_k": top_k,
        "stable_sorting": "valid",
        "coverage": coverage,
        "judgment_counts": {str(key): int(value) for key, value in counts.items()},
    }
    _print_json(payload)
    return 0


def _prepare_annotations(args: argparse.Namespace) -> int:
    config = _load_retrieval_config(args.config)
    downloaded = _annotation_paths(config)
    configured_train = _expected_annotation_paths(config)["train"]["topics"]
    if downloaded["train"]["topics"].resolve() != configured_train.resolve():
        raise ValueError(
            "dataset.train_topics_path must identify the official downloaded train TSV"
        )
    _, train_metadata = _validate_train_topics(config)
    _print_json(
        {
            "status": "ok",
            "train_topics": train_metadata,
            "dev_topics": _portable_repository_path(
                config, downloaded["dev"]["topics"]
            ),
            "train_qrels": _portable_repository_path(
                config, downloaded["train"]["qrels"]
            ),
            "dev_qrels": _portable_repository_path(
                config, downloaded["dev"]["qrels"]
            ),
        }
    )
    return 0


def _inspect_linux_environment(args: argparse.Namespace) -> int:
    config = _load_retrieval_config(args.config)
    report = _phase1_environment(config)
    try:
        _require_external_environment(config)
    except RuntimeError:
        _print_json({"status": "failed", "environment": report, "error": FULL_RETRIEVAL_ERROR})
        raise

    payload: dict[str, Any] = {"status": "valid", "environment": report}
    if args.check_index:
        payload["index_smoke"] = smoke_prebuilt_index(
            index_name=str(config["retrieval"]["index_name"]),
            language=str(config["dataset"]["language"]),
            query_text="Когда начался Карибский кризис?",
        )
    _print_json(payload)
    return 0


def _retrieval_topics_argument(config: dict[str, Any], split: str) -> str:
    if split == "train":
        _, metadata = _validate_train_topics(config)
        return str(metadata["path"])
    if split == "dev":
        return str(config["reproduction_gate"]["official_topic"])
    raise ValueError(f"unsupported retrieval split: {split}")


def _build_bm25_command(
    config: dict[str, Any],
    *,
    split: str,
    target: Path,
    topics_argument: str | None = None,
) -> list[str]:
    topics = topics_argument or _retrieval_topics_argument(config, split)
    return [
        sys.executable,
        "-m",
        "pyserini.search.lucene",
        "--threads",
        str(config["retrieval"]["threads"]),
        "--batch-size",
        str(config["retrieval"]["batch_size"]),
        "--language",
        str(config["dataset"]["language"]),
        "--topics",
        topics,
        "--index",
        str(config["retrieval"]["index_name"]),
        "--output",
        str(target),
        "--bm25",
        "--hits",
        str(config["retrieval"]["retrieval_hits"][split]),
    ]


def _zero_hit_query_ids(depth_audit: dict[str, object]) -> list[str]:
    rows = depth_audit.get("zero_hit_queries", [])
    if not isinstance(rows, list):
        raise ValueError("retrieval-depth audit has invalid zero_hit_queries")
    return [str(row["query_id"]) for row in rows]


def _require_no_zero_hits(
    depth_audit: dict[str, object],
    *,
    context: str,
) -> None:
    query_ids = _zero_hit_query_ids(depth_audit)
    if query_ids:
        raise ValueError(f"{context}; zero-hit query_id values: " + ", ".join(query_ids))


def _validate_raw_trec_depth(
    raw_run: pd.DataFrame,
    *,
    requested_top_k: int,
) -> None:
    source_ranked = raw_run.rename(columns={"source_rank": "bm25_rank"})[
        ["split", "query_id", "docid", "bm25_rank", "bm25_score"]
    ]
    validate_top_k(source_ranked, requested_top_k, require_exact=False)


def _run_bm25(args: argparse.Namespace) -> int:
    config = _load_retrieval_config(args.config)
    _require_external_environment(config)
    splits = list(config["splits"]) if args.split == "all" else [args.split]
    run_keys = {"train": "train_run", "dev": "dev_run"}
    targets = [
        _resolve_repository_path(config, config["artifacts"][run_keys[split]])
        for split in splits
    ]
    _ensure_writable_targets(targets, overwrite=args.overwrite)
    annotation_paths = _expected_annotation_paths(config)

    results: dict[str, Any] = {}
    for split, target in zip(splits, targets, strict=True):
        train_topics_metadata: dict[str, Any] | None = None
        if split == "train":
            queries, train_topics_metadata = _validate_train_topics(config)
            topics_argument = str(train_topics_metadata["path"])
        else:
            queries = load_topics(annotation_paths[split]["topics"], split=split)
            expected_queries = int(config["dataset"]["expected_rows"]["dev_queries"])
            if len(queries) != expected_queries:
                raise ValueError(
                    f"official dev topics contain {len(queries)} queries; "
                    f"expected {expected_queries}"
                )
            topics_argument = _retrieval_topics_argument(config, split)
        retrieval_hits = int(config["retrieval"]["retrieval_hits"][split])
        temporary = target.with_suffix(target.suffix + ".tmp")
        command = _build_bm25_command(
            config,
            split=split,
            target=temporary,
            topics_argument=topics_argument,
        )
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=_repository_root(config),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            temporary.unlink(missing_ok=True)
            output = "\n".join(
                part.strip() for part in (result.stdout, result.stderr) if part.strip()
            )
            raise RuntimeError(f"official Pyserini retrieval failed: {output}")
        elapsed = time.perf_counter() - started
        if not temporary.is_file():
            raise RuntimeError("official Pyserini retrieval did not create its run file")
        work_path = _resolve_repository_path(
            config, config["paths"]["work_dir"]
        ) / f"retrieval_{split}.json"
        try:
            raw_run = read_trec_run(str(temporary), split=split)
            _validate_raw_trec_depth(
                raw_run,
                requested_top_k=retrieval_hits,
            )
            validated = normalize_bm25_run(
                raw_run[["split", "query_id", "docid", "bm25_score"]]
            )
            depth_audit = build_retrieval_depth_audit(
                validated,
                queries,
                requested_top_k=retrieval_hits,
            )
        except (ValueError, RuntimeError) as exc:
            preserved = _portable_repository_path(config, temporary)
            raise ValueError(
                f"BM25 run validation failed: {exc}. Raw run preserved at {preserved}"
            ) from exc

        run_report = {
            "split": split,
            "seconds": elapsed,
            "queries": int(len(queries)),
            "rows": int(len(raw_run)),
            "hits_per_query": retrieval_hits,
            "topics_argument": topics_argument,
            "train_topics": train_topics_metadata,
            "command": command,
            "run": _portable_repository_path(config, target),
            "raw_validation_path": _portable_repository_path(config, temporary),
            "depth_audit": depth_audit,
        }
        zero_hit_ids = _zero_hit_query_ids(depth_audit)
        if zero_hit_ids:
            run_report["status"] = "FAIL_ZERO_HIT_QUERIES"
            _write_json(work_path, run_report)
            raise ValueError(
                "BM25 run contains zero-hit queries: "
                + ", ".join(zero_hit_ids)
                + ". Raw run preserved at "
                + _portable_repository_path(config, temporary)
            )
        validate_query_coverage(validated, queries)
        # Preserve Pyserini's raw output byte-for-byte for official evaluation.
        temporary.replace(target)
        run_report["status"] = "PASS"
        run_report["raw_validation_path"] = None
        _write_json(work_path, run_report)
        results[split] = {
            "run": _portable_repository_path(config, target),
            "queries": int(len(queries)),
            "rows": int(len(raw_run)),
            "hits_per_query": retrieval_hits,
            "seconds": elapsed,
            "depth_audit": depth_audit,
        }
    _print_json({"status": "ok", "retrieval": results})
    return 0


def _stable_truncate_trec_run(
    source: Path,
    target: Path,
    *,
    split: str,
    top_k: int,
    source_top_k: int | None = None,
) -> pd.DataFrame:
    """Write a separate deterministic top-k run without mutating its source."""

    source_hash = _sha256(source)
    raw = read_trec_run(str(source), split=split)
    raw_limit = source_top_k or int(pd.to_numeric(raw["source_rank"]).max())
    _validate_raw_trec_depth(raw, requested_top_k=raw_limit)
    normalized = normalize_bm25_run(
        raw[["split", "query_id", "docid", "bm25_score"]],
        top_k=top_k,
    )
    validate_top_k(normalized, top_k, require_exact=False)
    validate_stable_order(normalized)
    temporary = target.with_suffix(target.suffix + ".tmp")
    write_trec_run(
        normalized,
        str(temporary),
        tag="rusearchrank-stable-top100",
    )
    if _sha256(source) != source_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("source dev top-1000 run changed during stable truncation")
    temporary.replace(target)
    # Candidate scores and ranks come from the exact portable top-k file.
    portable = read_trec_run(str(target), split=split)
    portable = normalize_bm25_run(
        portable[["split", "query_id", "docid", "bm25_score"]],
        top_k=top_k,
    )
    validate_top_k(portable, top_k, require_exact=False)
    validate_stable_order(portable)
    return portable


def _build_candidate_cache(args: argparse.Namespace) -> int:
    config = _load_retrieval_config(args.config)
    _require_external_environment(config)
    reproduction = _read_json(_audit_path(config, "reproduction"))
    if reproduction.get("status") != "PASS" or reproduction.get("gate_passed") is not True:
        raise ValueError(
            "candidate cache requires a passed official dev reproduction gate"
        )
    dev_run_path = _artifact_path(config, "dev_run")
    if reproduction.get("source_run") != str(dev_run_path):
        raise ValueError(
            "candidate cache requires an audit of the configured dev top-1000 run"
        )
    if reproduction.get("source_run_sha256") != _sha256(dev_run_path):
        raise ValueError(
            "candidate cache requires a reproduction gate for the current dev run"
        )
    dev_source_hash = _sha256(dev_run_path)
    dev_top100_path = _artifact_path(config, "dev_top100_run")
    train_candidates_path = _artifact_path(config, "train_candidates")
    dev_candidates_path = _artifact_path(config, "dev_candidates")
    queries_path = _artifact_path(config, "queries")
    passages_path = _artifact_path(config, "passages")
    output_paths = [
        dev_top100_path,
        train_candidates_path,
        dev_candidates_path,
        queries_path,
        passages_path,
    ]
    _ensure_writable_targets(output_paths, overwrite=args.overwrite)
    annotation_paths = _annotation_paths(config)
    top_k = int(config["retrieval"]["candidate_depth"])
    dev_top100 = _stable_truncate_trec_run(
        dev_run_path,
        dev_top100_path,
        split="dev",
        top_k=top_k,
        source_top_k=int(config["retrieval"]["retrieval_hits"]["dev"]),
    )
    if _sha256(dev_run_path) != dev_source_hash:
        raise RuntimeError("dev top-1000 run was modified after reproduction")

    candidate_frames: dict[str, pd.DataFrame] = {}
    depth_audits: dict[str, dict[str, object]] = {}
    query_frames: list[pd.DataFrame] = []
    for split in ("train", "dev"):
        queries = load_topics(annotation_paths[split]["topics"], split=split)
        qrels = load_qrels(annotation_paths[split]["qrels"], split=split)
        if split == "dev":
            normalized = dev_top100
        else:
            run_path = _artifact_path(config, "train_run")
            if not run_path.is_file():
                raise ValueError(f"BM25 run does not exist: {run_path}")
            raw_run = read_trec_run(str(run_path), split=split)
            _validate_raw_trec_depth(raw_run, requested_top_k=top_k)
            normalized = normalize_bm25_run(
                raw_run[["split", "query_id", "docid", "bm25_score"]]
            )
        validate_top_k(normalized, top_k, require_exact=False)
        depth_audit = build_retrieval_depth_audit(
            normalized,
            queries,
            requested_top_k=top_k,
        )
        _require_no_zero_hits(
            depth_audit,
            context=f"cannot build {split} candidate cache",
        )
        validate_query_coverage(normalized, queries)
        candidates = join_qrels(normalized, qrels)
        candidates = candidates.loc[:, list(CANDIDATE_COLUMNS)]
        validate_candidate_schema(candidates)
        candidate_frames[split] = candidates
        depth_audits[split] = depth_audit
        query_frames.append(queries)

    queries = pd.concat(query_frames, ignore_index=True)
    validate_queries(queries)
    candidate_docids = set(
        pd.concat(
            [frame["docid"] for frame in candidate_frames.values()], ignore_index=True
        ).astype("string")
    )
    extraction_started = time.perf_counter()
    passages = stream_candidate_passages(
        candidate_docids,
        dataset_name=str(config["dataset"]["corpus_source"]),
        language=str(config["dataset"]["language"]),
        revision=str(config["dataset"]["corpus_revision"]),
    )
    extraction_seconds = time.perf_counter() - extraction_started
    validate_passages(passages)
    if set(passages["docid"].astype("string")) != candidate_docids:
        raise ValueError("passage extraction did not produce exact candidate coverage")

    _atomic_parquet(candidate_frames["train"], train_candidates_path)
    _atomic_parquet(candidate_frames["dev"], dev_candidates_path)
    _atomic_parquet(queries, queries_path)
    _atomic_parquet(passages, passages_path)
    work_path = _resolve_repository_path(
        config, config["paths"]["work_dir"]
    ) / "candidate_cache.json"
    _write_json(
        work_path,
        {
            "passage_extraction_seconds": extraction_seconds,
            "unique_passages": int(len(passages)),
            "train_rows": int(len(candidate_frames["train"])),
            "dev_rows": int(len(candidate_frames["dev"])),
            "dev_top1000_run": str(dev_run_path),
            "dev_top1000_sha256": dev_source_hash,
            "dev_top100_run": str(dev_top100_path),
            "dev_top100_sha256": _sha256(dev_top100_path),
            "retrieval_depth": depth_audits,
        },
    )
    _print_json(
        {
            "status": "ok",
            "train_rows": int(len(candidate_frames["train"])),
            "dev_rows": int(len(candidate_frames["dev"])),
            "queries": int(len(queries)),
            "unique_passages": int(len(passages)),
            "passage_extraction_seconds": extraction_seconds,
        }
    )
    return 0


def _resolve_trec_eval_executable(config: dict[str, Any]) -> Path:
    configured = str(
        config["reproduction_gate"].get("trec_eval_executable", "trec_eval")
    )
    if Path(configured).is_absolute() or "/" in configured:
        candidate = _resolve_repository_path(config, configured)
    else:
        located = shutil.which(configured)
        if located is None:
            raise ValueError(
                "official trec_eval binary was not found on PATH; install NIST "
                "trec_eval or set reproduction_gate.trec_eval_executable"
            )
        candidate = Path(located).resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError(f"trec_eval executable is not an executable file: {candidate}")
    return candidate


def _require_nonempty_regular_file(path: Path, *, label: str) -> None:
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def _run_trec_eval_binary(executable: Path, arguments: list[str]) -> dict[str, Any]:
    command = [str(executable), *arguments]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=1800,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "official trec_eval command failed "
            f"with exit code {exc.returncode}: {' '.join(command)}\n"
            f"stdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}"
        ) from exc
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _probe_trec_eval(executable: Path) -> dict[str, Any]:
    command = [str(executable), "-h"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"trec_eval -h failed with exit code {exc.returncode}: "
            f"{exc.stderr or exc.stdout or ''}"
        ) from exc
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _evaluate_bm25(args: argparse.Namespace) -> int:
    config = _load_retrieval_config(args.config)
    environment = _require_external_environment(config)
    output = _audit_path(config, "reproduction")
    _ensure_writable_targets([output], overwrite=args.overwrite)
    dev_run_path = _resolve_repository_path(config, config["artifacts"]["dev_run"])
    dev_qrels_path = _expected_annotation_paths(config)["dev"]["qrels"]
    _require_nonempty_regular_file(dev_qrels_path, label="official dev qrels")
    _require_nonempty_regular_file(
        dev_run_path, label="untouched official dev top-1000 run"
    )
    executable = _resolve_trec_eval_executable(config)
    version_probe = _probe_trec_eval(executable)
    source_run_sha256 = _sha256(dev_run_path)
    source_run_size_bytes = dev_run_path.stat().st_size

    # The official tool evaluates the untouched 1,000-hit dev run first.
    ndcg_execution = _run_trec_eval_binary(
        executable,
        [
            "-c",
            "-M",
            "100",
            "-m",
            "ndcg_cut.10",
            str(dev_qrels_path),
            str(dev_run_path),
        ],
    )
    recall_execution = _run_trec_eval_binary(
        executable,
        [
            "-c",
            "-m",
            "recall.100",
            str(dev_qrels_path),
            str(dev_run_path),
        ],
    )
    parsed_metrics = {
        "ndcg_cut_10": parse_trec_eval_metric(
            str(ndcg_execution["stdout"]), "ndcg_cut_10"
        ),
        "recall_100": parse_trec_eval_metric(
            str(recall_execution["stdout"]), "recall_100"
        ),
    }
    official_tool = {
        "ndcg_at_10": parsed_metrics["ndcg_cut_10"],
        "recall_at_100": parsed_metrics["recall_100"],
    }
    if (
        _sha256(dev_run_path) != source_run_sha256
        or dev_run_path.stat().st_size != source_run_size_bytes
    ):
        raise RuntimeError("official dev top-1000 run changed during evaluation")

    published = {
        "ndcg_at_10": float(config["reproduction_gate"]["official_ndcg_at_10"]),
        "recall_at_100": float(
            config["reproduction_gate"]["official_recall_at_100"]
        ),
    }
    tolerances = {
        "ndcg_at_10": float(
            config["reproduction_gate"]["ndcg_at_10_tolerance"]
        ),
        "recall_at_100": float(
            config["reproduction_gate"]["recall_at_100_tolerance"]
        ),
    }
    rows = reproduction_rows(official=published, local=official_tool, tolerances=tolerances)
    gate_passed = all(row["status"] == "PASS" for row in rows)
    expected_metrics = {
        "ndcg_cut_10": published["ndcg_at_10"],
        "recall_100": published["recall_at_100"],
    }
    metric_tolerances = {
        "ndcg_cut_10": tolerances["ndcg_at_10"],
        "recall_100": tolerances["recall_at_100"],
    }
    per_metric = [
        {
            "metric": metric,
            "actual": parsed_metrics[metric],
            "expected": expected_metrics[metric],
            "tolerance": metric_tolerances[metric],
            "absolute_difference": abs(
                parsed_metrics[metric] - expected_metrics[metric]
            ),
            "pass": abs(parsed_metrics[metric] - expected_metrics[metric])
            <= metric_tolerances[metric],
        }
        for metric in ("ndcg_cut_10", "recall_100")
    ]
    payload = {
        "status": "PASS" if gate_passed else "FAIL",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "official_source": config["sources"]["pyserini_miracl_2cr"],
        "official_commands": {
            "retrieval": config["reproduction_gate"]["official_retrieval_command"],
            "ndcg": ndcg_execution["command"],
            "recall": recall_execution["command"],
        },
        "trec_eval": {
            "executable_path": str(executable),
            "version": str(config["reproduction_gate"]["trec_eval_version"]),
            "source": str(config["reproduction_gate"]["trec_eval_source"]),
            "version_probe": version_probe,
            "executions": {
                "ndcg_cut_10": ndcg_execution,
                "recall_100": recall_execution,
            },
        },
        "environment": {
            "python": environment["python"],
            "java_major": environment["java_major"],
            "pyserini": environment["pyserini"],
        },
        "metrics": rows,
        "parsed_metrics": parsed_metrics,
        "expected_values": expected_metrics,
        "tolerances": metric_tolerances,
        "per_metric": per_metric,
        "overall_pass": gate_passed,
        "qrels": {
            "path": _portable_repository_path(config, dev_qrels_path),
            "size_bytes": dev_qrels_path.stat().st_size,
            "sha256": _sha256(dev_qrels_path),
        },
        "source_run": _portable_repository_path(config, dev_run_path),
        "source_run_sha256": source_run_sha256,
        "source_run_size_bytes": source_run_size_bytes,
        "source_run_hits_per_query": int(
            config["retrieval"]["retrieval_hits"]["dev"]
        ),
        "candidate_depth_after_reproduction": int(
            config["retrieval"]["candidate_depth"]
        ),
        "official_tool_values": official_tool,
        "gate_passed": gate_passed,
    }
    _write_json(output, payload)
    _print_json(payload)
    return 0 if gate_passed else 1


def _audit_qrels(args: argparse.Namespace) -> int:
    config = _load_retrieval_config(args.config)
    output = _audit_path(config, "qrels")
    _ensure_writable_targets([output], overwrite=args.overwrite)
    annotation_paths = _annotation_paths(config)
    queries = pd.read_parquet(_artifact_path(config, "queries"))
    validate_queries(queries)
    split_reports: dict[str, Any] = {}
    for split in ("train", "dev"):
        split_queries = queries.loc[queries["split"].astype("string").eq(split)]
        qrels = load_qrels(annotation_paths[split]["qrels"], split=split)
        candidates = read_candidate_table(_artifact_path(config, f"{split}_candidates"))
        validate_candidate_schema(candidates)
        split_reports[split] = build_qrels_split_audit(
            queries=split_queries,
            qrels=qrels,
            candidates=candidates,
        )
    payload = {
        "status": "COMPLETED",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "annotations_revision": config["dataset"]["annotations_revision"],
        "splits": split_reports,
        "max_judged_negatives_recommendation": None,
        "note": "No training-sampling decision is made in Phase 1A.",
    }
    _write_json(output, payload)
    _print_json(payload)
    return 0


def _package_phase1(args: argparse.Namespace) -> int:
    config = _load_retrieval_config(args.config)
    archive_path = Path(str(config["archive"]["path"]))
    manifest_path = _audit_path(config, "manifest")
    _ensure_writable_targets([archive_path, manifest_path], overwrite=args.overwrite)

    reproduction_path = _audit_path(config, "reproduction")
    qrels_path = _audit_path(config, "qrels")
    reproduction = _read_json(reproduction_path)
    qrels_audit = _read_json(qrels_path)
    dev_top1000_path = _artifact_path(config, "dev_run")
    dev_top100_path = _artifact_path(config, "dev_top100_run")
    train_run_path = _artifact_path(config, "train_run")
    if reproduction.get("status") != "PASS" or reproduction.get("gate_passed") is not True:
        raise ValueError("a passed official reproduction gate is required before packaging")
    if reproduction.get("source_run") != str(dev_top1000_path):
        raise ValueError("reproduction audit does not identify the configured dev top-1000 run")
    if reproduction.get("source_run_sha256") != _sha256(dev_top1000_path):
        raise ValueError("dev top-1000 run no longer matches its reproduction audit")
    if qrels_audit.get("status") != "COMPLETED":
        raise ValueError("a completed qrels audit is required before packaging")
    if not bool(config["archive"].get("include_runs", False)):
        raise ValueError("Phase 1A archive must include all provenance runs")

    candidates = {
        split: read_candidate_table(_artifact_path(config, f"{split}_candidates"))
        for split in ("train", "dev")
    }
    for frame in candidates.values():
        validate_candidate_schema(frame)
        validate_top_k(
            frame,
            int(config["retrieval"]["candidate_depth"]),
            require_exact=False,
        )
        validate_stable_order(frame)
    queries = pd.read_parquet(_artifact_path(config, "queries"))
    passages = pd.read_parquet(_artifact_path(config, "passages"))
    validate_queries(queries)
    validate_passages(passages)
    all_docids = set(
        pd.concat([frame["docid"] for frame in candidates.values()]).astype("string")
    )
    if all_docids != set(passages["docid"].astype("string")):
        raise ValueError("passages.parquet does not exactly cover candidate docids")
    for split, frame in candidates.items():
        expected = queries.loc[queries["split"].astype("string").eq(split)]
        validate_query_coverage(frame, expected)

    top_k = int(config["retrieval"]["candidate_depth"])
    raw_dev = read_trec_run(str(dev_top1000_path), split="dev")
    _validate_raw_trec_depth(
        raw_dev,
        requested_top_k=int(config["retrieval"]["retrieval_hits"]["dev"]),
    )
    expected_dev_top100 = normalize_bm25_run(
        raw_dev[["split", "query_id", "docid", "bm25_score"]], top_k=top_k
    )
    portable_dev = read_trec_run(str(dev_top100_path), split="dev")
    _validate_raw_trec_depth(portable_dev, requested_top_k=top_k)
    actual_dev_top100 = normalize_bm25_run(
        portable_dev[["split", "query_id", "docid", "bm25_score"]], top_k=top_k
    )
    validate_top_k(actual_dev_top100, top_k, require_exact=False)
    validate_stable_order(actual_dev_top100)
    run_columns = ["split", "query_id", "docid", "bm25_rank", "bm25_score"]
    if not expected_dev_top100[run_columns].equals(actual_dev_top100[run_columns]):
        raise ValueError("dev top-100 run is not the stable truncation of dev top-1000")
    candidate_core = candidates["dev"][run_columns].reset_index(drop=True).copy()
    for column in ("split", "query_id", "docid"):
        candidate_core[column] = candidate_core[column].astype("string")
    candidate_core["bm25_rank"] = candidate_core["bm25_rank"].astype("int64")
    candidate_core["bm25_score"] = candidate_core["bm25_score"].astype("float64")
    if not candidate_core.equals(actual_dev_top100[run_columns]):
        raise ValueError("dev candidate cache does not match the portable dev top-100 run")

    payload_paths = [
        dev_top1000_path,
        dev_top100_path,
        train_run_path,
        _artifact_path(config, "train_candidates"),
        _artifact_path(config, "dev_candidates"),
        _artifact_path(config, "queries"),
        _artifact_path(config, "passages"),
        reproduction_path,
        qrels_path,
    ]
    missing = [str(path) for path in payload_paths if not path.is_file()]
    if missing:
        raise ValueError(f"phase1 package inputs are missing: {', '.join(missing)}")

    work_dir = _resolve_repository_path(config, config["paths"]["work_dir"])
    timings: dict[str, Any] = {}
    for name in ("retrieval_train", "retrieval_dev", "candidate_cache"):
        path = work_dir / f"{name}.json"
        if path.is_file():
            timings[name] = _read_json(path)
    _, current_train_topics = _validate_train_topics(config)
    retrieval_train = timings.get("retrieval_train")
    if not isinstance(retrieval_train, dict) or retrieval_train.get(
        "train_topics"
    ) != current_train_topics:
        raise ValueError(
            "train topics metadata does not match the source used by train retrieval"
        )
    text_lengths = passages["text"].astype("string").str.len()
    runtime_environment = _phase1_environment(config)
    manifest = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": config["dataset"]["name"],
            "language": config["dataset"]["language"],
            "corpus_revision": config["dataset"]["corpus_revision"],
            "annotations_revision": config["dataset"]["annotations_revision"],
            "train_topics": current_train_topics,
        },
        "retrieval": {
            "engine": config["retrieval"]["engine"],
            "index_name": config["retrieval"]["index_name"],
            "analyzer": config["retrieval"]["analyzer"],
            "candidate_depth": config["retrieval"]["candidate_depth"],
            "retrieval_hits": config["retrieval"]["retrieval_hits"],
            "threads": config["retrieval"]["threads"],
            "batch_size": config["retrieval"]["batch_size"],
            "bm25_k1": config["retrieval"]["bm25_k1"],
            "bm25_b": config["retrieval"]["bm25_b"],
            "official_command": config["reproduction_gate"][
                "official_retrieval_command"
            ],
            "actual_command": (
                "python -m rusearchrank.cli run-bm25 "
                "--config configs/retrieval.yaml --split {train|dev}"
            ),
        },
        "environment": {
            "platform": runtime_environment["platform"],
            "platform_detail": runtime_environment["platform_detail"],
            "python": runtime_environment["python"],
            "java_major": runtime_environment["java_major"],
            "java_version_output": runtime_environment["java"]["output"],
            "pyserini": runtime_environment["pyserini"],
            "cpu_count": runtime_environment["cpu_count"],
            "memory_bytes": runtime_environment["memory_bytes"],
        },
        "git": {
            "head": _run_capture(["git", "rev-parse", "HEAD"])["output"] or None,
            "branch": _run_capture(["git", "branch", "--show-current"])["output"] or None,
        },
        "scale": {
            "train_queries": int(candidates["train"]["query_id"].nunique()),
            "dev_queries": int(candidates["dev"]["query_id"].nunique()),
            "train_candidate_rows": int(len(candidates["train"])),
            "dev_candidate_rows": int(len(candidates["dev"])),
            "unique_passages": int(len(passages)),
        },
        "passages": {
            "fraction_with_title": float(passages["title"].astype("string").ne("").mean()),
            "empty_titles": int(passages["title"].astype("string").eq("").sum()),
            "text_length_characters": {
                "min": int(text_lengths.min()),
                "median": float(text_lengths.median()),
                "p90": float(text_lengths.quantile(0.9)),
                "max": int(text_lengths.max()),
            },
        },
        "official_values": {
            "ndcg_at_10": config["reproduction_gate"]["official_ndcg_at_10"],
            "recall_at_100": config["reproduction_gate"]["official_recall_at_100"],
        },
        "local_values": reproduction.get("official_tool_values"),
        "gate_passed": True,
        "retrieval_depth": timings.get("candidate_cache", {}).get(
            "retrieval_depth"
        ),
        "provenance_chain": {
            "flow": [
                "dev top-1000 run",
                "official evaluation PASS",
                "stable truncation (bm25_score DESC, docid ASC; ranks 1..100)",
                "dev top-100 run",
                "three-state candidate cache",
            ],
            "dev_top1000": {
                "path": str(dev_top1000_path),
                "sha256": _sha256(dev_top1000_path),
                "size_bytes": dev_top1000_path.stat().st_size,
            },
            "reproduction_audit": {
                "path": str(reproduction_path),
                "status": reproduction["status"],
                "source_run_sha256": reproduction["source_run_sha256"],
            },
            "dev_top100": {
                "path": str(dev_top100_path),
                "sha256": _sha256(dev_top100_path),
                "size_bytes": dev_top100_path.stat().st_size,
            },
            "candidate_cache": {
                "train": str(_artifact_path(config, "train_candidates")),
                "dev": str(_artifact_path(config, "dev_candidates")),
                "judgment_states": ["relevant", "non_relevant", "unjudged"],
            },
        },
        "timings": timings,
        "artifacts": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in payload_paths
        ],
    }
    _write_json(manifest_path, manifest)
    archive_inputs = [*payload_paths, manifest_path]
    temporary_archive = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in archive_inputs:
            archive.write(path, arcname=path.as_posix())
    expected_archive_names = [path.as_posix() for path in archive_inputs]
    with zipfile.ZipFile(temporary_archive) as archive:
        if archive.namelist() != expected_archive_names:
            temporary_archive.unlink(missing_ok=True)
            raise RuntimeError("ZIP contents do not match the exact Phase 1A allowlist")
    temporary_archive.replace(archive_path)
    _print_json(
        {
            "status": "ok",
            "archive": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
            "contents": [path.as_posix() for path in archive_inputs],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rusearchrank.cli",
        description="RuSearchRank Phase 0 checks and guarded Phase 1A retrieval",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser(
        "inspect-data", help="inspect small official MIRACL Russian samples"
    )
    data_parser.add_argument("--sample-size", type=int, default=3)
    data_parser.add_argument("--timeout", type=float, default=30.0)
    data_parser.add_argument(
        "--output", default=str(DEFAULT_AUDIT_DIR / "miracl_schema.json")
    )
    data_parser.set_defaults(func=_inspect_data)

    annotations_parser = subparsers.add_parser(
        "prepare-annotations",
        help="download and validate revision-pinned MIRACL topics and qrels",
    )
    annotations_parser.add_argument(
        "--config", default=str(DEFAULT_RETRIEVAL_CONFIG)
    )
    annotations_parser.set_defaults(func=_prepare_annotations)

    checkpoint_parser = subparsers.add_parser(
        "inspect-checkpoint", help="inspect tokenizer/config and optionally run model inference"
    )
    checkpoint_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    checkpoint_parser.add_argument("--revision", default="main")
    checkpoint_parser.add_argument("--max-length", type=int, default=256)
    checkpoint_parser.add_argument(
        "--with-model",
        action="store_true",
        help="download model weights and run CPU/MPS inference",
    )
    checkpoint_parser.add_argument(
        "--output", default=str(DEFAULT_AUDIT_DIR / "checkpoint_contract.json")
    )
    checkpoint_parser.set_defaults(func=_inspect_checkpoint)

    validation_parser = subparsers.add_parser(
        "validate-candidates", help="validate a CSV, TSV, or Parquet candidate table"
    )
    validation_parser.add_argument("path")
    validation_parser.add_argument(
        "--config", default=str(DEFAULT_RETRIEVAL_CONFIG)
    )
    validation_parser.set_defaults(func=_validate_candidates)

    environment_parser = subparsers.add_parser(
        "environment-report", help="capture Python, Java, platform, disk, and MPS facts"
    )
    environment_parser.add_argument(
        "--output", default=str(DEFAULT_AUDIT_DIR / "environment.json")
    )
    environment_parser.set_defaults(func=_environment_report)

    linux_parser = subparsers.add_parser(
        "inspect-linux-environment",
        help="validate the exact external retrieval environment",
    )
    linux_parser.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    linux_parser.add_argument(
        "--check-index",
        action="store_true",
        help="download/open the official index and run one smoke query",
    )
    linux_parser.set_defaults(func=_inspect_linux_environment)

    bm25_parser = subparsers.add_parser(
        "run-bm25", help="run guarded full Pyserini BM25 retrieval"
    )
    bm25_parser.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    bm25_parser.add_argument(
        "--split", choices=("train", "dev", "all"), default="all"
    )
    bm25_parser.add_argument("--overwrite", action="store_true")
    bm25_parser.set_defaults(func=_run_bm25)

    cache_parser = subparsers.add_parser(
        "build-candidate-cache",
        help="join qrels and stream only top-100 candidate passages",
    )
    cache_parser.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    cache_parser.add_argument("--overwrite", action="store_true")
    cache_parser.set_defaults(func=_build_candidate_cache)

    evaluate_parser = subparsers.add_parser(
        "evaluate-bm25", help="compare internal metrics with official Pyserini evaluation"
    )
    evaluate_parser.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    evaluate_parser.add_argument("--overwrite", action="store_true")
    evaluate_parser.set_defaults(func=_evaluate_bm25)

    audit_parser = subparsers.add_parser(
        "audit-qrels", help="write qrels and candidate-coverage audit"
    )
    audit_parser.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    audit_parser.add_argument("--overwrite", action="store_true")
    audit_parser.set_defaults(func=_audit_qrels)

    package_parser = subparsers.add_parser(
        "package-phase1", help="validate and package only portable Phase 1 results"
    )
    package_parser.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    package_parser.add_argument("--overwrite", action="store_true")
    package_parser.set_defaults(func=_package_phase1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
