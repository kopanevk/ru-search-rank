"""Phase 0 checks and the guarded Linux/Colab Phase 1A command line."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Sequence
import zipfile

import pandas as pd
import pyarrow.parquet as pq
import yaml

from .corpus import (
    CORPUS_FIELDS,
    CorpusAccessError,
    HubShardSource,
    LocalShardSource,
    MissingCandidatePassagesError,
    ShardSource,
    assert_no_dataset_script,
    extract_candidate_passages,
    iter_shard_rows,
    resolve_hub_shards,
    shard_names,
    sort_shard_names,
)
from .data import (
    CANDIDATE_COLUMNS,
    download_annotation_files,
    inspect_miracl_ru,
    load_qrels,
    load_topics,
    read_candidate_table,
    validate_candidate_schema,
    validate_passages,
    validate_queries,
)
from . import data as data_module
from .evaluation import (
    assert_candidate_set_invariant,
    build_qrels_split_audit,
    classify_bm25_tie_break_audit,
    evaluate_bm25_metrics,
    paired_ranking_comparison,
    parse_trec_eval_metric,
    parse_trec_eval_per_query,
    raw_score_tie_statistics,
    reproduction_rows,
    sparse_judgment_diagnostics,
    stratified_delta_summary,
)
from . import rerank as rerank_module
from . import training_data as training_data_module
from . import training as training_module
from . import phase3_eval as phase3_eval_module
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
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(
            f"diagnostic temporary JSON already exists and was preserved: {temporary}"
        )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _read_json(temporary)
        temporary.replace(path)
    except Exception as exc:
        raise RuntimeError(
            f"failed to write JSON atomically; diagnostic temporary file: {temporary}"
        ) from exc


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
    repository_root = _repository_root(config)
    disk = shutil.disk_usage(repository_root)
    java_home = os.environ.get("JAVA_HOME")
    return {
        "platform": platform.system(),
        "platform_detail": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "java": java,
        "java_major": _java_major_version(str(java.get("output", ""))),
        "java_home": java_home,
        "java_home_executable": (
            str(Path(java_home) / "bin/java") if java_home else None
        ),
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
    java_home = report.get("java_home")
    java_home_ok = bool(java_home) and (Path(str(java_home)) / "bin/java").is_file()
    pyserini_ok = report["pyserini"] == str(expected["pyserini"])
    disk_required = int(expected["minimum_free_disk_gib"]) * 1024**3
    disk_ok = int(report["disk_free_bytes"]) >= disk_required
    checks = {
        "platform_linux": platform.system() == "Linux",
        f"python_{expected['python']}": python_ok,
        f"java_{expected['java']}": java_ok,
        "java_home_bin_java": java_home_ok,
        f"pyserini_{expected['pyserini']}": pyserini_ok,
        f"free_disk_gib_at_least_{expected['minimum_free_disk_gib']}": disk_ok,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            FULL_RETRIEVAL_ERROR
            + " Failed checks: "
            + ", ".join(failed)
            + ". Observed: "
            + json.dumps(
                {
                    "platform": report["platform"],
                    "python": report["python"],
                    "java_major": report["java_major"],
                    "java_home": report["java_home"],
                    "pyserini": report["pyserini"],
                    "disk_free_bytes": report["disk_free_bytes"],
                },
                ensure_ascii=False,
            )
        )
    return report


class StageError(RuntimeError):
    """A stage failure that names its command, paths, cause, and recovery cell."""

    def __init__(
        self,
        *,
        stage: str,
        command: str,
        inputs: dict[str, str],
        outputs: dict[str, str],
        root_cause: str,
        reusable: str,
        repeat_cell: str,
        rollback_status: str | None = None,
    ) -> None:
        self.stage = stage
        payload = {
            "stage": stage,
            "command": command,
            "input_paths": inputs,
            "output_or_temp_paths": outputs,
            "root_cause": root_cause,
            "safe_to_reuse": reusable,
            "repeat_cell": repeat_cell,
        }
        if rollback_status is not None:
            payload["rollback_status"] = rollback_status
        super().__init__(json.dumps(payload, ensure_ascii=False, indent=2))


def _corpus_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Read and validate the pinned static-shard corpus contract."""

    dataset = config["dataset"]
    language = str(dataset["language"])
    shard_count = int(dataset["corpus_shard_count"])
    expected = shard_names(language, shard_count)
    configured = [str(name) for name in dataset["corpus_shard_files"]]
    if configured != expected:
        raise ValueError(
            "dataset.corpus_shard_files must list the official shards in numeric "
            f"order 0..{shard_count - 1}; expected {expected[:3]}... got "
            f"{configured[:3]}..."
        )
    assert_no_dataset_script(configured)
    if str(dataset.get("corpus_repo_type", "dataset")) != "dataset":
        raise ValueError("dataset.corpus_repo_type must be 'dataset'")
    revision = str(dataset["corpus_revision"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "dataset.corpus_revision must be an immutable 40-character commit SHA; "
            f"got {revision!r}"
        )
    batch_rows = int(dataset["corpus_passage_batch_rows"])
    if batch_rows <= 0:
        raise ValueError("dataset.corpus_passage_batch_rows must be positive")
    cache_dir = dataset.get("corpus_cache_dir")
    local_dir = dataset.get("corpus_local_dir")
    return {
        "repo_id": str(dataset["corpus_source"]),
        "revision": revision,
        "language": language,
        "shard_count": shard_count,
        "shards": configured,
        "batch_rows": batch_rows,
        "cache_dir": (
            _resolve_repository_path(config, str(cache_dir)) if cache_dir else None
        ),
        "local_dir": (
            _resolve_repository_path(config, str(local_dir)) if local_dir else None
        ),
    }


def _corpus_source(config: dict[str, Any]) -> ShardSource:
    settings = _corpus_settings(config)
    if settings["local_dir"] is not None:
        # Explicit offline override for already-materialized official shards.
        return LocalShardSource(
            language=settings["language"],
            shard_count=settings["shard_count"],
            directory=settings["local_dir"],
        )
    cache_dir = settings["cache_dir"]
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return HubShardSource(
        language=settings["language"],
        shard_count=settings["shard_count"],
        repo_id=settings["repo_id"],
        revision=settings["revision"],
        cache_dir=cache_dir,
    )


def _artifact_path(config: dict[str, Any], key: str) -> Path:
    return _resolve_repository_path(config, config["artifacts"][key])


def _audit_path(config: dict[str, Any], key: str) -> Path:
    return _resolve_repository_path(config, config["audits"][key])


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


def _annotation_paths_for_split(
    config: dict[str, Any], split: str
) -> dict[str, Path]:
    """Download and validate exactly one annotation split.

    Phase 1 keeps the historical all-splits command as its default.  Phase 3
    uses this narrower path so the train annotations can be materialized while
    final-evaluation annotations remain unopened and absent until selection.
    """

    if split != "train":
        raise ValueError("the isolated single-split path only materializes train")
    raw_dir = _resolve_repository_path(config, config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "topics": raw_dir / f"topics.miracl-v1.0-ru-{split}.tsv",
        "qrels": raw_dir / f"qrels.miracl-v1.0-ru-{split}.tsv",
    }
    pending: dict[Path, Path] = {}
    for kind, path in paths.items():
        if path.is_file():
            continue
        payload = data_module._fetch_text(  # type: ignore[attr-defined]
            str(config["dataset"][kind][split]), 120.0
        )
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        if temporary.exists():
            raise RuntimeError(
                f"diagnostic annotation download already exists: {temporary}"
            )
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        pending[path] = temporary

    topics_source = pending.get(paths["topics"], paths["topics"])
    qrels_source = pending.get(paths["qrels"], paths["qrels"])
    topics = load_topics(topics_source, split=split)
    qrels = load_qrels(qrels_source, split=split)
    expected_topics = int(config["dataset"]["expected_rows"][f"{split}_queries"])
    expected_qrels = int(config["dataset"]["expected_rows"][f"{split}_qrels"])
    if len(topics) != expected_topics or len(qrels) != expected_qrels:
        raise RuntimeError(
            f"official {split} annotations have unexpected row counts: "
            f"queries={len(topics)} (expected {expected_topics}), "
            f"qrels={len(qrels)} (expected {expected_qrels})"
        )
    if qrels.duplicated(["query_id", "docid"], keep=False).any():
        raise RuntimeError(f"official {split} qrels contain duplicate pairs")
    if (qrels["relevance_grade"] < 0).any():
        raise RuntimeError(f"official {split} qrels contain negative grades")
    for final_path, temporary in pending.items():
        temporary.replace(final_path)
    return paths


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(
            f"diagnostic temporary Parquet already exists and was preserved: {temporary}"
        )
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        reloaded = pd.read_parquet(temporary)
        if len(reloaded) != len(frame) or list(reloaded.columns) != list(frame.columns):
            raise ValueError("temporary Parquet row count or schema changed during write")
        temporary.replace(path)
    except Exception as exc:
        raise RuntimeError(
            f"failed to write Parquet atomically; diagnostic temporary file: {temporary}"
        ) from exc


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


def _run_streaming_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> dict[str, Any]:
    """Stream both pipes live and persist their complete contents and return code."""

    print("+", shlex.join(command), flush=True)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        payload = {
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
        _write_json(log_path, payload)
        raise RuntimeError(
            f"failed to start command; complete log: {log_path}: {exc}"
        ) from exc

    def pump(stream: Any, destination: Any, collected: list[str]) -> None:
        for line in iter(stream.readline, ""):
            collected.append(line)
            print(line, end="", file=destination, flush=True)
        stream.close()

    threads = [
        threading.Thread(
            target=pump,
            args=(process.stdout, sys.stdout, stdout_lines),
            daemon=True,
        ),
        threading.Thread(
            target=pump,
            args=(process.stderr, sys.stderr, stderr_lines),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    returncode = process.wait()
    for thread in threads:
        thread.join()
    payload = {
        "command": command,
        "cwd": str(cwd),
        "returncode": returncode,
        "stdout": "".join(stdout_lines),
        "stderr": "".join(stderr_lines),
    }
    _write_json(log_path, payload)
    return payload


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
    requested_split = str(getattr(args, "split", "all"))
    if requested_split == "all":
        downloaded = _annotation_paths(config)
    else:
        downloaded = {
            requested_split: _annotation_paths_for_split(config, requested_split)
        }
    if requested_split != "all":
        split_paths = downloaded[requested_split]
        configured_train = _expected_annotation_paths(config)["train"]["topics"]
        if split_paths["topics"].resolve() != configured_train.resolve():
            raise ValueError(
                "dataset.train_topics_path must identify the official downloaded "
                "train TSV"
            )
        _, train_metadata = _validate_train_topics(config)
        payload: dict[str, Any] = {
            "status": "ok",
            "split": "train",
            "train_topics": train_metadata,
            "train_qrels": _portable_repository_path(config, split_paths["qrels"]),
        }
        _print_json(payload)
        return 0

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


def _validate_retrieval_config_contract(config: dict[str, Any]) -> dict[str, Any]:
    required_nested = {
        "dataset": {
            "name",
            "language",
            "corpus_source",
            "corpus_revision",
            "corpus_repo_type",
            "corpus_shard_count",
            "corpus_shard_files",
            "corpus_cache_dir",
            "corpus_passage_batch_rows",
            "annotations_revision",
            "train_topics_path",
            "topics",
            "qrels",
            "expected_rows",
        },
        "environment": {
            "python",
            "java",
            "pyserini",
            "minimum_free_disk_gib",
        },
        "retrieval": {
            "engine",
            "index_name",
            "candidate_depth",
            "retrieval_hits",
            "threads",
            "batch_size",
            "bm25_k1",
            "bm25_b",
        },
        "artifacts": {
            "train_run",
            "dev_run",
            "dev_top100_run",
            "train_candidates",
            "dev_candidates",
            "queries",
            "passages",
        },
        "audits": {"reproduction", "qrels", "manifest"},
        "reproduction_gate": {
            "trec_eval_executable",
            "trec_eval_version",
            "official_topic",
            "official_ndcg_at_10",
            "official_recall_at_100",
            "ndcg_at_10_tolerance",
            "recall_at_100_tolerance",
        },
        "archive": {"path", "include_runs"},
        "paths": {"repository_root", "raw_dir", "work_dir"},
    }
    for section, required_keys in required_nested.items():
        value = config.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"retrieval config section {section!r} must be a mapping")
        missing = sorted(required_keys.difference(value))
        if missing:
            raise ValueError(
                f"retrieval config section {section!r} is missing: {', '.join(missing)}"
            )
    if config.get("splits") != ["train", "dev"]:
        raise ValueError("retrieval config splits must be [train, dev]")
    for name in ("topics", "qrels"):
        if set(config["dataset"][name]) != {"train", "dev"}:
            raise ValueError(f"dataset.{name} must contain exactly train and dev")
    expected_row_keys = {
        "train_queries",
        "dev_queries",
        "train_qrels",
        "dev_qrels",
    }
    if set(config["dataset"]["expected_rows"]) != expected_row_keys:
        raise ValueError("dataset.expected_rows has an unexpected key set")
    if any(
        int(config["dataset"]["expected_rows"][key]) <= 0
        for key in expected_row_keys
    ):
        raise ValueError("all dataset.expected_rows values must be positive")
    root = _repository_root(config)
    if not root.is_dir():
        raise ValueError(f"configured repository root does not exist: {root}")
    expected_hits = config["retrieval"].get("retrieval_hits")
    if expected_hits != {"train": 100, "dev": 1000}:
        raise ValueError(
            "retrieval_hits must preserve train=100 and official dev=1000"
        )
    if int(config["retrieval"].get("candidate_depth", 0)) != 100:
        raise ValueError("candidate_depth must be 100")
    if config["retrieval"].get("index_name") != "miracl-v1.0-ru":
        raise ValueError("official index_name must be miracl-v1.0-ru")
    if config["reproduction_gate"].get("official_topic") != "miracl-v1.0-ru-dev":
        raise ValueError("official dev topic ID must be miracl-v1.0-ru-dev")
    _corpus_settings(config)

    configured_paths: list[Path] = []
    configured_paths.extend(_artifact_path(config, key) for key in config["artifacts"])
    configured_paths.extend(_audit_path(config, key) for key in config["audits"])
    configured_paths.append(_resolve_repository_path(config, config["archive"]["path"]))
    for path in configured_paths:
        _portable_repository_path(config, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=".rusearchrank-write-probe-",
                delete=True,
            ) as probe:
                probe.write(b"ok")
                probe.flush()
        except OSError as exc:
            raise ValueError(f"output directory is not writable: {path.parent}") from exc
    return {
        "repository_root": str(root),
        "config": _portable_repository_path(config, Path(config["_config_path"])),
        "output_directories": sorted({str(path.parent) for path in configured_paths}),
    }


def _validate_official_annotations(
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    paths = _expected_annotation_paths(config)
    report: dict[str, Any] = {}
    for split in ("train", "dev"):
        topics_path = paths[split]["topics"]
        qrels_path = paths[split]["qrels"]
        _require_nonempty_regular_file(topics_path, label=f"official {split} topics")
        _require_nonempty_regular_file(qrels_path, label=f"official {split} qrels")
        topics = load_topics(topics_path, split=split)
        qrels = load_qrels(qrels_path, split=split)
        expected_topics = int(config["dataset"]["expected_rows"][f"{split}_queries"])
        expected_qrels = int(config["dataset"]["expected_rows"][f"{split}_qrels"])
        if len(topics) != expected_topics:
            raise ValueError(
                f"official {split} topics have {len(topics)} rows; expected {expected_topics}"
            )
        if len(qrels) != expected_qrels:
            raise ValueError(
                f"official {split} qrels have {len(qrels)} rows; expected {expected_qrels}"
            )
        if qrels[["split", "query_id", "docid"]].astype("string").duplicated().any():
            raise ValueError(f"official {split} qrels contain duplicate query-doc pairs")
        if set(qrels["relevance_grade"].astype("int64")) - {0, 1}:
            raise ValueError(
                f"official {split} qrels contain unsupported relevance values"
            )
        report[split] = {
            "topics": _file_metadata(config, topics_path),
            "qrels": _file_metadata(config, qrels_path),
            "query_count": len(topics),
            "qrels_count": len(qrels),
        }
    _, train_topics = _validate_train_topics(config)
    report["train_topics_contract"] = train_topics
    report["dev_registered_topic"] = config["reproduction_gate"]["official_topic"]
    return paths, report


def _preflight_retrieval(
    config: dict[str, Any], *, check_index: bool
) -> dict[str, Any]:
    config_report = _validate_retrieval_config_contract(config)
    environment = _require_external_environment(config)
    if importlib.util.find_spec("pyserini") is None:
        raise ValueError("Pyserini cannot be imported from the active Python environment")
    try:
        from pyserini.search import get_topics

        registered_dev_topics = get_topics(
            str(config["reproduction_gate"]["official_topic"])
        )
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Pyserini cannot resolve the registered official dev topics "
            f"{config['reproduction_gate']['official_topic']!r}"
        ) from exc
    expected_dev_queries = int(config["dataset"]["expected_rows"]["dev_queries"])
    if not registered_dev_topics or len(registered_dev_topics) != expected_dev_queries:
        raise ValueError(
            "registered official dev topics have an unexpected query count: "
            f"{len(registered_dev_topics) if registered_dev_topics else 0}; "
            f"expected {expected_dev_queries}"
        )
    trec_eval = _resolve_trec_eval_executable(config)
    trec_eval_probe = _probe_trec_eval(trec_eval)
    _, annotations = _validate_official_annotations(config)
    report: dict[str, Any] = {
        "stage": "retrieval",
        "config": config_report,
        "environment": environment,
        "trec_eval": {
            "path": str(trec_eval),
            "expected_version": str(config["reproduction_gate"]["trec_eval_version"]),
            "probe": trec_eval_probe,
        },
        "annotations": annotations,
        "registered_dev_topics": {
            "id": config["reproduction_gate"]["official_topic"],
            "query_count": len(registered_dev_topics),
        },
        "index_name": config["retrieval"]["index_name"],
        "index_checked": check_index,
    }
    if check_index:
        report["index_smoke"] = smoke_prebuilt_index(
            index_name=str(config["retrieval"]["index_name"]),
            language=str(config["dataset"]["language"]),
            query_text="Когда начался Карибский кризис?",
        )
    return report


def _preflight_candidate_cache(config: dict[str, Any]) -> dict[str, Any]:
    retrieval_report = _preflight_retrieval(config, check_index=False)
    paths, _ = _validate_official_annotations(config)
    run_reports: dict[str, Any] = {}
    for split, key in (("train", "train_run"), ("dev", "dev_run")):
        run_path = _artifact_path(config, key)
        topics = load_topics(paths[split]["topics"], split=split)
        requested_top_k = int(config["retrieval"]["retrieval_hits"][split])
        raw, normalized, depth = _validate_retrieval_run(
            run_path,
            split=split,
            queries=topics,
            requested_top_k=requested_top_k,
        )
        _require_no_zero_hits(
            depth,
            context=f"candidate-cache preflight rejected {split} run",
        )
        validate_query_coverage(normalized, topics)
        run_reports[split] = {
            **_file_metadata(config, run_path),
            "rows": len(raw),
            "retrieval_depth": depth,
        }
    reproduction, audit_path, dev_hash = _require_reproduction_gate(config)
    return {
        "stage": "candidate-cache",
        "retrieval_preflight": retrieval_report,
        "runs": run_reports,
        "reproduction": {
            "path": _portable_repository_path(config, audit_path),
            "status": reproduction["status"],
            "overall_pass": reproduction["overall_pass"],
            "dev_top1000_sha256": dev_hash,
        },
    }


def _preflight_command(args: argparse.Namespace) -> int:
    if args.stage == "rerank":
        if args.check_index:
            raise ValueError("--check-index is only valid for retrieval preflight")
        rerank_config = rerank_module.load_rerank_config(args.config)
        report = rerank_module.preflight_rerank(rerank_config)
        _print_json({"status": "PASS", "preflight": report})
        return 0
    config = _load_retrieval_config(args.config)
    if args.stage == "retrieval":
        report = _preflight_retrieval(config, check_index=args.check_index)
    elif args.stage == "candidate-cache":
        if args.check_index:
            raise ValueError("--check-index is only valid for retrieval preflight")
        report = _preflight_candidate_cache(config)
    else:
        if args.check_index:
            raise ValueError("--check-index is only valid for retrieval preflight")
        report = _preflight_candidate_cache(config)
        report["stage"] = "package"
        for key in ("train_candidates", "dev_candidates", "queries", "passages"):
            _require_nonempty_regular_file(
                _artifact_path(config, key), label=f"package input {key}"
            )
        for key in ("qrels",):
            _require_nonempty_regular_file(
                _audit_path(config, key), label=f"package audit {key}"
            )
    _print_json({"status": "PASS", "preflight": report})
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


def _validate_retrieval_run(
    path: Path,
    *,
    split: str,
    queries: pd.DataFrame,
    requested_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw_run = read_trec_run(str(path), split=split)
    _validate_raw_trec_depth(raw_run, requested_top_k=requested_top_k)
    normalized = normalize_bm25_run(
        raw_run[["split", "query_id", "docid", "bm25_score"]]
    )
    depth_audit = build_retrieval_depth_audit(
        normalized,
        queries,
        requested_top_k=requested_top_k,
    )
    return raw_run, normalized, depth_audit


_RUN_CORE_COLUMNS = ["split", "query_id", "docid", "bm25_rank", "bm25_score"]


def _ranked_raw_trec(raw_run: pd.DataFrame) -> pd.DataFrame:
    ranked = raw_run.rename(columns={"source_rank": "bm25_rank"})[
        _RUN_CORE_COLUMNS
    ].copy()
    for column in ("split", "query_id", "docid"):
        ranked[column] = ranked[column].astype("string")
    ranked["bm25_rank"] = ranked["bm25_rank"].astype("int64")
    ranked["bm25_score"] = ranked["bm25_score"].astype("float64")
    return ranked


def _expected_stable_truncation(
    source: Path,
    *,
    split: str,
    top_k: int,
    source_top_k: int | None,
) -> pd.DataFrame:
    raw = read_trec_run(str(source), split=split)
    raw_limit = source_top_k or int(pd.to_numeric(raw["source_rank"]).max())
    _validate_raw_trec_depth(raw, requested_top_k=raw_limit)
    expected = normalize_bm25_run(
        raw[["split", "query_id", "docid", "bm25_score"]],
        top_k=top_k,
    )
    validate_top_k(expected, top_k, require_exact=False)
    validate_stable_order(expected)
    return expected


def _validate_stable_trec_derivation(
    source: Path,
    target: Path,
    *,
    split: str,
    top_k: int,
    source_top_k: int | None = None,
) -> pd.DataFrame:
    """Prove that ``target`` is the deterministic top-k derived from ``source``."""

    expected = _expected_stable_truncation(
        source,
        split=split,
        top_k=top_k,
        source_top_k=source_top_k,
    )
    raw_target = read_trec_run(str(target), split=split)
    _validate_raw_trec_depth(raw_target, requested_top_k=top_k)
    actual = _ranked_raw_trec(raw_target)
    validate_top_k(actual, top_k, require_exact=False)
    validate_stable_order(actual)
    expected_core = expected[_RUN_CORE_COLUMNS].reset_index(drop=True).copy()
    expected_core["bm25_score"] = expected_core["bm25_score"].map(
        lambda value: float(f"{float(value):.8f}")
    )
    if not expected_core.equals(actual[_RUN_CORE_COLUMNS].reset_index(drop=True)):
        raise ValueError(
            f"{target} is not the stable score-desc/docid-asc top-{top_k} "
            f"derivation of {source}"
        )
    return actual


def _run_bm25(args: argparse.Namespace) -> int:
    config = _load_retrieval_config(args.config)
    _preflight_retrieval(config, check_index=False)
    splits = list(config["splits"]) if args.split == "all" else [args.split]
    run_keys = {"train": "train_run", "dev": "dev_run"}
    targets = [_artifact_path(config, run_keys[split]) for split in splits]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
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
        work_path = _resolve_repository_path(
            config, config["paths"]["work_dir"]
        ) / f"retrieval_{split}.json"
        command_target = target.with_suffix(target.suffix + ".tmp")
        command = _build_bm25_command(
            config,
            split=split,
            target=command_target,
            topics_argument=topics_argument,
        )
        if target.exists() and not args.overwrite:
            try:
                raw_run, normalized, depth_audit = _validate_retrieval_run(
                    target,
                    split=split,
                    queries=queries,
                    requested_top_k=retrieval_hits,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                raise ValueError(
                    f"existing {split} run is invalid and was preserved at "
                    f"{_portable_repository_path(config, target)}. Review it, then use "
                    f"--overwrite to rerun retrieval. Validation error: {exc}"
                ) from exc
            zero_hit_ids = _zero_hit_query_ids(depth_audit)
            if zero_hit_ids:
                _write_json(
                    work_path,
                    {
                        "status": "FAIL_ZERO_HIT_QUERIES",
                        "action": "rejected_existing_run",
                        "split": split,
                        "run": _portable_repository_path(config, target),
                        "run_sha256": _sha256(target),
                        "depth_audit": depth_audit,
                    },
                )
                raise ValueError(
                    f"existing {split} run contains zero-hit query_id values: "
                    + ", ".join(zero_hit_ids)
                    + f". Run preserved at {_portable_repository_path(config, target)}"
                )
            validate_query_coverage(normalized, queries)
            run_report = {
                "status": "PASS",
                "action": "reused_valid_run",
                "split": split,
                "seconds": 0.0,
                "queries": int(len(queries)),
                "rows": int(len(raw_run)),
                "hits_per_query": retrieval_hits,
                "topics_argument": topics_argument,
                "train_topics": train_topics_metadata,
                "command_if_rebuilt": command,
                "run": _portable_repository_path(config, target),
                "run_sha256": _sha256(target),
                "run_size_bytes": target.stat().st_size,
                "raw_validation_path": None,
                "depth_audit": depth_audit,
            }
            _write_json(work_path, run_report)
            results[split] = {
                "action": "reused_valid_run",
                "run": _portable_repository_path(config, target),
                "queries": int(len(queries)),
                "rows": int(len(raw_run)),
                "hits_per_query": retrieval_hits,
                "seconds": 0.0,
                "depth_audit": depth_audit,
            }
            continue

        temporary = target.with_suffix(target.suffix + ".tmp")
        if temporary.exists():
            raise ValueError(
                f"diagnostic temporary run already exists: "
                f"{_portable_repository_path(config, temporary)}. Preserve or move it "
                "before starting a new retrieval."
            )
        started = time.perf_counter()
        log_path = _resolve_repository_path(
            config, config["paths"]["work_dir"]
        ) / f"retrieval_{split}_command.json"
        result = _run_streaming_process(
            command,
            cwd=_repository_root(config),
            env=os.environ.copy(),
            log_path=log_path,
        )
        if result["returncode"] != 0:
            preserved = (
                _portable_repository_path(config, temporary)
                if temporary.exists()
                else "not created"
            )
            raise RuntimeError(
                "official Pyserini retrieval failed with return code "
                f"{result['returncode']}; complete log: "
                f"{_portable_repository_path(config, log_path)}; temporary run: {preserved}"
            )
        elapsed = time.perf_counter() - started
        if not temporary.is_file():
            raise RuntimeError("official Pyserini retrieval did not create its run file")
        try:
            raw_run, validated, depth_audit = _validate_retrieval_run(
                temporary,
                split=split,
                queries=queries,
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
                "BM25 run contains zero-hit query_id values: "
                + ", ".join(zero_hit_ids)
                + ". Raw run preserved at "
                + _portable_repository_path(config, temporary)
            )
        validate_query_coverage(validated, queries)
        # Preserve Pyserini's raw output byte-for-byte for official evaluation.
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(target)
        run_report["status"] = "PASS"
        run_report["action"] = "retrieved"
        run_report["run_sha256"] = _sha256(target)
        run_report["run_size_bytes"] = target.stat().st_size
        run_report["command_log"] = _portable_repository_path(config, log_path)
        run_report["raw_validation_path"] = None
        _write_json(work_path, run_report)
        results[split] = {
            "action": "retrieved",
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

    _require_nonempty_regular_file(source, label=f"source {split} TREC run")
    source_hash = _sha256(source)
    normalized = _expected_stable_truncation(
        source,
        split=split,
        top_k=top_k,
        source_top_k=source_top_k,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(
            f"diagnostic temporary TREC run already exists and was preserved: {temporary}"
        )
    write_trec_run(
        normalized,
        str(temporary),
        tag="rusearchrank-stable-top100",
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    portable = _validate_stable_trec_derivation(
        source,
        temporary,
        split=split,
        top_k=top_k,
        source_top_k=source_top_k,
    )
    if _sha256(source) != source_hash:
        raise RuntimeError(
            "source dev top-1000 run changed during stable truncation; "
            f"diagnostic temporary file preserved at {temporary}"
        )
    temporary.replace(target)
    return portable


def _load_or_create_stable_trec_run(
    source: Path,
    target: Path,
    *,
    split: str,
    top_k: int,
    source_top_k: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, str]:
    """Reuse a proven derived run, or create it without touching the raw source."""

    existed_before = target.exists()
    if existed_before:
        try:
            validated = _validate_stable_trec_derivation(
                source,
                target,
                split=split,
                top_k=top_k,
                source_top_k=source_top_k,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            if not overwrite:
                raise ValueError(
                    f"existing derived run is invalid or stale and was preserved: {target}. "
                    "Use --overwrite only after reviewing it. "
                    f"Validation error: {exc}"
                ) from exc
        else:
            return validated, "reused_valid"
    created = _stable_truncate_trec_run(
        source,
        target,
        split=split,
        top_k=top_k,
        source_top_k=source_top_k,
    )
    return created, "recreated" if existed_before else "created"


def _file_metadata(config: dict[str, Any], path: Path) -> dict[str, Any]:
    _require_nonempty_regular_file(path, label="provenance input")
    return {
        "path": _portable_repository_path(config, path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _require_reproduction_gate(
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path, str]:
    audit_path = _audit_path(config, "reproduction")
    reproduction = _read_json(audit_path)
    if (
        reproduction.get("status") != "PASS"
        or reproduction.get("gate_passed") is not True
        or reproduction.get("overall_pass") is not True
    ):
        raise ValueError(
            "candidate cache requires status=PASS and overall_pass=true in the "
            "official dev reproduction audit"
        )
    dev_run_path = _artifact_path(config, "dev_run")
    _require_nonempty_regular_file(
        dev_run_path, label="untouched official dev top-1000 run"
    )
    portable_dev_run = _portable_repository_path(config, dev_run_path)
    if reproduction.get("source_run") != portable_dev_run:
        raise ValueError(
            "candidate cache requires an audit of the configured dev top-1000 run"
        )
    current_hash = _sha256(dev_run_path)
    if reproduction.get("source_run_sha256") != current_hash:
        raise ValueError(
            "candidate cache requires a reproduction gate for the current dev run; "
            "the top-1000 SHA-256 changed after evaluation"
        )
    if reproduction.get("source_run_size_bytes") != dev_run_path.stat().st_size:
        raise ValueError(
            "candidate cache requires a reproduction audit with the current dev run size"
        )
    return reproduction, audit_path, current_hash


def _candidate_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        "train_candidates": _artifact_path(config, "train_candidates"),
        "dev_candidates": _artifact_path(config, "dev_candidates"),
        "queries": _artifact_path(config, "queries"),
        "passages": _artifact_path(config, "passages"),
    }


def _candidate_input_metadata(
    config: dict[str, Any],
    annotation_paths: dict[str, dict[str, Path]],
    *,
    reproduction_path: Path,
    dev_top100_path: Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        "train_run": _artifact_path(config, "train_run"),
        "dev_top1000_run": _artifact_path(config, "dev_run"),
        "dev_top100_run": dev_top100_path,
        "reproduction_audit": reproduction_path,
        "train_topics": annotation_paths["train"]["topics"],
        "dev_topics": annotation_paths["dev"]["topics"],
        "train_qrels": annotation_paths["train"]["qrels"],
        "dev_qrels": annotation_paths["dev"]["qrels"],
    }
    return {name: _file_metadata(config, path) for name, path in paths.items()}


def _candidate_locations(
    candidate_frames: dict[str, pd.DataFrame],
    *,
    only_docids: set[str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    context: dict[str, list[dict[str, str]]] = {}
    for frame in candidate_frames.values():
        for row in frame[["split", "query_id", "docid"]].itertuples(index=False):
            docid = str(row.docid)
            if only_docids is not None and docid not in only_docids:
                continue
            locations = context.setdefault(docid, [])
            if len(locations) < 3:
                locations.append(
                    {"split": str(row.split), "query_id": str(row.query_id)}
                )
    return context


def _validate_candidate_cache_artifacts(
    config: dict[str, Any],
    *,
    dev_top100: pd.DataFrame,
    expected_inputs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    outputs = _candidate_output_paths(config)
    candidates = {
        split: read_candidate_table(outputs[f"{split}_candidates"])
        for split in ("train", "dev")
    }
    top_k = int(config["retrieval"]["candidate_depth"])
    for split, frame in candidates.items():
        validate_candidate_schema(frame)
        validate_top_k(frame, top_k, require_exact=False)
        validate_stable_order(frame)
        if set(frame["split"].astype("string")) != {split}:
            raise ValueError(f"{split} candidate Parquet contains another split")
    queries = pd.read_parquet(outputs["queries"])
    passages = pd.read_parquet(outputs["passages"])
    validate_queries(queries)
    validate_passages(passages)
    for split, frame in candidates.items():
        expected_queries = queries.loc[queries["split"].astype("string").eq(split)]
        validate_query_coverage(frame, expected_queries)
    all_docids = set(
        pd.concat([frame["docid"] for frame in candidates.values()]).astype("string")
    )
    passage_docids = set(passages["docid"].astype("string"))
    if all_docids != passage_docids:
        missing = sorted(all_docids.difference(passage_docids))
        extra = sorted(passage_docids.difference(all_docids))
        raise ValueError(
            "passages.parquet does not exactly cover candidate docids: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    actual_dev = candidates["dev"][_RUN_CORE_COLUMNS].reset_index(drop=True).copy()
    expected_dev = dev_top100[_RUN_CORE_COLUMNS].reset_index(drop=True).copy()
    for frame in (actual_dev, expected_dev):
        for column in ("split", "query_id", "docid"):
            frame[column] = frame[column].astype("string")
        frame["bm25_rank"] = frame["bm25_rank"].astype("int64")
        frame["bm25_score"] = frame["bm25_score"].astype("float64")
    if not actual_dev.equals(expected_dev):
        raise ValueError("dev candidate cache does not match the stable dev top-100 run")

    work_path = _resolve_repository_path(
        config, config["paths"]["work_dir"]
    ) / "candidate_cache.json"
    work = _read_json(work_path)
    if expected_inputs is not None and work.get("input_artifacts") != expected_inputs:
        raise ValueError("candidate cache provenance is stale or does not match its inputs")
    expected_outputs = {
        name: _file_metadata(config, path) for name, path in outputs.items()
    }
    if work.get("output_artifacts") != expected_outputs:
        raise ValueError("candidate cache output hashes do not match its provenance audit")
    return {
        "candidates": candidates,
        "queries": queries,
        "passages": passages,
        "work": work,
    }


def _build_candidate_cache(args: argparse.Namespace) -> int:
    config = _load_retrieval_config(args.config)
    _preflight_candidate_cache(config)
    _, reproduction_path, dev_source_hash = _require_reproduction_gate(config)
    dev_run_path = _artifact_path(config, "dev_run")
    dev_top100_path = _artifact_path(config, "dev_top100_run")
    annotation_paths = _annotation_paths(config)
    top_k = int(config["retrieval"]["candidate_depth"])
    dev_top100, dev_top100_action = _load_or_create_stable_trec_run(
        dev_run_path,
        dev_top100_path,
        split="dev",
        top_k=top_k,
        source_top_k=int(config["retrieval"]["retrieval_hits"]["dev"]),
        overwrite=args.overwrite,
    )
    if _sha256(dev_run_path) != dev_source_hash:
        raise RuntimeError("dev top-1000 run was modified after reproduction")

    output_paths = _candidate_output_paths(config)
    existing = {name for name, path in output_paths.items() if path.exists()}
    if existing and existing != set(output_paths) and not args.overwrite:
        missing = sorted(set(output_paths).difference(existing))
        raise ValueError(
            "partial candidate cache detected; existing outputs were preserved: "
            f"{sorted(existing)}; missing outputs: {missing}. Review them, then use "
            "--overwrite to rebuild only the cache (raw BM25 runs are never removed)."
        )
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    input_metadata = _candidate_input_metadata(
        config,
        annotation_paths,
        reproduction_path=reproduction_path,
        dev_top100_path=dev_top100_path,
    )
    if existing == set(output_paths) and not args.overwrite:
        validated = _validate_candidate_cache_artifacts(
            config,
            dev_top100=dev_top100,
            expected_inputs=input_metadata,
        )
        _print_json(
            {
                "status": "ok",
                "action": "reused_valid_candidate_cache",
                "dev_top100_action": dev_top100_action,
                "train_rows": int(len(validated["candidates"]["train"])),
                "dev_rows": int(len(validated["candidates"]["dev"])),
                "queries": int(len(validated["queries"])),
                "unique_passages": int(len(validated["passages"])),
            }
        )
        return 0

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
    work_dir = _resolve_repository_path(config, config["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    staging_passages = work_dir / "passages.staging.parquet"
    corpus_source = _corpus_source(config)
    extraction_started = time.perf_counter()
    try:
        extraction_report = extract_candidate_passages(
            candidate_docids,
            staging_passages,
            source=corpus_source,
            batch_rows=int(_corpus_settings(config)["batch_rows"]),
            log=lambda message: print(message, flush=True),
        )
    except MissingCandidatePassagesError as exc:
        preview_docids = set(exc.missing_docids[:10])
        context = _candidate_locations(
            candidate_frames,
            only_docids=preview_docids,
        )
        detailed = MissingCandidatePassagesError(
            exc.missing_docids,
            candidate_context=context,
            shards_visited=exc.shards_visited,
            lines_visited=exc.lines_visited,
        )
        raise StageError(
            stage="build-candidate-cache/passage-extraction",
            command=(
                "python -m rusearchrank.cli build-candidate-cache "
                "--config configs/retrieval.yaml"
            ),
            inputs={
                "train_run": _portable_repository_path(
                    config, _artifact_path(config, "train_run")
                ),
                "dev_top100_run": _portable_repository_path(config, dev_top100_path),
                "corpus_repo": str(corpus_source.describe()),
            },
            outputs={"staging_passages": str(staging_passages)},
            root_cause=str(detailed),
            reusable=(
                "raw BM25 runs, dev top-100 run and the reproduction audit are "
                "untouched and can be reused"
            ),
            repeat_cell="Cell 12 (build-candidate-cache) after Cell 11 smoke passes",
        ) from exc
    except CorpusAccessError as exc:
        raise StageError(
            stage="build-candidate-cache/corpus-access",
            command=(
                "python -m rusearchrank.cli build-candidate-cache "
                "--config configs/retrieval.yaml"
            ),
            inputs={"corpus_repo": str(corpus_source.describe())},
            outputs={"staging_passages": str(staging_passages)},
            root_cause=f"{type(exc).__name__}: {exc}",
            reusable="every artifact produced by Cells 8-10 is untouched",
            repeat_cell="Cell 11 (REAL COLAB SMOKE), then Cell 12",
        ) from exc
    extraction_seconds = time.perf_counter() - extraction_started

    current_inputs = _candidate_input_metadata(
        config,
        annotation_paths,
        reproduction_path=reproduction_path,
        dev_top100_path=dev_top100_path,
    )
    if current_inputs != input_metadata or _sha256(dev_run_path) != dev_source_hash:
        raise RuntimeError("candidate-cache inputs changed during passage extraction")

    _atomic_parquet(candidate_frames["train"], output_paths["train_candidates"])
    _atomic_parquet(candidate_frames["dev"], output_paths["dev_candidates"])
    _atomic_parquet(queries, output_paths["queries"])
    staging_passages.replace(output_paths["passages"])
    # The passage table is validated row by row while it is written, so the
    # published file only needs its docid column read back for exact coverage.
    published = pq.read_table(output_paths["passages"], columns=["docid"])
    published_docids = set(published.column("docid").to_pylist())
    passage_count = published.num_rows
    if published_docids != candidate_docids or passage_count != len(candidate_docids):
        missing = sorted(candidate_docids.difference(published_docids))
        extra = sorted(published_docids.difference(candidate_docids))
        raise ValueError(
            "passage extraction did not produce exact candidate coverage: "
            f"missing={missing[:10]}, extra={extra[:10]}, rows={passage_count}"
        )
    work_path = work_dir / "candidate_cache.json"
    output_metadata = {
        name: _file_metadata(config, path) for name, path in output_paths.items()
    }
    _write_json(
        work_path,
        {
            "status": "PASS",
            "producer_command": (
                "python -m rusearchrank.cli build-candidate-cache "
                "--config configs/retrieval.yaml"
            ),
            "passage_extraction_seconds": extraction_seconds,
            "passage_extraction": extraction_report,
            "unique_passages": int(passage_count),
            "train_rows": int(len(candidate_frames["train"])),
            "dev_rows": int(len(candidate_frames["dev"])),
            "dev_top1000_run": _portable_repository_path(config, dev_run_path),
            "dev_top1000_sha256": dev_source_hash,
            "dev_top100_run": _portable_repository_path(config, dev_top100_path),
            "dev_top100_sha256": _sha256(dev_top100_path),
            "dev_top100_action": dev_top100_action,
            "retrieval_depth": depth_audits,
            "input_artifacts": input_metadata,
            "output_artifacts": output_metadata,
        },
    )
    _print_json(
        {
            "status": "ok",
            "action": "rebuilt_candidate_cache" if existing else "created_candidate_cache",
            "dev_top100_action": dev_top100_action,
            "train_rows": int(len(candidate_frames["train"])),
            "dev_rows": int(len(candidate_frames["dev"])),
            "queries": int(len(queries)),
            "unique_passages": int(passage_count),
            "passage_extraction_seconds": extraction_seconds,
            "shards_visited": extraction_report["shards_visited"],
            "lines_visited": extraction_report["lines_visited"],
            "early_stop": extraction_report["early_stop"],
        }
    )
    return 0


@dataclasses.dataclass(frozen=True)
class _SingleShardSource(ShardSource):
    """One already-resolved shard, used by the smoke test only."""

    shard_name: str = ""
    shard_path: Path = Path(".")

    def names(self) -> list[str]:
        return [self.shard_name]

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "single_shard",
            "shard": self.shard_name,
            "path": str(self.shard_path),
            "loader": "gzip + json.loads",
            "dataset_script_used": False,
            "trust_remote_code": False,
        }

    def local_path(self, name: str) -> Path:
        if name != self.shard_name:
            raise ValueError(f"unexpected shard request: {name}")
        return self.shard_path


def _smoke_corpus_access(args: argparse.Namespace) -> int:
    """Run a cheap but completely real corpus/Parquet/ZIP round trip.

    Every step touches the same code the full candidate cache uses: the pinned
    Hugging Face revision, one real static JSONL.GZ shard, real gzip and JSON
    parsing, a real Parquet write and read back, a real manifest, a real ZIP,
    real extraction and a real SHA-256 revalidation. Nothing is downloaded
    beyond a single shard and no production artifact is written.
    """

    config = _load_retrieval_config(args.config)
    settings = _corpus_settings(config)
    checks: list[dict[str, Any]] = []

    def record(name: str, detail: dict[str, Any]) -> None:
        checks.append({"check": name, "status": "PASS", **detail})

    work_root = _resolve_repository_path(config, config["paths"]["work_dir"])
    work_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix="rusearchrank-corpus-smoke-", dir=work_root)
    )
    started = datetime.now(timezone.utc)
    cleaned = False
    try:
        if args.shards_dir:
            shards_directory = _resolve_repository_path(config, args.shards_dir)
            source: ShardSource = LocalShardSource(
                language=settings["language"],
                shard_count=settings["shard_count"],
                directory=shards_directory,
            )
            resolution = {
                "kind": "local_directory",
                "directory": str(shards_directory),
                "shards": source.names(),
            }
        else:
            source = _corpus_source(config)
            resolution = resolve_hub_shards(
                repo_id=settings["repo_id"],
                revision=settings["revision"],
                language=settings["language"],
                shard_count=settings["shard_count"],
            )
        # 1. pinned revision availability, 3. no dataset script is ever executed
        record(
            "pinned_revision_available",
            {
                "repo_id": settings["repo_id"],
                "revision": settings["revision"],
                "resolution": resolution,
            },
        )
        assert_no_dataset_script(source.names())
        record(
            "no_dataset_script_loader",
            {
                "denied": ["miracl-corpus.py"],
                "loader": source.describe()["loader"],
                "datasets_load_dataset_used": False,
                "trust_remote_code": False,
            },
        )
        names = source.names()
        if names != sort_shard_names(names) or len(names) != settings["shard_count"]:
            raise ValueError("official shard list is not in numeric order")
        record(
            "numeric_shard_order",
            {"first": names[:3], "last": names[-3:], "count": len(names)},
        )

        # 2. first shard availability
        shard_name = names[int(args.shard_index)]
        shard_path = source.local_path(shard_name)
        record(
            "shard_available",
            {
                "shard": shard_name,
                "local_path": str(shard_path),
                "size_bytes": shard_path.stat().st_size,
            },
        )

        # 4-7. real gzip open, valid JSON, required fields, string docid
        rows: list[dict[str, str]] = []
        max_rows = int(args.max_rows)
        stream = iter_shard_rows(shard_path, shard=shard_name, max_rows=max_rows)
        with contextlib.closing(stream):
            for _, record_row in stream:
                rows.append(record_row)
        if len(rows) < int(args.min_passages):
            raise ValueError(
                f"shard {shard_name} yielded only {len(rows)} rows; "
                f"at least {args.min_passages} are required for the smoke test"
            )
        if any(not isinstance(row["docid"], str) for row in rows):
            raise ValueError("corpus docid must always stay a string")
        record(
            "gzip_json_fields_and_string_docid",
            {
                "shard": shard_name,
                "rows_parsed": len(rows),
                "fields": list(CORPUS_FIELDS),
                "docid_type": "str",
                "first_docids": [row["docid"] for row in rows[:3]],
            },
        )

        # 8. filter several real passages exactly like the candidate cache does
        selected = [row["docid"] for row in rows[:: max(1, len(rows) // 25)]][:25]
        if len(selected) < min(25, len(rows)):
            selected = [row["docid"] for row in rows[:25]]
        requested = set(selected)
        passages_path = temporary_root / "smoke_passages.parquet"
        extraction = extract_candidate_passages(
            requested,
            passages_path,
            source=_SingleShardSource(
                language=settings["language"],
                shard_count=1,
                shard_name=shard_name,
                shard_path=shard_path,
            ),
            batch_rows=max(1, len(requested) // 3),
        )
        record(
            "real_passage_filtering",
            {
                "requested": len(requested),
                "found": extraction["found_docids"],
                "batches_written": extraction["batches_written"],
                "lines_visited": extraction["lines_visited"],
                "early_stop": extraction["early_stop"],
            },
        )

        # 9-10. real Parquet write and read back
        reloaded = pd.read_parquet(passages_path)
        validate_passages(reloaded)
        if set(reloaded["docid"].astype("string")) != requested:
            raise ValueError("smoke Parquet does not cover the requested docids")
        record(
            "parquet_write_and_read_back",
            {
                "path": str(passages_path),
                "rows": int(len(reloaded)),
                "columns": list(reloaded.columns),
                "size_bytes": passages_path.stat().st_size,
            },
        )

        # 11. real manifest with sizes and hashes
        manifest_path = temporary_root / "smoke_manifest.json"
        passages_hash = _sha256(passages_path)
        manifest = {
            "status": "PASS",
            "created_at": started.isoformat(),
            "artifacts": [
                {
                    "path": passages_path.name,
                    "size_bytes": passages_path.stat().st_size,
                    "sha256": passages_hash,
                    "row_count": int(len(reloaded)),
                    "schema": {
                        column: str(dtype) for column, dtype in reloaded.dtypes.items()
                    },
                    "producer_command": (
                        "python -m rusearchrank.cli smoke-corpus-access "
                        "--config configs/retrieval.yaml"
                    ),
                    "input_hashes": {"corpus_shard": shard_name},
                }
            ],
        }
        _write_json(manifest_path, manifest)
        record("temporary_manifest", {"path": str(manifest_path), "entries": 1})

        # 12-14. real ZIP, real extraction, real hash revalidation
        archive_path = temporary_root / "smoke_results.zip"
        members = [passages_path.name, manifest_path.name]
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.write(passages_path, arcname=passages_path.name)
            archive.write(manifest_path, arcname=manifest_path.name)
        with zipfile.ZipFile(archive_path) as archive:
            if archive.namelist() != members:
                raise ValueError("smoke ZIP does not contain the exact member list")
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"smoke ZIP CRC validation failed for {corrupt}")
            for info in archive.infolist():
                member = Path(info.filename)
                if member.is_absolute() or ".." in member.parts or info.is_dir():
                    raise ValueError(f"unsafe smoke ZIP member: {info.filename}")
            extraction_root = temporary_root / "extracted"
            archive.extractall(extraction_root)
        record(
            "zip_created_and_extracted",
            {
                "path": str(archive_path),
                "size_bytes": archive_path.stat().st_size,
                "members": members,
            },
        )
        extracted_manifest = _read_json(extraction_root / manifest_path.name)
        if extracted_manifest != manifest:
            raise ValueError("smoke manifest changed after ZIP extraction")
        extracted_passages = extraction_root / passages_path.name
        extracted_hash = _sha256(extracted_passages)
        if extracted_hash != passages_hash:
            raise ValueError("smoke Parquet SHA-256 changed after ZIP extraction")
        extracted_frame = pd.read_parquet(extracted_passages)
        validate_passages(extracted_frame)
        if len(extracted_frame) != len(reloaded):
            raise ValueError("smoke Parquet row count changed after ZIP extraction")
        record(
            "hash_revalidated_after_extraction",
            {"sha256": extracted_hash, "rows": int(len(extracted_frame))},
        )

        payload = {
            "status": "PASS",
            "stage": "real-corpus-smoke",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "source": source.describe(),
            "shard": shard_name,
            "rows_parsed": len(rows),
            "requested_passages": len(requested),
            "extraction": extraction,
            "checks": checks,
            "full_corpus_downloaded": False,
            "production_artifacts_written": [],
            "temporary_root": str(temporary_root),
        }
        report_path = _resolve_repository_path(config, args.output)
        _write_json(report_path, payload)
        payload["report"] = _portable_repository_path(config, report_path)
        _print_json(payload)
        # 15. cleanup only after a successful report
        shutil.rmtree(temporary_root, ignore_errors=True)
        cleaned = True
        return 0
    finally:
        if not cleaned and temporary_root.is_dir():
            print(
                "corpus smoke failed; diagnostic files preserved at "
                f"{temporary_root}",
                file=sys.stderr,
            )


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
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "official trec_eval command timed out: "
            f"{' '.join(command)}\nstdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
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
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"trec_eval -h timed out: {exc}") from exc
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
    annotation_paths, _ = _validate_official_annotations(config)
    queries = pd.read_parquet(_artifact_path(config, "queries"))
    validate_queries(queries)
    split_reports: dict[str, Any] = {}
    for split in ("train", "dev"):
        split_queries = queries.loc[queries["split"].astype("string").eq(split)]
        qrels = load_qrels(annotation_paths[split]["qrels"], split=split)
        candidates = read_candidate_table(_artifact_path(config, f"{split}_candidates"))
        validate_candidate_schema(candidates)
        report = build_qrels_split_audit(
            queries=split_queries,
            qrels=qrels,
            candidates=candidates,
        )
        report["inputs"] = {
            "topics": _file_metadata(config, annotation_paths[split]["topics"]),
            "qrels": _file_metadata(config, annotation_paths[split]["qrels"]),
            "candidates": _file_metadata(
                config, _artifact_path(config, f"{split}_candidates")
            ),
            "queries": _file_metadata(config, _artifact_path(config, "queries")),
        }
        split_reports[split] = report
    payload = {
        "status": "PASS",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "annotations_revision": config["dataset"]["annotations_revision"],
        "splits": split_reports,
        "max_judged_negatives_recommendation": None,
        "note": "No training-sampling decision is made in Phase 1A.",
    }
    _write_json(output, payload)
    _print_json(payload)
    return 0


def _tabular_contract(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        return {
            "row_count": int(len(frame)),
            "schema": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        }
    if path.suffix.lower() == ".trec":
        with path.open("r", encoding="utf-8") as stream:
            row_count = sum(1 for line in stream if line.strip())
        return {
            "row_count": row_count,
            "schema": {
                "query_id": "string",
                "q0": "string",
                "docid": "string",
                "rank": "int64",
                "score": "float64",
                "tag": "string",
            },
        }
    return {"row_count": None, "schema": None}


def _artifact_manifest_entry(
    config: dict[str, Any],
    path: Path,
    *,
    producer_command: str,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        **_file_metadata(config, path),
        **_tabular_contract(path),
        "producer_command": producer_command,
        "input_hashes": input_hashes,
    }


def _validate_manifest_artifacts(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_paths: list[Path],
    expected_input_hashes: dict[str, dict[str, str]] | None = None,
) -> None:
    if manifest.get("status") != "PASS":
        raise ValueError("candidate-cache manifest status is not PASS")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("candidate-cache manifest artifacts must be a list")
    by_path = {
        str(entry.get("path")): entry for entry in entries if isinstance(entry, dict)
    }
    expected_names = {
        _portable_repository_path(config, path) for path in expected_paths
    }
    if set(by_path) != expected_names:
        raise ValueError("manifest artifact paths do not match the exact allowlist")
    for path in expected_paths:
        name = _portable_repository_path(config, path)
        entry = by_path[name]
        if entry.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"stale manifest size for {name}")
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"stale manifest SHA-256 for {name}")
        if "producer_command" not in entry or "input_hashes" not in entry:
            raise ValueError(f"manifest provenance is incomplete for {name}")
        if expected_input_hashes is not None and entry.get("input_hashes") != (
            expected_input_hashes[name]
        ):
            raise ValueError(f"stale manifest input hashes for {name}")
        if not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in entry.get("input_hashes", {}).values()
        ):
            raise ValueError(f"manifest contains an invalid input hash for {name}")
        contract = _tabular_contract(path)
        if entry.get("row_count") != contract["row_count"]:
            raise ValueError(f"stale manifest row count for {name}")
        if entry.get("schema") != contract["schema"]:
            raise ValueError(f"stale manifest schema for {name}")


def _validate_phase1_archive(
    config: dict[str, Any],
    archive_path: Path,
    manifest_path: Path,
    *,
    payload_paths: list[Path],
    expected_input_hashes: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    _require_nonempty_regular_file(archive_path, label="Phase 1 ZIP")
    _require_nonempty_regular_file(manifest_path, label="candidate-cache manifest")
    manifest = _read_json(manifest_path)
    _validate_manifest_artifacts(
        config,
        manifest,
        expected_paths=payload_paths,
        expected_input_hashes=expected_input_hashes,
    )
    allowlist = [
        *[_portable_repository_path(config, path) for path in payload_paths],
        _portable_repository_path(config, manifest_path),
    ]
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if names != allowlist or len(set(names)) != len(names):
            raise ValueError("ZIP contents do not match the exact ordered allowlist")
        for info in archive.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts or info.is_dir():
                raise ValueError(f"unsafe or unexpected ZIP member: {info.filename}")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"ZIP CRC validation failed for {corrupt}")
        archived_manifest = json.loads(
            archive.read(_portable_repository_path(config, manifest_path)).decode("utf-8")
        )
        if archived_manifest != manifest:
            raise ValueError("archived manifest differs from the validated manifest")
        with tempfile.TemporaryDirectory(prefix="rusearchrank-zip-verify-") as directory:
            extraction_root = Path(directory)
            archive.extractall(extraction_root)
            for entry in manifest["artifacts"]:
                extracted = extraction_root / str(entry["path"])
                if not extracted.is_file():
                    raise ValueError(f"ZIP extraction is missing {entry['path']}")
                if extracted.stat().st_size != entry["size_bytes"]:
                    raise ValueError(f"extracted size mismatch for {entry['path']}")
                if _sha256(extracted) != entry["sha256"]:
                    raise ValueError(f"extracted SHA-256 mismatch for {entry['path']}")
            extracted_manifest = _read_json(
                extraction_root / _portable_repository_path(config, manifest_path)
            )
            if extracted_manifest != manifest:
                raise ValueError("manifest changed after ZIP extraction")
    return {
        "path": _portable_repository_path(config, archive_path),
        "size_bytes": archive_path.stat().st_size,
        "sha256": _sha256(archive_path),
        "contents": allowlist,
    }


def _package_phase1(args: argparse.Namespace) -> int:
    config = _load_retrieval_config(args.config)
    _preflight_candidate_cache(config)
    archive_path = _resolve_repository_path(config, config["archive"]["path"])
    manifest_path = _audit_path(config, "manifest")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    reproduction_path = _audit_path(config, "reproduction")
    qrels_path = _audit_path(config, "qrels")
    reproduction, _, _ = _require_reproduction_gate(config)
    qrels_audit = _read_json(qrels_path)
    dev_top1000_path = _artifact_path(config, "dev_run")
    dev_top100_path = _artifact_path(config, "dev_top100_run")
    train_run_path = _artifact_path(config, "train_run")
    if reproduction.get("source_run") != _portable_repository_path(
        config, dev_top1000_path
    ):
        raise ValueError("reproduction audit does not identify the configured dev top-1000 run")
    if reproduction.get("source_run_sha256") != _sha256(dev_top1000_path):
        raise ValueError("dev top-1000 run no longer matches its reproduction audit")
    if qrels_audit.get("status") != "PASS":
        raise ValueError("a passed qrels audit is required before packaging")
    if not isinstance(qrels_audit.get("splits"), dict):
        raise ValueError("qrels audit is missing split reports")
    if not bool(config["archive"].get("include_runs", False)):
        raise ValueError("Phase 1A archive must include all provenance runs")

    top_k = int(config["retrieval"]["candidate_depth"])
    actual_dev_top100 = _validate_stable_trec_derivation(
        dev_top1000_path,
        dev_top100_path,
        split="dev",
        top_k=top_k,
        source_top_k=int(config["retrieval"]["retrieval_hits"]["dev"]),
    )
    annotation_paths, _ = _validate_official_annotations(config)
    expected_cache_inputs = _candidate_input_metadata(
        config,
        annotation_paths,
        reproduction_path=reproduction_path,
        dev_top100_path=dev_top100_path,
    )
    cache = _validate_candidate_cache_artifacts(
        config,
        dev_top100=actual_dev_top100,
        expected_inputs=expected_cache_inputs,
    )
    candidates = cache["candidates"]
    queries = cache["queries"]
    passages = cache["passages"]

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

    for split in ("train", "dev"):
        split_audit = qrels_audit["splits"].get(split, {})
        expected_qrels_inputs = {
            "topics": _file_metadata(config, annotation_paths[split]["topics"]),
            "qrels": _file_metadata(config, annotation_paths[split]["qrels"]),
            "candidates": _file_metadata(
                config, _artifact_path(config, f"{split}_candidates")
            ),
            "queries": _file_metadata(config, _artifact_path(config, "queries")),
        }
        if split_audit.get("inputs") != expected_qrels_inputs:
            raise ValueError(f"{split} qrels audit is stale or bound to other inputs")

    cache_input_hashes = {
        name: str(metadata["sha256"])
        for name, metadata in expected_cache_inputs.items()
    }
    train_candidates_path = _artifact_path(config, "train_candidates")
    dev_candidates_path = _artifact_path(config, "dev_candidates")
    queries_path = _artifact_path(config, "queries")
    passages_path = _artifact_path(config, "passages")
    train_candidate_hash = _sha256(train_candidates_path)
    dev_candidate_hash = _sha256(dev_candidates_path)
    queries_hash = _sha256(queries_path)
    artifact_input_hashes = {
        _portable_repository_path(config, train_run_path): {
            "train_topics": cache_input_hashes["train_topics"],
        },
        _portable_repository_path(config, dev_top1000_path): {
            "dev_topics": cache_input_hashes["dev_topics"],
        },
        _portable_repository_path(config, dev_top100_path): {
            "dev_top1000_run": cache_input_hashes["dev_top1000_run"],
            "reproduction_audit": cache_input_hashes["reproduction_audit"],
        },
        _portable_repository_path(config, train_candidates_path): {
            "train_run": cache_input_hashes["train_run"],
            "train_topics": cache_input_hashes["train_topics"],
            "train_qrels": cache_input_hashes["train_qrels"],
        },
        _portable_repository_path(config, dev_candidates_path): {
            "dev_top100_run": cache_input_hashes["dev_top100_run"],
            "dev_topics": cache_input_hashes["dev_topics"],
            "dev_qrels": cache_input_hashes["dev_qrels"],
            "reproduction_audit": cache_input_hashes["reproduction_audit"],
        },
        _portable_repository_path(config, queries_path): {
            "train_topics": cache_input_hashes["train_topics"],
            "dev_topics": cache_input_hashes["dev_topics"],
        },
        _portable_repository_path(config, passages_path): {
            "train_candidates": train_candidate_hash,
            "dev_candidates": dev_candidate_hash,
        },
        _portable_repository_path(config, reproduction_path): {
            "dev_top1000_run": cache_input_hashes["dev_top1000_run"],
            "dev_qrels": cache_input_hashes["dev_qrels"],
        },
        _portable_repository_path(config, qrels_path): {
            "train_qrels": cache_input_hashes["train_qrels"],
            "dev_qrels": cache_input_hashes["dev_qrels"],
            "train_candidates": train_candidate_hash,
            "dev_candidates": dev_candidate_hash,
            "queries": queries_hash,
        },
    }

    existing_package = {
        "archive": archive_path.exists(),
        "manifest": manifest_path.exists(),
    }
    # A freshly cloned repository ships a placeholder manifest and no ZIP, so a
    # missing ZIP beside an existing manifest is the normal first-run state.
    # It is only reported as partial when the caller refused to overwrite.
    if (
        any(existing_package.values())
        and not all(existing_package.values())
        and not args.overwrite
    ):
        raise ValueError(
            "partial package output detected and preserved: "
            f"{existing_package}; manifest={_portable_repository_path(config, manifest_path)}, "
            f"archive={_portable_repository_path(config, archive_path)}. "
            "Rerun the packaging cell with --overwrite to rebuild both from the "
            "validated candidate cache."
        )
    if all(existing_package.values()) and not args.overwrite:
        try:
            archive_report = _validate_phase1_archive(
                config,
                archive_path,
                manifest_path,
                payload_paths=payload_paths,
                expected_input_hashes=artifact_input_hashes,
            )
        except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as exc:
            raise ValueError(
                "existing Phase 1 package is invalid or stale and was preserved; "
                f"use --overwrite only after review. Validation error: {exc}"
            ) from exc
        _print_json(
            {"status": "ok", "action": "reused_valid_package", **archive_report}
        )
        return 0

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
    producers = {
        _portable_repository_path(config, train_run_path): (
            "python -m rusearchrank.cli run-bm25 --config configs/retrieval.yaml "
            "--split train"
        ),
        _portable_repository_path(config, dev_top1000_path): (
            "python -m rusearchrank.cli run-bm25 --config configs/retrieval.yaml "
            "--split dev"
        ),
        _portable_repository_path(config, dev_top100_path): (
            "python -m rusearchrank.cli build-candidate-cache "
            "--config configs/retrieval.yaml"
        ),
        _portable_repository_path(config, reproduction_path): (
            "python -m rusearchrank.cli evaluate-bm25 "
            "--config configs/retrieval.yaml"
        ),
        _portable_repository_path(config, qrels_path): (
            "python -m rusearchrank.cli audit-qrels --config configs/retrieval.yaml"
        ),
    }
    cache_producer = (
        "python -m rusearchrank.cli build-candidate-cache "
        "--config configs/retrieval.yaml"
    )
    for key in ("train_candidates", "dev_candidates", "queries", "passages"):
        producers[_portable_repository_path(config, _artifact_path(config, key))] = (
            cache_producer
        )
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
            "java_home": runtime_environment["java_home"],
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
                "path": _portable_repository_path(config, dev_top1000_path),
                "sha256": _sha256(dev_top1000_path),
                "size_bytes": dev_top1000_path.stat().st_size,
            },
            "reproduction_audit": {
                "path": _portable_repository_path(config, reproduction_path),
                "status": reproduction["status"],
                "source_run_sha256": reproduction["source_run_sha256"],
            },
            "dev_top100": {
                "path": _portable_repository_path(config, dev_top100_path),
                "sha256": _sha256(dev_top100_path),
                "size_bytes": dev_top100_path.stat().st_size,
            },
            "candidate_cache": {
                "train": _portable_repository_path(
                    config, _artifact_path(config, "train_candidates")
                ),
                "dev": _portable_repository_path(
                    config, _artifact_path(config, "dev_candidates")
                ),
                "judgment_states": [
                    "relevant",
                    "judged_non_relevant",
                    "unjudged",
                ],
                "input_hashes": cache_input_hashes,
            },
        },
        "timings": timings,
        "artifacts": [
            _artifact_manifest_entry(
                config,
                path,
                producer_command=producers[_portable_repository_path(config, path)],
                input_hashes=artifact_input_hashes[
                    _portable_repository_path(config, path)
                ],
            )
            for path in payload_paths
        ],
    }
    _write_json(manifest_path, manifest)
    archive_inputs = [*payload_paths, manifest_path]
    temporary_archive = archive_path.with_name(
        f"{archive_path.name}.tmp.{os.getpid()}"
    )
    if temporary_archive.exists():
        raise ValueError(
            f"diagnostic temporary ZIP already exists and was preserved: {temporary_archive}"
        )
    with zipfile.ZipFile(
        temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in archive_inputs:
            archive.write(path, arcname=_portable_repository_path(config, path))
    with temporary_archive.open("rb") as stream:
        os.fsync(stream.fileno())
    try:
        _validate_phase1_archive(
            config,
            temporary_archive,
            manifest_path,
            payload_paths=payload_paths,
            expected_input_hashes=artifact_input_hashes,
        )
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"temporary ZIP validation failed; preserved at {temporary_archive}: {exc}"
        ) from exc
    temporary_archive.replace(archive_path)
    archive_report = _validate_phase1_archive(
        config,
        archive_path,
        manifest_path,
        payload_paths=payload_paths,
        expected_input_hashes=artifact_input_hashes,
    )
    _print_json({"status": "ok", "action": "created_package", **archive_report})
    return 0


def _rerank_file_metadata(
    config: dict[str, Any], path: Path
) -> dict[str, Any]:
    _require_nonempty_regular_file(path, label="Phase 2 input")
    return {
        "path": rerank_module.portable_path(config, path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _execute_rerank_trec_eval(
    config: dict[str, Any],
    *,
    run_path: Path,
    executable: Path,
    trec_eval_provenance: dict[str, Any],
) -> dict[str, Any]:
    qrels_path = rerank_module.resolve_path(config, config["inputs"]["qrels"])
    executions: dict[str, Any] = {}
    command_specs = {
        "ndcg_at_10": (config["evaluation"]["ndcg_command"], "ndcg_cut_10"),
        "recall_at_100": (config["evaluation"]["recall_command"], "recall_100"),
        "mrr_at_10": (config["evaluation"]["mrr_command"], "recip_rank"),
        "per_query_ndcg_at_10": (
            config["evaluation"]["per_query_command"],
            "ndcg_cut_10",
        ),
    }
    parsed: dict[str, Any] = {}
    for name, (arguments, metric) in command_specs.items():
        execution = _run_trec_eval_binary(
            executable,
            [*[str(value) for value in arguments], str(qrels_path), str(run_path)],
        )
        execution["command"] = [
            str(executable),
            *[str(value) for value in arguments],
            rerank_module.portable_path(config, qrels_path),
            rerank_module.portable_path(config, run_path),
        ]
        execution["executable"] = str(executable)
        execution["expected_release_version"] = str(
            config["evaluation"]["trec_eval_expected_release"]
        )
        execution["binary_reported_version"] = trec_eval_provenance[
            "binary_reported_version"
        ]
        execution["version_probe"] = trec_eval_provenance["version_probe"]
        executions[name] = execution
        if name == "per_query_ndcg_at_10":
            parsed[name] = parse_trec_eval_per_query(str(execution["stdout"]), metric)
        else:
            parsed[name] = parse_trec_eval_metric(str(execution["stdout"]), metric)
        execution["parsed"] = parsed[name]
    return {
        "executable": str(executable),
        "expected_release_version": str(
            config["evaluation"]["trec_eval_expected_release"]
        ),
        "binary_reported_version": trec_eval_provenance[
            "binary_reported_version"
        ],
        "version_probe": trec_eval_provenance["version_probe"],
        "build_provenance": trec_eval_provenance,
        "run": _rerank_file_metadata(config, run_path),
        "executions": executions,
        "parsed": parsed,
    }


def _trec_python_ranking(
    run: pd.DataFrame, *, docid_ascending: bool
) -> pd.DataFrame:
    ordered = run[["query_id", "docid", "bm25_score"]].copy()
    ordered[["query_id", "docid"]] = ordered[["query_id", "docid"]].astype(
        "string"
    )
    ordered["bm25_score"] = pd.to_numeric(
        ordered["bm25_score"], errors="raise"
    ).astype("float64")
    ordered = ordered.sort_values(
        ["query_id", "bm25_score", "docid"],
        ascending=[True, False, docid_ascending],
        kind="mergesort",
    )
    ordered["bm25_rank"] = (
        ordered.groupby("query_id", sort=False).cumcount().add(1).astype("int64")
    )
    return ordered[["query_id", "docid", "bm25_rank"]].reset_index(drop=True)


def _python_metric_vector(report: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["query_id"]): float(row["ndcg_at_10"])
        for row in report["per_query"]
    }


def _vector_difference(
    left: dict[str, float], right: dict[str, float]
) -> float:
    if set(left) != set(right):
        return float("inf")
    return max(abs(float(left[key]) - float(right[key])) for key in left)


def _raw_score_tie_count(frame: pd.DataFrame, score_column: str) -> int:
    counts = frame.groupby(["query_id", score_column], dropna=False).size()
    return int(sum(int(value) - 1 for value in counts if int(value) > 1))


def _metric_common_provenance(
    config: dict[str, Any],
    *,
    score_sidecar: dict[str, Any],
    input_artifacts: dict[str, Any],
    evaluation_fingerprint: str,
    score_ties: dict[str, Any],
    ranking_score_tie_rows: int,
) -> dict[str, Any]:
    components = score_sidecar.get("fingerprint_components")
    if not isinstance(components, dict):
        raise ValueError("score sidecar is missing fingerprint components")
    evaluation_commit, evaluation_git_dirty = rerank_module.git_provenance(
        rerank_module.repository_root(config)
    )
    return {
        "input_artifacts": input_artifacts,
        "model_id": str(config["model"]["id"]),
        "model_revision": str(config["model"]["revision"]),
        "tokenizer_revision": str(config["model"]["tokenizer_revision"]),
        "device": str(score_sidecar["device"]),
        "dtype": str(score_sidecar["dtype"]),
        "batch_size": int(score_sidecar["batch_size"]),
        "max_length": int(config["input"]["max_length"]),
        **score_ties,
        "ranking_score_tie_rows": int(ranking_score_tie_rows),
        "token_accounting": score_sidecar["token_accounting"],
        "peak_rss_bytes": int(score_sidecar["peak_rss_bytes"]),
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
        "config_sha256": _sha256(Path(str(config["_config_path"]))),
        "input_fingerprint": str(score_sidecar["input_fingerprint"]),
        "evaluation_fingerprint": evaluation_fingerprint,
        "score_producer_commit": str(components["git_commit"]),
        "evaluation_commit": evaluation_commit,
        "evaluation_git_dirty": evaluation_git_dirty,
        "score_encoding": rerank_module.SCORE_ENCODING,
    }


_EVALUATION_REPORT_ORDER = (
    "baseline",
    "system",
    "comparison",
    "depth_profile",
)


def _validate_evaluation_generation(
    reports: dict[str, dict[str, Any]],
    *,
    evaluation_fingerprint: str,
    staging_directory: Path,
) -> None:
    if tuple(reports) != _EVALUATION_REPORT_ORDER:
        raise ValueError("evaluation generation does not contain the exact four reports")
    serialized = json.dumps(reports, ensure_ascii=False, sort_keys=True)
    if str(staging_directory) in serialized or ".tmp." in serialized:
        raise ValueError("evaluation reports contain temporary paths")

    def reject_unapproved_absolute_paths(value: Any, *, key: str = "") -> None:
        if isinstance(value, dict):
            for nested_key, nested in value.items():
                reject_unapproved_absolute_paths(nested, key=str(nested_key))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                if (
                    key == "command"
                    and index == 0
                    and isinstance(nested, str)
                    and Path(nested).is_absolute()
                ):
                    continue
                reject_unapproved_absolute_paths(nested, key=key)
        elif (
            isinstance(value, str)
            and Path(value).is_absolute()
            and key not in {"binary_path", "source_path", "executable"}
        ):
            raise ValueError(
                f"evaluation reports contain an unapproved absolute path at {key}: {value}"
            )

    reject_unapproved_absolute_paths(reports)
    provenances: list[dict[str, Any]] = []
    input_artifacts: list[dict[str, Any]] = []
    for name, report in reports.items():
        if report.get("status") != "PASS":
            raise ValueError(f"temporary {name} report status is not PASS")
        if report.get("evaluation_fingerprint") != evaluation_fingerprint:
            raise ValueError(f"temporary {name} report has a different fingerprint")
        provenance = report.get("trec_eval_provenance")
        if not isinstance(provenance, dict) or provenance.get(
            "evaluation_protocol_status"
        ) != "PASS":
            raise ValueError(f"temporary {name} report lacks valid trec_eval provenance")
        provenances.append(provenance)
        artifacts = report.get("input_artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(f"temporary {name} report lacks input artifact hashes")
        input_artifacts.append(artifacts)
    if any(value != provenances[0] for value in provenances[1:]):
        raise ValueError("temporary reports disagree on trec_eval provenance")
    if any(value != input_artifacts[0] for value in input_artifacts[1:]):
        raise ValueError("temporary reports disagree on model/input hashes")

    baseline_value = float(reports["baseline"]["official"]["value"])
    system_value = float(reports["system"]["official"]["value"])
    official_depths = [
        entry
        for entry in reports["depth_profile"].get("depths", [])
        if entry.get("official") is True
    ]
    if len(official_depths) != 1 or not math.isclose(
        float(official_depths[0]["ndcg_at_10"]), system_value, abs_tol=1e-12
    ):
        raise ValueError("system and official depth-profile metrics disagree")
    paired_delta = float(reports["comparison"]["paired"]["mean_delta"])
    if not math.isclose(system_value - baseline_value, paired_delta, abs_tol=1e-4):
        raise ValueError("baseline/system/comparison metrics are inconsistent")
    strata = reports["comparison"].get("stratified_mean_delta")
    if not isinstance(strata, dict):
        raise ValueError("comparison report is missing strata")
    invariants = strata.get("invariants")
    if not isinstance(invariants, dict) or invariants.get(
        "each_query_in_exactly_one_stratum"
    ) is not True:
        raise ValueError("comparison strata are not a disjoint complete partition")


def _stage_evaluation_generation(
    reports: dict[str, dict[str, Any]],
    *,
    staging_directory: Path,
    evaluation_fingerprint: str,
) -> dict[str, Path]:
    staged_paths: dict[str, Path] = {}
    for name in _EVALUATION_REPORT_ORDER:
        path = staging_directory / f"{name}.json"
        _write_json(path, reports[name])
        staged_paths[name] = path
    loaded = {name: _read_json(path) for name, path in staged_paths.items()}
    _validate_evaluation_generation(
        loaded,
        evaluation_fingerprint=evaluation_fingerprint,
        staging_directory=staging_directory,
    )
    return staged_paths


def _publish_evaluation_generation(
    *,
    config: dict[str, Any],
    output_paths: dict[str, Path],
    staged_paths: dict[str, Path],
    replace: Any | None = None,
) -> dict[str, Any]:
    if tuple(output_paths) != _EVALUATION_REPORT_ORDER or tuple(staged_paths) != (
        _EVALUATION_REPORT_ORDER
    ):
        raise ValueError("evaluation publication requires the exact ordered generation")
    work_dir = rerank_module.resolve_path(config, config["paths"]["work_dir"])
    backup_dir = work_dir / (
        "metrics-backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    existing = {name: path.is_file() for name, path in output_paths.items()}
    replace_path = replace or os.replace
    moved_old: list[str] = []
    published_new: list[str] = []
    try:
        if any(existing.values()):
            backup_dir.mkdir(parents=True, exist_ok=False)
            for name in _EVALUATION_REPORT_ORDER:
                if existing[name]:
                    replace_path(output_paths[name], backup_dir / output_paths[name].name)
                    moved_old.append(name)
        for name in _EVALUATION_REPORT_ORDER:
            output_paths[name].parent.mkdir(parents=True, exist_ok=True)
            replace_path(staged_paths[name], output_paths[name])
            published_new.append(name)
    except Exception as exc:
        rollback_errors: list[str] = []
        for name in reversed(published_new):
            try:
                replace_path(output_paths[name], staged_paths[name])
            except Exception as rollback_exc:
                rollback_errors.append(f"remove_new:{name}:{rollback_exc}")
        for name in reversed(moved_old):
            try:
                replace_path(backup_dir / output_paths[name].name, output_paths[name])
            except Exception as rollback_exc:
                rollback_errors.append(f"restore_old:{name}:{rollback_exc}")
        rollback_status = "PASS" if not rollback_errors else (
            "FAIL: " + "; ".join(rollback_errors)
        )
        raise StageError(
            stage="evaluate-rerank/publication",
            command=(
                "python -m rusearchrank.cli evaluate-rerank "
                "--config configs/rerank.yaml --split dev --overwrite"
            ),
            inputs={
                "previous_generation": str(backup_dir),
                "temporary_generation": str(next(iter(staged_paths.values())).parent),
            },
            outputs={name: str(path) for name, path in output_paths.items()},
            root_cause=f"publication failed after {published_new}: {exc}",
            reusable=(
                "production metrics are restored and score-Parquet/TREC runs remain reusable"
                if not rollback_errors
                else "manual recovery is required from the listed backup generation"
            ),
            repeat_cell="Cell 12: repeat evaluate-rerank only after reviewing rollback_status",
            rollback_status=rollback_status,
        ) from exc
    return {
        "rollback_status": "NOT_NEEDED",
        "previous_generation": (
            rerank_module.portable_path(config, backup_dir)
            if any(existing.values())
            else None
        ),
    }


def _smoke_rerank(args: argparse.Namespace) -> int:
    config = rerank_module.load_rerank_config(args.config)
    report = rerank_module.run_smoke_rerank(
        config,
        limit=int(args.limit),
        requested_device=str(args.device),
        output=args.output,
    )
    _print_json(report)
    return 0


def _rerank_score(
    args: argparse.Namespace,
    *,
    scorer: rerank_module.PairScorer | None = None,
) -> int:
    config = rerank_module.load_rerank_config(args.config)
    try:
        rerank_module.validate_smoke_gate(config)
    except (ValueError, OSError) as exc:
        raise StageError(
            stage="rerank-score/smoke-gate",
            command=(
                "python -m rusearchrank.cli rerank-score "
                "--config configs/rerank.yaml --split dev"
            ),
            inputs={
                "smoke_report": rerank_module.portable_path(
                    config,
                    rerank_module.resolve_path(config, config["audits"]["smoke"]),
                ),
                "config": rerank_module.portable_path(
                    config, Path(str(config["_config_path"]))
                ),
            },
            outputs={
                "scores": rerank_module.portable_path(
                    config,
                    rerank_module.resolve_path(
                        config, config["artifacts"]["scores"]
                    ),
                )
            },
            root_cause=str(exc),
            reusable="all Phase 1 inputs and compatible partial score shards remain reusable",
            repeat_cell=(
                "Cell 9: rerun `smoke-rerank --limit 64`; only then repeat Cell 10"
            ),
        ) from exc
    report = rerank_module.run_rerank_scoring(
        config,
        split=str(args.split),
        requested_device=str(args.device),
        batch_size_override=args.batch_size,
        overwrite=bool(args.overwrite),
        scorer=scorer,
        checkpoint=getattr(args, "checkpoint", None),
    )
    _print_json(report)
    return 0


def _build_rerank_run(args: argparse.Namespace) -> int:
    config = rerank_module.load_rerank_config(args.config)
    report = rerank_module.build_rerank_run(
        config,
        split=str(args.split),
        depth=int(args.depth),
        overwrite=bool(args.overwrite),
    )
    _print_json(report)
    return 0


def _rerank_evaluation_preflight(config: dict[str, Any]) -> dict[str, Any]:
    rerank_module.validate_phase1_inputs(config)
    score_path = rerank_module.resolve_path(config, config["artifacts"]["scores"])
    score_sidecar = rerank_module.validate_current_score_sidecar(
        config, score_path=score_path
    )
    official_depth = int(config["protocol"]["official_depth"])
    depths = [
        *[int(value) for value in config["protocol"]["diagnostic_depths"]],
        official_depth,
    ]
    run_paths = {depth: rerank_module.rerank_run_path(config, depth) for depth in depths}
    for depth, path in run_paths.items():
        _require_nonempty_regular_file(path, label=f"rerank K={depth} run")
    bm25_path = rerank_module.resolve_path(config, config["inputs"]["bm25_run"])
    qrels_path = rerank_module.resolve_path(config, config["inputs"]["qrels"])
    candidates_path = rerank_module.resolve_path(config, config["inputs"]["candidates"])
    for label, path in (
        ("BM25 run", bm25_path),
        ("qrels", qrels_path),
        ("candidate cache", candidates_path),
    ):
        _require_nonempty_regular_file(path, label=label)

    executable = rerank_module.resolve_trec_eval(config)
    trec_eval_provenance = rerank_module.validate_trec_eval_build_provenance(
        config, executable=executable
    )
    qrels = load_qrels(qrels_path, split="dev")
    query_ids = list(dict.fromkeys(qrels["query_id"].astype("string").map(str)))
    candidates = pd.read_parquet(candidates_path)
    candidates = candidates.loc[candidates["split"].astype("string").eq("dev")].copy()
    bm25_run = read_trec_run(str(bm25_path), split="dev")
    candidate_invariant = assert_candidate_set_invariant(candidates, bm25_run)
    rerank_runs = {
        depth: read_trec_run(str(path), split="dev")
        for depth, path in run_paths.items()
    }
    for run in rerank_runs.values():
        assert_candidate_set_invariant(candidates, run)
        assert_candidate_set_invariant(bm25_run, run)

    current_hashes = {
        "config": _sha256(Path(str(config["_config_path"]))),
        "evaluation_source": rerank_module.evaluation_source_sha256(config)[0],
        "scoring_source": score_sidecar["scoring_source_sha256"],
        "scores": _sha256(score_path),
        "bm25_run": _sha256(bm25_path),
        "qrels": _sha256(qrels_path),
        "candidates": _sha256(candidates_path),
        **{f"rerank_k{depth}": _sha256(path) for depth, path in run_paths.items()},
    }
    evaluation_fingerprint = rerank_module.canonical_json_sha256(current_hashes)
    output_paths = {
        name: rerank_module.resolve_path(config, config["metrics"][name])
        for name in _EVALUATION_REPORT_ORDER
    }
    work_dir = rerank_module.resolve_path(config, config["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(prefix="evaluation-generation-", dir=work_dir)
    )
    return {
        "score_path": score_path,
        "score_sidecar": score_sidecar,
        "official_depth": official_depth,
        "depths": depths,
        "run_paths": run_paths,
        "bm25_path": bm25_path,
        "qrels_path": qrels_path,
        "candidates_path": candidates_path,
        "executable": executable,
        "trec_eval_provenance": trec_eval_provenance,
        "qrels": qrels,
        "query_ids": query_ids,
        "candidates": candidates,
        "bm25_run": bm25_run,
        "candidate_invariant": candidate_invariant,
        "rerank_runs": rerank_runs,
        "evaluation_fingerprint": evaluation_fingerprint,
        "output_paths": output_paths,
        "staging_directory": staging_directory,
    }


def _evaluate_rerank_impl(args: argparse.Namespace) -> int:
    config = rerank_module.load_rerank_config(args.config)
    if args.split != "dev":
        raise ValueError("Phase 2 evaluation only supports dev")
    try:
        preflight = _rerank_evaluation_preflight(config)
    except Exception as exc:
        output_paths = {
            name: rerank_module.resolve_path(config, config["metrics"][name])
            for name in _EVALUATION_REPORT_ORDER
        }
        raise StageError(
            stage="evaluate-rerank/preflight",
            command=(
                "python -m rusearchrank.cli evaluate-rerank "
                "--config configs/rerank.yaml --split dev --overwrite"
            ),
            inputs={
                "config": str(config["_config_path"]),
                "trec_eval_provenance": str(
                    rerank_module.resolve_path(
                        config, config["evaluation"]["trec_eval_provenance_path"]
                    )
                ),
            },
            outputs={name: str(path) for name, path in output_paths.items()},
            root_cause=str(exc),
            reusable=(
                "all existing production metrics, score-Parquet, and TREC runs "
                "are byte-for-byte untouched"
            ),
            repeat_cell="Cell 12: fix the named preflight input, then repeat evaluate-rerank",
            rollback_status="NOT_STARTED",
        ) from exc
    score_path = preflight["score_path"]
    score_sidecar = preflight["score_sidecar"]
    official_depth = preflight["official_depth"]
    depths = preflight["depths"]
    run_paths = preflight["run_paths"]
    bm25_path = preflight["bm25_path"]
    qrels_path = preflight["qrels_path"]
    candidates_path = preflight["candidates_path"]
    executable = preflight["executable"]
    trec_eval_provenance = preflight["trec_eval_provenance"]
    qrels = preflight["qrels"]
    query_ids = preflight["query_ids"]
    candidates = preflight["candidates"]
    bm25_run = preflight["bm25_run"]
    candidate_invariant = preflight["candidate_invariant"]
    rerank_runs = preflight["rerank_runs"]
    evaluation_fingerprint = preflight["evaluation_fingerprint"]
    output_paths = preflight["output_paths"]
    staging_directory = preflight["staging_directory"]
    setattr(args, "_evaluation_staging_directory", str(staging_directory))
    existing = {name: path.exists() for name, path in output_paths.items()}
    if all(existing.values()) and not args.overwrite:
        reports = {name: _read_json(path) for name, path in output_paths.items()}
        if all(
            report.get("status") == "PASS"
            and report.get("evaluation_fingerprint") == evaluation_fingerprint
            for report in reports.values()
        ):
            _print_json(
                {
                    "status": "PASS",
                    "action": "reused_valid_evaluation",
                    "evaluation_fingerprint": evaluation_fingerprint,
                    "outputs": {
                        name: rerank_module.portable_path(config, path)
                        for name, path in output_paths.items()
                    },
                }
            )
            return 0
        raise ValueError(
            "existing evaluation outputs are stale or invalid and were preserved; "
            "use --overwrite after review"
        )
    if any(existing.values()) and not all(existing.values()) and not args.overwrite:
        raise ValueError(
            f"partial evaluation output detected and preserved: {existing}; "
            "use --overwrite after review"
        )
    baseline_trec = _execute_rerank_trec_eval(
        config,
        run_path=bm25_path,
        executable=executable,
        trec_eval_provenance=trec_eval_provenance,
    )
    depth_trec = {
        depth: _execute_rerank_trec_eval(
            config,
            run_path=path,
            executable=executable,
            trec_eval_provenance=trec_eval_provenance,
        )
        for depth, path in run_paths.items()
    }

    tolerance = float(config["evaluation"]["python_vs_trec_eval_tolerance"])
    baseline_vector = baseline_trec["parsed"]["per_query_ndcg_at_10"]
    baseline_rank_desc = _trec_python_ranking(bm25_run, docid_ascending=False)
    baseline_rank_asc = _trec_python_ranking(bm25_run, docid_ascending=True)
    baseline_python_desc = evaluate_bm25_metrics(
        baseline_rank_desc, qrels, query_ids=query_ids
    )
    baseline_python_asc = evaluate_bm25_metrics(
        baseline_rank_asc, qrels, query_ids=query_ids
    )
    desc_diff = _vector_difference(
        _python_metric_vector(baseline_python_desc), baseline_vector
    )
    asc_diff = _vector_difference(
        _python_metric_vector(baseline_python_asc), baseline_vector
    )
    tie_break_audit = classify_bm25_tie_break_audit(
        docno_desc_max_abs_difference=desc_diff,
        docno_asc_max_abs_difference=asc_diff,
        tolerance=tolerance,
    )
    if tie_break_audit["conclusion"] == "no_policy_matches":
        raise ValueError(
            "BM25 tie-break audit concluded no_policy_matches: "
            + json.dumps(tie_break_audit, ensure_ascii=False, sort_keys=True)
        )
    if tie_break_audit["conclusion"] == "docno_desc":
        baseline_ranking = baseline_rank_desc
        baseline_python = baseline_python_desc
        tie_break_audit["python_representative"] = "docno_desc"
    else:
        baseline_ranking = baseline_rank_asc
        baseline_python = baseline_python_asc
        tie_break_audit["python_representative"] = (
            "docno_asc"
            if tie_break_audit["conclusion"] == "docno_asc"
            else "docno_asc_metric_equivalent_representative_not_inferred_policy"
        )
    aggregate_baseline_difference = abs(
        float(baseline_python["aggregate"]["ndcg_at_10"])
        - float(baseline_trec["parsed"]["ndcg_at_10"])
    )
    if aggregate_baseline_difference > tolerance:
        raise ValueError("Python BM25 nDCG@10 differs from official trec_eval")
    python_by_depth: dict[int, dict[str, Any]] = {}
    sparse_by_depth: dict[int, dict[str, Any]] = {}
    comparison_by_depth: dict[int, dict[str, Any]] = {}
    baseline_sparse = sparse_judgment_diagnostics(
        candidates=candidates,
        qrels=qrels,
        ranking=baseline_ranking,
        bm25_ranking=baseline_ranking,
        query_ids=query_ids,
    )
    for depth, run in rerank_runs.items():
        ranking = run.rename(columns={"source_rank": "bm25_rank"})[
            ["query_id", "docid", "bm25_rank"]
        ]
        python_report = evaluate_bm25_metrics(
            ranking, qrels, query_ids=query_ids
        )
        python_by_depth[depth] = python_report
        if depth == official_depth:
            difference = abs(
                float(python_report["aggregate"]["ndcg_at_10"])
                - float(depth_trec[depth]["parsed"]["ndcg_at_10"])
            )
            if difference > tolerance:
                raise ValueError(
                    f"Python reranker nDCG@10 differs from trec_eval by {difference}"
                )
        sparse_by_depth[depth] = sparse_judgment_diagnostics(
            candidates=candidates,
            qrels=qrels,
            ranking=ranking,
            bm25_ranking=baseline_ranking,
            query_ids=query_ids,
        )
        comparison_by_depth[depth] = paired_ranking_comparison(
            baseline_vector,
            depth_trec[depth]["parsed"]["per_query_ndcg_at_10"],
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            seed=int(config["evaluation"]["bootstrap_seed"]),
            confidence=float(config["evaluation"]["bootstrap_confidence"]),
        )

    official_trec = depth_trec[official_depth]
    official_sparse = sparse_by_depth[official_depth]
    official_comparison = comparison_by_depth[official_depth]
    recall_difference = abs(
        float(baseline_trec["parsed"]["recall_at_100"])
        - float(official_trec["parsed"]["recall_at_100"])
    )
    if recall_difference > 1e-12:
        raise ValueError(
            "trec_eval recall.100 differs despite exact candidate-set equality"
        )
    sparse_inversion_delta = {
        "queries_with_unjudged_above_relevant": int(
            official_sparse["queries_with_unjudged_above_relevant"]
            - baseline_sparse["queries_with_unjudged_above_relevant"]
        ),
        "pairwise_unjudged_relevant_inversions": int(
            official_sparse["pairwise_unjudged_relevant_inversions"]
            - baseline_sparse["pairwise_unjudged_relevant_inversions"]
        ),
    }
    baseline_vector_python = _python_metric_vector(baseline_python)
    system_vector_python = _python_metric_vector(
        python_by_depth[official_depth]
    )
    stratification = stratified_delta_summary(
        candidates=candidates,
        baseline_per_query=baseline_vector_python,
        system_per_query=system_vector_python,
        oracle_per_query=baseline_sparse["oracle_per_query"],
    )
    expected_at_oracle = (
        int(stratification["no_relevant_in_candidates"]["query_count"])
        + int(stratification["already_at_oracle"]["query_count"])
    )
    if int(baseline_sparse["queries_at_oracle_under_bm25"]) != expected_at_oracle:
        raise RuntimeError(
            "queries_at_oracle_under_bm25 is inconsistent with the disjoint "
            "no-relevant/already-at-oracle strata"
        )
    if int(baseline_sparse["queries_without_relevant_candidate"]) != int(
        stratification["no_relevant_in_candidates"]["query_count"]
    ):
        raise RuntimeError(
            "no_relevant_in_candidates stratum disagrees with sparse diagnostics"
        )
    pooling_bias_suspected = bool(
        float(official_comparison["mean_delta"]) > 0
        and float(official_sparse["judged_at_10"])
        < float(baseline_sparse["judged_at_10"]) - 0.05
    )

    score_table = pq.read_table(score_path, columns=["query_id", "docid", "score"])
    raw_scores = score_table.to_pandas()
    raw_tie_stats = raw_score_tie_statistics(raw_scores)
    baseline_ties = _raw_score_tie_count(bm25_run, "bm25_score")
    input_artifacts = {
        "scores": _rerank_file_metadata(config, score_path),
        "candidates": _rerank_file_metadata(config, candidates_path),
        "qrels": _rerank_file_metadata(config, qrels_path),
        "bm25_run": _rerank_file_metadata(config, bm25_path),
        **{
            f"rerank_run_k{depth}": _rerank_file_metadata(config, path)
            for depth, path in run_paths.items()
        },
    }
    baseline_common = _metric_common_provenance(
        config,
        score_sidecar=score_sidecar,
        input_artifacts=input_artifacts,
        evaluation_fingerprint=evaluation_fingerprint,
        score_ties=raw_tie_stats,
        ranking_score_tie_rows=baseline_ties,
    )
    system_common = _metric_common_provenance(
        config,
        score_sidecar=score_sidecar,
        input_artifacts=input_artifacts,
        evaluation_fingerprint=evaluation_fingerprint,
        score_ties=raw_tie_stats,
        ranking_score_tie_rows=int(raw_tie_stats["rows_in_raw_score_ties"]),
    )
    sanity_reference = float(config["evaluation"]["expected_bm25_ndcg_at_10"])
    sanity_difference = float(baseline_trec["parsed"]["ndcg_at_10"]) - sanity_reference
    baseline_payload = {
        "status": "PASS",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "official": {
            "metric": "standard_nDCG@10",
            "value": float(baseline_trec["parsed"]["ndcg_at_10"]),
            "tool": "NIST trec_eval",
            "expected_release_version": str(
                config["evaluation"]["trec_eval_expected_release"]
            ),
            "binary_reported_version": trec_eval_provenance[
                "binary_reported_version"
            ],
        },
        "diagnostic": {
            "diagnostic": True,
            "recall_at_100": float(baseline_trec["parsed"]["recall_at_100"]),
            "mrr_at_10": float(baseline_trec["parsed"]["mrr_at_10"]),
            "sparse_judgments": baseline_sparse,
        },
        "trec_eval": baseline_trec,
        "trec_eval_provenance": trec_eval_provenance,
        "python_cross_check": {
            "ndcg_at_10": float(baseline_python["aggregate"]["ndcg_at_10"]),
            "absolute_difference": aggregate_baseline_difference,
            "tolerance": tolerance,
            "tie_break": tie_break_audit,
            "per_query_vector_checked": len(baseline_vector_python),
        },
        "phase1_sanity_reference": {
            "expected": sanity_reference,
            "recomputed": float(baseline_trec["parsed"]["ndcg_at_10"]),
            "difference": sanity_difference,
            "difference_exceeds_0_0005": abs(sanity_difference) > 0.0005,
            "explanation_if_exceeded": (
                "The Phase 2 baseline uses -M 100 and trec_eval score/docno "
                "tie-breaking on the explicit top-100 run."
                if abs(sanity_difference) > 0.0005
                else None
            ),
        },
        **baseline_common,
    }
    system_payload = {
        "status": "PASS",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "official": {
            "metric": "standard_nDCG@10",
            "value": float(official_trec["parsed"]["ndcg_at_10"]),
            "tool": "NIST trec_eval",
            "expected_release_version": str(
                config["evaluation"]["trec_eval_expected_release"]
            ),
            "binary_reported_version": trec_eval_provenance[
                "binary_reported_version"
            ],
            "depth": official_depth,
        },
        "diagnostic": {
            "diagnostic": True,
            "recall_at_100": float(official_trec["parsed"]["recall_at_100"]),
            "mrr_at_10": float(official_trec["parsed"]["mrr_at_10"]),
            "sparse_judgments": official_sparse,
        },
        "trec_eval": official_trec,
        "trec_eval_provenance": trec_eval_provenance,
        "python_cross_check": {
            "ndcg_at_10": float(
                python_by_depth[official_depth]["aggregate"]["ndcg_at_10"]
            ),
            "absolute_difference": abs(
                float(python_by_depth[official_depth]["aggregate"]["ndcg_at_10"])
                - float(official_trec["parsed"]["ndcg_at_10"])
            ),
            "tolerance": tolerance,
            "tie_break_irrelevant": True,
        },
        **system_common,
    }
    comparison_payload = {
        "status": "PASS",
        "trec_eval_provenance": trec_eval_provenance,
        "official_metric": "standard_nDCG@10",
        "paired": official_comparison,
        "recall_at_100_invariant": {
            **candidate_invariant,
            "trec_eval_absolute_difference": recall_difference,
            "strict_set_equality_is_primary": True,
        },
        "sparse_judgment_delta": sparse_inversion_delta,
        "stratified_mean_delta": stratification,
        "stratification_metric_source": (
            "python_full_precision_evaluate_bm25_metrics_for_bm25_system_and_oracle"
        ),
        "pooling_bias_suspected": pooling_bias_suspected,
        "condensed_ndcg_at_10": {
            "diagnostic": True,
            "bm25": float(baseline_sparse["condensed_ndcg_at_10"]),
            "reranker": float(official_sparse["condensed_ndcg_at_10"]),
            "delta": float(
                official_sparse["condensed_ndcg_at_10"]
                - baseline_sparse["condensed_ndcg_at_10"]
            ),
        },
        **system_common,
    }
    depth_entries: list[dict[str, Any]] = []
    for depth in depths:
        sparse = sparse_by_depth[depth]
        paired = comparison_by_depth[depth]
        depth_entries.append(
            {
                "depth": depth,
                "official": depth == official_depth,
                "diagnostic": depth != official_depth,
                "ndcg_at_10": float(depth_trec[depth]["parsed"]["ndcg_at_10"]),
                "judged_at_10": float(sparse["judged_at_10"]),
                "condensed_ndcg_at_10": float(sparse["condensed_ndcg_at_10"]),
                "improved": int(paired["improved"]),
                "degraded": int(paired["degraded"]),
                "tie": int(paired["tie"]),
                "candidate_set_invariant": True,
                "run": _rerank_file_metadata(config, run_paths[depth]),
                "trec_eval": depth_trec[depth],
            }
        )
    depth_payload = {
        "status": "PASS",
        "trec_eval_provenance": trec_eval_provenance,
        "baseline_sha256": _sha256(bm25_path),
        "depths": depth_entries,
        **system_common,
    }
    reports = {
        "baseline": baseline_payload,
        "system": system_payload,
        "comparison": comparison_payload,
        "depth_profile": depth_payload,
    }
    try:
        staged_paths = _stage_evaluation_generation(
            reports,
            staging_directory=staging_directory,
            evaluation_fingerprint=evaluation_fingerprint,
        )
    except Exception as exc:
        raise StageError(
            stage="evaluate-rerank/temporary-calculation-validation",
            command=(
                "python -m rusearchrank.cli evaluate-rerank "
                "--config configs/rerank.yaml --split dev --overwrite"
            ),
            inputs={
                "score_parquet": str(score_path),
                "trec_eval_provenance": str(
                    rerank_module.resolve_path(
                        config, config["evaluation"]["trec_eval_provenance_path"]
                    )
                ),
            },
            outputs={
                "temporary_generation": str(staging_directory),
                **{name: str(path) for name, path in output_paths.items()},
            },
            root_cause=str(exc),
            reusable=(
                "all existing production metrics, score-Parquet, and TREC runs "
                "are byte-for-byte untouched"
            ),
            repeat_cell="Cell 12: inspect the temporary generation, then repeat evaluation",
            rollback_status="NOT_STARTED",
        ) from exc
    publication = _publish_evaluation_generation(
        config=config,
        output_paths=output_paths,
        staged_paths=staged_paths,
    )
    try:
        staging_directory.rmdir()
    except OSError:
        pass
    _print_json(
        {
            "status": "PASS",
            "action": "evaluated",
            "official_ndcg_at_10": system_payload["official"]["value"],
            "recomputed_bm25_ndcg_at_10": baseline_payload["official"]["value"],
            "mean_delta": official_comparison["mean_delta"],
            "pooling_bias_suspected": pooling_bias_suspected,
            "evaluation_fingerprint": evaluation_fingerprint,
            "trec_eval_provenance": trec_eval_provenance,
            "publication": publication,
            "outputs": {
                name: rerank_module.portable_path(config, path)
                for name, path in output_paths.items()
            },
        }
    )
    return 0


def _evaluate_rerank(args: argparse.Namespace) -> int:
    try:
        return _evaluate_rerank_impl(args)
    except StageError:
        raise
    except Exception as exc:
        config = rerank_module.load_rerank_config(args.config)
        output_paths = {
            name: rerank_module.resolve_path(config, config["metrics"][name])
            for name in _EVALUATION_REPORT_ORDER
        }
        temporary = str(
            getattr(
                args,
                "_evaluation_staging_directory",
                rerank_module.resolve_path(config, config["paths"]["work_dir"]),
            )
        )
        raise StageError(
            stage="evaluate-rerank/calculation-validation",
            command=(
                "python -m rusearchrank.cli evaluate-rerank "
                "--config configs/rerank.yaml --split dev --overwrite"
            ),
            inputs={
                "config": str(config["_config_path"]),
                "temporary_generation": temporary,
            },
            outputs={name: str(path) for name, path in output_paths.items()},
            root_cause=str(exc),
            reusable=(
                "all existing production metrics, score-Parquet, and TREC runs "
                "are byte-for-byte untouched"
            ),
            repeat_cell="Cell 12: inspect the temporary generation, then repeat evaluation",
            rollback_status="NOT_STARTED",
        ) from exc


def _package_phase2(args: argparse.Namespace) -> int:
    config = rerank_module.load_rerank_config(args.config)
    report = rerank_module.package_phase2(config, overwrite=bool(args.overwrite))
    _print_json(report)
    return 0


def _load_phase3_config(args: argparse.Namespace) -> dict[str, Any]:
    return training_data_module.load_finetune_config(args.config)


def _build_training_split_phase3(args: argparse.Namespace) -> int:
    report = training_data_module.build_training_split(
        _load_phase3_config(args), overwrite=bool(args.overwrite)
    )
    _print_json(report)
    return 0


def _build_training_pairs_phase3(args: argparse.Namespace) -> int:
    report = training_data_module.build_training_pairs(
        _load_phase3_config(args),
        regime=str(args.regime),
        overwrite=bool(args.overwrite),
    )
    _print_json(report)
    return 0


def _validate_checkpoint_phase3(args: argparse.Namespace) -> int:
    report = training_module.validate_checkpoint(
        _load_phase3_config(args), checkpoint=str(args.checkpoint)
    )
    _print_json(report)
    return 0


def _smoke_finetune_phase3(args: argparse.Namespace) -> int:
    report = training_module.smoke_finetune(
        _load_phase3_config(args), limit_pairs=int(args.limit_pairs)
    )
    _print_json(report)
    return 0


def _finetune_phase3(args: argparse.Namespace) -> int:
    report = training_module.run_finetune(
        _load_phase3_config(args),
        run_id=str(args.run_id),
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
    )
    _print_json(report)
    return 0


def _select_checkpoint_phase3(args: argparse.Namespace) -> int:
    report = phase3_eval_module.select_checkpoint(
        _load_phase3_config(args), overwrite=bool(args.overwrite)
    )
    _print_json(report)
    return 0


def _prepare_dev_evaluation_phase3(args: argparse.Namespace) -> int:
    report = phase3_eval_module.prepare_dev_evaluation(_load_phase3_config(args))
    _print_json(report)
    return 0


def _score_finetuned_phase3(args: argparse.Namespace) -> int:
    report = phase3_eval_module.score_finetuned(
        _load_phase3_config(args), overwrite=bool(args.overwrite)
    )
    _print_json(report)
    return 0


def _evaluate_phase3(args: argparse.Namespace) -> int:
    report = phase3_eval_module.evaluate_phase3(
        _load_phase3_config(args), overwrite=bool(args.overwrite)
    )
    _print_json(report)
    return 0


def _package_phase3(args: argparse.Namespace) -> int:
    report = phase3_eval_module.package_phase3(
        _load_phase3_config(args), overwrite=bool(args.overwrite)
    )
    _print_json(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rusearchrank.cli",
        description=(
            "RuSearchRank Phase 0 checks, guarded Phase 1 retrieval, and "
            "Phase 2 zero-shot reranking plus isolated Phase 3 fine-tuning"
        ),
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
    annotations_parser.add_argument(
        "--split",
        choices=("all", "train"),
        default="all",
        help="materialize all annotations or exactly one split",
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

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="validate the environment and all inputs before a heavy Phase 1 stage",
    )
    preflight_parser.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    preflight_parser.add_argument(
        "--stage",
        choices=("retrieval", "candidate-cache", "package", "rerank"),
        required=True,
    )
    preflight_parser.add_argument(
        "--check-index",
        action="store_true",
        help="open/download and smoke-test the official prebuilt index",
    )
    preflight_parser.set_defaults(func=_preflight_command)

    bm25_parser = subparsers.add_parser(
        "run-bm25", help="run guarded full Pyserini BM25 retrieval"
    )
    bm25_parser.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    bm25_parser.add_argument(
        "--split", choices=("train", "dev", "all"), default="all"
    )
    bm25_parser.add_argument("--overwrite", action="store_true")
    bm25_parser.set_defaults(func=_run_bm25)

    smoke_parser = subparsers.add_parser(
        "smoke-corpus-access",
        help=(
            "real cheap corpus/Parquet/ZIP round trip against the pinned "
            "revision; required before the full candidate cache"
        ),
    )
    smoke_parser.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    smoke_parser.add_argument(
        "--shard-index", type=int, default=0, help="which official shard to sample"
    )
    smoke_parser.add_argument(
        "--max-rows", type=int, default=2000, help="rows parsed from the shard"
    )
    smoke_parser.add_argument(
        "--min-passages",
        type=int,
        default=25,
        help="minimum real passages the smoke must filter and materialize",
    )
    smoke_parser.add_argument(
        "--shards-dir",
        default=None,
        help="use already-materialized local shards instead of the Hub",
    )
    smoke_parser.add_argument(
        "--output", default=str(DEFAULT_AUDIT_DIR / "corpus_smoke.json")
    )
    smoke_parser.set_defaults(func=_smoke_corpus_access)

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

    rerank_smoke_parser = subparsers.add_parser(
        "smoke-rerank",
        help="run a real pinned-model forward and temporary Phase 2 round trip",
    )
    rerank_smoke_parser.add_argument(
        "--config", default=str(rerank_module.DEFAULT_RERANK_CONFIG)
    )
    rerank_smoke_parser.add_argument("--limit", type=int, default=64)
    rerank_smoke_parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    rerank_smoke_parser.add_argument(
        "--output", default="reports/audit/rerank_smoke.json"
    )
    rerank_smoke_parser.set_defaults(func=_smoke_rerank)

    rerank_score_parser = subparsers.add_parser(
        "rerank-score",
        help="score dev BM25 candidates with guarded sharding and resume",
    )
    rerank_score_parser.add_argument(
        "--config", default=str(rerank_module.DEFAULT_RERANK_CONFIG)
    )
    rerank_score_parser.add_argument("--split", choices=("dev",), required=True)
    rerank_score_parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    rerank_score_parser.add_argument("--batch-size", type=int, default=None)
    rerank_score_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional local fine-tuned checkpoint directory.",
    )
    rerank_score_parser.add_argument("--overwrite", action="store_true")
    rerank_score_parser.set_defaults(func=_rerank_score)

    rerank_run_parser = subparsers.add_parser(
        "build-rerank-run",
        help="build a rank-preserving TREC run from raw float32 scores",
    )
    rerank_run_parser.add_argument(
        "--config", default=str(rerank_module.DEFAULT_RERANK_CONFIG)
    )
    rerank_run_parser.add_argument("--split", choices=("dev",), required=True)
    rerank_run_parser.add_argument(
        "--depth", type=int, choices=(10, 20, 50, 100), default=100
    )
    rerank_run_parser.add_argument("--overwrite", action="store_true")
    rerank_run_parser.set_defaults(func=_build_rerank_run)

    rerank_evaluate_parser = subparsers.add_parser(
        "evaluate-rerank",
        help="run official NIST evaluation and Phase 2 diagnostics",
    )
    rerank_evaluate_parser.add_argument(
        "--config", default=str(rerank_module.DEFAULT_RERANK_CONFIG)
    )
    rerank_evaluate_parser.add_argument(
        "--split", choices=("dev",), required=True
    )
    rerank_evaluate_parser.add_argument("--overwrite", action="store_true")
    rerank_evaluate_parser.set_defaults(func=_evaluate_rerank)

    phase2_package_parser = subparsers.add_parser(
        "package-phase2",
        help="snapshot, manifest, package, and revalidate Phase 2 results",
    )
    phase2_package_parser.add_argument(
        "--config", default=str(rerank_module.DEFAULT_RERANK_CONFIG)
    )
    phase2_package_parser.add_argument("--overwrite", action="store_true")
    phase2_package_parser.set_defaults(func=_package_phase2)

    phase3_split_parser = subparsers.add_parser(
        "build-training-split",
        help="materialize the isolated train-fit/train-validation query split",
    )
    phase3_split_parser.add_argument(
        "--config", default=str(training_data_module.DEFAULT_FINETUNE_CONFIG)
    )
    phase3_split_parser.add_argument("--overwrite", action="store_true")
    phase3_split_parser.set_defaults(func=_build_training_split_phase3)

    phase3_pairs_parser = subparsers.add_parser(
        "build-training-pairs",
        help="materialize one preregistered Phase 3 pair regime",
    )
    phase3_pairs_parser.add_argument(
        "--config", default=str(training_data_module.DEFAULT_FINETUNE_CONFIG)
    )
    phase3_pairs_parser.add_argument(
        "--regime", choices=training_data_module.PAIR_REGIMES, required=True
    )
    phase3_pairs_parser.add_argument("--overwrite", action="store_true")
    phase3_pairs_parser.set_defaults(func=_build_training_pairs_phase3)

    phase3_checkpoint_parser = subparsers.add_parser(
        "validate-checkpoint",
        help="evaluate S0 or a local checkpoint on train validation groups",
    )
    phase3_checkpoint_parser.add_argument(
        "--config", default=str(training_data_module.DEFAULT_FINETUNE_CONFIG)
    )
    phase3_checkpoint_parser.add_argument("--checkpoint", default="base")
    phase3_checkpoint_parser.set_defaults(func=_validate_checkpoint_phase3)

    phase3_smoke_parser = subparsers.add_parser(
        "smoke-finetune",
        help="run the real-model two-step Phase 3 CUDA smoke gate",
    )
    phase3_smoke_parser.add_argument(
        "--config", default=str(training_data_module.DEFAULT_FINETUNE_CONFIG)
    )
    phase3_smoke_parser.add_argument("--limit-pairs", type=int, default=64)
    phase3_smoke_parser.set_defaults(func=_smoke_finetune_phase3)

    phase3_train_parser = subparsers.add_parser(
        "finetune", help="run a registered CUDA fine-tuning experiment"
    )
    phase3_train_parser.add_argument(
        "--config", default=str(training_data_module.DEFAULT_FINETUNE_CONFIG)
    )
    phase3_train_parser.add_argument(
        "--run-id", choices=("C1", "A1", "A2", "B1"), required=True
    )
    phase3_train_parser.add_argument("--resume", action="store_true")
    phase3_train_parser.add_argument("--overwrite", action="store_true")
    phase3_train_parser.set_defaults(func=_finetune_phase3)

    phase3_select_parser = subparsers.add_parser(
        "select-checkpoint",
        help="select and atomically publish the validation winner before dev access",
    )
    phase3_select_parser.add_argument(
        "--config", default=str(training_data_module.DEFAULT_FINETUNE_CONFIG)
    )
    phase3_select_parser.add_argument("--overwrite", action="store_true")
    phase3_select_parser.set_defaults(func=_select_checkpoint_phase3)

    phase3_prepare_parser = subparsers.add_parser(
        "prepare-dev-evaluation",
        help="append the ledger and materialize guarded evaluation qrels",
    )
    phase3_prepare_parser.add_argument(
        "--config", default=str(training_data_module.DEFAULT_FINETUNE_CONFIG)
    )
    phase3_prepare_parser.set_defaults(func=_prepare_dev_evaluation_phase3)

    phase3_score_parser = subparsers.add_parser(
        "score-finetuned",
        help="score exactly the published best_finetuned checkpoint",
    )
    phase3_score_parser.add_argument(
        "--config", default=str(training_data_module.DEFAULT_FINETUNE_CONFIG)
    )
    phase3_score_parser.add_argument("--overwrite", action="store_true")
    phase3_score_parser.set_defaults(func=_score_finetuned_phase3)

    phase3_evaluate_parser = subparsers.add_parser(
        "evaluate-phase3",
        help="run final BM25/zero-shot/fine-tuned comparison and diagnostics",
    )
    phase3_evaluate_parser.add_argument(
        "--config", default=str(training_data_module.DEFAULT_FINETUNE_CONFIG)
    )
    phase3_evaluate_parser.add_argument("--overwrite", action="store_true")
    phase3_evaluate_parser.set_defaults(func=_evaluate_phase3)

    phase3_package_parser = subparsers.add_parser(
        "package-phase3",
        help="build and revalidate the exact Phase 3 result and model archives",
    )
    phase3_package_parser.add_argument(
        "--config", default=str(training_data_module.DEFAULT_FINETUNE_CONFIG)
    )
    phase3_package_parser.add_argument("--overwrite", action="store_true")
    phase3_package_parser.set_defaults(func=_package_phase3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except StageError as exc:
        print(f"error: stage {exc.stage} failed\n{exc}", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
