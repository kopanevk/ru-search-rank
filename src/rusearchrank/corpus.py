"""Revision-pinned MIRACL corpus access over official static JSONL.GZ shards.

The official ``miracl/miracl-corpus`` repository still ships a legacy
``miracl-corpus.py`` dataset script. Modern ``datasets`` releases refuse to
execute repository scripts, so ``load_dataset("miracl/miracl-corpus", "ru")``
fails with *Dataset scripts are no longer supported*. Phase 1 therefore never
touches the ``datasets`` loader: it resolves the immutable revision through the
Hugging Face Hub, downloads the numbered static shards
``miracl-corpus-v1.0-<language>/docs-<index>.jsonl.gz`` and parses them line by
line. Nothing here imports ``datasets`` and nothing enables remote code.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Iterator
import zlib

import pyarrow as pa
import pyarrow.parquet as pq


CORPUS_FIELDS = ("docid", "title", "text")
SHARD_TEMPLATE = "miracl-corpus-v1.0-{language}/docs-{index}.jsonl.gz"
_SHARD_PATTERN = re.compile(
    r"^miracl-corpus-v1\.0-(?P<language>[a-z]{2,3})/docs-(?P<index>\d+)\.jsonl\.gz$"
)
_LOADER_DENYLIST = ("miracl-corpus.py", "dataset_infos.json")
PASSAGE_SCHEMA = pa.schema(
    [
        pa.field("docid", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ]
)


class CorpusAccessError(RuntimeError):
    """The pinned repository, revision, or shard could not be resolved."""


class CorpusRowError(ValueError):
    """A corpus line is missing, malformed, or violates the passage contract."""

    def __init__(self, message: str, *, shard: str, line_number: int) -> None:
        self.shard = shard
        self.line_number = line_number
        super().__init__(f"{shard}:{line_number}: {message}")


class MissingCandidatePassagesError(ValueError):
    """Candidate docids were absent from the revision-pinned corpus shards."""

    def __init__(
        self,
        missing_docids: list[str],
        *,
        candidate_context: dict[str, list[dict[str, str]]] | None = None,
        shards_visited: int | None = None,
        lines_visited: int | None = None,
    ) -> None:
        self.missing_docids = list(missing_docids)
        self.shards_visited = shards_visited
        self.lines_visited = lines_visited
        details: list[str] = []
        for docid in self.missing_docids[:10]:
            locations = (candidate_context or {}).get(docid, [])
            location_text = ", ".join(
                f"split={row.get('split')}, query_id={row.get('query_id')}"
                for row in locations[:3]
            )
            details.append(
                f"docid={docid}" + (f" ({location_text})" if location_text else "")
            )
        scanned = ""
        if shards_visited is not None and lines_visited is not None:
            scanned = f" after scanning {shards_visited} shards / {lines_visited} lines"
        super().__init__(
            f"{len(self.missing_docids)} candidate passages missing from corpus"
            + scanned
            + "; first entries: "
            + "; ".join(details)
        )


def shard_index(name: str) -> int:
    """Return the numeric shard index encoded in an official shard path."""

    match = _SHARD_PATTERN.match(name)
    if match is None:
        raise ValueError(f"not an official MIRACL corpus shard name: {name!r}")
    return int(match.group("index"))


def sort_shard_names(names: Iterable[str]) -> list[str]:
    """Sort shard paths numerically (0, 1, 2, ... 19), never lexicographically."""

    return sorted(names, key=shard_index)


def shard_names(language: str, shard_count: int) -> list[str]:
    """Build the exact ordered official shard list for one language."""

    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty string")
    if not isinstance(shard_count, int) or shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    names = [
        SHARD_TEMPLATE.format(language=language, index=index)
        for index in range(shard_count)
    ]
    # Guard the ordering contract itself: numeric, not lexicographic.
    if names != sort_shard_names(names):
        raise RuntimeError("generated shard names are not in numeric order")
    return names


def assert_no_dataset_script(names: Iterable[str]) -> None:
    """Reject any attempt to route corpus access through a repository script."""

    offending = sorted(
        name for name in names if Path(name).name in _LOADER_DENYLIST
    )
    if offending:
        raise CorpusAccessError(
            "corpus access must use static JSONL.GZ shards, never a dataset "
            f"script; refusing: {', '.join(offending)}"
        )


def parse_corpus_line(raw_line: str, *, shard: str, line_number: int) -> dict[str, str]:
    """Parse and validate one official corpus line into a string-only record."""

    stripped = raw_line.strip()
    if not stripped:
        raise CorpusRowError("blank corpus line", shard=shard, line_number=line_number)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise CorpusRowError(
            f"invalid JSON ({exc.msg})", shard=shard, line_number=line_number
        ) from exc
    if not isinstance(payload, dict):
        raise CorpusRowError(
            f"corpus row must be a JSON object, found {type(payload).__name__}",
            shard=shard,
            line_number=line_number,
        )
    missing = [field for field in CORPUS_FIELDS if field not in payload]
    if missing:
        raise CorpusRowError(
            "corpus row is missing required fields: " + ", ".join(missing),
            shard=shard,
            line_number=line_number,
        )
    docid = payload["docid"]
    if not isinstance(docid, str):
        raise CorpusRowError(
            f"docid must stay a string, found {type(docid).__name__}",
            shard=shard,
            line_number=line_number,
        )
    if not docid.strip():
        raise CorpusRowError("empty docid", shard=shard, line_number=line_number)
    title = payload["title"]
    text = payload["text"]
    if title is not None and not isinstance(title, str):
        raise CorpusRowError(
            f"title must be a string or null, found {type(title).__name__}",
            shard=shard,
            line_number=line_number,
        )
    if not isinstance(text, str):
        raise CorpusRowError(
            f"text must be a string, found {type(text).__name__}",
            shard=shard,
            line_number=line_number,
        )
    return {"docid": docid, "title": "" if title is None else title, "text": text}


def iter_shard_rows(
    path: str | Path,
    *,
    shard: str,
    max_rows: int | None = None,
) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield ``(line_number, record)`` from one gzip-compressed JSONL shard."""

    shard_path = Path(path)
    if not shard_path.is_file():
        raise CorpusAccessError(f"corpus shard is not a regular file: {shard_path}")
    if shard_path.stat().st_size == 0:
        raise CorpusAccessError(f"corpus shard is empty: {shard_path}")
    emitted = 0
    try:
        stream = gzip.open(shard_path, "rt", encoding="utf-8")
    except OSError as exc:  # pragma: no cover - open() rarely fails on its own.
        raise CorpusAccessError(f"cannot open corpus shard {shard_path}: {exc}") from exc
    with stream:
        line_number = 0
        while True:
            try:
                raw_line = stream.readline()
            except (
                OSError,
                EOFError,
                gzip.BadGzipFile,
                UnicodeDecodeError,
                zlib.error,
            ) as exc:
                raise CorpusAccessError(
                    f"corrupted corpus shard {shard} at {shard_path} "
                    f"after {line_number} lines: {type(exc).__name__}: {exc}"
                ) from exc
            if not raw_line:
                break
            line_number += 1
            if not raw_line.strip():
                continue
            yield line_number, parse_corpus_line(
                raw_line, shard=shard, line_number=line_number
            )
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return


@dataclass(frozen=True)
class ShardSource:
    """Base contract: an ordered list of shard names resolvable to local files."""

    language: str
    shard_count: int

    def names(self) -> list[str]:
        return shard_names(self.language, self.shard_count)

    def describe(self) -> dict[str, Any]:  # pragma: no cover - overridden below.
        raise NotImplementedError

    def local_path(self, name: str) -> Path:  # pragma: no cover - overridden below.
        raise NotImplementedError


@dataclass(frozen=True)
class HubShardSource(ShardSource):
    """Official Hugging Face shards pinned to one immutable revision."""

    repo_id: str = "miracl/miracl-corpus"
    revision: str = ""
    cache_dir: Path | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "huggingface_hub",
            "repo_id": self.repo_id,
            "repo_type": "dataset",
            "revision": self.revision,
            "language": self.language,
            "shard_count": self.shard_count,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "loader": "hf_hub_download + gzip + json.loads",
            "dataset_script_used": False,
            "trust_remote_code": False,
        }

    def local_path(self, name: str) -> Path:
        assert_no_dataset_script([name])
        if not self.revision or len(self.revision) < 7:
            raise CorpusAccessError(
                "corpus access requires an immutable pinned revision; "
                f"got {self.revision!r}"
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - packaged dependency.
            raise CorpusAccessError(
                "corpus access requires the huggingface_hub package"
            ) from exc
        try:
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=name,
                repo_type="dataset",
                revision=self.revision,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
        except Exception as exc:  # Hub errors are re-raised with full context.
            raise CorpusAccessError(
                "failed to download official corpus shard "
                f"{name!r} from {self.repo_id!r} at revision {self.revision!r} "
                f"(repo_type=dataset, cache_dir={self.cache_dir}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return Path(downloaded)


@dataclass(frozen=True)
class LocalShardSource(ShardSource):
    """Already-materialized shards, used by offline fixtures and reruns."""

    directory: Path = Path(".")

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "local_directory",
            "directory": str(self.directory),
            "language": self.language,
            "shard_count": self.shard_count,
            "loader": "gzip + json.loads",
            "dataset_script_used": False,
            "trust_remote_code": False,
        }

    def local_path(self, name: str) -> Path:
        assert_no_dataset_script([name])
        path = Path(self.directory) / name
        if not path.is_file():
            raise CorpusAccessError(f"local corpus shard does not exist: {path}")
        return path


def resolve_hub_shards(
    *,
    repo_id: str,
    revision: str,
    language: str,
    shard_count: int,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Prove the pinned revision and every expected shard exist before download."""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - packaged dependency.
        raise CorpusAccessError(
            "corpus access requires the huggingface_hub package"
        ) from exc
    expected = shard_names(language, shard_count)
    assert_no_dataset_script(expected)
    api = HfApi()
    try:
        info = api.dataset_info(repo_id, revision=revision, timeout=timeout)
    except Exception as exc:
        raise CorpusAccessError(
            f"pinned corpus revision is unavailable: repo_id={repo_id!r}, "
            f"revision={revision!r}: {type(exc).__name__}: {exc}"
        ) from exc
    resolved = str(getattr(info, "sha", "") or "")
    if resolved and revision not in {resolved, resolved[: len(revision)]}:
        raise CorpusAccessError(
            f"corpus revision {revision!r} resolved to a different commit {resolved!r}"
        )
    try:
        paths = api.get_paths_info(
            repo_id, expected, repo_type="dataset", revision=revision
        )
    except Exception as exc:
        raise CorpusAccessError(
            f"cannot list pinned corpus shards for {repo_id!r} at {revision!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    sizes = {
        str(entry.path): int(getattr(entry, "size", 0) or 0)
        for entry in paths
        if str(getattr(entry, "path", ""))
    }
    missing = [name for name in expected if name not in sizes]
    if missing:
        raise CorpusAccessError(
            f"pinned corpus revision {revision!r} is missing shards: "
            + ", ".join(missing[:5])
        )
    return {
        "repo_id": repo_id,
        "revision": revision,
        "resolved_sha": resolved,
        "shard_count": len(expected),
        "shards": expected,
        "shard_sizes_bytes": {name: sizes[name] for name in expected},
        "dataset_script_used": False,
        "trust_remote_code": False,
    }


class _PassageBatchWriter:
    """Append validated passages to a temporary Parquet file in fixed batches."""

    def __init__(self, temporary_path: Path, *, batch_rows: int) -> None:
        if batch_rows <= 0:
            raise ValueError("batch_rows must be a positive integer")
        self.temporary_path = temporary_path
        self.batch_rows = batch_rows
        self._writer: pq.ParquetWriter | None = None
        self._docids: list[str] = []
        self._titles: list[str] = []
        self._texts: list[str] = []
        self.rows_written = 0
        self.batches_written = 0

    def append(self, record: dict[str, str]) -> None:
        self._docids.append(record["docid"])
        self._titles.append(record["title"])
        self._texts.append(record["text"])
        if len(self._docids) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self._docids:
            return
        table = pa.Table.from_pydict(
            {"docid": self._docids, "title": self._titles, "text": self._texts},
            schema=PASSAGE_SCHEMA,
        )
        if self._writer is None:
            self.temporary_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.temporary_path, PASSAGE_SCHEMA)
        self._writer.write_table(table)
        self.rows_written += len(self._docids)
        self.batches_written += 1
        self._docids.clear()
        self._titles.clear()
        self._texts.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            with self.temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())


def _validate_temporary_passages(path: Path, *, expected_rows: int) -> dict[str, Any]:
    """Re-read the temporary Parquet and enforce the passage contract."""

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    if [field.name for field in schema] != list(CORPUS_FIELDS):
        raise ValueError(
            f"temporary passage Parquet has unexpected columns: {schema.names}"
        )
    if any(not pa.types.is_string(field.type) for field in schema):
        raise ValueError("temporary passage Parquet columns must all be strings")
    row_count = int(parquet_file.metadata.num_rows)
    if row_count != expected_rows:
        raise ValueError(
            f"temporary passage Parquet holds {row_count} rows, expected {expected_rows}"
        )
    seen: set[str] = set()
    empty_titles = 0
    for batch in parquet_file.iter_batches(batch_size=50_000, columns=list(CORPUS_FIELDS)):
        for docid, title, text in zip(
            batch.column("docid").to_pylist(),
            batch.column("title").to_pylist(),
            batch.column("text").to_pylist(),
            strict=True,
        ):
            if not isinstance(docid, str) or not docid.strip():
                raise ValueError(f"temporary passage Parquet has an invalid docid: {docid!r}")
            if docid in seen:
                raise ValueError(
                    f"temporary passage Parquet has duplicate docid: {docid!r}"
                )
            seen.add(docid)
            if text is None or not str(text).strip():
                raise ValueError(f"passage {docid!r} has empty text")
            if title is None or not str(title).strip():
                empty_titles += 1
    return {"row_count": row_count, "empty_titles": empty_titles}


def extract_candidate_passages(
    candidate_docids: set[str] | frozenset[str],
    output_path: str | Path,
    *,
    source: ShardSource,
    batch_rows: int = 50_000,
    candidate_context: dict[str, list[dict[str, str]]] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stream official shards and materialize only the requested passages.

    The corpus is never loaded into memory or into a single dict: shards are read
    line by line, matched rows are appended in fixed batches to a temporary
    Parquet file, and reading stops as soon as no candidate docid remains. The
    final path is only published after the temporary file passes revalidation.
    """

    emit = log if log is not None else (lambda message: None)
    required = {str(docid) for docid in candidate_docids}
    if not required:
        raise ValueError("candidate_docids must contain non-empty identifiers")
    if any(not docid.strip() for docid in required):
        raise ValueError("candidate_docids must contain non-empty identifiers")

    final_path = Path(output_path)
    temporary_path = final_path.with_name(f"{final_path.name}.partial")
    preserved_partial: str | None = None
    if temporary_path.exists():
        # A partial file is never trusted, never published, and never silently
        # deleted: it is moved aside so a rerun can proceed without manual steps.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        preserved = temporary_path.with_name(f"{temporary_path.name}.stale.{stamp}")
        temporary_path.replace(preserved)
        preserved_partial = str(preserved)
        emit(
            "passage extraction: an interrupted run left "
            f"{temporary_path}; it was preserved as {preserved} and a fresh "
            "extraction starts now"
        )

    names = source.names()
    remaining = set(required)
    duplicate_docids: list[str] = []
    shards_visited = 0
    lines_visited = 0
    early_stop = False
    started = time.perf_counter()
    writer = _PassageBatchWriter(temporary_path, batch_rows=batch_rows)
    emit(
        f"passage extraction: {len(required)} unique candidate docids, "
        f"{len(names)} shards, batch_rows={batch_rows}"
    )
    try:
        for name in names:
            if not remaining:
                early_stop = True
                emit(f"passage extraction: early stop before {name}")
                break
            shard_path = source.local_path(name)
            shards_visited += 1
            shard_hits = 0
            rows = iter_shard_rows(shard_path, shard=name)
            with contextlib.closing(rows):
                for _, record in rows:
                    lines_visited += 1
                    docid = record["docid"]
                    if docid not in required:
                        continue
                    if docid not in remaining:
                        duplicate_docids.append(docid)
                        continue
                    if not record["text"].strip():
                        raise ValueError(
                            f"candidate passage {docid!r} has empty text in shard {name}"
                        )
                    writer.append(record)
                    remaining.discard(docid)
                    shard_hits += 1
                    if not remaining:
                        break
            emit(
                f"passage extraction: {name} -> {shard_hits} matched, "
                f"{len(remaining)} remaining, {lines_visited} lines read"
            )
        writer.close()
    except BaseException:
        with contextlib.suppress(Exception):
            writer.close()
        raise

    if remaining:
        raise MissingCandidatePassagesError(
            sorted(remaining),
            candidate_context=candidate_context,
            shards_visited=shards_visited,
            lines_visited=lines_visited,
        )
    if duplicate_docids:
        raise ValueError(
            f"{len(duplicate_docids)} duplicate candidate docids in the corpus; "
            "first entries: " + ", ".join(sorted(set(duplicate_docids))[:10])
        )
    if not temporary_path.is_file():
        raise RuntimeError("passage extraction produced no temporary Parquet file")

    validation = _validate_temporary_passages(
        temporary_path, expected_rows=len(required)
    )
    temporary_path.replace(final_path)
    elapsed = time.perf_counter() - started
    report = {
        "status": "PASS",
        "source": source.describe(),
        "required_docids": len(required),
        "found_docids": validation["row_count"],
        "missing_docids": 0,
        "duplicate_corpus_docids": 0,
        "extra_rows": validation["row_count"] - len(required),
        "empty_titles": validation["empty_titles"],
        "rows_written": writer.rows_written,
        "batches_written": writer.batches_written,
        "batch_rows": batch_rows,
        "shards_total": len(names),
        "shards_visited": shards_visited,
        "shards_skipped_by_early_stop": len(names) - shards_visited,
        "lines_visited": lines_visited,
        "early_stop": early_stop,
        # Duplicates can only be observed in the region actually streamed; the
        # early stop deliberately trades a full-corpus duplicate scan for speed.
        "duplicate_detection_scope": "scanned_lines_only",
        "preserved_partial_from_previous_run": preserved_partial,
        "output_path": str(final_path),
        "output_size_bytes": final_path.stat().st_size,
        "seconds": elapsed,
    }
    emit(
        "passage extraction: wrote "
        f"{report['rows_written']} passages in {report['batches_written']} batches "
        f"from {shards_visited}/{len(names)} shards in {elapsed:.1f}s"
    )
    return report
