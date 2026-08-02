"""Helpers that build real gzip-compressed MIRACL-shaped corpus shards.

Passage extraction is only trustworthy when it is exercised against genuine
``.jsonl.gz`` bytes, so tests never mock the corpus reader: they write real
shards with the official ``miracl-corpus-v1.0-<language>/docs-<index>.jsonl.gz``
layout and read them back through the production code path.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Sequence


def shard_path(directory: Path, *, language: str, index: int) -> Path:
    return Path(directory) / f"miracl-corpus-v1.0-{language}/docs-{index}.jsonl.gz"


def write_shard(
    directory: Path,
    *,
    language: str,
    index: int,
    rows: Iterable[dict[str, object]],
) -> Path:
    """Write one real gzip JSONL shard and return its path."""

    path = shard_path(directory, language=language, index=index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_corpus(
    directory: Path,
    *,
    language: str = "ru",
    shards: Sequence[Sequence[dict[str, object]]],
) -> list[Path]:
    """Write a whole numbered shard set (docs-0 .. docs-N) and return the paths."""

    return [
        write_shard(directory, language=language, index=index, rows=rows)
        for index, rows in enumerate(shards)
    ]


def passage(docid: str, *, title: str = "Заголовок", text: str | None = None) -> dict[str, str]:
    return {
        "docid": docid,
        "title": title,
        "text": text if text is not None else f"Русский текст пассажа {docid}.",
    }


def filler(prefix: str, count: int, *, start: int = 0) -> list[dict[str, str]]:
    """Non-candidate rows so extraction really has to skip and stream."""

    return [passage(f"{prefix}{index}#0") for index in range(start, start + count)]
