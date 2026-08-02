"""MIRACL data loading and the normalized Phase 1 candidate contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import ssl
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover - datasets normally supplies certifi.
    certifi = None


CANDIDATE_COLUMNS = (
    "split",
    "query_id",
    "docid",
    "bm25_rank",
    "bm25_score",
    "relevance_grade",
    "judgment",
    "relevance",
    "is_judged",
)
JUDGMENTS = frozenset({"relevant", "non_relevant", "unjudged"})

_MIRACL_ANNOTATIONS = "miracl/miracl"
_MIRACL_CORPUS = "miracl/miracl-corpus"
_HF_API = "https://huggingface.co/api/datasets"
_HF_RESOLVE = "https://huggingface.co/datasets"
_HF_ROWS = "https://datasets-server.huggingface.co/rows"


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _normalised_ids(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: frame[column].astype("string") for column in columns},
        index=frame.index,
    )


def validate_candidate_schema(candidates: pd.DataFrame) -> None:
    """Validate the normalized Phase 1 candidate table contract.

    Validation is deliberately strict about the relationship between
    ``judgment``, ``relevance`` and ``is_judged``. In particular, missing
    qrels are represented as ``unjudged`` and never as a negative label.
    """

    _require_columns(candidates, set(CANDIDATE_COLUMNS), "candidate table")

    identifier_columns = ("split", "query_id", "docid")
    null_ids = candidates[list(identifier_columns)].isna()
    if null_ids.any().any():
        bad_columns = [column for column in null_ids if null_ids[column].any()]
        raise ValueError(f"candidate identifiers contain nulls: {', '.join(bad_columns)}")

    ids = _normalised_ids(candidates, identifier_columns)
    empty_ids = ids.apply(lambda column: column.str.strip().eq(""))
    if empty_ids.any().any():
        raise ValueError("candidate identifiers must not be empty strings")

    duplicate_pairs = ids.duplicated(["split", "query_id", "docid"], keep=False)
    if duplicate_pairs.any():
        raise ValueError("duplicate (query_id, docid) pairs are not allowed")

    ranks = pd.to_numeric(candidates["bm25_rank"], errors="coerce")
    invalid_ranks = ranks.isna() | ~np.isfinite(ranks) | ranks.le(0) | ranks.mod(1).ne(0)
    if invalid_ranks.any():
        raise ValueError("bm25_rank must contain positive integers")

    rank_keys = pd.DataFrame(
        {
            "split": ids["split"],
            "query_id": ids["query_id"],
            "bm25_rank": ranks.astype("int64"),
        },
        index=candidates.index,
    )
    if rank_keys.duplicated(["split", "query_id", "bm25_rank"], keep=False).any():
        raise ValueError("duplicate bm25_rank values within a query are not allowed")

    scores = pd.to_numeric(candidates["bm25_score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores).all():
        raise ValueError("bm25_score must contain finite numeric values")

    judgment = candidates["judgment"].astype("string")
    invalid_judgments = ~judgment.isin(JUDGMENTS)
    if invalid_judgments.any():
        values = sorted(judgment[invalid_judgments].dropna().unique().tolist())
        raise ValueError(f"invalid judgment values: {values}")

    bool_values = candidates["is_judged"]
    actual_booleans = bool_values.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not actual_booleans.all():
        raise ValueError("is_judged must contain booleans")

    relevance = pd.to_numeric(candidates["relevance"], errors="coerce")
    non_numeric = candidates["relevance"].notna() & relevance.isna()
    if non_numeric.any():
        raise ValueError("relevance must contain 0, 1, or null")

    grades = pd.to_numeric(candidates["relevance_grade"], errors="coerce")
    non_numeric_grades = candidates["relevance_grade"].notna() & grades.isna()
    invalid_grades = grades.notna() & (
        ~np.isfinite(grades) | grades.lt(0) | grades.mod(1).ne(0)
    )
    if non_numeric_grades.any() or invalid_grades.any():
        raise ValueError("relevance_grade must contain non-negative integers or null")

    relevant = (
        judgment.eq("relevant")
        & grades.gt(0)
        & relevance.eq(1)
        & bool_values.eq(True)
    )
    non_relevant = (
        judgment.eq("non_relevant")
        & grades.eq(0)
        & relevance.eq(0)
        & bool_values.eq(True)
    )
    unjudged = (
        judgment.eq("unjudged")
        & candidates["relevance_grade"].isna()
        & candidates["relevance"].isna()
        & bool_values.eq(False)
    )
    valid_state = (relevant | non_relevant | unjudged).fillna(False)
    if not valid_state.all():
        rows = candidates.index[~valid_state].tolist()[:5]
        raise ValueError(f"inconsistent judgment fields at rows: {rows}")


def attach_qrels(candidates: pd.DataFrame, qrels: pd.DataFrame) -> pd.DataFrame:
    """Attach qrels while preserving grade and a distinct unjudged state."""

    base_columns = {"split", "query_id", "docid", "bm25_rank", "bm25_score"}
    _require_columns(candidates, base_columns, "BM25 candidates")
    _require_columns(qrels, {"query_id", "docid"}, "qrels")
    grade_columns = [
        column for column in ("relevance_grade", "relevance") if column in qrels.columns
    ]
    if not grade_columns:
        raise ValueError("qrels is missing relevance or relevance_grade")
    if len(grade_columns) == 2:
        left = pd.to_numeric(qrels["relevance_grade"], errors="coerce")
        right = pd.to_numeric(qrels["relevance"], errors="coerce")
        if not left.equals(right):
            raise ValueError("qrels relevance and relevance_grade columns conflict")
    source_grade = grade_columns[0]

    if candidates[["split", "query_id", "docid"]].isna().any().any():
        raise ValueError("BM25 candidate identifiers must not be null")
    if qrels[["query_id", "docid", source_grade]].isna().any().any():
        raise ValueError("qrels identifiers and relevance must not be null")

    base = candidates.drop(
        columns=["relevance_grade", "judgment", "relevance", "is_judged"],
        errors="ignore",
    ).copy()
    base["split"] = base["split"].astype("string")
    base["query_id"] = base["query_id"].astype("string")
    base["docid"] = base["docid"].astype("string")
    if base.duplicated(["split", "query_id", "docid"], keep=False).any():
        raise ValueError("duplicate (query_id, docid) pairs are not allowed")

    merge_keys = ["query_id", "docid"]
    if "split" in qrels.columns:
        merge_keys.insert(0, "split")
    elif base["split"].nunique() != 1:
        raise ValueError("qrels without split can only be joined to one candidate split")

    labels = qrels[merge_keys + [source_grade]].copy()
    if "split" in labels.columns:
        labels["split"] = labels["split"].astype("string")
    labels["query_id"] = labels["query_id"].astype("string")
    labels["docid"] = labels["docid"].astype("string")
    if labels.duplicated(merge_keys, keep=False).any():
        raise ValueError("qrels contain duplicate (query_id, docid) pairs")

    numeric_grade = pd.to_numeric(labels[source_grade], errors="coerce")
    invalid_grade = (
        numeric_grade.isna()
        | ~np.isfinite(numeric_grade)
        | numeric_grade.lt(0)
        | numeric_grade.mod(1).ne(0)
    )
    if invalid_grade.any():
        raise ValueError("qrels relevance grades must be non-negative integers")
    labels = labels.drop(columns=[source_grade]).assign(
        relevance_grade=numeric_grade.astype("Int64")
    )

    merged = base.merge(
        labels,
        on=merge_keys,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(merged) != len(base):
        raise ValueError("qrels join changed the number of candidate rows")
    merged["relevance_grade"] = merged["relevance_grade"].astype("Int64")
    merged["relevance"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    merged.loc[merged["relevance_grade"].gt(0), "relevance"] = 1
    merged.loc[merged["relevance_grade"].eq(0), "relevance"] = 0
    merged["is_judged"] = merged["relevance_grade"].notna().astype("bool")
    merged["judgment"] = pd.Series("unjudged", index=merged.index, dtype="string")
    merged.loc[merged["relevance_grade"].gt(0), "judgment"] = "relevant"
    merged.loc[merged["relevance_grade"].eq(0), "judgment"] = "non_relevant"

    ordered = list(CANDIDATE_COLUMNS) + [
        column for column in merged.columns if column not in CANDIDATE_COLUMNS
    ]
    merged = merged.loc[:, ordered]
    validate_candidate_schema(merged)
    return merged


def validate_queries(queries: pd.DataFrame) -> None:
    """Validate the normalized ``split/query_id/query_text`` table."""

    columns = ["split", "query_id", "query_text"]
    _require_columns(queries, set(columns), "queries table")
    if queries[columns].isna().any().any():
        raise ValueError("queries contain null values")
    normalized = queries[columns].astype("string")
    if normalized.apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("queries contain empty values")
    if normalized.duplicated(["split", "query_id"], keep=False).any():
        raise ValueError("queries contain duplicate (split, query_id) pairs")


def validate_passages(passages: pd.DataFrame) -> None:
    """Validate the normalized unique candidate-passage table."""

    columns = ["docid", "title", "text"]
    _require_columns(passages, set(columns), "passages table")
    if passages[["docid", "text"]].isna().any().any():
        raise ValueError("passages contain null docid or text")
    docids = passages["docid"].astype("string")
    if docids.str.strip().eq("").any():
        raise ValueError("passages contain empty docid")
    if docids.duplicated(keep=False).any():
        raise ValueError("passages contain duplicate docid values")
    if passages["text"].astype("string").str.strip().eq("").any():
        raise ValueError("passages contain empty text")


def extract_passages_from_rows(
    rows: Any, candidate_docids: set[str]
) -> pd.DataFrame:
    """Retain only requested passages from an iterable of official corpus rows."""

    requested = {str(docid) for docid in candidate_docids}
    if not requested or any(not docid.strip() for docid in requested):
        raise ValueError("candidate_docids must contain non-empty identifiers")

    found: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or not {"docid", "title", "text"}.issubset(row):
            raise ValueError("corpus row does not match docid/title/text schema")
        docid = str(row["docid"])
        if docid not in requested:
            continue
        title = "" if row["title"] is None else str(row["title"])
        if row["text"] is None or not str(row["text"]).strip():
            raise ValueError(f"candidate passage {docid!r} has empty text")
        text = str(row["text"])
        content = (title, text)
        if docid in found and found[docid] != content:
            raise ValueError(f"candidate passage {docid!r} has conflicting content")
        found[docid] = content
        if len(found) == len(requested):
            break

    missing = sorted(requested.difference(found))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"candidate passages missing from corpus: {preview}")

    passages = pd.DataFrame(
        [
            {"docid": docid, "title": found[docid][0], "text": found[docid][1]}
            for docid in sorted(found)
        ],
        columns=["docid", "title", "text"],
    ).astype({"docid": "string", "title": "string", "text": "string"})
    validate_passages(passages)
    return passages


def stream_candidate_passages(
    candidate_docids: set[str],
    *,
    dataset_name: str,
    language: str,
    revision: str,
) -> pd.DataFrame:
    """Stream the official corpus and materialize only requested documents."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only by the Linux runner.
        raise RuntimeError("candidate passage extraction requires the datasets package") from exc

    corpus = load_dataset(
        dataset_name,
        language,
        split="train",
        revision=revision,
        streaming=True,
    )
    return extract_passages_from_rows(corpus, candidate_docids)


def read_candidate_table(path: str | Path) -> pd.DataFrame:
    """Read a candidate table from Parquet, CSV, or TSV."""

    candidate_path = Path(path)
    if not candidate_path.is_file():
        raise ValueError(f"candidate file does not exist: {candidate_path}")
    suffix = candidate_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(candidate_path)
    if suffix == ".csv":
        return pd.read_csv(candidate_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(candidate_path, sep="\t")
    raise ValueError("candidate path must end in .parquet, .pq, .csv, .tsv, or .txt")


def _fetch_text(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": "rusearchrank-phase0/0.1"})
    context = ssl.create_default_context(cafile=certifi.where() if certifi else None)
    try:
        with urlopen(request, timeout=timeout, context=context) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"failed to fetch {url}: {exc}") from exc


def _fetch_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        payload = json.loads(_fetch_text(url, timeout))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"source returned invalid JSON: {url}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"source returned an unexpected JSON shape: {url}")
    return payload


def _parse_topics(text: str) -> pd.DataFrame:
    topics = pd.read_csv(
        StringIO(text),
        sep="\t",
        names=["query_id", "query"],
        dtype="string",
        keep_default_na=False,
    )
    if topics.empty or topics[["query_id", "query"]].isna().any().any():
        raise RuntimeError("MIRACL topics are empty or malformed")
    return topics


def _parse_qrels(text: str) -> pd.DataFrame:
    qrels = pd.read_csv(
        StringIO(text),
        sep="\t",
        names=["query_id", "iteration", "docid", "relevance"],
        dtype={"query_id": "string", "iteration": "string", "docid": "string"},
        keep_default_na=False,
    )
    qrels["relevance"] = pd.to_numeric(qrels["relevance"], errors="raise").astype("Int64")
    if qrels.empty or not qrels["relevance"].isin([0, 1]).all():
        raise RuntimeError("MIRACL qrels are empty or contain non-binary relevance")
    return qrels


def load_topics(path: str | Path, *, split: str) -> pd.DataFrame:
    """Load an official two-column MIRACL topics TSV."""

    topic_path = Path(path)
    if not topic_path.is_file():
        raise ValueError(f"topics file does not exist: {topic_path}")
    topics = _parse_topics(topic_path.read_text(encoding="utf-8"))
    normalized = topics.rename(columns={"query": "query_text"})
    normalized.insert(0, "split", split)
    normalized = normalized[["split", "query_id", "query_text"]].astype("string")
    validate_queries(normalized)
    return normalized


def load_qrels(path: str | Path, *, split: str) -> pd.DataFrame:
    """Load an official four-column MIRACL qrels TSV and preserve the grade."""

    qrels_path = Path(path)
    if not qrels_path.is_file():
        raise ValueError(f"qrels file does not exist: {qrels_path}")
    qrels = _parse_qrels(qrels_path.read_text(encoding="utf-8")).rename(
        columns={"relevance": "relevance_grade"}
    )
    qrels.insert(0, "split", split)
    return qrels[["split", "query_id", "iteration", "docid", "relevance_grade"]]


def download_annotation_files(
    *,
    topics_urls: dict[str, str],
    qrels_urls: dict[str, str],
    raw_dir: str | Path,
    expected_rows: dict[str, int],
    timeout: float = 120.0,
) -> dict[str, dict[str, Path]]:
    """Download revision-pinned topics/qrels and validate their exact row counts."""

    destination = Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, Path]] = {}
    for split in ("train", "dev"):
        if split not in topics_urls or split not in qrels_urls:
            raise ValueError(f"missing official annotation URL for split {split!r}")
        topics_path = destination / f"topics.miracl-v1.0-ru-{split}.tsv"
        qrels_path = destination / f"qrels.miracl-v1.0-ru-{split}.tsv"
        for url, path in ((topics_urls[split], topics_path), (qrels_urls[split], qrels_path)):
            if not path.is_file():
                payload = _fetch_text(str(url), timeout)
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(payload, encoding="utf-8")
                temporary.replace(path)

        topics = load_topics(topics_path, split=split)
        qrels = load_qrels(qrels_path, split=split)
        expected_query_count = int(expected_rows[f"{split}_queries"])
        expected_qrels_count = int(expected_rows[f"{split}_qrels"])
        if len(topics) != expected_query_count or len(qrels) != expected_qrels_count:
            raise RuntimeError(
                f"official {split} annotations have unexpected row counts: "
                f"queries={len(topics)} (expected {expected_query_count}), "
                f"qrels={len(qrels)} (expected {expected_qrels_count})"
            )
        if qrels.duplicated(["query_id", "docid"], keep=False).any():
            raise RuntimeError(f"official {split} qrels contain duplicate pairs")
        if (qrels["relevance_grade"] < 0).any():
            raise RuntimeError(f"official {split} qrels contain negative grades")
        result[split] = {"topics": topics_path, "qrels": qrels_path}
    return result


def inspect_miracl_ru(sample_size: int = 3, timeout: float = 30.0) -> dict[str, Any]:
    """Inspect real Russian MIRACL records without downloading the corpus.

    The corpus rows come from the official Hugging Face Dataset Viewer API.
    Topics and qrels are small files and are read in full from the official
    dataset repository so their row counts and relevance values are checked.
    """

    if not isinstance(sample_size, int) or not 1 <= sample_size <= 20:
        raise ValueError("sample_size must be an integer between 1 and 20")

    corpus_meta = _fetch_json(f"{_HF_API}/{_MIRACL_CORPUS}", timeout)
    annotations_meta = _fetch_json(f"{_HF_API}/{_MIRACL_ANNOTATIONS}", timeout)
    corpus_revision = corpus_meta.get("sha")
    annotations_revision = annotations_meta.get("sha")
    if not corpus_revision or not annotations_revision:
        raise RuntimeError("Hugging Face metadata did not include resolved revisions")

    rows_url = f"{_HF_ROWS}?{urlencode({'dataset': _MIRACL_CORPUS, 'config': 'ru', 'split': 'train', 'offset': 0, 'length': sample_size})}"
    corpus_payload = _fetch_json(rows_url, timeout)
    corpus_rows = [entry.get("row", {}) for entry in corpus_payload.get("rows", [])]
    expected_corpus_fields = {"docid", "title", "text"}
    if len(corpus_rows) != sample_size or any(
        set(row) != expected_corpus_fields for row in corpus_rows
    ):
        raise RuntimeError("MIRACL corpus sample does not match docid/title/text contract")

    split_reports: dict[str, Any] = {}
    for split in ("train", "dev"):
        base = f"{_HF_RESOLVE}/{_MIRACL_ANNOTATIONS}/resolve/{annotations_revision}/miracl-v1.0-ru"
        topics_url = f"{base}/topics/topics.miracl-v1.0-ru-{split}.tsv"
        qrels_url = f"{base}/qrels/qrels.miracl-v1.0-ru-{split}.tsv"
        topics = _parse_topics(_fetch_text(topics_url, timeout))
        qrels = _parse_qrels(_fetch_text(qrels_url, timeout))
        split_reports[split] = {
            "queries": {
                "fields": {"query_id": "string", "query": "string"},
                "row_count": int(len(topics)),
                "sample": topics.head(sample_size).to_dict(orient="records"),
                "source_url": topics_url,
            },
            "qrels": {
                "fields": {
                    "query_id": "string",
                    "iteration": "string",
                    "docid": "string",
                    "relevance": "int64",
                },
                "row_count": int(len(qrels)),
                "iteration_values": sorted(qrels["iteration"].unique().tolist()),
                "relevance_values": sorted(int(value) for value in qrels["relevance"].unique()),
                "sample": qrels.head(sample_size).astype(
                    {"relevance": "int64"}
                ).to_dict(orient="records"),
                "source_url": qrels_url,
            },
        }

    features = {
        feature["name"]: feature.get("type", {}).get("dtype", "unknown")
        for feature in corpus_payload.get("features", [])
    }
    return {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "language": "ru",
        "sources": {
            "corpus_dataset": _MIRACL_CORPUS,
            "corpus_revision": corpus_revision,
            "annotations_dataset": _MIRACL_ANNOTATIONS,
            "annotations_revision": annotations_revision,
        },
        "corpus": {
            "fields": features,
            "sample_size": len(corpus_rows),
            "sample": corpus_rows,
            "sample_api_url": rows_url,
            "viewer_partial": bool(corpus_payload.get("partial", False)),
            "full_corpus_downloaded": False,
        },
        "splits": split_reports,
    }
