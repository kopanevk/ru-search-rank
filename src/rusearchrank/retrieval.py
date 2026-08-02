"""Deterministic BM25 retrieval and run transformations for Phase 1A."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import attach_qrels


_RUN_COLUMNS = {"split", "query_id", "docid", "bm25_score"}


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"BM25 run is missing required columns: {', '.join(missing)}")


def check_duplicate_documents(run: pd.DataFrame) -> None:
    """Reject duplicate documents within one query."""

    _require_columns(run, {"split", "query_id", "docid"})
    if run[["split", "query_id", "docid"]].isna().any().any():
        raise ValueError("BM25 run identifiers must not be null")
    keys = run[["split", "query_id", "docid"]].astype("string")
    if keys.duplicated(["split", "query_id", "docid"], keep=False).any():
        raise ValueError("BM25 run contains duplicate (query_id, docid) pairs")


def stable_sort_bm25(run: pd.DataFrame) -> pd.DataFrame:
    """Sort each query by score descending and docid ascending."""

    _require_columns(run, _RUN_COLUMNS)
    check_duplicate_documents(run)
    sorted_run = run.drop(columns=["bm25_rank"], errors="ignore").copy()
    sorted_run["split"] = sorted_run["split"].astype("string")
    sorted_run["query_id"] = sorted_run["query_id"].astype("string")
    sorted_run["docid"] = sorted_run["docid"].astype("string")
    scores = pd.to_numeric(sorted_run["bm25_score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores).all():
        raise ValueError("bm25_score must contain finite numeric values")
    sorted_run["bm25_score"] = scores.astype("float64")

    query_keys = pd.MultiIndex.from_frame(sorted_run[["split", "query_id"]])
    sorted_run["_query_order"] = pd.factorize(query_keys, sort=False)[0]
    sorted_run = sorted_run.sort_values(
        ["_query_order", "bm25_score", "docid"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop(columns="_query_order")
    sorted_run["bm25_rank"] = (
        sorted_run.groupby(["split", "query_id"], sort=False)
        .cumcount()
        .add(1)
        .astype("int64")
    )

    core = ["split", "query_id", "docid", "bm25_rank", "bm25_score"]
    extra = [column for column in sorted_run.columns if column not in core]
    return sorted_run.loc[:, core + extra].reset_index(drop=True)


def validate_top_k(run: pd.DataFrame, top_k: int, *, require_exact: bool = True) -> None:
    """Validate duplicate-free, contiguous ranks bounded by ``top_k``."""

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    _require_columns(run, {"split", "query_id", "docid", "bm25_rank"})
    check_duplicate_documents(run)

    ranks = pd.to_numeric(run["bm25_rank"], errors="coerce")
    invalid = ranks.isna() | ~np.isfinite(ranks) | ranks.le(0) | ranks.mod(1).ne(0)
    if invalid.any():
        raise ValueError("bm25_rank must contain positive integers")

    keys = pd.DataFrame(
        {
            "split": run["split"].astype("string"),
            "query_id": run["query_id"].astype("string"),
            "bm25_rank": ranks.astype("int64"),
        }
    )
    if keys.duplicated(["split", "query_id", "bm25_rank"], keep=False).any():
        raise ValueError("duplicate bm25_rank values within a query are not allowed")

    for (split, query_id), group in keys.groupby(["split", "query_id"], sort=False):
        observed = sorted(group["bm25_rank"].tolist())
        expected = list(range(1, len(observed) + 1))
        if observed != expected:
            raise ValueError(
                f"query {(split, query_id)!r} has non-contiguous BM25 ranks"
            )
        if len(observed) > top_k:
            raise ValueError(
                f"query {(split, query_id)!r} has more than top_k={top_k} rows"
            )
        if require_exact and len(observed) != top_k:
            raise ValueError(
                f"query {(split, query_id)!r} does not contain exactly top_k={top_k} rows"
            )


def normalize_bm25_run(run: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    """Return a deterministic BM25 run and optionally retain only top-k rows."""

    normalized = stable_sort_bm25(run)
    if top_k is not None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        normalized = normalized[normalized["bm25_rank"].le(top_k)].reset_index(drop=True)
        validate_top_k(normalized, top_k, require_exact=False)
    return normalized


def join_qrels(run: pd.DataFrame, qrels: pd.DataFrame) -> pd.DataFrame:
    """Join qrels onto a normalized run using the three-state judgment contract."""

    _require_columns(
        run, {"split", "query_id", "docid", "bm25_rank", "bm25_score"}
    )
    return attach_qrels(run, qrels)


def validate_query_coverage(run: pd.DataFrame, expected_queries: pd.DataFrame) -> None:
    """Require every source query to have a complete retrieval group."""

    _require_columns(run, {"split", "query_id"})
    missing = sorted({"split", "query_id"}.difference(expected_queries.columns))
    if missing:
        raise ValueError(f"expected queries are missing columns: {', '.join(missing)}")
    observed = set(
        map(tuple, run[["split", "query_id"]].astype("string").drop_duplicates().to_numpy())
    )
    expected = set(
        map(
            tuple,
            expected_queries[["split", "query_id"]]
            .astype("string")
            .drop_duplicates()
            .to_numpy(),
        )
    )
    missing_queries = sorted(expected.difference(observed))
    extra_queries = sorted(observed.difference(expected))
    if missing_queries or extra_queries:
        raise ValueError(
            "BM25 query coverage mismatch: "
            f"missing={missing_queries[:10]}, extra={extra_queries[:10]}"
        )


def build_retrieval_depth_audit(
    run: pd.DataFrame,
    topics: pd.DataFrame,
    *,
    requested_top_k: int,
) -> dict[str, object]:
    """Summarize variable retrieval depth while preserving full topic coverage."""

    validate_top_k(run, requested_top_k, require_exact=False)
    required_topics = {"split", "query_id", "query_text"}
    missing_columns = sorted(required_topics.difference(topics.columns))
    if missing_columns:
        raise ValueError(
            "topics are missing required columns: " + ", ".join(missing_columns)
        )
    normalized_topics = topics[["split", "query_id", "query_text"]].copy()
    for column in ("split", "query_id", "query_text"):
        normalized_topics[column] = normalized_topics[column].astype("string")
    if normalized_topics[["split", "query_id"]].duplicated(keep=False).any():
        raise ValueError("topics contain duplicate (split, query_id) pairs")

    expected_keys = set(
        map(tuple, normalized_topics[["split", "query_id"]].to_numpy())
    )
    observed_keys = set(map(tuple, run[["split", "query_id"]].astype("string").to_numpy()))
    unknown = sorted(observed_keys.difference(expected_keys))
    if unknown:
        raise ValueError(f"BM25 run contains unknown query_id values: {unknown[:10]}")

    counts = (
        run.assign(
            split=run["split"].astype("string"),
            query_id=run["query_id"].astype("string"),
        )
        .groupby(["split", "query_id"], sort=False)
        .size()
        .rename("candidate_count")
        .reset_index()
    )
    coverage = normalized_topics.merge(
        counts,
        on=["split", "query_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    coverage["candidate_count"] = (
        coverage["candidate_count"].fillna(0).astype("int64")
    )
    coverage["shortfall"] = requested_top_k - coverage["candidate_count"]
    positive_short = coverage["candidate_count"].between(1, requested_top_k - 1)
    zero_hit = coverage["candidate_count"].eq(0)

    def rows_for(mask: pd.Series) -> list[dict[str, object]]:
        return [
            {
                "query_id": str(row.query_id),
                "query_text": str(row.query_text),
                "candidate_count": int(row.candidate_count),
                "shortfall": int(row.shortfall),
            }
            for row in coverage.loc[mask].itertuples(index=False)
        ]

    candidate_counts = coverage["candidate_count"]
    return {
        "requested_top_k": requested_top_k,
        "query_count": int(len(coverage)),
        "queries_in_run": int(len(counts)),
        "full_depth_query_count": int(candidate_counts.eq(requested_top_k).sum()),
        "short_query_count": int(positive_short.sum()),
        "zero_hit_query_count": int(zero_hit.sum()),
        "minimum_candidate_count": int(candidate_counts.min()),
        "maximum_candidate_count": int(candidate_counts.max()),
        "total_candidate_rows": int(candidate_counts.sum()),
        "total_shortfall": int(coverage["shortfall"].sum()),
        "short_queries": rows_for(positive_short),
        "zero_hit_queries": rows_for(zero_hit),
    }


def validate_stable_order(run: pd.DataFrame) -> None:
    """Require rows and ranks to equal the deterministic score/docid ordering."""

    _require_columns(run, _RUN_COLUMNS | {"bm25_rank"})
    expected = stable_sort_bm25(run)
    columns = ["split", "query_id", "docid", "bm25_rank", "bm25_score"]
    actual_core = run.loc[:, columns].reset_index(drop=True).copy()
    for column in ("split", "query_id", "docid"):
        actual_core[column] = actual_core[column].astype("string")
    actual_core["bm25_rank"] = pd.to_numeric(
        actual_core["bm25_rank"], errors="coerce"
    ).astype("int64")
    actual_core["bm25_score"] = pd.to_numeric(
        actual_core["bm25_score"], errors="coerce"
    ).astype("float64")
    expected_core = expected.loc[:, columns]
    if not actual_core.equals(expected_core):
        raise ValueError("BM25 rows are not in deterministic score-desc/docid-asc order")


def read_trec_run(path: str, *, split: str) -> pd.DataFrame:
    """Read a six-column TREC run into the normalized BM25 run schema."""

    columns = ["query_id", "q0", "docid", "source_rank", "bm25_score", "tag"]
    run = pd.read_csv(
        path,
        sep=r"\s+",
        names=columns,
        dtype={"query_id": "string", "q0": "string", "docid": "string", "tag": "string"},
        keep_default_na=False,
    )
    if run.empty:
        raise ValueError(f"TREC run is empty: {path}")
    run.insert(0, "split", split)
    return run[["split", "query_id", "docid", "source_rank", "bm25_score", "tag"]]


def write_trec_run(run: pd.DataFrame, path: str, *, tag: str = "rusearchrank-bm25") -> None:
    """Write a deterministic normalized run in standard six-column TREC format."""

    _require_columns(run, _RUN_COLUMNS | {"bm25_rank"})
    validate_stable_order(run)
    output = pd.DataFrame(
        {
            "query_id": run["query_id"].astype("string"),
            "q0": "Q0",
            "docid": run["docid"].astype("string"),
            "rank": run["bm25_rank"].astype("int64"),
            "score": run["bm25_score"].astype("float64"),
            "tag": tag,
        }
    )
    output.to_csv(path, sep=" ", header=False, index=False, float_format="%.8f")


def run_pyserini_bm25(
    queries: pd.DataFrame,
    *,
    index_name: str,
    language: str,
    top_k: int,
    threads: int,
    k1: float = 0.9,
    b: float = 0.4,
) -> pd.DataFrame:
    """Run full BM25 in Linux; Pyserini is imported lazily by design."""

    required = {"split", "query_id", "query_text"}
    missing = sorted(required.difference(queries.columns))
    if missing:
        raise ValueError(f"queries are missing columns: {', '.join(missing)}")
    if queries.empty or queries["split"].nunique() != 1:
        raise ValueError("run_pyserini_bm25 expects exactly one non-empty split")
    if queries[["split", "query_id", "query_text"]].isna().any().any():
        raise ValueError("queries contain null values")
    if queries["query_id"].astype("string").duplicated().any():
        raise ValueError("queries contain duplicate query_id values")

    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as exc:  # pragma: no cover - Linux integration path only.
        raise RuntimeError("Pyserini is required for full BM25 retrieval") from exc

    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    searcher.set_language(language)
    searcher.set_bm25(k1, b)
    query_ids = queries["query_id"].astype("string").tolist()
    query_texts = queries["query_text"].astype("string").tolist()
    hits_by_query = searcher.batch_search(
        query_texts,
        query_ids,
        k=top_k,
        threads=threads,
    )

    split = str(queries["split"].iloc[0])
    rows: list[dict[str, object]] = []
    for query_id in query_ids:
        for hit in hits_by_query.get(query_id, []):
            rows.append(
                {
                    "split": split,
                    "query_id": query_id,
                    "docid": str(hit.docid),
                    "bm25_score": float(hit.score),
                }
            )
    if not rows:
        raise RuntimeError("Pyserini returned no BM25 results")
    normalized = normalize_bm25_run(pd.DataFrame(rows))
    validate_top_k(normalized, top_k, require_exact=False)
    validate_query_coverage(normalized, queries)
    return normalized


def smoke_prebuilt_index(
    *, index_name: str, language: str, query_text: str
) -> dict[str, object]:
    """Open the official index and return a small validated BM25 result."""

    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as exc:  # pragma: no cover - Linux integration path only.
        raise RuntimeError("Pyserini is required for index smoke retrieval") from exc

    # Opening by the official catalog key is the definitive availability check;
    # Pyserini raises before retrieval if the key or archive cannot be resolved.
    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    searcher.set_language(language)
    searcher.set_bm25(0.9, 0.4)
    hits = searcher.search(query_text, k=3)
    if not hits:
        raise RuntimeError("prebuilt index smoke retrieval returned no results")
    first = hits[0]
    score = float(first.score)
    if not str(first.docid).strip() or not np.isfinite(score):
        raise RuntimeError("prebuilt index returned an invalid docid or score")
    return {"docid": str(first.docid), "score": score, "hits": len(hits)}
