"""Offline-checkable ranking metrics and Phase 1 audit summaries."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _normalise_qrels(qrels: pd.DataFrame) -> pd.DataFrame:
    _require_columns(qrels, {"query_id", "docid"}, "qrels")
    source = "relevance_grade" if "relevance_grade" in qrels.columns else "relevance"
    if source not in qrels.columns:
        raise ValueError("qrels is missing relevance or relevance_grade")
    normalized = qrels[["query_id", "docid", source]].copy()
    normalized.columns = ["query_id", "docid", "relevance_grade"]
    if normalized.isna().any().any():
        raise ValueError("qrels contain null identifiers or grades")
    normalized[["query_id", "docid"]] = normalized[["query_id", "docid"]].astype(
        "string"
    )
    if normalized.duplicated(["query_id", "docid"], keep=False).any():
        raise ValueError("qrels contain duplicate (query_id, docid) pairs")
    grades = pd.to_numeric(normalized["relevance_grade"], errors="coerce")
    invalid = (
        grades.isna()
        | ~np.isfinite(grades)
        | grades.lt(0)
        | grades.mod(1).ne(0)
    )
    if invalid.any():
        raise ValueError("qrels grades must be non-negative integers")
    normalized["relevance_grade"] = grades.astype("int64")
    return normalized


def _dcg(grades: Iterable[int], cutoff: int) -> float:
    return float(
        sum(
            (2.0 ** int(grade) - 1.0) / math.log2(rank + 1.0)
            for rank, grade in enumerate(list(grades)[:cutoff], start=1)
        )
    )


def evaluate_bm25_metrics(
    candidates: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    query_ids: Iterable[str] | None = None,
    ndcg_cutoff: int = 10,
    recall_cutoff: int = 100,
) -> dict[str, object]:
    """Compute standard and condensed metrics over a fixed query universe.

    Standard nDCG keeps unjudged documents in place. Condensed nDCG first
    removes every unjudged document from the full candidate list and only then
    applies the cutoff. Both variants use the same qrels-derived ideal DCG.
    """

    if ndcg_cutoff <= 0 or recall_cutoff <= 0:
        raise ValueError("metric cutoffs must be positive")
    _require_columns(candidates, {"query_id", "docid", "bm25_rank"}, "candidates")
    ranking = candidates[["query_id", "docid", "bm25_rank"]].copy()
    if ranking[["query_id", "docid", "bm25_rank"]].isna().any().any():
        raise ValueError("candidates contain null query_id, docid, or rank")
    ranking[["query_id", "docid"]] = ranking[["query_id", "docid"]].astype("string")
    ranks = pd.to_numeric(ranking["bm25_rank"], errors="coerce")
    invalid_ranks = ranks.isna() | ~np.isfinite(ranks) | ranks.le(0) | ranks.mod(1).ne(0)
    if invalid_ranks.any():
        raise ValueError("candidate ranks must be positive integers")
    ranking["bm25_rank"] = ranks.astype("int64")
    if ranking.duplicated(["query_id", "docid"], keep=False).any():
        raise ValueError("candidates contain duplicate (query_id, docid) pairs")
    if ranking.duplicated(["query_id", "bm25_rank"], keep=False).any():
        raise ValueError("candidates contain duplicate ranks within a query")
    ranking = ranking.sort_values(["query_id", "bm25_rank"], kind="mergesort")

    labels = _normalise_qrels(qrels)
    grade_lookup = {
        (str(row.query_id), str(row.docid)): int(row.relevance_grade)
        for row in labels.itertuples(index=False)
    }
    qrels_by_query = {
        str(query_id): group["relevance_grade"].astype("int64").tolist()
        for query_id, group in labels.groupby("query_id", sort=False)
    }

    if query_ids is None:
        universe = list(dict.fromkeys(labels["query_id"].astype("string").tolist()))
    else:
        universe = list(dict.fromkeys(str(query_id) for query_id in query_ids))
    if not universe:
        raise ValueError("metric query universe must not be empty")

    ranking_groups = {
        str(query_id): group.sort_values("bm25_rank", kind="mergesort")
        for query_id, group in ranking.groupby("query_id", sort=False)
    }
    per_query: list[dict[str, object]] = []
    for query_id in universe:
        group = ranking_groups.get(query_id)
        docids = [] if group is None else group["docid"].astype("string").tolist()
        retrieved_grades = [grade_lookup.get((query_id, docid), 0) for docid in docids]
        judged = [(query_id, docid) in grade_lookup for docid in docids]

        ideal_grades = sorted(qrels_by_query.get(query_id, []), reverse=True)
        ideal_dcg = _dcg(ideal_grades, ndcg_cutoff)
        standard_dcg = _dcg(retrieved_grades, ndcg_cutoff)
        judged_grades = [
            grade for grade, is_judged in zip(retrieved_grades, judged, strict=True) if is_judged
        ]
        condensed_dcg = _dcg(judged_grades, ndcg_cutoff)
        ndcg = standard_dcg / ideal_dcg if ideal_dcg else 0.0
        condensed_ndcg = condensed_dcg / ideal_dcg if ideal_dcg else 0.0

        relevant_total = sum(grade > 0 for grade in ideal_grades)
        relevant_retrieved = sum(grade > 0 for grade in retrieved_grades[:recall_cutoff])
        recall = relevant_retrieved / relevant_total if relevant_total else 0.0
        hit = float(relevant_retrieved > 0)
        judged_at_10 = sum(judged[:ndcg_cutoff]) / ndcg_cutoff
        if condensed_ndcg + 1e-12 < ndcg:
            raise ValueError("condensed nDCG is unexpectedly below standard nDCG")
        if math.isclose(judged_at_10, 1.0) and not math.isclose(
            condensed_ndcg, ndcg, abs_tol=1e-12
        ):
            raise ValueError("fully judged top-10 changed condensed nDCG")

        per_query.append(
            {
                "query_id": query_id,
                "ndcg_at_10": ndcg,
                "recall_at_100": recall,
                "hit_at_100": hit,
                "judged_at_10": judged_at_10,
                "condensed_ndcg_at_10": condensed_ndcg,
                "retrieved": len(docids),
                "relevant_total": relevant_total,
                "relevant_retrieved_at_100": relevant_retrieved,
            }
        )

    metric_names = (
        "ndcg_at_10",
        "recall_at_100",
        "hit_at_100",
        "judged_at_10",
        "condensed_ndcg_at_10",
    )
    aggregate = {
        metric: float(np.mean([float(row[metric]) for row in per_query]))
        for metric in metric_names
    }
    return {"query_count": len(universe), "aggregate": aggregate, "per_query": per_query}


def parse_trec_eval_metric(output: str, metric: str) -> float:
    """Extract one ``all`` aggregate from official NIST trec_eval output."""

    matches: list[float] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == metric and parts[1] == "all":
            try:
                matches.append(float(parts[2]))
            except ValueError as exc:
                raise ValueError(f"invalid trec_eval value in line: {line}") from exc
    if len(matches) != 1 or not math.isfinite(matches[0]):
        raise ValueError(f"expected one finite {metric!r} aggregate from trec_eval")
    return matches[0]


def reproduction_rows(
    *,
    official: dict[str, float],
    local: dict[str, float],
    tolerances: dict[str, float],
) -> list[dict[str, object]]:
    """Build the explicit PASS/FAIL rows used in the reproduction audit."""

    rows: list[dict[str, object]] = []
    for metric, official_value in official.items():
        if metric not in local or metric not in tolerances:
            raise ValueError(f"missing local value or tolerance for {metric}")
        difference = abs(float(local[metric]) - float(official_value))
        tolerance = float(tolerances[metric])
        rows.append(
            {
                "metric": metric,
                "official": float(official_value),
                "local": float(local[metric]),
                "absolute_difference": difference,
                "tolerance": tolerance,
                "status": "PASS" if difference <= tolerance else "FAIL",
            }
        )
    return rows


def distribution_summary(values: pd.Series) -> dict[str, float | int | None]:
    """Return the min/median/p90/max shape required by the qrels audit."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"min": None, "median": None, "p90": None, "max": None}
    return {
        "min": int(numeric.min()),
        "median": float(numeric.median()),
        "p90": float(numeric.quantile(0.9)),
        "max": int(numeric.max()),
    }


def build_qrels_split_audit(
    *, queries: pd.DataFrame, qrels: pd.DataFrame, candidates: pd.DataFrame
) -> dict[str, object]:
    """Summarize one fully generated split for ``qrels_audit.json``."""

    _require_columns(queries, {"query_id"}, "queries")
    _require_columns(candidates, {"query_id", "judgment", "bm25_rank"}, "candidates")
    labels = _normalise_qrels(qrels)
    query_ids = queries["query_id"].astype("string").drop_duplicates()
    counts = labels.assign(
        positive=labels["relevance_grade"].gt(0),
        negative=labels["relevance_grade"].eq(0),
    ).groupby("query_id", sort=False)[["positive", "negative"]].sum()
    counts = counts.reindex(query_ids, fill_value=0)
    candidate_queries = set(candidates["query_id"].astype("string"))
    retrieved_positive_pairs = candidates.loc[
        candidates["judgment"].astype("string").eq("relevant"), ["query_id", "docid"]
    ].drop_duplicates()
    top10 = candidates.loc[pd.to_numeric(candidates["bm25_rank"]).le(10)]
    judged_at_10 = float(top10["judgment"].astype("string").ne("unjudged").sum()) / (
        10 * len(query_ids)
    )
    return {
        "queries": int(len(query_ids)),
        "qrels": int(len(labels)),
        "relevant_judgments": int(labels["relevance_grade"].gt(0).sum()),
        "non_relevant_judgments": int(labels["relevance_grade"].eq(0).sum()),
        "queries_without_known_positive": int(counts["positive"].eq(0).sum()),
        "positives_per_query": distribution_summary(counts["positive"]),
        "judged_negatives_per_query": distribution_summary(counts["negative"]),
        "fraction_queries_without_judged_negative": float(counts["negative"].eq(0).mean()),
        "candidate_judgment_coverage": float(
            candidates["judgment"].astype("string").ne("unjudged").mean()
        ),
        "mean_judged_at_10": judged_at_10,
        "known_positives_retrieved_at_100": int(len(retrieved_positive_pairs)),
        "retrieval_failures": int(sum(query_id not in candidate_queries for query_id in query_ids)),
    }
