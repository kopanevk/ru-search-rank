"""Offline-checkable ranking metrics and Phase 1 audit summaries."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

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

    _require_columns(queries, {"query_id", "query_text"}, "queries")
    _require_columns(
        candidates,
        {"query_id", "docid", "judgment", "bm25_rank"},
        "candidates",
    )
    labels = _normalise_qrels(qrels)
    normalized_queries = queries[["query_id", "query_text"]].copy()
    normalized_queries[["query_id", "query_text"]] = normalized_queries[
        ["query_id", "query_text"]
    ].astype("string")
    if normalized_queries["query_id"].duplicated(keep=False).any():
        raise ValueError("queries contain duplicate query_id values within a split")
    query_ids = normalized_queries["query_id"]
    unknown_qrels = sorted(set(labels["query_id"]).difference(query_ids))
    if unknown_qrels:
        raise ValueError(f"qrels contain unknown query_id values: {unknown_qrels[:10]}")
    unknown_candidates = sorted(
        set(candidates["query_id"].astype("string")).difference(query_ids)
    )
    if unknown_candidates:
        raise ValueError(
            f"candidates contain unknown query_id values: {unknown_candidates[:10]}"
        )
    counts = labels.assign(
        positive=labels["relevance_grade"].gt(0),
        negative=labels["relevance_grade"].eq(0),
    ).groupby("query_id", sort=False)[["positive", "negative"]].sum()
    counts = counts.reindex(query_ids, fill_value=0)
    normalized_candidates = candidates[["query_id", "docid", "judgment", "bm25_rank"]].copy()
    normalized_candidates[["query_id", "docid", "judgment"]] = normalized_candidates[
        ["query_id", "docid", "judgment"]
    ].astype("string")
    if normalized_candidates.duplicated(["query_id", "docid"], keep=False).any():
        raise ValueError("candidates contain duplicate (query_id, docid) pairs")
    semantic_check = normalized_candidates.merge(
        labels,
        on=["query_id", "docid"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    expected_judgment = pd.Series(
        "unjudged", index=semantic_check.index, dtype="string"
    )
    expected_judgment.loc[semantic_check["relevance_grade"].gt(0)] = "relevant"
    expected_judgment.loc[
        semantic_check["relevance_grade"].eq(0)
    ] = "judged_non_relevant"
    if not semantic_check["judgment"].reset_index(drop=True).equals(
        expected_judgment.reset_index(drop=True)
    ):
        bad_rows = semantic_check.loc[
            semantic_check["judgment"].ne(expected_judgment),
            ["query_id", "docid", "judgment", "relevance_grade"],
        ].head(10)
        raise ValueError(
            "candidate judgment semantics disagree with qrels: "
            + str(bad_rows.to_dict(orient="records"))
        )
    candidate_queries = set(normalized_candidates["query_id"])
    retrieved_positive_pairs = candidates.loc[
        candidates["judgment"].astype("string").eq("relevant"), ["query_id", "docid"]
    ].drop_duplicates()
    top10 = normalized_candidates.loc[
        pd.to_numeric(normalized_candidates["bm25_rank"]).le(10)
    ].copy()
    top10["judged"] = top10["judgment"].ne("unjudged")
    top10_per_query = top10.groupby("query_id", sort=False).agg(
        retrieved_at_10=("docid", "size"),
        judged_at_10=("judged", "sum"),
    )
    candidate_per_query = normalized_candidates.assign(
        judged=normalized_candidates["judgment"].ne("unjudged")
    ).groupby("query_id", sort=False).agg(
        candidate_count=("docid", "size"),
        judged_candidate_count=("judged", "sum"),
    )
    coverage = normalized_queries.merge(
        candidate_per_query,
        left_on="query_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    ).merge(
        top10_per_query,
        left_on="query_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    numeric_columns = [
        "candidate_count",
        "judged_candidate_count",
        "retrieved_at_10",
        "judged_at_10",
    ]
    coverage[numeric_columns] = coverage[numeric_columns].fillna(0).astype("int64")
    coverage["judged_fraction"] = np.where(
        coverage["candidate_count"].gt(0),
        coverage["judged_candidate_count"] / coverage["candidate_count"],
        0.0,
    )
    coverage["judged_fraction_at_10"] = np.where(
        coverage["retrieved_at_10"].gt(0),
        coverage["judged_at_10"] / coverage["retrieved_at_10"],
        0.0,
    )
    candidate_pairs = normalized_candidates[["query_id", "docid"]].drop_duplicates()
    qrels_pairs = labels[["query_id", "docid"]]
    judged_overlap = candidate_pairs.merge(
        qrels_pairs,
        on=["query_id", "docid"],
        how="inner",
        validate="one_to_one",
    )
    judgment_counts = {
        state: int(normalized_candidates["judgment"].eq(state).sum())
        for state in ("relevant", "judged_non_relevant", "unjudged")
    }
    zero_judged = coverage.loc[
        coverage["judged_candidate_count"].eq(0), ["query_id", "query_text"]
    ]
    return {
        "queries": int(len(query_ids)),
        "unique_qrels_query_ids": int(labels["query_id"].nunique()),
        "qrels": int(len(labels)),
        "unique_qrels_query_doc_pairs": int(
            len(labels[["query_id", "docid"]].drop_duplicates())
        ),
        "relevance_values": sorted(
            int(value) for value in labels["relevance_grade"].unique()
        ),
        "relevant_judgments": int(labels["relevance_grade"].gt(0).sum()),
        "judged_non_relevant_judgments": int(
            labels["relevance_grade"].eq(0).sum()
        ),
        "queries_without_known_positive": int(counts["positive"].eq(0).sum()),
        "positives_per_query": distribution_summary(counts["positive"]),
        "judged_negatives_per_query": distribution_summary(counts["negative"]),
        "fraction_queries_without_judged_negative": float(counts["negative"].eq(0).mean()),
        "candidate_rows": int(len(normalized_candidates)),
        "candidate_judgment_counts": judgment_counts,
        "candidate_qrels_overlap_pairs": int(len(judged_overlap)),
        "candidate_judgment_coverage": float(coverage["judged_fraction"].mean()),
        "mean_judged_at_10": float(coverage["judged_fraction_at_10"].mean()),
        "per_query_candidate_count": distribution_summary(coverage["candidate_count"]),
        "per_query_judged_count": distribution_summary(
            coverage["judged_candidate_count"]
        ),
        "zero_judged_query_count": int(len(zero_judged)),
        "zero_judged_queries": zero_judged.to_dict(orient="records"),
        "per_query_coverage": coverage.to_dict(orient="records"),
        "known_positives_retrieved_at_100": int(len(retrieved_positive_pairs)),
        "retrieval_failures": int(sum(query_id not in candidate_queries for query_id in query_ids)),
    }


def parse_trec_eval_per_query(output: str, metric: str) -> dict[str, float]:
    """Parse the finite per-query vector emitted by ``trec_eval -q``."""

    values: dict[str, float] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] != metric or parts[1] == "all":
            continue
        query_id = str(parts[1])
        if query_id in values:
            raise ValueError(f"duplicate {metric} value for query {query_id!r}")
        try:
            value = float(parts[2])
        except ValueError as exc:
            raise ValueError(f"invalid trec_eval per-query line: {line}") from exc
        if not math.isfinite(value):
            raise ValueError(f"non-finite trec_eval value for query {query_id!r}")
        values[query_id] = value
    if not values:
        raise ValueError(f"trec_eval output contains no per-query {metric!r} values")
    return values


def classify_bm25_tie_break_audit(
    *, docno_desc_max_abs_difference: float, docno_asc_max_abs_difference: float,
    tolerance: float
) -> dict[str, Any]:
    """Describe what per-query metric vectors can and cannot infer about ties."""

    if tolerance < 0 or not all(
        math.isfinite(value)
        for value in (
            docno_desc_max_abs_difference,
            docno_asc_max_abs_difference,
            tolerance,
        )
    ):
        raise ValueError("tie-break audit requires finite non-negative tolerance")
    desc_match = docno_desc_max_abs_difference <= tolerance
    asc_match = docno_asc_max_abs_difference <= tolerance
    if asc_match and not desc_match:
        conclusion = "docno_asc"
        explanation = "Only docno ASC reproduces the trec_eval per-query vector."
    elif desc_match and not asc_match:
        conclusion = "docno_desc"
        explanation = "Only docno DESC reproduces the trec_eval per-query vector."
    elif asc_match and desc_match:
        conclusion = "inconclusive_metric_equivalent"
        explanation = (
            "Both docno policies are metric-equivalent within tolerance; the "
            "available metric cannot empirically identify trec_eval's actual "
            "tie policy."
        )
    else:
        conclusion = "no_policy_matches"
        explanation = (
            "Neither docno ASC nor docno DESC reproduces the trec_eval per-query "
            "vector within tolerance."
        )
    return {
        "empirical_method": "compare full per-query nDCG@10 vectors",
        "docno_desc_max_abs_difference": float(
            docno_desc_max_abs_difference
        ),
        "docno_asc_max_abs_difference": float(docno_asc_max_abs_difference),
        "tolerance": float(tolerance),
        "docno_desc_matches": desc_match,
        "docno_asc_matches": asc_match,
        "conclusion": conclusion,
        "explanation": explanation,
    }


def raw_score_tie_statistics(scores: pd.DataFrame) -> dict[str, Any]:
    """Count exact stored-float32 ties under the production ranking rule."""

    _require_columns(scores, {"query_id", "docid", "score"}, "raw scores")
    frame = scores[["query_id", "docid", "score"]].copy()
    if frame.isna().any().any():
        raise ValueError("raw scores contain nulls")
    frame[["query_id", "docid"]] = frame[["query_id", "docid"]].astype(
        "string"
    )
    if frame.duplicated(["query_id", "docid"], keep=False).any():
        raise ValueError("raw scores contain duplicate candidate keys")
    numeric = pd.to_numeric(frame["score"], errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("raw scores contain NaN or infinite values")
    stored = numeric.astype("float32")
    if not np.array_equal(
        numeric.to_numpy(dtype=np.float64), stored.to_numpy(dtype=np.float64)
    ):
        raise ValueError("raw scores are not exact stored float32 values")
    frame["score"] = stored
    ordered = frame.sort_values(
        ["query_id", "score", "docid"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ordered["rank"] = (
        ordered.groupby("query_id", sort=False).cumcount().add(1).astype("int64")
    )
    tie_groups = 0
    rows_in_ties = 0
    queries_with_any: set[str] = set()
    queries_with_top10: set[str] = set()
    crossing_rank10 = 0
    largest = 0
    for (query_id, _), group in ordered.groupby(
        ["query_id", "score"], sort=False, dropna=False
    ):
        size = int(len(group))
        if size < 2:
            continue
        ranks = group["rank"].to_numpy(dtype=np.int64)
        tie_groups += 1
        rows_in_ties += size
        largest = max(largest, size)
        queries_with_any.add(str(query_id))
        if int((ranks <= 10).sum()) >= 2:
            queries_with_top10.add(str(query_id))
        if int(ranks.min()) <= 10 < int(ranks.max()):
            crossing_rank10 += 1
    return {
        "raw_score_tie_definition": (
            "exact_float32_equality_within_query_id_on_stored_raw_logits"
        ),
        "raw_score_tie_groups": tie_groups,
        "rows_in_raw_score_ties": rows_in_ties,
        "queries_with_any_raw_score_tie": len(queries_with_any),
        "queries_with_top10_raw_score_tie": len(queries_with_top10),
        "ties_crossing_rank10_boundary": crossing_rank10,
        "largest_raw_score_tie_group": largest,
    }


def assert_candidate_set_invariant(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> dict[str, Any]:
    """Prove Recall@100's load-bearing invariant by exact key-set equality."""

    for label, frame in (("left", left), ("right", right)):
        _require_columns(frame, {"query_id", "docid"}, label)
        if frame[["query_id", "docid"]].isna().any().any():
            raise ValueError(f"{label} candidate keys contain nulls")
        if frame[["query_id", "docid"]].astype("string").duplicated().any():
            raise ValueError(f"{label} candidate keys contain duplicates")
    left_keys = set(
        map(tuple, left[["query_id", "docid"]].astype("string").to_numpy())
    )
    right_keys = set(
        map(tuple, right[["query_id", "docid"]].astype("string").to_numpy())
    )
    if left_keys != right_keys:
        missing = sorted(left_keys.difference(right_keys))[:10]
        extra = sorted(right_keys.difference(left_keys))[:10]
        raise ValueError(
            f"Recall@100 candidate-set invariant failed: missing={missing}, extra={extra}"
        )
    return {
        "candidate_set_invariant": True,
        "pair_count": len(left_keys),
        "query_count": int(left["query_id"].astype("string").nunique()),
    }


def paired_bootstrap(
    deltas: Iterable[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260802,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile CI for the paired mean, reproducible for a fixed seed."""

    values = np.asarray(list(deltas), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires a non-empty finite vector")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap resamples/confidence are invalid")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample_indices = rng.integers(0, values.size, size=values.size)
        means[index] = float(values[sample_indices].mean())
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "mean_delta": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def paired_ranking_comparison(
    baseline: Mapping[str, float],
    system: Mapping[str, float],
    *,
    resamples: int = 10_000,
    seed: int = 20260802,
    confidence: float = 0.95,
    tie_tolerance: float = 1e-9,
) -> dict[str, Any]:
    if set(baseline) != set(system):
        missing = sorted(set(baseline).difference(system))[:10]
        extra = sorted(set(system).difference(baseline))[:10]
        raise ValueError(
            f"paired metric query universes differ: missing={missing}, extra={extra}"
        )
    query_ids = sorted(baseline)
    deltas = np.asarray(
        [float(system[qid]) - float(baseline[qid]) for qid in query_ids],
        dtype=np.float64,
    )
    if not np.isfinite(deltas).all():
        raise ValueError("paired deltas contain non-finite values")
    ties = np.abs(deltas) < tie_tolerance
    return {
        "query_count": len(query_ids),
        "improved": int((deltas >= tie_tolerance).sum()),
        "degraded": int((deltas <= -tie_tolerance).sum()),
        "tie": int(ties.sum()),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "min_delta": float(deltas.min()),
        "max_delta": float(deltas.max()),
        "tie_tolerance": float(tie_tolerance),
        "per_query": [
            {
                "query_id": query_id,
                "baseline_ndcg_at_10": float(baseline[query_id]),
                "system_ndcg_at_10": float(system[query_id]),
                "delta": float(delta),
            }
            for query_id, delta in zip(query_ids, deltas, strict=True)
        ],
        "paired_bootstrap": paired_bootstrap(
            deltas,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
        ),
    }


def _normalise_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    _require_columns(ranking, {"query_id", "docid"}, "ranking")
    rank_column = "rank" if "rank" in ranking.columns else "bm25_rank"
    _require_columns(ranking, {rank_column}, "ranking")
    result = ranking[["query_id", "docid", rank_column]].copy()
    result.columns = ["query_id", "docid", "bm25_rank"]
    result[["query_id", "docid"]] = result[["query_id", "docid"]].astype("string")
    ranks = pd.to_numeric(result["bm25_rank"], errors="coerce")
    invalid = ranks.isna() | ~np.isfinite(ranks) | ranks.le(0) | ranks.mod(1).ne(0)
    if invalid.any():
        raise ValueError("ranking contains invalid ranks")
    result["bm25_rank"] = ranks.astype("int64")
    if result.duplicated(["query_id", "docid"], keep=False).any():
        raise ValueError("ranking contains duplicate query-doc pairs")
    if result.duplicated(["query_id", "bm25_rank"], keep=False).any():
        raise ValueError("ranking contains duplicate ranks within a query")
    return result


def _candidate_states(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"query_id", "docid", "judgment", "relevance_grade"}
    _require_columns(candidates, required, "candidate judgments")
    states = candidates[list(required)].copy()
    states[["query_id", "docid", "judgment"]] = states[
        ["query_id", "docid", "judgment"]
    ].astype("string")
    if states.duplicated(["query_id", "docid"], keep=False).any():
        raise ValueError("candidate judgments contain duplicate keys")
    grades = pd.to_numeric(states["relevance_grade"], errors="coerce")
    states["is_judged"] = grades.notna()
    states["is_relevant"] = grades.fillna(0).gt(0) & states["is_judged"]
    expected_unjudged = states["judgment"].eq("unjudged")
    if not np.array_equal(
        expected_unjudged.fillna(False).to_numpy(dtype=bool),
        (~states["is_judged"]).to_numpy(dtype=bool),
    ):
        raise ValueError("unjudged candidates do not preserve null relevance grades")
    return states


def _oracle_ranking(candidates: pd.DataFrame) -> pd.DataFrame:
    states = _candidate_states(candidates)
    bm25_column = "bm25_rank" if "bm25_rank" in candidates.columns else None
    if bm25_column is None:
        states["source_rank"] = states.groupby("query_id", sort=False).cumcount() + 1
    else:
        states["source_rank"] = pd.to_numeric(
            candidates[bm25_column], errors="raise"
        ).astype("int64")
    states = states.sort_values(
        ["query_id", "is_relevant", "source_rank", "docid"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    states["bm25_rank"] = states.groupby("query_id", sort=False).cumcount() + 1
    return states[["query_id", "docid", "bm25_rank"]]


def _oracle_stratum_memberships(
    *,
    candidates: pd.DataFrame,
    baseline_per_query: Mapping[str, float],
    oracle_per_query: Mapping[str, float],
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    if set(baseline_per_query) != set(oracle_per_query):
        raise ValueError("oracle stratification query universes differ")
    states = _candidate_states(candidates)
    relevant_by_query = states.groupby("query_id", sort=False)["is_relevant"].any()
    memberships: dict[str, set[str]] = {
        "no_relevant_in_candidates": set(),
        "already_at_oracle": set(),
        "improvable": set(),
    }
    for query_id in sorted(baseline_per_query):
        baseline = float(baseline_per_query[query_id])
        oracle = float(oracle_per_query[query_id])
        if not math.isfinite(baseline) or not math.isfinite(oracle):
            raise ValueError(f"non-finite oracle stratum metric for {query_id!r}")
        if not bool(relevant_by_query.get(query_id, False)):
            label = "no_relevant_in_candidates"
        elif math.isclose(
            baseline,
            oracle,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            label = "already_at_oracle"
        elif baseline < oracle:
            label = "improvable"
        else:
            raise ValueError(
                f"BM25 nDCG@10 exceeds candidate oracle for {query_id!r}: "
                f"baseline={baseline}, oracle={oracle}"
            )
        memberships[label].add(str(query_id))
    sets = list(memberships.values())
    union = set().union(*sets)
    pairwise_disjoint = all(
        sets[left].isdisjoint(sets[right])
        for left in range(len(sets))
        for right in range(left + 1, len(sets))
    )
    expected = {str(query_id) for query_id in baseline_per_query}
    exactly_once = pairwise_disjoint and union == expected
    query_count_sum = sum(len(values) for values in sets)
    if not exactly_once or query_count_sum != len(expected):
        raise RuntimeError("oracle strata are not an exact query partition")
    at_oracle = (
        len(memberships["no_relevant_in_candidates"])
        + len(memberships["already_at_oracle"])
    )
    invariants = {
        "strata_pairwise_disjoint": pairwise_disjoint,
        "strata_cover_all_queries": union == expected,
        "each_query_in_exactly_one_stratum": exactly_once,
        "stratum_query_count_sum": query_count_sum,
        "query_count": len(expected),
        "queries_at_oracle_under_bm25": at_oracle,
        "queries_at_oracle_consistency": (
            at_oracle
            == len(memberships["no_relevant_in_candidates"])
            + len(memberships["already_at_oracle"])
        ),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }
    return memberships, invariants


def sparse_judgment_diagnostics(
    *,
    candidates: pd.DataFrame,
    qrels: pd.DataFrame,
    ranking: pd.DataFrame,
    bm25_ranking: pd.DataFrame | None = None,
    query_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute standard/condensed and unjudged-vs-relevant diagnostics."""

    system = _normalise_ranking(ranking)
    states = _candidate_states(candidates)
    candidate_keys = set(
        map(tuple, states[["query_id", "docid"]].astype("string").to_numpy())
    )
    system_keys = set(
        map(tuple, system[["query_id", "docid"]].astype("string").to_numpy())
    )
    if system_keys != candidate_keys:
        raise ValueError("diagnostic ranking does not exactly cover candidate keys")
    labels = _normalise_qrels(qrels)
    universe = (
        list(dict.fromkeys(str(value) for value in query_ids))
        if query_ids is not None
        else list(dict.fromkeys(labels["query_id"].astype("string").tolist()))
    )
    standard = evaluate_bm25_metrics(
        system,
        labels,
        query_ids=universe,
        ndcg_cutoff=10,
        recall_cutoff=100,
    )
    metric_by_query = {
        str(row["query_id"]): row for row in standard["per_query"]
    }
    joined = system.merge(
        states[["query_id", "docid", "is_judged", "is_relevant"]],
        on=["query_id", "docid"],
        how="inner",
        validate="one_to_one",
    )
    ranking_groups = {
        str(query_id): group.sort_values("bm25_rank", kind="mergesort")
        for query_id, group in joined.groupby("query_id", sort=False)
    }
    candidate_groups = {
        str(query_id): group
        for query_id, group in states.groupby("query_id", sort=False)
    }
    queries_without_judged = 0
    queries_without_relevant = 0
    unjudged_counts: list[int] = []
    inversion_queries = 0
    inversion_pairs = 0
    eligible: list[str] = []
    for query_id in universe:
        candidate_group = candidate_groups.get(query_id)
        if candidate_group is None or not bool(candidate_group["is_judged"].any()):
            queries_without_judged += 1
        else:
            eligible.append(query_id)
        if candidate_group is None or not bool(candidate_group["is_relevant"].any()):
            queries_without_relevant += 1
        group = ranking_groups.get(query_id)
        if group is None:
            unjudged_counts.append(10)
            continue
        top10 = group.loc[group["bm25_rank"].le(10)]
        unjudged_top = top10.loc[~top10["is_judged"]]
        unjudged_counts.append(int(len(unjudged_top)))
        relevant_ranks = group.loc[group["is_relevant"], "bm25_rank"].astype("int64")
        query_inversions = sum(
            int((relevant_ranks > int(unjudged_rank)).sum())
            for unjudged_rank in unjudged_top["bm25_rank"]
        )
        inversion_pairs += query_inversions
        inversion_queries += int(query_inversions > 0)

    oracle = _oracle_ranking(candidates)
    oracle_metrics = evaluate_bm25_metrics(
        oracle,
        labels,
        query_ids=universe,
        ndcg_cutoff=10,
        recall_cutoff=100,
    )
    oracle_by_query = {
        str(row["query_id"]): float(row["ndcg_at_10"])
        for row in oracle_metrics["per_query"]
    }
    at_oracle = None
    bm25_oracle_strata = None
    if bm25_ranking is not None:
        baseline = evaluate_bm25_metrics(
            _normalise_ranking(bm25_ranking),
            labels,
            query_ids=universe,
            ndcg_cutoff=10,
            recall_cutoff=100,
        )
        baseline_by_query = {
            str(row["query_id"]): float(row["ndcg_at_10"])
            for row in baseline["per_query"]
        }
        memberships, oracle_invariants = _oracle_stratum_memberships(
            candidates=candidates,
            baseline_per_query=baseline_by_query,
            oracle_per_query=oracle_by_query,
        )
        at_oracle = int(oracle_invariants["queries_at_oracle_under_bm25"])
        bm25_oracle_strata = {
            label: {
                "query_count": len(query_ids_for_label),
                "query_ids": sorted(query_ids_for_label),
            }
            for label, query_ids_for_label in memberships.items()
        }
        bm25_oracle_strata["invariants"] = oracle_invariants
    subset_rows = [metric_by_query[query_id] for query_id in eligible]
    subset = {
        "query_count": len(eligible),
        "ndcg_at_10": float(np.mean([row["ndcg_at_10"] for row in subset_rows]))
        if subset_rows
        else 0.0,
        "condensed_ndcg_at_10": float(
            np.mean([row["condensed_ndcg_at_10"] for row in subset_rows])
        )
        if subset_rows
        else 0.0,
        "judged_at_10": float(np.mean([row["judged_at_10"] for row in subset_rows]))
        if subset_rows
        else 0.0,
    }
    return {
        "diagnostic": True,
        "judged_at_10": float(standard["aggregate"]["judged_at_10"]),
        "mean_unjudged_in_top10": float(np.mean(unjudged_counts)),
        "condensed_ndcg_at_10": float(
            standard["aggregate"]["condensed_ndcg_at_10"]
        ),
        "queries_without_judged_candidate": int(queries_without_judged),
        "queries_without_relevant_candidate": int(queries_without_relevant),
        "queries_at_oracle_under_bm25": (
            int(at_oracle) if at_oracle is not None else None
        ),
        "bm25_oracle_strata": bm25_oracle_strata,
        "oracle_ndcg_at_10_over_candidates": float(
            oracle_metrics["aggregate"]["ndcg_at_10"]
        ),
        "queries_with_unjudged_above_relevant": int(inversion_queries),
        "pairwise_unjudged_relevant_inversions": int(inversion_pairs),
        "excluding_queries_without_judged_candidate": subset,
        "per_query_metrics": standard["per_query"],
        "oracle_per_query": oracle_by_query,
    }


def stratified_delta_summary(
    *,
    candidates: pd.DataFrame,
    baseline_per_query: Mapping[str, float],
    system_per_query: Mapping[str, float],
    oracle_per_query: Mapping[str, float],
) -> dict[str, Any]:
    if set(baseline_per_query) != set(system_per_query) or set(baseline_per_query) != set(
        oracle_per_query
    ):
        raise ValueError("stratification query universes differ")
    memberships, invariants = _oracle_stratum_memberships(
        candidates=candidates,
        baseline_per_query=baseline_per_query,
        oracle_per_query=oracle_per_query,
    )
    result: dict[str, Any] = {}
    for label, query_ids in memberships.items():
        deltas = [
            float(system_per_query[query_id])
            - float(baseline_per_query[query_id])
            for query_id in sorted(query_ids)
        ]
        result[label] = {
            "query_count": len(query_ids),
            "query_ids": sorted(query_ids),
            "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        }
    result["invariants"] = invariants
    return result


def rank_candidates_by_score(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    score_column: str = "score",
) -> pd.DataFrame:
    """Join exact candidate keys and rank float scores with the frozen tie-break."""

    _require_columns(candidates, {"query_id", "docid"}, "candidates")
    _require_columns(scores, {"query_id", "docid", score_column}, "scores")
    base = candidates[["query_id", "docid"]].copy()
    values = scores[["query_id", "docid", score_column]].copy()
    for label, frame in (("candidates", base), ("scores", values)):
        frame[["query_id", "docid"]] = frame[["query_id", "docid"]].astype("string")
        if frame.duplicated(["query_id", "docid"], keep=False).any():
            raise ValueError(f"{label} contain duplicate query-document keys")
    left = set(map(tuple, base[["query_id", "docid"]].to_numpy()))
    right = set(map(tuple, values[["query_id", "docid"]].to_numpy()))
    if left != right:
        raise ValueError("score keys do not exactly match candidate keys")
    numeric = pd.to_numeric(values[score_column], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError("ranking scores must be finite")
    values[score_column] = numeric.astype("float32")
    ranked = base.merge(
        values,
        on=["query_id", "docid"],
        how="inner",
        validate="one_to_one",
        sort=False,
    ).sort_values(
        ["query_id", score_column, "docid"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked["rank"] = (
        ranked.groupby("query_id", sort=False).cumcount().add(1).astype("int64")
    )
    return ranked[["query_id", "docid", "rank", score_column]].reset_index(drop=True)


def evaluate_ranked_ndcg_at_10(
    ranking: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    query_ids: Iterable[str],
) -> dict[str, Any]:
    """Evaluate full ranked groups with qrels-derived ideal DCG at cutoff 10."""

    _require_columns(ranking, {"query_id", "docid"}, "ranking")
    rank_column = "rank" if "rank" in ranking.columns else "bm25_rank"
    _require_columns(ranking, {rank_column}, "ranking")
    normalized = ranking[["query_id", "docid", rank_column]].copy()
    normalized.columns = ["query_id", "docid", "bm25_rank"]
    report = evaluate_bm25_metrics(
        normalized,
        qrels,
        query_ids=query_ids,
        ndcg_cutoff=10,
        recall_cutoff=100,
    )
    return {
        "query_count": int(report["query_count"]),
        "ndcg_at_10": float(report["aggregate"]["ndcg_at_10"]),
        "per_query": {
            str(row["query_id"]): float(row["ndcg_at_10"])
            for row in report["per_query"]
        },
    }


def mean_reciprocal_rank_at_10(
    ranking: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    query_ids: Iterable[str],
) -> dict[str, Any]:
    """Compute diagnostic MRR@10 over an explicit query universe."""

    normalized_qrels = _normalise_qrels(qrels)
    relevant = set(
        map(
            tuple,
            normalized_qrels.loc[
                normalized_qrels["relevance_grade"].gt(0), ["query_id", "docid"]
            ].astype("string").to_numpy(),
        )
    )
    normalized = _normalise_ranking(ranking)
    groups = {
        str(query_id): group.sort_values("bm25_rank", kind="mergesort")
        for query_id, group in normalized.groupby("query_id", sort=False)
    }
    per_query: dict[str, float] = {}
    for query_id in dict.fromkeys(str(value) for value in query_ids):
        reciprocal = 0.0
        group = groups.get(query_id)
        if group is not None:
            for row in group.loc[group["bm25_rank"].le(10)].itertuples(index=False):
                if (query_id, str(row.docid)) in relevant:
                    reciprocal = 1.0 / int(row.bm25_rank)
                    break
        per_query[query_id] = reciprocal
    if not per_query:
        raise ValueError("MRR query universe must not be empty")
    return {
        "mrr_at_10": float(np.mean(list(per_query.values()))),
        "per_query": per_query,
    }
