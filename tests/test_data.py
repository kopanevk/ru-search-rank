from __future__ import annotations

import pandas as pd
import pytest

from rusearchrank.data import (
    attach_qrels,
    extract_passages_from_rows,
    validate_candidate_schema,
    validate_passages,
    validate_queries,
)


def valid_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["dev", "dev", "dev"],
            "query_id": ["q1", "q1", "q1"],
            "docid": ["d1", "d2", "d3"],
            "bm25_rank": [1, 2, 3],
            "bm25_score": [3.0, 2.0, 1.0],
            "relevance_grade": pd.Series([2, 0, pd.NA], dtype="Int64"),
            "judgment": ["relevant", "non_relevant", "unjudged"],
            "relevance": pd.Series([1, 0, pd.NA], dtype="Int64"),
            "is_judged": [True, True, False],
        }
    )


def test_all_three_judgment_states_are_valid() -> None:
    validate_candidate_schema(valid_candidates())


@pytest.mark.parametrize(
    ("grade", "judgment", "relevance", "is_judged"),
    [
        (0, "relevant", 1, True),
        (1, "relevant", 0, True),
        (1, "relevant", 1, False),
        (1, "non_relevant", 0, True),
        (0, "non_relevant", 1, True),
        (pd.NA, "unjudged", 0, False),
        (pd.NA, "unjudged", pd.NA, True),
        (pd.NA, "unknown", pd.NA, False),
    ],
)
def test_inconsistent_judgment_combinations_fail(
    grade: object, judgment: str, relevance: object, is_judged: bool
) -> None:
    candidates = valid_candidates().iloc[[0]].copy()
    candidates.loc[candidates.index[0], [
        "relevance_grade",
        "judgment",
        "relevance",
        "is_judged",
    ]] = [grade, judgment, relevance, is_judged]
    with pytest.raises(ValueError):
        validate_candidate_schema(candidates)


def test_duplicate_query_document_fails() -> None:
    candidates = pd.concat([valid_candidates(), valid_candidates().iloc[[0]]], ignore_index=True)
    candidates.loc[3, "bm25_rank"] = 4
    with pytest.raises(ValueError, match=r"duplicate \(query_id, docid\)"):
        validate_candidate_schema(candidates)


def test_duplicate_rank_within_query_fails() -> None:
    candidates = valid_candidates()
    candidates.loc[2, "bm25_rank"] = 2
    with pytest.raises(ValueError, match="duplicate bm25_rank"):
        validate_candidate_schema(candidates)


@pytest.mark.parametrize("rank", [0, -1, 1.5, "not-a-rank", pd.NA])
def test_invalid_rank_fails(rank: object) -> None:
    candidates = valid_candidates()
    candidates["bm25_rank"] = candidates["bm25_rank"].astype("object")
    candidates.loc[0, "bm25_rank"] = rank
    with pytest.raises(ValueError, match="positive integers"):
        validate_candidate_schema(candidates)


@pytest.mark.parametrize("column", ["split", "query_id", "docid"])
def test_null_identifier_fails(column: str) -> None:
    candidates = valid_candidates()
    candidates.loc[0, column] = pd.NA
    with pytest.raises(ValueError, match="identifiers contain nulls"):
        validate_candidate_schema(candidates)


def test_attach_qrels_preserves_grade_and_three_states_without_row_growth() -> None:
    run = valid_candidates()[
        ["split", "query_id", "docid", "bm25_rank", "bm25_score"]
    ]
    qrels = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "docid": ["d1", "d2"],
            "relevance": [2, 0],
        }
    )
    result = attach_qrels(run, qrels)
    assert len(result) == len(run)
    assert result["relevance_grade"].tolist() == [2, 0, pd.NA]
    assert result["judgment"].tolist() == ["relevant", "non_relevant", "unjudged"]
    assert result["is_judged"].tolist() == [True, True, False]
    assert pd.isna(result.loc[2, "relevance"])


def test_duplicate_qrels_are_rejected_before_join() -> None:
    run = valid_candidates()[
        ["split", "query_id", "docid", "bm25_rank", "bm25_score"]
    ]
    qrels = pd.DataFrame(
        {"query_id": ["q1", "q1"], "docid": ["d1", "d1"], "relevance": [1, 1]}
    )
    with pytest.raises(ValueError, match="qrels contain duplicate"):
        attach_qrels(run, qrels)


def test_conflicting_qrels_grade_columns_are_rejected() -> None:
    run = valid_candidates().iloc[[0]][
        ["split", "query_id", "docid", "bm25_rank", "bm25_score"]
    ]
    qrels = pd.DataFrame(
        {
            "query_id": ["q1"],
            "docid": ["d1"],
            "relevance": [1],
            "relevance_grade": [0],
        }
    )
    with pytest.raises(ValueError, match="conflict"):
        attach_qrels(run, qrels)


def test_negative_qrels_grade_is_rejected() -> None:
    run = valid_candidates().iloc[[0]][
        ["split", "query_id", "docid", "bm25_rank", "bm25_score"]
    ]
    qrels = pd.DataFrame(
        {"query_id": ["q1"], "docid": ["d1"], "relevance_grade": [-1]}
    )
    with pytest.raises(ValueError, match="non-negative"):
        attach_qrels(run, qrels)


def test_queries_and_passages_contracts() -> None:
    queries = pd.DataFrame(
        {"split": ["dev"], "query_id": ["q1"], "query_text": ["русский запрос"]}
    )
    passages = pd.DataFrame(
        {"docid": ["d1"], "title": [""], "text": ["Непустой текст"]}
    )
    validate_queries(queries)
    validate_passages(passages)


def test_extract_passages_keeps_only_candidates_and_unicode() -> None:
    rows = [
        {"docid": "skip", "title": "X", "text": "not retained"},
        {"docid": "d2", "title": None, "text": "Русский текст"},
        {"docid": "d1", "title": "Заголовок", "text": "Ещё текст"},
    ]
    result = extract_passages_from_rows(rows, {"d1", "d2"})
    assert result["docid"].tolist() == ["d1", "d2"]
    assert result.loc[result["docid"].eq("d2"), "title"].item() == ""
    assert "Русский" in result.loc[result["docid"].eq("d2"), "text"].item()


def test_extract_passages_rejects_missing_and_conflicting_content() -> None:
    with pytest.raises(ValueError, match="missing from corpus"):
        extract_passages_from_rows([], {"d1"})
    rows = [
        {"docid": "d1", "title": "A", "text": "one"},
        {"docid": "d1", "title": "A", "text": "two"},
        {"docid": "d2", "title": "B", "text": "ok"},
    ]
    with pytest.raises(ValueError, match="conflicting content"):
        extract_passages_from_rows(rows, {"d1", "d2"})
