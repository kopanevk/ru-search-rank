from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rusearchrank.evaluation import (
    assert_candidate_set_invariant,
    paired_bootstrap,
    parse_trec_eval_per_query,
)
from rusearchrank.rerank import (
    SCORE_SCHEMA,
    build_input_fingerprint,
    derive_rerank_ranking,
    format_document,
    key_set_sha256,
    plan_query_shards,
    prepare_pair,
    render_rank_preserving_trec,
    score_schema_json,
    source_tree_sha256,
    token_accounting,
    validate_rank_preserving_trec,
    validate_score_table,
)


class TinyTokenizer:
    """Pair tokenizer with XLM-R-like four special tokens."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @staticmethod
    def _tokens(text: str) -> list[int]:
        # Character tokens make NBSP preservation and exact lengths observable.
        return [10 + (ord(character) % 101) for character in text]

    def __call__(
        self,
        query: str,
        document: str,
        *,
        truncation: bool | str,
        add_special_tokens: bool,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        self.calls.append(
            {
                "query": query,
                "document": document,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        query_tokens = self._tokens(query)
        document_tokens = self._tokens(document)
        if truncation == "only_second" and max_length is not None:
            room = max_length - len(query_tokens) - 4
            if room < 0:
                raise ValueError("query is too long")
            document_tokens = document_tokens[:room]
        ids = [0, *query_tokens, 2, 2, *document_tokens, 2]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def score_table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=SCORE_SCHEMA)


def valid_rows() -> list[dict[str, object]]:
    return [
        {
            "query_id": "q1",
            "docid": "d1",
            "score": np.float32(1.0),
            "pair_tokens_before_truncation": np.int32(8),
            "pair_tokens_after_truncation": np.int32(8),
            "truncated": False,
        },
        {
            "query_id": "q1",
            "docid": "d2",
            "score": np.float32(0.5),
            "pair_tokens_before_truncation": np.int32(12),
            "pair_tokens_after_truncation": np.int32(10),
            "truncated": True,
        },
    ]


@pytest.mark.parametrize("title", [None, "", "   "])
def test_document_format_omits_separator_for_empty_title(title: str | None) -> None:
    assert format_document(title, "текст") == "текст"


def test_pair_format_preserves_nbsp_and_query_document_order() -> None:
    tokenizer = TinyTokenizer()
    pair = prepare_pair(
        tokenizer,
        query_id="q",
        docid="d",
        query_text="что\xa0это",
        title=" Заголовок ",
        text="текст\xa0без нормализации",
        max_length=320,
    )
    assert tokenizer.calls[0]["query"] == "что\xa0это"
    assert tokenizer.calls[0]["document"] == " Заголовок \nтекст\xa0без нормализации"
    assert pair.truncated is False


@pytest.mark.parametrize(
    ("query", "document", "max_length", "expected_before", "expected_after", "truncated"),
    [
        ("qq", "ddd", 10, 9, 9, False),
        ("qq", "dddd", 10, 10, 10, False),
        ("qq", "ddddd", 10, 11, 10, True),
    ],
)
def test_token_accounting_exact_cases(
    query: str,
    document: str,
    max_length: int,
    expected_before: int,
    expected_after: int,
    truncated: bool,
) -> None:
    pair = prepare_pair(
        TinyTokenizer(),
        query_id="q",
        docid="d",
        query_text=query,
        title=None,
        text=document,
        max_length=max_length,
    )
    assert pair.pair_tokens_before_truncation == expected_before
    assert pair.pair_tokens_after_truncation == expected_after
    assert pair.truncated is truncated


def test_only_second_keeps_the_entire_query() -> None:
    tokenizer = TinyTokenizer()
    prepare_pair(
        tokenizer,
        query_id="q",
        docid="d",
        query_text="длинный запрос",
        title=None,
        text="д" * 1000,
        max_length=64,
    )
    assert tokenizer.calls[0]["query"] == tokenizer.calls[1]["query"]
    assert tokenizer.calls[1]["truncation"] == "only_second"


def test_aggregate_token_accounting_includes_padding_and_upper_bound() -> None:
    table = score_table(valid_rows())
    accounting = token_accounting(table, processed_tokens=24, max_length=320)
    assert accounting["sum_pair_tokens_after_truncation"] == 18
    assert accounting["padding_tokens"] == 6
    assert accounting["processed_tokens"] == 24
    assert accounting["processed_tokens_upper_bound"] == 2 * 320
    with pytest.raises(ValueError, match="exceed upper bound"):
        token_accounting(table, processed_tokens=641, max_length=320)


def _ranking(scores: list[float], docids: list[str]) -> pd.DataFrame:
    candidates = pd.DataFrame(
        {
            "query_id": ["q"] * len(docids),
            "docid": docids,
            "bm25_rank": range(1, len(docids) + 1),
        }
    )
    score_frame = pd.DataFrame(
        {
            "query_id": ["q"] * len(docids),
            "docid": docids,
            "score": np.asarray(scores, dtype=np.float32),
        }
    )
    result, _ = derive_rerank_ranking(
        candidates, score_frame, depth=100, official_depth=100
    )
    return result


def test_exact_float32_tie_uses_docid_ascending() -> None:
    result = _ranking([1.0, 1.0], ["z", "a"])
    assert result.sort_values("rank")["docid"].tolist() == ["a", "z"]


def test_last_bit_score_difference_wins_over_docid() -> None:
    high = np.nextafter(np.float32(1.0), np.float32(2.0), dtype=np.float32)
    result = _ranking([float(high), 1.0], ["z", "a"])
    assert result.sort_values("rank")["docid"].tolist() == ["z", "a"]


def test_scores_are_not_rounded_before_ordering() -> None:
    result = _ranking([1.0000001, 1.0], ["z", "a"])
    assert result.sort_values("rank")["docid"].tolist() == ["z", "a"]


def test_rank_preserving_trec_is_tie_break_independent_and_hides_raw_score(
    tmp_path: Path,
) -> None:
    ranking = _ranking([0.1234567, -9.25, 0.1234566], ["b", "c", "a"])
    rendered = render_rank_preserving_trec(ranking, tag="fixture")
    assert "0.1234567" not in rendered
    assert "-9.25" not in rendered
    path = tmp_path / "run.trec"
    path.write_text(rendered, encoding="utf-8")
    proof = validate_rank_preserving_trec(
        path,
        expected_keys={("q", "a"), ("q", "b"), ("q", "c")},
        expected_tag="fixture",
    )
    assert proof["docno_asc_desc_tie_break_independent"] is True


def test_query_shard_plan_is_deterministic_and_exact() -> None:
    first = plan_query_shards(["q3", "q1", "q2", "q1"], 2)
    second = plan_query_shards(reversed(["q3", "q1", "q2", "q1"]), 2)
    assert first == second == [("q1", "q2"), ("q3",)]
    assert [qid for shard in first for qid in shard] == ["q1", "q2", "q3"]


def fingerprint_components() -> dict[str, object]:
    return {
        "implementation_version": "2.0.0",
        "score_schema_version": 1,
        "source_tree_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": True,
        "config_sha256": "c" * 64,
        "candidates_sha256": "d" * 64,
        "queries_sha256": "e" * 64,
        "passages_sha256": "f" * 64,
        "model_id": "model",
        "model_revision": "1" * 40,
        "tokenizer_revision": "1" * 40,
        "max_length": 320,
        "truncation": "only_second",
        "pair_order": "query_document",
        "title_separator": "\n",
        "batch_size": 64,
        "device": "cuda",
        "dtype": "float16",
        "shard_queries": 64,
        "seed": 20260802,
        "python_version": "3.12.0",
        "torch_version": "2.8.0",
        "transformers_version": "5.0.0",
        "tokenizers_version": "0.22.0",
    }


@pytest.mark.parametrize("field", list(fingerprint_components()))
def test_each_input_fingerprint_component_is_load_bearing(field: str) -> None:
    base = fingerprint_components()
    changed = copy.deepcopy(base)
    value = changed[field]
    changed[field] = not value if isinstance(value, bool) else f"{value}-changed"
    assert build_input_fingerprint(base) != build_input_fingerprint(changed)


def test_source_tree_hash_changes_when_any_source_byte_changes(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    config_path = tmp_path / "rerank.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    config = {
        "_config_path": str(config_path),
        "paths": {"repository_root": "."},
        "implementation": {"source_files": ["b.py", "a.py"]},
    }
    before, before_files = source_tree_sha256(config)
    (tmp_path / "b.py").write_text("B", encoding="utf-8")
    after, after_files = source_tree_sha256(config)
    assert before != after
    assert before_files["a.py"] == after_files["a.py"]
    assert before_files["b.py"] != after_files["b.py"]


def test_score_schema_and_validation_reject_bad_values(tmp_path: Path) -> None:
    table = score_table(valid_rows())
    assert validate_score_table(table, expected_rows=2)["schema_json"] == score_schema_json()
    duplicate = score_table([valid_rows()[0], valid_rows()[0]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_score_table(duplicate)
    for bad_score in (float("nan"), float("inf")):
        rows = valid_rows()
        rows[0]["score"] = np.float32(bad_score)
        with pytest.raises(ValueError, match="NaN or infinite"):
            validate_score_table(score_table(rows))
    rows = valid_rows()
    rows[0]["pair_tokens_after_truncation"] = np.int32(0)
    with pytest.raises(ValueError, match="positive"):
        validate_score_table(score_table(rows))
    nullable_schema = pa.schema(
        [pa.field(field.name, field.type, nullable=True) for field in SCORE_SCHEMA]
    )
    nullable = pa.Table.from_pylist(valid_rows(), schema=nullable_schema)
    path = tmp_path / "nullable.parquet"
    pq.write_table(nullable, path)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_score_table(path)


def test_depth_rerank_preserves_tail_order_and_candidate_set() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": ["q"] * 12,
            "docid": [f"d{index:02d}" for index in range(12)],
            "bm25_rank": range(1, 13),
        }
    )
    scores = pd.DataFrame(
        {
            "query_id": ["q"] * 12,
            "docid": candidates["docid"],
            "score": np.arange(12, dtype=np.float32),
        }
    )
    ranking, _ = derive_rerank_ranking(candidates, scores, depth=10)
    ordered = ranking.sort_values("rank")
    assert ordered.iloc[:10]["docid"].tolist() == [
        f"d{index:02d}" for index in reversed(range(10))
    ]
    assert ordered.iloc[10:]["docid"].tolist() == ["d10", "d11"]
    assert set(zip(ranking.query_id, ranking.docid)) == set(
        zip(candidates.query_id, candidates.docid)
    )


def test_trec_eval_per_query_parser_and_bootstrap_reproducibility() -> None:
    output = "ndcg_cut_10 q2 0.5000\nndcg_cut_10 q1 1.0000\nndcg_cut_10 all 0.7500\n"
    assert parse_trec_eval_per_query(output, "ndcg_cut_10") == {
        "q2": 0.5,
        "q1": 1.0,
    }
    first = json.dumps(paired_bootstrap([0.1, -0.2, 0.3]), sort_keys=True)
    second = json.dumps(paired_bootstrap([0.1, -0.2, 0.3]), sort_keys=True)
    assert first.encode() == second.encode()


def test_recall_invariant_is_exact_key_set_equality() -> None:
    left = pd.DataFrame({"query_id": ["q"], "docid": ["d"]})
    right = left.copy()
    assert assert_candidate_set_invariant(left, right)["candidate_set_invariant"]
    with pytest.raises(ValueError, match="invariant failed"):
        assert_candidate_set_invariant(
            left, pd.DataFrame({"query_id": ["q"], "docid": ["other"]})
        )


def test_key_set_hash_is_order_independent() -> None:
    assert key_set_sha256([("q2", "d"), ("q1", "z")]) == key_set_sha256(
        [("q1", "z"), ("q2", "d")]
    )
