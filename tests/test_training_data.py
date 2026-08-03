from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from rusearchrank.training_data import (
    PAIR_SCHEMA,
    QUERY_SPLIT_SCHEMA,
    assert_not_dev,
    atomic_write_parquet,
    build_pair_frame,
    build_query_split_frame,
    load_finetune_config,
    manifest_declared_hashes,
    materialize_control_pairs,
    merge_small_strata,
    sample_weak_negatives,
)


def candidate_fixture(query_count: int = 40) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_number in range(query_count):
        query_id = f"q{query_number:03d}"
        for rank in range(1, 101):
            if query_number == 0 and rank <= 12:
                grade: int | pd._libs.missing.NAType = 1
                judgment = "relevant"
            elif query_number == 0 and 13 <= rank <= 32:
                grade = 0
                judgment = "judged_non_relevant"
            elif query_number > 0 and rank == 1:
                grade = 1
                judgment = "relevant"
            elif query_number > 0 and query_number % 2 == 0 and 2 <= rank <= 12:
                grade = 0
                judgment = "judged_non_relevant"
            else:
                grade = pd.NA
                judgment = "unjudged"
            rows.append(
                {
                    "split": "train",
                    "query_id": query_id,
                    "docid": f"{query_id}-d{rank:03d}",
                    "bm25_rank": rank,
                    "bm25_score": float(101 - rank),
                    "relevance_grade": grade,
                    "judgment": judgment,
                    "relevance": (
                        1
                        if judgment == "relevant"
                        else 0
                        if judgment == "judged_non_relevant"
                        else pd.NA
                    ),
                    "is_judged": judgment != "unjudged",
                }
            )
    frame = pd.DataFrame(rows)
    frame["relevance_grade"] = frame["relevance_grade"].astype("Int64")
    frame["relevance"] = frame["relevance"].astype("Int64")
    return frame


def all_fit_split(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query_id, group in candidates.groupby("query_id", sort=True):
        relevant = int(group["judgment"].eq("relevant").sum())
        negatives = int(group["judgment"].eq("judged_non_relevant").sum())
        bucket = str(relevant) if relevant <= 2 else "3+"
        rows.append(
            {
                "query_id": str(query_id),
                "split_role": "train_fit",
                "n_relevant_in_candidates": relevant,
                "n_judged_negatives_in_candidates": negatives,
                "relevant_bucket": bucket,
                "has_judged_negative": negatives > 0,
                "stratum": f"({bucket},{'T' if negatives else 'F'})",
                "merged_stratum": f"({bucket},{'T' if negatives else 'F'})",
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/candidates/dev_top100.parquet",
        "artifacts/runs/dev_bm25_top100.trec",
        "artifacts/runs/dev_rerank_x.trec",
        "qrels.miracl-v1.0-ru-dev.tsv",
    ],
)
def test_assert_not_dev_rejects_all_frozen_markers(path: str) -> None:
    with pytest.raises(ValueError, match="isolated evaluation boundary"):
        assert_not_dev(path)


def test_split_is_deterministic_disjoint_and_preserves_full_groups(tmp_path: Path) -> None:
    candidates = candidate_fixture()
    first, first_manifest = build_query_split_frame(candidates)
    second, second_manifest = build_query_split_frame(candidates.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(first, second)
    assert first_manifest == second_manifest
    fit = set(first.loc[first.split_role.eq("train_fit"), "query_id"])
    validation = set(first.loc[first.split_role.eq("train_validation"), "query_id"])
    assert not fit & validation
    assert fit | validation == set(first.query_id)
    assert first_manifest["dev_query_count"] == 0
    path1 = tmp_path / "one.parquet"
    path2 = tmp_path / "two.parquet"
    atomic_write_parquet(path1, first, schema=QUERY_SPLIT_SCHEMA)
    atomic_write_parquet(path2, second, schema=QUERY_SPLIT_SCHEMA)
    assert path1.read_bytes() == path2.read_bytes()


def test_training_frame_rejects_source_evaluation_rows_before_filtering() -> None:
    candidates = candidate_fixture()
    candidates.loc[candidates.index[0], "split"] = "dev"
    with pytest.raises(ValueError, match="isolated evaluation rows"):
        build_query_split_frame(candidates)


def test_small_strata_merge_is_deterministic_and_reaches_minimum() -> None:
    sizes = {
        "(0,F)": 3,
        "(0,T)": 4,
        "(1,F)": 30,
        "(1,T)": 1,
        "(2,F)": 0,
        "(2,T)": 0,
        "(3+,F)": 5,
        "(3+,T)": 6,
    }
    first = merge_small_strata(sizes, min_stratum_size=20)
    second = merge_small_strata(copy.deepcopy(sizes), min_stratum_size=20)
    assert first == second
    mapping, order, members = first
    assert set(mapping) == {name for name, size in sizes.items() if size}
    assert list(members) == order
    merged_sizes = {
        group: sum(sizes[name] for name in group_members)
        for group, group_members in members.items()
    }
    assert len(merged_sizes) == 1 or min(merged_sizes.values()) >= 20


def test_weak_sampling_is_deterministic_random_within_bucket_and_preserves_nulls() -> None:
    group = candidate_fixture(1)
    # Make all ranks 26-100 available as unjudged for an observable sample.
    first = sample_weak_negatives("q000", group, global_seed=20260803)
    second = sample_weak_negatives("q000", group.sample(frac=1, random_state=2), global_seed=20260803)
    assert first == second
    assert len(first) == 8
    selected_26_50 = [row["bm25_rank"] for row in first if row["bucket_id"] == "26-50"]
    assert selected_26_50 != sorted(range(33, 33 + len(selected_26_50)))
    assert group.loc[group.judgment.eq("unjudged"), "relevance"].isna().all()


def test_weak_quota_is_up_to_eight_and_bucket_fallback_round_robins() -> None:
    group = candidate_fixture(1).iloc[:32].copy()
    group.loc[:, "judgment"] = "judged_non_relevant"
    group.loc[:, "relevance_grade"] = 0
    group.loc[:, "relevance"] = 0
    group.loc[:, "is_judged"] = True
    for rank in (26, 51, 52, 76, 77):
        index = group.index[group.bm25_rank.eq(rank)]
        if len(index):
            group.loc[index, ["judgment", "relevance_grade", "relevance", "is_judged"]] = [
                "unjudged",
                pd.NA,
                pd.NA,
                False,
            ]
    # Rebuild a minimal five-document source because the sliced fixture lacks 51+.
    source = candidate_fixture(1)
    weak = source.loc[source.bm25_rank.isin([26, 51, 52, 76, 77])].copy()
    weak["judgment"] = "unjudged"
    weak["relevance_grade"] = pd.NA
    weak["relevance"] = pd.NA
    weak["is_judged"] = False
    selected = sample_weak_negatives("few", weak, global_seed=20260803)
    assert len(selected) == 5
    assert {row["bucket_id"] for row in selected} == {"26-50", "51-75", "76-100"}


def test_global_round_robin_covers_12_positives_and_keeps_both_sources() -> None:
    candidates = candidate_fixture()
    split = all_fit_split(candidates)
    pairs, section = build_pair_frame(
        candidates,
        split,
        regime="weak_negatives",
        config=load_finetune_config(),
    )
    q0 = pairs.loc[pairs.query_id.eq("q000")]
    assert len(q0) == 16
    assert q0.positive_docid.nunique() == 12
    assert q0.negative_source.value_counts().to_dict() == {
        "judged_non_relevant": 8,
        "weak_unjudged": 8,
    }
    assert section["negative_source_distribution"]["weak_unjudged"] > 0
    assert not pairs.duplicated(["query_id", "positive_docid", "negative_docid"]).any()


def test_a_cap_and_population_difference_are_explicit() -> None:
    candidates = candidate_fixture()
    split = all_fit_split(candidates)
    config = load_finetune_config()
    judged, section_a = build_pair_frame(candidates, split, regime="judged_only", config=config)
    weak, section_b = build_pair_frame(candidates, split, regime="weak_negatives", config=config)
    assert judged.groupby("query_id").size().max() <= 16
    assert weak.groupby("query_id").size().max() <= 16
    assert section_b["usable_query_count"] > section_a["usable_query_count"]
    assert set(judged.negative_source) == {"judged_non_relevant"}
    assert set(weak.pair_weight.unique()) == {0.5, 1.0}
    assert section_b["weak_weight_share_per_query"]["per_query"]


def test_control_selection_and_shuffle_ignore_physical_parquet_order(tmp_path: Path) -> None:
    candidates = candidate_fixture()
    split = all_fit_split(candidates)
    judged, _ = build_pair_frame(
        candidates, split, regime="judged_only", config=load_finetune_config()
    )
    first, first_section = materialize_control_pairs(
        judged,
        split,
        target_pairs=45,
        merged_strata_order=["(1,F)", "(1,T)", "(3+,T)"],
    )
    second, second_section = materialize_control_pairs(
        judged.sample(frac=1, random_state=91),
        split.sample(frac=1, random_state=19),
        target_pairs=45,
        merged_strata_order=["(1,F)", "(1,T)", "(3+,T)"],
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_section == second_section
    one = tmp_path / "one.parquet"
    two = tmp_path / "two.parquet"
    atomic_write_parquet(one, first, schema=PAIR_SCHEMA)
    atomic_write_parquet(two, second, schema=PAIR_SCHEMA)
    assert one.read_bytes() == two.read_bytes()
    assert pq.read_table(one).schema.equals(PAIR_SCHEMA, check_metadata=False)
    required_audit_fields = {
        "excluded_query_counts",
        "judged_pairs_before_cap",
        "judged_pairs_after_cap",
        "weak_pairs_before_cap",
        "weak_pairs_after_cap",
        "queries_with_weak_pairs_before_cap",
        "queries_with_weak_pairs_after_cap",
        "weak_bucket_distribution",
        "label_audit",
        "population_disclosure",
        "weight_disclosure",
        "heuristic_disclosure",
    }
    assert required_audit_fields.issubset(first_section)


def test_control_rejects_validation_query_groups() -> None:
    candidates = candidate_fixture()
    split = all_fit_split(candidates)
    judged, _ = build_pair_frame(
        candidates, split, regime="judged_only", config=load_finetune_config()
    )
    leaked_query = str(judged.iloc[0].query_id)
    split.loc[split.query_id.eq(leaked_query), "split_role"] = "train_validation"
    with pytest.raises(ValueError, match="train_validation"):
        materialize_control_pairs(judged, split, target_pairs=20)


def test_manifest_hash_parser_binds_named_inputs_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"files":[{"path":"payload.bin","sha256":"'
        + digest
        + '","input_hashes":{"dev_qrels":"'
        + digest
        + '"}}]}\n',
        encoding="utf-8",
    )
    assert manifest_declared_hashes(manifest) == {
        "payload.bin": digest,
        "dev_qrels": digest,
    }
    manifest.write_text(
        '{"files":[{"input_hashes":{"dev_qrels":"'
        + digest
        + '"}},{"input_hashes":{"dev_qrels":"'
        + "b" * 64
        + '"}}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting"):
        manifest_declared_hashes(manifest)
