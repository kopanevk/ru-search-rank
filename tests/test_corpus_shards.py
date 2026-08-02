"""Static-shard corpus access: ordering, parsing, streaming, and failure paths.

Every test here reads real gzip bytes. The previous implementation called
``load_dataset("miracl/miracl-corpus", "ru", streaming=True)``, which modern
``datasets`` refuses to execute because the repository ships a
``miracl-corpus.py`` dataset script; nothing in this module can pass through
that path.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import shard_fixtures
from rusearchrank.corpus import (
    CorpusAccessError,
    CorpusRowError,
    HubShardSource,
    LocalShardSource,
    MissingCandidatePassagesError,
    assert_no_dataset_script,
    extract_candidate_passages,
    iter_shard_rows,
    parse_corpus_line,
    shard_index,
    shard_names,
    sort_shard_names,
)


def _local_source(directory: Path, shard_count: int) -> LocalShardSource:
    return LocalShardSource(
        language="ru", shard_count=shard_count, directory=directory
    )


# --------------------------------------------------------------------------
# Shard list, ordering, and the banned dataset script
# --------------------------------------------------------------------------


def test_official_russian_shard_list_is_exact_and_numeric() -> None:
    names = shard_names("ru", 20)
    assert len(names) == 20
    assert names[0] == "miracl-corpus-v1.0-ru/docs-0.jsonl.gz"
    assert names[19] == "miracl-corpus-v1.0-ru/docs-19.jsonl.gz"
    assert [shard_index(name) for name in names] == list(range(20))


def test_shards_sort_numerically_not_lexicographically() -> None:
    lexicographic = sorted(shard_names("ru", 20))
    assert lexicographic[1].endswith("docs-1.jsonl.gz")
    assert lexicographic[2].endswith("docs-10.jsonl.gz")  # the trap
    assert sort_shard_names(lexicographic) == shard_names("ru", 20)


def test_shard_list_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        shard_names("", 20)
    with pytest.raises(ValueError):
        shard_names("ru", 0)
    with pytest.raises(ValueError, match="not an official MIRACL corpus shard"):
        shard_index("miracl-corpus-v1.0-ru/docs-x.jsonl.gz")


def test_dataset_script_is_refused_everywhere() -> None:
    with pytest.raises(CorpusAccessError, match="never a dataset script"):
        assert_no_dataset_script(["miracl-corpus-v1.0-ru/docs-0.jsonl.gz", "miracl-corpus.py"])
    source = HubShardSource(
        language="ru",
        shard_count=20,
        repo_id="miracl/miracl-corpus",
        revision="d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
    )
    with pytest.raises(CorpusAccessError, match="never a dataset script"):
        source.local_path("miracl-corpus.py")


def test_hub_source_requires_an_immutable_pinned_revision() -> None:
    source = HubShardSource(
        language="ru", shard_count=20, repo_id="miracl/miracl-corpus", revision="main"
    )
    with pytest.raises(CorpusAccessError, match="immutable pinned revision"):
        source.local_path("miracl-corpus-v1.0-ru/docs-0.jsonl.gz")


def test_hub_source_describes_a_script_free_loader() -> None:
    described = HubShardSource(
        language="ru",
        shard_count=20,
        repo_id="miracl/miracl-corpus",
        revision="d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
    ).describe()
    assert described["loader"] == "hf_hub_download + gzip + json.loads"
    assert described["dataset_script_used"] is False
    assert described["trust_remote_code"] is False
    assert described["repo_type"] == "dataset"


# --------------------------------------------------------------------------
# JSONL.GZ parsing
# --------------------------------------------------------------------------


def test_parses_a_real_jsonl_gz_shard(tmp_path: Path) -> None:
    shard_fixtures.write_shard(
        tmp_path,
        language="ru",
        index=0,
        rows=[shard_fixtures.passage("7#0"), shard_fixtures.passage("7#1")],
    )
    path = shard_fixtures.shard_path(tmp_path, language="ru", index=0)
    rows = list(iter_shard_rows(path, shard="docs-0"))
    assert [number for number, _ in rows] == [1, 2]
    assert [row["docid"] for _, row in rows] == ["7#0", "7#1"]
    assert all(isinstance(row["docid"], str) for _, row in rows)
    assert "Русский" in rows[0][1]["text"]


def test_max_rows_limits_a_real_stream(tmp_path: Path) -> None:
    shard_fixtures.write_shard(
        tmp_path, language="ru", index=0, rows=shard_fixtures.filler("d", 50)
    )
    path = shard_fixtures.shard_path(tmp_path, language="ru", index=0)
    assert len(list(iter_shard_rows(path, shard="docs-0", max_rows=7))) == 7


def test_numeric_docid_is_rejected_so_ids_stay_strings() -> None:
    with pytest.raises(CorpusRowError, match="docid must stay a string"):
        parse_corpus_line(
            json.dumps({"docid": 7, "title": "t", "text": "x"}),
            shard="docs-0",
            line_number=3,
        )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('{"docid": "d", "title": "t"}', "missing required fields: text"),
        ('{"docid": "d", "text": "x"}', "missing required fields: title"),
        ('{"docid": "", "title": "t", "text": "x"}', "empty docid"),
        ('{"docid": "d", "title": 5, "text": "x"}', "title must be a string"),
        ('{"docid": "d", "title": "t", "text": 5}', "text must be a string"),
        ('["not", "an", "object"]', "must be a JSON object"),
    ],
)
def test_malformed_rows_name_the_shard_and_line(payload: str, expected: str) -> None:
    with pytest.raises(CorpusRowError, match=expected) as error:
        parse_corpus_line(payload, shard="docs-3", line_number=41)
    assert "docs-3:41" in str(error.value)


def test_malformed_json_names_the_shard_and_line(tmp_path: Path) -> None:
    path = shard_fixtures.shard_path(tmp_path, language="ru", index=0)
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(shard_fixtures.passage("d0")) + "\n")
        stream.write("{not json\n")
    with pytest.raises(CorpusRowError, match="invalid JSON"):
        list(iter_shard_rows(path, shard="docs-0"))


def test_corrupted_gzip_is_reported_as_a_corpus_access_error(tmp_path: Path) -> None:
    path = shard_fixtures.shard_path(tmp_path, language="ru", index=0)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x1f\x8b\x08\x00 definitely not gzip payload")
    with pytest.raises(CorpusAccessError, match="corrupted corpus shard"):
        list(iter_shard_rows(path, shard="docs-0"))


def test_missing_and_empty_shards_are_reported(tmp_path: Path) -> None:
    with pytest.raises(CorpusAccessError, match="not a regular file"):
        list(iter_shard_rows(tmp_path / "absent.jsonl.gz", shard="docs-0"))
    empty = tmp_path / "empty.jsonl.gz"
    empty.write_bytes(b"")
    with pytest.raises(CorpusAccessError, match="is empty"):
        list(iter_shard_rows(empty, shard="docs-0"))
    with pytest.raises(CorpusAccessError, match="local corpus shard does not exist"):
        _local_source(tmp_path, 1).local_path("miracl-corpus-v1.0-ru/docs-0.jsonl.gz")


# --------------------------------------------------------------------------
# Streaming extraction
# --------------------------------------------------------------------------


def test_extraction_streams_batches_and_stops_early(tmp_path: Path) -> None:
    shard_fixtures.write_corpus(
        tmp_path,
        shards=[
            [*shard_fixtures.filler("a", 10), shard_fixtures.passage("want-1")],
            [shard_fixtures.passage("want-2"), *shard_fixtures.filler("b", 10)],
            shard_fixtures.filler("c", 10),
        ],
    )
    output = tmp_path / "passages.parquet"
    messages: list[str] = []
    report = extract_candidate_passages(
        {"want-1", "want-2"},
        output,
        source=_local_source(tmp_path, 3),
        batch_rows=1,
        log=messages.append,
    )
    assert report["found_docids"] == 2
    assert report["rows_written"] == 2
    assert report["batches_written"] == 2
    assert report["early_stop"] is True
    assert report["shards_visited"] == 2
    assert report["shards_skipped_by_early_stop"] == 1
    assert report["lines_visited"] == 12
    assert report["source"]["dataset_script_used"] is False
    assert any("early stop" in message for message in messages)
    table = pq.read_table(output)
    assert table.num_rows == 2
    assert table.schema.names == ["docid", "title", "text"]
    assert not (tmp_path / "passages.parquet.partial").exists()


def test_extraction_writes_many_batches_for_a_large_request(tmp_path: Path) -> None:
    wanted = [f"w{index}#0" for index in range(120)]
    shard_fixtures.write_corpus(
        tmp_path,
        shards=[[shard_fixtures.passage(docid) for docid in wanted]],
    )
    output = tmp_path / "passages.parquet"
    report = extract_candidate_passages(
        set(wanted), output, source=_local_source(tmp_path, 1), batch_rows=25
    )
    assert report["rows_written"] == 120
    assert report["batches_written"] == 5
    parquet_file = pq.ParquetFile(output)
    assert parquet_file.metadata.num_rows == 120
    assert parquet_file.num_row_groups == 5


def test_missing_passages_report_split_query_and_scan_progress(tmp_path: Path) -> None:
    shard_fixtures.write_corpus(tmp_path, shards=[[shard_fixtures.passage("present")]])
    with pytest.raises(MissingCandidatePassagesError) as error:
        extract_candidate_passages(
            {"present", "absent"},
            tmp_path / "passages.parquet",
            source=_local_source(tmp_path, 1),
            candidate_context={"absent": [{"split": "train", "query_id": "375"}]},
        )
    message = str(error.value)
    assert "1 candidate passages missing" in message
    assert "docid=absent" in message
    assert "split=train" in message
    assert "query_id=375" in message
    assert "after scanning 1 shards / 1 lines" in message
    assert not (tmp_path / "passages.parquet").exists()  # never published


def test_duplicate_corpus_docid_in_the_scanned_region_blocks_publication(
    tmp_path: Path,
) -> None:
    # The duplicate must fall inside the streamed region: extraction stops as
    # soon as nothing remains, so duplicates after that point are never scanned.
    shard_fixtures.write_corpus(
        tmp_path,
        shards=[
            [
                shard_fixtures.passage("dup"),
                shard_fixtures.passage("dup", text="Другой текст"),
                shard_fixtures.passage("second"),
            ]
        ],
    )
    with pytest.raises(ValueError, match="duplicate candidate docids"):
        extract_candidate_passages(
            {"dup", "second"},
            tmp_path / "passages.parquet",
            source=_local_source(tmp_path, 1),
        )
    assert not (tmp_path / "passages.parquet").exists()


def test_duplicate_detection_scope_is_reported_as_bounded(tmp_path: Path) -> None:
    shard_fixtures.write_corpus(tmp_path, shards=[[shard_fixtures.passage("d1")]])
    report = extract_candidate_passages(
        {"d1"}, tmp_path / "passages.parquet", source=_local_source(tmp_path, 1)
    )
    assert report["duplicate_detection_scope"] == "scanned_lines_only"


def test_empty_candidate_text_blocks_publication(tmp_path: Path) -> None:
    shard_fixtures.write_corpus(
        tmp_path, shards=[[shard_fixtures.passage("blank", text="   ")]]
    )
    with pytest.raises(ValueError, match="has empty text"):
        extract_candidate_passages(
            {"blank"}, tmp_path / "passages.parquet", source=_local_source(tmp_path, 1)
        )
    assert not (tmp_path / "passages.parquet").exists()


def test_empty_titles_are_allowed_and_counted(tmp_path: Path) -> None:
    shard_fixtures.write_corpus(
        tmp_path,
        shards=[
            [
                shard_fixtures.passage("with-title", title="Заголовок"),
                {"docid": "no-title", "title": None, "text": "Текст"},
                shard_fixtures.passage("blank-title", title=""),
            ]
        ],
    )
    report = extract_candidate_passages(
        {"with-title", "no-title", "blank-title"},
        tmp_path / "passages.parquet",
        source=_local_source(tmp_path, 1),
    )
    assert report["empty_titles"] == 2
    assert report["found_docids"] == 3


def test_empty_candidate_set_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty identifiers"):
        extract_candidate_passages(
            set(), tmp_path / "passages.parquet", source=_local_source(tmp_path, 1)
        )
    with pytest.raises(ValueError, match="non-empty identifiers"):
        extract_candidate_passages(
            {"  "}, tmp_path / "passages.parquet", source=_local_source(tmp_path, 1)
        )


def test_interrupted_partial_parquet_is_preserved_not_deleted(tmp_path: Path) -> None:
    shard_fixtures.write_corpus(tmp_path, shards=[[shard_fixtures.passage("d1")]])
    output = tmp_path / "passages.parquet"
    partial = tmp_path / "passages.parquet.partial"
    partial.write_bytes(b"interrupted run bytes")
    messages: list[str] = []
    report = extract_candidate_passages(
        {"d1"}, output, source=_local_source(tmp_path, 1), log=messages.append
    )
    assert report["preserved_partial_from_previous_run"] is not None
    preserved = Path(report["preserved_partial_from_previous_run"])
    assert preserved.read_bytes() == b"interrupted run bytes"
    assert not partial.exists()
    assert output.is_file()
    assert any("preserved" in message for message in messages)


# --------------------------------------------------------------------------
# Hub failure paths (no network: the client is replaced by explicit stubs)
# --------------------------------------------------------------------------


class _StubPath:
    def __init__(self, path: str, size: int = 1) -> None:
        self.path = path
        self.size = size


def _install_stub_hub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset_info: object,
    paths_info: object,
) -> None:
    import huggingface_hub

    class StubApi:
        def dataset_info(self, repo_id: str, **_: object) -> object:
            if isinstance(dataset_info, Exception):
                raise dataset_info
            return dataset_info

        def get_paths_info(self, repo_id: str, paths: list[str], **_: object) -> object:
            if isinstance(paths_info, Exception):
                raise paths_info
            return paths_info

    monkeypatch.setattr(huggingface_hub, "HfApi", StubApi)


class _StubInfo:
    def __init__(self, sha: str) -> None:
        self.sha = sha


def test_hub_unavailable_is_reported_with_repo_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub_hub(
        monkeypatch,
        dataset_info=ConnectionError("hub is unreachable"),
        paths_info=[],
    )
    from rusearchrank.corpus import resolve_hub_shards

    with pytest.raises(CorpusAccessError) as error:
        resolve_hub_shards(
            repo_id="miracl/miracl-corpus",
            revision="d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
            language="ru",
            shard_count=20,
        )
    message = str(error.value)
    assert "pinned corpus revision is unavailable" in message
    assert "miracl/miracl-corpus" in message
    assert "hub is unreachable" in message


def test_revision_that_resolves_elsewhere_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub_hub(
        monkeypatch, dataset_info=_StubInfo("f" * 40), paths_info=[]
    )
    from rusearchrank.corpus import resolve_hub_shards

    with pytest.raises(CorpusAccessError, match="resolved to a different commit"):
        resolve_hub_shards(
            repo_id="miracl/miracl-corpus",
            revision="d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
            language="ru",
            shard_count=20,
        )


def test_missing_shard_at_the_pinned_revision_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "d921ec7e349ce0d28daf30b2da9da5ee698bef0d"
    present = [
        _StubPath(name) for name in shard_names("ru", 20)[:19]
    ]  # docs-19 is missing
    _install_stub_hub(
        monkeypatch, dataset_info=_StubInfo(revision), paths_info=present
    )
    from rusearchrank.corpus import resolve_hub_shards

    with pytest.raises(CorpusAccessError, match="missing shards"):
        resolve_hub_shards(
            repo_id="miracl/miracl-corpus",
            revision=revision,
            language="ru",
            shard_count=20,
        )


def test_interrupted_shard_download_names_repo_revision_and_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub

    def interrupted(**_: object) -> str:
        raise OSError("connection reset while downloading")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", interrupted)
    source = HubShardSource(
        language="ru",
        shard_count=20,
        repo_id="miracl/miracl-corpus",
        revision="d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
        cache_dir=tmp_path,
    )
    with pytest.raises(CorpusAccessError) as error:
        source.local_path("miracl-corpus-v1.0-ru/docs-0.jsonl.gz")
    message = str(error.value)
    assert "failed to download official corpus shard" in message
    assert "docs-0.jsonl.gz" in message
    assert "d921ec7e349ce0d28daf30b2da9da5ee698bef0d" in message
    assert str(tmp_path) in message
    assert "connection reset" in message


def test_shard_source_never_touches_a_dataset_script(tmp_path: Path) -> None:
    shard_fixtures.write_corpus(tmp_path, shards=[[shard_fixtures.passage("d1")]])
    (tmp_path / "miracl-corpus.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    report = extract_candidate_passages(
        {"d1"}, tmp_path / "passages.parquet", source=_local_source(tmp_path, 1)
    )
    assert report["status"] == "PASS"
    assert report["source"]["dataset_script_used"] is False
