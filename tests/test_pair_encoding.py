from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rusearchrank.pair_encoding import build_document, encode_pair


GOLDEN = Path(__file__).parent / "fixtures/pair_encoding_golden.json"


def _load_complete_pinned_tokenizer(
    fixture: dict[str, object], transformers: object
):
    from huggingface_hub import hf_hub_download

    expected = fixture["tokenizer_payload_sha256"]
    assert isinstance(expected, dict)
    override = os.environ.get("RUSEARCHRANK_PINNED_TOKENIZER_DIR")
    if override:
        directory = Path(override).resolve()
        for filename, digest in expected.items():
            path = directory / str(filename)
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        return transformers.AutoTokenizer.from_pretrained(  # type: ignore[attr-defined]
            directory, local_files_only=True
        )
    resolved: list[Path] = []
    try:
        for filename, digest in expected.items():
            path = Path(
                hf_hub_download(
                    repo_id=str(fixture["model_id"]),
                    filename=str(filename),
                    revision=str(fixture["tokenizer_revision"]),
                    local_files_only=True,
                )
            )
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            resolved.append(path)
    except OSError:
        pytest.skip("complete pinned tokenizer payload is not in the offline cache")
    snapshot_directories = {path.resolve().parent for path in resolved}
    assert len(snapshot_directories) == 1
    return transformers.AutoTokenizer.from_pretrained(  # type: ignore[attr-defined]
        snapshot_directories.pop(), local_files_only=True
    )


class CharacterTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        query_ids = [10 + ord(value) % 211 for value in query]
        document_ids = [10 + ord(value) % 211 for value in document]
        if truncation == "only_second" and max_length is not None:
            room = max_length - len(query_ids) - 4
            if room < 0:
                raise ValueError("query exceeds the pair limit")
            document_ids = document_ids[:room]
        ids = [0, *query_ids, 2, 2, *document_ids, 2]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _hashes(encoded: dict[str, object]) -> dict[str, str]:
    result = {
        "input_ids_sha256": hashlib.sha256(
            np.asarray(encoded["input_ids"], dtype="<i8").tobytes()
        ).hexdigest(),
        "attention_mask_sha256": hashlib.sha256(
            bytes(encoded["attention_mask"])
        ).hexdigest(),
    }
    if "token_type_ids" in encoded:
        result["token_type_ids_sha256"] = hashlib.sha256(
            bytes(encoded["token_type_ids"])
        ).hexdigest()
    return result


def test_all_64_pinned_tokenizer_golden_pairs_match_byte_for_byte() -> None:
    transformers = pytest.importorskip("transformers")
    fixture = json.loads(GOLDEN.read_text(encoding="utf-8"))
    tokenizer = _load_complete_pinned_tokenizer(fixture, transformers)
    assert len(fixture["cases"]) == 64
    for case in fixture["cases"]:
        encoded = encode_pair(
            tokenizer,
            case["query"],
            build_document(case["title"], case["text"]),
            max_length=fixture["max_length"],
        )
        hashes = _hashes(encoded)
        assert hashes["input_ids_sha256"] == case["input_ids_sha256"], case["case_id"]
        assert hashes["attention_mask_sha256"] == case["attention_mask_sha256"], case["case_id"]
        if "token_type_ids_sha256" in case:
            assert hashes["token_type_ids_sha256"] == case["token_type_ids_sha256"]
        assert encoded["tokens_before"] == case["tokens_before"]
        assert encoded["tokens_after"] == case["tokens_after"]
        assert encoded["truncated"] is case["truncated"]


@pytest.mark.parametrize("title", [None, "", " ", "\t", "\n"])
def test_empty_or_whitespace_title_omits_separator(title: str | None) -> None:
    assert build_document(title, "текст") == "текст"


def test_nbsp_and_query_document_order_are_preserved() -> None:
    tokenizer = CharacterTokenizer()
    encode_pair(
        tokenizer,
        "что\xa0это",
        build_document(" Заголовок ", "текст\xa0без нормализации"),
    )
    assert tokenizer.calls[0]["query"] == "что\xa0это"
    assert tokenizer.calls[0]["document"] == " Заголовок \nтекст\xa0без нормализации"


@pytest.mark.parametrize(
    ("document", "expected_before", "expected_after", "truncated"),
    [("d" * 3, 9, 9, False), ("d" * 4, 10, 10, False), ("d" * 5, 11, 10, True)],
)
def test_token_accounting_short_exact_and_over(
    document: str, expected_before: int, expected_after: int, truncated: bool
) -> None:
    encoded = encode_pair(CharacterTokenizer(), "qq", document, max_length=10)
    assert encoded["tokens_before"] == expected_before
    assert encoded["tokens_after"] == expected_after
    assert encoded["truncated"] is truncated


def test_only_second_keeps_entire_query_with_long_document() -> None:
    tokenizer = CharacterTokenizer()
    encode_pair(tokenizer, "длинный запрос", "д" * 1000, max_length=64)
    assert tokenizer.calls[0]["query"] == tokenizer.calls[1]["query"]
    assert tokenizer.calls[1]["truncation"] == "only_second"


def test_changed_max_length_is_detectable_against_golden() -> None:
    fixture = json.loads(GOLDEN.read_text(encoding="utf-8"))
    over = next(case for case in fixture["cases"] if case["tokens_before"] > 320)
    transformers = pytest.importorskip("transformers")
    tokenizer = _load_complete_pinned_tokenizer(fixture, transformers)
    mutated = encode_pair(
        tokenizer,
        over["query"],
        build_document(over["title"], over["text"]),
        max_length=319,
    )
    assert _hashes(mutated)["input_ids_sha256"] != over["input_ids_sha256"]


def test_changed_truncation_strategy_is_detectable_against_golden() -> None:
    fixture = json.loads(GOLDEN.read_text(encoding="utf-8"))
    case = next(
        value for value in fixture["cases"] if value["case_id"] == "very_long_query_0"
    )
    transformers = pytest.importorskip("transformers")
    tokenizer = _load_complete_pinned_tokenizer(fixture, transformers)
    mutated = tokenizer(
        case["query"],
        build_document(case["title"], case["text"]),
        truncation="longest_first",
        max_length=fixture["max_length"],
        add_special_tokens=True,
    )
    assert _hashes(dict(mutated))["input_ids_sha256"] != case["input_ids_sha256"]
