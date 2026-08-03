"""Shared, byte-stable query/document encoding for training and scoring.

Phase 3 deliberately has one implementation of the cross-encoder input
contract.  The function does not normalize text: whitespace (including NBSP),
case, Unicode code points, and embedded newlines are passed to the pinned
tokenizer exactly as supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_document(title: Any, text: Any) -> str:
    """Return the exact document string used by the cross-encoder."""

    if text is None or not isinstance(text, str):
        raise ValueError("passage text must be a string")
    if title is not None and not isinstance(title, str):
        raise ValueError("passage title must be a string or null")
    return f"{title}\n{text}" if title and title.strip() else text


def _plain_encoding(value: Mapping[str, Any]) -> dict[str, list[int]]:
    """Convert tokenizer containers to unbatched plain integer lists."""

    result: dict[str, list[int]] = {}
    for name, raw in value.items():
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if raw and isinstance(raw[0], list):
            if len(raw) != 1:
                raise ValueError("pair tokenizer unexpectedly returned a batch")
            raw = raw[0]
        result[str(name)] = [int(token) for token in raw]
    if "input_ids" not in result or not result["input_ids"]:
        raise ValueError("tokenizer returned no input_ids")
    if "attention_mask" not in result:
        raise ValueError("tokenizer returned no attention_mask")
    if len(result["attention_mask"]) != len(result["input_ids"]):
        raise ValueError("tokenizer returned mismatched input_ids and attention_mask")
    return result


def encode_pair(
    tokenizer: Any,
    query: str,
    document: str,
    max_length: int = 320,
) -> dict[str, Any]:
    """Encode ``(query, document)`` with exact only-second truncation semantics.

    Token accounting is based on a first, untruncated encoding.  A second
    tokenizer call is made only when that encoding exceeds ``max_length``.
    The returned mapping contains model inputs plus ``tokens_before``,
    ``tokens_after``, and ``truncated``.
    """

    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if not isinstance(document, str):
        raise ValueError("document must be a string")
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    encoded_full = _plain_encoding(
        tokenizer(
            query,
            document,
            truncation=False,
            add_special_tokens=True,
        )
    )
    tokens_before = len(encoded_full["input_ids"])
    if tokens_before <= max_length:
        encoded = encoded_full
        truncated = False
    else:
        encoded = _plain_encoding(
            tokenizer(
                query,
                document,
                truncation="only_second",
                max_length=max_length,
                add_special_tokens=True,
            )
        )
        truncated = True

    tokens_after = len(encoded["input_ids"])
    if tokens_before <= 0 or tokens_after <= 0 or tokens_after > max_length:
        raise ValueError(
            "invalid pair token accounting: "
            f"before={tokens_before}, after={tokens_after}, max_length={max_length}"
        )
    if truncated and tokens_after >= tokens_before:
        raise ValueError("only_second truncation did not shorten an overlength pair")

    result: dict[str, Any] = {
        name: values
        for name, values in encoded.items()
        if name in {"input_ids", "attention_mask", "token_type_ids"}
    }
    result.update(
        {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "truncated": truncated,
        }
    )
    return result


__all__ = ["build_document", "encode_pair"]
