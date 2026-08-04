from __future__ import annotations

import pytest

from rusearchrank.cli import build_parser


def test_cli_help_uses_russian_service_labels() -> None:
    help_text = build_parser().format_help()

    assert "позиционные аргументы:" in help_text
    assert "параметры:" in help_text
    assert "показать эту справку и выйти" in help_text
    assert "positional arguments:" not in help_text
    assert "show this help message" not in help_text


def test_cli_translates_invalid_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        build_parser().parse_args(["unknown-command"])

    error = capsys.readouterr().err
    assert "ошибка: команда: недопустимое значение 'unknown-command'" in error
    assert "допустимые значения:" in error
    assert "invalid choice" not in error
    assert "choose from" not in error


def test_cli_translates_missing_required_parameter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        build_parser().parse_args(["finetune"])

    error = capsys.readouterr().err
    assert "не указаны обязательные параметры: --run-id" in error
    assert "arguments are required" not in error
