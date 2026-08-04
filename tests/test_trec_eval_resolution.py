from __future__ import annotations

from pathlib import Path

import pytest

from rusearchrank.trec_eval import resolve_trec_eval_executable


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


def test_resolution_precedence_is_cli_config_environment_path(tmp_path: Path) -> None:
    cli = _executable(tmp_path / "cli" / "trec_eval")
    configured = _executable(tmp_path / "configured" / "trec_eval")
    environment = _executable(tmp_path / "environment" / "trec_eval")
    path_binary = _executable(tmp_path / "path" / "trec_eval")

    assert resolve_trec_eval_executable(
        cli_path=cli,
        configured_path=configured,
        environment={"TREC_EVAL_PATH": str(environment)},
        which=lambda _: str(path_binary),
    ) == cli
    assert resolve_trec_eval_executable(
        configured_path=configured,
        environment={"TREC_EVAL_PATH": str(environment)},
        which=lambda _: str(path_binary),
    ) == configured
    assert resolve_trec_eval_executable(
        environment={"TREC_EVAL_PATH": str(environment)},
        which=lambda _: str(path_binary),
    ) == environment
    assert resolve_trec_eval_executable(
        environment={}, which=lambda _: str(path_binary)
    ) == path_binary


def test_relative_config_path_is_resolved_from_repository_root(tmp_path: Path) -> None:
    executable = _executable(tmp_path / "tools" / "trec_eval")
    assert resolve_trec_eval_executable(
        configured_path="tools/trec_eval",
        repository_root=tmp_path,
        environment={},
        which=lambda _: None,
    ) == executable


@pytest.mark.parametrize("kind", ("missing", "directory", "not_executable"))
def test_invalid_explicit_path_fails_before_path_fallback(
    tmp_path: Path, kind: str
) -> None:
    candidate = tmp_path / kind / "trec_eval"
    if kind == "directory":
        candidate.mkdir(parents=True)
    elif kind == "not_executable":
        candidate.parent.mkdir(parents=True)
        candidate.write_text("not executable\n", encoding="utf-8")
        candidate.chmod(0o644)
    with pytest.raises(ValueError, match="trec_eval из источника configuration"):
        resolve_trec_eval_executable(
            configured_path=candidate,
            environment={},
            which=lambda _: "/should/not/be/used",
        )


def test_clear_error_when_all_sources_are_absent() -> None:
    with pytest.raises(ValueError, match="--trec-eval-path.*TREC_EVAL_PATH.*PATH"):
        resolve_trec_eval_executable(environment={}, which=lambda _: None)
