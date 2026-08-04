"""Resolution and executable checks for the official ``trec_eval`` binary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
import shutil


TREC_EVAL_ENVIRONMENT_VARIABLE = "TREC_EVAL_PATH"


def _as_nonempty(value: str | Path | None) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def resolve_trec_eval_executable(
    *,
    cli_path: str | Path | None = None,
    configured_path: str | Path | None = None,
    repository_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> Path:
    """Resolve ``trec_eval`` by CLI, config, environment, then ``PATH``."""

    env = os.environ if environment is None else environment
    path_lookup = shutil.which if which is None else which
    candidates = (
        ("CLI --trec-eval-path", _as_nonempty(cli_path)),
        ("configuration", _as_nonempty(configured_path)),
        (
            TREC_EVAL_ENVIRONMENT_VARIABLE,
            _as_nonempty(env.get(TREC_EVAL_ENVIRONMENT_VARIABLE)),
        ),
    )
    source = "PATH"
    value: str | None = None
    for candidate_source, candidate_value in candidates:
        if candidate_value is not None:
            source = candidate_source
            value = candidate_value
            break

    if value is None:
        located = path_lookup("trec_eval")
        if located is None:
            raise ValueError(
                "trec_eval не найден: задайте --trec-eval-path, настройку "
                "trec_eval_executable, переменную TREC_EVAL_PATH или добавьте "
                "исполняемый файл в PATH"
            )
        executable = Path(located).resolve()
    else:
        candidate = Path(value).expanduser()
        looks_like_path = candidate.is_absolute() or "/" in value or "\\" in value
        if looks_like_path:
            if not candidate.is_absolute():
                if source in {"CLI --trec-eval-path", "configuration"}:
                    base = (
                        Path(repository_root).resolve()
                        if repository_root is not None
                        else Path.cwd().resolve()
                    )
                else:
                    base = Path.cwd().resolve()
                candidate = base / candidate
            executable = candidate.resolve()
        else:
            located = path_lookup(value)
            if located is None:
                raise ValueError(
                    f"trec_eval из источника {source} не найден в PATH: {value}"
                )
            executable = Path(located).resolve()

    if not executable.exists():
        raise ValueError(
            f"trec_eval из источника {source} не существует: {executable}"
        )
    if not executable.is_file():
        raise ValueError(
            f"trec_eval из источника {source} не является обычным файлом: "
            f"{executable}"
        )
    if not os.access(executable, os.X_OK):
        raise ValueError(
            f"trec_eval из источника {source} не является исполняемым файлом: "
            f"{executable}"
        )
    return executable
