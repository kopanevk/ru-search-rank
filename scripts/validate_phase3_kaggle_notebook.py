#!/usr/bin/env python3
"""Совместимый вход для проверки канонического notebook этапа 3."""

from __future__ import annotations

import sys

from validate_phase3_notebook import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, SyntaxError) as exc:
        print(f"Проверка production notebook не пройдена: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
