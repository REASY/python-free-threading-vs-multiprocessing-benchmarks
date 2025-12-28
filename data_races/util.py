from __future__ import annotations

import os
from typing import Callable, TypeVar

T = TypeVar("T")


def env_parse(name: str, default: T, *, parse: Callable[[str], T], type_name: str) -> T:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return parse(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a {type_name}, got {raw!r}") from exc


def env_int(name: str, default: int) -> int:
    return env_parse(
        name,
        default,
        parse=lambda s: int(s.replace("_", "")),
        type_name="int",
    )


def env_float(name: str, default: float) -> float:
    return env_parse(
        name,
        default,
        parse=float,
        type_name="float",
    )


def env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"
