"""Dataclass-typed arguments stored in hyper-parameters as the ``class_path`` / ``init_args`` mapping of a config file, since a checkpoint must hold primitives only."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence, TypeVar

T = TypeVar("T")


def to_mapping(value) -> dict:
    """`value` as a ``class_path`` / ``init_args`` mapping."""
    return {
        "class_path": f"{type(value).__module__}.{type(value).__name__}",
        "init_args": asdict(value),
    }


def from_mapping(value: T | Mapping[str, object], options: Sequence[type]) -> T:
    """An instance of one of `options`, given as one or as its mapping."""
    if isinstance(value, tuple(options)):
        return value
    by_name = {option.__name__: option for option in options}
    name = str(value["class_path"]).rsplit(".", 1)[-1]
    if name not in by_name:
        raise ValueError(
            f"Expected one of {sorted(by_name)}, got {value['class_path']!r}"
        )
    return by_name[name](**dict(value.get("init_args", {})))
