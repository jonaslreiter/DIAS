"""Dataclass-typed arguments that have to survive a checkpoint.

Hyper-parameters must be primitives: `torch.load` defaults to
``weights_only=True`` since torch 2.6 and refuses to unpickle a dataclass,
which makes the whole checkpoint unloadable rather than just that one field.
So an argument typed as one of a few dataclasses is stored as the same
``class_path`` / ``init_args`` mapping jsonargparse writes for it in a config
file, and rebuilt from that mapping on the way back in.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence, TypeVar

T = TypeVar("T")


def to_mapping(value) -> dict:
    """The primitive form of a dataclass-typed argument."""
    return {
        "class_path": f"{type(value).__module__}.{type(value).__name__}",
        "init_args": asdict(value),
    }


def from_mapping(value: T | Mapping[str, object], options: Sequence[type]) -> T:
    """Accept one of `options`, or the mapping a checkpoint stores it as."""
    if isinstance(value, tuple(options)):
        return value
    by_name = {option.__name__: option for option in options}
    name = str(value["class_path"]).rsplit(".", 1)[-1]
    if name not in by_name:
        raise ValueError(
            f"Expected one of {sorted(by_name)}, got {value['class_path']!r}"
        )
    return by_name[name](**dict(value.get("init_args", {})))
