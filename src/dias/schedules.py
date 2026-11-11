"""Schedules of a value over training, as functions of the global step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from dias.hparams import from_mapping


@dataclass(frozen=True)
class Constant:
    value: float = 1.0

    def __call__(self, step: int) -> float:
        return self.value


@dataclass(frozen=True)
class Exponential:
    """``floor + (start - floor) * factor ** (step / every)``, and `floor` from the start if ``every = 0``."""

    start: float = 1.0
    floor: float = 0.0
    factor: float = 0.1
    every: int = 500

    def __post_init__(self):
        if self.start <= self.floor:
            raise ValueError(
                f"Expected start above floor, got start={self.start}, floor={self.floor}"
            )
        if self.factor <= 0:
            raise ValueError(f"Expected a positive factor, got {self.factor}")

    def __call__(self, step: int) -> float:
        if self.every == 0:
            return self.floor
        return self.floor + (self.start - self.floor) * self.factor ** (
            step / self.every
        )


@dataclass(frozen=True)
class PiecewiseLinear:
    """Linear between `keypoints`, constant outside them."""

    keypoints: dict[int, float] = field(default_factory=lambda: {0: 1.0})

    def __call__(self, step: int) -> float:
        steps = sorted(self.keypoints)
        if step <= steps[0]:
            return self.keypoints[steps[0]]
        if step >= steps[-1]:
            return self.keypoints[steps[-1]]
        after = next(s for s in steps if s > step)
        before = steps[steps.index(after) - 1]
        fraction = (step - before) / (after - before)
        return self.keypoints[before] + fraction * (
            self.keypoints[after] - self.keypoints[before]
        )


Schedule = Constant | Exponential | PiecewiseLinear


def as_schedule(value: "Schedule | Mapping[str, object] | None") -> Schedule:
    """A schedule, given as one, as its mapping, or as None for a constant 1."""
    if value is None:
        return Constant(1.0)
    return from_mapping(value, (Constant, Exponential, PiecewiseLinear))
