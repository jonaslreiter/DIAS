"""Values that change over the course of training.

Each schedule is a pure function of the global step: nothing here holds state,
so a schedule can be queried at any step, in any order, without being "advanced"
first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Constant:
    value: float = 1.0

    def __call__(self, step: int) -> float:
        return self.value


@dataclass(frozen=True)
class Exponential:
    """Decay from `start` toward `floor`, by `factor` every `every` steps.

    ``floor + (start - floor) * factor ** (step / every)``

    The rate and the asymptote are independent: raising `floor` does not change
    how fast the value falls. That is what the noise experiments need -- the
    same decay as the clean setup, but bottoming out at 0.1 rather than 0,
    because the exact precondition product is too strict a check under noise.

    ``every = 0`` is the degenerate case: decay is instantaneous, so the value
    is `floor` from the first step. That is the limit as every -> 0+, written
    out because the general form would divide by zero.
    """

    start: float = 1.0
    floor: float = 0.0
    factor: float = 0.1
    every: int = 500

    def __post_init__(self):
        # The curve approaches `floor` without crossing it.
        if self.start <= self.floor:
            raise ValueError(
                f"Expected start above floor, got start={self.start}, "
                f"floor={self.floor}"
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
    """Linear interpolation between `keypoints`, held flat outside their range."""

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


_BY_NAME = {cls.__name__: cls for cls in (Constant, Exponential, PiecewiseLinear)}


def as_schedule(value: "Schedule | Mapping[str, object] | None") -> Schedule:
    """Accept a schedule, the mapping a checkpoint stores it as, or None.

    None means a constant 1.0. Callers must default to None rather than to a
    schedule instance: jsonargparse merges a configured schedule's fields onto
    the default's instead of replacing it, and a Constant default would turn an
    Exponential in the config into a mixture that matches neither.
    """
    if value is None:
        return Constant(1.0)
    if isinstance(value, tuple(_BY_NAME.values())):
        return value
    name = str(value["class_path"]).rsplit(".", 1)[-1]
    if name not in _BY_NAME:
        raise ValueError(f"Unknown schedule {value['class_path']!r}")
    return _BY_NAME[name](**dict(value.get("init_args", {})))


def schedule_config(schedule: Schedule) -> dict:
    """The mapping form, for hyperparameters.

    Checkpoints must hold primitives: torch.load defaults to weights_only=True
    since 2.6, which refuses to unpickle a dataclass and makes the whole file
    unloadable -- not just this field.
    """
    return {
        "class_path": f"{__name__}.{type(schedule).__name__}",
        "init_args": asdict(schedule),
    }
