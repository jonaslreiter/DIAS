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
    """Geometric decay from `v0`, reaching `vT` at step `T` and continuing past it.

    ``T = 0`` is the degenerate case: the decay is instantaneous, so the value
    is 0 from the first step. That is the limit of the expression below as
    T -> 0+, written out because it would otherwise divide by zero.
    """

    v0: float = 1.0
    vT: float = 0.1
    T: int = 500

    def __call__(self, step: int) -> float:
        if self.T == 0:
            return 0.0
        return self.v0 * (self.vT / self.v0) ** (step / self.T)


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
