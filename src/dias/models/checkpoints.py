"""Reading checkpoints written by earlier versions of the model.

A checkpoint records the version it was written with; its absence means
version 1, which is what the paper's runs are. Loading tries the stock loader
first, so current checkpoints take the fast path, and only a failure triggers a
migration and one retry. The file on disk is never modified.
"""

from __future__ import annotations

from typing import Callable

from dias.schedules import (
    Constant,
    Exponential,
    PiecewiseLinear,
    as_schedule,
    schedule_config,
)

CHECKPOINT_VERSION = 3
VERSION_KEY = "dias_checkpoint_version"


def _legacy_schedule(spec: dict | None):
    """v1's ``{"type": ..., ...}`` schedule dict as a schedule object."""
    if not spec:
        return Constant(1.0)
    kind = spec.get("type")
    if kind == "exponential":
        return Exponential(v0=spec["v0"], vT=spec["vT"], T=spec["T"])
    if kind == "linear":
        keypoints = {int(k): float(v) for k, v in spec["keypoints"].items()}
        # A single keypoint is a constant, which is how v1 spelled one.
        return (
            Constant(next(iter(keypoints.values())))
            if len(keypoints) == 1
            else (PiecewiseLinear(keypoints))
        )
    raise ValueError(f"Unknown v1 schedule type {kind!r}")


def _to_v2(checkpoint: dict) -> bool:
    """v1 held a dict of named loss-weight schedules; v2 takes one tau schedule."""
    hparams = checkpoint.get("hyper_parameters")
    if hparams is None or "loss_weight_schedule" not in hparams:
        return False
    legacy = hparams.pop("loss_weight_schedule") or {}
    hparams["gmean_temperature"] = schedule_config(
        _legacy_schedule(legacy.get("gmean_temperature"))
    )
    return True


def _to_v3(checkpoint: dict) -> bool:
    """v2 added `gmean_offset` to tau after the schedule; v3 folds it in.

    The offset was an asymptote the schedule had to be pre-scaled around, so
    v0 and vT meant "the decaying part" rather than tau itself. Shifting both
    endpoints up by the offset and recording it as `floor` reproduces the same
    trajectory exactly.
    """
    hparams = checkpoint.get("hyper_parameters")
    if hparams is None or "gmean_offset" not in hparams:
        return False
    offset = float(hparams.pop("gmean_offset") or 0.0)
    schedule = as_schedule(hparams.get("gmean_temperature"))
    if offset:
        if isinstance(schedule, Exponential):
            schedule = Exponential(
                v0=schedule.v0 + offset,
                vT=schedule.vT + offset,
                T=schedule.T,
                floor=offset,
            )
        elif isinstance(schedule, Constant):
            schedule = Constant(schedule.value + offset)
        else:
            schedule = PiecewiseLinear(
                {step: value + offset for step, value in schedule.keypoints.items()}
            )
    hparams["gmean_temperature"] = schedule_config(schedule)
    return True


MIGRATIONS: tuple[tuple[int, str, Callable[[dict], bool]], ...] = (
    (2, "one typed tau schedule instead of a dict of named schedules", _to_v2),
    (3, "the tau offset folded into the schedule as its floor", _to_v3),
)


def migrate(checkpoint: dict) -> dict:
    """Bring `checkpoint` up to the current version, in place.

    A rule that finds nothing to change is not an error: not every older
    checkpoint contains everything a later version renamed.
    """
    version = checkpoint.get(VERSION_KEY, 1)
    for target, _description, step in MIGRATIONS:
        if target > version:
            step(checkpoint)
    checkpoint[VERSION_KEY] = CHECKPOINT_VERSION
    return checkpoint
