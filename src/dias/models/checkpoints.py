"""Reading checkpoints written by earlier versions of the model.

A checkpoint records the version it was written with; its absence means
version 1, which is what the paper's runs are. Loading is driven by that stamp
rather than by a failed load, because Lightning drops hyper-parameters it does
not recognise instead of raising -- so a renamed one would quietly fall back to
its default and change what an archived run means.

Each rule reads and writes plain dictionaries, never the live schedule or model
classes. A rule that built today's objects would break as soon as those classes
changed again, which is precisely the situation it exists to handle.
"""

from __future__ import annotations

from typing import Callable

CHECKPOINT_VERSION = 4
VERSION_KEY = "dias_checkpoint_version"

EXPONENTIAL = "dias.schedules.Exponential"
CONSTANT = "dias.schedules.Constant"
PIECEWISE_LINEAR = "dias.schedules.PiecewiseLinear"


def _spec(class_path: str, **init_args) -> dict:
    return {"class_path": class_path, "init_args": init_args}


def _to_v2(checkpoint: dict) -> bool:
    """v1 held a dict of named loss-weight schedules; v2 takes one tau schedule."""
    hparams = checkpoint.get("hyper_parameters")
    if hparams is None or "loss_weight_schedule" not in hparams:
        return False
    legacy = (hparams.pop("loss_weight_schedule") or {}).get("gmean_temperature")
    hparams["gmean_temperature"] = _v1_schedule_spec(legacy)
    return True


def _v1_schedule_spec(legacy: dict | None) -> dict:
    if not legacy:
        return _spec(CONSTANT, value=1.0)
    kind = legacy.get("type")
    if kind == "exponential":
        return _spec(EXPONENTIAL, v0=legacy["v0"], vT=legacy["vT"], T=legacy["T"])
    if kind == "linear":
        keypoints = {int(k): float(v) for k, v in legacy["keypoints"].items()}
        if len(keypoints) == 1:
            return _spec(CONSTANT, value=next(iter(keypoints.values())))
        return _spec(PIECEWISE_LINEAR, keypoints=keypoints)
    raise ValueError(f"Unknown v1 schedule type {kind!r}")


def _to_v3(checkpoint: dict) -> bool:
    """v2 added `gmean_offset` to tau after the schedule; v3 folds it in.

    The offset was an asymptote the schedule had to be pre-scaled around, so
    shifting both endpoints up by it and recording it as the floor reproduces
    the same trajectory exactly.
    """
    hparams = checkpoint.get("hyper_parameters")
    if hparams is None or "gmean_offset" not in hparams:
        return False
    offset = float(hparams.pop("gmean_offset") or 0.0)
    spec = hparams.get("gmean_temperature") or _spec(CONSTANT, value=1.0)
    args = dict(spec["init_args"])
    if offset:
        if spec["class_path"] == EXPONENTIAL:
            args = {
                "v0": args["v0"] + offset,
                "vT": args["vT"] + offset,
                "T": args["T"],
                "floor": offset,
            }
        elif spec["class_path"] == CONSTANT:
            args = {"value": args["value"] + offset}
        else:
            args = {"keypoints": {s: v + offset for s, v in args["keypoints"].items()}}
    hparams["gmean_temperature"] = _spec(spec["class_path"], **args)
    return True


def _to_v4(checkpoint: dict) -> bool:
    """v3 anchored the decay on its value at step T; v4 states the rate directly.

    ``vT`` had to be recomputed whenever the floor moved, since it was the value
    at T rather than a rate. ``factor`` per ``every`` steps is independent of
    the floor and describes the same curve.
    """
    hparams = checkpoint.get("hyper_parameters")
    spec = (hparams or {}).get("gmean_temperature")
    if not spec or spec["class_path"] != EXPONENTIAL or "v0" not in spec["init_args"]:
        return False
    args = spec["init_args"]
    floor = args.get("floor", 0.0)
    span = args["v0"] - floor
    hparams["gmean_temperature"] = _spec(
        EXPONENTIAL,
        start=args["v0"],
        floor=floor,
        factor=(args["vT"] - floor) / span,
        every=args["T"],
    )
    return True


MIGRATIONS: tuple[tuple[int, str, Callable[[dict], bool]], ...] = (
    (2, "one typed tau schedule instead of a dict of named schedules", _to_v2),
    (3, "the tau offset folded into the schedule as its floor", _to_v3),
    (4, "the tau decay stated as a rate rather than a value at step T", _to_v4),
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
