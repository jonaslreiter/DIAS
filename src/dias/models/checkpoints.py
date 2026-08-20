"""Reading checkpoints written before the models/ split.

Version 1 is what the paper's runs are: everything written before the models/
split, which stamped nothing. Later versions each retire something a
constructor no longer accepts. The intermediate numbers the refactor passed
through never left the branch that produced them, so there is nothing to
migrate from and no rule for them.

Loading is driven by the stamp rather than by a failed load, because Lightning
drops hyper-parameters it does not recognise instead of raising: a renamed one
would quietly fall back to its default and change what an archived run means.

The rules read and write plain dictionaries, never the live schedule or model
classes. A rule that built today's objects would break as soon as those classes
changed again, which is precisely the situation it exists to handle.
"""

from __future__ import annotations

import re
from typing import Callable

CHECKPOINT_VERSION = 3
VERSION_KEY = "dias_checkpoint_version"

EXPONENTIAL = "dias.schedules.Exponential"
CONSTANT = "dias.schedules.Constant"
PIECEWISE_LINEAR = "dias.schedules.PiecewiseLinear"

# v1 kept the whole schema flat on the prediction module and declared the
# encoder's layers one attribute at a time.
_SCHEMA_RENAMES = {
    "graph_edit.slot_encodings": "graph_edit.selector.slot_encodings",
    "graph_edit.slot_weight_logits": "graph_edit.selector.slot_weight_logits",
    "graph_edit.preconditions": "graph_edit.preconditions.logits",
    "graph_edit.effects": "graph_edit.effects.logits",
}
_ENCODER_LAYER = re.compile(r"^dynamics_model\.(conv|ln)(\d+)\.")
_ENCODER_LISTS = {"conv": "convs", "ln": "norms"}

# Derived from the constructor arguments, and rebuilt from them on load.
_DROPPED_KEYS = ("graph_edit.action_arities_t", "_action_rel_lookup")


def _spec(class_path: str, **init_args) -> dict:
    return {"class_path": class_path, "init_args": init_args}


def _migrate_state_dict(state: dict) -> None:
    """Rename v1's parameters and drop what is no longer stored."""
    if any(key.startswith("graph_edit.effects_mlp.") for key in state):
        raise ValueError(
            "This MLP-baseline checkpoint predates the activations between the "
            "effect MLP's linear layers, which makes its weights mean something "
            "different than they do now; there is nothing to migrate them to. "
            "Retrain the run."
        )
    for old, new in _SCHEMA_RENAMES.items():
        if old in state:
            state[new] = state.pop(old)
    for key in [key for key in state if _ENCODER_LAYER.match(key)]:
        layer = _ENCODER_LAYER.match(key)
        kind, index = layer.group(1), int(layer.group(2))
        renamed = (
            f"state_encoder.{_ENCODER_LISTS[kind]}.{index - 1}.{key[layer.end():]}"
        )
        state[renamed] = state.pop(key)
    for key in _DROPPED_KEYS:
        state.pop(key, None)


def _migrate_tau_schedule(hparams: dict) -> None:
    """Rebuild tau from v1's schedule dictionary and its separate offset.

    v1 held a collection of named loss-weight schedules that only ever had one
    member, and added `gmean_offset` to whatever it returned. The offset was an
    asymptote the schedule had to be pre-scaled around, so it becomes the
    floor; the decay itself was anchored on its value at step T rather than
    stated as a rate, so ``factor`` per ``every`` steps replaces it. The
    trajectory is identical, which is what the noise runs depend on.
    """
    if "loss_weight_schedule" not in hparams and "gmean_offset" not in hparams:
        return
    legacy = (hparams.pop("loss_weight_schedule", None) or {}).get("gmean_temperature")
    offset = float(hparams.pop("gmean_offset", 0.0) or 0.0)

    if not legacy:
        hparams["gmean_temperature"] = _spec(CONSTANT, value=1.0 + offset)
        return

    kind = legacy.get("type")
    if kind == "exponential":
        hparams["gmean_temperature"] = _spec(
            EXPONENTIAL,
            start=legacy["v0"] + offset,
            floor=offset,
            factor=legacy["vT"] / legacy["v0"],
            every=legacy["T"],
        )
        return
    if kind == "linear":
        keypoints = {int(k): float(v) + offset for k, v in legacy["keypoints"].items()}
        hparams["gmean_temperature"] = (
            _spec(CONSTANT, value=next(iter(keypoints.values())))
            if len(keypoints) == 1
            else _spec(PIECEWISE_LINEAR, keypoints=keypoints)
        )
        return
    raise ValueError(f"Unknown v1 schedule type {kind!r}")


def _from_v1(checkpoint: dict) -> None:
    """Bring the layout the paper's runs were written with up to date."""
    if checkpoint.get("state_dict") is not None:
        _migrate_state_dict(checkpoint["state_dict"])
    if checkpoint.get("hyper_parameters") is not None:
        _migrate_tau_schedule(checkpoint["hyper_parameters"])


def _from_v2(checkpoint: dict) -> None:
    """Drop the plotting switches; reporting is a callback's business now."""
    (checkpoint.get("hyper_parameters") or {}).pop("parameter_plotting", None)


MIGRATIONS: tuple[tuple[int, str, Callable[[dict], None]], ...] = (
    (2, "the models/ split: one module per part of the schema", _from_v1),
    (3, "plotting left the model for a callback", _from_v2),
)


def migrate(checkpoint: dict) -> dict:
    """Bring `checkpoint` up to the current version, in place.

    A rule that finds nothing to change is not an error: not every older
    checkpoint contains everything a later version renamed.
    """
    version = checkpoint.get(VERSION_KEY, 1)
    if version > CHECKPOINT_VERSION:
        raise ValueError(
            f"This checkpoint was written by a newer version of dias "
            f"(format {version}, this one reads {CHECKPOINT_VERSION})."
        )
    for target, _description, apply in MIGRATIONS:
        if target > version:
            apply(checkpoint)
    checkpoint[VERSION_KEY] = CHECKPOINT_VERSION
    return checkpoint
