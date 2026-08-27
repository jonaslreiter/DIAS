"""Bringing an old checkpoint layout up to the current format.

Driven by the version stamp, not by a failed load: Lightning drops hyper-parameters it does not recognise instead of raising. The rules read and write plain dictionaries, never the live classes.
"""

from __future__ import annotations

import re

CHECKPOINT_VERSION = 1
VERSION_KEY = "dias_checkpoint_version"
# An absent stamp: written before the stamp existed.
UNVERSIONED = 0

EXPONENTIAL = "dias.schedules.Exponential"
CONSTANT = "dias.schedules.Constant"
PIECEWISE_LINEAR = "dias.schedules.PiecewiseLinear"

# The unversioned layout kept the whole schema flat on the prediction module and declared the encoder's layers one attribute at a time.
_SCHEMA_RENAMES = {
    "graph_edit.slot_encodings": "graph_edit.selector.slot_encodings",
    "graph_edit.slot_weight_logits": "graph_edit.selector.slot_weight_logits",
    "graph_edit.preconditions": "graph_edit.preconditions.logits",
    "graph_edit.effects": "graph_edit.effects.logits",
}
_ENCODER_LAYER = re.compile(r"^dynamics_model\.(conv|ln)(\d+)\.")
_ENCODER_LISTS = {"conv": "convs", "ln": "norms"}
# Encoder convolutions after the rename, for locating the last one.
_ENCODER_CONV = re.compile(r"^state_encoder\.convs\.(\d+)\.")

# Derived from the constructor arguments, and rebuilt from them on load.
_DROPPED_KEYS = ("graph_edit.action_arities_t", "_action_rel_lookup")

# Hyper-parameters the constructor no longer takes; neither ever affected any weights.
_DROPPED_HPARAMS = ("parameter_plotting", "batch_size")


def _spec(class_path: str, **init_args) -> dict:
    return {"class_path": class_path, "init_args": init_args}


def _drop_final_encoder_bias(state: dict) -> None:
    """Remove the last convolution's bias, which no gradient ever reached (see `RelationalEncoder`). Which one is last follows `encoder_hidden_dims`, so the index is read off the state dict."""
    indices = [
        int(match.group(1))
        for key in state
        if (match := _ENCODER_CONV.match(key)) is not None
    ]
    if indices:
        state.pop(f"state_encoder.convs.{max(indices)}.bias", None)


def _migrate_state_dict(state: dict) -> None:
    """Rename the unversioned parameters and drop what is no longer stored."""
    if any(key.startswith("graph_edit.effects_mlp.") for key in state):
        raise ValueError(
            "This MLP-baseline checkpoint predates the activations between the effect MLP's linear layers, which makes its weights mean something different than they do now; there is nothing to migrate them to. Retrain the run."
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
    # After the rename, so it sees the current spelling either way.
    _drop_final_encoder_bias(state)


def _migrate_tau_schedule(hparams: dict) -> None:
    """Rebuild tau from the unversioned schedule dictionary and its offset.

    The offset was the asymptote, so it becomes the floor; the decay was anchored on its value at step T, so ``factor`` per ``every`` steps replaces it. Same trajectory.
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
    raise ValueError(f"Unknown schedule type {kind!r}")


def _from_unversioned(checkpoint: dict) -> None:
    """Bring the pre-versioning layout up to date. Finding nothing to change is not an error."""
    if checkpoint.get("state_dict") is not None:
        _migrate_state_dict(checkpoint["state_dict"])
    hparams = checkpoint.get("hyper_parameters")
    if hparams is not None:
        _migrate_tau_schedule(hparams)
        for key in _DROPPED_HPARAMS:
            hparams.pop(key, None)


def migrate(checkpoint: dict) -> dict:
    """Bring `checkpoint` up to the current version, in place."""
    version = checkpoint.get(VERSION_KEY, UNVERSIONED)
    if version > CHECKPOINT_VERSION:
        raise ValueError(
            f"This checkpoint was written by a newer version of dias (format {version}, this one reads {CHECKPOINT_VERSION})."
        )
    if version < CHECKPOINT_VERSION:
        _from_unversioned(checkpoint)
    checkpoint[VERSION_KEY] = CHECKPOINT_VERSION
    return checkpoint
