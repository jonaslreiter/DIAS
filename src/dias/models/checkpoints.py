"""Reading checkpoints written before the format was versioned.

The current format is version 1. Everything older is the single layout that
came before it -- what the paper's runs were written with, which carried no
stamp at all -- so an absent stamp means "older than 1" and one rule brings it
up to date. The refactor that produced version 1 passed through several
intermediate layouts, but none of them left the branch, so there is no format
between the two and no rule for one.

**Version 1 is still being written.** Until this work reaches main, another
change to the layout belongs in `_from_unversioned` beside the rest; nothing
carries a version-1 stamp that anyone else has to keep reading. After the
merge that stops being true, and a change to what the model stores needs
version 2 and a rule of its own -- editing this one would change what an
already-published checkpoint means.

Loading is driven by the stamp rather than by a failed load, because Lightning
drops hyper-parameters it does not recognise instead of raising: a renamed one
would quietly fall back to its default and change what an archived run means.

The rules read and write plain dictionaries, never the live schedule or model
classes. A rule that built today's objects would break as soon as those classes
changed again, which is precisely the situation it exists to handle.
"""

from __future__ import annotations

import re

CHECKPOINT_VERSION = 1
VERSION_KEY = "dias_checkpoint_version"
# What an absent stamp means: written before the stamp existed.
UNVERSIONED = 0

EXPONENTIAL = "dias.schedules.Exponential"
CONSTANT = "dias.schedules.Constant"
PIECEWISE_LINEAR = "dias.schedules.PiecewiseLinear"

# The unversioned layout kept the whole schema flat on the prediction module
# and declared the encoder's layers one attribute at a time.
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

# Hyper-parameters the constructor no longer takes. Plotting is a callback's
# business now, and the batch size was the datamodule's all along -- the model
# stored it and never read it. Neither affected any weights.
_DROPPED_HPARAMS = ("parameter_plotting", "batch_size")


def _spec(class_path: str, **init_args) -> dict:
    return {"class_path": class_path, "init_args": init_args}


def _drop_final_encoder_bias(state: dict) -> None:
    """Remove the last convolution's bias, which the loss never saw.

    Nothing is normalised after that convolution, so its bias added the same
    vector to every object embedding; the slot scores are a plain inner
    product, so it reached the assignment as a per-slot constant, which
    Sinkhorn's row scaling removes exactly. Its gradient was therefore float32
    round-off, which Adam normalised into a random walk. Discarding a learned
    value is safe precisely because no learning ever reached it.

    Which convolution is last follows `encoder_hidden_dims`, so the index is
    read off the state dict rather than assumed.
    """
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
    # After the rename, so it sees the current spelling either way.
    _drop_final_encoder_bias(state)


def _migrate_tau_schedule(hparams: dict) -> None:
    """Rebuild tau from the unversioned schedule dictionary and its offset.

    That layout held a collection of named loss-weight schedules that only ever
    had one member, and added `gmean_offset` to whatever it returned. The
    offset was an asymptote the schedule had to be pre-scaled around, so it
    becomes the floor; the decay itself was anchored on its value at step T
    rather than stated as a rate, so ``factor`` per ``every`` steps replaces it.
    The trajectory is identical, which is what the noise runs depend on.
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
    """Bring the layout the paper's runs were written with up to date.

    Finding nothing to change is not an error: not every old checkpoint
    contains everything version 1 renamed.
    """
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
            f"This checkpoint was written by a newer version of dias "
            f"(format {version}, this one reads {CHECKPOINT_VERSION})."
        )
    if version < CHECKPOINT_VERSION:
        _from_unversioned(checkpoint)
    checkpoint[VERSION_KEY] = CHECKPOINT_VERSION
    return checkpoint
