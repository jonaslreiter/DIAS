"""Reading checkpoints written in an earlier format.

A checkpoint stores the format it was written in under `VERSION_KEY`. Format 2 (the paper's runs) differs from the current one only in the names of some hyper-parameters and in two arguments that no longer exist.
"""

from __future__ import annotations

CHECKPOINT_VERSION = 3
VERSION_KEY = "dias_checkpoint_version"
OLDEST_READABLE = 2

RENAMED_HPARAMS = {
    "learn_preconditions": "gate_preconditions",
    "random_node_embeddings": "random_node_features",
    "state_dependent_effects": "mlp_effects",
    "schema_threshold": "derivation_threshold",
}
# Arguments that no longer exist, with the value the current model has fixed.
RETIRED_HPARAMS = {"successor_relations": True, "pcgrad_norm_cap": True}


def _from_version_2(hparams: dict) -> None:
    for old, new in RENAMED_HPARAMS.items():
        if old in hparams:
            hparams[new] = hparams.pop(old)
    for name, value in RETIRED_HPARAMS.items():
        if hparams.pop(name, value) != value:
            raise ValueError(
                f"A checkpoint with {name} other than {value} cannot be read."
            )


def migrate(checkpoint: dict) -> dict:
    """`checkpoint`, in place, in the current format."""
    version = checkpoint.get(VERSION_KEY)
    if version is None or not OLDEST_READABLE <= version <= CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint format {version} cannot be read; this version reads formats {OLDEST_READABLE} to {CHECKPOINT_VERSION}."
        )
    if version < 3:
        _from_version_2(checkpoint.get("hyper_parameters") or {})
    checkpoint[VERSION_KEY] = CHECKPOINT_VERSION
    return checkpoint
