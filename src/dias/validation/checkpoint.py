"""Loading a trained run: its model, its datamodule settings, its domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from dias.models.dias import DIAS
from dias.schema import DomainSchema
from dias.schema.domain import EQUALITY_BEFORE_ARGUMENT

# Sampling parameters PddlGraphDataModule no longer accepts; dropped so older checkpoints still load.
RETIRED_HPARAMS = frozenset(
    {
        "method",
        "num_val",
        "num_test",
        "balanced_min_per_label",
        "balanced_max_steps",
        "inapplicable_actions_ratio",
    }
)


@dataclass(frozen=True)
class TrainedRun:
    """One finished training run, read back for validation."""

    checkpoint_path: str
    model: DIAS
    datamodule_hparams: dict
    schema: DomainSchema

    @property
    def domain_path(self) -> str:
        return self.datamodule_hparams["domain_path"]


def resolve_best_checkpoint(checkpoint_dir: Path) -> Path:
    """The ``best-epoch*.ckpt`` most recently written in `checkpoint_dir`.

    Re-running a seed reuses the directory and Lightning keeps the old file under a ``-v1`` suffix, so mtime picks the current attempt rather than an arbitrary glob hit.
    """
    matches = list(checkpoint_dir.glob("best-epoch*.ckpt"))
    if not matches:
        raise FileNotFoundError(
            f"No best-epoch*.ckpt found in {checkpoint_dir} (run did not finish?)"
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_run(model_id: str, logs_path: str) -> TrainedRun:
    """Load the model and domain metadata of one run under `logs_path`."""
    checkpoint_path = resolve_best_checkpoint(
        Path(logs_path) / "dias" / model_id / "checkpoints"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    hparams = {
        key: value
        for key, value in checkpoint["datamodule_hyper_parameters"].items()
        if not key.startswith("_") and key not in RETIRED_HPARAMS
    }
    # Paths in a checkpoint are relative to the repository root, where the run was launched.
    for key in ("domain_path", "problem_path", "pt_file"):
        if hparams.get(key) is not None:
            hparams[key] = str(Path(hparams[key]).resolve())

    model = DIAS.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    model.to("cpu")
    model.eval()

    return TrainedRun(
        checkpoint_path=str(checkpoint_path),
        model=model,
        datamodule_hparams=hparams,
        schema=DomainSchema.from_pddl(
            hparams["domain_path"],
            hparams["problem_path"],
            include_statics=hparams.get("include_statics", False),
            include_equality=hparams.get("include_equality", EQUALITY_BEFORE_ARGUMENT),
            default_statics=hparams.get("default_statics"),
            action_arguments=hparams.get("action_arguments", False),
            strips_plus_actions=hparams.get("strips_plus_actions", False),
        ),
    )
