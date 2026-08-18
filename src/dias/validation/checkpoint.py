"""Loading a trained run: its model, its datamodule settings, its domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from dias.models.dias import DIAS
from dias.schema import DomainSchema

# Sampling parameters that PddlGraphDataModule no longer accepts. Dropping them
# is what lets checkpoints from before the sampling refactor still load.
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
    """Return the ``best-epoch*.ckpt`` most recently written in `checkpoint_dir`.

    Re-running a seed (``launch.py --seeds``) reuses the run id and therefore
    the checkpoint directory, and Lightning does not delete the previous
    attempt's file: it side-steps the name clash with a ``-v1`` suffix.
    Selecting by mtime picks the current attempt; glob order is arbitrary and
    could silently validate the stale one.
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
    # Paths in a checkpoint were relative to the directory the run was launched
    # from, which is the repository root.
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
            default_statics=hparams.get("default_statics"),
            action_arguments=hparams.get("action_arguments", False),
            strips_plus_actions=hparams.get("strips_plus_actions", False),
        ),
    )
