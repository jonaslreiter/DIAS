"""Loading a trained run: its model, its datamodule settings, its domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from dias.models.dias import DIAS
from dias.schema import DomainSchema
from dias.schema.export import learned_domain

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
        # In the order the model was trained on, which the domain file alone no longer fixes for every run.
        schema=DomainSchema.from_run(hparams, model.hparams["relation_arities"]),
    )


def learned_pddl(run: TrainedRun, threshold: float | None = None) -> tuple[str, float]:
    """The run's learned domain and the threshold it was binarized at: the one written beside its checkpoint, unless a `threshold` is given or none was written, which extracts it anew."""
    written = Path(run.checkpoint_path).with_suffix(".pddl")
    if threshold is None and written.exists():
        return written.read_text(), run.model.hparams.schema_threshold
    if threshold is None:
        threshold = run.model.hparams.schema_threshold
    return (
        learned_domain(run.model, run.schema, run.datamodule_hparams, threshold),
        threshold,
    )
