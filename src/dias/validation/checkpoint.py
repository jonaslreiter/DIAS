"""Loading a trained run: its model, data settings and domain metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from dias.models.dias import DIAS
from dias.schema import DomainSchema
from dias.schema.export import learned_domain


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


def resolve_checkpoint(checkpoint_dir: Path, name: str = "final") -> Path:
    """The newest ``<name>-epoch*.ckpt`` in `checkpoint_dir`: ``final`` is the model at the end of training, ``best`` the one with the lowest validation loss."""
    matches = list(checkpoint_dir.glob(f"{name}-epoch*.ckpt"))
    if not matches:
        raise FileNotFoundError(
            f"No {name}-epoch*.ckpt found in {checkpoint_dir} (run did not start?)"
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_run(model_id: str, logs_path: str, checkpoint: str = "final") -> TrainedRun:
    """Load the model and domain metadata of one run under `logs_path`, from its `checkpoint` (see `resolve_checkpoint`)."""
    checkpoint_path = resolve_checkpoint(
        Path(logs_path) / "dias" / model_id / "checkpoints", checkpoint
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    hparams = {
        key: value
        for key, value in checkpoint["datamodule_hyper_parameters"].items()
        if not key.startswith("_")
    }
    # Relative to the directory the run was launched from.
    for key in ("domain_path", "problem_path"):
        hparams[key] = str(Path(hparams[key]).resolve())

    model = DIAS.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    model.to("cpu")
    model.eval()

    return TrainedRun(
        checkpoint_path=str(checkpoint_path),
        model=model,
        datamodule_hparams=hparams,
        schema=DomainSchema.from_run(hparams, model.hparams["relation_arities"]),
    )


def learned_pddl(run: TrainedRun, threshold: float | None = None) -> tuple[str, float]:
    """The run's learned domain and its derivation threshold: the domain written beside the checkpoint, or derived anew at `threshold` if given."""
    written = Path(run.checkpoint_path).with_suffix(".pddl")
    if threshold is None and written.exists():
        return written.read_text(), run.model.hparams.derivation_threshold
    if threshold is None:
        threshold = run.model.hparams.derivation_threshold
    return (
        learned_domain(run.model, run.schema, run.datamodule_hparams, threshold),
        threshold,
    )
