"""Helpers the test modules share."""

import tempfile
import unittest
from pathlib import Path


def temp_dir(case: unittest.TestCase) -> Path:
    """A temporary directory, removed when `case` finishes. Not `TestCase.enterContext`, which is 3.11+ while this package supports 3.10."""
    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    return Path(directory.name)


MAX_STEPS = 10000


def write_run(log_dir: Path, run_id: str, step=MAX_STEPS, max_steps=MAX_STEPS) -> None:
    """A run directory as training leaves it: the config it resolved to, and both checkpoints at `step`."""
    run_dir = log_dir / "dias" / run_id
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    run_dir.joinpath("config.yaml").write_text(f"trainer:\n  max_steps: {max_steps}\n")
    (checkpoints / f"final-epoch=0-step={step}.ckpt").touch()
    (checkpoints / f"best-epoch=0-step={step - 50}.ckpt").touch()
