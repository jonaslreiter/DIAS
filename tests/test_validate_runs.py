"""Tests for a seed's result JSON: its status, what `collect_results` reads of it, and resuming from it. A seed is one number per held-out problem, so a seed missing one is re-run, never averaged beside the others."""

import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import validate_runs
from collect_results import collect_local
from support import temp_dir
from validate_runs import _write_json, validate_run

PROBLEMS = ["test-1", "test-2", "test-3"]


def scored(tp=90, fp=10, fn=10):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
    }


class SeedStatusTest(unittest.TestCase):
    def setUp(self):
        self.logs = temp_dir(self)

    def status(self, per_problem, seed=0, **kwargs):
        _write_json(self.logs, "exp", "driverlog", seed, "run", per_problem, **kwargs)
        path = self.logs / "validation" / "exp" / "driverlog" / f"seed{seed}.json"
        return json.loads(path.read_text())["status"]

    def test_every_problem_scored_is_ok(self):
        self.assertEqual(self.status({p: scored() for p in PROBLEMS}), "ok")

    def test_a_failed_problem_makes_the_seed_partial(self):
        per_problem = {"test-1": scored(), "test-2": {"error": "CUDA out of memory"}}
        self.assertEqual(self.status(per_problem), "partial")

    def test_a_problem_not_yet_scored_makes_the_seed_partial(self):
        per_problem = {"test-1": scored()}
        self.assertEqual(self.status(per_problem, expected=PROBLEMS), "partial")

    def test_no_problem_scored_is_a_failure(self):
        self.assertEqual(
            self.status({"test-1": {"error": "boom"}}), "validation_failed"
        )

    def test_the_collector_reads_only_complete_seeds(self):
        self.status({p: scored() for p in PROBLEMS}, seed=0)
        self.status({"test-1": scored(), "test-2": {"error": "OOM"}}, seed=1)
        self.assertEqual(len(collect_local("exp", self.logs)["driverlog"]), 1)

    def test_the_collector_still_reads_results_without_a_status(self):
        path = self.logs / "validation" / "exp" / "driverlog"
        path.mkdir(parents=True)
        (path / "seed0.json").write_text(
            json.dumps({"domain": "driverlog", "per_problem": {"test-1": scored()}})
        )
        self.assertEqual(len(collect_local("exp", self.logs)["driverlog"]), 1)


class StandInValidator:
    calls: list = []

    def __init__(self, model_id, logs_path, problem_path, **kwargs):
        self.problem = Path(problem_path).stem

    def setup(self, num_samples, sample_population, seed):
        self.seed = seed

    def validate(self, progress_path=None):
        StandInValidator.calls.append((self.problem, self.seed))
        return scored(tp=50)


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.logs = temp_dir(self)
        checkpoints = self.logs / "dias" / "mlp-gripper-s0" / "checkpoints"
        checkpoints.mkdir(parents=True)
        self.checkpoint = checkpoints / "best-epoch=0-step=10.ckpt"
        self.checkpoint.touch()
        self.result = self.logs / "validation" / "mlp" / "gripper" / "seed0.json"
        StandInValidator.calls = []
        patcher = mock.patch.object(
            validate_runs, "validator_class", return_value=StandInValidator
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def validate(self, resume=True):
        return validate_run(
            "mlp",
            "gripper",
            0,
            "mlp",
            {"num_samples": 5, "sample_population": 10},
            self.logs,
            log_wandb=False,
            num_workers=1,
            resume=resume,
        )

    def write_partial(self, **fields):
        """Two problems scored and the third failed, written after the checkpoint."""
        self.result.parent.mkdir(parents=True)
        per_problem = {"test-1": scored(tp=1), "test-2": scored(tp=2)}
        per_problem["test-3"] = {"error": "CUDA out of memory"}
        self.result.write_text(
            json.dumps(
                {
                    "sample_seed": 3449441584,
                    "threshold": 0.5,
                    "per_problem": per_problem,
                    **fields,
                }
            )
        )
        past = time.time() - 3600
        os.utime(self.checkpoint, (past, past))

    def test_only_the_failed_problem_is_scored_again_with_the_same_sample(self):
        self.write_partial()
        per_problem = self.validate()
        self.assertEqual(StandInValidator.calls, [("test-3", 3449441584)])
        self.assertEqual([per_problem[p]["tp"] for p in PROBLEMS], [1, 2, 50])
        payload = json.loads(self.result.read_text())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["checkpoint"], str(self.checkpoint))

    def test_results_older_than_the_checkpoint_are_refused(self):
        self.write_partial()
        future = time.time() + 3600
        os.utime(self.checkpoint, (future, future))
        with self.assertRaises(SystemExit):
            self.validate()

    def test_results_recording_another_checkpoint_are_refused(self):
        self.write_partial(checkpoint="/elsewhere/best.ckpt", checkpoint_mtime=1.0)
        with self.assertRaises(SystemExit):
            self.validate()

    def test_without_resume_every_problem_is_scored(self):
        self.write_partial()
        self.validate(resume=False)
        self.assertEqual([call[0] for call in StandInValidator.calls], PROBLEMS)


if __name__ == "__main__":
    unittest.main()
