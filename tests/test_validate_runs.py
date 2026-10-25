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
from dias.validation.workers import StateDoesNotFit
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

    def test_a_problem_out_of_memory_leaves_the_seed_out(self):
        per_problem = {
            "test-1": scored(),
            "test-2": {"error": "state 7 does not fit", "out_of_memory": True},
        }
        self.assertEqual(self.status(per_problem), "out_of_memory")

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
    # Problems with a state that does not fit.
    too_big: set = set()
    threshold = 0.5
    learned_domain_text = "(define (domain learned))"

    def __init__(self, problem_path, **kwargs):
        self.problem = Path(problem_path).stem
        self.kwargs = kwargs
        StandInValidator.instances.append(self)

    def setup(self, sampling, seed):
        self.seed = seed

    def validate(self, progress_path=None):
        StandInValidator.calls.append((self.problem, self.seed))
        if self.problem in StandInValidator.too_big:
            raise StateDoesNotFit(7, 2**30)
        return scored(tp=50)


SAMPLING = {"strategy": "walks", "num_samples": 5, "walk_length": 10, "min_walks": 2}


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.logs = temp_dir(self)
        checkpoints = self.logs / "dias" / "mlp-gripper-s0" / "checkpoints"
        checkpoints.mkdir(parents=True)
        self.checkpoint = checkpoints / "final-epoch=0-step=10.ckpt"
        self.checkpoint.touch()
        self.result = self.logs / "validation" / "mlp" / "gripper" / "seed0.json"
        StandInValidator.calls = []
        StandInValidator.instances = []
        StandInValidator.too_big = set()
        patcher = mock.patch.object(
            validate_runs, "validator_class", return_value=StandInValidator
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def validate(self, resume=True, checkpoint="final", sampling=SAMPLING):
        return validate_run(
            "mlp",
            "gripper",
            0,
            "mlp",
            {"sampling": sampling},
            self.logs,
            log_wandb=False,
            num_workers=1,
            resume=resume,
            checkpoint=checkpoint,
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
                    "force_injective": False,
                    "sampling": SAMPLING,
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

    def test_results_scored_otherwise_are_refused(self):
        """Forcing distinct parameters changes the domain scored, so the problems left must not be scored under the other reading."""
        self.write_partial(force_injective=True)
        with self.assertRaisesRegex(SystemExit, "force_injective"):
            self.validate()

    def test_results_from_before_force_injective_are_refused(self):
        self.write_partial()
        payload = json.loads(self.result.read_text())
        del payload["force_injective"]
        self.result.write_text(json.dumps(payload))
        with self.assertRaisesRegex(SystemExit, "force_injective None"):
            self.validate()

    def test_a_state_that_does_not_fit_ends_the_seed(self):
        StandInValidator.too_big = {"test-2"}
        per_problem = self.validate(resume=False)
        self.assertEqual(
            [problem for problem, _ in StandInValidator.calls], ["test-1", "test-2"]
        )
        self.assertTrue(per_problem["test-2"]["out_of_memory"])
        self.assertEqual(json.loads(self.result.read_text())["status"], "out_of_memory")

    def test_the_best_checkpoint_is_scored_when_asked(self):
        best = self.checkpoint.with_name("best-epoch=0-step=5.ckpt")
        best.touch()
        self.validate(resume=False, checkpoint="best")
        self.assertEqual(StandInValidator.instances[0].kwargs["checkpoint"], "best")
        self.assertEqual(json.loads(self.result.read_text())["checkpoint"], str(best))

    def test_the_scored_domain_is_kept_beside_the_results(self):
        self.validate(resume=False)
        self.assertEqual(
            self.result.with_suffix(".pddl").read_text(),
            StandInValidator.learned_domain_text,
        )

    def test_results_sampled_otherwise_are_refused(self):
        """The problems left must draw their states as the scored ones did, or a seed's three numbers would come from two samplings."""
        self.write_partial(sampling={**SAMPLING, "walk_length": 20})
        with self.assertRaisesRegex(SystemExit, "sampled"):
            self.validate()

    def test_results_from_before_the_sampling_key_were_sampled_breadth_first(self):
        """As the paper drew them, so they resume under that draw and under nothing else."""
        self.write_partial()
        payload = json.loads(self.result.read_text())
        del payload["sampling"]
        self.result.write_text(json.dumps(payload))
        with self.assertRaisesRegex(SystemExit, "bfs"):
            self.validate()
        self.validate(sampling={"strategy": "bfs"})
        self.assertEqual(StandInValidator.calls, [("test-3", 3449441584)])

    def test_the_result_records_its_sampling(self):
        self.validate(resume=False)
        self.assertEqual(json.loads(self.result.read_text())["sampling"], SAMPLING)

    def test_without_resume_every_problem_is_scored(self):
        self.write_partial()
        self.validate(resume=False)
        self.assertEqual([call[0] for call in StandInValidator.calls], PROBLEMS)


class L1RunTest(unittest.TestCase):
    def setUp(self):
        self.logs = temp_dir(self)
        StandInValidator.calls = []
        StandInValidator.instances = []
        StandInValidator.too_big = set()
        patcher = mock.patch.object(
            validate_runs, "validator_class", return_value=StandInValidator
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def validate(self, **kwargs):
        return validate_run(
            "l1",
            "gripper",
            0,
            "l1",
            {"sampling": SAMPLING},
            self.logs,
            **kwargs,
        )

    def test_l1_scores_its_learned_domain_with_no_run_behind_it(self):
        self.validate()
        kwargs = StandInValidator.instances[0].kwargs
        self.assertEqual(Path(kwargs["learned_domain_path"]).name, "gripper.pddl")
        self.assertEqual(Path(kwargs["original_domain_path"]).name, "domain.pddl")
        self.assertFalse(kwargs["force_injective"])
        payload = json.loads(
            (self.logs / "validation" / "l1" / "gripper" / "seed0.json").read_text()
        )
        self.assertEqual(payload["status"], "ok")
        self.assertNotIn("checkpoint", payload)

    def test_a_threshold_or_checkpoint_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "threshold"):
            self.validate(threshold=0.25)
        with self.assertRaisesRegex(SystemExit, "checkpoint"):
            self.validate(checkpoint="best")


if __name__ == "__main__":
    unittest.main()
