"""Tests for `--validate-only`, which submits the validation array over an already trained run instead of training first. Two things make it correct: no training array goes out and the validation array waits on no dependency, and a run that stopped short of `max_steps` is refused rather than scored."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import launch
from support import temp_dir


def plan_args(log_dir, **overrides):
    """The command line a launch is built from, as parse_args leaves it."""
    argv = [
        "--experiment",
        "exp",
        "--domains",
        "gripper",
        "--num-runs",
        "2",
        "--log-dir",
        str(log_dir),
    ]
    for flag, value in overrides.items():
        argv.append(f"--{flag.replace('_', '-')}")
        if value is not True:
            argv.append(str(value))
    return launch.parse_args(argv)


MAX_STEPS = 10000


def write_run(log_dir, run_id, step=MAX_STEPS, max_steps=MAX_STEPS):
    """A run directory as training leaves it: the config it resolved to, and both checkpoints at `step`."""
    run_dir = log_dir / "dias" / run_id
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    run_dir.joinpath("config.yaml").write_text(f"trainer:\n  max_steps: {max_steps}\n")
    (checkpoints / f"final-epoch=0-step={step}.ckpt").touch()
    (checkpoints / f"best-epoch=0-step={step - 50}.ckpt").touch()


def submitted(plan):
    """The sbatch command lines one domain's submission would issue."""
    with mock.patch.object(launch, "run_command", return_value="4242") as run:
        launch.submit_domain(plan, plan.points[0], "gripper")
    return [call.args[0] for call in run.call_args_list]


class ValidateOnlyTest(unittest.TestCase):
    def setUp(self):
        self.log_dir = temp_dir(self)
        write_run(self.log_dir, "exp-gripper-s0")
        write_run(self.log_dir, "exp-gripper-s1")

    def test_a_full_launch_submits_training_then_dependent_validation(self):
        plan = launch.build_plan(plan_args(self.log_dir))
        train, validate = submitted(plan)
        self.assertIn("--job-name=dias-train-gripper", train)
        self.assertIn("--job-name=dias-val-gripper", validate)
        self.assertIn("--dependency=aftercorr:4242", validate)

    def test_validate_only_submits_the_validation_array_alone(self):
        plan = launch.build_plan(plan_args(self.log_dir, validate_only=True))
        (validate,) = submitted(plan)
        self.assertIn("--job-name=dias-val-gripper", validate)
        # Nothing to wait for: the checkpoints it reads already exist.
        self.assertFalse([opt for opt in validate if opt.startswith("--dependency")])

    def test_validate_only_keeps_the_seeds_it_was_given(self):
        plan = launch.build_plan(plan_args(self.log_dir, validate_only=True))
        (validate,) = submitted(plan)
        self.assertIn("--array=0-1", validate)

    def test_a_run_that_never_trained_is_refused_before_anything_is_submitted(self):
        args = plan_args(self.log_dir, validate_only=True, num_runs=3)
        with self.assertRaises(SystemExit) as refusal:
            launch.build_plan(args)
        self.assertIn("exp-gripper-s2", str(refusal.exception))

    def test_allow_unfinished_submits_anyway(self):
        args = plan_args(
            self.log_dir, validate_only=True, num_runs=3, allow_unfinished=True
        )
        plan = launch.build_plan(args)
        self.assertEqual(plan.seeds, [0, 1, 2])

    def test_training_is_not_skipped_without_the_flag(self):
        plan = launch.build_plan(plan_args(self.log_dir))
        self.assertFalse(plan.validate_only)


class ValidationThresholdTest(unittest.TestCase):
    def exported(self, **overrides) -> str:
        plan = launch.build_plan(plan_args(temp_dir(self), **overrides))
        return next(opt for opt in submitted(plan)[-1] if opt.startswith("--export"))

    def test_by_default_validation_scores_the_domain_the_run_wrote(self):
        self.assertNotIn("THRESHOLD", self.exported())

    def test_a_threshold_is_handed_to_validation_only_when_given(self):
        self.assertIn("THRESHOLD=0.25", self.exported(validation_threshold=0.25))


class L1LaunchTest(unittest.TestCase):
    def args(self, *extra):
        return launch.parse_args(
            [
                "--experiment-type",
                "l1",
                "--experiment",
                "l1",
                "--domains",
                "gripper",
                "--log-dir",
                str(temp_dir(self)),
                *extra,
            ]
        )

    def test_l1_only_validates_once_per_domain(self):
        (validate,) = submitted(launch.build_plan(self.args()))
        self.assertIn("--array=0", validate)
        self.assertFalse([opt for opt in validate if opt.startswith("--dependency")])
        self.assertTrue([opt for opt in validate if "EXP_TYPE=l1" in opt])

    def test_options_for_a_training_run_are_refused(self):
        for extra in (
            ["--ablation", "no-pcgrad"],
            ["--set", "model.init_args.num_slots=3"],
            ["--validation-threshold", "0.25"],
        ):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(SystemExit, extra[0]):
                    launch.build_plan(self.args(*extra))


class FinishedTrainingTest(unittest.TestCase):
    """Both checkpoints are rewritten throughout a run, so only the step they reached says training finished."""

    def setUp(self):
        self.log_dir = temp_dir(self)

    def run_dir(self, run_id):
        return self.log_dir / "dias" / run_id

    def test_a_run_that_reached_max_steps_is_finished(self):
        write_run(self.log_dir, "done")
        self.assertTrue(launch.finished_training(self.run_dir("done")))

    def test_a_run_killed_half_way_is_not_finished_though_it_has_both_checkpoints(self):
        write_run(self.log_dir, "killed", step=4650)
        checkpoints = self.run_dir("killed") / "checkpoints"
        # The distinction the names alone cannot make: a timed-out run leaves both behind.
        self.assertTrue(list(checkpoints.glob("best-epoch*.ckpt")))
        self.assertTrue(list(checkpoints.glob("final-epoch*.ckpt")))
        self.assertFalse(launch.finished_training(self.run_dir("killed")))

    def test_max_steps_comes_from_the_run_not_from_a_constant(self):
        # A dataset-size sweep or a --set may have trained this run for fewer steps.
        write_run(self.log_dir, "short", step=500, max_steps=500)
        self.assertTrue(launch.finished_training(self.run_dir("short")))

    def test_a_directory_that_was_never_trained_is_not_finished(self):
        self.assertFalse(launch.finished_training(self.run_dir("absent")))

    def test_checkpoints_without_a_saved_config_cannot_be_judged_finished(self):
        write_run(self.log_dir, "no-config")
        (self.run_dir("no-config") / "config.yaml").unlink()
        self.assertFalse(launch.finished_training(self.run_dir("no-config")))

    def test_an_unbounded_max_steps_is_no_completion_criterion(self):
        write_run(self.log_dir, "unbounded", step=99, max_steps=-1)
        self.assertFalse(launch.finished_training(self.run_dir("unbounded")))


if __name__ == "__main__":
    unittest.main()
