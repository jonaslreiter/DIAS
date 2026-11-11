"""What `launch.py` submits: validation alone over finished runs, the Slurm resources, the overlays, the sweeps and L1."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import launch
import pipeline
from support import temp_dir, write_run


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

    def test_validation_scores_the_final_checkpoint_unless_asked_for_the_best(self):
        self.assertNotIn("CHECKPOINT", self.exported())
        self.assertIn("CHECKPOINT=best", self.exported(validation_checkpoint="best"))


class ResourcesTest(unittest.TestCase):
    def setUp(self):
        self.log_dir = temp_dir(self)

    def test_each_experiment_type_takes_its_sections_of_the_slurm_config(self):
        config = launch.load_yaml(pipeline.SLURM_CONFIG)
        for experiment_type, train, validation in (
            ("dias", "train", "validate"),
            ("noise", "train-noise", "validate-noise"),
            ("mlp", "train", "validate-mlp"),
        ):
            plan = launch.build_plan(
                plan_args(self.log_dir, experiment_type=experiment_type)
            )
            self.assertEqual(plan.train_resources, config[train])
            self.assertEqual(plan.validation_resources, config[validation])

    def test_the_validation_array_asks_for_its_resources(self):
        plan = launch.build_plan(plan_args(self.log_dir, experiment_type="mlp"))
        validation = submitted(plan)[-1]
        for option in launch.sbatch_resource_opts(
            launch.resolve_resources(plan.validation_resources, "gripper")
        ):
            self.assertIn(option, validation)


class ThresholdSweepTest(unittest.TestCase):
    def setUp(self):
        self.log_dir = temp_dir(self)
        write_run(self.log_dir, "exp-gripper-s0")
        write_run(self.log_dir, "exp-gripper-s1")

    def plan(self, **overrides):
        return launch.build_plan(
            plan_args(self.log_dir, sweep="threshold", sweep_values="0.25", **overrides)
        )

    def test_it_validates_the_existing_runs_into_an_experiment_per_threshold(self):
        (validate,) = submitted(self.plan())
        export = next(opt for opt in validate if opt.startswith("--export"))
        for variable in (
            "EXPERIMENT=exp,",
            "RESULTS_EXPERIMENT=exp-thr0.25",
            "THRESHOLD=0.25",
            "NO_WANDB=1",
        ):
            self.assertIn(variable, export)
        self.assertFalse([opt for opt in validate if opt.startswith("--dependency")])

    def test_runs_that_did_not_finish_are_refused(self):
        self.log_dir = temp_dir(self)
        write_run(self.log_dir, "exp-gripper-s0")
        write_run(self.log_dir, "exp-gripper-s1", step=5000)
        with self.assertRaisesRegex(SystemExit, "did not train"):
            self.plan()

    def test_an_override_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "--set"):
            self.plan(set="model.init_args.num_slots=3")


class LabelsTest(unittest.TestCase):
    def test_a_label_setting_composes_its_overlay_after_the_experiment_type(self):
        for labels, overlays in (
            ("names", ["experiment-noise.yaml"]),
            ("strips", ["experiment-noise.yaml", "labels-strips.yaml"]),
            ("strips-plus", ["experiment-noise.yaml", "labels-strips-plus.yaml"]),
        ):
            with self.subTest(labels=labels):
                plan = launch.build_plan(
                    plan_args(temp_dir(self), labels=labels, experiment_type="noise")
                )
                self.assertEqual([path.name for path in plan.overlays], overlays)


class NoiseSweepTest(unittest.TestCase):
    def test_flips_per_transition_set_a_quarter_added_and_a_quarter_removed(self):
        plan = launch.build_plan(
            plan_args(temp_dir(self), experiment_type="noise", sweep="noise")
        )
        self.assertEqual(
            [point.experiment for point in plan.points],
            ["exp-f1", "exp-f2", "exp-f4", "exp-f8"],
        )
        self.assertIn(
            "--data.init_args.noise.added_atoms=0.5", plan.points[1].overrides
        )
        self.assertIn(
            "--data.init_args.noise.removed_atoms=0.5", plan.points[1].overrides
        )

    def test_it_needs_the_noise_experiment_type(self):
        with self.assertRaisesRegex(SystemExit, "--experiment-type noise"):
            launch.build_plan(plan_args(temp_dir(self), sweep="noise"))


class L1LaunchTest(unittest.TestCase):
    def args(self, *extra, domains=("gripper",)):
        return launch.parse_args(
            [
                "--experiment-type",
                "l1",
                "--experiment",
                "l1",
                *(["--domains", *domains] if domains else []),
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

    def test_without_domains_it_validates_every_domain_l1_learned(self):
        # L1 learned no domain for satellite-recal.
        plan = launch.build_plan(self.args(domains=None))
        self.assertIn("satellite", plan.domains)
        self.assertNotIn("satellite-recal", plan.domains)

    def test_a_domain_l1_never_learned_is_refused_by_name(self):
        with self.assertRaisesRegex(SystemExit, "no domain for satellite-recal"):
            launch.build_plan(self.args(domains=("gripper", "satellite-recal")))

    def test_options_for_a_training_run_are_refused(self):
        for extra in (
            ["--labels", "strips"],
            ["--ablation", "no-pcgrad"],
            ["--set", "model.init_args.num_slots=3"],
            ["--validation-checkpoint", "best"],
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
