"""Tests for re-validating a finished experiment at other thresholds: it accepts the runs `launch.py --validate-only` accepts, and hands each validation array what `launch.py` would."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import launch_threshold_validation
from support import temp_dir, write_run


class ThresholdLaunchTest(unittest.TestCase):
    def setUp(self):
        self.log_dir = temp_dir(self)
        write_run(self.log_dir, "exp-gripper-s0")
        write_run(self.log_dir, "exp-gripper-s1")
        # Killed half way: both checkpoints exist, but training did not finish.
        write_run(self.log_dir, "exp-gripper-s2", step=5000)

    def submitted(self, *flags) -> list[list[str]]:
        """The sbatch command lines a launch over gripper at one threshold would issue."""
        argv = [
            "launch_threshold_validation.py",
            "--experiment",
            "exp",
            "--domains",
            "gripper",
            "--thresholds",
            "0.25",
            "--log-dir",
            str(self.log_dir),
            *flags,
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                launch_threshold_validation, "run_command", return_value="4242"
            ) as run,
            mock.patch("builtins.print"),
        ):
            launch_threshold_validation.main()
        return [call.args[0] for call in run.call_args_list]

    def test_finished_runs_are_validated_over_the_seeds_given(self):
        (command,) = self.submitted("--seeds", "0-1")
        self.assertIn("--array=0-1", command)
        self.assertIn(f"--output={self.log_dir.resolve()}/slurm/%x-%A_%a.out", command)

    def test_a_run_short_of_max_steps_is_refused_before_anything_is_submitted(self):
        with self.assertRaises(SystemExit) as refusal:
            self.submitted("--seeds", "0-2")
        self.assertIn("exp-gripper-s2", str(refusal.exception))

    def test_allow_unfinished_submits_anyway(self):
        (command,) = self.submitted("--seeds", "0-2", "--allow-unfinished")
        self.assertIn("--array=0-2", command)

    def test_resume_and_num_workers_reach_validation(self):
        (command,) = self.submitted("--seeds", "0", "--resume", "--num-workers", "4")
        export = next(opt for opt in command if opt.startswith("--export"))
        self.assertIn("RESUME=1", export)
        self.assertIn("NUM_WORKERS=4", export)


if __name__ == "__main__":
    unittest.main()
