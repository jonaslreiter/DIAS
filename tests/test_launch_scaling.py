"""Tests for launching the scaling study: its seeds and log directory mean what they mean for `experiments/launch.py`, and its collector reads where its runs write."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "scaling"))

import launch_scaling
from benchmark import SIZES_CONFIG, result_path
from collect_scaling import load_results
from pipeline import format_array
from support import temp_dir


class ScalingLaunchTest(unittest.TestCase):
    def setUp(self):
        self.log_dir = temp_dir(self)

    def submitted(self, *flags) -> list[str]:
        """The sbatch command line a launch over delivery would issue."""
        argv = [
            "launch_scaling.py",
            "--experiment",
            "exp",
            "--domains",
            "delivery",
            "--log-dir",
            str(self.log_dir),
            *flags,
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                launch_scaling, "run_command", return_value="4242"
            ) as run,
            mock.patch("builtins.print"),
        ):
            launch_scaling.main()
        (call,) = run.call_args_list
        return call.args[0]

    def test_seeds_take_ranges(self):
        self.assertIn("--array=0-4", self.submitted("--seeds", "0-4"))
        self.assertIn("--array=1,3-4", self.submitted("--seeds", "1", "3-4"))

    def test_num_runs_counts_from_zero(self):
        self.assertIn("--array=0-1", self.submitted("--num-runs", "2"))

    def test_without_either_the_seeds_of_sizes_yaml_run(self):
        seeds = yaml.safe_load(SIZES_CONFIG.read_text())["seeds"]
        self.assertIn(f"--array={format_array(sorted(seeds))}", self.submitted())

    def test_results_and_job_output_go_under_the_log_directory(self):
        command = self.submitted("--seeds", "0")
        self.assertIn(f"--output={self.log_dir.resolve()}/slurm/%x-%A_%a.out", command)
        export = next(opt for opt in command if opt.startswith("--export"))
        self.assertIn(f"LOGS_DIR={self.log_dir}", export)

    def test_the_collector_reads_where_a_run_writes(self):
        path = result_path(self.log_dir, "exp", "delivery", "n0012", 3)
        self.assertEqual(path.parents[2], self.log_dir / "scaling")
        path.parent.mkdir(parents=True)
        record = {"instance": "n0012", "seed": 3, "status": "ok"}
        path.write_text(json.dumps(record))
        self.assertEqual(
            load_results(self.log_dir, "exp", "delivery"), {"n0012": [record]}
        )


if __name__ == "__main__":
    unittest.main()
