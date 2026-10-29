"""Tests for `_seed_precision_recall`, which turns a seed's per-problem results into one P/R pair, for restricting a table to some seeds, for seeds that did not score every problem, and for reading a sweep's base value off the base runs. Every validator emits pooled tp/fp/fn, so every column of the paper table is micro-averaged the same way."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from collect_results import (
    _seed_precision_recall,
    collect_local,
    coverage_local,
    trained_values,
)
from support import temp_dir


def problem(tp, fp, fn, **extra):
    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }
    result.update(extra)
    return result


class SeedPrecisionRecallTest(unittest.TestCase):
    # A large problem the schema does well on, and a tiny one it fails.
    PER_PROBLEM = {"big": problem(90, 10, 10), "small": problem(0, 1, 1)}

    def test_counts_are_pooled_not_averaged(self):
        precision, recall = _seed_precision_recall(self.PER_PROBLEM)
        self.assertAlmostEqual(precision, 90 / 101)
        self.assertAlmostEqual(recall, 90 / 101)
        # The macro mean would have been 0.45; the two differ, so the test would catch a silent fall-back to it.
        self.assertNotAlmostEqual(precision, 0.45)

    def test_action_stats_and_pooled_counts_agree(self):
        with_stats = {
            name: {
                **{k: v for k, v in result.items() if k not in ("tp", "fp", "fn")},
                "action_stats": {
                    "a": {"tp": result["tp"], "fp": result["fp"], "fn": result["fn"]}
                },
            }
            for name, result in self.PER_PROBLEM.items()
        }
        self.assertEqual(
            _seed_precision_recall(with_stats),
            _seed_precision_recall(self.PER_PROBLEM),
        )

    def test_results_carrying_only_rates_fall_back_to_the_mean(self):
        rates_only = {
            name: {"precision": r["precision"], "recall": r["recall"]}
            for name, r in self.PER_PROBLEM.items()
        }
        precision, recall = _seed_precision_recall(rates_only)
        self.assertAlmostEqual(precision, 0.45)
        self.assertAlmostEqual(recall, 0.45)

    def test_failed_problems_are_skipped_and_empty_seeds_return_none(self):
        self.assertIsNone(_seed_precision_recall({}))
        self.assertIsNone(_seed_precision_recall({"p": {"error": "boom"}}))
        self.assertEqual(
            _seed_precision_recall({**self.PER_PROBLEM, "p": {"error": "boom"}}),
            _seed_precision_recall(self.PER_PROBLEM),
        )

    def test_counts_survive_the_json_round_trip(self):
        """validate_runs writes the result as JSON, which has no integer type.

        An earlier version keyed on `isinstance(v, int)` and silently fell back to the macro mean for every result read off disk.
        """
        on_disk = json.loads(json.dumps(self.PER_PROBLEM))
        self.assertEqual(
            _seed_precision_recall(on_disk),
            _seed_precision_recall(self.PER_PROBLEM),
        )

    def test_a_seed_with_no_predictions_scores_zero(self):
        self.assertEqual(_seed_precision_recall({"p": problem(0, 0, 5)}), (0.0, 0.0))


class SeedRestrictionTest(unittest.TestCase):
    """The ablations ran five seeds and `dias-names` ten, so their table compares `dias-names` on the same five."""

    def setUp(self):
        self.logs = temp_dir(self)
        directory = self.logs / "validation" / "exp" / "gripper"
        directory.mkdir(parents=True)
        for seed in range(10):
            # Seeds 5 and up are the only ones that are not sound.
            payload = {
                "domain": "gripper",
                "seed": seed,
                "status": "ok",
                "per_problem": {"test-1": problem(9, int(seed >= 5), 1)},
            }
            (directory / f"seed{seed}.json").write_text(json.dumps(payload))

    def test_every_seed_is_read_by_default(self):
        self.assertEqual(len(collect_local("exp", self.logs)["gripper"]), 10)

    def test_only_the_given_seeds_are_read(self):
        prs = collect_local("exp", self.logs, seeds=set(range(5)))["gripper"]
        self.assertEqual([precision for precision, _ in prs], [1.0] * 5)

    def test_a_result_without_a_seed_is_known_by_its_file_name(self):
        path = self.logs / "validation" / "exp" / "gripper" / "seed7.json"
        payload = json.loads(path.read_text())
        del payload["seed"]
        path.write_text(json.dumps(payload))
        self.assertEqual(len(collect_local("exp", self.logs, seeds={7})["gripper"]), 1)


class PartialSeedTest(unittest.TestCase):
    """A noise validation can run out of memory or be stopped partway; its fully scored problems still count with `include_partial`."""

    def setUp(self):
        self.logs = temp_dir(self)
        directory = self.logs / "validation" / "exp" / "satellite"
        directory.mkdir(parents=True)
        results = {
            0: ("ok", {p: problem(9, 0, 1) for p in ("test-1", "test-2", "test-3")}),
            # Stopped during test-3: only the problems it finished carry scores.
            1: (
                "partial",
                {"test-1": problem(1, 9, 0), "test-2": problem(1, 9, 0)},
            ),
            # Out of memory on its first problem.
            2: (
                "out_of_memory",
                {"test-1": {"error": "does not fit", "out_of_memory": True}},
            ),
        }
        for seed, (status, per_problem) in results.items():
            payload = {
                "domain": "satellite",
                "seed": seed,
                "status": status,
                "per_problem": per_problem,
            }
            (directory / f"seed{seed}.json").write_text(json.dumps(payload))

    def test_seeds_that_did_not_score_every_problem_are_left_out_by_default(self):
        self.assertEqual(len(collect_local("exp", self.logs)["satellite"]), 1)

    def test_partial_seeds_count_over_the_problems_they_scored(self):
        prs = collect_local("exp", self.logs, include_partial=True)["satellite"]
        self.assertEqual(sorted(prs), [(0.1, 1.0), (1.0, 0.9)])

    def test_coverage_counts_which_problems_each_incomplete_seed_scored(self):
        coverage = coverage_local("exp", self.logs)["satellite"]
        self.assertEqual(coverage, {("test-1", "test-2"): 1, (): 1})


class TrainedValueTest(unittest.TestCase):
    """`collect_results --sweep` reads a value without a sub-experiment from the base runs that trained at it."""

    def setUp(self):
        self.logs = temp_dir(self)
        for run, every, slots in (
            ("base-gripper-s0", 500, 5),
            ("base-tau100-gripper-s0", 100, 5),
        ):
            directory = self.logs / "dias" / run
            directory.mkdir(parents=True)
            # As Lightning saves it: the dataclass-typed schedule without its class_path / init_args wrapper.
            (directory / "config.yaml").write_text(
                f"model:\n  init_args:\n    num_slots: {slots}\n    precondition_tau:\n      start: 1.0\n      floor: 0.0\n      factor: 0.1\n      every: {every}\n"
            )

    def test_reads_a_plain_argument(self):
        self.assertEqual(
            trained_values("base", "model.init_args.num_slots", self.logs),
            {"gripper": 5.0},
        )

    def test_reads_a_schedule_field_saved_without_its_wrapper(self):
        values = trained_values(
            "base", "model.init_args.precondition_tau.init_args.every", self.logs
        )
        # The sweep's own sub-experiment, base-tau100, is not mistaken for a base run.
        self.assertEqual(values, {"gripper": 500.0})


if __name__ == "__main__":
    unittest.main()
