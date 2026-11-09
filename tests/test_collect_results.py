"""Collecting validation results: a seed's precision and recall micro-averaged over its problems, restricting to some seeds, seeds that did not score every problem, and a sweep value read off the base runs."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from collect_results import (
    aggregate,
    collect,
    coverage,
    rate,
    seed_precision_recall,
    trained_values,
)
from support import temp_dir


def problem(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn}


def write(logs: Path, domain: str, seed: int, status: str, per_problem: dict):
    directory = logs / "validation" / "exp" / domain
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "domain": domain,
        "seed": seed,
        "status": status,
        "per_problem": per_problem,
    }
    (directory / f"seed{seed}.json").write_text(json.dumps(payload))


class SeedPrecisionRecallTest(unittest.TestCase):
    PER_PROBLEM = {"big": problem(90, 10, 10), "small": problem(0, 1, 1)}

    def test_counts_are_pooled_not_averaged(self):
        precision, recall = seed_precision_recall(self.PER_PROBLEM)
        self.assertAlmostEqual(precision, 90 / 101)
        self.assertAlmostEqual(recall, 90 / 101)

    def test_failed_problems_are_skipped_and_empty_seeds_return_none(self):
        self.assertIsNone(seed_precision_recall({}))
        self.assertIsNone(seed_precision_recall({"p": {"error": "boom"}}))
        self.assertEqual(
            seed_precision_recall({**self.PER_PROBLEM, "p": {"error": "boom"}}),
            seed_precision_recall(self.PER_PROBLEM),
        )

    def test_a_seed_with_no_predictions_scores_zero(self):
        self.assertEqual(seed_precision_recall({"p": problem(0, 0, 5)}), (0.0, 0.0))


class AggregateTest(unittest.TestCase):
    def test_counts_and_means(self):
        self.assertEqual(
            aggregate([(1.0, 1.0), (1.0, 0.5), (0.5, 1.0)]),
            {"P": 2.5 / 3, "R": 2.5 / 3, "N_S": 2, "N_C": 2, "N_SC": 1, "n": 3},
        )

    def test_a_rate_below_one_is_never_shown_as_one(self):
        self.assertEqual(rate(0.999), "0.99")
        self.assertEqual(rate(1.0), "1.00")


class SeedRestrictionTest(unittest.TestCase):
    def setUp(self):
        self.logs = temp_dir(self)
        for seed in range(10):
            write(
                self.logs,
                "gripper",
                seed,
                "ok",
                {"test-1": problem(9, int(seed >= 5), 1)},
            )

    def test_every_seed_is_read_by_default(self):
        self.assertEqual(len(collect("exp", self.logs)["gripper"]), 10)

    def test_only_the_given_seeds_are_read(self):
        prs = collect("exp", self.logs, seeds=set(range(5)))["gripper"]
        self.assertEqual(aggregate(prs)["N_S"], 5)
        self.assertEqual(len(prs), 5)


class PartialSeedTest(unittest.TestCase):
    def setUp(self):
        self.logs = temp_dir(self)
        write(
            self.logs,
            "satellite",
            0,
            "ok",
            {p: problem(9, 0, 1) for p in ("test-1", "test-2", "test-3")},
        )
        write(
            self.logs,
            "satellite",
            1,
            "partial",
            {"test-1": problem(1, 9, 0), "test-2": problem(1, 9, 0)},
        )
        write(
            self.logs,
            "satellite",
            2,
            "out_of_memory",
            {"test-1": {"error": "does not fit", "out_of_memory": True}},
        )

    def test_partial_seeds_count_over_the_problems_they_scored(self):
        prs = collect("exp", self.logs)["satellite"]
        self.assertEqual(sorted(prs), [(0.1, 1.0), (1.0, 0.9)])

    def test_partial_seeds_are_left_out_on_request(self):
        self.assertEqual(
            len(collect("exp", self.logs, exclude_partial=True)["satellite"]), 1
        )

    def test_coverage_counts_which_problems_each_incomplete_seed_scored(self):
        self.assertEqual(
            coverage("exp", self.logs)["satellite"], {("test-1", "test-2"): 1, (): 1}
        )


class TrainedValueTest(unittest.TestCase):
    def setUp(self):
        self.logs = temp_dir(self)
        for run, every, slots in (
            ("base-gripper-s0", 500, 5),
            ("base-tau100-gripper-s0", 100, 5),
        ):
            directory = self.logs / "dias" / run
            directory.mkdir(parents=True)
            # As Lightning saves a dataclass-typed argument: without class_path and init_args.
            (directory / "config.yaml").write_text(
                f"model:\n  init_args:\n    num_slots: {slots}\n    precondition_tau:\n      start: 1.0\n      floor: 0.0\n      factor: 0.1\n      every: {every}\n"
            )

    def test_reads_a_plain_argument(self):
        self.assertEqual(
            trained_values("base", "model.init_args.num_slots", self.logs),
            {"gripper": 5.0},
        )

    def test_reads_a_schedule_field_and_not_a_sweep_experiment(self):
        values = trained_values(
            "base", "model.init_args.precondition_tau.init_args.every", self.logs
        )
        self.assertEqual(values, {"gripper": 500.0})


if __name__ == "__main__":
    unittest.main()
