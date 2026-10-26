"""Tests for `_seed_precision_recall`, which turns a seed's per-problem results into one P/R pair, and for restricting a table to some seeds. Every validator emits pooled tp/fp/fn, so every column of the paper table is micro-averaged the same way."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from collect_results import _seed_precision_recall, collect_local
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


if __name__ == "__main__":
    unittest.main()
