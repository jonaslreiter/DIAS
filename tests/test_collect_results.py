"""Tests for the estimator that turns per-problem results into one P/R pair.

Defect: the DIAS validators emitted `action_stats` and the MLP baseline did
not, so `_seed_precision_recall` micro-averaged one column of the paper table
and macro-averaged the other. Every validator now emits pooled tp/fp/fn.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_results import _seed_precision_recall


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
        # The macro mean would have been 0.45 -- the two differ, so the test
        # would catch a silent fall-back to it.
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

    def test_a_seed_with_no_predictions_scores_zero(self):
        self.assertEqual(_seed_precision_recall({"p": problem(0, 0, 5)}), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
