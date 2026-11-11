"""Tests for the ground-truth side of `experiments/schema_sizes.py`: arity from `:parameters`, effects and preconditions as distinct atoms, equalities left out."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from schema_sizes import ground_truth_sizes, matches


class GroundTruthSizesTest(unittest.TestCase):
    def test_blocks_3(self):
        sizes = ground_truth_sizes("blocks-3")
        # move: three parameters, four effect atoms, three preconditions besides its three inequalities.
        self.assertEqual(sizes["move"], {"arity": 3, "effects": 4, "preconditions": 3})
        self.assertEqual(sizes["stack"], {"arity": 2, "effects": 3, "preconditions": 3})

    def test_typed_parameters_count_once_and_types_are_not_preconditions(self):
        sizes = ground_truth_sizes("satellite")
        self.assertEqual(sizes["take_image"]["arity"], 4)
        self.assertEqual(sizes["turn_to"]["arity"], 3)


class MatchesTest(unittest.TestCase):
    def test_counts_every_domain_action_seed_triple(self):
        truth = {"a": {"arity": 2, "effects": 1, "preconditions": 1}}
        results = {
            "d": {
                "truth": truth,
                "learned": {
                    0: {"a": {"arity": 2, "effects": 1, "preconditions": 9}},
                    1: {"a": {"arity": 3, "effects": 1, "preconditions": 9}},
                },
                "rates": {},
            }
        }
        self.assertEqual(matches(results), {"arity": (1, 2), "effects": (2, 2)})


if __name__ == "__main__":
    unittest.main()
