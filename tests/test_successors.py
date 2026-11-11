"""How validation draws its states: training's random walk, several times over, and an equal share of the sample for every action schema."""

import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pymimir

from dias.validation import successors
from dias.validation.successors import sample_walk_states

PDDL = Path(__file__).resolve().parents[1] / "pddl"


def problem(domain: str, name: str = "test-1"):
    return pymimir.Problem(
        pymimir.Domain(PDDL / domain / "domain.pddl"), PDDL / domain / f"{name}.pddl"
    )


def applicable_in(states) -> Counter:
    return Counter(
        name
        for state in states
        for name in {
            a.get_action().get_name() for a in state.generate_applicable_actions()
        }
    )


class SampleWalkStatesTest(unittest.TestCase):
    def test_every_action_schema_applies_in_its_share_of_the_states(self):
        """Spanner's tighten_nut only applies at the gate, past every location, and the first 500 states a breadth-first search reaches on this problem include none there."""
        states = sample_walk_states(
            problem("spanner"), num_samples=30, walk_length=200, min_walks=1, seed=0
        )
        self.assertEqual(len({s.get_index() for s in states}), 30)
        counts = applicable_in(states)
        for name in ("pickup_spanner", "tighten_nut", "walk"):
            self.assertGreaterEqual(counts[name], 10, name)

    def test_the_seed_reproduces_the_states(self):
        def draw(seed):
            return [
                s.get_index()
                for s in sample_walk_states(
                    problem("blocks-3"),
                    num_samples=12,
                    walk_length=30,
                    min_walks=2,
                    seed=seed,
                )
            ]

        self.assertEqual(draw(3), draw(3))
        self.assertNotEqual(draw(3), draw(4))

    def test_walks_are_added_until_the_quota_is_met_and_no_further(self):
        walks = []
        original = successors.random_walk

        def counted(*args):
            walks.append(None)
            return original(*args)

        with mock.patch.object(successors, "random_walk", counted):
            sample_walk_states(
                problem("blocks-3"), num_samples=6, walk_length=30, min_walks=3, seed=0
            )
            self.assertEqual(len(walks), 3)
            walks.clear()
            # tighten_nut needs about 19 steps in, so walks of 5 never meet its share.
            with self.assertLogs(successors.logger, "WARNING"):
                states = sample_walk_states(
                    problem("spanner"),
                    num_samples=6,
                    walk_length=5,
                    min_walks=1,
                    seed=0,
                )
        self.assertEqual(len(walks), successors.MAX_WALKS)
        self.assertEqual(len(states), 6)
        self.assertEqual(applicable_in(states)["tighten_nut"], 0)


if __name__ == "__main__":
    unittest.main()
