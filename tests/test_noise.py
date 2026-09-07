"""Tests for the two adjacency-noise models."""

import unittest

import torch

from dias.data.noise import ExpectedCount, Ratio, RelationStats

# One unary and one binary relation, the only two arities a state can hold.
ARITIES = [1, 2]


def stats(states: torch.Tensor, is_fluent=(True, True)) -> RelationStats:
    return RelationStats.measure(states, ARITIES, list(is_fluent))


class RelationStatsTest(unittest.TestCase):
    def test_the_true_ratio_counts_only_cells_the_arity_allows(self):
        states = torch.tensor(
            [
                [
                    [[1, 1], [0, 1]],  # unary: the diagonal holds twice
                    [[1, 0], [0, 0]],  # binary: one of four cells holds
                ]
            ],
            dtype=torch.float32,
        )
        measured = stats(states)
        # Both diagonal cells hold, so there is nothing left to be false.
        self.assertEqual(measured.true_ratio[0], 0.0)
        self.assertEqual(measured.true_ratio[1], 1 / 3)

    def test_an_arity_with_no_tensor_representation_is_refused(self):
        with self.assertRaisesRegex(ValueError, "arity should be 1 or 2"):
            RelationStats.measure(torch.zeros(1, 1, 2, 2), [3], [True])


class RatioNoiseTest(unittest.TestCase):
    def test_it_flips_only_the_cells_a_relation_can_occupy(self):
        adjacency = torch.tensor(
            [[[[1, 1], [0, 0]], [[1, 0], [0, 0]]]], dtype=torch.float32
        )
        noisy = Ratio(remove=1.0, add=1.0).corrupt(
            adjacency,
            RelationStats(ARITIES, [1.0, 1.0], [True, True]),
            torch.Generator().manual_seed(3),
        )
        expected = torch.tensor(
            [[[[0, 1], [0, 1]], [[0, 1], [1, 1]]]], dtype=torch.float32
        )
        self.assertTrue(torch.equal(noisy, expected))

    def test_the_add_rate_is_scaled_by_how_often_a_relation_holds(self):
        """A relation that never holds gets no spurious atoms at all."""
        adjacency = torch.zeros(1, 2, 3, 3)
        noisy = Ratio(add=1.0).corrupt(
            adjacency,
            RelationStats(ARITIES, [0.0, 1.0], [True, True]),
            torch.Generator().manual_seed(0),
        )
        self.assertEqual(noisy[0, 0].sum().item(), 0.0)
        self.assertEqual(noisy[0, 1].sum().item(), 9.0)

    def test_a_model_at_rest_is_not_enabled(self):
        self.assertFalse(Ratio().enabled)
        self.assertTrue(Ratio(add=0.1).enabled)

    def test_negative_rates_are_refused(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            Ratio(remove=-0.1)


class ExpectedCountNoiseTest(unittest.TestCase):
    def pool(self) -> torch.Tensor:
        torch.manual_seed(0)
        return (torch.rand(8, 2, 5, 5) > 0.5).float()

    def test_the_budget_is_spent_per_state_not_per_cell(self):
        """Ten states, one removed atom each: about ten fewer atoms overall."""
        states = self.pool()
        noisy = ExpectedCount(removed_atoms=1.0).corrupt(
            states, stats(states), torch.Generator().manual_seed(5)
        )
        removed = (states - noisy).clamp(min=0).sum().item()
        self.assertEqual((noisy - states).clamp(min=0).sum().item(), 0.0)
        self.assertGreater(removed, 0.0)
        self.assertLess(removed, 4 * len(states))

    def test_a_static_relation_is_left_alone_unless_asked_for(self):
        states = self.pool()
        measured = stats(states, is_fluent=(True, False))
        untouched = ExpectedCount(20.0, 20.0, fluents_only=True).corrupt(
            states, measured, torch.Generator().manual_seed(5)
        )
        self.assertTrue(torch.equal(untouched[:, 1], states[:, 1]))
        self.assertFalse(torch.equal(untouched[:, 0], states[:, 0]))

        touched = ExpectedCount(20.0, 20.0, fluents_only=False).corrupt(
            states, measured, torch.Generator().manual_seed(5)
        )
        self.assertFalse(torch.equal(touched[:, 1], states[:, 1]))

    def test_a_budget_larger_than_the_state_flips_everything_once(self):
        """Positions are drawn without replacement, so nothing flips twice."""
        states = self.pool()[:1]
        noisy = ExpectedCount(1e4, 1e4).corrupt(
            states, stats(states), torch.Generator().manual_seed(1)
        )
        self.assertTrue(torch.equal(noisy[0, 1], 1 - states[0, 1]))

    def test_a_state_with_no_room_to_flip_is_returned_unchanged(self):
        states = torch.ones(1, 2, 3, 3)
        noisy = ExpectedCount(5.0, 5.0).corrupt(
            states, stats(states), torch.Generator().manual_seed(1)
        )
        self.assertTrue(torch.equal(noisy, states))

    def test_negative_budgets_are_refused(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ExpectedCount(added_atoms=-1.0)


if __name__ == "__main__":
    unittest.main()
