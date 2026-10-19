"""The loss terms, on predictions built by hand rather than by a model.

The golden fixture pins these end to end, but only for the default flags. The ablations, and what each term is counting, are checked here.
"""

import math
import unittest

import torch

from dias.models.objectives import DiasObjective
from dias.models.outputs import Channels, SchemaPrediction

ARITY_RELATIONS = [1, 2]
NUM_SLOTS = 3
SHAPE = (1, len(ARITY_RELATIONS), NUM_SLOTS, NUM_SLOTS)


def channels(ignore: float) -> Channels:
    """A schema tensor that ignores every cell to the given degree."""
    rest = (1 - ignore) / 2
    return Channels(
        ignore=torch.full(SHAPE, ignore),
        positive=torch.full(SHAPE, rest),
        negative=torch.full(SHAPE, rest),
    )


def prediction(adjacency: torch.Tensor, effects=0.5, preconditions=0.5):
    return SchemaPrediction(
        adjacency=adjacency,
        input_adjacency=torch.zeros_like(adjacency),
        delta=torch.zeros_like(adjacency),
        effects=channels(effects),
        preconditions=channels(preconditions),
        selection=torch.zeros(1, NUM_SLOTS, adjacency.size(-1)),
        correspondence=torch.zeros(1, NUM_SLOTS + 1, adjacency.size(-1)),
        precondition_satisfaction=torch.tensor([0.25]),
        precondition_product=torch.tensor([0.125]),
    )


class DiasObjectiveTest(unittest.TestCase):
    def objective(self, **flags):
        return DiasObjective(ARITY_RELATIONS, NUM_SLOTS, **flags)

    def losses(self, adjacency, target, objective=None, slot_weights=None):
        objective = objective or self.objective()
        return objective(
            prediction(adjacency),
            target,
            torch.ones(1, NUM_SLOTS) if slot_weights is None else slot_weights,
        )

    def test_a_perfect_prediction_costs_almost_nothing(self):
        target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]])
        losses = self.losses(target.clone(), target)
        self.assertLess(losses.adjacency.item(), 1e-5)
        self.assertEqual(losses.hamming_distance.item(), 0)

    def test_the_hamming_distance_counts_cells_on_the_wrong_side_of_a_half(self):
        target = torch.zeros(1, 2, 2, 2)
        target[0, 0, 0, 0] = 1.0
        predicted = torch.full((1, 2, 2, 2), 0.45)
        predicted[0, 0, 0, 0] = 0.6  # right, and the only one over the threshold
        self.assertEqual(self.losses(predicted, target).hamming_distance.item(), 0)

        predicted[0, 1, 1, 1] = 0.51  # a cell the target does not have
        self.assertEqual(self.losses(predicted, target).hamming_distance.item(), 1)

    def test_the_two_auxiliary_terms_are_independent(self):
        adjacency = torch.full((1, 2, 2, 2), 0.5)
        target = torch.zeros(1, 2, 2, 2)
        both, effects_only, preconditions_only, neither = (
            self.losses(adjacency, target, self.objective(**flags)).auxiliary
            for flags in (
                {},
                {"use_precondition_loss": False},
                {"use_effect_loss": False},
                {"use_effect_loss": False, "use_precondition_loss": False},
            )
        )
        self.assertEqual(neither.item(), 0.0)
        self.assertGreater(effects_only.item(), 0.0)
        self.assertGreater(preconditions_only.item(), 0.0)
        torch.testing.assert_close(both, effects_only + preconditions_only)

    def test_inactive_slots_make_a_precondition_cost_more(self):
        """L_pre is weighted by slot activation, so the two cannot be traded off."""
        adjacency = torch.full((1, 2, 2, 2), 0.5)
        target = torch.zeros(1, 2, 2, 2)
        objective = self.objective(use_effect_loss=False)
        active = self.losses(adjacency, target, objective, torch.ones(1, NUM_SLOTS))
        inactive = self.losses(
            adjacency, target, objective, torch.full((1, NUM_SLOTS), 0.1)
        )
        self.assertGreater(inactive.auxiliary.item(), active.auxiliary.item())

    def test_the_rejection_loss_replaces_the_restrictive_term(self):
        """Scored as the admitted fraction times the schema cells the restrictive term would have covered, so the normalizer and alpha keep their meaning."""
        adjacency = torch.full((1, 2, 2, 2), 0.5)
        target = torch.zeros(1, 2, 2, 2)
        objective = self.objective(use_effect_loss=False, precondition_loss="reject")
        admitted = torch.tensor([[0.25], [0.75]])
        losses = objective(
            prediction(adjacency), target, torch.ones(1, NUM_SLOTS), admitted
        )
        normalizer = objective._normalizer(adjacency)
        torch.testing.assert_close(
            losses.auxiliary, torch.tensor(0.5 * 12 / normalizer)
        )
        self.assertEqual(losses.precondition_admitted.item(), 0.5)
        self.assertIn("precondition_admitted", losses.logged(1.0))

        with self.assertRaisesRegex(ValueError, "needs `admitted`"):
            objective(prediction(adjacency), target, torch.ones(1, NUM_SLOTS))
        with self.assertRaisesRegex(ValueError, "precondition_loss must be one of"):
            self.objective(precondition_loss="literals")

    def test_only_the_rejection_loss_asks_for_sampled_ground_actions(self):
        self.assertFalse(self.objective().rejects)
        self.assertTrue(self.objective(precondition_loss="reject").rejects)
        self.assertFalse(
            self.objective(
                precondition_loss="reject", use_precondition_loss=False
            ).rejects
        )
        restrictive = self.losses(
            torch.full((1, 2, 2, 2), 0.5), torch.zeros(1, 2, 2, 2)
        )
        self.assertNotIn("precondition_admitted", restrictive.logged(1.0))

    def test_the_normalizer_counts_the_schema_once_per_auxiliary_term(self):
        # 3**1 + 3**2 slot tuples, for the effects and the preconditions.
        objective = self.objective()
        self.assertEqual(objective.schema_cells, 12)
        # 16 adjacency cells, plus 12 schema cells per sample per term.
        adjacency = torch.full((2, 2, 2, 2), 0.5)
        self.assertEqual(objective._normalizer(adjacency), 16 + 12 * 2 * 2)
        # Plus one applicability per sample when that term is on.
        applicable = self.objective(use_applicability_loss=True)
        self.assertEqual(applicable._normalizer(adjacency), 16 + 12 * 2 * 2 + 2)

    def test_the_primary_loss_is_the_adjacency_loss_by_default(self):
        losses = self.losses(torch.full((1, 2, 2, 2), 0.5), torch.zeros(1, 2, 2, 2))
        self.assertIsNone(losses.applicability)
        self.assertIs(losses.primary, losses.adjacency)
        self.assertNotIn("loss_applicability", losses.logged(1.0))

    def test_the_applicability_term_joins_the_primary_loss(self):
        """One cross-entropy per transition, of its precondition satisfaction against one, on the adjacency term's scale."""
        adjacency = torch.full((1, 2, 2, 2), 0.5)
        target = torch.zeros(1, 2, 2, 2)
        objective = self.objective(use_applicability_loss=True)
        losses = self.losses(adjacency, target, objective)
        normalizer = objective._normalizer(adjacency)
        # The hand-built prediction's preconditions hold to 0.25.
        torch.testing.assert_close(
            losses.applicability, -torch.log(torch.tensor(0.25)) / normalizer
        )
        torch.testing.assert_close(
            losses.primary, losses.adjacency + losses.applicability
        )
        torch.testing.assert_close(
            losses.total(0.5), losses.primary + 0.5 * losses.auxiliary
        )
        self.assertIn("loss_applicability", losses.logged(1.0))

    def test_without_observation_noise_the_loss_is_the_plain_cross_entropy(self):
        adjacency = torch.tensor([[[[0.9, 0.2], [0.0, 1.0]], [[0.3, 0.7], [0.5, 0.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]])
        plain = self.losses(adjacency, target)
        explicit = self.losses(adjacency, target, self.objective(observation_noise=0.0))
        self.assertTrue(torch.equal(plain.adjacency, explicit.adjacency))

    def test_observation_noise_bounds_what_a_contradicted_cell_costs(self):
        """A cell observed true against a certain prediction of false costs -log(noise), not the clamp's -log(1e-7); the agreeing cells cost -log(1 - noise) each."""
        adjacency = torch.zeros(1, 2, 2, 2)
        target = torch.zeros(1, 2, 2, 2)
        target[0, 0, 0, 0] = 1.0
        objective = self.objective(observation_noise=0.01)
        losses = self.losses(adjacency, target, objective)
        normalizer = objective._normalizer(adjacency)
        expected = (-math.log(0.01) - 7 * math.log(0.99)) / normalizer
        self.assertAlmostEqual(losses.adjacency.item(), expected, places=6)
        self.assertEqual(losses.hamming_distance.item(), 1)

    def test_observation_noise_must_leave_the_observation_informative(self):
        for noise in (-0.1, 0.5, 1.0):
            with self.assertRaises(ValueError):
                self.objective(observation_noise=noise)


if __name__ == "__main__":
    unittest.main()
