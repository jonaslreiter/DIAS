"""The loss terms, on predictions built by hand rather than by a model.

The golden fixture pins these end to end, but only for the default flags. The ablations, and what each term is counting, are checked here.
"""

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
        selection_logits=torch.zeros(1, NUM_SLOTS + 1, adjacency.size(-1)),
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

    def test_the_normalizer_counts_the_schema_once_per_auxiliary_term(self):
        # 3**1 + 3**2 slot tuples, for the effects and the preconditions.
        objective = self.objective()
        self.assertEqual(objective.schema_cells, 12)
        # 16 adjacency cells, plus 12 schema cells per sample per term.
        adjacency = torch.full((2, 2, 2, 2), 0.5)
        self.assertEqual(objective._normalizer(adjacency), 16 + 12 * 2 * 2)


if __name__ == "__main__":
    unittest.main()
