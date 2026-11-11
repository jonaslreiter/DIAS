import unittest

import torch

from dias.models.selection import (
    SlotSelector,
    batched_sinkhorn_d1d2,
    resolve_num_slots,
)


class SinkhornTest(unittest.TestCase):
    def test_returns_finite_assignment_and_accepts_row_mask(self):
        logits = torch.zeros(1, 3, 2)
        result = batched_sinkhorn_d1d2(logits, max_iter=20)
        self.assertEqual(tuple(result.shape), (1, 3, 2))
        self.assertTrue(torch.isfinite(result).all())
        masked = batched_sinkhorn_d1d2(
            logits, max_iter=20, row_mask=torch.tensor([[False, True, False]])
        )
        self.assertTrue(torch.equal(masked[:, 1], torch.zeros(1, 2)))

    def test_a_masked_row_stops_the_iteration_where_its_absence_would(self):
        """A masked row stays at -inf, so its change between iterations is NaN, which must not keep the iteration from converging: per_action_slots masks a row of every action narrower than the widest, and each further iteration is kept for the backward pass."""
        torch.manual_seed(0)
        logits = torch.randn(4, 5, 7)
        row_mask = torch.tensor([False, True, False, True, False]).expand(4, 5)
        masked = batched_sinkhorn_d1d2(logits, row_mask=row_mask)
        absent = batched_sinkhorn_d1d2(logits[:, [0, 2, 4]])
        # Up to round-off, since the masked call reduces over the masked rows too. Left iterating to max_iter, it lands farther than that from where its absence stops.
        torch.testing.assert_close(masked[:, [0, 2, 4]], absent, rtol=0.0, atol=1e-6)


class ResolveNumSlotsTest(unittest.TestCase):
    def test_per_action_slots_takes_the_widest_action(self):
        self.assertEqual(
            resolve_num_slots(
                None, num_objects=9, action_arities=[1, 3, 2], per_action_slots=True
            ),
            3,
        )

    def test_a_configured_count_the_instance_supports_is_kept(self):
        self.assertEqual(resolve_num_slots(5, num_objects=9, action_arities=[1, 2]), 5)

    def test_more_slots_than_objects_are_clamped_with_a_warning(self):
        """A slot no object can bind to is dead weight in every tensor."""
        with self.assertLogs("dias.models.selection", level="WARNING") as logs:
            resolved = resolve_num_slots(9, num_objects=4, action_arities=[1, 2])
        self.assertEqual(resolved, 4)
        self.assertIn("Clamping num_slots 9 to 4", logs.output[0])

    def test_a_count_below_the_widest_action_is_refused(self):
        with self.assertRaisesRegex(AssertionError, "must be >= max action arity"):
            resolve_num_slots(2, num_objects=9, action_arities=[1, 3])

    def test_an_instance_too_small_for_the_widest_action_is_refused(self):
        with self.assertRaisesRegex(AssertionError, "too few objects"):
            resolve_num_slots(5, num_objects=2, action_arities=[3])

    def test_the_two_ways_of_saying_nothing_are_refused(self):
        with self.assertRaisesRegex(AssertionError, "num_slots must be set"):
            resolve_num_slots(None, num_objects=9, action_arities=[1])
        with self.assertRaisesRegex(AssertionError, "action_arities"):
            resolve_num_slots(5, num_objects=9, per_action_slots=True)


class SlotWeightsTest(unittest.TestCase):
    def selector(self, **kwargs):
        torch.manual_seed(2)
        return SlotSelector(num_actions=2, num_slots=3, object_dim=4, **kwargs)

    def test_weights_are_the_sigmoid_of_the_logits(self):
        selector = self.selector()
        with torch.no_grad():
            selector.slot_weight_logits.copy_(
                torch.tensor([[2.0, -2.0, 0.0], [0.0, 0.0, 0.0]])
            )
        torch.testing.assert_close(
            selector.slot_weights(torch.tensor([0])),
            torch.sigmoid(torch.tensor([[2.0, -2.0, 0.0]])),
        )

    def test_without_slot_weights_every_slot_is_fully_active(self):
        selector = self.selector(use_slot_weights=False)
        with torch.no_grad():
            selector.slot_weight_logits.fill_(-5.0)
        self.assertTrue(
            torch.equal(selector.slot_weights(torch.tensor([0, 1])), torch.ones(2, 3))
        )

    def test_known_arguments_keep_their_full_weight(self):
        """A slot whose object is given is not something the model has to learn."""
        selector = self.selector(force_action_args=True)
        with torch.no_grad():
            selector.slot_weight_logits.fill_(-5.0)
        weights = selector.slot_weights(
            torch.tensor([0]), torch.tensor([[1, -1]])  # narrower than num_slots
        )
        self.assertEqual(weights[0, 0].item(), 1.0)
        self.assertLess(weights[0, 1].item(), 0.5)
        self.assertLess(weights[0, 2].item(), 0.5)


class AppliedSlotWeightLogitsTest(unittest.TestCase):
    """The logits a schema is read off must say what `forward` does, not what the parameter holds."""

    def selector(self, **kwargs):
        torch.manual_seed(2)
        return SlotSelector(num_actions=2, num_slots=3, object_dim=4, **kwargs)

    def test_learned_weights_are_the_parameter_itself(self):
        selector = self.selector()
        self.assertIs(
            selector.applied_slot_weight_logits(), selector.slot_weight_logits
        )

    def test_without_slot_weights_every_slot_reads_as_certain(self):
        """The parameter stays at its zero initialisation, which a threshold of 0.5 would read as every slot inactive."""
        selector = self.selector(use_slot_weights=False)
        logits = selector.applied_slot_weight_logits()
        self.assertTrue(torch.equal(torch.sigmoid(logits), torch.ones(2, 3)))

    def test_per_action_slots_read_as_exactly_the_arity(self):
        selector = self.selector(per_action_slots=True, action_arities=[1, 3])
        with torch.no_grad():
            selector.slot_weight_logits[1].fill_(-5.0)
        logits = selector.applied_slot_weight_logits()
        self.assertTrue(
            torch.equal(
                torch.sigmoid(logits),
                torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
