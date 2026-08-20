import unittest

import torch

from dias.models.selection import SlotSelector, batched_sinkhorn_d1d2


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


if __name__ == "__main__":
    unittest.main()
