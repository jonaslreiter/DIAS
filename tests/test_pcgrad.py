"""The gradient combiners, on gradients whose right answer is known by hand.

Every case here uses losses that are linear in the parameters, so each loss's
gradient *is* the vector it was built from and the expected result can be
written down rather than recomputed by the code under test.
"""

import unittest

import torch

from dias.models.pcgrad import PCGrad, SingleObjective, WeightedSum, gradient_combiner

AUXILIARY_WEIGHT = 2.0


def parameter(*values):
    return torch.nn.Parameter(torch.tensor(list(values)))


def linear_loss(parameter, direction):
    """A loss whose gradient with respect to `parameter` is exactly `direction`."""
    return (parameter * torch.tensor(direction)).sum()


class GradientCombinerTest(unittest.TestCase):
    def test_a_single_objective_ignores_the_auxiliary_loss(self):
        weights = parameter(0.0, 0.0)
        SingleObjective().backward(
            linear_loss(weights, [1.0, 2.0]),
            linear_loss(weights, [5.0, 5.0]),
            [weights],
        )
        self.assertEqual(weights.grad.tolist(), [1.0, 2.0])

    def test_a_weighted_sum_adds_the_scaled_auxiliary_gradient(self):
        weights = parameter(0.0, 0.0)
        WeightedSum(AUXILIARY_WEIGHT).backward(
            linear_loss(weights, [1.0, 2.0]),
            linear_loss(weights, [3.0, 0.0]),
            [weights],
        )
        self.assertEqual(weights.grad.tolist(), [1.0 + 2.0 * 3.0, 2.0])


class PCGradTest(unittest.TestCase):
    def combine(self, primary, auxiliary, **kwargs):
        weights = parameter(0.0, 0.0)
        PCGrad(AUXILIARY_WEIGHT, **kwargs).backward(
            linear_loss(weights, primary), linear_loss(weights, auxiliary), [weights]
        )
        return weights.grad

    def test_agreeing_gradients_are_summed_unchanged(self):
        # (1, 0) . (1, 1) > 0, and the auxiliary norm stays under the primary's.
        grad = self.combine([4.0, 0.0], [0.5, 0.5])
        torch.testing.assert_close(grad, torch.tensor([4.0 + 1.0, 1.0]))

    def test_a_conflicting_gradient_keeps_only_its_orthogonal_part(self):
        """g_aux - (g_adj . g_aux / |g_adj|^2) g_adj, then g_adj + that."""
        # The clip is disabled here: it has its own test, and the projected
        # gradient below is longer than the primary one, so it would fire.
        primary, auxiliary = [2.0, 0.0], [-1.0, 1.5]
        grad = self.combine(primary, auxiliary, dynamic_clip=False)

        g_primary = torch.tensor(primary)
        g_auxiliary = AUXILIARY_WEIGHT * torch.tensor(auxiliary)
        opposition = torch.dot(g_primary, g_auxiliary)
        self.assertLess(opposition.item(), 0.0)
        projected = (
            g_auxiliary - opposition / torch.dot(g_primary, g_primary) * g_primary
        )
        # The conflicting component is gone: what is left is orthogonal.
        torch.testing.assert_close(torch.dot(g_primary, projected), torch.tensor(0.0))
        torch.testing.assert_close(grad, g_primary + projected)

    def test_the_dynamic_clip_caps_the_auxiliary_gradient_at_the_primary_norm(self):
        primary, auxiliary = [0.5, 0.0], [0.0, 10.0]
        clipped = self.combine(primary, auxiliary)
        unclipped = self.combine(primary, auxiliary, dynamic_clip=False)

        g_primary = torch.tensor(primary)
        # Orthogonal already, so only the clip separates the two results.
        torch.testing.assert_close(
            unclipped, g_primary + AUXILIARY_WEIGHT * torch.tensor(auxiliary)
        )
        torch.testing.assert_close(
            torch.norm(clipped - g_primary), torch.norm(g_primary)
        )

    def test_a_parameter_only_one_loss_reaches_gets_that_gradient(self):
        primary_only = parameter(0.0, 0.0)
        auxiliary_only = parameter(0.0, 0.0)
        untouched = parameter(0.0, 0.0)
        PCGrad(AUXILIARY_WEIGHT).backward(
            linear_loss(primary_only, [1.0, 1.0]),
            linear_loss(auxiliary_only, [3.0, 3.0]),
            [primary_only, auxiliary_only, untouched],
        )
        self.assertEqual(primary_only.grad.tolist(), [1.0, 1.0])
        self.assertEqual(auxiliary_only.grad.tolist(), [6.0, 6.0])
        self.assertIsNone(untouched.grad)

    def test_a_vanishing_primary_gradient_leaves_the_auxiliary_one_alone(self):
        """There is no direction to project onto, so nothing is removed."""
        weights = parameter(0.0, 0.0)
        PCGrad(1.0, dynamic_clip=False).backward(
            linear_loss(weights, [0.0, 0.0]),
            linear_loss(weights, [-1.0, 2.0]),
            [weights],
        )
        self.assertEqual(weights.grad.tolist(), [-1.0, 2.0])


class FactoryTest(unittest.TestCase):
    def test_the_combiner_matches_the_configured_ablation(self):
        self.assertIsInstance(
            gradient_combiner(True, 1.0, auxiliary_active=False), SingleObjective
        )
        self.assertIsInstance(
            gradient_combiner(False, 1.0, auxiliary_active=True), WeightedSum
        )
        self.assertIsInstance(
            gradient_combiner(True, 1.0, auxiliary_active=True), PCGrad
        )


if __name__ == "__main__":
    unittest.main()
