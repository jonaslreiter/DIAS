"""The gradient combiners, on losses linear in the parameters, whose gradients are the vectors they are built from."""

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
    def combine(self, main, aux):
        weights = parameter(0.0, 0.0)
        PCGrad(AUXILIARY_WEIGHT).backward(
            linear_loss(weights, main), linear_loss(weights, aux), [weights]
        )
        return weights.grad

    def test_agreeing_gradients_are_summed_unchanged(self):
        grad = self.combine([4.0, 0.0], [0.5, 0.5])
        torch.testing.assert_close(grad, torch.tensor([4.0 + 1.0, 1.0]))

    def test_a_conflicting_gradient_keeps_only_its_orthogonal_part(self):
        """g_aux - (g_main . g_aux / |g_main|^2) g_main."""
        g_main = torch.tensor([2.0, 0.0])
        g_aux = torch.tensor([-1.0, 1.5])
        projected = PCGrad._project(g_main, g_aux)
        torch.testing.assert_close(projected, torch.tensor([0.0, 1.5]))
        torch.testing.assert_close(
            self.combine([2.0, 0.0], [-1.0, 1.5]),
            g_main + PCGrad._cap_norm(g_main, AUXILIARY_WEIGHT * projected),
        )

    def test_the_auxiliary_gradient_is_capped_at_the_main_norm(self):
        main = torch.tensor([0.5, 0.0])
        grad = self.combine([0.5, 0.0], [0.0, 10.0])
        torch.testing.assert_close(torch.norm(grad - main), torch.norm(main))

    def test_a_parameter_only_one_loss_reaches_gets_that_gradient(self):
        main_only = parameter(0.0, 0.0)
        aux_only = parameter(0.0, 0.0)
        untouched = parameter(0.0, 0.0)
        PCGrad(AUXILIARY_WEIGHT).backward(
            linear_loss(main_only, [1.0, 1.0]),
            linear_loss(aux_only, [3.0, 3.0]),
            [main_only, aux_only, untouched],
        )
        self.assertEqual(main_only.grad.tolist(), [1.0, 1.0])
        self.assertEqual(aux_only.grad.tolist(), [6.0, 6.0])
        self.assertIsNone(untouched.grad)

    def test_a_vanishing_main_gradient_is_not_projected_against(self):
        g_aux = torch.tensor([-1.0, 2.0])
        torch.testing.assert_close(PCGrad._project(torch.zeros(2), g_aux), g_aux)


class FactoryTest(unittest.TestCase):
    def test_the_combiner_matches_the_configured_ablation(self):
        self.assertIsInstance(
            gradient_combiner(True, 1.0, aux_active=False), SingleObjective
        )
        self.assertIsInstance(
            gradient_combiner(False, 1.0, aux_active=True), WeightedSum
        )
        self.assertIsInstance(gradient_combiner(True, 1.0, aux_active=True), PCGrad)


if __name__ == "__main__":
    unittest.main()
