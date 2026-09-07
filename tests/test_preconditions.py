"""Tests for the differentiable precondition gate."""

import unittest

import torch

from dias.models.preconditions import soft_conjunction


class SoftConjunctionTest(unittest.TestCase):
    def test_all_satisfied_gives_one(self):
        self.assertTrue(
            torch.equal(soft_conjunction(torch.ones(2, 3), dim=1), torch.ones(2))
        )

    def test_tau_one_is_the_geometric_mean(self):
        values = torch.tensor([[0.25, 1.0]])
        self.assertAlmostEqual(soft_conjunction(values, dim=1).item(), 0.5)

    def test_tau_zero_is_the_exact_product(self):
        values = torch.tensor([[0.25, 0.5, 1.0]])
        self.assertAlmostEqual(
            soft_conjunction(values, dim=1, tau=0.0).item(), 0.125, places=6
        )

    def test_a_zero_does_not_produce_an_infinite_gradient(self):
        values = torch.tensor([[0.0, 1.0]], requires_grad=True)
        result = soft_conjunction(values, dim=1)
        self.assertGreater(result.item(), 0.0)
        result.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())


if __name__ == "__main__":
    unittest.main()
