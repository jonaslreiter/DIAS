"""Tests for the dense adjacency representation."""

import unittest

import torch

from dias.adjacency import adj_to_pyg_data, valid_cells


class ValidCellsTest(unittest.TestCase):
    def test_binary_relations_use_the_whole_matrix(self):
        self.assertEqual(valid_cells(2, 3).tolist(), list(range(9)))

    def test_unary_relations_use_only_the_diagonal(self):
        self.assertEqual(valid_cells(1, 3).tolist(), [0, 4, 8])

    def test_other_arities_have_no_tensor_representation(self):
        self.assertIsNone(valid_cells(0, 3))
        self.assertIsNone(valid_cells(3, 3))


class AdjToPygDataTest(unittest.TestCase):
    def test_each_non_zero_becomes_an_edge_tagged_with_its_relation(self):
        adjacency = torch.tensor(
            [[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=torch.float32
        )
        data = adj_to_pyg_data(adjacency, num_objects=2)
        self.assertTrue(torch.equal(data.edge_index, torch.tensor([[0, 0], [0, 1]])))
        self.assertTrue(torch.equal(data.edge_attr, torch.tensor([0, 1])))
        self.assertEqual(data.num_nodes, 2)

    def test_an_empty_state_still_declares_its_objects(self):
        data = adj_to_pyg_data(torch.zeros(1, 2, 2), num_objects=2)
        self.assertEqual(tuple(data.edge_index.shape), (2, 0))
        self.assertEqual(tuple(data.edge_attr.shape), (0,))
        self.assertEqual(data.num_nodes, 2)


if __name__ == "__main__":
    unittest.main()
