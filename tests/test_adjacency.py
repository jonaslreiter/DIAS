"""Tests for the dense adjacency representation."""

import unittest

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj

from dias.adjacency import adj_to_pyg_data, dense_from_edges, valid_cells


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


class DenseFromEdgesTest(unittest.TestCase):
    """`dense_from_edges` replaced a per-relation `to_dense_adj` loop.

    The two differ where an edge occurs twice: `to_dense_adj` sums, this assigns. That cannot happen for edges built by `adj_to_pyg_data` from a 0/1 tensor, since they come from its non-zero cells. That is the fact the round-trip below pins.
    """

    def states(self) -> torch.Tensor:
        torch.manual_seed(0)
        return (torch.rand(3, 2, 4, 4) > 0.7).float()

    def batch(self, states: torch.Tensor) -> Batch:
        return Batch.from_data_list([adj_to_pyg_data(s, 4) for s in states])

    def test_a_batch_of_states_survives_the_round_trip(self):
        states = self.states()
        batch = self.batch(states)
        rebuilt = dense_from_edges(batch.edge_index, batch.edge_attr, batch.batch, 2, 4)
        self.assertTrue(torch.equal(rebuilt, states))

    def test_it_agrees_with_the_per_relation_to_dense_adj_loop(self):
        batch = self.batch(self.states())
        expected = torch.stack(
            [
                to_dense_adj(
                    batch.edge_index[:, batch.edge_attr == relation],
                    batch.batch,
                    max_num_nodes=4,
                )
                for relation in range(2)
            ],
            dim=1,
        )
        rebuilt = dense_from_edges(batch.edge_index, batch.edge_attr, batch.batch, 2, 4)
        self.assertTrue(torch.equal(rebuilt, expected))

    def test_a_state_with_no_atoms_is_all_zeros(self):
        batch = self.batch(torch.zeros(2, 2, 4, 4))
        rebuilt = dense_from_edges(batch.edge_index, batch.edge_attr, batch.batch, 2, 4)
        self.assertEqual(tuple(rebuilt.shape), (2, 2, 4, 4))
        self.assertEqual(rebuilt.sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
