import unittest

import torch
from torch_geometric.data import Batch, Data

from dias.models.utils import (
    batched_sinkhorn_d1d2,
    build_relation_change_edges,
    encode_relation_edges,
)


class ModelUtilsExtendedTest(unittest.TestCase):
    def test_encode_relation_edges_handles_empty_and_nonempty_inputs(self):
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_type = torch.tensor([0, 2])
        self.assertEqual(
            encode_relation_edges(edge_index, edge_type, 2, 3).tolist(), [3, 8]
        )
        empty = encode_relation_edges(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            2,
            3,
        )
        self.assertEqual(tuple(empty.shape), (0,))
        self.assertEqual(empty.dtype, torch.long)

    def test_build_relation_change_edges_reports_deletes_and_adds(self):
        before = Data(
            edge_index=torch.tensor([[0], [1]]),
            edge_attr=torch.tensor([0]),
            num_nodes=2,
        )
        after = Data(
            edge_index=torch.tensor([[1], [0]]),
            edge_attr=torch.tensor([1]),
            num_nodes=2,
        )
        batch_before = Batch.from_data_list([before])
        batch_after = Batch.from_data_list([after])
        edge_index, edge_type = build_relation_change_edges(
            batch_before, batch_after, num_relations=2, num_action_relations=1
        )
        self.assertEqual(tuple(edge_index.shape), (2, 4))
        self.assertEqual(edge_type.tolist(), [5, 7, 10, 12])

    def test_build_relation_change_edges_handles_none_and_no_changes(self):
        state = Data(
            edge_index=torch.tensor([[0], [1]]),
            edge_attr=torch.tensor([0]),
            num_nodes=2,
        )
        batch = Batch.from_data_list([state])
        empty_index, empty_type = build_relation_change_edges(
            batch, None, num_relations=1, num_action_relations=1
        )
        self.assertEqual(tuple(empty_index.shape), (2, 0))
        self.assertEqual(tuple(empty_type.shape), (0,))
        same_index, same_type = build_relation_change_edges(
            batch, batch, num_relations=1, num_action_relations=1
        )
        self.assertEqual(tuple(same_index.shape), (2, 0))
        self.assertEqual(tuple(same_type.shape), (0,))

    def test_sinkhorn_returns_finite_assignment_and_accepts_row_mask(self):
        logits = torch.zeros(1, 3, 2)
        result = batched_sinkhorn_d1d2(logits, max_iter=20)
        self.assertEqual(tuple(result.shape), (1, 3, 2))
        self.assertTrue(torch.isfinite(result).all())
        masked = batched_sinkhorn_d1d2(
            logits, max_iter=20, row_mask=torch.tensor([[False, True, False]])
        )
        self.assertTrue(torch.equal(masked[:, 1], torch.zeros(1, 2)))


if __name__ == "__main__":
    unittest.main()
