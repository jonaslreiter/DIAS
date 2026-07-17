import unittest

import torch

from dias.utils import (
    LossWeightSchedule,
    LossWeightScheduleCollection,
    adj_to_pyg_data,
    gmean,
    resolve_label_names,
)


class UtilsExtendedTest(unittest.TestCase):
    def test_linear_and_exponential_schedules(self):
        linear = LossWeightSchedule({"type": "linear", "keypoints": {0: 0.0, 10: 1.0}})
        self.assertEqual(linear(0), 0.0)
        self.assertEqual(linear(5), 0.5)
        self.assertEqual(linear(20), 1.0)

        exponential = LossWeightSchedule(
            {"type": "exponential", "v0": 1.0, "vT": 4.0, "T": 2}
        )
        self.assertEqual(exponential(0), 1.0)
        self.assertAlmostEqual(exponential(1), 2.0)

    def test_schedule_collection_updates_and_iterates(self):
        collection = LossWeightScheduleCollection(
            {"main": {"type": "linear", "keypoints": {0: 1.0, 2: 3.0}}}
        )
        self.assertEqual(list(collection), ["main"])
        self.assertEqual(collection.update_loss_weights(1), {"main": 2.0})
        self.assertEqual(collection["main"], 2.0)

    def test_resolve_label_names_fills_missing_values(self):
        class DataModule:
            action_names = ["move"]
            relation_names = None
            object_names = {"0": "robot"}

        self.assertEqual(
            resolve_label_names(DataModule(), 2, 2, 2),
            (["move", "Action 1"], ["Rel 0", "Rel 1"], ["robot", "Node 1"]),
        )

    def test_gmean_clamps_zeros_and_supports_temperature(self):
        values = torch.tensor([[0.0, 1.0], [0.25, 1.0]])
        result = gmean(values, dim=1)
        self.assertGreater(result[0], 0)
        self.assertAlmostEqual(result[1].item(), 0.5)
        self.assertTrue(torch.equal(gmean(torch.ones(2, 3), dim=1), torch.ones(2)))

    def test_adj_to_pyg_data_keeps_relation_edges(self):
        adjacency = torch.tensor(
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ],
            dtype=torch.float32,
        )
        data = adj_to_pyg_data(adjacency, num_objects=2)
        self.assertTrue(torch.equal(data.edge_index, torch.tensor([[0, 0], [0, 1]])))
        self.assertTrue(torch.equal(data.edge_attr, torch.tensor([0, 1])))
        self.assertEqual(data.num_nodes, 2)

    def test_adj_to_pyg_data_handles_empty_graph(self):
        data = adj_to_pyg_data(torch.zeros(1, 2, 2), num_objects=2)
        self.assertEqual(tuple(data.edge_index.shape), (2, 0))
        self.assertEqual(tuple(data.edge_attr.shape), (0,))


if __name__ == "__main__":
    unittest.main()
