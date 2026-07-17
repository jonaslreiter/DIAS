import unittest

import torch

from dias.models.prediction import PredictionModule


class PredictionModuleTest(unittest.TestCase):
    def test_dense_forward_handles_forced_arguments_and_tracks_training_state(self):
        torch.manual_seed(4)
        module = PredictionModule(
            num_objects=2,
            arity_relations=[1, 2],
            num_actions=1,
            num_slots=2,
            object_dim=3,
            force_action_args=True,
            learn_preconditions=False,
            action_arities=[2],
        )
        module.train()
        output = module(
            torch.randn(2, 3),
            y_action=torch.tensor([0]),
            y_action_args=torch.tensor([[1, -1]]),
            adj_dense=torch.zeros(1, 2, 2, 2),
        )
        self.assertEqual(tuple(output[0].shape), (1, 2, 2, 2))
        self.assertTrue(torch.isfinite(output[0]).all())
        self.assertEqual(module._last_y_action.tolist(), [0])
        self.assertEqual(tuple(module._last_selection.shape), (1, 2, 2))
        self.assertEqual(output[9][0, 0, 1].item(), 0.0)
        self.assertTrue(torch.isneginf(output[9]).any())

    def test_per_action_slots_masks_unused_slots_and_supports_graph_input(self):
        torch.manual_seed(7)
        module = PredictionModule(
            num_objects=2,
            arity_relations=[1, 2],
            num_actions=2,
            num_slots=2,
            object_dim=3,
            per_action_slots=True,
            action_arities=[1, 2],
        )
        module.eval()
        edge_index = torch.tensor([[0, 1], [0, 1]])
        edge_type = torch.tensor([0, 1])
        output = module(
            torch.randn(2, 3),
            edge_index=edge_index,
            edge_type=edge_type,
            batch_idx=torch.zeros(2, dtype=torch.long),
            y_action=torch.tensor([0]),
            y_action_args=torch.tensor([[-1, -1]]),
        )
        self.assertTrue(torch.all(output[8][:, 1] == 0))
        self.assertEqual(tuple(output[1].shape), (1, 2, 2, 2))
        self.assertTrue(torch.isfinite(output[0]).all())

    def test_forward_without_preconditions_does_not_gate_delta(self):
        torch.manual_seed(9)
        gated = PredictionModule(
            2, [2], 1, 1, 2, learn_preconditions=True, action_arities=[1]
        )
        ungated = PredictionModule(
            2, [2], 1, 1, 2, learn_preconditions=False, action_arities=[1]
        )
        ungated.load_state_dict(gated.state_dict())
        kwargs = {
            "y_action": torch.tensor([0]),
            "y_action_args": torch.tensor([[-1]]),
            "adj_dense": torch.zeros(1, 1, 2, 2),
        }
        node_embeddings = torch.randn(2, 2)
        gated_output = gated(node_embeddings, **kwargs)
        ungated_output = ungated(node_embeddings, **kwargs)
        self.assertFalse(torch.equal(gated_output[0], ungated_output[0]))
        self.assertTrue(torch.equal(gated_output[-1], ungated_output[-1]))


if __name__ == "__main__":
    unittest.main()
