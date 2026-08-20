import unittest

import torch

from dias.models.prediction import PredictionModule


class PredictionModuleTest(unittest.TestCase):
    def test_dense_forward_handles_forced_arguments_and_tracks_training_state(self):
        torch.manual_seed(4)
        module = PredictionModule(
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
            adjacency=torch.zeros(1, 2, 2, 2),
        )
        self.assertEqual(tuple(output.adjacency.shape), (1, 2, 2, 2))
        self.assertTrue(torch.isfinite(output.adjacency).all())
        self.assertEqual(tuple(output.selection.shape), (1, 2, 2))
        self.assertEqual(output.selection_logits[0, 0, 1].item(), 0.0)
        self.assertTrue(torch.isneginf(output.selection_logits).any())

    def test_per_action_slots_masks_unused_slots(self):
        torch.manual_seed(7)
        module = PredictionModule(
            arity_relations=[1, 2],
            num_actions=2,
            num_slots=2,
            object_dim=3,
            per_action_slots=True,
            action_arities=[1, 2],
        )
        module.eval()
        output = module(
            torch.randn(2, 3),
            adjacency=torch.eye(2).expand(1, 2, 2, 2),
            y_action=torch.tensor([0]),
            y_action_args=torch.tensor([[-1, -1]]),
        )
        self.assertTrue(torch.all(output.selection[:, 1] == 0))
        self.assertEqual(tuple(output.input_adjacency.shape), (1, 2, 2, 2))
        self.assertTrue(torch.isfinite(output.adjacency).all())

    def test_forward_without_preconditions_does_not_gate_delta(self):
        torch.manual_seed(9)
        gated = PredictionModule(
            [2], 1, 1, 2, learn_preconditions=True, action_arities=[1]
        )
        ungated = PredictionModule(
            [2], 1, 1, 2, learn_preconditions=False, action_arities=[1]
        )
        ungated.load_state_dict(gated.state_dict())
        kwargs = {
            "y_action": torch.tensor([0]),
            "y_action_args": torch.tensor([[-1]]),
            "adjacency": torch.zeros(1, 1, 2, 2),
        }
        node_embeddings = torch.randn(2, 2)
        gated_output = gated(node_embeddings, **kwargs)
        ungated_output = ungated(node_embeddings, **kwargs)
        self.assertFalse(torch.equal(gated_output.adjacency, ungated_output.adjacency))
        self.assertTrue(torch.equal(gated_output.delta, ungated_output.delta))

    def test_the_temperature_reaches_the_precondition_gate(self):
        """Tau arrives as a plain value; the floor is the schedule's business."""
        torch.manual_seed(11)
        module = PredictionModule([2], 1, 1, 2, action_arities=[1])
        kwargs = {
            "y_action": torch.tensor([0]),
            "y_action_args": torch.tensor([[-1]]),
            "adjacency": torch.zeros(1, 1, 2, 2),
        }
        node_embeddings = torch.randn(2, 2)
        relaxed = module(node_embeddings, gmean_temperature=0.5, **kwargs)
        stricter = module(node_embeddings, gmean_temperature=0.4, **kwargs)
        self.assertFalse(torch.allclose(relaxed.adjacency, stricter.adjacency))
        self.assertTrue(
            torch.allclose(
                relaxed.adjacency,
                module(node_embeddings, gmean_temperature=0.5, **kwargs).adjacency,
            )
        )


if __name__ == "__main__":
    unittest.main()
