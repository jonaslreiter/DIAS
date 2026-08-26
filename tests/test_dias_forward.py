import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dias.models.dias import DIAS


class DIASForwardTest(unittest.TestCase):
    def test_fixed_seed_forward_prediction(self):
        torch.manual_seed(17)
        model = DIAS(
            num_objects=2,
            arity_relations=[1, 2],
            num_actions=1,
            object_dim=4,
            num_slots=2,
            action_arities=[2],
            action_arg_positions=[[]],
            successor_relations=True,
        )
        model.eval()

        state = Data(
            edge_index=torch.tensor([[0, 1], [0, 1]]),
            edge_attr=torch.tensor([0, 1]),
            num_nodes=2,
        )
        state.y_action = torch.tensor([0])
        state.y_action_args = torch.tensor([[-1, -1]])
        next_state = Data(
            edge_index=torch.tensor([[0, 1], [0, 1]]),
            edge_attr=torch.tensor([0, 1]),
            num_nodes=2,
        )

        torch.manual_seed(23)
        prediction = model(
            Batch.from_data_list([state]), Batch.from_data_list([next_state])
        )

        self.assertEqual(prediction.adjacency.shape, (1, 2, 2, 2))
        # These values are what this seed builds; they moved once, deliberately,
        # when construction started drawing in the order the data flows.
        # float32 sums, so the tolerance has to sit above the accumulation error;
        # pinning more digits than this fails on any BLAS or torch version change.
        torch.testing.assert_close(
            prediction.adjacency.sum(), torch.tensor(2.2110720), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            prediction.delta.sum(), torch.tensor(0.2245795), rtol=0, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
