import unittest

import numpy as np
import torch

from dias.schema.derivation import derive_schema


class ProcessModelLogitsTest(unittest.TestCase):
    def test_derive_schema_forces_known_slots_and_masks_inactive_slots(self):
        shape = (2, 1, 3, 3, 3)
        precondition_logits = torch.zeros(shape)
        effect_logits = torch.zeros(shape)
        precondition_logits[..., 1] = 10.0
        effect_logits[..., -1] = 10.0
        slot_logits = torch.tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, -10.0]])
        result = derive_schema(
            effect_logits=effect_logits,
            precondition_logits=precondition_logits,
            slot_weight_logits=slot_logits,
            force_action_args=True,
            action_arguments=True,
            strips_plus_actions=False,
            strips_plus_positions={},
            action_names=["short", "long"],
            action_arities=[1, 2],
            num_slots=3,
        )
        add_effects, delete_effects, pos_preconditions, neg_preconditions, active = (
            result
        )
        self.assertEqual(active.tolist(), [[True, False, False], [True, True, False]])
        # One active slot for the first action, two for the second: 1 + 4 cells.
        self.assertEqual(delete_effects[0, 0, 0, 0], 1)
        self.assertEqual(delete_effects[0, 0, 0, 1], 0)
        self.assertEqual(delete_effects.sum(), 5)
        self.assertEqual(pos_preconditions.sum(), 5)
        self.assertEqual(add_effects.sum(), 0)
        self.assertEqual(neg_preconditions.sum(), 0)

    def test_derive_schema_uses_strips_plus_positions_and_random_flip(self):
        shape = (1, 1, 2, 2, 3)
        logits = torch.zeros(shape)
        slot_logits = torch.full((1, 2), -10.0)
        np.random.seed(4)
        result = derive_schema(
            logits,
            logits.clone(),
            slot_weight_logits=slot_logits,
            force_action_args=True,
            action_arguments=True,
            strips_plus_actions=True,
            strips_plus_positions={"action": [1]},
            action_names=["action"],
            action_arities=[2],
            num_slots=2,
            random_flip=True,
        )
        self.assertEqual(result[-1].tolist(), [[False, True]])
        self.assertEqual(sum(array.sum() for array in result[:4]), 4)

    def test_a_model_without_lifted_effects_decodes_only_its_preconditions(self):
        """The MLP-effects model predicts effects from the state, so there is no tensor."""
        shape = (1, 1, 2, 2, 3)
        precondition_logits = torch.zeros(shape)
        precondition_logits[..., 1] = 10.0
        add, delete, positive, negative, active = derive_schema(
            effect_logits=None,
            precondition_logits=precondition_logits,
            slot_weight_logits=torch.full((1, 2), 10.0),
            force_action_args=False,
            action_arguments=False,
            strips_plus_actions=False,
            strips_plus_positions={},
            action_names=["action"],
            action_arities=[2],
            num_slots=2,
        )
        self.assertIsNone(add)
        self.assertIsNone(delete)
        self.assertEqual(positive.sum(), 4)
        self.assertEqual(negative.sum(), 0)
        self.assertEqual(active.tolist(), [[True, True]])


if __name__ == "__main__":
    unittest.main()
