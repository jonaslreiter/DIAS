import unittest

import numpy as np
import torch

from dias.validation.utils import process_model_logits


class ValidationUtilsTest(unittest.TestCase):
    def test_process_model_logits_forces_known_slots_and_masks_inactive_slots(self):
        shape = (2, 1, 3, 3, 3)
        precondition_logits = torch.zeros(shape)
        effect_logits = torch.zeros(shape)
        precondition_logits[..., 1] = 10.0
        effect_logits[..., -1] = 10.0
        slot_logits = torch.tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, -10.0]])
        result = process_model_logits(
            precondition_logits,
            effect_logits,
            slot_logits,
            force_action_args=True,
            action_arguments=True,
            strips_plus_actions=False,
            free_pos_map={},
            action_names=["short", "long"],
            action_arities=[1, 2],
            num_slots=3,
        )
        add_effects, delete_effects, pos_preconditions, neg_preconditions, active = (
            result
        )
        self.assertEqual(active.tolist(), [[True, False, False], [True, True, False]])
        self.assertEqual(add_effects[0, 0, 0, 0], 1)
        self.assertEqual(add_effects[0, 0, 0, 1], 0)
        self.assertEqual(delete_effects[0].sum(), 0)
        self.assertEqual(pos_preconditions.sum(), 0)
        self.assertEqual(neg_preconditions.sum(), 5)

    def test_process_model_logits_uses_strips_plus_positions_and_random_flip(self):
        shape = (1, 1, 2, 2, 3)
        logits = torch.zeros(shape)
        slot_logits = torch.full((1, 2), -10.0)
        np.random.seed(4)
        result = process_model_logits(
            logits,
            logits.clone(),
            slot_logits,
            force_action_args=True,
            action_arguments=True,
            strips_plus_actions=True,
            free_pos_map={"action": [1]},
            action_names=["action"],
            action_arities=[2],
            num_slots=2,
            random_flip=True,
        )
        self.assertEqual(result[-1].tolist(), [[False, True]])
        self.assertEqual(sum(array.sum() for array in result[:4]), 4)


if __name__ == "__main__":
    unittest.main()
