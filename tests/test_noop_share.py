"""Tests for `experiments/noop_share.py`'s count: a transition is a no-op when its successor equals its state."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from noop_share import count_noops


class CountNoopsTest(unittest.TestCase):
    def test_counts_unchanged_transitions_per_action(self):
        states = torch.zeros(3, 2, 4, 4)
        next_states = states.clone()
        next_states[1, 0, 1, 2] = 1  # only the second transition changes the state
        base = SimpleNamespace(
            states=states,
            next_states=next_states,
            action_indices=torch.tensor([0, 0, 1]),
            action_names=["take_image", "turn_to"],
        )
        self.assertEqual(count_noops(base), {"take_image": (1, 2), "turn_to": (1, 1)})


if __name__ == "__main__":
    unittest.main()
