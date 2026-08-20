"""Tests for the three datasets a problem instance is read through."""

import unittest

import torch

from dias.data.datasets import (
    PddlGraphDataset,
    PddlTransitionDataset,
    StratifiedActionIterableDataset,
)


def make_data():
    return {
        "states": torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            ]
        ),
        "next_states": torch.zeros(2, 2, 2, 2),
        "action_indices": torch.tensor([0, 1]),
        "action_args": torch.tensor([[0, 1], [1, 0]]),
        "relation_names": ["clear", "on"],
        "relation_arities": [1, 2],
        "object_names": {"0": "a", "1": "b"},
        "action_names": ["move", "stack"],
        "action_arities": [2, 2],
        "num_objects": 2,
        "num_relations": 2,
    }


class TransitionDatasetTest(unittest.TestCase):
    def test_transition_dataset_exposes_samples_and_length(self):
        dataset = PddlTransitionDataset(make_data())
        self.assertEqual(len(dataset), 2)
        sample = dataset[1]
        self.assertTrue(torch.equal(sample["state"], dataset.states[1]))
        self.assertEqual(sample["action_index"].item(), 1)
        self.assertEqual(sample["action_args"].tolist(), [1, 0])


class FlipCountTest(unittest.TestCase):
    """The noise diagnostics ride on the graph, and only when there is noise."""

    def sample(self, **kwargs):
        base = PddlTransitionDataset(make_data())
        dataset = PddlGraphDataset(base, [0], **kwargs)
        return dataset[0][0]

    def test_a_clean_sample_carries_no_counts(self):
        self.assertFalse(hasattr(self.sample(), "num_flips_added"))

    def test_the_counts_cover_both_states_of_the_transition(self):
        clean = PddlTransitionDataset(make_data())
        noisy_states = clean.states[[0]].clone()
        noisy_states[0, 0, 0, 0] = 0.0  # one atom of the unary relation removed
        noisy_next = clean.next_states[[0]].clone()
        noisy_next[0, 1, 1, 0] = 1.0  # one atom of the binary relation added

        data = self.sample(noisy_states=noisy_states, noisy_next_states=noisy_next)
        self.assertEqual(data.num_flips_removed.tolist(), [1])
        self.assertEqual(data.num_flips_added.tolist(), [1])
        self.assertEqual(data.num_flips_removed_per_rel.tolist(), [[1, 0]])
        self.assertEqual(data.num_flips_added_per_rel.tolist(), [[0, 1]])
        self.assertEqual(data.num_true_predicates.tolist(), [1])


class StratifiedStreamTest(unittest.TestCase):
    """The stream is endless and balanced; training length comes from max_steps."""

    def stream(self, batch_size: int, indices=(0, 1)):
        return StratifiedActionIterableDataset(
            PddlTransitionDataset(make_data()), list(indices), batch_size, seed=0
        )

    def take(self, dataset, count: int) -> list[int]:
        actions = []
        for sample in dataset:
            actions.append(int(sample[0].y_action))
            if len(actions) == count:
                return actions

    def test_each_batch_covers_every_action_once(self):
        self.assertEqual(sorted(self.take(self.stream(2), 2)), [0, 1])

    def test_an_action_short_of_samples_is_still_drawn(self):
        """Draws are with replacement, so one sample can fill a whole batch."""
        self.assertEqual(self.take(self.stream(4, indices=[1]), 4), [1, 1, 1, 1])

    def test_an_empty_pool_is_refused_rather_than_hanging(self):
        with self.assertRaisesRegex(ValueError, "empty data pool"):
            self.stream(2, indices=[])


if __name__ == "__main__":
    unittest.main()
