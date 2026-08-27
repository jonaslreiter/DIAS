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


class GraphDatasetTest(unittest.TestCase):
    """A sample is built from the noisy transition where there is one."""

    def sample(self, **kwargs):
        base = PddlTransitionDataset(make_data())
        return PddlGraphDataset(base, [0], **kwargs)[0]

    def test_a_clean_sample_encodes_the_stored_transition(self):
        _, _, _, state, _, _, _ = self.sample()
        self.assertTrue(
            torch.equal(state, PddlTransitionDataset(make_data()).states[0])
        )

    def test_noisy_states_replace_the_clean_ones_in_the_graph(self):
        noisy_states = PddlTransitionDataset(make_data()).states[[0]].clone()
        noisy_states[0, 1, 1, 0] = 1.0  # one atom of the binary relation added

        state_graph, _, _, state, _, _, _ = self.sample(noisy_states=noisy_states)
        self.assertTrue(torch.equal(state, noisy_states[0]))
        self.assertEqual(state_graph.edge_index.size(1), int(noisy_states[0].sum()))


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
