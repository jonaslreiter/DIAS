import unittest

import torch

from dias.data.datamodule import PddlGraphDataModule, PddlTransitionDataset


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


class DataModuleTest(unittest.TestCase):
    def test_transition_dataset_exposes_samples_and_length(self):
        dataset = PddlTransitionDataset(make_data())
        self.assertEqual(len(dataset), 2)
        sample = dataset[1]
        self.assertTrue(torch.equal(sample["state"], dataset.states[1]))
        self.assertEqual(sample["action_index"].item(), 1)
        self.assertEqual(sample["action_args"].tolist(), [1, 0])

    def test_sampling_validation_rejects_invalid_combinations(self):
        def module(**values):
            data_module = PddlGraphDataModule.__new__(PddlGraphDataModule)
            defaults = {
                "pt_file": None,
                "metadata_only": False,
                "min_per_label_train": None,
                "min_per_label_valid": None,
                "num_train": None,
                "num_valid": None,
                "max_per_label_train": None,
                "max_per_label_valid": None,
            }
            defaults.update(values)
            for name, value in defaults.items():
                setattr(data_module, name, value)
            return data_module

        with self.assertRaisesRegex(ValueError, "at least one"):
            module()._validate_sampling_config()
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            module(
                min_per_label_train=0, min_per_label_valid=1
            )._validate_sampling_config()
        with self.assertRaisesRegex(ValueError, "set together"):
            module(
                num_train=2, num_valid=1, max_per_label_train=1
            )._validate_sampling_config()
        with self.assertRaisesRegex(ValueError, "must be >= min"):
            module(
                min_per_label_train=2,
                min_per_label_valid=2,
                max_per_label_train=1,
                max_per_label_valid=2,
            )._validate_sampling_config()
        with self.assertRaisesRegex(ValueError, "Ratio mismatch"):
            module(
                min_per_label_train=1,
                min_per_label_valid=1,
                num_train=3,
                num_valid=1,
            )._validate_sampling_config()

    def test_cap_per_label_limits_each_action(self):
        data_module = PddlGraphDataModule.__new__(PddlGraphDataModule)
        data_module.max_per_label_train = 1
        data_module.max_per_label_valid = 1
        samples = [
            {"action": name, "id": i} for i, name in enumerate(["a", "a", "a", "b"])
        ]
        kept = data_module._cap_per_label(samples)
        self.assertEqual([sample["id"] for sample in kept], [0, 1, 3])

    def test_relation_true_ratio_handles_unary_diagonal_and_binary_regions(self):
        data_module = PddlGraphDataModule.__new__(PddlGraphDataModule)
        data_module.arity_relations = [1, 2]
        data_module._base = PddlTransitionDataset(make_data())
        self.assertEqual(data_module._compute_relation_true_ratio(), [1 / 7, 1 / 7])

    def test_empty_data_uses_metadata_and_keeps_shapes(self):
        data_module = PddlGraphDataModule.__new__(PddlGraphDataModule)
        data_module.metadata = {
            "objects": {"0": "a"},
            "object_name_to_idx": {"a": 0},
            "predicates": [{"name": "clear", "arity": 1, "static": False}],
            "actions": [{"name": "go", "arity": 0}],
        }
        data_module.arity_relations = [1]
        data_module.include_statics = False
        data_module.default_statics = ["object"]
        empty = data_module._empty_data()
        self.assertEqual(tuple(empty["states"].shape), (0, 1, 1, 1))
        self.assertEqual(tuple(empty["action_args"].shape), (0, 0))
        self.assertEqual(empty["relation_names"], ["clear"])


if __name__ == "__main__":
    unittest.main()
