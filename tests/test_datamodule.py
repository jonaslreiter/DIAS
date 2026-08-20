import io
import unittest

import torch

from checkpoints import DOMAIN, TRAIN_PROBLEM
from dias.data.datamodule import PddlGraphDataModule, PddlTransitionDataset
from dias.data.noise import ExpectedCount, Ratio


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

    def test_min_per_label_floors_only_raise_the_per_domain_values(self):
        def module(**values):
            data_module = PddlGraphDataModule.__new__(PddlGraphDataModule)
            for name, value in values.items():
                setattr(data_module, name, value)
            return data_module

        raised = module(
            min_per_label_train=20,
            min_per_label_valid=10,
            min_per_label_train_floor=200,
            min_per_label_valid_floor=100,
        )
        raised._apply_min_per_label_floors()
        self.assertEqual(
            (raised.min_per_label_train, raised.min_per_label_valid), (200, 100)
        )

        kept = module(
            min_per_label_train=1000,
            min_per_label_valid=500,
            min_per_label_train_floor=200,
            min_per_label_valid_floor=100,
        )
        kept._apply_min_per_label_floors()
        self.assertEqual(
            (kept.min_per_label_train, kept.min_per_label_valid), (1000, 500)
        )

        # No floors / sampling by num_* instead: nothing to do.
        untouched = module(
            min_per_label_train=None,
            min_per_label_valid=None,
            min_per_label_train_floor=200,
            min_per_label_valid_floor=None,
        )
        untouched._apply_min_per_label_floors()
        self.assertIsNone(untouched.min_per_label_train)

    def test_cap_per_label_limits_each_action(self):
        data_module = PddlGraphDataModule.__new__(PddlGraphDataModule)
        data_module.max_per_label_train = 1
        data_module.max_per_label_valid = 1
        samples = [
            {"action": name, "id": i} for i, name in enumerate(["a", "a", "a", "b"])
        ]
        kept = data_module._cap_per_label(samples)
        self.assertEqual([sample["id"] for sample in kept], [0, 1, 3])


class NoiseArgumentTest(unittest.TestCase):
    """The noise model is a dataclass, and checkpoints may hold only primitives."""

    def build(self, **kwargs) -> PddlGraphDataModule:
        return PddlGraphDataModule(
            domain_path=str(DOMAIN),
            problem_path=str(TRAIN_PROBLEM),
            min_per_label_train=2,
            min_per_label_valid=1,
            **kwargs,
        )

    def test_a_configured_model_arrives_as_itself(self):
        noise = ExpectedCount(removed_atoms=1.0, added_atoms=2.0)
        self.assertEqual(self.build(noise=noise).noise, noise)

    def test_the_mapping_a_checkpoint_stores_rebuilds_the_model(self):
        stored = self.build(noise=Ratio(remove=0.25)).hparams["noise"]
        self.assertEqual(self.build(noise=stored).noise, Ratio(remove=0.25))

    def test_the_stored_form_survives_a_weights_only_load(self):
        """A dataclass here would make torch.load refuse the whole checkpoint."""
        stored = self.build(noise=ExpectedCount(added_atoms=1.0)).hparams["noise"]
        buffer = io.BytesIO()
        torch.save(stored, buffer)
        self.assertEqual(torch.load(io.BytesIO(buffer.getvalue())), stored)

    def test_no_noise_is_the_default(self):
        module = self.build()
        self.assertIsNone(module.noise)
        self.assertIsNone(module.hparams["noise"])


if __name__ == "__main__":
    unittest.main()
