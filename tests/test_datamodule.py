import io
import unittest

import torch

from checkpoints import DOMAIN, TRAIN_PROBLEM
from dias.data.datamodule import PddlGraphDataModule
from dias.data.noise import ExpectedCount, Ratio


class SamplingConfigTest(unittest.TestCase):
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


class DataloaderTest(unittest.TestCase):
    """One pass through setup and both loaders, on the smallest real domain."""

    def build(self, **kwargs) -> PddlGraphDataModule:
        module = PddlGraphDataModule(
            domain_path=str(DOMAIN),
            problem_path=str(TRAIN_PROBLEM),
            min_per_label_train=4,
            min_per_label_valid=2,
            batch_size=4,
            **kwargs,
        )
        module.setup()
        return module

    def first_batch(self, loader):
        batch_s1, batch_s2, *_ = next(iter(loader))
        self.assertEqual(batch_s1.num_graphs, batch_s2.num_graphs)
        return batch_s1

    def test_stratified_batches_stream_without_exhausting(self):
        module = self.build(batch_sampling_strategy="stratified")
        loader = module.train_dataloader()
        self.assertEqual(self.first_batch(loader).y_action.numel(), 4)

    def test_random_batches_walk_the_split(self):
        module = self.build(batch_sampling_strategy="random")
        for loader in (module.train_dataloader(), module.val_dataloader()):
            self.assertEqual(self.first_batch(loader).y_action.numel(), 4)


if __name__ == "__main__":
    unittest.main()
