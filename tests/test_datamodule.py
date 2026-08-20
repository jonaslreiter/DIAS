import io
import unittest

import torch

from checkpoints import DOMAIN, TRAIN_PROBLEM
from dias.data.datamodule import PddlGraphDataModule
from dias.data.noise import ExpectedCount, Ratio


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


class QuotaFloorTest(unittest.TestCase):
    def test_a_raised_minimum_does_not_reach_the_saved_hyperparameters(self):
        """An archived run must record the value its config asked for."""
        module = PddlGraphDataModule(
            domain_path=str(DOMAIN),
            problem_path=str(TRAIN_PROBLEM),
            min_per_label_train=2,
            min_per_label_valid=1,
            min_per_label_train_floor=20,
            min_per_label_valid_floor=10,
        )
        self.assertEqual(module.quota.split_weights, (20, 10))
        self.assertEqual(module.hparams["min_per_label_train"], 2)
        self.assertEqual(module.hparams["min_per_label_valid"], 1)


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
