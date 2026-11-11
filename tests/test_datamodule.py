import io
import unittest

import torch

from checkpoints import DOMAIN, TRAIN_PROBLEM
from dias.data.datamodule import PddlGraphDataModule
from dias.data.noise import ExpectedCount, Ratio


class NoiseArgumentTest(unittest.TestCase):
    """The noise model is stored as primitives."""

    def build(self, **kwargs) -> PddlGraphDataModule:
        return PddlGraphDataModule(
            domain_path=str(DOMAIN),
            problem_path=str(TRAIN_PROBLEM),
            min_per_action_train=2,
            min_per_action_valid=1,
            **kwargs,
        )

    def test_a_configured_model_arrives_as_itself(self):
        noise = ExpectedCount(removed_atoms=1.0, added_atoms=2.0)
        self.assertEqual(self.build(noise=noise).noise, noise)

    def test_the_mapping_a_checkpoint_stores_rebuilds_the_model(self):
        stored = self.build(noise=Ratio(remove=0.25)).hparams["noise"]
        self.assertEqual(self.build(noise=stored).noise, Ratio(remove=0.25))

    def test_the_stored_form_survives_a_weights_only_load(self):
        stored = self.build(noise=ExpectedCount(added_atoms=1.0)).hparams["noise"]
        buffer = io.BytesIO()
        torch.save(stored, buffer)
        self.assertEqual(torch.load(io.BytesIO(buffer.getvalue())), stored)

    def test_no_noise_is_the_default(self):
        module = self.build()
        self.assertIsNone(module.noise)
        self.assertIsNone(module.hparams["noise"])


class DataloaderTest(unittest.TestCase):

    def build(self, **kwargs) -> PddlGraphDataModule:
        module = PddlGraphDataModule(
            domain_path=str(DOMAIN),
            problem_path=str(TRAIN_PROBLEM),
            min_per_action_train=4,
            min_per_action_valid=2,
            batch_size=4,
            **kwargs,
        )
        module.setup()
        return module

    def first_batch(self, loader):
        batch_state, batch_next_state, *_ = next(iter(loader))
        self.assertEqual(batch_state.num_graphs, batch_next_state.num_graphs)
        return batch_state

    def test_both_loaders_stream_batches_of_the_configured_size(self):
        module = self.build()
        for loader in (module.train_dataloader(), module.val_dataloader()):
            self.assertEqual(self.first_batch(loader).y_action.numel(), 4)


if __name__ == "__main__":
    unittest.main()
