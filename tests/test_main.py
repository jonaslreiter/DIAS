"""The learned domain a run writes beside each of its checkpoints."""

import unittest
from types import SimpleNamespace

import torch
from lightning.pytorch.plugins.io import TorchCheckpointIO

from checkpoints import build_model, datamodule_hparams, domain_schema
from dias.main import LEARNED_DOMAIN, LearnedDomainCheckpoints
from dias.schema.export import learned_domain
from support import temp_dir


class LearnedDomainCheckpointsTest(unittest.TestCase):
    def test_the_learned_domain_is_saved_and_removed_with_its_checkpoint(self):
        directory = temp_dir(self)
        model, _ = build_model()
        datamodule = SimpleNamespace(
            schema=domain_schema(), hparams=datamodule_hparams()
        )
        trainer = SimpleNamespace(
            strategy=SimpleNamespace(checkpoint_io=TorchCheckpointIO()),
            datamodule=datamodule,
        )
        callback = LearnedDomainCheckpoints()
        callback.setup(trainer, model, "fit")
        callback.setup(trainer, model, "validate")
        self.assertIsInstance(trainer.strategy.checkpoint_io.io, TorchCheckpointIO)

        checkpoint = {"state_dict": model.state_dict()}
        callback.on_save_checkpoint(trainer, model, checkpoint)
        path = directory / "best-epoch=1-step=10.ckpt"
        trainer.strategy.checkpoint_io.save_checkpoint(checkpoint, path)
        written = path.with_suffix(".pddl")
        self.assertEqual(
            written.read_text(),
            learned_domain(model, datamodule.schema, datamodule.hparams, 0.5),
        )
        self.assertNotIn(LEARNED_DOMAIN, torch.load(path, weights_only=False))

        trainer.strategy.checkpoint_io.remove_checkpoint(path)
        self.assertFalse(path.exists())
        self.assertFalse(written.exists())


if __name__ == "__main__":
    unittest.main()
