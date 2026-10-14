"""The learned domain a run writes beside each of its checkpoints."""

import unittest
from types import SimpleNamespace

from checkpoints import build_model, datamodule_hparams, domain_schema
from dias.main import SchemaCheckpoint
from dias.schema.export import learned_domain
from support import temp_dir


class SchemaCheckpointTest(unittest.TestCase):
    def test_the_learned_domain_follows_the_checkpoints_held(self):
        directory = temp_dir(self)
        model, _ = build_model()
        datamodule = SimpleNamespace(
            schema=domain_schema(), hparams=datamodule_hparams()
        )
        trainer = SimpleNamespace(
            is_global_zero=True, lightning_module=model, datamodule=datamodule
        )
        callback = SchemaCheckpoint(dirpath=directory)
        checkpoint = directory / "best-epoch=1-step=10.ckpt"
        checkpoint.touch()
        callback.best_model_path = str(checkpoint)

        callback.on_train_end(trainer, model)
        written = checkpoint.with_suffix(".pddl")
        self.assertEqual(
            written.read_text(),
            learned_domain(model, datamodule.schema, datamodule.hparams, 0.5),
        )

        checkpoint.unlink()
        callback.on_train_end(trainer, model)
        self.assertFalse(written.exists())


if __name__ == "__main__":
    unittest.main()
