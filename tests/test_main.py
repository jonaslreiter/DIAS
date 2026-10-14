"""The learned domain a run writes beside each of its checkpoints."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from checkpoints import build_model, datamodule_hparams, domain_schema
from dias.main import SchemaCheckpoint
from dias.schema.export import learned_domain
from support import temp_dir


class SchemaCheckpointTest(unittest.TestCase):
    def test_the_learned_domain_is_written_and_removed_with_its_checkpoint(self):
        directory = temp_dir(self)
        model, _ = build_model()
        datamodule = SimpleNamespace(
            schema=domain_schema(), hparams=datamodule_hparams()
        )
        trainer = SimpleNamespace(
            is_global_zero=True,
            global_step=10,
            loggers=[],
            lightning_module=model,
            datamodule=datamodule,
            save_checkpoint=lambda path, weights_only: Path(path).touch(),
            strategy=SimpleNamespace(
                remove_checkpoint=lambda path: Path(path).unlink()
            ),
        )
        checkpoint = directory / "best-epoch=1-step=10.ckpt"
        callback = SchemaCheckpoint(dirpath=directory)

        callback._save_checkpoint(trainer, str(checkpoint))
        written = checkpoint.with_suffix(".pddl")
        self.assertEqual(
            written.read_text(),
            learned_domain(model, datamodule.schema, datamodule.hparams, 0.5),
        )

        callback._remove_checkpoint(trainer, str(checkpoint))
        self.assertFalse(checkpoint.exists())
        self.assertFalse(written.exists())


if __name__ == "__main__":
    unittest.main()
