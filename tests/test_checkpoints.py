"""Reading checkpoints of the current format and of format 2, in which the paper's runs were written."""

import unittest

import torch

from checkpoints import build_model, save
from dias.models.checkpoints import CHECKPOINT_VERSION, VERSION_KEY
from dias.models.dias import DIAS
from support import temp_dir


class CheckpointTest(unittest.TestCase):
    def write(self, version=CHECKPOINT_VERSION, **hparams):
        """A checkpoint of a small blocks-3 model in format `version`, with `hparams` stored in place of the model's."""
        directory = temp_dir(self)
        model, _ = build_model(random_node_features=False, derivation_threshold=0.75)
        save(directory, model, dict(model.hparams))
        path = next((directory / "dias" / "run0" / "checkpoints").glob("*.ckpt"))
        checkpoint = torch.load(path, weights_only=False)
        checkpoint[VERSION_KEY] = version
        if hparams:
            checkpoint["hyper_parameters"] = hparams
        torch.save(checkpoint, path)
        return path, model

    def format_2(self, **changes):
        """The model's hparams as format 2 named them."""
        _, model = self.write()
        hparams = dict(model.hparams)
        hparams["random_node_embeddings"] = hparams.pop("random_node_features")
        hparams["schema_threshold"] = hparams.pop("derivation_threshold")
        hparams["learn_preconditions"] = hparams.pop("gate_preconditions")
        hparams["state_dependent_effects"] = hparams.pop("mlp_effects")
        return (
            hparams | {"successor_relations": True, "pcgrad_norm_cap": True} | changes
        )

    def test_a_current_checkpoint_loads_as_saved(self):
        path, model = self.write()
        loaded = DIAS.load_from_checkpoint(str(path))
        self.assertEqual(dict(loaded.hparams), dict(model.hparams))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], value)

    def test_format_2_names_are_read_as_the_current_ones(self):
        path, model = self.write(2, **self.format_2())
        loaded = DIAS.load_from_checkpoint(str(path))
        self.assertFalse(loaded.hparams.random_node_features)
        self.assertEqual(loaded.hparams.derivation_threshold, 0.75)
        self.assertNotIn("successor_relations", loaded.hparams)
        self.assertEqual(dict(loaded.hparams), dict(model.hparams))

    def test_a_retired_argument_at_another_value_is_refused(self):
        path, _ = self.write(2, **self.format_2(successor_relations=False))
        with self.assertRaises(ValueError):
            DIAS.load_from_checkpoint(str(path))

    def test_formats_before_2_and_after_the_current_one_are_refused(self):
        for version in (None, 1, CHECKPOINT_VERSION + 1):
            with self.subTest(version=version):
                path, _ = self.write(version)
                with self.assertRaises(ValueError):
                    DIAS.load_from_checkpoint(str(path))


if __name__ == "__main__":
    unittest.main()
