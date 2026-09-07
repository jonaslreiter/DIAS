"""Executable record of the unversioned checkpoint layout: the keys a checkpoint with no format stamp has.

Nothing here reads a real one; the dict is built by hand, so the layout stays documented after the code that produced it is gone.
"""

import unittest
import unittest.mock
from pathlib import Path

import torch

from checkpoints import pre_split_name
from dias.hparams import to_mapping
from dias.models.checkpoints import CHECKPOINT_VERSION, VERSION_KEY
from dias.models.dias import DIAS
from dias.schedules import Constant, Exponential
from support import temp_dir

# Constructor arguments this layout spelled differently. Stated independently of dias.models.checkpoints, so a rename that forgets its migration rule fails here.
UNVERSIONED_HPARAM_NAMES = {
    "relation_arities": "arity_relations",
    "pcgrad_norm_cap": "pcgrad_dyn_clip",
    # Renamed after version 1 was published, so a version-1 checkpoint spells it this way too; see PublishedVersionOneTest.
    "aux_loss_alpha": "aux_loss_weight",
}
_CURRENT_HPARAM_NAMES = {old: new for new, old in UNVERSIONED_HPARAM_NAMES.items()}

UNVERSIONED_HPARAMS = {
    "num_objects": 4,
    "arity_relations": [1, 2, 2],
    "num_actions": 2,
    "object_dim": 8,
    "num_slots": 3,
    "action_arities": [2, 2],
    "action_arg_positions": [[0, 1], [0]],
    "successor_relations": True,
    "parameter_plotting": None,
    "batch_size": 2,
    "learning_rate": 0.005,
    "aux_loss_weight": 0.2,
    "force_action_args": False,
    "per_action_slots": False,
    "learn_preconditions": True,
    "random_node_embeddings": True,
    "pcgrad_dyn_clip": True,
    "gmean_offset": 0.0,
    "state_dependent_effects": False,
    "use_slot_weights": True,
    "use_precondition_loss": True,
    "use_effect_loss": True,
    "use_pcgrad": True,
    "loss_weight_schedule": {
        "gmean_temperature": {"type": "exponential", "v0": 1.0, "vT": 0.1, "T": 500}
    },
}

# That layout kept the schema flat on the prediction module and declared the encoder's layers one attribute at a time; `checkpoints.pre_split_name` maps today's names back.
UNVERSIONED_SCHEMA_KEYS = (
    "graph_edit.slot_encodings",
    "graph_edit.slot_weight_logits",
    "graph_edit.preconditions",
    "graph_edit.effects",
)

# Derived from constructor arguments yet persisted then; both are rebuilt on load now, so the migration drops them.
UNVERSIONED_DERIVED_KEYS = ("_action_rel_lookup", "graph_edit.action_arities_t")
UNVERSIONED_ENCODER_KEYS = tuple(
    f"dynamics_model.conv{i}.{suffix}"
    for i in range(1, 7)
    for suffix in ("weight", "root", "bias")
) + tuple(
    f"dynamics_model.ln{i}.{suffix}"
    for i in range(1, 6)
    for suffix in ("weight", "bias")
)

# Stored by that layout, not declared by the current encoder. Pins that it is dropped on load.
RETIRED_ENCODER_KEY = "dynamics_model.conv6.bias"


# Hyper-parameters that layout had which the current constructor no longer takes; each needs a matching rule in dias.models.checkpoints.
UNVERSIONED_ONLY_HPARAMS = {
    "loss_weight_schedule",
    "gmean_offset",
    "parameter_plotting",
    # Stored by the model, read only by the datamodule.
    "batch_size",
}

# What it becomes after migration.
CURRENT_HPARAMS = {
    _CURRENT_HPARAM_NAMES.get(key, key): value
    for key, value in UNVERSIONED_HPARAMS.items()
    if key not in UNVERSIONED_ONLY_HPARAMS
} | {
    # A gmean_offset of 0.0 leaves the schedule unchanged.
    "precondition_tau": to_mapping(
        Exponential(start=1.0, floor=0.0, factor=0.1, every=500)
    ),
    # Added later, so an unversioned checkpoint gets the constructor default.
    "encoder_hidden_dims": None,
}


def constructor_args() -> dict:
    """UNVERSIONED_HPARAMS as the current constructor accepts them.

    Only the hyper-parameters differ between the two layouts; the tensor shapes do not, so this still builds a state dict with the unversioned key set.
    """
    return {
        _CURRENT_HPARAM_NAMES.get(k, k): v
        for k, v in UNVERSIONED_HPARAMS.items()
        if k not in UNVERSIONED_ONLY_HPARAMS
    }


def build_unversioned_checkpoint(marker: float = 0.125) -> dict:
    """An unversioned checkpoint whose schema parameters are all `marker`.

    Only the names differ between the two: the tensors are the same shapes, so the current model supplies them and they are renamed back.
    """
    reference = DIAS(**constructor_args())
    state_dict = {
        pre_split_name(key): value.clone()
        for key, value in reference.state_dict().items()
    }
    state_dict["graph_edit.action_arities_t"] = torch.tensor(
        UNVERSIONED_HPARAMS["action_arities"]
    )
    state_dict["_action_rel_lookup"] = torch.tensor([[0, 1], [2, -1]])
    # Put back by hand, at a value no zero-initialised bias would have, so a migration that kept it is visible rather than plausible.
    state_dict[RETIRED_ENCODER_KEY] = torch.full(
        (UNVERSIONED_HPARAMS["object_dim"],), 0.75
    )
    for key in UNVERSIONED_SCHEMA_KEYS:
        state_dict[key] = torch.full_like(state_dict[key], marker)
    return {
        "state_dict": state_dict,
        "hyper_parameters": dict(UNVERSIONED_HPARAMS),
        "pytorch-lightning_version": "2.6.1",
        "epoch": 3,
        "global_step": 300,
    }


class CheckpointLayoutTest(unittest.TestCase):
    def write_unversioned_checkpoint(self, **kwargs) -> str:
        directory = temp_dir(self)
        path = directory / "unversioned.ckpt"
        torch.save(build_unversioned_checkpoint(**kwargs), path)
        return str(path)

    def test_the_unversioned_key_set_is_exactly_as_recorded(self):
        state_dict = build_unversioned_checkpoint()["state_dict"]
        expected = set(
            UNVERSIONED_SCHEMA_KEYS
            + UNVERSIONED_DERIVED_KEYS
            + UNVERSIONED_ENCODER_KEYS
        )
        self.assertEqual(set(state_dict), expected)

    def test_the_current_key_set_is_the_unversioned_one_renamed(self):
        """Nothing derived from a constructor argument is checkpointed."""
        model = DIAS(**constructor_args())
        self.assertEqual(
            {pre_split_name(key) for key in model.state_dict()},
            set(UNVERSIONED_SCHEMA_KEYS + UNVERSIONED_ENCODER_KEYS)
            - {RETIRED_ENCODER_KEY},
        )

    def test_an_unversioned_checkpoint_loads_and_preserves_schema_values(self):
        path = self.write_unversioned_checkpoint(marker=0.125)
        model = DIAS.load_from_checkpoint(path, map_location="cpu")

        loaded = model.state_dict()
        for key in (k for k in loaded if pre_split_name(k) in UNVERSIONED_SCHEMA_KEYS):
            # That layout stored the unary off-diagonal mask in the parameter, so those cells are -inf; the mask is applied in the forward pass now.
            finite = torch.isfinite(loaded[key])
            self.assertTrue(finite.any(), f"{key} is entirely masked")
            torch.testing.assert_close(
                loaded[key][finite],
                torch.full_like(loaded[key][finite], 0.125),
                rtol=0,
                atol=0,
            )

    def test_the_final_encoder_bias_is_dropped_on_load(self):
        """A parameter no gradient ever reached, so nothing is lost with it."""
        model = DIAS.load_from_checkpoint(
            self.write_unversioned_checkpoint(), map_location="cpu"
        )
        state = model.state_dict()
        self.assertNotIn(RETIRED_ENCODER_KEY, {pre_split_name(k) for k in state})
        self.assertFalse([k for k in state if k.endswith("convs.5.bias")])
        # The convolution itself is still there, weights and all.
        self.assertIn("encoder.convs.5.weight", state)

    def test_unversioned_hparams_migrate_to_the_current_layout(self):
        path = self.write_unversioned_checkpoint()
        model = DIAS.load_from_checkpoint(path, map_location="cpu")
        self.assertEqual(dict(model.hparams), CURRENT_HPARAMS)

    def test_a_current_checkpoint_loads_without_migrating(self):
        """The stock loader must handle it; migration is the fallback only."""
        directory = temp_dir(self)
        path = directory / "current.ckpt"
        model = DIAS(**constructor_args())
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": "2.6.1",
            "epoch": 3,
            "global_step": 300,
        }
        model.on_save_checkpoint(checkpoint)
        torch.save(checkpoint, path)
        self.assertEqual(checkpoint[VERSION_KEY], CHECKPOINT_VERSION)

        with unittest.mock.patch(
            "dias.models.dias.migrate", side_effect=AssertionError("migrated")
        ):
            reloaded = DIAS.load_from_checkpoint(path, map_location="cpu")
        self.assertEqual(dict(reloaded.hparams), dict(model.hparams))

    def test_an_unversioned_offset_becomes_the_schedules_floor(self):
        """The noise runs' tau trajectory has to survive the fold exactly."""
        directory = temp_dir(self)
        path = directory / "noise.ckpt"
        checkpoint = build_unversioned_checkpoint()
        checkpoint["hyper_parameters"]["gmean_offset"] = 0.1
        checkpoint["hyper_parameters"]["loss_weight_schedule"] = {
            "gmean_temperature": {
                "type": "exponential",
                "v0": 0.9,
                "vT": 0.09,
                "T": 500,
            }
        }
        torch.save(checkpoint, path)
        model = DIAS.load_from_checkpoint(path, map_location="cpu")

        def unversioned_tau(step):
            return 0.9 * (0.09 / 0.9) ** (step / 500) + 0.1

        for step in (0, 100, 500, 1000, 10000):
            self.assertAlmostEqual(
                model.precondition_tau(step), unversioned_tau(step), places=12
            )

    def test_an_unversioned_mlp_checkpoint_is_refused(self):
        """Its effect MLP had no activations, so its weights cannot be reused."""
        directory = temp_dir(self)
        path = directory / "mlp.ckpt"
        checkpoint = build_unversioned_checkpoint()
        del checkpoint["state_dict"]["graph_edit.effects"]
        checkpoint["state_dict"]["graph_edit.effects_mlp.1.weight"] = torch.zeros(2, 2)
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "Retrain"):
            DIAS.load_from_checkpoint(path, map_location="cpu")

    def test_a_checkpoint_from_a_newer_dias_is_refused(self):
        """Loading it as-is would mean guessing what its keys mean."""
        directory = temp_dir(self)
        path = directory / "future.ckpt"
        checkpoint = build_unversioned_checkpoint()
        checkpoint[VERSION_KEY] = CHECKPOINT_VERSION + 1
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "newer version of dias"):
            DIAS.load_from_checkpoint(path, map_location="cpu")

    def test_an_unversioned_checkpoint_missing_a_renamed_key_still_loads(self):
        """Not every old checkpoint contains everything a later version renamed."""
        directory = temp_dir(self)
        path = directory / "sparse.ckpt"
        checkpoint = build_unversioned_checkpoint()
        del checkpoint["hyper_parameters"]["loss_weight_schedule"]
        torch.save(checkpoint, path)
        model = DIAS.load_from_checkpoint(path, map_location="cpu")
        self.assertEqual(model.precondition_tau, Constant(1.0))


class PublishedVersionOneTest(unittest.TestCase):
    """Version 1 covers two spellings of one argument, so the stamp cannot tell them apart.

    ``aux_loss_weight`` became ``aux_loss_alpha`` after version 1 was published and no version 2 exists yet, so a checkpoint stamped 1 may spell it either way and the read path renames it whatever the stamp says. Lightning would otherwise drop the unrecognised name and fall back to the default, silently changing what an archived run means.
    """

    def write_version_one(self, name: str, value: float) -> Path:
        args = constructor_args() | {"aux_loss_alpha": value}
        model = DIAS(**args)
        hparams = dict(model.hparams)
        hparams[name] = hparams.pop("aux_loss_alpha")
        path = temp_dir(self) / "v1.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": hparams,
                "pytorch-lightning_version": "2.6.1",
                "epoch": 3,
                "global_step": 300,
                VERSION_KEY: CHECKPOINT_VERSION,
            },
            path,
        )
        return path

    def test_the_published_spelling_keeps_its_value(self):
        path = self.write_version_one("aux_loss_weight", 0.3)
        model = DIAS.load_from_checkpoint(path, map_location="cpu")
        self.assertEqual(model.aux_loss_alpha, 0.3)
        self.assertNotIn("aux_loss_weight", model.hparams)

    def test_the_current_spelling_is_untouched(self):
        path = self.write_version_one("aux_loss_alpha", 0.3)
        model = DIAS.load_from_checkpoint(path, map_location="cpu")
        self.assertEqual(model.aux_loss_alpha, 0.3)


if __name__ == "__main__":
    unittest.main()
