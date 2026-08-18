"""Executable record of the pre-refactor ("v1") checkpoint layout.

The paper's trained checkpoints have exactly the keys listed below. Nothing in
this file reads a real checkpoint: the dict is built by hand so the layout stays
documented even after the code that produced it is gone.

Every refactor step that renames a state-dict key or regroups a hyper-parameter
must extend the migration in `dias.models` and leave this test passing. That is
what keeps each step independently landable without breaking archived runs.
"""

import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch

from dias.models.checkpoints import CHECKPOINT_VERSION, VERSION_KEY
from dias.models.dias import DIAS
from dias.schedules import Constant, Exponential, schedule_config

V1_HPARAMS = {
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

# The schema parameters live under `graph_edit.*` and the encoder under
# `dynamics_model.conv{1..6}` / `ln{1..5}`. `_action_rel_lookup` and
# `graph_edit.action_arities_t` are derived from constructor arguments yet are
# persisted, which is why renaming them needs a migration entry today.
V1_SCHEMA_KEYS = (
    "graph_edit.slot_encodings",
    "graph_edit.slot_weight_logits",
    "graph_edit.preconditions",
    "graph_edit.effects",
)
V1_DERIVED_KEYS = ("_action_rel_lookup", "graph_edit.action_arities_t")
V1_ENCODER_KEYS = tuple(
    f"dynamics_model.conv{i}.{suffix}"
    for i in range(1, 7)
    for suffix in ("weight", "root", "bias")
) + tuple(
    f"dynamics_model.ln{i}.{suffix}"
    for i in range(1, 6)
    for suffix in ("weight", "bias")
)


# Hyper-parameters v1 had that the current constructor no longer takes. Each
# refactor step that renames or regroups one adds it here and a rule to
# dias.models.checkpoints, which is what keeps archived runs loadable.
V1_ONLY_HPARAMS = {"loss_weight_schedule"}

# What v1 becomes after migration.
V2_HPARAMS = {
    key: value for key, value in V1_HPARAMS.items() if key not in V1_ONLY_HPARAMS
} | {
    "gmean_temperature": schedule_config(Exponential(v0=1.0, vT=0.1, T=500)),
    # Added after v1, so a v1 checkpoint gets the constructor default.
    "encoder_hidden_dims": None,
}


def constructor_args() -> dict:
    """V1_HPARAMS as the current constructor accepts them.

    Only the hyper-parameters differ between versions; the tensor shapes do
    not, so this still builds a state dict with the v1 key set.
    """
    return {k: v for k, v in V1_HPARAMS.items() if k not in V1_ONLY_HPARAMS}


def build_v1_checkpoint(marker: float = 0.125) -> dict:
    """A v1 checkpoint whose schema parameters are all set to `marker`."""
    reference = DIAS(**constructor_args())
    state_dict = {k: v.clone() for k, v in reference.state_dict().items()}
    reference.teardown("build")
    for key in V1_SCHEMA_KEYS:
        state_dict[key] = torch.full_like(state_dict[key], marker)
    return {
        "state_dict": state_dict,
        "hyper_parameters": dict(V1_HPARAMS),
        "pytorch-lightning_version": "2.6.1",
        "epoch": 3,
        "global_step": 300,
    }


class CheckpointLayoutTest(unittest.TestCase):
    def write_v1_checkpoint(self, **kwargs) -> str:
        directory = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(directory) / "v1.ckpt"
        torch.save(build_v1_checkpoint(**kwargs), path)
        return str(path)

    def test_v1_key_set_is_exactly_as_recorded(self):
        model = DIAS(**constructor_args())
        self.addCleanup(model.teardown, "test")
        expected = set(V1_SCHEMA_KEYS + V1_DERIVED_KEYS + V1_ENCODER_KEYS)
        self.assertEqual(set(model.state_dict()), expected)

    def test_v1_checkpoint_loads_and_preserves_schema_values(self):
        path = self.write_v1_checkpoint(marker=0.125)
        model = DIAS.load_from_checkpoint(path, map_location="cpu")
        self.addCleanup(model.teardown, "test")

        loaded = model.state_dict()
        for key in V1_SCHEMA_KEYS:
            # -inf entries are the unary off-diagonal mask, applied at construction.
            finite = torch.isfinite(loaded[key])
            self.assertTrue(finite.any(), f"{key} is entirely masked")
            torch.testing.assert_close(
                loaded[key][finite],
                torch.full_like(loaded[key][finite], 0.125),
                rtol=0,
                atol=0,
            )

    def test_v1_hparams_migrate_to_the_current_layout(self):
        path = self.write_v1_checkpoint()
        model = DIAS.load_from_checkpoint(path, map_location="cpu")
        self.addCleanup(model.teardown, "test")
        self.assertEqual(dict(model.hparams), V2_HPARAMS)

    def test_a_current_checkpoint_loads_without_migrating(self):
        """The stock loader must handle it; migration is the fallback only."""
        directory = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(directory) / "v2.ckpt"
        model = DIAS(**constructor_args())
        self.addCleanup(model.teardown, "test")
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
        self.addCleanup(reloaded.teardown, "test")
        self.assertEqual(dict(reloaded.hparams), dict(model.hparams))

    def test_a_v1_checkpoint_missing_a_renamed_key_still_loads(self):
        """Not every old checkpoint contains everything a later version renamed."""
        directory = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(directory) / "sparse.ckpt"
        checkpoint = build_v1_checkpoint()
        del checkpoint["hyper_parameters"]["loss_weight_schedule"]
        torch.save(checkpoint, path)
        model = DIAS.load_from_checkpoint(path, map_location="cpu")
        self.addCleanup(model.teardown, "test")
        self.assertEqual(model.gmean_temperature, Constant(1.0))


if __name__ == "__main__":
    unittest.main()
