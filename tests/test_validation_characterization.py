"""End-to-end characterization of the DIAS validation path.

Phase 3 collapses three near-identical validator classes into one. That is only
provably behaviour-preserving if the current output is pinned first, and the
existing tests are unit-level: none of them would catch a transposed einsum, a
changed successor-enumeration order, or a flipped logit channel.

Rather than train a model, this plants a schema by hand: the `stack` action of
blocks-3 is written into the checkpoint exactly, and the other two actions are
left inert. The expected result is therefore known independently of the
numbers -- `stack` must be recovered perfectly, and everything `newtower` and
`move` would have reached must show up as false negatives.

That makes the assertion sensitive in both directions, which P = R = 0 from an
untrained model would not be. It exercises the whole chain: threshold
binarization, slot activation, the add/delete and positive/negative channel
convention, the transpiler, domain untyping and merging, pymimir grounding, and
the micro-averaged successor comparison.
"""

import random
import tempfile
import unittest
from pathlib import Path

import torch

from dias.data.datamodule import PddlGraphDataModule
from dias.models.dias import DIAS
from dias.validation.dias_validate import ModelValidation

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAIN = PROJECT_ROOT / "pddl" / "blocks-3" / "domain.pddl"
TRAIN_PROBLEM = PROJECT_ROOT / "pddl" / "blocks-3" / "train.pddl"
TEST_PROBLEM = PROJECT_ROOT / "pddl" / "blocks-3" / "test-1.pddl"

# Softmax over the three channels turns these into ~(1, 0, 0) or ~(0, 1, 0),
# which the 0.5 threshold reads as a clean off/on.
OFF, ON = 10.0, 10.0

SAMPLE_SEED = 0
NUM_SAMPLES = 8
SAMPLE_POPULATION = 20


def datamodule_hparams() -> dict:
    return {
        "domain_path": str(DOMAIN),
        "problem_path": str(TRAIN_PROBLEM),
        "include_statics": True,
        "action_arguments": False,
        "strips_plus_actions": False,
    }


def build_checkpoint(directory: Path) -> str:
    """Write a checkpoint whose only learned action is a correct `stack`."""
    metadata = PddlGraphDataModule(**datamodule_hparams(), metadata_only=True)
    metadata.setup()
    relation = {name: i for i, name in enumerate(metadata.relation_names)}
    action = {name: i for i, name in enumerate(metadata.action_names)}

    model_hparams = {
        "num_objects": metadata.num_objects,
        "arity_relations": list(metadata.arity_relations),
        "num_actions": metadata.num_actions,
        "object_dim": 8,
        "num_slots": max(metadata.action_arities),
        "action_arities": list(metadata.action_arities),
        "action_arg_positions": [list(p) for p in metadata.action_arg_positions],
        "successor_relations": True,
        "parameter_plotting": None,
        "batch_size": 2,
        "encoder_hidden_dims": [4, 8, 8],
    }
    torch.manual_seed(5)
    model = DIAS(**model_hparams)

    stack = action["stack"]
    with torch.no_grad():
        effects = model.graph_edit.effects
        preconditions = model.graph_edit.preconditions
        for tensor in (effects, preconditions):
            tensor[..., 0] = OFF
            tensor[..., 1] = 0.0
            tensor[..., 2] = 0.0

        model.graph_edit.slot_weight_logits.fill_(-ON)
        model.graph_edit.slot_weight_logits[stack, 0] = ON
        model.graph_edit.slot_weight_logits[stack, 1] = ON

        def enable(tensor, name, row, column, channel):
            tensor[stack, relation[name], row, column, 0] = 0.0
            tensor[stack, relation[name], row, column, channel] = ON

        # stack(?bm=slot0, ?bt=slot1):
        #   pre  (clear ?bm) (clear ?bt) (on-table ?bm) (not (eq ?bm ?bt))
        #   eff  (on ?bm ?bt) (not (clear ?bt)) (not (on-table ?bm))
        enable(preconditions, "clear", 0, 0, 1)
        enable(preconditions, "clear", 1, 1, 1)
        enable(preconditions, "on-table", 0, 0, 1)
        enable(preconditions, "eq", 0, 1, 2)
        enable(effects, "on", 0, 1, 1)
        enable(effects, "clear", 1, 1, 2)
        enable(effects, "on-table", 0, 0, 2)

    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": model_hparams,
        "datamodule_hyper_parameters": datamodule_hparams(),
        "pytorch-lightning_version": "2.6.1",
        "epoch": 1,
        "global_step": 10,
    }
    model.teardown("build")

    checkpoints = directory / "dias" / "run0" / "checkpoints"
    checkpoints.mkdir(parents=True)
    torch.save(checkpoint, checkpoints / "best-epoch=1-step=10.ckpt")
    return "run0"


class ValidationCharacterizationTest(unittest.TestCase):
    def validate(self) -> dict:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        model_id = build_checkpoint(directory)
        # setup() draws its states with the global `random` module.
        random.seed(SAMPLE_SEED)
        validation = ModelValidation(
            model_id, str(directory), problem_path=str(TEST_PROBLEM)
        )
        # ModelValidation writes the merged domain and problem to /tmp with
        # delete=False and never removes them, so the test does it instead.
        # Phase 2 moves this into a context-managed helper.
        for attribute in ("merged_domain_filename", "merged_problem_filename"):
            self.addCleanup(
                Path(getattr(validation, attribute)).unlink, missing_ok=True
            )
        validation.setup(num_samples=NUM_SAMPLES, sample_population=SAMPLE_POPULATION)
        return validation.validate()

    def test_planted_schema_reproduces_pinned_metrics(self):
        result = self.validate()

        self.assertEqual(result["total"], NUM_SAMPLES)
        # 54 of the original successors are reached, 8 spurious, 138 missed.
        self.assertAlmostEqual(result["precision"], 54 / 62, places=12)
        self.assertAlmostEqual(result["recall"], 54 / 192, places=12)
        self.assertEqual(result["matches"], 0)
        self.assertEqual(result["accuracy"], 0.0)

    def test_only_the_planted_action_is_recovered(self):
        stats = self.validate()["action_stats"]

        self.assertEqual(stats["stack"], {"tp": 54, "fp": 0, "fn": 0, "conflicts": 0})
        self.assertEqual(
            stats["newtower"], {"tp": 0, "fp": 8, "fn": 28, "conflicts": 36}
        )
        self.assertEqual(stats["move"], {"tp": 0, "fp": 8, "fn": 110, "conflicts": 118})

    def test_repeated_validation_is_deterministic(self):
        self.assertEqual(self.validate(), self.validate())


if __name__ == "__main__":
    unittest.main()
