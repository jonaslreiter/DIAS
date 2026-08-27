"""End-to-end characterization of the DIAS validation path.

Phase 3 collapses three near-identical validator classes into one. That is only provably behaviour-preserving if the current output is pinned first, and the existing tests are unit-level: none of them would catch a transposed einsum, a changed successor-enumeration order, or a flipped logit channel.

Rather than train a model, this plants a schema by hand: the `stack` action of blocks-3 is written into the checkpoint exactly, and the other two actions are left inert. The expected result is therefore known independently of the numbers: `stack` must be recovered perfectly, and everything `newtower` and `move` would have reached must show up as false negatives.

That makes the assertion sensitive in both directions, which P = R = 0 from an untrained model would not be. It exercises the whole chain: threshold binarization, slot activation, the add/delete and positive/negative channel convention, the transpiler, domain untyping and merging, pymimir grounding, and the micro-averaged successor comparison.
"""

import unittest
from pathlib import Path

import torch

from checkpoints import TEST_PROBLEM, build_model, domain_schema, save
from dias.validation.dias_validate import ModelValidation
from support import temp_dir

torch.set_num_threads(1)

# Softmax over the three channels turns these into ~(1, 0, 0) or ~(0, 1, 0), which the 0.5 threshold reads as a clean off/on.
OFF, ON = 10.0, 10.0

SAMPLE_SEED = 0
NUM_SAMPLES = 8
SAMPLE_POPULATION = 20


def build_checkpoint(directory: Path) -> str:
    """Write a checkpoint whose only learned action is a correct `stack`."""
    schema = domain_schema()
    relation = {name: i for i, name in enumerate(schema.relation_names)}
    action = {name: i for i, name in enumerate(schema.action_names)}
    model, hparams = build_model()

    stack = action["stack"]
    with torch.no_grad():
        effects = model.graph_edit.effects.logits
        preconditions = model.graph_edit.preconditions.logits
        for tensor in (effects, preconditions):
            tensor[..., 0] = OFF
            tensor[..., 1] = 0.0
            tensor[..., 2] = 0.0

        slot_weights = model.graph_edit.selector.slot_weight_logits
        slot_weights.fill_(-ON)
        slot_weights[stack, 0] = ON
        slot_weights[stack, 1] = ON

        def enable(tensor, name, row, column, channel):
            tensor[stack, relation[name], row, column, 0] = 0.0
            tensor[stack, relation[name], row, column, channel] = ON

        # stack(?bm=slot0, ?bt=slot1):
        # pre  (clear ?bm) (clear ?bt) (on-table ?bm) (not (eq ?bm ?bt))
        # eff  (on ?bm ?bt) (not (clear ?bt)) (not (on-table ?bm))
        enable(preconditions, "clear", 0, 0, 1)
        enable(preconditions, "clear", 1, 1, 1)
        enable(preconditions, "on-table", 0, 0, 1)
        enable(preconditions, "eq", 0, 1, 2)
        enable(effects, "on", 0, 1, 1)
        enable(effects, "clear", 1, 1, 2)
        enable(effects, "on-table", 0, 0, 2)

    return save(directory, model, hparams)


class ValidationCharacterizationTest(unittest.TestCase):
    def validate(self) -> dict:
        directory = temp_dir(self)
        model_id = build_checkpoint(directory)
        validation = ModelValidation(
            model_id, str(directory), problem_path=str(TEST_PROBLEM)
        )
        validation.setup(
            num_samples=NUM_SAMPLES,
            sample_population=SAMPLE_POPULATION,
            seed=SAMPLE_SEED,
        )
        return validation.validate()

    def test_planted_schema_reproduces_pinned_metrics(self):
        result = self.validate()

        self.assertEqual(result["total"], NUM_SAMPLES)
        # 54 of the original successors are reached, 138 missed, and 16 are spurious: 8 states, each produced by both of the inert learned actions. Counts are attributed to the action that produced them, so those 8 count twice: two operators, two wrong predictions.
        self.assertAlmostEqual(result["precision"], 54 / 70, places=12)
        self.assertAlmostEqual(result["recall"], 54 / 192, places=12)
        # No sampled state has both successor sets agreeing exactly, which is the state-level view the attributed counts do not carry.
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
        """Which states are drawn is seeded, so a result can be reproduced."""
        self.assertEqual(self.validate(), self.validate())

    def test_an_unseeded_run_records_the_seed_that_reproduces_it(self):
        """Unfixed is not the same as unrecorded.

        `experiments/configs/validation.yaml` leaves `sample_seed` unset, so each run scores on its own sample. What makes such a result checkable is that the seed it drew with is written into the result JSON and, passed back, redraws exactly those states.
        """
        directory = temp_dir(self)
        model_id = build_checkpoint(directory)

        def run(seed):
            validation = ModelValidation(
                model_id, str(directory), problem_path=str(TEST_PROBLEM)
            )
            validation.setup(
                num_samples=NUM_SAMPLES,
                sample_population=SAMPLE_POPULATION,
                seed=seed,
            )
            return validation.sample_seed, validation.validate()

        drawn, result = run(None)
        self.assertIsInstance(drawn, int)
        self.assertEqual(run(drawn), (drawn, result))


if __name__ == "__main__":
    unittest.main()
