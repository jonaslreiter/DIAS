"""End-to-end characterization of the DIAS validation path, which the unit tests do not cover: a transposed einsum, a changed enumeration order or a flipped logit channel would pass them all.

Rather than train a model, this plants a schema by hand: blocks-3's `stack` written into the checkpoint exactly, its inequality left to decoding as for any run without the `=` relation, and the other two actions left inert. The expected result then follows independently of the numbers, so the assertion is sensitive in both directions: `stack` recovered perfectly, everything the others would reach a false negative.
"""

import unittest
from pathlib import Path

import torch

from checkpoints import TEST_PROBLEM, build_model, domain_schema, save
from dias.validation.checkpoint import load_run
from dias.validation.dias import DiasValidator, read_learned_schemas
from support import temp_dir

torch.set_num_threads(1)

# Softmax over the three channels turns these into ~(1, 0, 0) or ~(0, 1, 0), which the 0.5 threshold reads as a clean off/on.
OFF, ON = 10.0, 10.0

SAMPLE_SEED = 0
NUM_SAMPLES = 8
SAMPLE_POPULATION = 20


def build_checkpoint(directory: Path, inequality: bool = False) -> str:
    """Write a checkpoint whose only learned action is `stack`, correct apart from its inequality, which needs a run from before `include_equality` to have a channel at all."""
    schema = domain_schema(legacy=inequality)
    relation = {name: i for i, name in enumerate(schema.relation_names)}
    action = {name: i for i, name in enumerate(schema.action_names)}
    model, hparams = build_model(legacy=inequality)

    stack = action["stack"]
    with torch.no_grad():
        effects = model.predictor.effects.logits
        preconditions = model.predictor.preconditions.logits
        for tensor in (effects, preconditions):
            tensor[..., 0] = OFF
            tensor[..., 1] = 0.0
            tensor[..., 2] = 0.0

        slot_weights = model.predictor.selector.slot_weight_logits
        slot_weights.fill_(-ON)
        slot_weights[stack, 0] = ON
        slot_weights[stack, 1] = ON

        def enable(tensor, name, row, column, channel):
            tensor[stack, relation[name], row, column, 0] = 0.0
            tensor[stack, relation[name], row, column, channel] = ON

        # stack(?bm=slot0, ?bt=slot1):
        # pre  (clear ?bm) (clear ?bt) (on-table ?bm) (not (= ?bm ?bt))
        # eff  (on ?bm ?bt) (not (clear ?bt)) (not (on-table ?bm))
        enable(preconditions, "clear", 0, 0, 1)
        enable(preconditions, "clear", 1, 1, 1)
        enable(preconditions, "on-table", 0, 0, 1)
        if inequality:
            enable(preconditions, "=", 0, 1, 2)
        enable(effects, "on", 0, 1, 1)
        enable(effects, "clear", 1, 1, 2)
        enable(effects, "on-table", 0, 0, 2)

    return save(directory, model, hparams, legacy=inequality)


class ValidationCharacterizationTest(unittest.TestCase):
    def validate(self) -> dict:
        directory = temp_dir(self)
        model_id = build_checkpoint(directory)
        validation = DiasValidator(
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
            validation = DiasValidator(
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


class InjectiveDecodingTest(unittest.TestCase):
    def schemas(self, inequality: bool, injective: bool) -> dict:
        directory = temp_dir(self)
        model_id = build_checkpoint(directory, inequality=inequality)
        run = load_run(model_id, str(directory))
        transpiler = read_learned_schemas(run, True, 0.5, injective=injective)
        return {schema.name: schema for schema in transpiler.action_schemas}

    def test_injective_decoding_writes_the_inequality_the_model_assumes(self):
        stack = self.schemas(inequality=False, injective=True)["stack"]
        self.assertIn("(not (eq ?x0 ?x1))", [str(atom) for atom in stack.preconditions])
        plain = self.schemas(inequality=False, injective=False)["stack"]
        self.assertNotIn("eq", " ".join(str(atom) for atom in plain.preconditions))

    def test_an_inequality_already_learned_is_not_repeated(self):
        stack = self.schemas(inequality=True, injective=True)["stack"]
        written = [
            str(atom) for atom in stack.preconditions if atom.predicate_name == "eq"
        ]
        self.assertEqual(written, ["(not (eq ?x0 ?x1))"])

    def score_stack(self, injective: bool) -> dict:
        directory = temp_dir(self)
        model_id = build_checkpoint(directory, inequality=False)
        validation = DiasValidator(
            model_id,
            str(directory),
            problem_path=str(TEST_PROBLEM),
            injective=injective,
        )
        validation.setup(
            num_samples=NUM_SAMPLES,
            sample_population=SAMPLE_POPULATION,
            seed=SAMPLE_SEED,
        )
        return validation.validate()["action_stats"]["stack"]

    def test_a_repeated_object_is_excluded_on_the_test_problem(self):
        """Without the inequality, stack(b, b) applies wherever b is clear on the table and puts it on itself; the distinctness written at decoding is what keeps it out, which takes the reflexive facts in the merged problem."""
        self.assertGreater(self.score_stack(injective=False)["fp"], 0)
        self.assertEqual(
            self.score_stack(injective=True),
            {"tp": 54, "fp": 0, "fn": 0, "conflicts": 0},
        )


if __name__ == "__main__":
    unittest.main()
