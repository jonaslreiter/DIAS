"""End-to-end characterization of the DIAS validation path on a planted schema: blocks-3's `stack` written into a checkpoint exactly, its distinctness left to the derivation, the other two actions inert."""

import json
import unittest
from pathlib import Path

import torch

from checkpoints import TEST_PROBLEM, build_model, domain_schema, save
from dias.schema.export import learned_domain
from dias.validation.checkpoint import load_run, resolve_checkpoint
from dias.validation.dias import DiasValidator
from dias.validation.successors import WalkSampling
from support import temp_dir

torch.set_num_threads(1)

OFF, ON = 10.0, 10.0

SAMPLE_SEED = 0
NUM_SAMPLES = 8
SAMPLING = WalkSampling(num_samples=NUM_SAMPLES, walk_length=20, min_walks=2)


def build_checkpoint(
    directory: Path, equality: bool = False, inequality: bool = False
) -> str:
    """Write a checkpoint whose only learned action is `stack`, correct apart from its inequality, which only a run with an `=` channel can learn."""
    schema = domain_schema(equality=equality)
    relation = {name: i for i, name in enumerate(schema.relation_names)}
    action = {name: i for i, name in enumerate(schema.action_names)}
    model, hparams = build_model(equality=equality)

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

    return save(directory, model, hparams, equality=equality)


class ValidationCharacterizationTest(unittest.TestCase):
    def validate(self, sampling=SAMPLING) -> dict:
        directory = temp_dir(self)
        model_id = build_checkpoint(directory)
        validation = DiasValidator(
            model_id, str(directory), problem_path=str(TEST_PROBLEM)
        )
        validation.setup(sampling, seed=SAMPLE_SEED)
        return validation.validate()

    def validate_with_progress(self, directory, model_id, progress) -> dict:
        validation = DiasValidator(
            model_id, str(directory), problem_path=str(TEST_PROBLEM)
        )
        validation.setup(SAMPLING, seed=SAMPLE_SEED)
        return validation.validate(progress)

    def test_a_resumed_validation_scores_only_the_states_left(self):
        """A validation stopped partway continues from its recorded states and adds up to the uninterrupted result."""
        directory = temp_dir(self)
        model_id = build_checkpoint(directory)
        progress = directory / "seed0.test-1.states.jsonl"
        uninterrupted = self.validate_with_progress(directory, model_id, progress)
        self.assertEqual(uninterrupted, self.validate())

        # Three states scored before a kill tore the fourth line.
        header, *states = progress.read_text().splitlines()
        progress.write_text("\n".join([header, *states[:3], '{"state": 5, "sco']))
        self.assertEqual(
            self.validate_with_progress(directory, model_id, progress), uninterrupted
        )
        self.assertEqual(len(progress.read_text().splitlines()), NUM_SAMPLES + 1)

    def test_progress_from_another_comparison_is_not_reused(self):
        directory = temp_dir(self)
        model_id = build_checkpoint(directory)
        progress = directory / "seed0.test-1.states.jsonl"
        self.validate_with_progress(directory, model_id, progress)
        header, *states = progress.read_text().splitlines()
        stale = json.loads(header) | {"scored": "another domain"}
        bogus = {"total": 1, "matches": 1, "action_stats": {}}
        progress.write_text(
            "\n".join(
                [json.dumps(stale)]
                + [json.dumps({"state": i, "score": bogus}) for i in range(NUM_SAMPLES)]
            )
        )
        self.assertEqual(
            self.validate_with_progress(directory, model_id, progress), self.validate()
        )
        self.assertEqual(progress.read_text().splitlines()[0], header)

    def test_planted_schema_reproduces_pinned_metrics(self):
        result = self.validate()

        self.assertEqual(result["total"], NUM_SAMPLES)
        # 65 successors reached, 112 missed, and the 8 states themselves produced by each of the two inert actions.
        self.assertAlmostEqual(result["precision"], 65 / 81, places=12)
        self.assertAlmostEqual(result["recall"], 65 / 177, places=12)
        self.assertEqual(result["matches"], 0)
        self.assertEqual(result["accuracy"], 0.0)

    def test_only_the_planted_action_is_recovered(self):
        stats = self.validate()["action_stats"]

        self.assertEqual(stats["stack"], {"tp": 65, "fp": 0, "fn": 0, "conflicts": 0})
        self.assertEqual(
            stats["newtower"], {"tp": 0, "fp": 8, "fn": 23, "conflicts": 31}
        )
        self.assertEqual(stats["move"], {"tp": 0, "fp": 8, "fn": 89, "conflicts": 97})

    def test_repeated_validation_is_deterministic(self):
        self.assertEqual(self.validate(), self.validate())

    def test_an_unseeded_run_records_the_seed_that_reproduces_it(self):
        """The seed an unseeded validation drew reproduces its states."""
        directory = temp_dir(self)
        model_id = build_checkpoint(directory)

        def run(seed):
            validation = DiasValidator(
                model_id, str(directory), problem_path=str(TEST_PROBLEM)
            )
            validation.setup(SAMPLING, seed=seed)
            return validation.sample_seed, validation.validate()

        drawn, result = run(None)
        self.assertIsInstance(drawn, int)
        self.assertEqual(run(drawn), (drawn, result))


class LearnedDomainTest(unittest.TestCase):
    def stack(self, equality: bool, inequality: bool = False) -> str:
        directory = temp_dir(self)
        run = load_run(
            build_checkpoint(directory, equality, inequality), str(directory)
        )
        text = learned_domain(run.model, run.schema, run.datamodule_hparams, 0.5)
        return text[text.index("(:action stack") : text.index("(:action newtower")]

    def test_a_run_without_equality_requires_distinct_parameters(self):
        self.assertIn("(not (= ?x0 ?x1))", self.stack(equality=False))

    def test_a_run_that_learned_equality_writes_only_what_it_learned(self):
        self.assertNotIn("equal", self.stack(equality=True))
        learned = self.stack(equality=True, inequality=True)
        self.assertIn("(not (equal ?x0 ?x1))", learned)
        self.assertNotIn("(= ", learned)

    def score_stack(self, equality: bool, force_injective: bool = False) -> dict:
        directory = temp_dir(self)
        model_id = build_checkpoint(directory, equality)
        validation = DiasValidator(
            model_id,
            str(directory),
            str(TEST_PROBLEM),
            force_injective=force_injective,
        )
        validation.setup(
            SAMPLING,
            seed=SAMPLE_SEED,
        )
        return validation.validate()["action_stats"]["stack"]

    def test_forcing_distinct_parameters_excludes_a_repeated_object(self):
        perfect = {"tp": 65, "fp": 0, "fn": 0, "conflicts": 0}
        self.assertEqual(self.score_stack(equality=False), perfect)
        self.assertGreater(self.score_stack(equality=True)["fp"], 0)
        self.assertEqual(self.score_stack(equality=True, force_injective=True), perfect)

    def test_the_domain_written_beside_the_checkpoint_is_scored_unless_a_threshold_is_given(
        self,
    ):
        directory = temp_dir(self)
        model_id = build_checkpoint(directory)
        run = load_run(model_id, str(directory))
        text = learned_domain(run.model, run.schema, run.datamodule_hparams, 0.5)
        written = resolve_checkpoint(
            directory / "dias" / model_id / "checkpoints"
        ).with_suffix(".pddl")
        written.write_text(text.replace("(clear ?x0) ", "", 1))

        def scored(threshold):
            return DiasValidator(
                model_id, str(directory), str(TEST_PROBLEM), threshold=threshold
            ).learned_domain_text

        self.assertEqual(scored(None), written.read_text())
        self.assertEqual(scored(0.5), text)


if __name__ == "__main__":
    unittest.main()
