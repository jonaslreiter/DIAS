"""End-to-end coverage for the MLP-effects validator.

Its effects come from an MLP over the state, so a schema cannot be planted by hand the way the DIAS validator's can; these pin the contract, the internal consistency of the counts and the worker-process path instead.
"""

import json
import os
import types
import unittest

# On CPU: the assertions are about bookkeeping, and a shared GPU makes this flaky.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from checkpoints import TEST_PROBLEM, build_model, save
from dias.validation.grounding import (
    flip_signature,
    signature_from_words,
    successor_keys,
)
from dias.validation.mlp_effects import MlpEffectsValidator
from dias.validation.successors import combine_scores, score_successors
from support import temp_dir

torch.set_num_threads(1)

SAMPLE_SEED = 0
NUM_SAMPLES = 6
SAMPLE_POPULATION = 20


class MlpEffectsTest(unittest.TestCase):
    def setUp(self):
        self.directory = temp_dir(self)
        model, hparams = build_model(state_dependent_effects=True)
        self.model_id = save(self.directory, model, hparams)

    def validate(self, progress_path=None, seed=SAMPLE_SEED, **kwargs) -> dict:
        validation = MlpEffectsValidator(
            self.model_id,
            str(self.directory),
            problem_path=str(TEST_PROBLEM),
            **{"batch_size": 256, **kwargs},
        )
        validation.setup(
            num_samples=NUM_SAMPLES,
            sample_population=SAMPLE_POPULATION,
            seed=seed,
        )
        return validation.validate(progress_path)

    def test_result_matches_the_shared_validation_contract(self):
        """The same payload the symbolic validators return, keys and all.

        All three validators finish through `successors.score_successors`, so this is one scorer's output rather than three that happen to agree.
        """
        result = self.validate()
        self.assertEqual(
            set(result),
            {
                "matches",
                "total",
                "accuracy",
                "precision",
                "recall",
                "tp",
                "fp",
                "fn",
                "action_stats",
            },
        )
        self.assertEqual(result["total"], NUM_SAMPLES)

    def test_the_reported_counts_are_the_per_action_counts(self):
        """One measure, not two views that have to be kept in agreement.

        `tp`/`fp`/`fn` are the sums of `action_stats`: a successor reached under the wrong operator is an error, and a successor two operators both reach is two predictions. There is no second, union-level count beside them to be confused with them.
        """
        result = self.validate()
        stats = result["action_stats"].values()

        for key in ("tp", "fp", "fn"):
            self.assertEqual(sum(s[key] for s in stats), result[key], key)
        self.assertGreater(result["fn"], 0, "an untrained MLP misses everything")

    def test_worker_processes_do_not_change_the_result(self):
        """Workers rebuild every state from its atoms, so a mistake there shows up as different counts rather than an error."""
        self.assertEqual(self.validate(num_workers=1), self.validate(num_workers=2))

    def test_batching_does_not_change_the_result(self):
        self.assertEqual(self.validate(batch_size=8), self.validate(batch_size=256))

    def test_a_resumed_validation_scores_only_the_states_left(self):
        progress = self.directory / "seed0.test-1.states.jsonl"
        uninterrupted = self.validate(progress)
        header, *states = progress.read_text().splitlines()

        # Two states scored before a kill tore the third line.
        progress.write_text("\n".join([header, *states[:2], '{"state": 4, "sco']))
        self.assertEqual(self.validate(progress), uninterrupted)
        self.assertEqual(len(progress.read_text().splitlines()), NUM_SAMPLES + 1)

    def test_progress_for_another_sample_is_not_reused(self):
        progress = self.directory / "seed0.test-1.states.jsonl"
        self.validate(progress, seed=SAMPLE_SEED + 1)
        self.assertEqual(self.validate(progress), self.validate())
        self.assertEqual(
            json.loads(progress.read_text().splitlines()[0])["sample_seed"],
            SAMPLE_SEED,
        )

    def test_a_static_effects_checkpoint_is_refused(self):
        directory = temp_dir(self)
        model, hparams = build_model()
        model_id = save(directory, model, hparams)
        with self.assertRaisesRegex(ValueError, "state_dependent_effects"):
            MlpEffectsValidator(
                model_id, str(directory), problem_path=str(TEST_PROBLEM)
            )


class BoundObjectGridTest(unittest.TestCase):
    """Predicting on the grid of bound objects must match `apply_schema` on the whole instance.

    The untrained effect MLP predicts almost nothing, which would make agreement trivial, so its last layer is scaled until successors flip often; bindings repeat objects and leave slots inactive, the two cases the small grid handles differently.
    """

    NUM_OBJECTS = 9
    NUM_GROUNDINGS = 4000

    def setUp(self):
        self.model, hparams = build_model(state_dependent_effects=True)
        self.model.retarget(self.NUM_OBJECTS)
        self.model.eval()
        last = self.model.predictor.effects.mlp[5]
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            last.weight.copy_(torch.randn(last.weight.shape, generator=generator) * 0.8)
            last.bias.copy_(torch.randn(last.bias.shape, generator=generator) * 2.0)

        rng = np.random.default_rng(3)
        self.num_relations = len(hparams["relation_arities"])
        self.state = rng.integers(
            0, 2, (self.num_relations, self.NUM_OBJECTS, self.NUM_OBJECTS)
        ).astype(np.uint8)
        self.actions = rng.integers(0, hparams["num_actions"], self.NUM_GROUNDINGS)
        self.slot_objects = rng.integers(
            0, self.NUM_OBJECTS, (self.NUM_GROUNDINGS, hparams["num_slots"])
        ).astype(np.int16)
        self.slot_objects[rng.random(self.slot_objects.shape) < 0.15] = -1
        self.keys = successor_keys(self.num_relations, self.NUM_OBJECTS)

    def full_grid(self, channels) -> list[int]:
        state = torch.from_numpy(self.state).float()
        selection = torch.zeros(
            self.NUM_GROUNDINGS, self.slot_objects.shape[1], self.NUM_OBJECTS
        )
        for g, row in enumerate(self.slot_objects):
            for slot, obj in enumerate(row):
                if obj >= 0:
                    selection[g, slot, obj] = 1.0
        with torch.no_grad():
            successor = self.model.apply_schema(
                state.expand(self.NUM_GROUNDINGS, -1, -1, -1),
                selection,
                torch.from_numpy(self.actions),
            )
        flips = ((successor > 0.5) != (state > 0.5)).numpy() & channels[:, None, None]
        return [flip_signature(self.keys, f) for f in flips]

    def bound_objects(self, channels) -> list[int]:
        with torch.no_grad():
            words = MlpEffectsValidator._successor_words(
                types.SimpleNamespace(model=self.model),
                torch.from_numpy(self.state).float(),
                torch.from_numpy(self.actions),
                self.slot_objects,
                torch.from_numpy(self.keys),
                torch.from_numpy(channels),
            )
        return [signature_from_words(high, low) for high, low in words.tolist()]

    def test_the_bound_objects_predict_what_the_whole_instance_does(self):
        excluded = np.ones(self.num_relations, dtype=bool)
        excluded[0] = False
        for channels in (np.ones(self.num_relations, dtype=bool), excluded):
            self.assertEqual(self.bound_objects(channels), self.full_grid(channels))

    def test_the_comparison_is_not_trivial(self):
        signatures = self.full_grid(np.ones(self.num_relations, dtype=bool))
        self.assertGreater(sum(s != 0 for s in signatures), self.NUM_GROUNDINGS // 2)
        repeated = sum(
            len(set(row[row >= 0])) < (row >= 0).sum() for row in self.slot_objects
        )
        self.assertGreater(repeated, self.NUM_GROUNDINGS // 10)


class SignatureTest(unittest.TestCase):
    def setUp(self):
        self.keys = successor_keys(11, 89)
        self.flips = np.random.default_rng(7).random((11, 89, 89)) < 0.01

    def test_every_process_draws_the_same_keys(self):
        self.assertTrue(np.array_equal(successor_keys(11, 89), self.keys))

    def test_an_unchanged_state_has_the_empty_signature(self):
        self.assertEqual(flip_signature(self.keys, np.zeros_like(self.flips)), 0)

    def test_one_more_flip_changes_the_signature(self):
        extra = self.flips.copy()
        extra[2, 40, 41] = not extra[2, 40, 41]
        self.assertNotEqual(
            flip_signature(self.keys, self.flips), flip_signature(self.keys, extra)
        )


class CombineScoresTest(unittest.TestCase):
    def test_combining_per_state_scores_is_scoring_the_states_together(self):
        rng = np.random.default_rng(0)

        def sets():
            names = rng.choice(["drive", "load", "walk"], rng.integers(1, 4), False)
            return {
                n: set(rng.integers(0, 30, rng.integers(0, 12)).tolist()) for n in names
            }

        states = [(sets(), sets()) for _ in range(40)] + [
            ({"load": {1}}, {"load": {1}})
        ]
        self.assertEqual(
            combine_scores(score_successors([s]) for s in states),
            score_successors(states),
        )


if __name__ == "__main__":
    unittest.main()
