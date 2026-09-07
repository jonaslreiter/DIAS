"""End-to-end coverage for the MLP-effects validator.

Its effects come from an MLP over the state, so a schema cannot be planted by hand the way the DIAS validator's can; these pin the contract, the internal consistency of the counts and the worker-process path instead.
"""

import os
import unittest
from pathlib import Path

# On CPU: the assertions are about bookkeeping, and a shared GPU makes this flaky.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from checkpoints import TEST_PROBLEM, build_model, save
from dias.validation.mlp_effects import MlpEffectsValidator
from support import temp_dir

torch.set_num_threads(1)

SAMPLE_SEED = 0
NUM_SAMPLES = 6
SAMPLE_POPULATION = 20


class MlpEffectsTest(unittest.TestCase):
    def validate(self, **kwargs) -> dict:
        directory = temp_dir(self)
        model, hparams = build_model(state_dependent_effects=True)
        model_id = save(directory, model, hparams)
        validation = MlpEffectsValidator(
            model_id,
            str(directory),
            problem_path=str(TEST_PROBLEM),
            **{"batch_size": 256, **kwargs},
        )
        validation.setup(
            num_samples=NUM_SAMPLES,
            sample_population=SAMPLE_POPULATION,
            seed=SAMPLE_SEED,
        )
        return validation.validate()

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
        """The phase-1 fan-out rebuilds every state in its worker, so a mistake there shows up as different counts rather than an error."""
        self.assertEqual(self.validate(num_workers=1), self.validate(num_workers=2))

    def test_a_static_effects_checkpoint_is_refused(self):
        directory = temp_dir(self)
        model, hparams = build_model()
        model_id = save(directory, model, hparams)
        with self.assertRaisesRegex(ValueError, "state_dependent_effects"):
            MlpEffectsValidator(
                model_id, str(directory), problem_path=str(TEST_PROBLEM)
            )


if __name__ == "__main__":
    unittest.main()
