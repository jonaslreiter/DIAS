"""End-to-end coverage for the MLP-effects baseline validator.

It had none, despite being 325 statements and one of the paper's two columns.
Its effects come from an MLP over the current state, so a schema cannot be
planted by hand the way the DIAS validator's can; these assertions pin the
contract, the internal consistency of the counts, and the worker-process path
instead.
"""

import os
import tempfile
import unittest
from pathlib import Path

# Keep the batched forward pass on CPU: the assertions are about bookkeeping,
# not throughput, and a shared GPU makes the test flaky on a busy node.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from checkpoints import TEST_PROBLEM, build_model, save
from dias.validation.mlp_baseline import ModelValidation

torch.set_num_threads(1)

SAMPLE_SEED = 0
NUM_SAMPLES = 6
SAMPLE_POPULATION = 20


class MlpBaselineTest(unittest.TestCase):
    def validate(self, **kwargs) -> dict:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        model, hparams = build_model(state_dependent_effects=True)
        model_id = save(directory, model, hparams)
        validation = ModelValidation(
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
        result = self.validate()
        self.assertEqual(
            set(result),
            {"matches", "total", "accuracy", "precision", "recall", "tp", "fp", "fn"},
        )
        self.assertEqual(result["total"], NUM_SAMPLES)

    def test_counts_agree_with_the_rates_they_are_reported_beside(self):
        result = self.validate()
        tp, fp, fn = result["tp"], result["fp"], result["fn"]
        self.assertEqual(result["precision"], tp / (tp + fp) if tp + fp else 0.0)
        self.assertEqual(result["recall"], tp / (tp + fn) if tp + fn else 0.0)
        # An untrained MLP recovers nothing, so every original successor is missed.
        self.assertGreater(fn, 0)
        self.assertEqual(tp, 0)

    def test_worker_processes_do_not_change_the_result(self):
        """The phase-1 fan-out reloads the domain in each worker and rebuilds
        every state from an atom payload; a mistake there would show up as
        different counts rather than as an error."""
        self.assertEqual(self.validate(num_workers=1), self.validate(num_workers=2))

    def test_a_static_effects_checkpoint_is_refused(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        model, hparams = build_model()
        model_id = save(directory, model, hparams)
        with self.assertRaisesRegex(ValueError, "state_dependent_effects"):
            ModelValidation(model_id, str(directory), problem_path=str(TEST_PROBLEM))


if __name__ == "__main__":
    unittest.main()
