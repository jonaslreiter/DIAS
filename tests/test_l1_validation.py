"""End-to-end coverage for the L1 validator, which had none.

L1 compares two PDDL domains directly, with no model involved, so a domain
validated against itself must score perfectly -- which pins the whole path:
untyping, merging, the reflexive-atom injection, pymimir grounding and the
successor comparison. Removing one precondition from the learned copy must then
lower the score.
"""

import random
import re
import tempfile
import unittest
from pathlib import Path

from dias.validation.l1 import ModelValidation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAIN = PROJECT_ROOT / "pddl" / "blocks-3" / "domain.pddl"
PROBLEM = PROJECT_ROOT / "pddl" / "blocks-3" / "test-1.pddl"

SAMPLE_SEED = 0
NUM_SAMPLES = 6
SAMPLE_POPULATION = 20


class L1ValidationTest(unittest.TestCase):
    def validate(self, learned_domain_text: str) -> dict:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        learned = directory / "learned.pddl"
        learned.write_text(learned_domain_text)
        random.seed(SAMPLE_SEED)
        validation = ModelValidation(
            str(DOMAIN), str(learned), str(directory), problem_path=str(PROBLEM)
        )
        validation.setup(num_samples=NUM_SAMPLES, sample_population=SAMPLE_POPULATION)
        return validation.validate()

    def test_a_domain_validated_against_itself_scores_perfectly(self):
        result = self.validate(DOMAIN.read_text())
        self.assertEqual(result["total"], NUM_SAMPLES)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)

    def test_dropping_a_precondition_costs_precision(self):
        # `stack` requires the moved block to be clear; without that it applies
        # in states it should not, producing successors the original cannot.
        weakened = re.sub(
            r"\(clear \?bm\)\s*", "", DOMAIN.read_text(), count=1
        )
        self.assertNotEqual(weakened, DOMAIN.read_text())
        result = self.validate(weakened)
        self.assertLess(result["precision"], 1.0)
        self.assertGreater(result["fp"], 0)

    def test_setup_is_required_before_validate(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        learned = directory / "learned.pddl"
        learned.write_text(DOMAIN.read_text())
        validation = ModelValidation(
            str(DOMAIN), str(learned), str(directory), problem_path=str(PROBLEM)
        )
        with self.assertRaisesRegex(RuntimeError, "setup"):
            validation.validate()


if __name__ == "__main__":
    unittest.main()
