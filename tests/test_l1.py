"""End-to-end coverage for the L1 validator, which compares two PDDL domains directly.

A domain validated against itself must score perfectly, pinning the whole path: untyping, merging, reflexive-atom injection, grounding and the successor comparison. Removing one precondition from the learned copy must lower the score, and spelling equality or the types another way must not change it.
"""

import random
import re
import sys
import unittest
from pathlib import Path

from dias.pddl.untype import problem_object_names
from dias.validation.l1 import L1Validator
from dias.validation.successors import WalkSampling
from support import temp_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pipeline import (  # noqa: E402
    available_domains,
    domain_pddl,
    l1_domain,
    validation_problems,
)

DOMAIN = PROJECT_ROOT / "pddl" / "blocks-3" / "domain.pddl"
PROBLEM = PROJECT_ROOT / "pddl" / "blocks-3" / "test-1.pddl"

SAMPLE_SEED = 0
NUM_SAMPLES = 6
SAMPLING = WalkSampling(num_samples=NUM_SAMPLES, walk_length=20, min_walks=2)


def equality_as_predicate(domain_text: str, name: str) -> str:
    """The domain as a planner without :equality wants it, and as L1 returns it: `=` written as a static predicate called `name`."""
    text = re.sub(r"\(\s*=\s+", f"({name} ", domain_text).replace(" :equality", "")
    return text.replace("(:predicates", f"(:predicates ({name} ?x ?y)", 1)


def with_reflexive_facts(problem_text: str, name: str) -> str:
    facts = " ".join(f"({name} {o} {o})" for o in problem_object_names(problem_text))
    return problem_text.replace("(:init", f"(:init {facts}", 1)


class L1ValidationTest(unittest.TestCase):
    def validate(
        self,
        learned_domain_text: str,
        original_domain_text: str | None = None,
        problem_text: str | None = None,
        injective: bool = False,
    ) -> dict:
        directory = temp_dir(self)
        learned = directory / "learned.pddl"
        learned.write_text(learned_domain_text)
        original = directory / "original.pddl"
        original.write_text(original_domain_text or DOMAIN.read_text())
        problem = directory / "problem.pddl"
        problem.write_text(problem_text or PROBLEM.read_text())
        random.seed(SAMPLE_SEED)
        validation = L1Validator(
            str(original),
            str(learned),
            str(directory),
            problem_path=str(problem),
            injective=injective,
        )
        validation.setup(SAMPLING)
        return validation.validate()

    def assertPerfect(self, result: dict) -> None:
        self.assertEqual(result["total"], NUM_SAMPLES)
        self.assertEqual((result["precision"], result["recall"]), (1.0, 1.0))

    def test_a_domain_validated_against_itself_scores_perfectly(self):
        result = self.validate(DOMAIN.read_text())
        self.assertEqual(result["total"], NUM_SAMPLES)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)

    def test_dropping_a_precondition_costs_precision(self):
        # `stack` requires the moved block to be clear; without that it applies in states it should not, producing successors the original cannot.
        weakened = re.sub(r"\(clear \?bm\)\s*", "", DOMAIN.read_text(), count=1)
        self.assertNotEqual(weakened, DOMAIN.read_text())
        result = self.validate(weakened)
        self.assertLess(result["precision"], 1.0)
        self.assertGreater(result["fp"], 0)

    def test_learned_equality_may_be_a_predicate(self):
        # No problem in pddl/ asserts `eq` or `equal`, so read as written, every schema using one would apply nowhere.
        for name in ("eq", "equal"):
            with self.subTest(name=name):
                self.assertPerfect(
                    self.validate(equality_as_predicate(DOMAIN.read_text(), name))
                )

    def test_so_may_the_original_with_or_without_its_facts(self):
        original = equality_as_predicate(DOMAIN.read_text(), "equal")
        for problem in (
            PROBLEM.read_text(),
            with_reflexive_facts(PROBLEM.read_text(), "equal"),
        ):
            with self.subTest(facts="(equal" in problem):
                self.assertPerfect(self.validate(DOMAIN.read_text(), original, problem))

    def test_uninformative_types_constrain_nothing(self):
        # How L1 types a parameter of an untyped domain.
        typed = DOMAIN.read_text().replace(
            ":parameters (?bm ?bt)", ":parameters (?bm - untyped ?bt - object)"
        )
        self.assertNotEqual(typed, DOMAIN.read_text())
        self.assertPerfect(self.validate(typed))

    def test_a_type_the_original_cannot_supply_is_refused(self):
        typed = DOMAIN.read_text().replace(
            ":parameters (?bm ?bt)", ":parameters (?bm ?bt - block)"
        )
        with self.assertRaisesRegex(ValueError, "block"):
            self.validate(typed)

    def test_injective_requires_the_learned_parameters_to_be_distinct(self):
        # Blocks without its distinctness, as L1 learns it: `stack` then also puts a block on itself.
        learned = re.sub(r"\s*\(not \(= \?\w+ \?\w+\)\)", "", DOMAIN.read_text())
        self.assertNotIn("(=", learned)
        self.assertGreater(self.validate(learned)["fp"], 0)
        self.assertPerfect(self.validate(learned, injective=True))

    def test_setup_is_required_before_validate(self):
        directory = temp_dir(self)
        learned = directory / "learned.pddl"
        learned.write_text(DOMAIN.read_text())
        validation = L1Validator(
            str(DOMAIN), str(learned), str(directory), problem_path=str(PROBLEM)
        )
        with self.assertRaisesRegex(RuntimeError, "setup"):
            validation.validate()


class CommittedL1DomainTest(unittest.TestCase):
    def test_every_l1_domain_merges_with_its_own_and_misses_no_successor(self):
        # L1's recall is 1.0 in every domain (the paper's Table 17), while a schema the merge breaks, by an undeclared predicate or a type no problem asserts, applies nowhere and misses every successor of its action.
        domains = [
            domain for domain in available_domains() if l1_domain(domain).is_file()
        ]
        self.assertTrue(domains)
        for domain in domains:
            with self.subTest(domain=domain):
                validation = L1Validator(
                    str(domain_pddl(domain)),
                    str(l1_domain(domain)),
                    str(temp_dir(self)),
                    problem_path=str(validation_problems(domain)[0]),
                )
                # Walks as long as validation's: shorter ones never make satellite's take_image applicable, which would leave it unchecked.
                validation.setup(
                    WalkSampling(num_samples=NUM_SAMPLES, min_walks=1),
                    seed=SAMPLE_SEED,
                )
                result = validation.validate()
                self.assertGreater(result["tp"], 0)
                self.assertEqual(result["fn"], 0)


if __name__ == "__main__":
    unittest.main()
