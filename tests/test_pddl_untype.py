"""Unit tests for compiling typed PDDL down to untyped PDDL.

Both functions used to take file paths and live in `validation.utils`, which
made them reachable only with a filesystem. They now take source text.
"""

import unittest

from dias.pddl.untype import extract_domain_header, strip_problem_types

DOMAIN = """(define (domain demo)
  (:requirements :strips :typing :equality)
  (:types child - parent parent - object)
  (:predicates
    (ready ?x - child)
    (linked ?x - child ?y - child)
  )

  (:action go
    :parameters (?x - child)
    :precondition (ready ?x)
    :effect (ready ?x))
)
"""

PROBLEM = """(define (problem p) (:domain demo)
(:objects c-1 - child)
(:init)
(:goal (ready c-1)))
"""


class StripProblemTypesTest(unittest.TestCase):
    def test_injects_the_type_hierarchy_and_reflexive_atoms(self):
        text = strip_problem_types(
            PROBLEM,
            DOMAIN,
            equality_predicate_name="=",
            reflexive_predicate_names=["same"],
        )
        self.assertIn("(:objects c-1)", text)
        # The whole ancestor chain, not just the declared type.
        self.assertIn("(child c-1)", text)
        self.assertIn("(parent c-1)", text)
        self.assertIn("(= c-1 c-1)", text)
        self.assertIn("(same c-1 c-1)", text)

    def test_object_names_containing_a_hyphen_survive(self):
        self.assertNotIn("- child", strip_problem_types(PROBLEM, DOMAIN))
        self.assertIn("c-1", strip_problem_types(PROBLEM, DOMAIN))

    def test_without_a_domain_types_are_dropped_but_nothing_is_injected(self):
        text = strip_problem_types(
            "(define (problem p) (:domain demo)\n"
            "(:objects a b - thing)\n(:init)\n(:goal (ready a)))\n"
        )
        self.assertIn("(:objects a b)", text)
        self.assertNotIn("- thing", text)
        self.assertNotIn("(thing a)", text)


class ExtractDomainHeaderTest(unittest.TestCase):
    def test_untypes_and_forces_negative_preconditions(self):
        header = extract_domain_header(DOMAIN)
        self.assertIn("(:requirements :negative-preconditions :equality)", header)
        self.assertNotIn(":typing", header)
        self.assertNotIn("(:types", header)
        self.assertNotIn("- child", header)
        self.assertNotIn("(:action", header)

    def test_equality_can_be_dropped_from_the_requirements(self):
        header = extract_domain_header(DOMAIN, keep_equality_requirement=False)
        self.assertIn("(:requirements :negative-preconditions)", header)

    def test_rebuilt_predicates_cover_relations_and_types(self):
        header = extract_domain_header(
            DOMAIN,
            rel_names=["=", "ready", "linked"],
            predicate_arities=[2, 1, 2],
            keep_equality_requirement=False,
        )
        self.assertIn("(ready ?x)", header)
        self.assertIn("(linked ?x0 ?x1)", header)
        # Types become unary predicates so they can act as static preconditions.
        self.assertIn("(child ?x)", header)
        self.assertIn("(parent ?x)", header)
        # "=" is a built-in enabled by :equality, never a declared predicate.
        self.assertNotIn("(= ", header)
        self.assertEqual(header.count("(ready"), 1)


if __name__ == "__main__":
    unittest.main()
