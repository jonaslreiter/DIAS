"""Unit tests for compiling typed PDDL down to untyped PDDL.

These used to be private static methods duplicated across the validator
classes, taking file paths, which
made them reachable only by constructing a validator. They now take source text.
"""

import unittest

from dias.pddl.untype import (
    extract_domain_header,
    inject_reflexive_atoms,
    problem_object_names,
    strip_problem_types,
    untype_action,
)

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


class UntypeActionTest(unittest.TestCase):
    ACTION = (
        "(:action push\n"
        "  :parameters (?p - player ?c - crate)\n"
        "  :precondition (and (at ?p) (= ?p ?c))\n"
        "  :effect (at ?c))"
    )

    def test_types_become_preconditions_and_equality_is_rewritten(self):
        text = untype_action(self.ACTION, equality_predicate_name="eq")
        self.assertIn(":parameters (?p ?c)", text)
        self.assertIn("(player ?p)", text)
        self.assertIn("(crate ?c)", text)
        self.assertIn("(eq ?p ?c)", text)
        self.assertNotIn("(= ", text)

    def test_equality_is_left_alone_without_a_predicate_name(self):
        self.assertIn("(= ?p ?c)", untype_action(self.ACTION))

    def test_prefix_and_ignored_types(self):
        text = untype_action(
            "(:action go :parameters (?x - untyped ?y - cell)"
            " :precondition (and) :effect (at ?y))",
            name_prefix="learned_",
            ignored_types=frozenset({"untyped"}),
        )
        self.assertIn("(:action learned_go", text)
        self.assertNotIn("untyped", text)
        self.assertIn("(cell ?y)", text)

    def test_a_malformed_block_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing parameters"):
            untype_action("(:action broken :parameters (?x))")


class ReflexiveAtomsTest(unittest.TestCase):
    def test_objects_are_read_without_their_types(self):
        self.assertEqual(problem_object_names(PROBLEM), ["c-1"])

    def test_atoms_are_injected_once_per_predicate_and_object(self):
        text = inject_reflexive_atoms(PROBLEM, ["eq", "equal"])
        self.assertIn("(eq c-1 c-1)", text)
        self.assertIn("(equal c-1 c-1)", text)

    def test_no_predicates_leaves_the_problem_untouched(self):
        self.assertEqual(inject_reflexive_atoms(PROBLEM, []), PROBLEM)


if __name__ == "__main__":
    unittest.main()
