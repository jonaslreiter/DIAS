"""Unit tests for compiling typed PDDL down to untyped PDDL.

The functions take and return PDDL source text, so a rewrite can be checked without constructing a validator.
"""

import unittest

from dias.pddl.untype import (
    extract_domain_header,
    problem_object_names,
    strip_problem_types,
    untype_action,
    with_equality_facts,
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
    def test_injects_the_type_hierarchy(self):
        text = strip_problem_types(PROBLEM, DOMAIN)
        self.assertIn("(:objects c-1)", text)
        # The whole ancestor chain, not just the declared type.
        self.assertIn("(child c-1)", text)
        self.assertIn("(parent c-1)", text)

    def test_object_names_containing_a_hyphen_survive(self):
        self.assertNotIn("- child", strip_problem_types(PROBLEM, DOMAIN))
        self.assertIn("c-1", strip_problem_types(PROBLEM, DOMAIN))

    def test_without_a_domain_types_are_dropped_but_nothing_is_injected(self):
        text = strip_problem_types(
            "(define (problem p) (:domain demo)\n(:objects a b - thing)\n(:init)\n(:goal (ready a)))\n"
        )
        self.assertIn("(:objects a b)", text)
        self.assertNotIn("- thing", text)
        self.assertNotIn("(thing a)", text)

    def test_a_type_declared_without_a_parent_is_still_asserted(self):
        # Satellite declares all four of its types this way.
        text = strip_problem_types(
            "(define (problem p) (:domain d)\n(:objects s1 - satellite d1 d2 - truck-bay)\n(:init)\n(:goal (ready s1)))\n",
            "(define (domain d) (:requirements :typing) (:types satellite truck-bay))",
        )
        self.assertIn("(:objects s1 d1 d2)", text)
        self.assertIn("(satellite s1)", text)
        self.assertIn("(truck-bay d2)", text)
        self.assertNotIn("(object", text)


class ExtractDomainHeaderTest(unittest.TestCase):
    def test_untypes_and_forces_negative_preconditions(self):
        header = extract_domain_header(DOMAIN)
        self.assertIn("(:requirements :negative-preconditions :equality)", header)
        self.assertNotIn(":typing", header)
        self.assertNotIn("(:types", header)
        self.assertNotIn("- child", header)
        self.assertNotIn("(:action", header)

    def test_rebuilt_predicates_cover_relations_and_types(self):
        header = extract_domain_header(
            DOMAIN,
            predicate_names=["=", "ready", "linked"],
            predicate_arities=[2, 1, 2],
        )
        self.assertIn("(ready ?x)", header)
        self.assertIn("(linked ?x0 ?x1)", header)
        # Types become unary predicates so they can act as static preconditions.
        self.assertIn("(child ?x)", header)
        self.assertIn("(parent ?x)", header)
        # "=" is a built-in enabled by :equality, never a declared predicate.
        self.assertNotIn("(= ", header)
        self.assertEqual(header.count("(ready"), 1)

    def test_hyphenated_names_survive_untyping(self):
        domain = "(define (domain d)\n  (:requirements :typing)\n  (:types loc)\n  (:predicates (on-table ?x) (link ?loc-from ?loc-to - loc))\n  (:action a :parameters () :precondition (and) :effect (and))\n)"
        header = extract_domain_header(domain)
        self.assertIn("(link ?loc-from ?loc-to)", header)
        header = extract_domain_header(
            domain, predicate_names=["link"], predicate_arities=[2]
        )
        # Declared by the original though no relation names it.
        self.assertIn("(on-table ?x)", header)


class UntypeActionTest(unittest.TestCase):
    ACTION = "(:action push\n  :parameters (?p - player ?c - crate)\n  :precondition (and (at ?p) (= ?p ?c))\n  :effect (at ?c))"

    def test_types_become_preconditions(self):
        text = untype_action(self.ACTION)
        self.assertIn(":parameters (?p ?c)", text)
        self.assertIn("(player ?p)", text)
        self.assertIn("(crate ?c)", text)
        self.assertIn("(= ?p ?c)", text)

    def test_prefix_and_ignored_types(self):
        text = untype_action(
            "(:action go :parameters (?x - untyped ?y - cell) :precondition (and) :effect (at ?y))",
            name_prefix="learned_",
            ignored_types=frozenset({"untyped"}),
        )
        self.assertIn("(:action learned_go", text)
        self.assertNotIn("untyped", text)
        self.assertIn("(cell ?y)", text)

    def test_a_malformed_block_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing parameters"):
            untype_action("(:action broken :parameters (?x))")


class EqualityFactsTest(unittest.TestCase):
    def test_every_object_equals_itself_in_place_of_the_facts_it_had(self):
        problem = PROBLEM.replace("(:init)", "(:init (eq c-1 c-1))").replace(
            "c-1 - child", "c-1 c-2 - child"
        )
        text = with_equality_facts(problem)
        self.assertIn("(equal c-1 c-1)", text)
        self.assertIn("(equal c-2 c-2)", text)
        self.assertNotIn("(eq ", text)

    def test_a_fact_between_two_objects_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not equality"):
            with_equality_facts(PROBLEM.replace("(:init)", "(:init (Equal c-1 c-2))"))


class ProblemObjectsTest(unittest.TestCase):
    def test_objects_are_read_without_their_types(self):
        self.assertEqual(problem_object_names(PROBLEM), ["c-1"])


if __name__ == "__main__":
    unittest.main()
