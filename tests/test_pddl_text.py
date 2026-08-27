"""Unit tests for PDDL source manipulation.

They are pure string functions, so none of this needs a checkpoint, a datamodule or a pymimir problem.
"""

import unittest

from dias.pddl import text

DOMAIN = """(define (domain sokoban)
  (:requirements :strips :typing :equality)
  (:types crate stone - locatable
          player - object)
  (:predicates
    (at ?x - locatable ?y - location)
    (clear ?c)
  )

  (:action push
    :parameters (?p - player ?c - crate ?from ?to - location)
    :precondition (and (at ?p ?from) (not (= ?p ?c)))
    :effect (and (at ?p ?to) (not (at ?p ?from)))
  )

  (:action move
    :parameters (?p ?q)
    :precondition (clear ?q)
    :effect (at ?p ?q)
  )
)
"""


class FindBalancedExprTest(unittest.TestCase):
    def test_returns_expression_and_resume_index(self):
        expr, end = text.find_balanced_expr("xx (a (b c) d) yy", 0)
        self.assertEqual(expr, "(a (b c) d)")
        self.assertEqual(end, 14)

    def test_rejects_missing_and_unbalanced_parens(self):
        with self.assertRaisesRegex(ValueError, "no '\\('"):
            text.find_balanced_expr("nothing here", 0)
        with self.assertRaisesRegex(ValueError, "Unbalanced"):
            text.find_balanced_expr("(a (b c)", 0)


class ExtractBlocksTest(unittest.TestCase):
    def test_extracts_each_action_block(self):
        blocks = text.extract_action_blocks(DOMAIN)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(blocks[0].startswith("(:action push"))
        self.assertTrue(blocks[1].startswith("(:action move"))
        # Nested parens inside a block must not terminate it early.
        self.assertIn(":effect", blocks[0])
        self.assertTrue(blocks[0].endswith(")"))

    def test_returns_empty_when_opener_absent(self):
        self.assertEqual(text.extract_blocks(DOMAIN, "(:derived"), [])


class TypedParametersTest(unittest.TestCase):
    def test_splits_typed_and_untyped_parameters(self):
        variables, typed = text.parse_typed_parameters(
            "(?p - player ?c - crate ?from ?to - location)"
        )
        self.assertEqual(variables, ["?p", "?c", "?from", "?to"])
        self.assertEqual(
            typed,
            [
                ("?p", "player"),
                ("?c", "crate"),
                ("?from", "location"),
                ("?to", "location"),
            ],
        )

    def test_untyped_parameters_have_no_types(self):
        variables, typed = text.parse_typed_parameters("(?p ?q)")
        self.assertEqual(variables, ["?p", "?q"])
        self.assertEqual(typed, [])


class TypeHierarchyTest(unittest.TestCase):
    def test_groups_keep_children_with_their_parent(self):
        self.assertEqual(
            text.parse_type_groups(DOMAIN),
            [(["crate", "stone"], "locatable"), (["player"], "object")],
        )

    def test_parents_flattens_the_groups(self):
        self.assertEqual(
            text.type_parents(DOMAIN),
            {"crate": "locatable", "stone": "locatable", "player": "object"},
        )

    def test_domain_without_types_yields_nothing(self):
        self.assertEqual(text.parse_type_groups("(define (domain d))"), [])
        self.assertEqual(text.type_parents("(define (domain d))"), {})


class EqualityTest(unittest.TestCase):
    def test_detects_the_equality_requirement(self):
        self.assertTrue(text.domain_uses_equality(DOMAIN))
        self.assertFalse(
            text.domain_uses_equality("(define (domain d) (:requirements :strips))")
        )

    def test_rewrites_equality_atoms_as_a_predicate(self):
        self.assertEqual(
            text.replace_equality_with_predicate("(and (= ?a ?b) (p ?a))", "eq"),
            "(and (eq ?a ?b) (p ?a))",
        )


class InjectPreconditionsTest(unittest.TestCase):
    def test_flattens_into_an_existing_conjunction(self):
        self.assertEqual(
            text.inject_extra_preconditions("(and (p ?x))", ["(q ?x)"]),
            "(and (q ?x) (p ?x))",
        )

    def test_wraps_a_bare_precondition(self):
        self.assertEqual(
            text.inject_extra_preconditions("(p ?x)", ["(q ?x)"]),
            "(and (q ?x) (p ?x))",
        )

    def test_handles_empty_preconditions_and_no_extras(self):
        self.assertEqual(text.inject_extra_preconditions("(and)", ["(q)"]), "(and (q))")
        self.assertEqual(text.inject_extra_preconditions("()", ["(q)"]), "(and (q))")
        self.assertEqual(text.inject_extra_preconditions("(p ?x)", []), "(p ?x)")


if __name__ == "__main__":
    unittest.main()
