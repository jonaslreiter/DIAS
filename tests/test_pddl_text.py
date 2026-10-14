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

    def test_a_type_without_a_parent_is_a_child_of_object(self):
        self.assertEqual(
            text.parse_type_groups(
                "(define (domain d) (:types satellite truck-bay - place mode))"
            ),
            [(["satellite", "truck-bay"], "place"), (["mode"], "object")],
        )


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


class DeclaredPredicatesTest(unittest.TestCase):
    def test_arities_count_variables_not_types(self):
        self.assertEqual(text.declared_predicates(DOMAIN), {"at": 2, "clear": 1})

    def test_comments_and_hyphenated_names(self):
        domain = "(define (domain d)\n  (:predicates (on-table ?x)\n    ;; ?x (a block) is held\n    (holding ?x)))"
        self.assertEqual(
            text.declared_predicates(domain), {"on-table": 1, "holding": 1}
        )


EQUAL_DOMAIN = """(define (domain d)
  (:requirements :strips :typing)
  (:types cell)
  (:predicates
    (at ?x - cell)
    (equal ?x - object ?y - object)
    (eq ?x)
  )
  (:action move
    :parameters (?from - cell ?to - cell)
    :precondition (and (at ?from) (not (equal ?from ?to)) (EQUAL ?to ?to))
    :effect (and (at ?to) (not (at ?from))))
)
"""


class EqualityPredicateTest(unittest.TestCase):
    def test_finds_a_binary_static_predicate_named_for_equality(self):
        # `eq` here is unary, so it cannot be equality.
        self.assertEqual(text.equality_predicates(EQUAL_DOMAIN), ["equal"])
        self.assertEqual(text.equality_predicates(DOMAIN), [])

    def test_a_predicate_an_action_changes_is_not_equality(self):
        changed = EQUAL_DOMAIN.replace("(at ?to) (not", "(at ?to) (equal ?to ?to) (not")
        self.assertEqual(text.equality_predicates(changed), [])

    def test_written_as_builtin(self):
        rewritten = text.write_equality_as_builtin(EQUAL_DOMAIN, ["equal"])
        self.assertIn("(not (= ?from ?to))", rewritten)
        self.assertIn("(= ?to ?to)", rewritten)
        self.assertNotIn("(equal", rewritten.lower())
        self.assertIn("(eq ?x)", rewritten)
        self.assertTrue(text.domain_uses_equality(rewritten))
        self.assertEqual(text.declared_predicates(rewritten), {"at": 1, "eq": 1})

    def test_written_as_builtin_without_requirements(self):
        domain = EQUAL_DOMAIN.replace("(:requirements :strips :typing)", "")
        rewritten = text.write_equality_as_builtin(domain, ["equal"])
        self.assertIn("(:requirements :equality)", rewritten)

    def test_nothing_to_rewrite_leaves_the_domain_alone(self):
        self.assertEqual(text.write_equality_as_builtin(DOMAIN, []), DOMAIN)

    def test_forced_distinct_parameters_are_pairwise(self):
        rewritten = text.force_distinct_parameters(DOMAIN)
        push = text.extract_action_blocks(rewritten)[0]
        for a, b in (("?p", "?c"), ("?p", "?from"), ("?c", "?to"), ("?from", "?to")):
            self.assertEqual(push.count(f"(not (= {a} {b}))"), 1)
        self.assertIn("(at ?p ?from)", push)
        move = text.extract_action_blocks(rewritten)[1]
        self.assertIn("(and (not (= ?p ?q)) (clear ?q))", move)
        self.assertEqual(text.force_distinct_parameters(rewritten), rewritten)

    def test_forcing_replaces_whatever_equality_was_learned(self):
        domain = "(define (domain d)\n  (:action a :parameters (?x ?y)\n    :precondition (and (= ?x ?x) (not(= ?y ?y)) (p ?x))\n    :effect (and (= ?x ?y) (not (p ?x)))))"
        action = text.extract_action_blocks(text.force_distinct_parameters(domain))[0]
        self.assertIn(":precondition (and (not (= ?x ?y)) (p ?x))", action)
        self.assertIn(":effect (and (not (p ?x)))", action)
        self.assertTrue(
            text.domain_uses_equality(text.force_distinct_parameters(EQUAL_DOMAIN))
        )

    def test_reflexive_facts_are_dropped(self):
        problem = "(define (problem p) (:domain d)\n(:objects a b - cell)\n(:init (at a) (equal a a)\n  (EQUAL b b))\n(:goal (at b)))"
        dropped = text.drop_equality_facts(problem, ["equal"])
        self.assertNotIn("(equal", dropped.lower())
        self.assertIn("(:init (at a))", dropped)

    def test_a_fact_between_two_objects_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not equality"):
            text.drop_equality_facts("(:init (equal a b))", ["equal"])


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
