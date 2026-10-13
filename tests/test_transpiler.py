import unittest

import numpy as np

from dias.pddl.transpiler import PDDLTranspiler
from dias.schema.action import ActionSchema, LiftedAtom


class TranspilerTest(unittest.TestCase):
    def test_ground_atom_string_equality_and_hash(self):
        atom = LiftedAtom("on", 2, [1, 0], True)
        self.assertEqual(str(atom), "(on ?x1 ?x0)")
        self.assertEqual(str(LiftedAtom("clear", 1, [0], False)), "(not (clear ?x0))")
        self.assertEqual(str(LiftedAtom("ready", 0, [], True)), "(ready)")
        self.assertEqual(atom, LiftedAtom("on", 2, [1, 0], True))
        self.assertNotEqual(atom, LiftedAtom("on", 2, [1, 0], False))
        self.assertEqual(len({atom, LiftedAtom("on", 2, [1, 0], True)}), 1)

    def test_action_schema_formats_empty_and_nonempty_sections(self):
        schema = ActionSchema(
            "stack",
            [LiftedAtom("clear", 1, [0], True)],
            [LiftedAtom("on", 2, [0, 1], True)],
        )
        text = str(schema)
        self.assertIn(":parameters (?x0 ?x1)", text)
        self.assertIn(":precondition (and (clear ?x0))", text)
        self.assertIn(":effect (and (on ?x0 ?x1))", text)
        empty = str(ActionSchema("noop", [], []))
        self.assertIn(":precondition ()", empty)
        self.assertIn(":effect ()", empty)

    def test_transpiler_generates_deduplicated_unary_and_binary_atoms(self):
        shape = (1, 2, 2, 2)
        add = np.zeros(shape, dtype=np.int64)
        delete = np.zeros(shape, dtype=np.int64)
        positive = np.zeros(shape, dtype=np.int64)
        negative = np.zeros(shape, dtype=np.int64)
        add[0, 0, 0, 1] = 1
        add[0, 0, 1, 0] = 1
        delete[0, 1, 1, 1] = 1
        positive[0, 0, 0, 0] = 1
        negative[0, 1, 0, 1] = 1
        transpiler = PDDLTranspiler(
            add,
            delete,
            positive,
            negative,
            ["on", "clear"],
            [2, 1],
            ["stack"],
        )
        schemas = transpiler.generate_action_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(
            [str(atom) for atom in schemas[0].preconditions],
            ["(on ?x0 ?x0)", "(not (clear ?x0))"],
        )
        self.assertEqual(
            [str(atom) for atom in schemas[0].effects],
            ["(on ?x0 ?x1)", "(on ?x1 ?x0)", "(not (clear ?x1))"],
        )

    def test_transpiler_validates_array_and_metadata_shapes(self):
        valid = np.zeros((1, 1, 1, 1), dtype=np.int64)
        kwargs = dict(
            add_effects=valid,
            delete_effects=valid.copy(),
            positive_preconditions=valid.copy(),
            negative_preconditions=valid.copy(),
            predicate_names=["ready"],
            predicate_arities=[0],
            action_names=["go"],
        )
        with self.assertRaisesRegex(TypeError, "must be a numpy array"):
            PDDLTranspiler(**{**kwargs, "add_effects": [[[[0]]]]})
        with self.assertRaisesRegex(ValueError, "exactly 4 dimensions"):
            PDDLTranspiler(**{**kwargs, "add_effects": np.zeros((1, 1, 1))})
        with self.assertRaisesRegex(ValueError, "only contain values"):
            PDDLTranspiler(**{**kwargs, "add_effects": np.full((1, 1, 1, 1), 2)})
        with self.assertRaisesRegex(ValueError, "same shape"):
            PDDLTranspiler(**{**kwargs, "delete_effects": np.zeros((2, 1, 1, 1))})
        with self.assertRaisesRegex(TypeError, "predicate_names"):
            PDDLTranspiler(**{**kwargs, "predicate_names": ("ready",)})
        with self.assertRaisesRegex(ValueError, "same length"):
            PDDLTranspiler(**{**kwargs, "predicate_arities": []})
        with self.assertRaisesRegex(ValueError, "one action name per action"):
            PDDLTranspiler(**{**kwargs, "action_names": []})

    def test_add_unreachable_action_references_original_predicates(self):
        valid = np.zeros((1, 1, 1, 1), dtype=np.int64)
        transpiler = PDDLTranspiler(
            valid,
            valid.copy(),
            valid.copy(),
            valid.copy(),
            ["ready"],
            [1],
            ["go"],
        )
        schema = transpiler.add_unreachable_action()
        self.assertEqual(str(schema.preconditions[0]), "(dummy_guard)")
        self.assertEqual(
            [str(atom) for atom in schema.effects],
            ["(ready ?x0)", "(dummy_guard)"],
        )
        self.assertIn("dummy_action_unreachable", str(transpiler))

    def test_add_unreachable_action_leaves_the_caller_lists_alone(self):
        valid = np.zeros((1, 1, 1, 1), dtype=np.int64)
        names, arities = ["ready"], [1]
        transpiler = PDDLTranspiler(
            valid, valid.copy(), valid.copy(), valid.copy(), names, arities, ["go"]
        )
        transpiler.add_unreachable_action()
        self.assertEqual(transpiler.predicate_names, ["ready", "dummy_guard"])
        self.assertEqual(names, ["ready"])
        self.assertEqual(arities, [1])

    def test_distinct_parameters_are_required_once_and_declared(self):
        on = np.zeros((1, 2, 3, 3), dtype=np.int64)
        on[0, 0, 0, 1] = 1
        pre = np.zeros_like(on)
        pre[0, 1, 1, 0] = 1  # (not (eq ?x1 ?x0)), already learned
        transpiler = PDDLTranspiler(
            on, np.zeros_like(on), np.zeros_like(on), pre, ["on", "eq"], [2, 2], ["a"]
        )
        transpiler.generate_action_schemas()
        transpiler.require_distinct("eq")
        self.assertEqual(
            [str(atom) for atom in transpiler.action_schemas[0].preconditions],
            ["(not (eq ?x1 ?x0))"],
        )
        self.assertEqual(transpiler.predicate_names, ["on", "eq"])

    def test_distinct_parameters_cover_named_slots_and_declare_the_predicate(self):
        empty = np.zeros((1, 1, 3, 3), dtype=np.int64)
        transpiler = PDDLTranspiler(
            empty, empty.copy(), empty.copy(), empty.copy(), ["on"], [2], ["a"]
        )
        transpiler.generate_action_schemas()
        transpiler.require_distinct("eq", [[0, 2]])
        self.assertEqual(
            [str(atom) for atom in transpiler.action_schemas[0].preconditions],
            ["(not (eq ?x0 ?x2))"],
        )
        self.assertEqual(transpiler.predicate_names, ["on", "eq"])
        self.assertEqual(transpiler.predicate_arities, [2, 2])

    def test_a_static_predicate_stays_out_of_the_unreachable_action(self):
        valid = np.zeros((1, 2, 1, 1), dtype=np.int64)
        transpiler = PDDLTranspiler(
            valid,
            valid.copy(),
            valid.copy(),
            valid.copy(),
            ["ready", "eq"],
            [1, 2],
            ["go"],
        )
        schema = transpiler.add_unreachable_action(static_predicate_names=("eq",))
        self.assertEqual(
            [str(atom) for atom in schema.effects], ["(ready ?x0)", "(dummy_guard)"]
        )


if __name__ == "__main__":
    unittest.main()
