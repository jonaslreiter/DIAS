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

    def test_distinct_parameters_are_required_pairwise_and_declared(self):
        on = np.zeros((1, 1, 3, 3), dtype=np.int64)
        on[0, 0, 0, 1] = 1
        on[0, 0, 1, 2] = 1
        transpiler = PDDLTranspiler(
            on,
            np.zeros_like(on),
            np.zeros_like(on),
            np.zeros_like(on),
            ["on"],
            [2],
            ["a"],
        )
        transpiler.generate_action_schemas()
        transpiler.require_distinct("=")
        transpiler.require_distinct("=")
        self.assertEqual(
            [str(atom) for atom in transpiler.action_schemas[0].preconditions],
            ["(not (= ?x0 ?x1))", "(not (= ?x0 ?x2))", "(not (= ?x1 ?x2))"],
        )
        self.assertEqual(transpiler.predicate_names, ["on", "="])

    def test_distinct_parameters_follow_parameters_no_atom_mentions(self):
        empty = np.zeros((1, 1, 3, 3), dtype=np.int64)
        transpiler = PDDLTranspiler(
            empty, empty.copy(), empty.copy(), empty.copy(), ["on"], [2], ["a"]
        )
        transpiler.generate_action_schemas()
        transpiler.action_schemas[0].parameters = [0, 2]
        transpiler.require_distinct("=")
        self.assertEqual(
            [str(atom) for atom in transpiler.action_schemas[0].preconditions],
            ["(not (= ?x0 ?x2))"],
        )
        self.assertIn(":parameters (?x0 ?x2)", str(transpiler.action_schemas[0]))


if __name__ == "__main__":
    unittest.main()
