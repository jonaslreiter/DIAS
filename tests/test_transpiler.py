import unittest

import numpy as np

from dias.validation.transpiler import ActionSchema, GroundAtom, PDDLTranspiler


class TranspilerTest(unittest.TestCase):
    def test_ground_atom_string_equality_and_hash(self):
        atom = GroundAtom("on", 2, [1, 0], True)
        self.assertEqual(str(atom), "(on ?x1 ?x0)")
        self.assertEqual(str(GroundAtom("clear", 1, [0], False)), "(not (clear ?x0))")
        self.assertEqual(str(GroundAtom("ready", 0, [], True)), "(ready)")
        self.assertEqual(atom, GroundAtom("on", 2, [1, 0], True))
        self.assertNotEqual(atom, GroundAtom("on", 2, [1, 0], False))
        self.assertEqual(len({atom, GroundAtom("on", 2, [1, 0], True)}), 1)

    def test_action_schema_formats_empty_and_nonempty_sections(self):
        schema = ActionSchema(
            "stack",
            [GroundAtom("clear", 1, [0], True)],
            [GroundAtom("on", 2, [0, 1], True)],
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
        with self.assertRaisesRegex(ValueError, "number of action names"):
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


if __name__ == "__main__":
    unittest.main()
