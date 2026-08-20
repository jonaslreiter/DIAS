"""Tests for the domain metadata every other module is a function of."""

import unittest

from checkpoints import DOMAIN, TRAIN_PROBLEM
from dias.schema.domain import read_metadata, tensor_relations


class ReadMetadataTest(unittest.TestCase):
    """Read against a real domain: fakes of the pymimir API pin nothing."""

    @classmethod
    def setUpClass(cls):
        import pymimir

        domain = pymimir.Domain(DOMAIN)
        cls.metadata = read_metadata(domain, pymimir.Problem(domain, TRAIN_PROBLEM))

    def test_objects_are_indexed_both_ways(self):
        self.assertEqual(self.metadata["num_objects"], 5)
        self.assertEqual(self.metadata["objects"]["0"], "b1")
        self.assertEqual(self.metadata["object_name_to_idx"]["b1"], 0)

    def test_predicates_come_back_sorted_with_their_arity_and_staticness(self):
        """Sorted by name, so the tensor channels have a stable order."""
        predicates = self.metadata["predicates"]
        self.assertEqual(
            [p["name"] for p in predicates],
            ["clear", "eq", "number", "object", "on", "on-table"],
        )
        self.assertEqual(
            {p["name"] for p in predicates if p["static"]}, {"eq", "number", "object"}
        )
        self.assertEqual(predicates[4], {"name": "on", "arity": 2, "static": False})

    def test_actions_keep_the_order_the_domain_declares_them_in(self):
        self.assertEqual(
            self.metadata["actions"],
            [
                {"name": "stack", "arity": 2},
                {"name": "newtower", "arity": 2},
                {"name": "move", "arity": 3},
            ],
        )


class TensorRelationsTest(unittest.TestCase):
    PREDICATES = [
        {"name": "one", "arity": 1, "static": False},
        {"name": "two", "arity": 2, "static": False},
        {"name": "three", "arity": 3, "static": False},
        {"name": "static", "arity": 1, "static": True},
        {"name": "object", "arity": 1, "static": True},
    ]

    def names(self, **kwargs) -> list[str]:
        return [p["name"] for p in tensor_relations(self.PREDICATES, **kwargs)]

    def test_arity_above_two_and_statics_are_dropped(self):
        self.assertEqual(self.names(), ["one", "two"])

    def test_asking_for_statics_still_drops_the_universal_ones(self):
        self.assertEqual(self.names(include_statics=True), ["one", "two", "static"])

    def test_which_statics_count_as_universal_can_be_named(self):
        self.assertEqual(
            self.names(include_statics=True, default_statics=["static"]),
            ["one", "two", "object"],
        )


if __name__ == "__main__":
    unittest.main()
