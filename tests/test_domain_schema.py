"""Tests for the domain metadata every other module is a function of."""

import unittest

from checkpoints import DOMAIN, TRAIN_PROBLEM, datamodule_hparams
from dias.schema import DomainSchema
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
            ["=", "clear", "number", "object", "on", "on-table"],
        )
        self.assertEqual(
            {p["name"] for p in predicates if p["static"]}, {"=", "number", "object"}
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

    def test_equality_is_a_relation_only_when_asked_for(self):
        predicates = self.PREDICATES + [{"name": "=", "arity": 2, "static": True}]
        names = lambda **kwargs: [
            p["name"]
            for p in tensor_relations(predicates, include_statics=True, **kwargs)
        ]
        self.assertEqual(names(), ["one", "two", "static"])
        self.assertEqual(names(include_equality=True), ["one", "two", "static", "="])


class CheckpointSchemaTest(unittest.TestCase):
    """A stored run rebuilds the relations it was trained with."""

    def relation_names(self, legacy: bool) -> tuple[str, ...]:
        checkpoint = {"datamodule_hyper_parameters": datamodule_hparams(legacy)}
        return DomainSchema.from_checkpoint(checkpoint).relation_names

    def test_a_run_from_before_the_argument_kept_equality(self):
        self.assertIn("=", self.relation_names(legacy=True))

    def test_a_current_run_has_no_equality_relation(self):
        self.assertNotIn("=", self.relation_names(legacy=False))


if __name__ == "__main__":
    unittest.main()
