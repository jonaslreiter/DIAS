"""Tests for the domain metadata every other module is a function of."""

import unittest

from dias.schema.domain import read_metadata, tensor_relations


class FakeTerm:
    def __init__(self, index):
        self.index = index

    def get_index(self):
        return self.index


class FakePredicate:
    def __init__(self, name, arity, static=False):
        self.name = name
        self.arity = arity
        self.static = static

    def get_name(self):
        return self.name

    def get_arity(self):
        return self.arity

    def is_static(self):
        return self.static


class FakeAtom:
    def __init__(self, predicate, indices):
        self.predicate = predicate
        self.indices = indices

    def get_predicate(self):
        return self.predicate

    def get_terms(self):
        return [FakeTerm(index) for index in self.indices]


class FakeObject:
    def __init__(self, index, name):
        self.index = index
        self.name = name

    def get_index(self):
        return self.index

    def get_name(self):
        return self.name


class FakeAction:
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

    def get_name(self):
        return self.name

    def get_arity(self):
        return self.arity


class FakeGroundAction:
    def __init__(self, action, objects, successor):
        self.action = action
        self.objects = objects
        self.successor = successor

    def get_action(self):
        return self.action

    def get_objects(self):
        return self.objects

    def apply(self, state):
        return self.successor


class FakeState:
    def __init__(self, index, atoms, actions=None):
        self.index = index
        self.atoms = atoms
        self.actions = actions or []

    def get_index(self):
        return self.index

    def get_atoms(self):
        return self.atoms

    def generate_applicable_actions(self):
        return self.actions


class FakeDomain:
    def __init__(self, name, predicates, actions):
        self.name = name
        self.predicates = predicates
        self.actions = actions

    def get_name(self):
        return self.name

    def get_predicates(self):
        return self.predicates

    def get_actions(self):
        return self.actions


class FakeProblem:
    def __init__(self, domain, initial, objects):
        self._domain = domain
        self.initial = initial
        self.objects = objects

    def get_initial_state(self):
        return self.initial

    def get_objects(self):
        return self.objects


class ReadMetadataTest(unittest.TestCase):
    def test_it_reports_the_domain_objects_predicates_and_actions(self):
        fluent = FakePredicate("ready", 1)
        static = FakePredicate("type", 1, static=True)
        state = FakeState(0, [FakeAtom(fluent, [1]), FakeAtom(static, [0])])
        action = FakeAction("go", 1)
        domain = FakeDomain("demo", [static, fluent], [action])
        problem = FakeProblem(domain, state, [FakeObject(0, "a"), FakeObject(1, "b")])

        metadata = read_metadata(domain, problem)
        self.assertEqual(metadata["domain_name"], "demo")
        self.assertEqual(metadata["objects"], {"0": "a", "1": "b"})
        self.assertEqual(metadata["object_name_to_idx"], {"a": 0, "b": 1})
        self.assertEqual(metadata["num_objects"], 2)
        self.assertEqual(metadata["actions"], [{"name": "go", "arity": 1}])
        # Sorted by name, so the tensor channels have a stable order.
        self.assertEqual([p["name"] for p in metadata["predicates"]], ["ready", "type"])


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
