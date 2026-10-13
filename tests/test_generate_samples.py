import unittest

from dias.data.generate_samples import (
    generate_bfs,
    generate_random_walk,
    reachable_states,
    samples_to_tensors,
    state_to_groundings,
)


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


class SampleGenerationTest(unittest.TestCase):
    def test_state_to_groundings_reads_only_the_fluents_unless_asked(self):
        fluent = FakePredicate("ready", 1)
        static = FakePredicate("type", 1, static=True)
        state = FakeState(0, [FakeAtom(fluent, [1]), FakeAtom(static, [0])])
        self.assertEqual(state_to_groundings(state), {"ready": [[1]]})
        self.assertEqual(
            state_to_groundings(state, include_statics=True),
            {"ready": [[1]], "type": [[0]]},
        )

    def test_random_walk_emits_samples_and_rejects_dead_initial_state(self):
        action = FakeAction("go", 1)
        predicate = FakePredicate("ready", 1)
        terminal = FakeState(1, [FakeAtom(predicate, [0])])
        initial = FakeState(0, [FakeAtom(predicate, [0])])
        initial.actions = [FakeGroundAction(action, [FakeObject(0, "a")], terminal)]
        domain = FakeDomain("demo", [predicate], [action])
        problem = FakeProblem(domain, initial, [FakeObject(0, "a")])
        samples = generate_random_walk(problem, min_total_samples=1, max_steps=3)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["action"], "go")
        self.assertEqual(samples[0]["action_args"], [0])

        initial.actions = []
        with self.assertRaisesRegex(ValueError, "no applicable actions"):
            generate_random_walk(problem)

    def test_random_walk_warns_when_quota_is_not_reached(self):
        action = FakeAction("go", 0)
        predicate = FakePredicate("ready", 1)
        state = FakeState(0, [FakeAtom(predicate, [0])])
        state.actions = [FakeGroundAction(action, [], state)]
        problem = FakeProblem(
            FakeDomain("demo", [predicate], [action]),
            state,
            [FakeObject(0, "a")],
        )
        samples = generate_random_walk(
            problem,
            min_per_action=2,
            max_steps=1,
            only_unseen_states=True,
        )
        self.assertEqual(len(samples), 1)

    def test_bfs_explores_new_states_and_emits_applicable_actions(self):
        action = FakeAction("go", 0)
        predicate = FakePredicate("ready", 1)
        terminal = FakeState(1, [FakeAtom(predicate, [0])])
        initial = FakeState(0, [FakeAtom(predicate, [0])])
        initial.actions = [FakeGroundAction(action, [], terminal)]
        problem = FakeProblem(
            FakeDomain("demo", [predicate], [action]),
            initial,
            [FakeObject(0, "a")],
        )
        samples = generate_bfs(problem, num_states=2)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["action"], "go")

    def test_an_action_applicable_to_one_object_twice_is_refused(self):
        """Even when the walk does not take it: the model could not learn the domain either way."""
        action = FakeAction("move", 2)
        predicate = FakePredicate("ready", 1)
        a, b = FakeObject(0, "a"), FakeObject(1, "b")
        terminal = FakeState(1, [FakeAtom(predicate, [0])])
        initial = FakeState(0, [FakeAtom(predicate, [0])])
        initial.actions = [
            FakeGroundAction(action, [a, b], terminal),
            FakeGroundAction(action, [a, a], initial),
        ]
        problem = FakeProblem(
            FakeDomain("demo", [predicate], [action]), initial, [a, b]
        )
        with self.assertRaisesRegex(ValueError, "injective"):
            generate_random_walk(problem, min_total_samples=1, max_steps=3)
        with self.assertRaisesRegex(ValueError, "injective"):
            generate_bfs(problem, num_states=2)

    def test_samples_to_tensors_encodes_unary_binary_and_actions(self):
        metadata = {
            "predicates": [
                {"name": "clear", "arity": 1, "static": False},
                {"name": "on", "arity": 2, "static": False},
            ],
            "actions": [{"name": "stack", "arity": 2}],
            "num_objects": 2,
            "objects": {"0": "a", "1": "b"},
            "object_name_to_idx": {"a": 0, "b": 1},
        }
        samples = [
            {
                "state": {"clear": [[0]], "on": [[0, 1]]},
                "next_state": {"clear": [[1]], "on": [[1, 0]]},
                "action": "stack",
                "action_args": [0, 1],
                "applicable_actions": {"stack": [[0, 1]]},
            }
        ]
        result = samples_to_tensors(samples, metadata)
        self.assertEqual(tuple(result["states"].shape), (1, 2, 2, 2))
        self.assertEqual(result["states"][0, 0, 0, 0].item(), 1.0)
        self.assertEqual(result["states"][0, 1, 0, 1].item(), 1.0)
        self.assertEqual(result["next_states"][0, 0, 1, 1].item(), 1.0)
        self.assertEqual(result["action_indices"].tolist(), [0])
        self.assertEqual(result["action_args"].tolist(), [[0, 1]])
        self.assertEqual(result["applicable_actions"], [{0: [(0, 1)]}])

    def test_samples_to_tensors_rejects_nullary_predicate_groundings(self):
        metadata = {
            "predicates": [{"name": "ready", "arity": 0, "static": False}],
            "actions": [{"name": "go", "arity": 0}],
            "num_objects": 1,
            "objects": {"0": "a"},
            "object_name_to_idx": {"a": 0},
        }
        sample = {
            "state": {"ready": [[]]},
            "next_state": {},
            "action": "go",
            "action_args": [],
            "applicable_actions": {},
        }
        with self.assertRaisesRegex(ValueError, "Nullary predicate"):
            samples_to_tensors([sample], metadata)

    def test_samples_to_tensors_rejects_domains_without_tensor_relations(self):
        metadata = {
            "predicates": [{"name": "too_many", "arity": 3, "static": False}],
            "actions": [],
            "num_objects": 1,
            "objects": {"0": "a"},
            "object_name_to_idx": {"a": 0},
        }
        with self.assertRaisesRegex(ValueError, "No predicates"):
            samples_to_tensors([], metadata)


if __name__ == "__main__":
    unittest.main()
