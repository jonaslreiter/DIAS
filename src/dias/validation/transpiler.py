import numpy as np


class GroundAtom:
    def __init__(
        self, predicate_name: str, arity: int, object_index: list[int], positive: bool
    ):

        assert arity == len(object_index)

        self.predicate_name = predicate_name
        self.arity = arity
        self.object_index = object_index
        self.positive = positive

    def __str__(self):
        args = " ".join(f"?x{i}" for i in self.object_index)
        atom = (
            f"({self.predicate_name})"
            if self.arity == 0
            else f"({self.predicate_name} {args})"
        )
        return atom if self.positive else f"(not {atom})"

    def __eq__(self, other):
        if not isinstance(other, GroundAtom):
            return NotImplemented
        return (
            self.predicate_name == other.predicate_name
            and self.arity == other.arity
            and self.positive == other.positive
            and self.object_index == other.object_index
        )

    def __hash__(self):
        return hash(
            (self.predicate_name, self.arity, self.positive, tuple(self.object_index))
        )


class ActionSchema:
    def __init__(
        self, name: str, preconditions: list[GroundAtom], effects: list[GroundAtom]
    ):
        self.name = name
        self.preconditions = preconditions
        self.effects = effects

    def __str__(self):
        all_indices = sorted(
            set(
                i
                for atom in self.preconditions + self.effects
                for i in atom.object_index
            )
        )
        params = " ".join(f"?x{i}" for i in all_indices)

        precs_list = [str(p) for p in self.preconditions]
        effs_list = [str(e) for e in self.effects]
        prec_str = "(and " + " ".join(precs_list) + ")" if precs_list else "()"
        eff_str = "(and " + " ".join(effs_list) + ")" if effs_list else "()"

        return f"(:action {self.name}\n  :parameters ({params})\n  :precondition {prec_str}\n  :effect {eff_str})\n\n"


class PDDLTranspiler:
    """
    args shape: (actions, predicates, slots, slots)
      - actions:    number of operators
      - predicates: number of predicates
      - slots:      number of object slots (padded to max arity)
    """

    def __init__(
        self,
        add_effects: np.ndarray,
        delete_effects: np.ndarray,
        positive_preconditions: np.ndarray,
        negative_preconditions: np.ndarray,
        predicate_names: list[str],
        predicate_arities: list[int],
        action_names: list[str],
    ):
        # Validate numpy arrays
        arrays = {
            "add_effects": add_effects,
            "delete_effects": delete_effects,
            "positive_preconditions": positive_preconditions,
            "negative_preconditions": negative_preconditions,
        }

        for name, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(arr)}")
            if arr.ndim != 4:
                raise ValueError(
                    f"{name} must have exactly 4 dimensions, got {arr.ndim}"
                )
            if not np.all(np.isin(arr, [0, 1])):
                raise ValueError(f"{name} must only contain values in {{0, 1}}")

        reference_shape = add_effects.shape
        for name, arr in arrays.items():
            if arr.shape != reference_shape:
                raise ValueError(
                    f"All arrays must have the same shape. "
                    f"add_effects has shape {reference_shape}, but {name} has shape {arr.shape}"
                )

        # Validate predicate names and arities
        if not isinstance(predicate_names, list) or not all(
            isinstance(n, str) for n in predicate_names
        ):
            raise TypeError("predicate_names must be a list of strings")
        if not isinstance(predicate_arities, list) or not all(
            isinstance(a, int) for a in predicate_arities
        ):
            raise TypeError("predicate_arities must be a list of ints")
        if len(predicate_names) != len(predicate_arities):
            raise ValueError(
                f"predicate_names and predicate_arities must have the same length, "
                f"got {len(predicate_names)} and {len(predicate_arities)}"
            )
        if len(action_names) != add_effects.shape[0]:
            raise ValueError(
                f"The number of action names must be equal to the number of action (first axis of effects and preconditions), "
                f"got {len(action_names)} and {add_effects.shape[0]}"
            )

        self.add_effects = add_effects
        self.delete_effects = delete_effects
        self.positive_preconditions = positive_preconditions
        self.negative_preconditions = negative_preconditions
        self.predicate_names = predicate_names
        self.predicate_arities = predicate_arities
        self.action_names = action_names

        self.num_actions = add_effects.shape[0]
        self.num_predicates = add_effects.shape[1]
        self.num_slots = add_effects.shape[2]

        self.action_schemas: list[ActionSchema] = []

    def generate_action_schemas(self):
        self.action_schemas.clear()

        for a, action_name in enumerate(self.action_names):
            # --- 1. PRECONDITIONS ---
            # Using a set to prevent duplicate unary predicates if s2 varies but s1 is the same
            precs = []

            # Positive Preconditions
            r_idx, s1_idx, s2_idx = np.where(self.positive_preconditions[a] == 1)
            for r, s1, s2 in zip(r_idx, s1_idx, s2_idx):
                object_indicies = [s1, s2]
                ga = GroundAtom(
                    self.predicate_names[r],
                    self.predicate_arities[r],
                    object_indicies[: self.predicate_arities[r]],
                    True,
                )
                if ga not in precs:
                    precs.append(ga)

            # Negative Preconditions
            r_idx, s1_idx, s2_idx = np.where(self.negative_preconditions[a] == 1)
            for r, s1, s2 in zip(r_idx, s1_idx, s2_idx):
                object_indicies = [s1, s2]
                ga = GroundAtom(
                    self.predicate_names[r],
                    self.predicate_arities[r],
                    object_indicies[: self.predicate_arities[r]],
                    False,
                )
                if ga not in precs:
                    precs.append(ga)

            # --- 2. EFFECTS ---
            effs = []

            # Add Effects (Positive)
            r_idx, s1_idx, s2_idx = np.where(self.add_effects[a] == 1)
            for r, s1, s2 in zip(r_idx, s1_idx, s2_idx):
                object_indicies = [s1, s2]
                ga = GroundAtom(
                    self.predicate_names[r],
                    self.predicate_arities[r],
                    object_indicies[: self.predicate_arities[r]],
                    True,
                )
                if ga not in effs:
                    effs.append(ga)

            # Delete Effects (Negative)
            r_idx, s1_idx, s2_idx = np.where(self.delete_effects[a] == 1)
            for r, s1, s2 in zip(r_idx, s1_idx, s2_idx):
                object_indicies = [s1, s2]
                ga = GroundAtom(
                    self.predicate_names[r],
                    self.predicate_arities[r],
                    object_indicies[: self.predicate_arities[r]],
                    False,
                )
                if ga not in effs:
                    effs.append(ga)

            self.action_schemas.append(
                ActionSchema(name=action_name, preconditions=precs, effects=effs)
            )

        return self.action_schemas

    def add_unreachable_action(
        self,
        action_name: str = "dummy_action_unreachable",
        guard_predicate_name: str = "dummy_guard",
    ):
        """Add an action that references all predicates but is never reachable.

        Unreachability is guaranteed by a dummy guard predicate (arity 0) that
        is required as a precondition but is never produced by any action and
        should never appear in any initial state.  This avoids contradictory
        preconditions that some PDDL compilers optimise away.
        """
        # Register the guard predicate on the transpiler
        self.predicate_names.append(guard_predicate_name)
        self.predicate_arities.append(0)
        self.num_predicates += 1

        # Build preconditions: the guard predicate makes this unreachable
        precs: list[GroundAtom] = [
            GroundAtom(guard_predicate_name, arity=0, object_index=[], positive=True)
        ]

        # Effects: reference every original predicate so they appear in the schema
        effs: list[GroundAtom] = []
        for pred_name, arity in zip(self.predicate_names, self.predicate_arities):
            object_indices = list(range(arity))
            effs.append(GroundAtom(pred_name, arity, object_indices, positive=True))

        schema = ActionSchema(name=action_name, preconditions=precs, effects=effs)
        self.action_schemas.append(schema)
        return schema

    def __str__(self):
        schema_str = ""

        for action_schema in self.action_schemas:
            schema_str += str(action_schema) + "\n\n"

        return schema_str.strip()
