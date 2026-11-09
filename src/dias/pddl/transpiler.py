"""Binarized schema tensors to lifted action schemas."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from dias.schema.action import ActionSchema, LiftedAtom


class PDDLTranspiler:
    """Action schemas from 0/1 effect and precondition tensors ``(A, R, M, M)``; a unary predicate reads its slot off the diagonal."""

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
                    f"All arrays must have the same shape. add_effects has shape {reference_shape}, but {name} has shape {arr.shape}"
                )

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
                f"predicate_names and predicate_arities must have the same length, got {len(predicate_names)} and {len(predicate_arities)}"
            )
        if len(action_names) != add_effects.shape[0]:
            raise ValueError(
                f"There must be one action name per action (first axis of the effects and preconditions), got {len(action_names)} and {add_effects.shape[0]}"
            )

        self.add_effects = add_effects
        self.delete_effects = delete_effects
        self.positive_preconditions = positive_preconditions
        self.negative_preconditions = negative_preconditions
        self.predicate_names = list(predicate_names)
        self.predicate_arities = list(predicate_arities)
        self.action_names = action_names

        self.num_actions = add_effects.shape[0]
        self.num_predicates = add_effects.shape[1]
        self.num_slots = add_effects.shape[2]

        self.action_schemas: list[ActionSchema] = []

    def _atoms(self, action: int, tensor: np.ndarray, positive: bool):
        """The distinct atoms one channel asserts for one action."""
        atoms: list[LiftedAtom] = []
        for r, s1, s2 in zip(*np.where(tensor[action] == 1)):
            arity = self.predicate_arities[r]
            atom = LiftedAtom(
                self.predicate_names[r], arity, [s1, s2][:arity], positive
            )
            if atom not in atoms:
                atoms.append(atom)
        return atoms

    def generate_action_schemas(self) -> list[ActionSchema]:
        self.action_schemas = [
            ActionSchema(
                name=name,
                preconditions=self._atoms(a, self.positive_preconditions, True)
                + self._atoms(a, self.negative_preconditions, False),
                effects=self._atoms(a, self.add_effects, True)
                + self._atoms(a, self.delete_effects, False),
            )
            for a, name in enumerate(self.action_names)
        ]
        return self.action_schemas

    def declare_predicate(self, name: str, arity: int) -> None:
        """Declare a predicate the tensors have no channel for."""
        if name not in self.predicate_names:
            self.predicate_names.append(name)
            self.predicate_arities.append(arity)
            self.num_predicates += 1

    def require_distinct(self, predicate_name: str) -> None:
        """Require every two parameters of every schema to differ, as ``(not (predicate_name ?a ?b))``."""
        self.declare_predicate(predicate_name, 2)
        for schema in self.action_schemas:
            for x, y in combinations(schema.parameters, 2):
                distinct = LiftedAtom(predicate_name, 2, [x, y], False)
                if distinct not in schema.preconditions:
                    schema.preconditions.append(distinct)

    def __str__(self):
        return "\n\n".join(str(s) for s in self.action_schemas).strip()
