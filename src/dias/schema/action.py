"""Lifted action schemas and their atoms, written as PDDL over the parameters ``?x<slot>``."""

from __future__ import annotations


class LiftedAtom:
    """A lifted atom over slots, e.g. ``(not (on ?x0 ?x1))``."""

    def __init__(
        self, predicate_name: str, arity: int, slots: list[int], positive: bool
    ):
        assert arity == len(slots)

        self.predicate_name = predicate_name
        self.arity = arity
        self.slots = slots
        self.positive = positive

    def __str__(self):
        args = " ".join(f"?x{i}" for i in self.slots)
        atom = (
            f"({self.predicate_name})"
            if self.arity == 0
            else f"({self.predicate_name} {args})"
        )
        return atom if self.positive else f"(not {atom})"

    def __eq__(self, other):
        if not isinstance(other, LiftedAtom):
            return NotImplemented
        return (
            self.predicate_name == other.predicate_name
            and self.arity == other.arity
            and self.positive == other.positive
            and self.slots == other.slots
        )

    def __hash__(self):
        return hash((self.predicate_name, self.arity, self.positive, tuple(self.slots)))


class ActionSchema:
    """An action schema, whose parameters are the slots its atoms use unless given."""

    def __init__(
        self,
        name: str,
        preconditions: list[LiftedAtom],
        effects: list[LiftedAtom],
        parameters: list[int] | None = None,
    ):
        self.name = name
        self.preconditions = preconditions
        self.effects = effects
        self._parameters = parameters

    @property
    def parameters(self) -> list[int]:
        if self._parameters is not None:
            return list(self._parameters)
        return sorted(
            {i for atom in self.preconditions + self.effects for i in atom.slots}
        )

    @parameters.setter
    def parameters(self, slots: list[int]) -> None:
        self._parameters = list(slots)

    def __str__(self):
        params = " ".join(f"?x{i}" for i in self.parameters)

        def conjunction(atoms: list[LiftedAtom]) -> str:
            return "(and " + " ".join(str(a) for a in atoms) + ")" if atoms else "()"

        return f"(:action {self.name}\n  :parameters ({params})\n  :precondition {conjunction(self.preconditions)}\n  :effect {conjunction(self.effects)})\n\n"
