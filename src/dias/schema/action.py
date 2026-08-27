"""Lifted action schemas and the atoms they are built from, serialising to PDDL. Slot indices are the parameter identity: an atom refers to ``?x{slot}``, and the parameter list is the union of the slots its atoms mention."""

from __future__ import annotations


class GroundAtom:
    """One (possibly negated) atom over slot indices, e.g. ``(not (on ?x0 ?x1))``."""

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
    """A named action with lifted preconditions and effects."""

    def __init__(
        self, name: str, preconditions: list[GroundAtom], effects: list[GroundAtom]
    ):
        self.name = name
        self.preconditions = preconditions
        self.effects = effects

    def __str__(self):
        slots = sorted(
            {i for atom in self.preconditions + self.effects for i in atom.object_index}
        )
        params = " ".join(f"?x{i}" for i in slots)

        def conjunction(atoms: list[GroundAtom]) -> str:
            return "(and " + " ".join(str(a) for a in atoms) + ")" if atoms else "()"

        return f"(:action {self.name}\n  :parameters ({params})\n  :precondition {conjunction(self.preconditions)}\n  :effect {conjunction(self.effects)})\n\n"
