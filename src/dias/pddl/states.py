"""Reading and re-hosting pymimir states."""

from __future__ import annotations

from pymimir.advanced.formalism import FluentGroundAtomList, ObjectList
from pymimir.wrapper_formalism import State


def state_fluents(state, predicate_names=None) -> frozenset[str]:
    """The state's atoms as strings, optionally restricted to some predicates."""
    atoms = state.get_atoms(ignore_static=False, ignore_derived=True)
    if predicate_names is None:
        return frozenset(str(atom) for atom in atoms)
    return frozenset(
        str(atom)
        for atom in atoms
        if atom.get_predicate().get_name() in predicate_names
    )


def transfer_state(source_state, target_problem) -> State:
    """Rebuild `source_state`'s atoms as a state of `target_problem`.

    Both problems must share object and predicate names -- typically the same
    problem file read against a different domain. Predicates that are static in
    the target keep the target's own definition rather than being copied.
    """
    advanced = target_problem._advanced_problem
    fluent_by_name = {
        p.get_name(): p for p in advanced.get_domain().get_fluent_predicates()
    }
    object_by_name = {o.get_name(): o for o in advanced.get_objects()}

    atoms = []
    for atom in source_state.get_atoms(ignore_static=False, ignore_derived=True):
        fluent = fluent_by_name.get(atom.get_predicate().get_name())
        if fluent is None:
            continue
        objects = ObjectList(
            [object_by_name[term.get_name()] for term in atom.get_terms()]
        )
        atoms.append(advanced.get_or_create_ground_atom(fluent, objects))

    repository = target_problem._search_context.get_state_repository()
    advanced_state, _ = repository.get_or_create_state(FluentGroundAtomList(atoms), [])
    return State(advanced_state, target_problem)
