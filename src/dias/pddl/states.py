"""Reading pymimir states and rebuilding them in another problem."""

from __future__ import annotations

from pymimir.advanced.formalism import FluentGroundAtomList, ObjectList
from pymimir.wrapper_formalism import State

# A state's atoms as ``(predicate, (object, ...))``.
StateAtoms = list[tuple[str, tuple[str, ...]]]


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


def state_atoms(state) -> StateAtoms:
    """The state's atoms, statics included, as plain data."""
    return [
        (
            atom.get_predicate().get_name(),
            tuple(term.get_name() for term in atom.get_terms()),
        )
        for atom in state.get_atoms(ignore_static=False, ignore_derived=True)
    ]


def rebuild_state(atoms: StateAtoms, target_problem) -> State:
    """The state of `target_problem` holding the atoms of its fluent predicates among `atoms`."""
    advanced = target_problem._advanced_problem
    fluent_by_name = {
        p.get_name(): p for p in advanced.get_domain().get_fluent_predicates()
    }
    object_by_name = {o.get_name(): o for o in advanced.get_objects()}
    ground = [
        advanced.get_or_create_ground_atom(
            fluent_by_name[predicate],
            ObjectList([object_by_name[name] for name in objects]),
        )
        for predicate, objects in atoms
        if predicate in fluent_by_name
    ]
    repository = target_problem._search_context.get_state_repository()
    advanced_state, _ = repository.get_or_create_state(FluentGroundAtomList(ground), [])
    return State(advanced_state, target_problem)


def transfer_state(source_state, target_problem) -> State:
    """Rebuild `source_state`'s atoms as a state of `target_problem`."""
    return rebuild_state(state_atoms(source_state), target_problem)
