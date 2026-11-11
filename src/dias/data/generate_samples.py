"""Sampling ``(s, a, s')`` transitions from a PDDL problem by random walk (paper, App. C.2)."""

import logging
import random
import time
from typing import Any, NamedTuple

from dias.schema.domain import tensor_relations

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL_SECONDS = 10.0


class _Progress:
    """Logs at most every ``PROGRESS_INTERVAL_SECONDS``, the first time after that long."""

    def __init__(self, message: str):
        self.message = message
        self.due = time.monotonic() + PROGRESS_INTERVAL_SECONDS

    def report(self, *values) -> None:
        now = time.monotonic()
        if now < self.due:
            return
        self.due = now + PROGRESS_INTERVAL_SECONDS
        logger.info(self.message, *values)


def state_to_groundings(state):
    """A pymimir state as ``{predicate: [[object index, ...], ...]}``."""
    result: dict[str, list[list[int]]] = {}
    for atom in state.get_atoms():
        result.setdefault(atom.get_predicate().get_name(), []).append(
            [t.get_index() for t in atom.get_terms()]
        )
    return result


def _groundings_by_action(actions):
    """Ground actions as ``{action: [(object index, ...), ...]}``."""
    groundings: dict[str, list[tuple[int, ...]]] = {}
    for ground_action in actions:
        groundings.setdefault(ground_action.get_action().get_name(), []).append(
            tuple(o.get_index() for o in ground_action.get_objects())
        )
    return groundings


def _require_injective(groundings_by_action: dict) -> None:
    """Refuse a state in which an action applies with one object in two parameters, since DIAS assumes injective bindings."""
    for name, groundings in groundings_by_action.items():
        for objects in groundings:
            if len(set(objects)) < len(objects):
                raise ValueError(
                    f"{name}{objects} binds one object to two parameters, but DIAS assumes injective bindings. A domain whose actions apply that way needs (not (= ?a ?b)) preconditions to be learned here."
                )


def _make_sample(state, ground_action, successor):
    applicable = _groundings_by_action(state.generate_applicable_actions())
    _require_injective(applicable)
    return {
        "state": state_to_groundings(state),
        "action": ground_action.get_action().get_name(),
        "action_args": [o.get_index() for o in ground_action.get_objects()],
        "next_state": state_to_groundings(successor),
        "applicable_actions": applicable,
    }


class WalkStep(NamedTuple):
    """One step of `random_walk`: the state, its applicable ground actions, and the action taken with its successor, both None at a dead end."""

    state: Any
    applicable: list
    action: Any
    successor: Any


def random_walk(problem, rng=random):
    """An endless random walk from the initial state: a uniformly drawn applicable ground action per step, and back to the initial state at a dead end. `rng` is anything with a `choice`."""
    initial = problem.get_initial_state()
    current = initial
    while True:
        applicable = list(current.generate_applicable_actions())
        if not applicable:
            yield WalkStep(current, applicable, None, None)
            current = initial
            continue
        action = rng.choice(applicable)
        successor = action.apply(current)
        yield WalkStep(current, applicable, action, successor)
        current = successor


def generate_random_walk(
    problem,
    min_total_samples: int = 0,
    min_per_action: int = 0,
    max_steps: int = 1_000_000,
):
    """The transitions of a random walk until it has ``min_total_samples`` and ``min_per_action`` of each action, or ``max_steps``."""
    initial = problem.get_initial_state()
    if not list(initial.generate_applicable_actions()):
        raise ValueError("initial state has no applicable actions.")

    action_names = [a.get_name() for a in problem._domain.get_actions()]
    walk = random_walk(problem)
    samples: list[dict] = []
    per_action: dict[str, int] = {}
    deadlocks = 0

    def quota_met() -> bool:
        return len(samples) >= min_total_samples and all(
            per_action.get(a, 0) >= min_per_action for a in action_names
        )

    steps = 0
    progress = _Progress("Random walk: %d steps, %d samples, %d/%d actions at quota.")
    while steps < max_steps:
        steps += 1
        progress.report(
            steps,
            len(samples),
            sum(per_action.get(a, 0) >= min_per_action for a in action_names),
            len(action_names),
        )
        step = next(walk)
        if step.action is None:
            deadlocks += 1
            continue

        samples.append(_make_sample(step.state, step.action, step.successor))
        name = step.action.get_action().get_name()
        per_action[name] = per_action.get(name, 0) + 1

        if quota_met():
            break

    if steps >= max_steps and not quota_met():
        missing = {
            a: min_per_action - per_action.get(a, 0)
            for a in action_names
            if per_action.get(a, 0) < min_per_action
        }
        logger.warning(
            "Random walk hit max_steps=%d before meeting its quota; missing per action: %s",
            max_steps,
            missing,
        )

    if deadlocks:
        logger.info(
            "Random walk restarted %d time(s) from the initial state.", deadlocks
        )
    logger.info(
        "Random walk: %d steps, %d samples, per action: %s",
        steps,
        len(samples),
        per_action,
    )
    return samples


def samples_to_tensors(samples, metadata, include_equality: bool = False):
    """Samples as adjacency tensors ``[N, R, O, O]`` with their action labels."""
    # Not at the top: validation workers import this module and must not import torch.
    import torch

    relation_info = tensor_relations(
        metadata["predicates"], include_equality=include_equality
    )
    relation_names = [r["name"] for r in relation_info]
    relation_arities = [r["arity"] for r in relation_info]
    pred_to_idx = {n: i for i, n in enumerate(relation_names)}

    n_samples = len(samples)
    n_rel = len(relation_names)
    n_obj = metadata["num_objects"]
    if n_rel == 0:
        raise ValueError("No predicates with arity 0, 1, or 2 in domain.")

    states = torch.zeros(n_samples, n_rel, n_obj, n_obj)
    next_states = torch.zeros(n_samples, n_rel, n_obj, n_obj)

    action_names = [a["name"] for a in metadata["actions"]]
    act_to_idx = {n: i for i, n in enumerate(action_names)}
    max_arity = max((a["arity"] for a in metadata["actions"]), default=0)
    action_indices = torch.zeros(n_samples, dtype=torch.long)
    action_args = torch.full((n_samples, max(max_arity, 1)), -1, dtype=torch.long)

    def fill(adj, groundings_by_pred, where):
        for pred_name, groundings in groundings_by_pred.items():
            if pred_name not in pred_to_idx:
                continue
            r = pred_to_idx[pred_name]
            for g in groundings:
                if len(g) == 2:
                    adj[r, g[0], g[1]] = 1.0
                elif len(g) == 1:
                    adj[r, g[0], g[0]] = 1.0
                else:
                    raise ValueError(f"Nullary predicate {pred_name} in {where}.")

    for i, sample in enumerate(samples):
        fill(states[i], sample["state"], "state")
        fill(next_states[i], sample["next_state"], "next_state")
        action_indices[i] = act_to_idx[sample["action"]]
        for j, obj_idx in enumerate(sample["action_args"]):
            action_args[i, j] = obj_idx

    applicable_actions = [
        {
            act_to_idx[name]: [tuple(g) for g in groundings]
            for name, groundings in s.get("applicable_actions", {}).items()
        }
        for s in samples
    ]

    return {
        "states": states,
        "next_states": next_states,
        "action_indices": action_indices,
        "action_args": action_args,
        "relation_names": relation_names,
        "relation_arities": relation_arities,
        "object_names": metadata["objects"],
        "object_name_to_idx": metadata["object_name_to_idx"],
        "action_names": action_names,
        "action_arities": [a["arity"] for a in metadata["actions"]],
        "applicable_actions": applicable_actions,
        "num_objects": n_obj,
        "num_relations": n_rel,
    }
