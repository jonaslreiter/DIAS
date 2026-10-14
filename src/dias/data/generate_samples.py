"""Generate (state, action, next_state) samples from PDDL domains via pymimir."""

import logging
import random
import time
from collections import deque
from typing import Any, NamedTuple

import torch

from dias.schema.domain import tensor_relations

logger = logging.getLogger(__name__)

# How often a long exploration says where it is. Anything shorter stays silent, so short runs are not narrated.
PROGRESS_INTERVAL_SECONDS = 10.0


class _Progress:
    """Report at most every ``PROGRESS_INTERVAL_SECONDS``, and only if asked twice."""

    def __init__(self, message: str):
        self.message = message
        self.due = time.monotonic() + PROGRESS_INTERVAL_SECONDS

    def report(self, *values) -> None:
        now = time.monotonic()
        if now < self.due:
            return
        self.due = now + PROGRESS_INTERVAL_SECONDS
        logger.info(self.message, *values)


def state_to_groundings(state, include_statics=False):
    """pymimir State -> {predicate_name: [[object_index, ...], ...]}."""
    result: dict[str, list[list[int]]] = {}
    for atom in state.get_atoms():
        pred = atom.get_predicate()
        if not include_statics and pred.is_static():
            continue
        result.setdefault(pred.get_name(), []).append(
            [t.get_index() for t in atom.get_terms()]
        )
    return result


def _groundings_by_action(actions):
    """List of ground actions -> {action_name: [(object_index, ...), ...]}."""
    groundings: dict[str, list[tuple[int, ...]]] = {}
    for ground_action in actions:
        groundings.setdefault(ground_action.get_action().get_name(), []).append(
            tuple(o.get_index() for o in ground_action.get_objects())
        )
    return groundings


def _require_injective(groundings_by_action: dict) -> None:
    """Refuse a state in which an action applies with one object in two parameters. The model binds injectively and validation requires distinct parameters, so such a domain is outside what DIAS learns; no domain in pddl/ has one."""
    for name, groundings in groundings_by_action.items():
        for objects in groundings:
            if len(set(objects)) < len(objects):
                raise ValueError(
                    f"{name}{objects} binds one object to two parameters, but DIAS assumes injective bindings. A domain whose actions apply that way needs (not (= ?a ?b)) preconditions to be learned here."
                )


def _make_sample(state, ground_action, successor, include_statics):
    # The executed action is one of the applicable ones, so checking those covers it.
    applicable = _groundings_by_action(state.generate_applicable_actions())
    _require_injective(applicable)
    return {
        "state": state_to_groundings(state, include_statics),
        "action": ground_action.get_action().get_name(),
        "action_args": [o.get_index() for o in ground_action.get_objects()],
        "next_state": state_to_groundings(successor, include_statics),
        "applicable_actions": applicable,
    }


class WalkStep(NamedTuple):
    """One step of `random_walk`: the state it was taken in, that state's applicable ground actions, and the action taken with its successor. At a dead end the last two are None and the walk goes back to the initial state."""

    state: Any
    applicable: list
    action: Any
    successor: Any


def random_walk(problem, rng=random):
    """The walk training and validation both sample from: from the initial state, a uniformly random applicable ground action per step, and back to the initial state at a dead end. It never ends; the caller takes as many steps as it needs.

    `rng` is anything with a `choice`; training draws from the global `random` that its seed sets, validation from a generator of its own.
    """
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
    only_unseen_states: bool = True,
    include_statics: bool = False,
):
    """Random walk until ``min_total_samples`` and the per-action quota are met."""
    initial = problem.get_initial_state()
    if not list(initial.generate_applicable_actions()):
        raise ValueError("initial state has no applicable actions.")

    action_names = [a.get_name() for a in problem._domain.get_actions()]
    walk = random_walk(problem)
    samples: list[dict] = []
    per_action: dict[str, int] = {}
    seen: set[int] = set()
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

        sid = step.state.get_index()
        if (not only_unseen_states) or sid not in seen:
            seen.add(sid)
            samples.append(
                _make_sample(step.state, step.action, step.successor, include_statics)
            )
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


def reachable_states(problem, limit: int) -> list:
    """Up to `limit` distinct states reachable from the initial one, breadth first."""
    initial = problem.get_initial_state()
    visited = {initial.get_index(): initial}
    queue = deque([initial])
    progress = _Progress("Reachable states: %d of at most %d, %d still to expand.")
    while queue and len(visited) < limit:
        progress.report(len(visited), limit, len(queue))
        current = queue.popleft()
        for action in current.generate_applicable_actions():
            successor = action.apply(current)
            if successor.get_index() in visited:
                continue
            visited[successor.get_index()] = successor
            queue.append(successor)
            if len(visited) >= limit:
                break

    logger.info("BFS reached %d distinct states.", len(visited))
    return list(visited.values())


def generate_bfs(problem, num_states: int, include_statics: bool = False):
    """One sample per applicable action of each state a BFS reaches."""
    return [
        _make_sample(state, action, action.apply(state), include_statics)
        for state in reachable_states(problem, num_states)
        for action in state.generate_applicable_actions()
    ]


def samples_to_tensors(
    samples,
    metadata,
    include_statics: bool = False,
    default_statics: list[str] = ["number", "object"],
    include_equality: bool = False,
):
    """Convert samples to adjacency-matrix tensors of shape [N, R, O, O]."""
    relation_info = tensor_relations(
        metadata["predicates"],
        include_statics=include_statics,
        include_equality=include_equality,
        default_statics=default_statics,
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
