"""Generate (state, action, next_state) samples from PDDL domains via pymimir."""

import random
import sys
from collections import deque


def state_to_dict(state, include_statics=False):
    """pymimir State -> {predicate_name: [[obj_idx, ...], ...]}."""
    result: dict[str, list[list[int]]] = {}
    for atom in state.get_atoms():
        pred = atom.get_predicate()
        if not include_statics and pred.is_static():
            continue
        result.setdefault(pred.get_name(), []).append(
            [t.get_index() for t in atom.get_terms()]
        )
    return result


def get_metadata(domain, problem):
    """Domain / problem metadata used downstream."""
    objects = {str(o.get_index()): o.get_name() for o in problem.get_objects()}
    object_name_to_idx = {o.get_name(): o.get_index() for o in problem.get_objects()}
    predicates = sorted(
        (
            {"name": p.get_name(), "arity": p.get_arity(), "static": p.is_static()}
            for p in domain.get_predicates()
        ),
        key=lambda p: p["name"],
    )
    actions = [
        {"name": a.get_name(), "arity": a.get_arity()} for a in domain.get_actions()
    ]
    return {
        "domain_name": domain.get_name(),
        "objects": objects,
        "object_name_to_idx": object_name_to_idx,
        "num_objects": len(objects),
        "predicates": predicates,
        "actions": actions,
    }


def filter_predicates_for_tensor_relations(
    predicates, *, include_statics=False, default_statics=None
):
    """Predicates kept as tensor relation channels (arity <= 2, static filter)."""
    default_statics = default_statics or ["number", "object"]
    keep = []
    for p in predicates:
        if p["arity"] > 2:
            continue
        if not include_statics and p["static"]:
            continue
        if include_statics and p["static"] and p["name"] in default_statics:
            continue
        keep.append(p)
    return keep


def _applicable_map(actions):
    """List of GroundedActions -> {action_name: [(arg_idx, ...), ...]}."""
    out: dict[str, list[tuple[int, ...]]] = {}
    for ga in actions:
        out.setdefault(ga.get_action().get_name(), []).append(
            tuple(o.get_index() for o in ga.get_objects())
        )
    return out


def _make_sample(state, ga, successor, include_statics):
    return {
        "state": state_to_dict(state, include_statics),
        "action": ga.get_action().get_name(),
        "action_args": [o.get_index() for o in ga.get_objects()],
        "next_state": state_to_dict(successor, include_statics),
        "applicable_actions": _applicable_map(state.generate_applicable_actions()),
        "_pymimir_state": state,
    }


def generate_random_walk(
    problem,
    min_num_samples: int = 0,
    min_per_label: int = 0,
    max_steps: int = 1_000_000,
    only_unseen_states: bool = True,
    include_statics: bool = False,
):
    """Random walk until ``min_num_samples`` and per-label quota are met."""
    initial = problem.get_initial_state()
    if not list(initial.generate_applicable_actions()):
        raise ValueError("initial state has no applicable actions.")

    action_names = [a.get_name() for a in problem._domain.get_actions()]
    current = initial
    samples: list[dict] = []
    per_label: dict[str, int] = {}
    seen: set[int] = set()
    deadlocks = 0

    def quota_met() -> bool:
        return len(samples) >= min_num_samples and all(
            per_label.get(a, 0) >= min_per_label for a in action_names
        )

    steps = 0
    while steps < max_steps:
        steps += 1
        app = list(current.generate_applicable_actions())
        if not app:
            deadlocks += 1
            current = initial
            continue

        ga = random.choice(app)
        successor = ga.apply(current)
        sid = current.get_index()
        if (not only_unseen_states) or sid not in seen:
            seen.add(sid)
            samples.append(_make_sample(current, ga, successor, include_statics))
            name = ga.get_action().get_name()
            per_label[name] = per_label.get(name, 0) + 1

        current = successor
        if quota_met():
            break

    if steps >= max_steps and not quota_met():
        missing = {
            a: min_per_label - per_label.get(a, 0)
            for a in action_names
            if per_label.get(a, 0) < min_per_label
        }
        print(
            f"[warn] random walk hit max_steps={max_steps} before meeting quota. "
            f"Missing per action: {missing}",
            file=sys.stderr,
        )

    if deadlocks:
        print(
            f"[info] Random walk restarted {deadlocks} time(s) from initial state.",
            file=sys.stderr,
        )
    print(
        f"[info] Random walk: {steps} steps, {len(samples)} samples, "
        f"per-label: {per_label}",
        file=sys.stderr,
    )
    return samples


def generate_bfs(problem, num_states: int, include_statics: bool = False):
    """BFS up to ``num_states`` states; emit one sample per applicable action."""
    initial = problem.get_initial_state()
    visited = {initial.get_index(): initial}
    queue = deque([initial])
    while queue and len(visited) < num_states:
        current = queue.popleft()
        for ga in current.generate_applicable_actions():
            succ = ga.apply(current)
            if succ.get_index() not in visited:
                visited[succ.get_index()] = succ
                queue.append(succ)
                if len(visited) >= num_states:
                    break
    print(f"  BFS explored {len(visited)} unique states.", file=sys.stderr)

    samples = []
    for state in visited.values():
        for ga in state.generate_applicable_actions():
            samples.append(_make_sample(state, ga, ga.apply(state), include_statics))
    return samples


def samples_to_tensors(
    samples,
    metadata,
    include_statics: bool = False,
    default_statics: list[str] = ["number", "object"],
):
    """Convert samples to adjacency-matrix tensors of shape [N, R, O, O]."""
    import torch

    relation_info = filter_predicates_for_tensor_relations(
        metadata["predicates"],
        include_statics=include_statics,
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
