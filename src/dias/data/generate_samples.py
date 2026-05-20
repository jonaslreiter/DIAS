#!/usr/bin/env python3
"""
generate_samples.py — Generate (state, action, next_state) samples from PDDL domains.

Uses pymimir to parse PDDL domains and problem instances, then generates
traces / rollouts using different exploration strategies.

Output formats
--------------
  json    (default) JSON file convenient for inspection and later conversion.
  tensor  PyTorch .pt file with adjacency-matrix tensors of shape
          ``[num_samples, num_relations, num_objects, num_objects]``.
          Only predicates with arity 0, 1, or 2 are included:
            arity 2 → full adjacency matrix,
            arity 1 → diagonal entries,
            arity 0 → all entries 1 when true, 0 when false.

Generation methods
------------------
  random_walk   Linear random walk from the initial state.
                Produces exactly `n` transition samples.

  bfs           BFS exploration from the initial state.
                Explores up to `n` unique states, then records every
                (state, applicable_action, successor) triple.

  rl_trace      RL-style trace: at each of the `n` steps all applicable
                actions and their successors are recorded, but only one
                randomly chosen action is followed.  Each sample carries
                a `followed` flag.

Usage examples
--------------
  # Random walk of 100 steps on blocks world (JSON)
  python generate_samples.py -d pddl_files/blocks_3/blocks_world.pddl \\
                             -p pddl_files/blocks_3/blocks_world_5.pddl \\
                             -m random_walk -n 100 -o samples.json

  # Same, but output adjacency-matrix tensors (.pt)
  python generate_samples.py -d pddl_files/blocks_3/blocks_world.pddl \\
                             -p pddl_files/blocks_3/blocks_world_5.pddl \\
                             -m random_walk -n 100 -f tensor -o samples.pt

  # BFS exploration of 50 states on the ferry domain
  python generate_samples.py -d pddl_files/ferry/ferry.pddl \\
                             -p pddl_files/ferry/ferry-4-4.pddl \\
                             -m bfs -n 50 -o samples.json

  # RL-style trace on gripper (tensor format)
  python generate_samples.py -d pddl_files/gripper/gripper.pddl \\
                             -p pddl_files/gripper/gripper_6_2.pddl \\
                             -m rl_trace -n 100 -f tensor -o gripper.pt

  # List all available domain / instance files
  python generate_samples.py --list-domains
"""

import argparse
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

PDDL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pddl_files")


# ─────────────────────────────────────────────────────────────────────
#  State parsing helpers
# ─────────────────────────────────────────────────────────────────────


def state_to_dict(state, include_statics=False):
    """Convert a pymimir State to a JSON-serialisable dictionary.

    Returns
    -------
    dict
        ``{predicate_name: [[obj_idx, ...], ...], ...}``
    """
    result = {}
    for atom in state.get_atoms():
        pred = atom.get_predicate()
        if not include_statics and pred.is_static():
            continue
        name = pred.get_name()
        objs = [t.get_index() for t in atom.get_terms()]
        result.setdefault(name, []).append(objs)
    return result


def get_metadata(domain, problem):
    """Extract domain / problem metadata for the output JSON."""
    objects = {}
    object_name_to_idx = {}
    for obj in problem.get_objects():
        idx = obj.get_index()
        name = obj.get_name()
        objects[str(idx)] = name
        object_name_to_idx[name] = idx

    predicates = []
    for pred in domain.get_predicates():
        predicates.append(
            {
                "name": pred.get_name(),
                "arity": pred.get_arity(),
                "static": pred.is_static(),
            }
        )
    predicates.sort(key=lambda p: p["name"])

    actions = []
    for act in domain.get_actions():
        actions.append(
            {
                "name": act.get_name(),
                "arity": act.get_arity(),
            }
        )

    return {
        "domain_name": domain.get_name(),
        "objects": objects,
        "object_name_to_idx": object_name_to_idx,
        "num_objects": len(objects),
        "predicates": predicates,
        "actions": actions,
    }


def filter_predicates_for_tensor_relations(
    predicates: list[dict],
    *,
    include_statics: bool = False,
    default_statics: list[str] | None = None,
) -> list[dict]:
    """Same predicate subset as :func:`samples_to_tensors` (tensor relation order).

    Each entry is a dict with keys ``name``, ``arity``, ``static`` (as in
    :func:`get_metadata` ``predicates``).
    """
    if default_statics is None:
        default_statics = ["number", "object"]
    relation_info: list[dict] = []
    for pred in predicates:
        if pred["arity"] > 2:
            continue
        if not include_statics and pred["static"]:
            continue
        if include_statics and pred["static"] and pred["name"] in default_statics:
            continue
        relation_info.append(pred)
    return relation_info


def get_tensor_schema_names_from_domain_file(
    domain_path: str | Path,
    *,
    include_statics: bool = False,
    default_statics: list[str] | None = None,
) -> tuple[list[str], list[str], list[int]]:
    """Relation and action names in the same order as training tensors.

    Pass the **same** ``include_statics`` / ``default_statics`` you used when
    building ``samples.pt`` / :class:`PddlDataModule`.

    Returns
    -------
    relation_names
        One name per tensor relation channel (arity ≤ 2, static filter applied).
    action_names
        One name per model action index (domain action declaration order).
    relation_arities
        Arity of each tensor channel (``0``, ``1``, or ``2``), same order as
        ``relation_names``. Matches encoding in :func:`samples_to_tensors`
        (unary uses diagonal ``[i, i]`` only; nullary uses a single broadcast cell).
    """
    import pymimir

    domain = pymimir.Domain(Path(domain_path))
    predicates = [
        {
            "name": pred.get_name(),
            "arity": pred.get_arity(),
            "static": pred.is_static(),
        }
        for pred in domain.get_predicates()
    ]
    relation_info = filter_predicates_for_tensor_relations(
        predicates,
        include_statics=include_statics,
        default_statics=default_statics,
    )
    relation_names = [r["name"] for r in relation_info]
    relation_arities = [int(r["arity"]) for r in relation_info]
    action_names = [act.get_name() for act in domain.get_actions()]
    return relation_names, action_names, relation_arities


def _shorten_mimir_parameter_name(mimir_name: str) -> str:
    """``?bm_0_0`` → ``?bm`` for readable PDDL; falls back if that would collide."""
    s = mimir_name.strip()
    if not s.startswith("?"):
        s = "?" + s
    rest = s[1:]
    base = rest.split("_")[0]
    return f"?{base}"


def get_domain_action_parameters_ordered(
    domain_path: str | Path,
) -> dict[str, list[str]]:
    """Action name → formal parameters in **domain declaration order** (readable names).

    pymimir exposes internal names like ``?bm_0_0``; we shorten to ``?bm`` when the
    shortened stem is unique for that action.
    """
    import pymimir

    domain = pymimir.Domain(Path(domain_path))
    out: dict[str, list[str]] = {}
    for act in domain.get_actions():
        raw = [v.get_name() for v in act.get_parameters()]
        short = [_shorten_mimir_parameter_name(n) for n in raw]
        if len(short) != len(set(short)):
            short = raw
        out[act.get_name()] = short
    return out


# ─────────────────────────────────────────────────────────────────────
#  Generation methods
# ─────────────────────────────────────────────────────────────────────
def generate_random_walk(
    problem,
    min_num_samples: int = 0,
    min_per_label: int = 0,
    max_steps: int = 1_000_000,
    only_unseen_states: bool = True,
    include_statics: bool = False,
):
    """Balanced random walk.

    Walks uniformly at random, collecting transition samples, and stops once
    every action schema that has been witnessed has at least ``min_per_label``
    samples — and every action in the domain that is reachable from the
    initial state has appeared at least once (so actions that never fire are
    simply skipped).

    Parameters
    ----------
    min_per_label
        Required number of samples per action label (the stopping criterion).
    only_unseen_states
        If ``True``, only transitions whose source state has not been recorded
        before are kept (the walk itself still proceeds through all states).
    max_steps
        Safety cap on the walk length.

    Behaviour
    ---------
    - On dead-ends the walk restarts from the initial state (like
      :func:`generate_random_walk`).
    - If ``max_steps`` is reached before the per-label quota is met, returns
      what has been collected along with a warning.
    """
    initial = problem.get_initial_state()
    action_names = [act.get_name() for act in problem._domain.get_actions()]
    current = initial
    samples = []
    per_label_count: dict[str, int] = {}
    seen_state_ids: set[int] = set()
    deadlocks = 0

    if not list(initial.generate_applicable_actions()):
        raise ValueError(
            "the initial state of the problem instance has no applicable actions."
        )

    steps = 0
    while steps < max_steps:
        steps += 1
        app_actions = list(current.generate_applicable_actions())

        if not app_actions:
            deadlocks += 1
            current = initial
            continue

        applicable = {}
        for ga_i in app_actions:
            name = ga_i.get_action().get_name()
            args = tuple(o.get_index() for o in ga_i.get_objects())
            applicable.setdefault(name, []).append(args)

        ga = random.choice(app_actions)
        successor = ga.apply(current)
        action_name = ga.get_action().get_name()

        sid = current.get_index()
        keep = (not only_unseen_states) or (sid not in seen_state_ids)
        if keep:
            seen_state_ids.add(sid)
            samples.append(
                {
                    "state": state_to_dict(current, include_statics),
                    "action": action_name,
                    "action_args": [o.get_index() for o in ga.get_objects()],
                    "next_state": state_to_dict(successor, include_statics),
                    "applicable_actions": applicable,
                    "_pymimir_state": current,
                    "_pymimir_next_state": successor,
                    "_pymimir_action": ga
                }
            )
            per_label_count[action_name] = per_label_count.get(action_name, 0) + 1

        current = successor

        if len(samples) >= min_num_samples and all(
            per_label_count.get(a, 0) >= min_per_label for a in action_names
        ):
            break

    if steps >= max_steps and not (
        len(samples) >= min_num_samples
        and all(per_label_count.get(a, 0) >= min_per_label for a in action_names)
    ):
        missing = {
            a: min_per_label - per_label_count.get(a, 0)
            for a in action_names
            if per_label_count.get(a, 0) < min_per_label
        }
        print(
            f"[warn] balanced random walk hit max_steps={max_steps} before "
            f"meeting the quota. Missing per action: {missing}",
            file=sys.stderr,
        )

    if deadlocks:
        print(
            f"[info] Balanced random walk restarted {deadlocks} time(s) from initial state after hitting dead-ends.",
            file=sys.stderr,
        )

    print(
        f"[info] Balanced random walk: {steps} steps, {len(samples)} samples kept, per-label counts: {per_label_count}",
        file=sys.stderr,
    )

    unique_states = {s["_pymimir_state"].get_index() for s in samples} | {
        s["_pymimir_next_state"].get_index() for s in samples
    }
    print(
        f"[info] Balanced random walk: {len(unique_states)} unique states across all samples.",
        file=sys.stderr,
    )

    return samples


def generate_bfs(problem, num_states, include_statics=False):
    """BFS exploration from the initial state.

    Explores up to *num_states* unique states, then for every explored
    state records one sample per applicable action (successor states
    outside the explored set are still included).
    """
    initial = problem.get_initial_state()

    # BFS to collect states
    visited = {initial.get_index(): initial}
    queue = deque([initial])

    while queue and len(visited) < num_states:
        current = queue.popleft()
        for ga in current.generate_applicable_actions():
            successor = ga.apply(current)
            sid = successor.get_index()
            if sid not in visited:
                visited[sid] = successor
                queue.append(successor)
                if len(visited) >= num_states:
                    break

    print(f"  BFS explored {len(visited)} unique states.", file=sys.stderr)

    samples = []
    for state in visited.values():
        s_dict = state_to_dict(state, include_statics)
        all_ga = list(state.generate_applicable_actions())
        applicable = {}
        for ga_i in all_ga:
            name = ga_i.get_action().get_name()
            args = tuple(o.get_index() for o in ga_i.get_objects())
            applicable.setdefault(name, []).append(args)
        for ga in all_ga:
            successor = ga.apply(state)
            samples.append(
                {
                    "state": s_dict,
                    "action": ga.get_action().get_name(),
                    "action_args": [o.get_index() for o in ga.get_objects()],
                    "next_state": state_to_dict(successor, include_statics),
                    "applicable_actions": applicable,
                    "_pymimir_state": state,
                }
            )

    return samples


# ─────────────────────────────────────────────────────────────────────
#  Tensor conversion
# ─────────────────────────────────────────────────────────────────────


def samples_to_tensors(
    samples,
    metadata,
    include_statics=False,
    default_statics: list[str] = ["number", "object"],
):
    """Convert samples to adjacency-matrix tensors.

    Only predicates with arity 0, 1, or 2 are kept.

    * arity 2 → ``tensor[sample, rel, i, j] = 1`` when ``pred(obj_i, obj_j)``
    * arity 1 → ``tensor[sample, rel, i, i] = 1`` when ``pred(obj_i)`` (diagonal)
      (broadcast the global boolean to every node pair)

    Returns
    -------
    dict
        Ready for ``torch.save()``.  Keys include *states* and *next_states*
        tensors of shape ``[N, R, O, O]``, action encoding tensors, and
        string metadata (relation / object / action names).
    """
    import torch

    # --- filter predicates to arity 0, 1, or 2 (shared with domain-file schema export) ---
    relation_info = filter_predicates_for_tensor_relations(
        metadata["predicates"],
        include_statics=include_statics,
        default_statics=default_statics,
    )

    relation_names = [r["name"] for r in relation_info]
    relation_arities = [r["arity"] for r in relation_info]
    pred_to_idx = {name: i for i, name in enumerate(relation_names)}

    num_samples = len(samples)
    num_relations = len(relation_names)
    num_objects = metadata["num_objects"]

    if num_relations == 0:
        raise ValueError(
            "No predicates with arity 0, 1, or 2 found in this domain — "
            "tensor format requires at least one such predicate."
        )

    states = torch.zeros(num_samples, num_relations, num_objects, num_objects)
    next_states = torch.zeros(num_samples, num_relations, num_objects, num_objects)

    # --- action encoding ---
    action_names = [a["name"] for a in metadata["actions"]]
    act_to_idx = {name: i for i, name in enumerate(action_names)}
    max_action_arity = (
        max(a["arity"] for a in metadata["actions"]) if metadata["actions"] else 0
    )

    action_indices = torch.zeros(num_samples, dtype=torch.long)
    action_args = torch.full(
        (num_samples, max(max_action_arity, 1)), -1, dtype=torch.long
    )

    # --- optional followed flag (rl_trace) ---
    has_followed = any("followed" in s for s in samples)
    followed = torch.zeros(num_samples, dtype=torch.bool) if has_followed else None

    for s_idx, sample in enumerate(samples):
        # state → adjacency matrices
        for pred_name, groundings in sample["state"].items():
            if pred_name not in pred_to_idx:
                continue
            r_idx = pred_to_idx[pred_name]
            for g in groundings:
                if len(g) == 2:
                    states[s_idx, r_idx, g[0], g[1]] = 1.0
                elif len(g) == 1:
                    states[s_idx, r_idx, g[0], g[0]] = 1.0
                elif len(g) == 0:
                    raise ValueError(f"Nullary predicate {pred_name} found in state.")

        # next_state → adjacency matrices
        for pred_name, groundings in sample["next_state"].items():
            if pred_name not in pred_to_idx:
                continue
            r_idx = pred_to_idx[pred_name]
            for g in groundings:
                if len(g) == 2:
                    next_states[s_idx, r_idx, g[0], g[1]] = 1.0
                elif len(g) == 1:
                    next_states[s_idx, r_idx, g[0], g[0]] = 1.0
                elif len(g) == 0:
                    raise ValueError(
                        f"Nullary predicate {pred_name} found in next state."
                    )

        # action
        action_indices[s_idx] = act_to_idx[sample["action"]]
        for a_idx, obj_idx in enumerate(sample["action_args"]):
            action_args[s_idx, a_idx] = obj_idx

        if has_followed and sample.get("followed"):
            followed[s_idx] = True

    applicable_actions = []
    for sample in samples:
        per_sample = {}
        for act_name, groundings in sample.get("applicable_actions", {}).items():
            per_sample[act_to_idx[act_name]] = [tuple(g) for g in groundings]
        applicable_actions.append(per_sample)

    result = {
        "states": states,  # [N, R, O, O]
        "next_states": next_states,  # [N, R, O, O]
        "action_indices": action_indices,  # [N]
        "action_args": action_args,  # [N, max_arity]
        "relation_names": relation_names,  # list[str]
        "relation_arities": relation_arities,  # list[int]
        "object_names": metadata["objects"],  # {idx_str: name}
        "object_name_to_idx": metadata["object_name_to_idx"],  # {name: idx}
        "action_names": action_names,  # list[str]
        "action_arities": [a["arity"] for a in metadata["actions"]],
        "applicable_actions": applicable_actions,  # list[{action_idx: [[args], ...]}]
        "num_objects": num_objects,
        "num_relations": num_relations,
    }

    if followed is not None:
        result["followed"] = followed  # [N]

    return result


# ─────────────────────────────────────────────────────────────────────
#  Domain listing
# ─────────────────────────────────────────────────────────────────────


def list_available_domains():
    """Scan pddl_files/ and print a tidy summary of domains / instances."""
    if not os.path.isdir(PDDL_DIR):
        print(f"PDDL directory not found: {PDDL_DIR}", file=sys.stderr)
        return

    print("Available PDDL domains and instances")
    print("=" * 50)

    for domain_dir in sorted(os.listdir(PDDL_DIR)):
        dir_path = os.path.join(PDDL_DIR, domain_dir)
        if not os.path.isdir(dir_path):
            continue

        domain_files = []
        instance_files = []

        for f in sorted(os.listdir(dir_path)):
            if not f.endswith(".pddl"):
                continue
            fpath = os.path.join(dir_path, f)
            try:
                with open(fpath, "r") as fh:
                    head = fh.read(500).lower().replace("\n", " ")
            except Exception:
                continue

            if "(define (domain" in head:
                domain_files.append(f)
            elif "(define (problem" in head:
                instance_files.append(f)

        if not domain_files and not instance_files:
            continue

        print(f"\n  {domain_dir}/")
        for df in domain_files:
            print(f"    [domain]   {df}")
        for pf in instance_files:
            print(f"    [instance] {pf}")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

METHODS = {
    "random_walk": generate_random_walk,
    "bfs": generate_bfs,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate (state, action, next_state) samples from PDDL domains.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Random walk (JSON)
  python generate_samples.py \\
      -d pddl_files/blocks_3/blocks_world.pddl \\
      -p pddl_files/blocks_3/blocks_world_5.pddl \\
      -m random_walk -n 100 -o samples.json

  # Random walk (adjacency-matrix tensors, .pt)
  python generate_samples.py \\
      -d pddl_files/blocks_3/blocks_world.pddl \\
      -p pddl_files/blocks_3/blocks_world_5.pddl \\
      -m random_walk -n 100 -f tensor -o samples.pt

  # BFS exploration (50 states)
  python generate_samples.py \\
      -d pddl_files/ferry/ferry.pddl \\
      -p pddl_files/ferry/ferry-4-4.pddl \\
      -m bfs -n 50 -o samples.json

  # RL-style trace (tensor format)
  python generate_samples.py \\
      -d pddl_files/gripper/gripper.pddl \\
      -p pddl_files/gripper/gripper_6_2.pddl \\
      -m rl_trace -n 100 -f tensor -o gripper.pt

  # List all available domain / instance files
  python generate_samples.py --list-domains
        """,
    )
    parser.add_argument(
        "-d",
        "--domain",
        type=str,
        help="Path to domain PDDL file",
    )
    parser.add_argument(
        "-p",
        "--problem",
        type=str,
        help="Path to problem / instance PDDL file",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=list(METHODS.keys()),
        help="Generation method: " + ", ".join(METHODS.keys()),
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        help="Number of steps (random_walk, rl_trace) or states to explore (bfs)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path (default: write to stdout for json format)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["json", "tensor"],
        default="json",
        help=(
            "Output format (default: json).  'tensor' saves a .pt file "
            "with adjacency-matrix tensors of shape "
            "[num_samples, num_relations, num_objects, num_objects].  "
            "Predicates with arity 0, 1, or 2 are included."
        ),
    )
    parser.add_argument(
        "--include-statics",
        action="store_true",
        help="Include static predicates in state representations",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List available domain / problem files and exit",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    #  --list-domains: standalone mode (no pymimir needed)
    # ------------------------------------------------------------------
    if args.list_domains:
        list_available_domains()
        return

    # ------------------------------------------------------------------
    #  Validate required arguments
    # ------------------------------------------------------------------
    if not all([args.domain, args.problem, args.method, args.num]):
        parser.error(
            "Arguments -d/--domain, -p/--problem, -m/--method, -n/--num are all required for sample generation."
        )
    if not os.path.isfile(args.domain):
        parser.error(f"Domain file not found: {args.domain}")
    if not os.path.isfile(args.problem):
        parser.error(f"Problem file not found: {args.problem}")

    # ------------------------------------------------------------------
    #  Initialise pymimir (import here so --list-domains works w/o it)
    # ------------------------------------------------------------------
    import pymimir  # noqa: E402

    random.seed(args.seed)

    print(f"Loading domain:  {args.domain}", file=sys.stderr)
    print(f"Loading problem: {args.problem}", file=sys.stderr)

    domain = pymimir.Domain(Path(args.domain))
    problem = pymimir.Problem(domain, Path(args.problem))

    metadata = get_metadata(domain, problem)
    print(
        f"Domain: {metadata['domain_name']},  "
        f"Objects: {metadata['num_objects']},  "
        f"Actions: {[a['name'] for a in metadata['actions']]}",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    #  Generate samples
    # ------------------------------------------------------------------
    print(
        f"Generating samples  method={args.method}  n={args.num}  seed={args.seed} ...",
        file=sys.stderr,
    )
    generator = METHODS[args.method]
    samples = generator(problem, args.num, args.include_statics)
    print(f"Generated {len(samples)} samples.", file=sys.stderr)

    # ------------------------------------------------------------------
    #  Build & write output
    # ------------------------------------------------------------------
    if args.format == "tensor":
        import torch

        tensor_data = samples_to_tensors(
            samples, metadata, include_statics=args.include_statics
        )
        tensor_data["generation_config"] = {
            "domain_file": args.domain,
            "problem_file": args.problem,
            "method": args.method,
            "num": args.num,
            "seed": args.seed,
            "include_statics": args.include_statics,
        }

        out_path = args.output or "samples.pt"
        if not out_path.endswith(".pt"):
            out_path += ".pt"

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        torch.save(tensor_data, out_path)

        S = tensor_data["states"]
        print(
            f"Tensor saved to {out_path}\n"
            f"  states       shape: {list(S.shape)}  "
            f"(N={S.shape[0]}, R={S.shape[1]}, O={S.shape[2]})\n"
            f"  next_states  shape: {list(S.shape)}\n"
            f"  relations:   {tensor_data['relation_names']}\n"
            f"  rel arities: {tensor_data['relation_arities']}",
            file=sys.stderr,
        )

    else:  # json
        json_samples = [
            {k: v for k, v in s.items() if k != "_pymimir_state"} for s in samples
        ]
        output = {
            "metadata": metadata,
            "generation_config": {
                "domain_file": args.domain,
                "problem_file": args.problem,
                "method": args.method,
                "num": args.num,
                "seed": args.seed,
                "include_statics": args.include_statics,
            },
            "num_samples": len(json_samples),
            "samples": json_samples,
        }

        json_str = json.dumps(output, indent=2)

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                f.write(json_str)
            print(f"Samples written to {args.output}", file=sys.stderr)
        else:
            print(json_str)


if __name__ == "__main__":
    main()
