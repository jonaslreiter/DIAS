"""Sampling states and scoring a learned domain against the original one.

Validation is behavioural: the learned schemas are correct to the extent that
they generate the same successor states as the domain they were learned from.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from tqdm import tqdm

from dias.data.generate_samples import generate_bfs
from dias.pddl.states import state_fluents, transfer_state

logger = logging.getLogger(__name__)

# Learned actions are merged into the original domain under this prefix,
# which is the only thing telling the two apart when scoring.
LEARNED_ACTION_PREFIX = "learned_"


def sample_reachable_states(problem, num_samples: int, sample_population: int) -> list:
    """Draw `num_samples` distinct states from a BFS over `problem`."""
    if num_samples > sample_population:
        logger.warning(
            "num_samples (%d) exceeds the population, clamping to %d.",
            num_samples,
            sample_population,
        )
        num_samples = sample_population

    seen: set[int] = set()
    population: list = []
    for sample in generate_bfs(problem, sample_population):
        state = sample["_pymimir_state"]
        if state.get_index() not in seen:
            seen.add(state.get_index())
            population.append(state)
        if len(population) >= sample_population:
            break

    if num_samples > len(population):
        logger.warning(
            "num_samples (%d) exceeds the reachable states (%d), clamping.",
            num_samples,
            len(population),
        )
        num_samples = len(population)

    return random.sample(population, num_samples)


def transfer_states(states: list, target_problem, source_domain, target_domain) -> list:
    """Re-host each state on `target_problem`, checking the shared fluents agree.

    The two domains only overlap on the predicates both declare, so that is the
    view the transfer can be verified on.
    """
    shared = {p.get_name() for p in source_domain.get_predicates()} & {
        p.get_name() for p in target_domain.get_predicates()
    }

    transferred = []
    for state in states:
        target_state = transfer_state(state, target_problem)
        source_atoms = state_fluents(state, shared)
        target_atoms = state_fluents(target_state, shared)
        if source_atoms != target_atoms:
            raise AssertionError(
                "State transfer mismatch on the predicates both domains declare:\n"
                f"  shared predicates: {sorted(shared)}\n"
                f"  only in source: {source_atoms - target_atoms}\n"
                f"  only in target: {target_atoms - source_atoms}"
            )
        transferred.append(target_state)
    return transferred


def compare_successors(states: list, learned_action_prefix: str) -> dict:
    """Score the learned actions of a merged domain against the original ones.

    Every state carries both sets of actions, so applying all of them and
    comparing the successors needs no planning. Precision and recall are
    micro-averaged: one pooled tp/fp/fn over all states, so a state with many
    successors weighs more than one with few.
    """
    matches = 0
    tp_total = fp_total = fn_total = 0
    action_tp: dict[str, int] = defaultdict(int)
    action_fp: dict[str, int] = defaultdict(int)
    action_fn: dict[str, int] = defaultdict(int)
    prefix_length = len(learned_action_prefix)

    for state in tqdm(states, total=len(states), desc="Validating states"):
        original_by_action = defaultdict(set)
        learned_by_action = defaultdict(set)
        for action in state.generate_applicable_actions():
            successor = action.apply(state).get_index()
            name = action.get_action().get_name()
            if name.startswith(learned_action_prefix):
                learned_by_action[name[prefix_length:]].add(successor)
            else:
                original_by_action[name].add(successor)

        original = set().union(*original_by_action.values())
        learned = set().union(*learned_by_action.values())

        true_positives = len(original & learned)
        tp_total += true_positives
        fp_total += len(learned) - true_positives
        fn_total += len(original) - true_positives
        matches += original == learned

        for name in set(original_by_action) | set(learned_by_action):
            expected = original_by_action.get(name, set())
            produced = learned_by_action.get(name, set())
            hits = len(expected & produced)
            action_tp[name] += hits
            action_fp[name] += len(produced) - hits
            action_fn[name] += len(expected) - hits

    return {
        "matches": matches,
        "total": len(states),
        "accuracy": matches / len(states),
        "precision": _ratio(tp_total, fp_total),
        "recall": _ratio(tp_total, fn_total),
        # Pooled counts, so several problems can be micro-averaged downstream
        # instead of averaging their precisions.
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "action_stats": {
            name: {
                "tp": action_tp[name],
                "fp": action_fp[name],
                "fn": action_fn[name],
                "conflicts": action_fp[name] + action_fn[name],
            }
            for name in set(action_tp) | set(action_fp) | set(action_fn)
        },
    }


def _ratio(true_positives: int, errors: int) -> float:
    total = true_positives + errors
    return true_positives / total if total else 0.0
