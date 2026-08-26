"""Sampling states and scoring a learned domain against the original one.

Validation is behavioural: the learned schemas are correct to the extent that
they generate the same successor states as the domain they were learned from.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from tqdm import tqdm

from dias.data.generate_samples import reachable_states
from dias.pddl.states import state_fluents, transfer_state

logger = logging.getLogger(__name__)

# Learned actions are merged into the original domain under this prefix,
# which is the only thing telling the two apart when scoring.
LEARNED_ACTION_PREFIX = "learned_"


def resolve_sample_seed(seed: int | None) -> int:
    """The seed to draw with, drawn itself when the caller named none.

    An *unfixed* draw and an *unrecorded* one are different things. The states
    a result is computed on stay a fresh sample per run -- nothing in this
    pipeline pins a draw to a constant -- but the seed that produced them is
    written into the result, so passing it back reproduces those exact states.
    """
    return random.randrange(2**32) if seed is None else seed


def sample_reachable_states(
    problem,
    num_samples: int,
    sample_population: int,
    seed: int,
) -> list:
    """Draw `num_samples` distinct states from a BFS over `problem`.

    `seed` decides which ones, and callers get it from `resolve_sample_seed`,
    so there is no path here that draws without a seed something recorded. The
    draw comes from a generator of its own, never the global one, so it neither
    depends on nor disturbs anything else's random state.
    """
    if num_samples > sample_population:
        logger.warning(
            "num_samples (%d) exceeds the population, clamping to %d.",
            num_samples,
            sample_population,
        )
        num_samples = sample_population

    population = reachable_states(problem, sample_population)
    if num_samples > len(population):
        logger.warning(
            "num_samples (%d) exceeds the reachable states (%d), clamping.",
            num_samples,
            len(population),
        )
        num_samples = len(population)

    logger.info(
        "Sampling %d of the %d reachable states found, seeded %d.",
        num_samples,
        len(population),
        seed,
    )
    return random.Random(seed).sample(population, num_samples)


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
