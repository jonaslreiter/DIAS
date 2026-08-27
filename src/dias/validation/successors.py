"""Sampling states and scoring a learned domain against the original one.

Validation is behavioural: the learned schemas are correct to the extent that they generate the same successor states as the domain they were learned from.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Iterable, Mapping

from tqdm import tqdm

from dias.data.generate_samples import reachable_states
from dias.pddl.states import state_fluents, transfer_state

logger = logging.getLogger(__name__)

# Learned actions are merged into the original domain under this prefix, which is the only thing telling the two apart when scoring.
LEARNED_ACTION_PREFIX = "learned_"

# One state's successors: what the original domain reaches, and what the learned one reached, each keyed by action name.
SuccessorSets = tuple[Mapping[str, set], Mapping[str, set]]


def resolve_sample_seed(seed: int | None) -> int:
    """The seed to draw with, drawn itself when the caller named none.

    An *unfixed* draw and an *unrecorded* one are different things. The states a result is computed on stay a fresh sample per run (nothing in this pipeline pins a draw to a constant), but the seed that produced them is written into the result, so passing it back reproduces those exact states.
    """
    return random.randrange(2**32) if seed is None else seed


def sample_reachable_states(
    problem,
    num_samples: int,
    sample_population: int,
    seed: int,
) -> list:
    """Draw `num_samples` distinct states from a BFS over `problem`.

    `seed` decides which ones, and callers get it from `resolve_sample_seed`, so there is no path here that draws without a seed something recorded. The draw comes from a generator of its own, never the global one, so it neither depends on nor disturbs anything else's random state.
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

    The two domains only overlap on the predicates both declare, so that is the view the transfer can be verified on.
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
                f"State transfer mismatch on the predicates both domains declare:\n  shared predicates: {sorted(shared)}\n  only in source: {source_atoms - target_atoms}\n  only in target: {target_atoms - source_atoms}"
            )
        transferred.append(target_state)
    return transferred


def score_successors(per_state: Iterable[SuccessorSets]) -> dict:
    """Score produced successors against expected ones, state by state.

    Each element is one state's ``(expected, produced)`` pair, both keyed by action name. A successor is whatever identifies one: a pymimir state index where the learned domain is symbolic, a binarized adjacency where a network produced it. So this is the one scorer for every validator, and the only thing they have to agree on is that equal states compare equal.

    **Counts are attributed to actions, and that is the measure.** A successor the learned domain reaches under the wrong operator is a false positive there and a false negative under the operator that should have reached it, not a hit. The alternative (pooling the union across actions first, and asking only which states are reachable in one step) scores the learned domain as a transition model rather than as a set of action schemas, and this repository learns schemas. The two agree almost always (3330 of 3348 archived results) and never disagreed about whether a run was sound or complete, but where they part the attributed one is the claim being made: one archived ablation reproduces 12% of transitions under the union view and 0% under this one, because it had permuted the operators.

    ``matches`` and ``accuracy`` keep the state-level view: a state counts as a match when the two successor *sets* agree exactly, which is the "same search tree" question, and holding at every state is the same thing as the union view scoring perfectly.

    Micro-averaged: one pooled tp/fp/fn over every state and action, so a state with many successors weighs more than one with few.
    """
    states = 0
    matches = 0
    action_tp: dict[str, int] = defaultdict(int)
    action_fp: dict[str, int] = defaultdict(int)
    action_fn: dict[str, int] = defaultdict(int)

    for expected_by_action, produced_by_action in per_state:
        states += 1
        matches += set().union(*expected_by_action.values(), set()) == set().union(
            *produced_by_action.values(), set()
        )

        for name in set(expected_by_action) | set(produced_by_action):
            expected = expected_by_action.get(name, set())
            produced = produced_by_action.get(name, set())
            hits = len(expected & produced)
            action_tp[name] += hits
            action_fp[name] += len(produced) - hits
            action_fn[name] += len(expected) - hits

    tp_total = sum(action_tp.values())
    fp_total = sum(action_fp.values())
    fn_total = sum(action_fn.values())
    return {
        "matches": matches,
        "total": states,
        "accuracy": matches / states if states else 0.0,
        "precision": _ratio(tp_total, fp_total),
        "recall": _ratio(tp_total, fn_total),
        # Totals, so several problems can be micro-averaged downstream instead of averaging their precisions. They are the sums of `action_stats`.
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


def compare_successors(states: list, learned_action_prefix: str) -> dict:
    """Score a merged domain's learned actions against its original ones.

    Every state carries both sets of actions, so applying all of them and comparing the successors needs no planning. A successor is its pymimir state index; the scoring itself is `score_successors`.
    """
    prefix_length = len(learned_action_prefix)

    def sets_for(state) -> SuccessorSets:
        expected: dict[str, set] = defaultdict(set)
        produced: dict[str, set] = defaultdict(set)
        for action in state.generate_applicable_actions():
            successor = action.apply(state).get_index()
            name = action.get_action().get_name()
            if name.startswith(learned_action_prefix):
                produced[name[prefix_length:]].add(successor)
            else:
                expected[name].add(successor)
        return expected, produced

    return score_successors(
        sets_for(state)
        for state in tqdm(states, total=len(states), desc="Validating states")
    )


def _ratio(true_positives: int, errors: int) -> float:
    total = true_positives + errors
    return true_positives / total if total else 0.0
