"""Sampling states and scoring a learned domain against the original one, behaviourally: the learned schemas are correct to the extent that they generate the same successor states."""

from __future__ import annotations

import dataclasses
import logging
import random
from collections import Counter, defaultdict
from itertools import islice
from typing import ClassVar, Iterable, Mapping

from tqdm import tqdm

from dias.data.generate_samples import random_walk, reachable_states
from dias.pddl.states import state_fluents, transfer_state

logger = logging.getLogger(__name__)

# Learned actions are merged into the original domain under this prefix, which is what tells the two apart when scoring.
LEARNED_ACTION_PREFIX = "learned_"

# One state's successors, expected and produced, each keyed by action name.
SuccessorSets = tuple[Mapping[str, set], Mapping[str, set]]

# A bound on the walks `sample_walk_states` adds while some action schema is short of its share, for a problem in which a schema never applies. At validation.yaml's settings no domain in pddl/ needs more than 20 (Hanoi, and Sokoban test-1).
MAX_WALKS = 100


def resolve_sample_seed(seed: int | None) -> int:
    """The seed to draw with, drawn itself when the caller named none. It is recorded in the result, so passing it back reproduces those exact states."""
    return random.randrange(2**32) if seed is None else seed


def sample_walk_states(
    problem,
    num_samples: int,
    walk_length: int,
    min_walks: int,
    seed: int,
) -> list:
    """Draw `num_samples` distinct states of `problem` from independent random walks, an equal share for each action schema.

    Each walk is the one training samples from (`generate_samples.random_walk`), `walk_length` steps from the initial state. At least `min_walks` are taken, and more until every action schema is applicable in at least its share of the distinct states visited, as training walks until each action name has its quota; a walk caught where an action can no longer apply (a box pushed into a corner) is followed by a fresh one. Each schema's share is then drawn uniformly from the visited states in which it applies, so recall is scored on every schema, and the sample is filled up uniformly from the rest if a share runs short. The generator is seeded with `seed` and is nobody else's.
    """
    rng = random.Random(seed)
    action_names = sorted(a.get_name() for a in problem.get_domain().get_actions())
    shares = {
        name: num_samples // len(action_names) + (i < num_samples % len(action_names))
        for i, name in enumerate(action_names)
    }

    # State index -> (state, the action schemas applicable in it), in the order first visited.
    visited: dict[int, tuple] = {}
    applicable_in: Counter = Counter()

    def quota_met() -> bool:
        return len(visited) >= num_samples and all(
            applicable_in[name] >= share for name, share in shares.items()
        )

    walks = 0
    while walks < min_walks or not quota_met():
        if walks == MAX_WALKS:
            logger.warning(
                "%d walks left the quota unmet; visited %d states, applicable in per action schema: %s.",
                walks,
                len(visited),
                {name: applicable_in[name] for name in action_names},
            )
            break
        for step in islice(random_walk(problem, rng), walk_length):
            index = step.state.get_index()
            if index not in visited:
                names = frozenset(a.get_action().get_name() for a in step.applicable)
                visited[index] = (step.state, names)
                applicable_in.update(names)
        walks += 1

    if len(visited) < num_samples:
        logger.warning(
            "num_samples (%d) exceeds the states visited (%d), clamping.",
            num_samples,
            len(visited),
        )
    chosen: dict[int, object] = {}
    for name in action_names:
        candidates = [
            index
            for index, (_, names) in visited.items()
            if name in names and index not in chosen
        ]
        for index in rng.sample(candidates, min(shares[name], len(candidates))):
            chosen[index] = visited[index][0]
    rest = [index for index in visited if index not in chosen]
    for index in rng.sample(rest, min(num_samples - len(chosen), len(rest))):
        chosen[index] = visited[index][0]

    logger.info(
        "Drew %d states from %d walks of %d steps (%d distinct states visited), seeded %d.",
        len(chosen),
        walks,
        walk_length,
        len(visited),
        seed,
    )
    return list(chosen.values())


def sample_reachable_states(
    problem,
    num_samples: int,
    sample_population: int,
    seed: int,
) -> list:
    """Draw `num_samples` distinct states from a BFS over `problem`, from a generator of its own rather than the global one."""
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


class Sampling:
    """How a validator draws its states from a held-out problem: one of the dataclasses below, named in validation.yaml's `sampling` block by its `strategy`."""

    strategy: ClassVar[str]

    def draw(self, problem, seed: int) -> list:
        raise NotImplementedError

    def record(self) -> dict:
        """The settings as results record them and validation.yaml states them; `sampling_from_record` reads them back."""
        return {"strategy": self.strategy, **dataclasses.asdict(self)}


@dataclasses.dataclass(frozen=True)
class WalkSampling(Sampling):
    """`sample_walk_states`: training's random walk on the held-out problem, an equal share of the states for each action schema."""

    strategy: ClassVar[str] = "walks"
    num_samples: int = 500
    walk_length: int = 2000
    min_walks: int = 5

    def draw(self, problem, seed: int) -> list:
        return sample_walk_states(
            problem, self.num_samples, self.walk_length, self.min_walks, seed
        )


@dataclasses.dataclass(frozen=True)
class BfsSampling(Sampling):
    """`sample_reachable_states`, the paper's draw: `num_samples` of the first `sample_population` states a breadth-first search reaches. It stays within a few actions of the initial state on the larger problems, and never makes satellite's take_image applicable, nor spanner's tighten_nut on test-2 and test-3."""

    strategy: ClassVar[str] = "bfs"
    num_samples: int = 250
    sample_population: int = 500

    def draw(self, problem, seed: int) -> list:
        return sample_reachable_states(
            problem, self.num_samples, self.sample_population, seed
        )


SAMPLINGS = {sampling.strategy: sampling for sampling in (WalkSampling, BfsSampling)}

# What a result recording no sampling was drawn with: before the walks there was only the breadth-first draw, and validation.yaml never changed its settings.
UNRECORDED_SAMPLING = BfsSampling()


def sampling_from_record(record: Mapping) -> Sampling:
    """The sampling a `Sampling.record()`, or validation.yaml's `sampling` block, describes; settings it leaves out take their defaults."""
    settings = dict(record)
    strategy = settings.pop("strategy", None)
    if strategy not in SAMPLINGS:
        raise ValueError(
            f"sampling strategy {strategy!r} is none of {sorted(SAMPLINGS)}"
        )
    sampling = SAMPLINGS[strategy]
    known = {field.name for field in dataclasses.fields(sampling)}
    if set(settings) - known:
        raise ValueError(
            f"{strategy} sampling takes {sorted(known)}, not {sorted(set(settings) - known)}"
        )
    return sampling(**settings)


def transfer_states(states: list, target_problem, source_domain, target_domain) -> list:
    """Re-host each state on `target_problem`, checking that the predicates both domains declare agree."""
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

    The one scorer for every validator: a successor is a pymimir state index or a signature of the cells it flips, and all they must agree on is that equal successors compare equal.

    Counts are attributed to actions, so a successor reached under the wrong operator is a false positive there and a false negative under the right one. ``matches`` keeps the state-level view: the two successor sets agreed exactly. Micro-averaged over every state and action.
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

    return _summary(states, matches, action_tp, action_fp, action_fn)


def combine_scores(scores: Iterable[dict]) -> dict:
    """`score_successors` over the union of disjoint state sets, from each set's own result: every count is a sum over states."""
    states = 0
    matches = 0
    action_tp: dict[str, int] = defaultdict(int)
    action_fp: dict[str, int] = defaultdict(int)
    action_fn: dict[str, int] = defaultdict(int)
    for score in scores:
        states += score["total"]
        matches += score["matches"]
        for name, stats in score["action_stats"].items():
            action_tp[name] += stats["tp"]
            action_fp[name] += stats["fp"]
            action_fn[name] += stats["fn"]

    return _summary(states, matches, action_tp, action_fp, action_fn)


def compare_successors(states: list, learned_action_prefix: str) -> dict:
    """Score a merged domain's learned actions against its original ones. Every state carries both sets, so this needs no planning: apply all of them and compare state indices."""
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


def _summary(
    states: int, matches: int, action_tp: dict, action_fp: dict, action_fn: dict
) -> dict:
    tp_total = sum(action_tp.values())
    fp_total = sum(action_fp.values())
    fn_total = sum(action_fn.values())
    return {
        "matches": matches,
        "total": states,
        "accuracy": matches / states if states else 0.0,
        "precision": _ratio(tp_total, fp_total),
        "recall": _ratio(tp_total, fn_total),
        # Totals (the sums of `action_stats`), so problems can be micro-averaged downstream.
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
