"""Drawing test states and scoring the successors a learned domain generates against the original's (paper, App. C.3)."""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Iterable, Mapping

from tqdm import tqdm

from dias.data.generate_samples import random_walk
from dias.pddl.states import state_fluents, transfer_state

logger = logging.getLogger(__name__)

# The prefix of the learned actions in a domain holding both.
LEARNED_ACTION_PREFIX = "learned_"

# One state's successors, expected and produced, by action name.
SuccessorSets = tuple[Mapping[str, set], Mapping[str, set]]

MAX_WALKS = 100


def resolve_sample_seed(seed: int | None) -> int:
    """`seed`, or a freshly drawn one."""
    return random.randrange(2**32) if seed is None else seed


def sample_walk_states(
    problem,
    num_samples: int,
    walk_length: int,
    min_walks: int,
    seed: int,
) -> list:
    """`num_samples` distinct states of `problem`, an equal share for each action schema.

    Walks of `walk_length` steps from the initial state are repeated at least `min_walks` and at most `MAX_WALKS` times, until every action schema is applicable in its share of the distinct states visited. Each schema's share is drawn uniformly from the visited states in which it applies, and any shortfall from the remaining ones.
    """
    rng = random.Random(seed)
    action_names = sorted(a.get_name() for a in problem.get_domain().get_actions())
    shares = {
        name: num_samples // len(action_names) + (i < num_samples % len(action_names))
        for i, name in enumerate(action_names)
    }

    # State index -> (state, the action schemas applicable in it).
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


@dataclasses.dataclass(frozen=True)
class WalkSampling:
    """How validation draws a problem's test states: `sample_walk_states` with these settings."""

    num_samples: int = 500
    walk_length: int = 2000
    min_walks: int = 5

    def draw(self, problem, seed: int) -> list:
        return sample_walk_states(
            problem, self.num_samples, self.walk_length, self.min_walks, seed
        )

    def record(self) -> dict:
        return dataclasses.asdict(self)


def sampling_from_record(record: Mapping) -> WalkSampling:
    """The sampling a `WalkSampling.record()` or validation.yaml's `sampling` block describes, settings left out at their defaults."""
    settings = dict(record)
    if settings.pop("strategy", "walks") != "walks":
        raise ValueError(
            f"validation samples by walks only, not {record['strategy']!r}"
        )
    known = {field.name for field in dataclasses.fields(WalkSampling)}
    if set(settings) - known:
        raise ValueError(
            f"sampling takes {sorted(known)}, not {sorted(set(settings) - known)}"
        )
    return WalkSampling(**settings)


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
    """tp, fp and fn of the produced successors against the expected ones, per action and summed. A successor reached by the wrong action counts against both. ``matches`` counts the states whose successor sets agree."""
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
    """`score_successors` over several disjoint sets of states, from their results."""
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


def successor_sets(state, learned_action_prefix: str) -> SuccessorSets:
    """The successors of `state` in a domain holding the original and the learned actions, by action, original and learned apart."""
    prefix_length = len(learned_action_prefix)
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


def read_progress(path: Path, header: dict) -> dict[int, dict]:
    """The state scores recorded in `path` under `header`. A file with another header is started over."""
    lines = path.read_text().splitlines() if path.exists() else []
    scores: dict[int, dict] = {}
    if lines and lines[0] == json.dumps(header):
        for line in lines[1:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            scores[record["state"]] = record["score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [header] + [{"state": i, "score": s} for i, s in scores.items()]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return scores


def append_progress(path: Path, index: int, score: dict) -> None:
    """Record one state's score."""
    with path.open("a") as progress:
        progress.write(json.dumps({"state": index, "score": score}) + "\n")


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
