"""Scoring a checkpoint whose effects are predicted from the state by an MLP. There is no lifted effect tensor to write into PDDL, so only the preconditions become a learned domain, and the successors themselves come from the model."""

from __future__ import annotations

import json
import multiprocessing as _mp
import re
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import islice
from pathlib import Path

import numpy as np
import pymimir
import torch
from tqdm import tqdm

from dias.pddl import text as pddl_text
from dias.pddl.untype import (
    extract_domain_header,
    strip_problem_types,
    untype_action,
    with_equality_facts,
)
from dias.schema.action import ActionSchema, LiftedAtom

from .artifacts import PddlWorkspace
from .checkpoint import learned_pddl, load_run
from .grounding import ground_state, init_worker, signature_from_words, successor_keys
from .successors import (
    LEARNED_ACTION_PREFIX,
    combine_scores,
    resolve_sample_seed,
    sample_reachable_states,
    score_successors,
    transfer_states,
)

UNREACHABLE_GUARD = "dummy_guard"


def _unreachable_action(predicates: list[tuple[str, int]]) -> ActionSchema:
    """An action that never applies, with an effect on every predicate but `equal`: with every learned effect empty, pymimir would otherwise take them all for static."""
    return ActionSchema(
        name="dummy_action_unreachable",
        preconditions=[LiftedAtom(UNREACHABLE_GUARD, 0, [], True)],
        effects=[
            LiftedAtom(name, arity, list(range(arity)), True)
            for name, arity in predicates
            if name != pddl_text.EQUALITY_PREDICATE
        ],
    )


def _xor_reduce(words: torch.Tensor) -> torch.Tensor:
    """XOR along dimension 1 of ``(B, M, 2)`` words, pairwise."""
    width = 1 << (words.shape[1] - 1).bit_length()
    words = torch.nn.functional.pad(words, (0, 0, 0, width - words.shape[1]))
    while words.shape[1] > 1:
        words = words[:, 0::2] ^ words[:, 1::2]
    return words[:, 0]


def _read_progress(path: Path, header: dict) -> dict[int, dict]:
    """The state scores an interrupted run with this `header` left in `path`, rewritten to hold just those."""
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


class MlpEffectsValidator:
    def __init__(
        self,
        model_id,
        logs_path,
        problem_path,
        threshold=None,
        force_injective=False,
        checkpoint="final",
        batch_size: int = 65536,
        num_workers: int = 1,
        states_per_worker: int = 4,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.states_per_worker = states_per_worker

        run = load_run(model_id, logs_path, checkpoint)
        self.checkpoint_path = str(run.checkpoint_path)
        self.schema = run.schema
        self.model = run.model

        if self.model.schema_logits().effects is not None:
            raise ValueError(
                "MlpEffectsValidator expects a model trained with state_dependent_effects=True. A model with lifted effects is scored by DiasValidator."
            )

        self.learned_domain_text, self.threshold = learned_pddl(run, threshold)
        learned_domain_text = pddl_text.equality_as_predicate(self.learned_domain_text)
        if force_injective:
            learned_domain_text = pddl_text.force_distinct_parameters(
                learned_domain_text
            )

        original_domain_path = run.domain_path
        original_domain_text = Path(original_domain_path).read_text()

        relation_names = list(self.schema.relation_names)
        relation_arities = list(self.schema.relation_arities)
        self.relation_names = relation_names
        self.relation_arities = relation_arities
        self.eq_channel_idx = (
            relation_names.index("=") if "=" in relation_names else None
        )
        self.action_names = list(self.schema.action_names)
        self.num_slots = int(self.model.num_slots)

        # A learned action's parameters are its active slots, named ?x{slot}; grounding reads bindings back through them.
        blocks = {
            re.match(r"\(\s*:action\s+(\S+)", block).group(1): block
            for block in pddl_text.extract_action_blocks(learned_domain_text)
        }
        self.active_slot_indices = [
            [
                int(variable[2:])
                for variable in pddl_text.parse_typed_parameters(
                    pddl_text.find_balanced_expr(
                        blocks[name], blocks[name].find(":parameters")
                    )[0]
                )[0]
            ]
            for name in self.action_names
        ]
        actions = [
            untype_action(blocks[name], name_prefix=LEARNED_ACTION_PREFIX)
            for name in self.action_names
        ]

        predicates = [
            (
                (
                    pddl_text.EQUALITY_PREDICATE
                    if pddl_text.is_equality_predicate(name, arity)
                    else name
                ),
                arity,
            )
            for name, arity in zip(relation_names, relation_arities)
        ]
        if (pddl_text.EQUALITY_PREDICATE, 2) not in predicates:
            predicates.append((pddl_text.EQUALITY_PREDICATE, 2))
        predicates.append((UNREACHABLE_GUARD, 0))
        header = extract_domain_header(
            pddl_text.equality_as_predicate(original_domain_text),
            predicate_names=[name for name, _ in predicates],
            predicate_arities=[arity for _, arity in predicates],
        )

        self.workspace = PddlWorkspace()
        self.learned_domain_filename = self.workspace.write(
            "learned-domain.pddl",
            "\n\n".join([header, *actions, str(_unreachable_action(predicates))])
            + "\n)\n",
        )
        self.learned_problem_filename = self.workspace.write(
            "learned-problem.pddl",
            strip_problem_types(
                with_equality_facts(Path(problem_path).read_text()),
                original_domain_text,
            ),
        )

        self.learned_domain = pymimir.Domain(Path(self.learned_domain_filename))
        self.learned_problem = pymimir.Problem(
            self.learned_domain, Path(self.learned_problem_filename)
        )

        self.original_domain_path = str(original_domain_path)
        self.original_problem_path = str(problem_path)
        self.original_domain = pymimir.Domain(Path(original_domain_path))
        self.original_problem = pymimir.Problem(
            self.original_domain, Path(problem_path)
        )

        self.num_objects = len(self.original_problem.get_objects())
        self.model.retarget(self.num_objects)
        self.model.eval()

        # Adjacencies index objects as the training tensors did, and workers read learned bindings by the same indices.
        object_indices = {
            o.get_name(): o.get_index() for o in self.original_problem.get_objects()
        }
        assert sorted(object_indices.values()) == list(range(self.num_objects))
        assert all(
            object_indices[o.get_name()] == o.get_index()
            for o in self.learned_problem.get_objects()
        )

        self.relation_name_to_idx = {n: i for i, n in enumerate(relation_names)}
        self.action_name_to_idx = {n: i for i, n in enumerate(self.action_names)}

        self.state_pairs: list[tuple] | None = None

    def setup(
        self,
        num_samples: int = 250,
        sample_population: int = 500,
        seed: int | None = None,
    ):
        self.sample_seed = resolve_sample_seed(seed)
        original_states = sample_reachable_states(
            self.original_problem, num_samples, sample_population, self.sample_seed
        )
        learned_states = transfer_states(
            original_states,
            self.learned_problem,
            self.original_domain,
            self.learned_domain,
        )
        self.state_pairs = list(zip(original_states, learned_states))

    def validate(self, progress_path: str | Path | None = None) -> dict:
        """Score every sampled state: ground it in a worker, predict its successors on the GPU, compare.

        With `progress_path`, each state's score is appended as it finishes, and a later run for the same checkpoint, problem and sample seed skips those states.
        """
        if self.state_pairs is None:
            raise RuntimeError("\033[91mCall setup() before validate().\033[0m")

        # One serializable atom payload per state pair; statics come from :init in the worker.
        payloads = [
            [
                (
                    atom.get_predicate().get_name(),
                    tuple(t.get_name() for t in atom.get_terms()),
                )
                for atom in orig_state.get_atoms(
                    ignore_static=False, ignore_derived=True
                )
            ]
            for orig_state, _ in self.state_pairs
        ]
        config = {
            "original_domain_path": self.original_domain_path,
            "original_problem_path": self.original_problem_path,
            "learned_domain_path": self.learned_domain_filename,
            "learned_problem_path": self.learned_problem_filename,
            "relation_arities": list(self.relation_arities),
            "relation_name_to_idx": dict(self.relation_name_to_idx),
            "action_name_to_idx": dict(self.action_name_to_idx),
            "active_slot_indices": self.active_slot_indices,
            "num_relations": len(self.relation_names),
            "num_objects": self.num_objects,
            "num_slots": self.num_slots,
            "eq_channel_idx": self.eq_channel_idx,
            "learned_action_prefix": LEARNED_ACTION_PREFIX,
        }

        scores: dict[int, dict] = {}
        if progress_path is not None:
            progress_path = Path(progress_path)
            header = {
                "checkpoint": self.checkpoint_path,
                "checkpoint_mtime": Path(self.checkpoint_path).stat().st_mtime,
                "problem": str(Path(self.original_problem_path).resolve()),
                "sample_seed": self.sample_seed,
            }
            scores = _read_progress(progress_path, header)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        keys = torch.from_numpy(
            successor_keys(len(self.relation_names), self.num_objects)
        ).to(device)
        channels = torch.ones(len(self.relation_names), dtype=torch.bool, device=device)
        if self.eq_channel_idx is not None:
            channels[self.eq_channel_idx] = False

        todo = [i for i in range(len(payloads)) if i not in scores]
        with tqdm(total=len(payloads), initial=len(scores), desc="states") as bar:
            for index, (adjacency, expected, actions, slot_objects) in self._ground(
                todo, payloads, config
            ):
                produced = self._predict(
                    adjacency, actions, slot_objects, keys, channels
                )
                scores[index] = score_successors([(expected, produced)])
                if progress_path is not None:
                    with progress_path.open("a") as progress:
                        progress.write(
                            json.dumps({"state": index, "score": scores[index]}) + "\n"
                        )
                bar.update()

        return combine_scores(scores[i] for i in range(len(payloads)))

    def _ground(self, todo, payloads, config):
        """Yield ``(index, grounding)`` for the states in `todo`, grounding at most `num_workers` at once."""
        if self.num_workers <= 1:
            init_worker(config)
            for index in todo:
                yield index, ground_state(payloads[index])
            return
        options = {
            "max_workers": self.num_workers,
            "mp_context": _mp.get_context("spawn"),
            "initializer": init_worker,
            "initargs": (config,),
        }
        if sys.version_info >= (3, 11):
            options["max_tasks_per_child"] = self.states_per_worker
        queue = iter(todo)
        with ProcessPoolExecutor(**options) as pool:
            running = {
                pool.submit(ground_state, payloads[index]): index
                for index in islice(queue, self.num_workers)
            }
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    yield running.pop(future), future.result()
                    for index in islice(queue, 1):
                        running[pool.submit(ground_state, payloads[index])] = index

    def _predict(self, adjacency, actions, slot_objects, keys, channels) -> dict:
        """The successor signatures the model predicts for one state's groundings, keyed by action name."""
        state = torch.from_numpy(adjacency).to(keys.device, dtype=torch.float32)
        produced: dict[str, set[int]] = defaultdict(set)
        with torch.no_grad():
            for start in range(0, len(actions), self.batch_size):
                batch = slice(start, start + self.batch_size)
                action_idx = torch.from_numpy(actions[batch].astype(np.int64)).to(
                    keys.device
                )
                words = self._successor_words(
                    state, action_idx, slot_objects[batch], keys, channels
                )
                for action in torch.unique(action_idx).tolist():
                    rows = torch.unique(words[action_idx == action], dim=0)
                    produced[self.action_names[action]].update(
                        signature_from_words(high, low) for high, low in rows.tolist()
                    )
        return produced

    def _successor_words(self, state, actions, slot_objects, keys, channels):
        """Signature words ``(B, 2)`` for a batch of groundings of `state`.

        `apply_schema` runs on the ``K x K`` grid of bound objects instead of the whole instance: the projected effect is zero outside those cells, so it is the same computation. A slot repeating an earlier slot's object shares its row and column, so both accumulate in one cell as on the full grid.
        """
        objects = torch.from_numpy(slot_objects.astype(np.int64)).to(state.device)
        batch, num_slots = objects.shape
        active = objects >= 0
        same = (
            (objects[:, :, None] == objects[:, None, :])
            & active[:, :, None]
            & active[:, None, :]
        )
        position = same.to(torch.uint8).argmax(dim=2)
        occupied = active & (position == torch.arange(num_slots, device=state.device))
        cells = occupied[:, :, None] & occupied[:, None, :]

        selection = torch.zeros(batch, num_slots, num_slots, device=state.device)
        selection.scatter_(2, position[:, :, None], active[:, :, None].float())
        index = objects.clamp(min=0)
        grid = state[:, index[:, :, None], index[:, None, :]].permute(1, 0, 2, 3)
        grid = grid * cells[:, None]

        successor = self.model.apply_schema(grid, selection, actions)
        flips = (
            ((successor > 0.5) != (grid > 0.5))
            & cells[:, None]
            & channels[:, None, None]
        )
        flipped_keys = keys[:, index[:, :, None], index[:, None, :]].permute(
            1, 0, 2, 3, 4
        )
        flipped_keys = torch.where(flips[..., None], flipped_keys, 0)
        return _xor_reduce(flipped_keys.reshape(batch, -1, 2))
