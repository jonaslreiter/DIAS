"""The MLP-effects validator's worker side: a state's expected successors and the groundings of its learned actions. A successor is identified by the XOR of a random 128-bit key per cell it flips. Imports no torch, since workers run under an address-space limit."""

from __future__ import annotations

from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np
import pymimir

from dias.pddl.states import rebuild_state

_KEY_SEED = 0x5EED_D1A5

_WORKER: dict = {}


def successor_keys(num_relations: int, num_objects: int) -> np.ndarray:
    """One key per cell as two int64 words, ``(R, O, O, 2)``, the same in every process."""
    return np.random.default_rng(_KEY_SEED).integers(
        np.iinfo(np.int64).min,
        np.iinfo(np.int64).max,
        size=(num_relations, num_objects, num_objects, 2),
        dtype=np.int64,
        endpoint=True,
    )


def signature_from_words(high: int, low: int) -> int:
    mask = (1 << 64) - 1
    return ((int(high) & mask) << 64) | (int(low) & mask)


def flip_signature(keys: np.ndarray, flips: np.ndarray) -> int:
    """The signature of a successor differing from its state at `flips`, ``(R, O, O)``."""
    words = np.bitwise_xor.reduce(keys[flips], axis=0) if flips.any() else (0, 0)
    return signature_from_words(words[0], words[1])


def init_worker(config: dict) -> None:
    _WORKER.clear()
    _WORKER.update(
        config=config,
        original_domain=pymimir.Domain(Path(config["original_domain_path"])),
        learned_domain=pymimir.Domain(Path(config["learned_domain_path"])),
        keys=successor_keys(config["num_relations"], config["num_objects"]),
        learned_actions={},
    )


def _adjacency(state, config: dict) -> np.ndarray:
    num_objects = config["num_objects"]
    adjacency = np.zeros(
        (config["num_relations"], num_objects, num_objects), dtype=np.uint8
    )
    for atom in state.get_atoms(ignore_static=False, ignore_derived=True):
        r = config["relation_name_to_idx"].get(atom.get_predicate().get_name())
        if r is None:
            continue
        terms = atom.get_terms()
        arity = config["relation_arities"][r]
        if arity == 2:
            adjacency[r, terms[0].get_index(), terms[1].get_index()] = 1
        elif arity == 1:
            adjacency[r, terms[0].get_index(), terms[0].get_index()] = 1
        elif arity == 0:
            adjacency[r, 0, 0] = 1
    return adjacency


def ground_state(payload) -> tuple:
    """A state's adjacency, its expected successor signatures by action, and per learned grounding its action index and slot binding ``(M,)``, -1 for an inactive slot. Grounded on problems of its own, since pymimir keeps every action a problem grounds."""
    config = _WORKER["config"]
    channels = np.ones(config["num_relations"], dtype=bool)
    if config["eq_channel_idx"] is not None:
        channels[config["eq_channel_idx"]] = False

    original_problem = pymimir.Problem(
        _WORKER["original_domain"], Path(config["original_problem_path"])
    )
    learned_problem = pymimir.Problem(
        _WORKER["learned_domain"], Path(config["learned_problem_path"])
    )
    original_state = rebuild_state(payload, original_problem)
    learned_state = rebuild_state(payload, learned_problem)
    adjacency = _adjacency(original_state, config)

    expected: dict[str, set[int]] = defaultdict(set)
    for ground_action in original_state.generate_applicable_actions():
        successor = _adjacency(ground_action.apply(original_state), config)
        flips = (successor != adjacency) & channels[:, None, None]
        expected[ground_action.get_action().get_name()].add(
            flip_signature(_WORKER["keys"], flips)
        )

    actions = array("h")
    slot_objects = array("h")
    learned = _WORKER["learned_actions"]
    prefix = config["learned_action_prefix"]
    for ground_action in learned_state.generate_applicable_actions():
        action = ground_action.get_action()
        key = action.get_index()
        if key not in learned:
            name = action.get_name()
            learned[key] = (
                config["action_name_to_idx"][name[len(prefix) :]]
                if name.startswith(prefix)
                else None
            )
        action_idx = learned[key]
        if action_idx is None:
            continue
        row = [-1] * config["num_slots"]
        for slot, obj in zip(
            config["active_slot_indices"][action_idx], ground_action.get_objects()
        ):
            row[slot] = obj.get_index()
        actions.append(action_idx)
        slot_objects.extend(row)

    return (
        adjacency,
        dict(expected),
        np.frombuffer(actions, dtype=np.int16).copy(),
        np.frombuffer(slot_objects, dtype=np.int16)
        .reshape(-1, config["num_slots"])
        .copy(),
    )
