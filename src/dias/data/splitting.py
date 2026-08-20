"""Dividing the sampled transitions into a train and a validation split.

The split is stratified by action: each action that has at least two samples
gets one reserved in each split before the rest is dealt out at the requested
ratio, so no action can be missing from a split just by being rare.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def cap_per_label(samples: list[dict], cap: int) -> list[dict]:
    """Keep at most `cap` samples of each action, in the order they were drawn."""
    kept: list[dict] = []
    kept_count: dict[str, int] = {}
    dropped: dict[str, int] = {}
    for sample in samples:
        action = sample["action"]
        if kept_count.get(action, 0) < cap:
            kept.append(sample)
            kept_count[action] = kept_count.get(action, 0) + 1
        else:
            dropped[action] = dropped.get(action, 0) + 1
    if dropped:
        logger.info("Capped each action at %d; dropped %s", cap, dropped)
    return kept


def stratified_split(
    action_indices: torch.Tensor,
    train_weight: int,
    valid_weight: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split the sample indices `train_weight` : `valid_weight`, by action."""
    total = len(action_indices)
    if total == 0:
        return [], []
    target = int(round(total * train_weight / (train_weight + valid_weight)))
    # Both splits must be non-empty, whatever the ratio asks for.
    target = max(1, min(total - 1, target))

    generator = torch.Generator().manual_seed(seed)
    by_action: dict[int, list[int]] = {}
    for index, action in enumerate(action_indices.tolist()):
        by_action.setdefault(action, []).append(index)

    train, valid, pool, singletons = [], [], [], []
    for indices in by_action.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[position] for position in order]
        if len(shuffled) >= 2:
            train.append(shuffled[0])
            valid.append(shuffled[1])
            pool.extend(shuffled[2:])
        else:
            # An action with a single sample can only be learned from, not scored.
            singletons.extend(shuffled)
    train.extend(singletons)

    extra = min(max(0, target - len(train)), len(pool))
    order = torch.randperm(len(pool), generator=generator).tolist()
    train.extend(pool[position] for position in order[:extra])
    valid.extend(pool[position] for position in order[extra:])

    logger.info(
        "Split %d samples into train=%d, valid=%d at ratio %d:%d, "
        "reserving one per action per split where there are at least two.",
        total,
        len(train),
        len(valid),
        train_weight,
        valid_weight,
    )
    return train, valid


def check_every_action_survives(
    action_indices: torch.Tensor,
    action_names: list[str],
    splits: dict[str, list[int]],
) -> None:
    """Refuse a split that lost an action the pool had.

    Stratified batching draws one batch per action, so an action missing from
    the validation split would be scored on nothing at all.
    """
    sampled = set(action_indices.tolist())
    for name, indices in splits.items():
        if not indices:
            continue
        missing = sampled - set(action_indices[indices].tolist())
        if missing:
            raise ValueError(
                f"Stratified '{name}' split is missing action(s) "
                f"{[action_names[action] for action in sorted(missing)]} that "
                f"appear in the full pool. Increase min_per_label_* / num_* or "
                f"relax the train:valid ratio."
            )
