"""Corrupting observed states, for the noise ablations.

Both models flip cells of the adjacency tensor; they differ in how the budget
is chosen. `ratio` flips each eligible cell independently with a fixed
probability, `expected_count` spends a Poisson budget across relations.
"""

from __future__ import annotations

import torch

from dias.adjacency import valid_cells


def add_adj_noise(
    adj: torch.Tensor,
    arities: list,
    remove_ratio: float,
    add_ratio: float | list[float] | torch.Tensor,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Return a noisy copy of a multi-relational adjacency tensor ``[R, O, O]``.

    For each relation slice the valid region depends on arity:
      - arity 2: the full ``[O, O]`` matrix.
      - arity 1: only diagonal entries ``(i, i)``.
      - other arities: skipped.

    Within the valid region, each existing 1-entry is independently flipped to
    0 with probability ``remove_ratio`` and each existing 0-entry is flipped to
    1 with probability ``add_ratio``. ``add_ratio`` may be a scalar or a
    per-relation sequence of length ``R``.
    """
    R, O, O2 = adj.shape
    assert O == O2, f"adj must be [R, O, O], got {adj.shape}"
    assert len(arities) == R, f"arities length {len(arities)} != R={R}"

    if isinstance(add_ratio, (int, float)):
        add_ratio_per_rel = [float(add_ratio)] * R
    else:
        add_ratio_per_rel = [float(x) for x in add_ratio]
        assert (
            len(add_ratio_per_rel) == R
        ), f"add_ratio length {len(add_ratio_per_rel)} != R={R}"

    noisy = adj.clone()
    for r, ar in enumerate(arities):
        valid_idx = valid_cells(ar, O)
        if valid_idx is None:
            continue

        flat = noisy[r].reshape(-1)
        vals = flat[valid_idx]
        ones_pos = valid_idx[vals == 1]
        zeros_pos = valid_idx[vals == 0]

        if remove_ratio > 0 and ones_pos.numel() > 0:
            mask = torch.rand(ones_pos.numel(), generator=generator) < remove_ratio
            flat[ones_pos[mask]] = 0
        if add_ratio_per_rel[r] > 0 and zeros_pos.numel() > 0:
            mask = (
                torch.rand(zeros_pos.numel(), generator=generator)
                < add_ratio_per_rel[r]
            )
            flat[zeros_pos[mask]] = 1

    return noisy


def add_adj_noise_expected_count(
    adj: torch.Tensor,
    arities: list[int],
    expected_remove_count: float,
    expected_add_count: float,
    generator: torch.Generator = None,
    relation_mask: list[bool] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a noisy copy using per-tensor Poisson flip budgets.

    For each relation, compute the eligible true and false atoms in the valid
    region. The balancing weight is ``min(num_true, num_false)``, so relations
    closest to a 50/50 true-false split receive the largest share of the noise
    budget. Removal and addition counts are sampled independently per relation
    from Poisson distributions whose rates are proportional to those weights,
    then clipped to the available eligible atoms and applied without replacement.
    ``relation_mask`` (length ``R``) disables noise on specific relations
    (e.g. statics) when an entry is ``False``.
    """
    R, O, O2 = adj.shape
    assert O == O2, f"adj must be [R, O, O], got {adj.shape}"
    assert len(arities) == R, f"arities length {len(arities)} != R={R}"

    if relation_mask is None:
        enabled_relations = [True] * R
    else:
        enabled_relations = [bool(x) for x in relation_mask]
        assert (
            len(enabled_relations) == R
        ), f"relation_mask length {len(enabled_relations)} != R={R}"

    noisy = adj.clone()
    ones_positions_per_rel: list[torch.Tensor] = []
    zeros_positions_per_rel: list[torch.Tensor] = []
    weights: list[float] = []

    for r, ar in enumerate(arities):
        if not enabled_relations[r] or ar not in (1, 2):
            ones_positions_per_rel.append(torch.empty(0, dtype=torch.long))
            zeros_positions_per_rel.append(torch.empty(0, dtype=torch.long))
            weights.append(0.0)
            continue

        valid_idx = valid_cells(ar, O)
        flat = noisy[r].reshape(-1)
        vals = flat[valid_idx]
        ones_pos = valid_idx[vals == 1]
        zeros_pos = valid_idx[vals == 0]
        ones_positions_per_rel.append(ones_pos)
        zeros_positions_per_rel.append(zeros_pos)
        weights.append(float(min(ones_pos.numel(), zeros_pos.numel())))

    total_weight = sum(weights)
    if total_weight <= 0.0:
        return noisy

    def _sample_poisson_count(rate: float) -> int:
        if rate <= 0.0:
            return 0
        return int(torch.poisson(torch.tensor([rate]), generator=generator).item())

    def _pick_positions(candidates: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0 or candidates.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        if count >= candidates.numel():
            return candidates
        perm = torch.randperm(candidates.numel(), generator=generator)
        return candidates[perm[:count]]

    for r in range(R):
        weight = weights[r]
        if weight <= 0.0:
            continue
        flat = noisy[r].reshape(-1)
        ones_pos = ones_positions_per_rel[r]
        zeros_pos = zeros_positions_per_rel[r]

        remove_count = min(
            _sample_poisson_count(expected_remove_count * weight / total_weight),
            ones_pos.numel(),
        )
        add_count = min(
            _sample_poisson_count(expected_add_count * weight / total_weight),
            zeros_pos.numel(),
        )
        remove_positions = _pick_positions(ones_pos, remove_count)
        add_positions = _pick_positions(zeros_pos, add_count)
        if remove_positions.numel() > 0:
            flat[remove_positions] = 0
        if add_positions.numel() > 0:
            flat[add_positions] = 1

    return noisy
