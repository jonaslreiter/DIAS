"""Cells of a schema tensor that no lifted action could mean.

Effects and preconditions are laid out over slot pairs per relation, which is one cell too many for a unary relation: an off-diagonal cell has no reading, so it is pinned to the ignore channel. Pinned on the way out of the module that owns the tensor, so no effect source can forget it.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .outputs import IGNORE


def unary_offdiagonal_mask(
    arity_relations: Sequence[int], num_slots: int
) -> torch.Tensor:
    """``(1, R, k, k, 1)``, true where a relation cannot relate the slot pair."""
    is_unary = torch.tensor([arity == 1 for arity in arity_relations])
    off_diagonal = ~torch.eye(num_slots, dtype=torch.bool)
    return is_unary.view(1, -1, 1, 1, 1) & off_diagonal.view(
        1, 1, num_slots, num_slots, 1
    )


def forbid(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pin every masked cell of `logits` to the ignore channel. Those cells take no gradient, so whatever the parameter holds there never matters."""
    blocked = logits.new_full((logits.shape[-1],), -torch.inf)
    blocked[IGNORE] = 0.0
    return torch.where(mask, blocked, logits)
