"""Cells of a schema tensor that no lifted action could ever mean.

Effects and preconditions are laid out over *slot pairs* for every relation,
which is one cell too many for a unary relation: it says something about a
single slot, so an off-diagonal cell has no reading. Those cells are pinned to
the ignore channel rather than left free to fit noise.

The pinning happens on the way out of the module that owns the tensor, not on
the parameter itself, so no effect source can forget it.
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
    """Pin every masked cell of `logits` to the ignore channel.

    The masked values take no gradient, so what the parameter holds there never
    matters -- including the ``-inf`` that checkpoints written before the mask
    moved out of the parameter still carry.
    """
    blocked = logits.new_full((logits.shape[-1],), -torch.inf)
    blocked[IGNORE] = 0.0
    return torch.where(mask, blocked, logits)
