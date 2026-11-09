"""The off-diagonal cells of unary relations, which schema tensors fix to the ignore channel."""

from __future__ import annotations

from typing import Sequence

import torch

from .outputs import IGNORE


def unary_offdiagonal_mask(
    relation_arities: Sequence[int], num_slots: int
) -> torch.Tensor:
    """``(1, R, M, M, 1)``, true off the diagonal of unary relations."""
    is_unary = torch.tensor([arity == 1 for arity in relation_arities])
    off_diagonal = ~torch.eye(num_slots, dtype=torch.bool)
    return is_unary.view(1, -1, 1, 1, 1) & off_diagonal.view(
        1, 1, num_slots, num_slots, 1
    )


def forbid(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`logits` with every masked cell fixed to ``[0, -inf, -inf]``."""
    blocked = logits.new_full((logits.shape[-1],), -torch.inf)
    blocked[IGNORE] = 0.0
    return torch.where(mask, blocked, logits)
