"""The precondition parameters `P_pre` and their relaxed conjunction `p_pre` (paper, Sec. 4.3)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .masking import forbid, unary_offdiagonal_mask


class Preconditions(nn.Module):
    """Per-action precondition logits over ignore / positive / negative, arity-masked."""

    def __init__(
        self, num_actions: int, relation_arities: Sequence[int], num_slots: int
    ):
        super().__init__()
        self.logits = nn.Parameter(
            0.1
            * torch.randn(num_actions, len(relation_arities), num_slots, num_slots, 3)
        )
        self.register_buffer(
            "mask",
            unary_offdiagonal_mask(relation_arities, num_slots),
            persistent=False,
        )

    def forward(self, y_action: torch.Tensor) -> torch.Tensor:
        """Masked logits for one batch of actions, ``(B, R, M, M, 3)``."""
        return forbid(self.logits[y_action], self.mask)

    def action_logits(self) -> torch.Tensor:
        """Masked logits for every action, ``(A, R, M, M, 3)``."""
        return forbid(self.logits, self.mask)


def soft_conjunction(values: torch.Tensor, dim: int, tau: float = 1.0):
    """The product of `values` along `dim` to the power ``1 / (tau * n + 1 - tau)`` (paper, Eq. 8): the geometric mean at tau = 1, the product at tau = 0."""
    logs = torch.log(values.clamp(min=1e-8))
    return torch.exp(logs.sum(dim=dim) / (1 + tau * (values.shape[dim] - 1)))
