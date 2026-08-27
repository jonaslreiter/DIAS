"""What an action requires of a state, and whether that holds."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .masking import forbid, unary_offdiagonal_mask


class Preconditions(nn.Module):
    """One cell per (action, relation, slot, slot), over ignore / require-true / require-false, arity-masked on the way out."""

    def __init__(
        self, num_actions: int, arity_relations: Sequence[int], num_slots: int
    ):
        super().__init__()
        self.logits = nn.Parameter(
            0.1
            * torch.randn(num_actions, len(arity_relations), num_slots, num_slots, 3)
        )
        self.register_buffer(
            "mask",
            unary_offdiagonal_mask(arity_relations, num_slots),
            persistent=False,
        )

    def forward(self, y_action: torch.Tensor) -> torch.Tensor:
        """Masked logits for one batch of actions, ``(B, R, k, k, 3)``."""
        return forbid(self.logits[y_action], self.mask)

    def action_logits(self) -> torch.Tensor:
        """Masked logits for every action, ``(A, R, k, k, 3)``."""
        return forbid(self.logits, self.mask)


def soft_conjunction(values: torch.Tensor, dim: int, temperature: float = 1.0):
    """A relaxed product of `values` along `dim`.

    The exact product collapses to zero as soon as one atom fails, carrying almost no gradient back to the others. Dividing the sum of logs by ``1 + temperature*(n-1)`` softens that: geometric mean at 1, exact product at 0. Training anneals from one to the other.
    """
    # Clamped so log stays finite: zeros are common here, and log(0) has an unstable backward.
    logs = torch.log(values.clamp(min=1e-8, max=1.0))
    return torch.exp(logs.sum(dim=dim) / (1 + temperature * (values.shape[dim] - 1)))
