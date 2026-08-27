"""What an action changes about a state: one learned tensor per action (DIAS), or predicted from the state (the MLP baseline)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .masking import forbid, unary_offdiagonal_mask

HIDDEN_DIM = 64


class EffectSource(nn.Module):
    """Per-action effect logits over ignore / add / delete. Subclasses supply the raw logits; the arity mask is applied here, so no source can skip it."""

    def __init__(self, arity_relations: Sequence[int], num_slots: int):
        super().__init__()
        self.register_buffer(
            "mask",
            unary_offdiagonal_mask(arity_relations, num_slots),
            persistent=False,
        )

    def forward(
        self,
        y_action: torch.Tensor,
        adjacency: torch.Tensor,
        selection: torch.Tensor,
    ) -> torch.Tensor:
        """Masked logits for one batch of actions, ``(B, R, k, k, 3)``."""
        return forbid(self._logits(y_action, adjacency, selection), self.mask)

    def _logits(
        self,
        y_action: torch.Tensor,
        adjacency: torch.Tensor,
        selection: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def action_logits(self) -> torch.Tensor | None:
        """Masked logits for every action, or None if they depend on the state."""
        raise NotImplementedError


class StaticEffects(EffectSource):
    """One learned effect tensor per action: the DIAS model."""

    def __init__(
        self, num_actions: int, arity_relations: Sequence[int], num_slots: int
    ):
        super().__init__(arity_relations, num_slots)
        self.logits = nn.Parameter(
            0.1
            * torch.randn(num_actions, len(arity_relations), num_slots, num_slots, 3)
        )

    def _logits(self, y_action, adjacency, selection):
        return self.logits[y_action]

    def action_logits(self) -> torch.Tensor:
        return forbid(self.logits, self.mask)


class StateDependentEffects(EffectSource):
    """Effects predicted from the current state: the MLP baseline. The state is projected onto the action's slots first, so the network sees a lifted state rather than the instance."""

    def __init__(
        self, num_actions: int, arity_relations: Sequence[int], num_slots: int
    ):
        super().__init__(arity_relations, num_slots)
        num_relations = len(arity_relations)
        cells = num_relations * num_slots * num_slots
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cells, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, num_actions * cells * 3),
            nn.Unflatten(1, (num_actions, num_relations, num_slots, num_slots, 3)),
        )

    def _logits(self, y_action, adjacency, selection):
        lifted_state = torch.einsum(
            "bai,brij,bcj->brac", selection, adjacency, selection
        )
        per_action = self.mlp(lifted_state)
        return per_action[
            torch.arange(y_action.size(0), device=y_action.device), y_action
        ]

    def action_logits(self) -> None:
        return None
