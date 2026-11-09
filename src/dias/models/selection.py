"""The selection `S̃^a` of objects into an action's slots (paper, Sec. 4.1 and App. B.4)."""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def resolve_num_slots(
    num_slots: int | None,
    num_objects: int,
    action_arities: Sequence[int] | None = None,
    per_action_slots: bool = False,
) -> int:
    """The number of slots `M`: the widest action's arity with `per_action_slots`, otherwise `num_slots`, capped at the number of objects."""
    if per_action_slots:
        assert (
            action_arities is not None
        ), "action_arities must be provided when per_action_slots=True"
        return max(action_arities)

    assert (
        num_slots is not None
    ), "num_slots must be set explicitly when per_action_slots=False"
    widest_action = max(action_arities) if action_arities is not None else 0
    assert (
        num_slots >= widest_action
    ), f"num_slots ({num_slots}) must be >= max action arity ({widest_action})"
    if num_slots <= num_objects:
        return num_slots

    logger.warning(
        "Clamping num_slots %d to %d, the number of objects in this instance.",
        num_slots,
        num_objects,
    )
    assert (
        num_objects >= widest_action
    ), f"Clamped num_slots ({num_objects}) is smaller than max action arity ({widest_action}); problem instance has too few objects for this domain."
    return num_objects


class SlotSelector(nn.Module):
    """Queries `Q^a` matched against the keys `K` by a rectangular Sinkhorn normalization, scaled by the slot weights `σ(w^a)`."""

    def __init__(
        self,
        num_actions: int,
        num_slots: int,
        object_dim: int,
        action_arities: Sequence[int] | None = None,
        per_action_slots: bool = False,
        force_action_args: bool = False,
        use_slot_weights: bool = True,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.object_dim = object_dim
        self.per_action_slots = per_action_slots
        self.force_action_args = force_action_args
        self.use_slot_weights = use_slot_weights

        self.slot_queries = nn.Parameter(
            0.1 * torch.randn(num_actions, num_slots, object_dim)
        )
        self.slot_weight_logits = nn.Parameter(torch.zeros(num_actions, num_slots))
        self.register_buffer(
            "action_arities",
            (
                torch.tensor(action_arities, dtype=torch.long)
                if action_arities is not None
                else torch.full((num_actions,), num_slots, dtype=torch.long)
            ),
            persistent=False,
        )

        if per_action_slots:
            with torch.no_grad():
                self.slot_weight_logits.masked_fill_(self.unused_slots(), -torch.inf)

    def unused_slots(self, y_action: torch.Tensor | None = None) -> torch.Tensor:
        """Slots beyond an action's arity, for the given actions or for all."""
        arities = (
            self.action_arities if y_action is None else self.action_arities[y_action]
        )
        slots = torch.arange(self.num_slots, device=arities.device)
        return slots.unsqueeze(0) >= arities.unsqueeze(1)

    def forward(
        self,
        embeddings: torch.Tensor,
        y_action: torch.Tensor,
        y_action_args: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, O, d)`` embeddings to a ``(B, M, O)`` selection, and the correspondence matrix ``C^a`` with its slack row."""
        batch, num_objects, _ = embeddings.shape
        queries = self.slot_queries[y_action]  # (B, k, D)
        scores = torch.bmm(queries, embeddings.transpose(1, 2)) / self.object_dim**0.5

        correspondence = torch.zeros(
            (batch, self.num_slots + 1, num_objects),
            device=scores.device,
            dtype=torch.float32,
        )
        correspondence[:, :-1, :] = scores

        unused = self.unused_slots(y_action) if self.per_action_slots else None
        row_mask = None if unused is None else F.pad(unused, (0, 1), value=False)

        if self.force_action_args and y_action_args is not None:
            self._pin_known_arguments(correspondence, y_action_args)

        selection = batched_sinkhorn_d1d2(correspondence, row_mask=row_mask)[:, :-1, :]
        if unused is not None:
            selection = selection.clone()
            selection[unused] = 0.0
        else:
            weights = self.slot_weights(y_action, y_action_args)
            selection = selection * weights.unsqueeze(-1)
        return selection, correspondence

    def slot_weights(
        self, y_action: torch.Tensor, y_action_args: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``σ(w^a)``, ``(B, M)``; a slot whose object the action label names has weight 1."""
        logits = self.slot_weight_logits[y_action]
        if not self.use_slot_weights:
            return torch.ones_like(logits)

        weights = torch.sigmoid(logits)
        if self.force_action_args and y_action_args is not None:
            forced = y_action_args >= 0
            if forced.any():
                if forced.shape[1] < weights.shape[1]:
                    forced = F.pad(forced, (0, weights.shape[1] - forced.shape[1]))
                weights = torch.where(forced, torch.ones_like(weights), weights)
        return weights

    def applied_slot_weight_logits(self) -> torch.Tensor:
        """``(A, M)`` logits of the weights `forward` applies: infinite for a slot always used at full weight, which is every slot without `use_slot_weights` and every slot within the arity with `per_action_slots`."""
        if self.use_slot_weights and not self.per_action_slots:
            return self.slot_weight_logits
        logits = torch.full_like(self.slot_weight_logits, torch.inf)
        if self.per_action_slots:
            logits.masked_fill_(self.unused_slots(), -torch.inf)
        return logits

    @staticmethod
    def _pin_known_arguments(
        correspondence: torch.Tensor, y_action_args: torch.Tensor
    ) -> None:
        """Bind each slot whose object the label names to that object, and to nothing else."""
        known = y_action_args >= 0  # (B, k)
        if not known.any():
            return
        samples, slots = known.nonzero(as_tuple=True)
        objects = y_action_args[samples, slots]

        row_mask = torch.zeros(
            correspondence.shape[:2] + (1,),
            device=correspondence.device,
            dtype=torch.bool,
        )
        row_mask[samples, slots, 0] = True
        correspondence.masked_fill_(row_mask, -torch.inf)

        column_mask = torch.zeros(
            (correspondence.size(0), correspondence.size(2)),
            device=correspondence.device,
            dtype=torch.bool,
        )
        column_mask[samples, objects] = True
        correspondence.masked_fill_(column_mask.unsqueeze(1), -torch.inf)

        correspondence[samples, slots, objects] = 0.0


def batched_sinkhorn_d1d2(
    correspondence: torch.Tensor,
    max_iter: int = 10000,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sinkhorn D1D2 in log space (Brun et al., 2022) on ``C^a`` with a slack row, which absorbs the mass of objects no slot selects."""
    tol = 1.0e-2
    batch_size, num_rows, _ = correspondence.shape
    slack_row = num_rows - 1

    log_row = torch.zeros(
        batch_size,
        num_rows,
        device=correspondence.device,
        dtype=correspondence.dtype,
    )
    log_col = torch.zeros(
        batch_size,
        correspondence.size(-1),
        device=correspondence.device,
        dtype=correspondence.dtype,
    )
    if row_mask is not None:
        log_row = log_row.masked_fill(row_mask, -torch.inf)

    for _ in range(max_iter):
        previous_row, previous_col = log_row, log_col
        log_row = -torch.logsumexp(correspondence + log_col.unsqueeze(-2), dim=-1)
        log_row[:, slack_row] = 0.0
        if row_mask is not None:
            log_row = log_row.masked_fill(row_mask, -torch.inf)
        log_col = -torch.logsumexp(correspondence + log_row.unsqueeze(-1), dim=-2)

        row_change = (log_row - previous_row).abs()
        if row_mask is not None:
            row_change = row_change.masked_fill(row_mask, 0.0)
        if max(row_change.max(), (log_col - previous_col).abs().max()) < tol:
            break

    return torch.exp(log_row.unsqueeze(-1) + correspondence + log_col.unsqueeze(-2))
