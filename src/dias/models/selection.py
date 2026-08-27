"""Binding an action's slots to the objects of a state."""

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
    """How many slots every action gets, given what the instance supports.

    With `per_action_slots` each action uses exactly as many slots as its own arity, so the widest action sizes the tensors. Otherwise the count is configured, but capped at the number of objects: a slot that no object can bind to is dead weight in every tensor indexed by it.
    """
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
    """A soft assignment of slots to objects, one per sample.

    Each (action, slot) has a learned query that scores the encoded objects; Sinkhorn normalisation turns those scores into an assignment. The extended matrix has a slack row and no slack column, so a slot may bind to nothing while every object still has to be accounted for.
    """

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

        self.slot_encodings = nn.Parameter(
            0.1 * torch.randn(num_actions, num_slots, object_dim)
        )
        self.slot_weight_logits = nn.Parameter(torch.zeros(num_actions, num_slots))
        # Derived from the constructor arguments, so it is rebuilt on load rather than carried in the checkpoint.
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
        """``(B, N, D)`` embeddings to a ``(B, k, N)`` selection and its logits."""
        batch, num_objects, _ = embeddings.shape
        queries = self.slot_encodings[y_action]  # (B, k, D)
        scores = torch.bmm(queries, embeddings.transpose(1, 2)) / self.object_dim**0.5

        logits = torch.zeros(
            (batch, self.num_slots + 1, num_objects),
            device=scores.device,
            dtype=torch.float32,
        )
        logits[:, :-1, :] = scores

        unused = self.unused_slots(y_action) if self.per_action_slots else None
        row_mask = None if unused is None else F.pad(unused, (0, 1), value=False)

        if self.force_action_args and y_action_args is not None:
            self._pin_known_arguments(logits, y_action_args)

        selection = batched_sinkhorn_d1d2(logits, row_mask=row_mask)[:, :-1, :]
        if unused is not None:
            selection = selection.clone()
            selection[unused] = 0.0
        else:
            weights = self.slot_weights(y_action, y_action_args)
            selection = selection * weights.unsqueeze(-1)
        return selection, logits

    def slot_weights(
        self, y_action: torch.Tensor, y_action_args: torch.Tensor | None = None
    ) -> torch.Tensor:
        """How strongly each slot participates, ``(B, k)``.

        Without learned slot weights every slot is fully active. Given the action's arguments, slots whose object is already known keep their full weight: their binding is not something the model has to learn.
        """
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

    @staticmethod
    def _pin_known_arguments(logits: torch.Tensor, y_action_args: torch.Tensor) -> None:
        """Force slots with a known object onto it, by hard-masking the logits.

        The slot's row and the object's column are ruled out everywhere else, leaving Sinkhorn a single admissible cell for that pair.
        """
        known = y_action_args >= 0  # (B, k)
        if not known.any():
            return
        samples, slots = known.nonzero(as_tuple=True)
        objects = y_action_args[samples, slots]

        row_mask = torch.zeros(
            logits.shape[:2] + (1,), device=logits.device, dtype=torch.bool
        )
        row_mask[samples, slots, 0] = True
        logits.masked_fill_(row_mask, -torch.inf)

        column_mask = torch.zeros(
            (logits.size(0), logits.size(2)), device=logits.device, dtype=torch.bool
        )
        column_mask[samples, objects] = True
        logits.masked_fill_(column_mask.unsqueeze(1), -torch.inf)

        logits[samples, slots, objects] = 0.0


def batched_sinkhorn_d1d2(
    log_A: torch.Tensor,
    max_iter: int = 10000,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched Sinkhorn D1D2 in log-space (https://hal.science/hal-03768664)."""
    tol = 1.0e-2
    batch_size, np1, _ = log_A.shape
    n = np1 - 1

    log_x = torch.zeros(batch_size, np1, device=log_A.device, dtype=log_A.dtype)
    log_y = torch.zeros(
        batch_size, log_A.size(-1), device=log_A.device, dtype=log_A.dtype
    )
    if row_mask is not None:
        log_x = log_x.masked_fill(row_mask, -torch.inf)

    for _ in range(max_iter):
        log_x_prev, log_y_prev = log_x, log_y
        log_x = -torch.logsumexp(log_A + log_y.unsqueeze(-2), dim=-1)
        log_x[:, n] = 0.0
        if row_mask is not None:
            log_x = log_x.masked_fill(row_mask, -torch.inf)
        log_y = -torch.logsumexp(log_A + log_x.unsqueeze(-1), dim=-2)

        if (
            max(
                (log_x - log_x_prev).abs().max(),
                (log_y - log_y_prev).abs().max(),
            )
            < tol
        ):
            break

    return torch.exp(log_x.unsqueeze(-1) + log_A + log_y.unsqueeze(-2))
