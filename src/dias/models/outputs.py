"""What the prediction module returns.

The schema tensors encode three mutually exclusive states per cell, softmaxed
over the last axis: leave the atom alone, assert it, or negate it. Effects read
that as ignore / add / delete and preconditions as ignore / positive /
negative, but it is one convention and it is decoded in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE, ASSERT, NEGATE = 0, 1, -1


@dataclass(frozen=True)
class Channels:
    """One schema tensor as probabilities, split by channel."""

    ignore: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor

    @classmethod
    def from_logits(cls, logits: torch.Tensor) -> "Channels":
        probabilities = F.softmax(logits, dim=-1)
        return cls(
            ignore=probabilities[..., IGNORE],
            positive=probabilities[..., ASSERT],
            negative=probabilities[..., NEGATE],
        )


@dataclass(frozen=True)
class SchemaPrediction:
    """One forward pass: the successor state, and everything it was built from."""

    adjacency: torch.Tensor
    """Predicted successor ``A' = A + pre * delta``, per relation."""

    input_adjacency: torch.Tensor
    """The state the action was applied to."""

    delta: torch.Tensor
    """The change the action would make, before precondition gating."""

    effects: Channels
    preconditions: Channels

    selection: torch.Tensor
    """Soft slot-to-object assignment, ``(batch, slots, objects)``."""

    selection_logits: torch.Tensor
    """Assignment scores before Sinkhorn, with the slack row appended."""

    precondition_satisfaction: torch.Tensor
    """Relaxed measure of whether the preconditions hold, one per sample."""

    precondition_product: torch.Tensor
    """The unrelaxed product, detached: a diagnostic, never a training signal."""
