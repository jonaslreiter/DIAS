"""What the prediction module returns.

Schema tensors hold three mutually exclusive states per cell, softmaxed over the last axis: ignore the atom, assert it, negate it. Effects read that as ignore / add / delete and preconditions as ignore / positive / negative.
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

    correspondence: torch.Tensor
    """Assignment scores before Sinkhorn, with the slack row appended."""

    precondition_satisfaction: torch.Tensor
    """Relaxed measure of whether the preconditions hold, one per sample."""

    precondition_product: torch.Tensor
    """The unrelaxed product, detached: a diagnostic, never a training signal."""


@dataclass(frozen=True)
class SchemaLogits:
    """A trained model's lifted schema, masked and detached. `effects` is None when they are predicted from the state."""

    preconditions: torch.Tensor
    effects: torch.Tensor | None
    slot_weight_logits: torch.Tensor
