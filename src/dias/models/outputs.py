"""What the model returns. Schema tensors have three channels per cell, softmaxed: ignore, positive and negative, read as add and delete for effects."""

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
    """The predicted successor ``A'``."""

    input_adjacency: torch.Tensor
    """The state the action was applied to."""

    delta: torch.Tensor
    """The change the action makes, before gating by `p_pre`."""

    effects: Channels
    preconditions: Channels

    selection: torch.Tensor
    """``S̃^a``, ``(B, M, O)``."""

    correspondence: torch.Tensor
    """``C^a`` with the slack row."""

    precondition_satisfaction: torch.Tensor
    """``p_pre``, one per transition."""

    precondition_product: torch.Tensor
    """The product at tau = 0, detached."""


@dataclass(frozen=True)
class SchemaLogits:
    """A model's lifted schema logits, masked and detached; `effects` is None for MLP effects."""

    preconditions: torch.Tensor
    effects: torch.Tensor | None
    slot_weight_logits: torch.Tensor
