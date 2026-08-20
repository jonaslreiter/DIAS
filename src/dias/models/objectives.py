"""What training minimises."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .outputs import SchemaPrediction


@dataclass(frozen=True)
class Losses:
    """The loss terms of one step, and the diagnostics logged beside them."""

    adjacency: torch.Tensor
    """How far the predicted successor state is from the observed one."""

    auxiliary: torch.Tensor
    """The two terms pushing the schema toward ignoring what it can."""

    hamming_distance: torch.Tensor
    """Cells the thresholded prediction got wrong. Never trained on."""

    precondition_satisfaction: torch.Tensor
    precondition_product: torch.Tensor

    def total(self, auxiliary_weight: float) -> torch.Tensor:
        return self.adjacency + auxiliary_weight * self.auxiliary

    def logged(self, auxiliary_weight: float) -> dict[str, torch.Tensor]:
        """What one step reports, under the names a run's history uses."""
        return {
            "loss": self.total(auxiliary_weight).detach(),
            "loss_adjacency": self.adjacency.detach(),
            "loss_aux": self.auxiliary.detach(),
            "hamming_distance": self.hamming_distance,
            "pre_mean": self.precondition_satisfaction,
            "pre_prod_mean": self.precondition_product,
        }


class DiasObjective:
    """Reproduce the observed successor state, with two nudges toward simplicity.

    `L_adj` is the objective proper, a cross-entropy over every cell of the
    predicted successor. `L_eff` asks the schema to ignore every effect it can
    and `L_pre` every precondition it can, which is what stops it explaining a
    transition with a schema that also fires everywhere else. The two are
    returned separately from `L_adj`, never summed with it: how the two
    gradients are reconciled is `pcgrad`'s decision, not this one's.
    """

    # Probabilities are clamped off 0 and 1 before the cross-entropy, whose
    # gradient is unbounded at both ends.
    EPSILON = 1e-7

    def __init__(
        self,
        arity_relations: Sequence[int],
        num_slots: int,
        use_effect_loss: bool = True,
        use_precondition_loss: bool = True,
    ):
        # One schema cell per slot tuple a relation of that arity ranges over.
        self.schema_cells = int(sum(num_slots**arity for arity in arity_relations))
        self.use_effect_loss = use_effect_loss
        self.use_precondition_loss = use_precondition_loss

    def __call__(
        self,
        prediction: SchemaPrediction,
        target: torch.Tensor,
        slot_weights: torch.Tensor,
    ) -> Losses:
        predicted = prediction.adjacency
        # Under AMP the cross-entropies would run in half precision, where a
        # sum over every cell of every relation loses too much.
        with torch.amp.autocast(device_type=predicted.device.type, enabled=False):
            normalizer = self._normalizer(predicted)
            adjacency = (
                F.binary_cross_entropy(
                    predicted.float().clamp(self.EPSILON, 1 - self.EPSILON),
                    target.float().clamp(0.0, 1.0),
                    reduction="sum",
                )
                / normalizer
            )
            auxiliary = self._auxiliary(prediction, slot_weights) / normalizer

        return Losses(
            adjacency=adjacency,
            auxiliary=auxiliary,
            hamming_distance=((predicted > 0.5) ^ (target > 0.5)).sum(),
            precondition_satisfaction=prediction.precondition_satisfaction.mean().detach(),
            precondition_product=prediction.precondition_product.mean(),
        )

    def _auxiliary(
        self, prediction: SchemaPrediction, slot_weights: torch.Tensor
    ) -> torch.Tensor:
        """`L_eff + L_pre`, either of which can be ablated away."""
        ignore_effects = prediction.effects.ignore
        # Full precision, so a sum over every schema cell still accumulates
        # exactly when the terms themselves are half under AMP.
        terms = torch.zeros((), device=ignore_effects.device, dtype=torch.float32)
        if self.use_effect_loss:
            terms = terms + _cross_entropy_against_one(ignore_effects)
        if self.use_precondition_loss:
            # Weighting by the slot weights makes an unrestrictive precondition
            # as costly as an active slot, so the two cannot be traded off.
            weight_pairs = (
                slot_weights[:, None, :, None] * slot_weights[:, None, None, :]
            )
            restrictive = ((1 - prediction.preconditions.ignore) * weight_pairs).clamp(
                self.EPSILON, 1 - self.EPSILON
            )
            terms = terms + _cross_entropy_against_one(restrictive)
        return terms

    def _normalizer(self, predicted: torch.Tensor) -> int:
        """What the summed terms are divided by.

        Every cell the prediction could have got wrong, plus the schema cells
        the auxiliary terms range over -- once for the effects, once for the
        preconditions. Summing and dividing rather than averaging keeps the
        three terms on one scale; averaged separately, the adjacency term
        would shrink as the instance grew while the schema terms stayed put.
        """
        return predicted.numel() + self.schema_cells * predicted.size(0) * 2


def _cross_entropy_against_one(probabilities: torch.Tensor) -> torch.Tensor:
    """How far `probabilities` are from all being 1, summed."""
    return F.binary_cross_entropy(
        probabilities, torch.ones_like(probabilities), reduction="sum"
    )
