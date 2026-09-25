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

    def total(self, alpha: float) -> torch.Tensor:
        return self.adjacency + alpha * self.auxiliary

    def logged(self, alpha: float) -> dict[str, torch.Tensor]:
        """What one step reports, under the names a run's history uses."""
        return {
            "loss": self.total(alpha).detach(),
            "loss_adjacency": self.adjacency.detach(),
            "loss_aux": self.auxiliary.detach(),
            "hamming_distance": self.hamming_distance,
            "precondition_satisfaction": self.precondition_satisfaction,
            "precondition_product": self.precondition_product,
        }


class DiasObjective:
    """Reproduce the observed successor state, with two nudges toward simplicity.

    `L_adj` is a cross-entropy over every cell of the predicted successor. `L_eff` asks the schema to ignore every effect it can and `L_pre` every precondition it can, which stops it explaining a transition with a schema that fires everywhere else. The two are returned beside `L_adj`, never summed with it: that is `pcgrad`'s decision.
    """

    # Clamp off 0 and 1: the cross-entropy gradient is unbounded at both ends.
    EPSILON = 1e-7

    def __init__(
        self,
        relation_arities: Sequence[int],
        num_slots: int,
        use_effect_loss: bool = True,
        use_precondition_loss: bool = True,
    ):
        # One schema cell per slot tuple a relation of that arity ranges over.
        self.schema_cells = int(sum(num_slots**arity for arity in relation_arities))
        self.use_effect_loss = use_effect_loss
        self.use_precondition_loss = use_precondition_loss

    def __call__(
        self,
        prediction: SchemaPrediction,
        target: torch.Tensor,
        slot_weights: torch.Tensor,
    ) -> Losses:
        predicted = prediction.adjacency
        # In half precision a sum over every cell of every relation loses too much.
        with torch.amp.autocast(device_type=predicted.device.type, enabled=False):
            normalizer = self._normalizer(predicted)
            adjacency = (
                F.binary_cross_entropy(
                    predicted.float().clamp(self.EPSILON, 1 - self.EPSILON),
                    target.float(),
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
        # Full precision, so the sum over every schema cell accumulates exactly.
        terms = torch.zeros((), device=ignore_effects.device, dtype=torch.float32)
        if self.use_effect_loss:
            terms = terms + _cross_entropy_against_one(ignore_effects)
        if self.use_precondition_loss:
            # Weighting by slot weights makes an unrestrictive precondition as costly as an active slot, so the two cannot be traded off.
            weight_pairs = (
                slot_weights[:, None, :, None] * slot_weights[:, None, None, :]
            )
            # Off zero only: a masked cell is exactly zero, and nothing here exceeds one.
            restrictive = ((1 - prediction.preconditions.ignore) * weight_pairs).clamp(
                min=self.EPSILON
            )
            terms = terms + _cross_entropy_against_one(restrictive)
        return terms

    def _normalizer(self, predicted: torch.Tensor) -> int:
        """Every cell the prediction could have got wrong, plus the schema cells each auxiliary term ranges over.

        Summing and dividing keeps the three terms on one scale; averaged separately, the adjacency term would shrink as the instance grew.
        """
        return predicted.numel() + self.schema_cells * predicted.size(0) * 2


def _cross_entropy_against_one(probabilities: torch.Tensor) -> torch.Tensor:
    """How far `probabilities` are from all being 1, summed."""
    return F.binary_cross_entropy(
        probabilities, torch.ones_like(probabilities), reduction="sum"
    )
