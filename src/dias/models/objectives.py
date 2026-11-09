"""The training objective: `L_main` and `L_aux` (paper, Sec. 4.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .outputs import SchemaPrediction


@dataclass(frozen=True)
class Losses:
    """The loss terms of one step and the diagnostics logged beside them, each divided by `N`."""

    main: torch.Tensor
    """`L_main`: the cross-entropy of the predicted successor, plus the applicability term when it is on."""

    aux: torch.Tensor
    """`L_eff + L_pre`, before the weight alpha."""

    hamming_distance: torch.Tensor
    """Cells of the successor the thresholded prediction gets wrong."""

    precondition_satisfaction: torch.Tensor
    precondition_product: torch.Tensor

    applicability: torch.Tensor | None = None
    """The applicability term, when it is on."""

    def total(self, alpha: float) -> torch.Tensor:
        return self.main + alpha * self.aux

    def logged(self, alpha: float) -> dict[str, torch.Tensor]:
        logged = {
            "loss": self.total(alpha).detach(),
            "loss_main": self.main.detach(),
            "loss_aux": self.aux.detach(),
            "hamming_distance": self.hamming_distance,
            "precondition_satisfaction": self.precondition_satisfaction,
            "precondition_product": self.precondition_product,
        }
        if self.applicability is not None:
            logged["loss_applicability"] = self.applicability.detach()
        return logged


class DiasObjective:
    """`L_main = L_adj / N` and `L_aux = (L_eff + L_pre) / N`, returned separately for the gradient combiner.

    With `use_applicability_loss`, `L_main` adds a cross-entropy per transition asking its preconditions to hold at the bound objects. With `observation_noise` ε, `L_adj` scores the observed successor against ``ε + (1 - 2ε) Â'``.
    """

    EPSILON = 1e-7

    def __init__(
        self,
        relation_arities: Sequence[int],
        num_slots: int,
        use_effect_loss: bool = True,
        use_precondition_loss: bool = True,
        use_applicability_loss: bool = False,
        observation_noise: float = 0.0,
    ):
        if not 0.0 <= observation_noise < 0.5:
            raise ValueError(
                f"observation_noise must lie in [0, 0.5), not {observation_noise!r}"
            )
        # Schema cells per action: M^|p| per relation p.
        self.schema_cells = int(sum(num_slots**arity for arity in relation_arities))
        self.use_effect_loss = use_effect_loss
        self.use_precondition_loss = use_precondition_loss
        self.use_applicability_loss = use_applicability_loss
        self.observation_noise = observation_noise

    def __call__(
        self,
        prediction: SchemaPrediction,
        target: torch.Tensor,
        slot_weights: torch.Tensor,
    ) -> Losses:
        predicted = prediction.adjacency
        with torch.amp.autocast(device_type=predicted.device.type, enabled=False):
            normalizer = self._normalizer(predicted)
            main = (
                F.binary_cross_entropy(
                    self.observed(predicted.float()).clamp(
                        self.EPSILON, 1 - self.EPSILON
                    ),
                    target.float(),
                    reduction="sum",
                )
                / normalizer
            )
            aux = self._aux(prediction, slot_weights) / normalizer
            applicability = None
            if self.use_applicability_loss:
                satisfaction = prediction.precondition_satisfaction.float()
                applicability = (
                    _cross_entropy_against_one(satisfaction.clamp(min=self.EPSILON))
                    / normalizer
                )
                main = main + applicability

        return Losses(
            main=main,
            aux=aux,
            hamming_distance=((predicted > 0.5) ^ (target > 0.5)).sum(),
            precondition_satisfaction=prediction.precondition_satisfaction.mean().detach(),
            precondition_product=prediction.precondition_product.mean(),
            applicability=applicability,
        )

    def observed(self, predicted: torch.Tensor) -> torch.Tensor:
        """The predicted successor seen through a channel flipping each atom with probability `observation_noise`."""
        if self.observation_noise == 0.0:
            return predicted
        return self.observation_noise + (1 - 2 * self.observation_noise) * predicted

    def _aux(
        self, prediction: SchemaPrediction, slot_weights: torch.Tensor
    ) -> torch.Tensor:
        """`L_eff + L_pre`, either of which may be off."""
        ignore_effects = prediction.effects.ignore
        terms = torch.zeros((), device=ignore_effects.device, dtype=torch.float32)
        if self.use_effect_loss:
            terms = terms + _cross_entropy_against_one(ignore_effects)
        if self.use_precondition_loss:
            weight_pairs = (
                slot_weights[:, None, :, None] * slot_weights[:, None, None, :]
            )
            restrictive = ((1 - prediction.preconditions.ignore) * weight_pairs).clamp(
                min=self.EPSILON
            )
            terms = terms + _cross_entropy_against_one(restrictive)
        return terms

    def _normalizer(self, predicted: torch.Tensor) -> int:
        """`N` (paper, Eq. 12) summed over the batch, plus one cell per transition for the applicability term."""
        batch = predicted.size(0)
        applicability = batch if self.use_applicability_loss else 0
        return predicted.numel() + self.schema_cells * batch * 2 + applicability


def _cross_entropy_against_one(probabilities: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(
        probabilities, torch.ones_like(probabilities), reduction="sum"
    )
