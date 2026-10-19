"""What training minimises."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .outputs import SchemaPrediction

PRECONDITION_LOSSES = ("restrict", "reject")


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

    precondition_admitted: torch.Tensor | None = None
    """Mean fraction of the sampled ground actions the preconditions admit. Only the rejection loss samples them."""

    applicability: torch.Tensor | None = None
    """How far the preconditions are from holding at the bound objects of the observed transitions. Part of the primary loss, and only computed when that term is on."""

    @property
    def primary(self) -> torch.Tensor:
        """What the auxiliary gradient may not override: `L_adj`, plus the applicability term when it is on."""
        if self.applicability is None:
            return self.adjacency
        return self.adjacency + self.applicability

    def total(self, alpha: float) -> torch.Tensor:
        return self.primary + alpha * self.auxiliary

    def logged(self, alpha: float) -> dict[str, torch.Tensor]:
        """What one step reports, under the names a run's history uses."""
        logged = {
            "loss": self.total(alpha).detach(),
            "loss_adjacency": self.adjacency.detach(),
            "loss_aux": self.auxiliary.detach(),
            "hamming_distance": self.hamming_distance,
            "precondition_satisfaction": self.precondition_satisfaction,
            "precondition_product": self.precondition_product,
        }
        if self.precondition_admitted is not None:
            logged["precondition_admitted"] = self.precondition_admitted
        if self.applicability is not None:
            logged["loss_applicability"] = self.applicability.detach()
        return logged


class DiasObjective:
    """Reproduce the observed successor state, with two nudges toward simplicity.

    `L_adj` is a cross-entropy over every cell of the predicted successor. `L_eff` asks the schema to ignore every effect it can and `L_pre` to restrict every precondition it can, which stops it explaining a transition with a schema that fires everywhere else. The two are returned beside `L_adj`, never summed with it: that is `pcgrad`'s decision.

    `L_pre` comes in two forms. `restrict`, the paper's, asks every precondition cell to require or forbid its atom. `reject` asks the preconditions to reject as many uniformly sampled ground actions of the batch's states as they can (see `rejection.py`), and needs their satisfaction passed in as `admitted`.

    With `use_applicability_loss`, the primary loss also asks the preconditions to hold at the bound objects of every observed transition, since every observed action was applicable. `L_adj` alone says so only where the action changed something: where its add effects already held, failing preconditions explain the transition as well as holding ones, and `L_pre` then pushes in a literal no observation refutes, such as satellite's calibrate forbidding the `calibrated` it adds.
    """

    # Clamp off 0 and 1: the cross-entropy gradient is unbounded at both ends.
    EPSILON = 1e-7

    def __init__(
        self,
        relation_arities: Sequence[int],
        num_slots: int,
        use_effect_loss: bool = True,
        use_precondition_loss: bool = True,
        precondition_loss: str = "restrict",
        use_applicability_loss: bool = False,
    ):
        if precondition_loss not in PRECONDITION_LOSSES:
            raise ValueError(
                f"precondition_loss must be one of {PRECONDITION_LOSSES}, not {precondition_loss!r}"
            )
        # One schema cell per slot tuple a relation of that arity ranges over.
        self.schema_cells = int(sum(num_slots**arity for arity in relation_arities))
        self.use_effect_loss = use_effect_loss
        self.use_precondition_loss = use_precondition_loss
        self.precondition_loss = precondition_loss
        self.use_applicability_loss = use_applicability_loss

    @property
    def rejects(self) -> bool:
        """Whether `L_pre` is the rejection loss, and so needs `admitted`."""
        return self.use_precondition_loss and self.precondition_loss == "reject"

    def __call__(
        self,
        prediction: SchemaPrediction,
        target: torch.Tensor,
        slot_weights: torch.Tensor,
        admitted: torch.Tensor | None = None,
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
            auxiliary = self._auxiliary(prediction, slot_weights, admitted) / normalizer
            applicability = None
            if self.use_applicability_loss:
                # Off zero only, as a satisfaction never exceeds one.
                satisfaction = prediction.precondition_satisfaction.float()
                applicability = (
                    _cross_entropy_against_one(satisfaction.clamp(min=self.EPSILON))
                    / normalizer
                )

        return Losses(
            adjacency=adjacency,
            auxiliary=auxiliary,
            hamming_distance=((predicted > 0.5) ^ (target > 0.5)).sum(),
            precondition_satisfaction=prediction.precondition_satisfaction.mean().detach(),
            precondition_product=prediction.precondition_product.mean(),
            precondition_admitted=(
                None if admitted is None else admitted.mean().detach()
            ),
            applicability=applicability,
        )

    def _auxiliary(
        self,
        prediction: SchemaPrediction,
        slot_weights: torch.Tensor,
        admitted: torch.Tensor | None,
    ) -> torch.Tensor:
        """`L_eff + L_pre`, either of which can be ablated away."""
        ignore_effects = prediction.effects.ignore
        # Full precision, so the sum over every schema cell accumulates exactly.
        terms = torch.zeros((), device=ignore_effects.device, dtype=torch.float32)
        if self.use_effect_loss:
            terms = terms + _cross_entropy_against_one(ignore_effects)
        if self.rejects:
            if admitted is None:
                raise ValueError(
                    "The rejection loss needs `admitted`, the satisfaction of sampled ground actions."
                )
            # Counted as the schema cells of the restrictive term it replaces, which is what the normalizer counts, so alpha weighs the two alike.
            batch = prediction.adjacency.size(0)
            terms = terms + admitted.float().mean() * self.schema_cells * batch
        elif self.use_precondition_loss:
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
        """Every cell the prediction could have got wrong, plus the schema cells each auxiliary term ranges over, plus one applicability per transition when that term is on.

        Summing and dividing keeps the terms on one scale; averaged separately, the adjacency term would shrink as the instance grew.
        """
        batch = predicted.size(0)
        applicability = batch if self.use_applicability_loss else 0
        return predicted.numel() + self.schema_cells * batch * 2 + applicability


def _cross_entropy_against_one(probabilities: torch.Tensor) -> torch.Tensor:
    """How far `probabilities` are from all being 1, summed."""
    return F.binary_cross_entropy(
        probabilities, torch.ones_like(probabilities), reduction="sum"
    )
