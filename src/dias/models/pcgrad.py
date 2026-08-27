"""Turning two losses into one gradient per parameter."""

from __future__ import annotations

from typing import Iterable, Protocol

import torch


class GradientCombiner(Protocol):
    """Populates ``parameter.grad`` from a primary and an auxiliary loss."""

    def backward(
        self,
        primary: torch.Tensor,
        auxiliary: torch.Tensor,
        parameters: Iterable[torch.nn.Parameter],
    ) -> None: ...


class SingleObjective:
    """No auxiliary objective at all: an ordinary backward pass."""

    def backward(self, primary, auxiliary, parameters) -> None:
        primary.backward()


class WeightedSum:
    """One backward pass over ``primary + weight * auxiliary``."""

    def __init__(self, weight: float):
        self.weight = weight

    def backward(self, primary, auxiliary, parameters) -> None:
        (primary + self.weight * auxiliary).backward()


class PCGrad:
    """Where the two gradients conflict, keep the primary one's direction.

    The auxiliary objective asks the schema to explain less, which past a point is at odds with explaining the transition at all. Rather than let a weight arbitrate globally, this projects the auxiliary gradient onto the plane orthogonal to the primary one wherever the two point against each other, per parameter, so the auxiliary term can only act in directions the primary objective is indifferent to.
    """

    # Below this the primary gradient carries no direction to project onto.
    NEGLIGIBLE = 1e-12

    def __init__(self, weight: float, dynamic_clip: bool = True):
        self.weight = weight
        self.dynamic_clip = dynamic_clip

    def backward(self, primary, auxiliary, parameters) -> None:
        parameters = list(parameters)
        # The weight is applied inside autograd rather than to the gradient afterwards, which is what the archived runs did.
        primary_grads = torch.autograd.grad(
            primary, parameters, retain_graph=True, allow_unused=True
        )
        auxiliary_grads = torch.autograd.grad(
            self.weight * auxiliary, parameters, retain_graph=True, allow_unused=True
        )
        for parameter, primary_grad, auxiliary_grad in zip(
            parameters, primary_grads, auxiliary_grads
        ):
            combined = self._combine(primary_grad, auxiliary_grad)
            if combined is not None:
                parameter.grad = combined

    def _combine(
        self, primary: torch.Tensor | None, auxiliary: torch.Tensor | None
    ) -> torch.Tensor | None:
        """The gradient one parameter ends up with, or None to leave it alone.

        A parameter only one of the two losses reaches gets that one's gradient; a parameter neither reaches keeps whatever it had.
        """
        if primary is None or auxiliary is None:
            reached = auxiliary if primary is None else primary
            return None if reached is None else reached.clone()

        auxiliary = self._project(primary, auxiliary)
        if self.dynamic_clip:
            auxiliary = self._clip_to(primary, auxiliary)
        return (primary + auxiliary).clone()

    @classmethod
    def _project(cls, primary: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        """Remove the component of `auxiliary` that opposes `primary`."""
        flat_primary = primary.reshape(-1)
        opposition = torch.dot(flat_primary, auxiliary.reshape(-1))
        if opposition >= 0:
            return auxiliary
        squared_norm = torch.dot(flat_primary, flat_primary)
        if squared_norm <= cls.NEGLIGIBLE:
            return auxiliary
        return auxiliary - (opposition / squared_norm) * primary

    @classmethod
    def _clip_to(cls, primary: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        """Keep the auxiliary gradient no longer than the primary one.

        Orthogonality alone does not bound the step the auxiliary term takes; without this it can dominate early, when the primary gradient is small.
        """
        primary_norm = torch.norm(primary)
        auxiliary_norm = torch.norm(auxiliary)
        if auxiliary_norm > primary_norm and auxiliary_norm > cls.NEGLIGIBLE:
            return auxiliary * (primary_norm / auxiliary_norm)
        return auxiliary


def gradient_combiner(
    use_pcgrad: bool,
    auxiliary_weight: float,
    auxiliary_active: bool,
    dynamic_clip: bool = True,
) -> GradientCombiner:
    """Pick how the two losses are reconciled, once, at construction."""
    if not auxiliary_active:
        return SingleObjective()
    if not use_pcgrad:
        return WeightedSum(auxiliary_weight)
    return PCGrad(auxiliary_weight, dynamic_clip=dynamic_clip)
