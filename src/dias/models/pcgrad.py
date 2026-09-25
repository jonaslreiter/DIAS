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
    """One backward pass over ``primary + alpha * auxiliary``."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def backward(self, primary, auxiliary, parameters) -> None:
        (primary + self.alpha * auxiliary).backward()


class PCGrad:
    """Where the two gradients conflict, keep the primary one's direction.

    Per parameter, the auxiliary gradient is projected onto the plane orthogonal to the primary one wherever they oppose, so it can only act in directions the primary objective is indifferent to.
    """

    # Below this the primary gradient is noise, typically once training has converged, so a conflict with it says nothing about a wrong direction: the auxiliary gradient is left unprojected, and the norm cap bounds how much it can do.
    NEGLIGIBLE = 1e-12

    def __init__(self, alpha: float, norm_cap: bool = True):
        self.alpha = alpha
        self.norm_cap = norm_cap

    def backward(self, primary, auxiliary, parameters) -> None:
        parameters = list(parameters)
        # Alpha is applied inside autograd rather than to the gradient afterwards.
        primary_grads = torch.autograd.grad(
            primary, parameters, retain_graph=True, allow_unused=True
        )
        auxiliary_grads = torch.autograd.grad(
            self.alpha * auxiliary, parameters, retain_graph=True, allow_unused=True
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
        """The gradient one parameter ends up with, or None to leave it alone."""
        if primary is None or auxiliary is None:
            reached = auxiliary if primary is None else primary
            return None if reached is None else reached.clone()

        auxiliary = self._project(primary, auxiliary)
        if self.norm_cap:
            auxiliary = self._cap_norm(primary, auxiliary)
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
    def _cap_norm(cls, primary: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        """Keep the auxiliary gradient no longer than the primary one: orthogonality alone does not bound it, and it can dominate early."""
        primary_norm = torch.norm(primary)
        auxiliary_norm = torch.norm(auxiliary)
        if auxiliary_norm > primary_norm:
            return auxiliary * (primary_norm / auxiliary_norm)
        return auxiliary


def gradient_combiner(
    use_pcgrad: bool,
    alpha: float,
    auxiliary_active: bool,
    norm_cap: bool = True,
) -> GradientCombiner:
    """Pick how the two losses are reconciled, once, at construction."""
    if not auxiliary_active:
        return SingleObjective()
    if not use_pcgrad:
        return WeightedSum(alpha)
    return PCGrad(alpha, norm_cap=norm_cap)
