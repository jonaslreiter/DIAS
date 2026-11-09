"""Combining the gradients of `L_main` and `L_aux` (paper, App. B.3)."""

from __future__ import annotations

from typing import Iterable, Protocol

import torch


class GradientCombiner(Protocol):
    """Sets ``parameter.grad`` from the main and the auxiliary loss."""

    def backward(
        self,
        main: torch.Tensor,
        aux: torch.Tensor,
        parameters: Iterable[torch.nn.Parameter],
    ) -> None: ...


class SingleObjective:
    """The main loss alone."""

    def backward(self, main, aux, parameters) -> None:
        main.backward()


class WeightedSum:
    """``main + alpha * aux``."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def backward(self, main, aux, parameters) -> None:
        (main + self.alpha * aux).backward()


class PCGrad:
    """Per parameter tensor, the auxiliary gradient without its component against the main one, capped at the main gradient's norm."""

    # Below this squared norm, the main gradient is not projected against.
    NEGLIGIBLE = 1e-12

    def __init__(self, alpha: float):
        self.alpha = alpha

    def backward(self, main, aux, parameters) -> None:
        parameters = list(parameters)
        main_grads = torch.autograd.grad(
            main, parameters, retain_graph=True, allow_unused=True
        )
        aux_grads = torch.autograd.grad(
            self.alpha * aux, parameters, retain_graph=True, allow_unused=True
        )
        for parameter, main_grad, aux_grad in zip(parameters, main_grads, aux_grads):
            combined = self._combine(main_grad, aux_grad)
            if combined is not None:
                parameter.grad = combined

    def _combine(
        self, main: torch.Tensor | None, aux: torch.Tensor | None
    ) -> torch.Tensor | None:
        """One parameter's gradient, or None if neither loss reaches it."""
        if main is None or aux is None:
            reached = aux if main is None else main
            return None if reached is None else reached.clone()
        aux = self._cap_norm(main, self._project(main, aux))
        return (main + aux).clone()

    @classmethod
    def _project(cls, main: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """`aux` without its component against `main`."""
        flat_main = main.reshape(-1)
        opposition = torch.dot(flat_main, aux.reshape(-1))
        if opposition >= 0:
            return aux
        squared_norm = torch.dot(flat_main, flat_main)
        if squared_norm <= cls.NEGLIGIBLE:
            return aux
        return aux - (opposition / squared_norm) * main

    @staticmethod
    def _cap_norm(main: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """`aux` scaled down to the norm of `main` if longer."""
        main_norm = torch.norm(main)
        aux_norm = torch.norm(aux)
        if aux_norm > main_norm:
            return aux * (main_norm / aux_norm)
        return aux


def gradient_combiner(
    use_pcgrad: bool, alpha: float, aux_active: bool
) -> GradientCombiner:
    if not aux_active:
        return SingleObjective()
    if not use_pcgrad:
        return WeightedSum(alpha)
    return PCGrad(alpha)
