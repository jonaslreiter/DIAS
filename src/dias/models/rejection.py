"""How many ground actions the learned preconditions admit, estimated from the states of one batch.

The rejection loss asks the preconditions to reject as many ground actions as they can, where a ground action binds every slot of an action to its own object, as a planner grounds the learned schema. Counting precondition literals instead rewards binding a parameter that occurs only in preconditions to whichever object satisfies the most of them, which can admit ground actions the domain does not have; counting what the preconditions reject cannot be gamed that way, because a literal that holds at every ground action rejects nothing.
"""

from __future__ import annotations

import torch

from .outputs import Channels
from .preconditions import soft_conjunction


def sample_bindings(
    num_states: int,
    num_actions: int,
    samples: int,
    num_objects: int,
    num_slots: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``(S, A, K, k)`` object indices: `samples` uniform ground actions per state and action, each binding its slots to distinct objects."""
    keys = torch.rand(num_states, num_actions, samples, num_objects, device=device)
    return keys.argsort(dim=-1)[..., :num_slots]


def lift(adjacency: torch.Tensor, bindings: torch.Tensor) -> torch.Tensor:
    """``(S, A, K, R, k, k)``: every state read over the slots of each of its bindings, the one-hot case of `prediction.project`."""
    states = torch.arange(adjacency.size(0), device=adjacency.device)
    by_cell = adjacency.permute(0, 2, 3, 1)  # (S, O, O, R)
    lifted = by_cell[
        states.view(-1, 1, 1, 1, 1), bindings.unsqueeze(-1), bindings.unsqueeze(-2)
    ]
    return lifted.permute(0, 1, 2, 5, 3, 4)


def admission(
    adjacency: torch.Tensor,
    preconditions: Channels,
    slot_weights: torch.Tensor,
    bindings: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """``(S, A, K)``: the relaxed precondition satisfaction of each of the given ground actions.

    `adjacency` is ``(S, R, O, O)``, `preconditions` holds every action's ``(A, R, k, k)`` channels, `slot_weights` is ``(A, k)`` and `bindings` is ``(S, A, K, k)``. Slots are weighted as in the forward pass, so an inactive slot restricts nothing, and the product over cells is relaxed by the same `tau`.
    """
    lifted = lift(adjacency, bindings)
    weight_pairs = (slot_weights[:, :, None] * slot_weights[:, None, :]).unsqueeze(1)
    required = (preconditions.positive * weight_pairs)[None, :, None]
    forbidden = (preconditions.negative * weight_pairs)[None, :, None]
    # As in the forward pass: an atom satisfies the preconditions unless it is required and absent, or forbidden and present.
    satisfaction = ((1 - required * (1 - lifted)) * (1 - forbidden * lifted)).flatten(
        start_dim=3
    )
    return soft_conjunction(satisfaction, dim=-1, tau=tau)


def admitted_fraction(
    adjacency: torch.Tensor,
    preconditions: Channels,
    slot_weights: torch.Tensor,
    samples: int,
    tau: float = 1.0,
) -> torch.Tensor:
    """``(S, A)``: per state and action, the `admission` of `samples` uniform ground actions, averaged. One minus it is the fraction the preconditions reject."""
    num_states, _, _, num_objects = adjacency.shape
    num_actions, _, num_slots, _ = preconditions.positive.shape
    bindings = sample_bindings(
        num_states, num_actions, samples, num_objects, num_slots, adjacency.device
    )
    return admission(adjacency, preconditions, slot_weights, bindings, tau).mean(dim=-1)
