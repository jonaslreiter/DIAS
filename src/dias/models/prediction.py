"""Predicting the successor state ``A' = A + p_pre * delta`` (paper, Sec. 4.1 to 4.3)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .effects import MlpEffects, StaticEffects
from .outputs import Channels, SchemaPrediction
from .preconditions import Preconditions, soft_conjunction
from .selection import SlotSelector


def project(lifted: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
    """A tensor over slot pairs as one over object pairs, ``S^T P S``."""
    return torch.einsum("bai,brac,bcj->brij", selection, lifted, selection)


class SuccessorPredictor(nn.Module):
    """Binds the action's slots to objects, grounds its lifted schema on them and applies it to the state."""

    def __init__(
        self,
        relation_arities: Sequence[int],
        num_actions: int,
        num_slots: int,
        object_dim: int,
        force_action_args: bool = False,
        per_action_slots: bool = False,
        gate_preconditions: bool = True,
        action_arities: Sequence[int] | None = None,
        mlp_effects: bool = False,
        use_slot_weights: bool = True,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_slots = num_slots
        self.num_relations = len(relation_arities)
        self.gate_preconditions = gate_preconditions

        self.selector = SlotSelector(
            num_actions=num_actions,
            num_slots=num_slots,
            object_dim=object_dim,
            action_arities=action_arities,
            per_action_slots=per_action_slots,
            force_action_args=force_action_args,
            use_slot_weights=use_slot_weights,
        )
        self.preconditions = Preconditions(num_actions, relation_arities, num_slots)
        effects = MlpEffects if mlp_effects else StaticEffects
        self.effects = effects(num_actions, relation_arities, num_slots)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        y_action: torch.Tensor,
        y_action_args: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> SchemaPrediction:
        # Sinkhorn runs in float32.
        node_embeddings = node_embeddings.float()
        adjacency = adjacency.float()
        batch, _, _, num_objects = adjacency.shape

        selection, correspondence = self.selector(
            node_embeddings.view(batch, num_objects, -1), y_action, y_action_args
        )

        preconditions = Channels.from_logits(self.preconditions(y_action))
        required = project(preconditions.positive, selection)
        forbidden = project(preconditions.negative, selection)
        satisfaction = (
            (1 - required * (1 - adjacency)) * (1 - forbidden * adjacency)
        ).flatten(start_dim=1)
        fulfilled = soft_conjunction(satisfaction, dim=-1, tau=tau)

        effects = Channels.from_logits(self.effects(y_action, adjacency, selection))
        delta = self.delta(adjacency, effects, selection)

        if self.gate_preconditions:
            successor = adjacency + fulfilled[:, None, None, None] * delta
        else:
            successor = adjacency + delta

        return SchemaPrediction(
            adjacency=successor,
            input_adjacency=adjacency,
            delta=delta,
            effects=effects,
            preconditions=preconditions,
            selection=selection,
            correspondence=correspondence,
            precondition_satisfaction=fulfilled,
            precondition_product=satisfaction.prod(dim=-1).detach(),
        )

    def apply(
        self,
        adjacency: torch.Tensor,
        selection: torch.Tensor,
        y_action: torch.Tensor,
    ) -> torch.Tensor:
        """The successor state for a known binding, preconditions not consulted."""
        effects = Channels.from_logits(self.effects(y_action, adjacency, selection))
        return adjacency + self.delta(adjacency, effects, selection)

    def delta(
        self,
        adjacency: torch.Tensor,
        effects: Channels,
        selection: torch.Tensor,
    ) -> torch.Tensor:
        """The change the action makes: add where absent, delete where present."""
        added = project(effects.positive, selection)
        deleted = project(effects.negative, selection)
        return (1 - adjacency) * added - adjacency * deleted
