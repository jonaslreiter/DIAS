"""Applying one lifted action to one state, differentiably."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .effects import StateDependentEffects, StaticEffects
from .outputs import Channels, SchemaPrediction
from .preconditions import Preconditions, soft_conjunction
from .selection import SlotSelector


def project(lifted: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
    """Read a tensor over slot pairs as one over object pairs."""
    return torch.einsum("bai,brac,bcj->brij", selection, lifted, selection)


class PredictionModule(nn.Module):
    """Slot-based graph-edit predictor.

    The action's slots are bound to the state's objects, its lifted effects and preconditions are projected onto that binding, and the resulting change is applied to the state: ``A' = A + pre * delta``, where ``pre`` is how well the preconditions hold (when ``learn_preconditions=True``).
    """

    def __init__(
        self,
        arity_relations: Sequence[int],
        num_actions: int,
        num_slots: int,
        object_dim: int,
        force_action_args: bool = False,
        per_action_slots: bool = False,
        learn_preconditions: bool = True,
        action_arities: Sequence[int] | None = None,
        state_dependent_effects: bool = False,
        use_slot_weights: bool = True,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_slots = num_slots
        self.num_relations = len(arity_relations)
        self.learn_preconditions = learn_preconditions

        self.selector = SlotSelector(
            num_actions=num_actions,
            num_slots=num_slots,
            object_dim=object_dim,
            action_arities=action_arities,
            per_action_slots=per_action_slots,
            force_action_args=force_action_args,
            use_slot_weights=use_slot_weights,
        )
        self.preconditions = Preconditions(num_actions, arity_relations, num_slots)
        effect_source = (
            StateDependentEffects if state_dependent_effects else StaticEffects
        )
        self.effects = effect_source(num_actions, arity_relations, num_slots)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        y_action: torch.Tensor,
        y_action_args: torch.Tensor | None = None,
        gmean_temperature: float = 1.0,
    ) -> SchemaPrediction:
        # Sinkhorn / log-space are float16-unsafe under AMP.
        node_embeddings = node_embeddings.float()
        adjacency = adjacency.float()
        batch, _, _, num_objects = adjacency.shape

        selection, selection_logits = self.selector(
            node_embeddings.view(batch, num_objects, -1), y_action, y_action_args
        )

        # Do the preconditions hold?
        preconditions = Channels.from_logits(self.preconditions(y_action))
        required = project(preconditions.positive, selection)
        forbidden = project(preconditions.negative, selection)
        # An atom satisfies the preconditions unless it is required and absent, or forbidden and present.
        satisfaction = (
            (1 - required * (1 - adjacency)) * (1 - forbidden * adjacency)
        ).flatten(start_dim=1)
        fulfilled = soft_conjunction(
            satisfaction, dim=-1, temperature=gmean_temperature
        )

        # What would the action change?
        effects = Channels.from_logits(self.effects(y_action, adjacency, selection))
        delta = self.delta(adjacency, effects, selection)

        if self.learn_preconditions:
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
            selection_logits=selection_logits,
            precondition_satisfaction=fulfilled,
            precondition_product=satisfaction.prod(dim=-1).detach(),
        )

    def apply(
        self,
        adjacency: torch.Tensor,
        selection: torch.Tensor,
        y_action: torch.Tensor,
    ) -> torch.Tensor:
        """The successor state, for a caller that already knows the binding.

        The preconditions are not consulted. A caller holding a binding got it by grounding the learned schema through PDDL, which is where applicability was decided; asking again here would decide it twice, by two different means.
        """
        effects = Channels.from_logits(self.effects(y_action, adjacency, selection))
        return adjacency + self.delta(adjacency, effects, selection)

    def delta(
        self,
        adjacency: torch.Tensor,
        effects: Channels,
        selection: torch.Tensor,
    ) -> torch.Tensor:
        """What the action changes: add where absent, delete where present."""
        added = project(effects.positive, selection)
        deleted = project(effects.negative, selection)
        return (1 - adjacency) * added - adjacency * deleted
