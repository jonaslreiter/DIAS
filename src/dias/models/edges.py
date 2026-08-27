"""How the encoder's edge types are numbered, and the edges themselves."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeTypeLayout(nn.Module):
    """The one place that knows what an edge type id means.

    The encoder reads one graph per transition, whose edges carry a type from four consecutive bands. Over `R` relations and `A` action-argument positions:

    band | edge types
    --- | ---
    `[0, R)` | the state's atoms
    `[R, 2R)` | the same, reversed
    `[2R, 2R + A)` | action self-loops: which argument of the action this object was passed as
    `[2R + A, 2R + A + 4R)` | what the transition changed: deleted, deleted reversed, added, added reversed

    The last band exists only with `successor_relations`, which shows the encoder the answer.
    """

    def __init__(
        self,
        num_relations: int,
        num_actions: int,
        action_arg_positions: Sequence[Sequence[int]] | None = None,
        successor_relations: bool = False,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.successor_relations = successor_relations

        # One relation per used (action, argument position) pair.
        positions = action_arg_positions or [[] for _ in range(num_actions)]
        width = max((max(p) + 1 for p in positions if p), default=1)
        lookup = torch.full((num_actions, width), -1, dtype=torch.long)
        self.num_action_relations = 0
        for action, argument_positions in enumerate(positions):
            for position in argument_positions:
                lookup[action, position] = self.num_action_relations
                self.num_action_relations += 1
        # Derived from the constructor arguments, so it is not checkpointed.
        self.register_buffer("_argument_relation", lookup, persistent=False)

    @property
    def num_edge_types(self) -> int:
        return (
            2 * self.num_relations
            + self.num_action_relations
            + (4 * self.num_relations if self.successor_relations else 0)
        )

    def forward(self, batch_s1, batch_s2=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Every edge the encoder sees, as one ``(edge_index, edge_type)`` pair."""
        num_objects = batch_s1.batch.size(0) // batch_s1.y_action.size(0)
        parts = [
            (batch_s1.edge_index, batch_s1.edge_attr),
            (batch_s1.edge_index.flip(0), batch_s1.edge_attr + self.num_relations),
            self._action_edges(batch_s1.y_action, batch_s1.y_action_args, num_objects),
        ]
        if self.successor_relations and batch_s2 is not None:
            parts.append(self._change_edges(batch_s1, batch_s2))
        return (
            torch.cat([index for index, _ in parts], dim=1),
            torch.cat([types for _, types in parts], dim=0),
        )

    def _action_edges(
        self,
        y_action: torch.Tensor,
        y_action_args: torch.Tensor,
        num_objects: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A self-loop on each object the action was applied to, typed by the argument position it filled."""
        width = self._argument_relation.size(1)
        arguments = y_action_args[:, :width]
        if arguments.size(1) < width:
            arguments = F.pad(arguments, (0, width - arguments.size(1)), value=-1)

        samples, positions = (arguments >= 0).nonzero(as_tuple=True)
        relations = self._argument_relation[y_action[samples], positions]
        # A position the action does not declare as an argument has no relation.
        used = relations >= 0
        samples, positions, relations = samples[used], positions[used], relations[used]

        nodes = samples * num_objects + arguments[samples, positions]
        return (
            torch.stack([nodes, nodes], dim=0),
            2 * self.num_relations + relations,
        )

    def _change_edges(self, batch_s1, batch_s2) -> tuple[torch.Tensor, torch.Tensor]:
        """The atoms the transition deleted and added, in both directions."""
        total_nodes = batch_s1.batch.size(0)
        before = _edge_codes(
            batch_s1.edge_index, batch_s1.edge_attr, total_nodes, self.num_relations
        )
        after = _edge_codes(
            batch_s2.edge_index, batch_s2.edge_attr, total_nodes, self.num_relations
        )
        base = 2 * self.num_relations + self.num_action_relations

        indices: list[torch.Tensor] = []
        types: list[torch.Tensor] = []
        for changed, batch, offset in (
            (~torch.isin(before, after), batch_s1, base),
            (~torch.isin(after, before), batch_s2, base + 2 * self.num_relations),
        ):
            if not changed.any():
                continue
            edge_index = batch.edge_index[:, changed]
            edge_type = batch.edge_attr[changed]
            indices += [edge_index, edge_index.flip(0)]
            types += [
                edge_type + offset,
                edge_type + offset + self.num_relations,
            ]

        if not indices:
            return (
                batch_s1.edge_index.new_empty((2, 0)),
                batch_s1.edge_attr.new_empty((0,)),
            )
        return torch.cat(indices, dim=1), torch.cat(types, dim=0)


def _edge_codes(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    num_relations: int,
) -> torch.Tensor:
    """One integer per atom, so two states can be compared with `isin`."""
    if edge_type.numel() == 0:
        return edge_type.new_empty((0,), dtype=torch.long)
    source, target = edge_index
    return (
        source.long() * num_nodes + target.long()
    ) * num_relations + edge_type.long()
