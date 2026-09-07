"""Turning a state's graph into one embedding per object."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv

DEFAULT_HIDDEN_DIMS = (32, 64, 64, 64, 64)


class RelationalEncoder(nn.Module):
    """Relational graph convolutions with LayerNorm + SiLU between layers, one per hidden dimension plus a final bias-free one back down to `object_dim`. Per-edge-type weights are what distinguish a relation from its reverse and both from an action self-loop."""

    def __init__(
        self,
        object_dim: int,
        num_edge_types: int,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        random_node_embeddings: bool = True,
    ):
        super().__init__()
        self.object_dim = object_dim
        self.random_node_embeddings = random_node_embeddings

        widths = [object_dim, *hidden_dims, object_dim]
        last = len(widths) - 2
        self.convs = nn.ModuleList(
            FastRGCNConv(
                in_dim,
                out_dim,
                num_edge_types,
                aggr="add",
                # No bias on the last one: nothing is normalised after it, so a bias there is a per-slot constant that Sinkhorn removes exactly. The loss cannot see it, and Adam turns its round-off gradient into a random walk.
                bias=index != last,
            )
            for index, (in_dim, out_dim) in enumerate(zip(widths, widths[1:]))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for dim in hidden_dims)

    def initial_features(self, num_nodes: int, device=None) -> torch.Tensor:
        """Where every object starts, before any message passing.

        Half the dimensions are seeded with noise: objects carry no features of their own, so two in symmetric positions are otherwise indistinguishable. Off is the `random_node_embeddings` ablation.
        """
        features = torch.zeros(
            num_nodes, self.object_dim, device=device, dtype=torch.float32
        )
        if self.random_node_embeddings:
            half = self.object_dim // 2
            features[:, :half] = 0.1 * torch.randn(
                num_nodes, half, device=device, dtype=torch.float32
            )
        return features

    def forward(self, x, edge_index, edge_type):
        # One norm per hidden layer, so zip leaves the last convolution bare.
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.silu(conv(x, edge_index, edge_type)))
        return self.convs[-1](x, edge_index, edge_type)
