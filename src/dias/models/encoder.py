"""The R-GCN computing the object embeddings, the keys `K` (paper, Sec. 4.1 and App. B.2)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv

DEFAULT_HIDDEN_DIMS = (32, 64, 64, 64, 64)


class RelationalEncoder(nn.Module):
    """Relational graph convolutions, one per hidden width plus a last one without bias back to `object_dim`, each but the last followed by SiLU and layer normalization."""

    def __init__(
        self,
        object_dim: int,
        num_edge_types: int,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        random_node_features: bool = True,
    ):
        super().__init__()
        self.object_dim = object_dim
        self.random_node_features = random_node_features

        widths = [object_dim, *hidden_dims, object_dim]
        last = len(widths) - 2
        self.convs = nn.ModuleList(
            FastRGCNConv(
                in_dim,
                out_dim,
                num_edge_types,
                aggr="add",
                bias=index != last,
            )
            for index, (in_dim, out_dim) in enumerate(zip(widths, widths[1:]))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for dim in hidden_dims)

    def initial_features(self, num_nodes: int, device=None) -> torch.Tensor:
        """The initial node features: the first half of the dimensions drawn from ``0.1 * N(0, 1)``, the rest zero, or all zero without `random_node_features`."""
        features = torch.zeros(
            num_nodes, self.object_dim, device=device, dtype=torch.float32
        )
        if self.random_node_features:
            half = self.object_dim // 2
            features[:, :half] = 0.1 * torch.randn(
                num_nodes, half, device=device, dtype=torch.float32
            )
        return features

    def forward(self, x, edge_index, edge_type):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.silu(conv(x, edge_index, edge_type)))
        return self.convs[-1](x, edge_index, edge_type)
