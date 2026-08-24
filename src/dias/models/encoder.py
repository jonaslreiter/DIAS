"""Turning a state's graph into one embedding per object."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv

DEFAULT_HIDDEN_DIMS = (32, 64, 64, 64, 64)


class RelationalEncoder(nn.Module):
    """Relational graph convolutions with LayerNorm + SiLU between layers.

    One convolution per hidden dimension, plus a final, bias-free one back down
    to `latent_dim`. Each convolution has its own weights per edge type, which is
    what lets the encoder distinguish a relation from its reverse and both from
    an action self-loop.
    """

    def __init__(
        self,
        latent_dim: int,
        num_edge_types: int,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        random_node_embeddings: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.random_node_embeddings = random_node_embeddings

        widths = [latent_dim, *hidden_dims, latent_dim]
        last = len(widths) - 2
        self.convs = nn.ModuleList(
            FastRGCNConv(
                in_dim,
                out_dim,
                num_edge_types,
                aggr="add",
                # The last convolution has no bias. Nothing is normalised after
                # it, so its bias would add the same vector to every object;
                # the slot scores are a plain inner product, so that arrives at
                # the assignment as a per-slot constant, which Sinkhorn's row
                # scaling removes exactly. The loss could not see it, and Adam
                # would turn its round-off gradient into a random walk.
                bias=index != last,
            )
            for index, (in_dim, out_dim) in enumerate(zip(widths, widths[1:]))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for dim in hidden_dims)

    def initial_features(self, num_nodes: int, device=None) -> torch.Tensor:
        """Where every object starts, before any message passing.

        Half the dimensions are seeded with noise, because objects carry no
        features of their own: two objects in symmetric positions are otherwise
        indistinguishable to the encoder, and no assignment can separate them.
        Turning this off is the `random_node_embeddings` ablation.
        """
        features = torch.zeros(
            num_nodes, self.latent_dim, device=device, dtype=torch.float32
        )
        if self.random_node_embeddings:
            half = self.latent_dim // 2
            features[:, :half] = 0.1 * torch.randn(
                num_nodes, half, device=device, dtype=torch.float32
            )
        return features

    def forward(self, x, edge_index, edge_type):
        # One norm per hidden layer, so zip leaves the last convolution bare.
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.silu(conv(x, edge_index, edge_type)))
        return self.convs[-1](x, edge_index, edge_type)
