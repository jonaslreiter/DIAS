from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv

DEFAULT_HIDDEN_DIMS = (32, 64, 64, 64, 64)


class GNNModule(nn.Module):
    """Relational graph convolutions with LayerNorm + SiLU between layers.

    One convolution per hidden dimension, plus a final one back down to
    ``latent_dim``.
    """

    def __init__(
        self,
        latent_dim: int,
        number_relations: int,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
    ):
        super().__init__()
        self.hidden_dims = tuple(hidden_dims)

        # Attributes are named conv1.. / ln1.. rather than held in ModuleLists so
        # that existing checkpoints keep loading; the layout changes in a later
        # step, together with its migration.
        widths = [latent_dim, *self.hidden_dims]
        for index, (in_dim, out_dim) in enumerate(zip(widths, widths[1:]), start=1):
            setattr(
                self,
                f"conv{index}",
                FastRGCNConv(in_dim, out_dim, number_relations, aggr="add"),
            )
        setattr(
            self,
            f"conv{len(self.hidden_dims) + 1}",
            FastRGCNConv(
                self.hidden_dims[-1], latent_dim, number_relations, aggr="add"
            ),
        )
        for index, dim in enumerate(self.hidden_dims, start=1):
            setattr(self, f"ln{index}", nn.LayerNorm(dim))

    @property
    def _hidden_layers(self):
        return [
            (getattr(self, f"conv{i}"), getattr(self, f"ln{i}"))
            for i in range(1, len(self.hidden_dims) + 1)
        ]

    def forward(self, x, edge_index, edge_type):
        for conv, ln in self._hidden_layers:
            x = ln(F.silu(conv(x, edge_index, edge_type)))

        return getattr(self, f"conv{len(self.hidden_dims) + 1}")(
            x, edge_index, edge_type
        )
