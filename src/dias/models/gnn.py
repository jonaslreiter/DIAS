import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv


class GNNModule(nn.Module):
    """6-layer FastRGCN with LayerNorm + SiLU between layers."""

    def __init__(self, latent_dim: int, number_relations: int):
        super().__init__()

        self.conv1 = FastRGCNConv(latent_dim, 32, number_relations, aggr="add")
        self.conv2 = FastRGCNConv(32, 64, number_relations, aggr="add")
        self.conv3 = FastRGCNConv(64, 64, number_relations, aggr="add")
        self.conv4 = FastRGCNConv(64, 64, number_relations, aggr="add")
        self.conv5 = FastRGCNConv(64, 64, number_relations, aggr="add")
        self.conv6 = FastRGCNConv(64, latent_dim, number_relations, aggr="add")

        self.ln1 = nn.LayerNorm(32)
        self.ln2 = nn.LayerNorm(64)
        self.ln3 = nn.LayerNorm(64)
        self.ln4 = nn.LayerNorm(64)
        self.ln5 = nn.LayerNorm(64)

    def forward(self, x, edge_index, edge_type):
        for conv, ln in (
            (self.conv1, self.ln1),
            (self.conv2, self.ln2),
            (self.conv3, self.ln3),
            (self.conv4, self.ln4),
            (self.conv5, self.ln5),
        ):
            x = ln(F.silu(conv(x, edge_index, edge_type)))

        return self.conv6(x, edge_index, edge_type)
