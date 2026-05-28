import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv


class DynamicsModelRGCNST(nn.Module):
    """6-layer FastRGCN with LayerNorm + SiLU between layers."""

    def __init__(self, latent_dim: int, number_relations: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.number_relations = number_relations

        def conv(in_d, out_d):
            return FastRGCNConv(in_d, out_d, number_relations, aggr="add")

        self.conv1 = conv(latent_dim, 32)
        self.conv2 = conv(32, 64)
        self.conv3 = conv(64, 64)
        self.conv4 = conv(64, 64)
        self.conv5 = conv(64, 64)
        self.conv6 = conv(64, latent_dim)

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
