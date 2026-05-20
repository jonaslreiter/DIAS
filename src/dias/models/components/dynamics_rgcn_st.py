import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv


class DynamicsModelRGCNST(nn.Module):
    def __init__(
        self,
        latent_dim,
        number_relations,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.number_relations = number_relations

        self.conv1 = FastRGCNConv(
            self.latent_dim,
            32,
            number_relations,
            aggr="add",
        )
        self.conv2 = FastRGCNConv(
            32,
            64,
            number_relations,
            aggr="add",
        )
        self.conv3 = FastRGCNConv(
            64,
            64,
            number_relations,
            aggr="add",
        )
        self.conv4 = FastRGCNConv(
            64,
            64,
            number_relations,
            aggr="add",
        )
        self.conv5 = FastRGCNConv(
            64,
            64,
            number_relations,
            aggr="add",
        )
        self.conv6 = FastRGCNConv(
            64,
            self.latent_dim,
            number_relations,
            aggr="add",
        )

        self.ln1 = nn.LayerNorm(32)
        self.ln2 = nn.LayerNorm(64)
        self.ln3 = nn.LayerNorm(64)
        self.ln4 = nn.LayerNorm(64)
        self.ln5 = nn.LayerNorm(64)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.silu(x)
        x = self.ln1(x)

        x = self.conv2(x, edge_index, edge_type)
        x = F.silu(x)
        x = self.ln2(x)

        x = self.conv3(x, edge_index, edge_type)
        x = F.silu(x)
        x = self.ln3(x)

        x = self.conv4(x, edge_index, edge_type)
        x = F.silu(x)
        x = self.ln4(x)

        x = self.conv5(x, edge_index, edge_type)
        x = F.silu(x)
        x = self.ln5(x)

        x = self.conv6(x, edge_index, edge_type)

        return x
