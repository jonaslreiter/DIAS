"""A state as the adjacency tensor ``[R, O, O]``: a unary relation on the diagonal, a binary one everywhere, and its PyG graph."""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def valid_cells(arity: int, num_objects: int) -> torch.Tensor | None:
    """Flat indices of the cells a relation of `arity` occupies, or None for an arity other than 1 or 2."""
    if arity == 2:
        return torch.arange(num_objects * num_objects)
    if arity == 1:
        return torch.arange(num_objects) * (num_objects + 1)
    return None


def adj_to_pyg_data(adj: torch.Tensor, num_objects: int) -> Data:
    """A PyG graph with one edge per atom, typed by its relation."""
    relation, source, target = adj.nonzero(as_tuple=True)
    return Data(
        edge_index=torch.stack([source, target], dim=0),
        edge_attr=relation,
        num_nodes=num_objects,
    )


def dense_from_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    batch_index: torch.Tensor,
    num_relations: int,
    num_objects: int,
) -> torch.Tensor:
    """``[B, R, O, O]`` from a PyG batch of graphs over one instance."""
    adjacency = torch.zeros(
        batch_index.size(0) // num_objects,
        num_relations,
        num_objects,
        num_objects,
        device=edge_index.device,
        dtype=torch.float32,
    )
    graph = batch_index[edge_index[0]]
    source = edge_index[0] - graph * num_objects
    target = edge_index[1] - graph * num_objects
    adjacency[graph, edge_type, source, target] = 1.0
    return adjacency
