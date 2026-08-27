"""The dense multi-relational adjacency tensor: a state is one ``[R, O, O]``, and which cells of a relation's slice are meaningful depends on its arity."""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def valid_cells(arity: int, num_objects: int) -> torch.Tensor | None:
    """Flat indices of the cells a relation of `arity` can occupy: the whole matrix if binary, the diagonal if unary, None otherwise."""
    if arity == 2:
        return torch.arange(num_objects * num_objects)
    if arity == 1:
        return torch.arange(num_objects) * (num_objects + 1)
    return None


def adj_to_pyg_data(adj: torch.Tensor, num_objects: int) -> Data:
    """``[R, O, O]`` to a PyG graph, one edge per non-zero with its relation id."""
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
    """``[B, R, O, O]`` from one PyG batch's flat edge list. Every graph describes the same instance, so a node's graph and position follow from its global index."""
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
