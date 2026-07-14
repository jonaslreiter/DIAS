import torch


def batched_sinkhorn_d1d2(
    log_A: torch.Tensor,
    max_iter: int = 10000,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched Sinkhorn D1D2 in log-space (https://hal.science/hal-03768664)."""
    tol = 1.0e-2
    batch_size, np1, _ = log_A.shape
    n = np1 - 1

    log_x = torch.zeros(batch_size, np1, device=log_A.device, dtype=log_A.dtype)
    log_y = torch.zeros(
        batch_size, log_A.size(-1), device=log_A.device, dtype=log_A.dtype
    )
    if row_mask is not None:
        log_x = log_x.masked_fill(row_mask, -torch.inf)

    for _ in range(max_iter):
        log_x_prev, log_y_prev = log_x, log_y
        log_x = -torch.logsumexp(log_A + log_y.unsqueeze(-2), dim=-1)
        log_x[:, n] = 0.0
        if row_mask is not None:
            log_x = log_x.masked_fill(row_mask, -torch.inf)
        log_y = -torch.logsumexp(log_A + log_x.unsqueeze(-1), dim=-2)

        if (
            max(
                (log_x - log_x_prev).abs().max(),
                (log_y - log_y_prev).abs().max(),
            )
            < tol
        ):
            break

    return torch.exp(log_x.unsqueeze(-1) + log_A + log_y.unsqueeze(-2))


def encode_relation_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    num_relations: int,
) -> torch.Tensor:
    if edge_type.numel() == 0:
        return edge_type.new_empty((0,), dtype=torch.long)
    src, dst = edge_index
    return (src.long() * num_nodes + dst.long()) * num_relations + edge_type.long()


def build_relation_change_edges(
    batch_s1,
    batch_s2,
    num_relations: int,
    num_action_relations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    empty_ei = batch_s1.edge_index.new_empty((2, 0))
    empty_et = batch_s1.edge_attr.new_empty((0,))
    if batch_s2 is None:
        return empty_ei, empty_et

    total_nodes = batch_s1.batch.size(0)
    s1_codes = encode_relation_edges(
        batch_s1.edge_index, batch_s1.edge_attr, total_nodes, num_relations
    )
    s2_codes = encode_relation_edges(
        batch_s2.edge_index, batch_s2.edge_attr, total_nodes, num_relations
    )

    deleted = ~torch.isin(s1_codes, s2_codes)
    added = ~torch.isin(s2_codes, s1_codes)
    base = 2 * num_relations + num_action_relations

    ei_parts: list[torch.Tensor] = []
    et_parts: list[torch.Tensor] = []

    if deleted.any():
        edge_index = batch_s1.edge_index[:, deleted]
        edge_type = batch_s1.edge_attr[deleted]
        ei_parts.extend((edge_index, edge_index.flip(0)))
        et_parts.extend((edge_type + base, edge_type + base + num_relations))

    if added.any():
        edge_index = batch_s2.edge_index[:, added]
        edge_type = batch_s2.edge_attr[added]
        ei_parts.extend((edge_index, edge_index.flip(0)))
        et_parts.extend(
            (edge_type + base + 2 * num_relations, edge_type + base + 3 * num_relations)
        )

    if not ei_parts:
        return empty_ei, empty_et
    return torch.cat(ei_parts, dim=1), torch.cat(et_parts, dim=0)
