import torch
import torch.nn as nn
import torch.nn.functional as F

from dias.models.utils import batched_sinkhorn_d1d2
from dias.utils import gmean


class PredictionModule(nn.Module):
    """Slot-based graph-edit predictor.

    Builds slot-to-node attention via Sinkhorn normalisation, then combines
    learnable per-action add/delete/precondition tensors with the selection to
    produce a predicted next-state adjacency ``A' = A + Δ`` (gated by
    precondition fulfilment when ``learn_preconditions=True``).
    """

    def __init__(
        self,
        num_objects: int,
        arity_relations: list[int],
        num_actions: int,
        num_slots: int,
        object_dim: int,
        force_action_args: bool = False,
        per_action_slots: bool = False,
        learn_preconditions: bool = True,
        action_arities: list[int] | None = None,
        gumbel_slot_noise: bool = False,
        gumbel_softmax: bool = False,
        gmean_offset: float = 0.0,
    ):
        super().__init__()
        self.num_objects = num_objects
        self.arity_relations = arity_relations
        self.num_actions = num_actions
        self.num_slots = num_slots
        self.object_dim = object_dim
        self.force_action_args = force_action_args
        self.per_action_slots = per_action_slots
        self.learn_preconditions = learn_preconditions
        self.gumbel_slot_noise = gumbel_slot_noise
        self.gumbel_softmax = gumbel_softmax
        self.gmean_offset = gmean_offset
        self.num_relations = len(arity_relations)

        action_arities_t = (
            torch.tensor(action_arities, dtype=torch.long)
            if action_arities is not None
            else torch.full((num_actions,), num_slots, dtype=torch.long)
        )
        self.register_buffer("action_arities_t", action_arities_t)

        self.slot_encodings = nn.Parameter(
            0.1 * torch.randn(num_actions, num_slots, object_dim)
        )
        self.slot_weight_logits = nn.Parameter(torch.zeros(num_actions, num_slots))
        self.preconditions = nn.Parameter(
            0.1 * torch.randn(num_actions, self.num_relations, num_slots, num_slots, 3)
        )
        self.effects = nn.Parameter(
            0.1 * torch.randn(num_actions, self.num_relations, num_slots, num_slots, 3)
        )

        if per_action_slots:
            with torch.no_grad():
                unused = torch.arange(num_slots).unsqueeze(
                    0
                ) >= action_arities_t.unsqueeze(1)
                self.slot_weight_logits.masked_fill_(unused, -torch.inf)

        # Unary relations: only diagonal slot pairs can be non-zero.
        relation_is_unary = torch.tensor(
            [a == 1 for a in arity_relations], dtype=torch.bool
        )
        diag = torch.eye(num_slots, dtype=torch.bool).view(
            1, 1, num_slots, num_slots, 1
        )
        unary_offdiag = relation_is_unary.view(1, self.num_relations, 1, 1, 1) & ~diag
        with torch.no_grad():
            for p in (self.preconditions, self.effects):
                p.masked_fill_(unary_offdiag, -torch.inf)
                p[..., 0].masked_fill_(unary_offdiag.squeeze(-1), 0)

        self._last_selection = None
        self._last_y_action = None

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
        y_action: torch.Tensor = None,
        y_action_args: torch.Tensor | None = None,
        adj_dense: torch.Tensor | None = None,
        gmean_temperature: float = 1.0,
    ):
        # Sinkhorn / log-space are float16-unsafe under AMP.
        node_embeddings = node_embeddings.float()

        B = y_action.size(0)
        n_nodes = self.num_objects
        k = self.num_slots

        if adj_dense is not None:
            adj_batch = adj_dense.float()
        else:
            adj_batch = torch.zeros(
                B,
                self.num_relations,
                n_nodes,
                n_nodes,
                device=node_embeddings.device,
                dtype=torch.float32,
            )
            graph_ids = batch_idx[edge_index[0]]
            local_src = edge_index[0] - graph_ids * n_nodes
            local_dst = edge_index[1] - graph_ids * n_nodes
            adj_batch[graph_ids, edge_type, local_src, local_dst] = 1.0

        embeddings_batch = node_embeddings.view(B, n_nodes, -1)
        slot_encodings = self.slot_encodings[y_action]  # (B, k, D)
        attention = torch.bmm(slot_encodings, embeddings_batch.transpose(1, 2)) / (
            self.object_dim**0.5
        )

        if self.training and self.gumbel_slot_noise:
            gumbel_noise = (
                torch.distributions.gumbel.Gumbel(0.0, 1.0)
                .sample(attention.shape)
                .to(attention.device)
            )
            logits = attention + gumbel_noise
        else:
            logits = attention

        # Extended logit matrix: (B, k+1, n_nodes) — slack row, no slack column.
        logits_ext = torch.zeros(
            (B, k + 1, n_nodes), device=attention.device, dtype=torch.float32
        )
        logits_ext[:, :-1, :] = logits

        if self.per_action_slots:
            arity_per_sample = self.action_arities_t[y_action]
            unused_mask = torch.arange(k, device=logits.device).unsqueeze(
                0
            ) >= arity_per_sample.unsqueeze(1)
            sinkhorn_row_mask = F.pad(unused_mask, (0, 1), value=False)  # (B, k+1)
        else:
            unused_mask = None
            sinkhorn_row_mask = None

        # Force selection of known action arguments by hard-masking logits.
        if self.force_action_args and y_action_args is not None:
            known = y_action_args >= 0  # (B, k)
            if known.any():
                b_idx, s_idx = known.nonzero(as_tuple=True)
                obj_idx = y_action_args[b_idx, s_idx]

                row_mask = torch.zeros(
                    B, logits_ext.size(1), 1, device=logits_ext.device, dtype=torch.bool
                )
                row_mask[b_idx, s_idx, 0] = True
                logits_ext.masked_fill_(row_mask, -torch.inf)

                col_mask = torch.zeros(
                    B, logits_ext.size(2), device=logits_ext.device, dtype=torch.bool
                )
                col_mask[b_idx, obj_idx] = True
                logits_ext.masked_fill_(col_mask.unsqueeze(1), -torch.inf)

                logits_ext[b_idx, s_idx, obj_idx] = 0.0

        selection_all = batched_sinkhorn_d1d2(logits_ext, row_mask=sinkhorn_row_mask)

        if self.per_action_slots:
            selection = selection_all[:, :-1, :].clone()
            selection[unused_mask] = 0.0
        else:
            selection = selection_all[:, :-1, :]
            slot_weights = torch.sigmoid(self.slot_weight_logits[y_action])  # (B, k)
            # Don't downweight forced selections.
            if self.force_action_args and y_action_args is not None:
                forced = y_action_args >= 0
                if forced.any():
                    if forced.shape[1] < slot_weights.shape[1]:
                        forced = F.pad(
                            forced, (0, slot_weights.shape[1] - forced.shape[1])
                        )
                    slot_weights = torch.where(
                        forced, torch.ones_like(slot_weights), slot_weights
                    )
            selection = selection * slot_weights.unsqueeze(-1)

        if self.training:
            self._last_selection = selection.detach().cpu()
            self._last_y_action = y_action.detach().cpu()

        if self.training and self.gumbel_softmax:
            preconditions = F.gumbel_softmax(
                self.preconditions[y_action], hard=False, dim=-1
            )
            effects = F.gumbel_softmax(self.effects[y_action], hard=False, dim=-1)
        else:
            preconditions = F.softmax(self.preconditions[y_action], dim=-1)
            effects = F.softmax(self.effects[y_action], dim=-1)

        ignore_effects, add_effects, del_effects = (
            effects[..., 0],
            effects[..., 1],
            effects[..., -1],
        )
        ignore_precond, pos_precond, neg_precond = (
            preconditions[..., 0],
            preconditions[..., 1],
            preconditions[..., -1],
        )

        def project(slots):
            return torch.einsum("bai,brac,bcj->brij", selection, slots, selection)

        add_proj = project(add_effects)
        del_proj = project(del_effects)
        pos_proj = project(pos_precond)
        neg_proj = project(neg_precond)

        precond_fulfilled = (
            (1 - pos_proj * (1 - adj_batch)) * (1 - neg_proj * adj_batch)
        ).flatten(start_dim=1)
        pre = gmean(
            precond_fulfilled, dim=-1, temperature=gmean_temperature + self.gmean_offset
        )
        self._last_pre_mean = pre.mean().detach()
        self._last_pre_prod_mean = precond_fulfilled.prod(dim=-1).mean().detach()

        delta = (1 - adj_batch) * add_proj - adj_batch * del_proj
        if self.learn_preconditions:
            adj_out_probs = adj_batch + pre[:, None, None, None] * delta
        else:
            adj_out_probs = adj_batch + delta

        return (
            adj_out_probs,
            adj_batch,
            ignore_effects,
            add_effects,
            del_effects,
            ignore_precond,
            pos_precond,
            neg_precond,
            selection,
            logits_ext,
            delta,
        )
