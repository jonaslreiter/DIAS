import torch
import torch.nn as nn
import torch.nn.functional as F

from dias.utils import gmean


def _batched_sinkhorn_d1d2(
    log_A: torch.Tensor,
    max_iter: int = 10000,
    temperature: float = 1.0,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Batched Sinkhorn D1D2 algorithm in log-space,
    based on https://hal.science/hal-03768664/document.

    Iterates until the max-norm change in the dual scaling vectors
    (log_x, log_y) falls below ``tol``, or ``max_iter`` iterations
    are reached.

    log_A: (B, n+1, m+1) logit matrix (arbitrary real values).
    max_iter: maximum number of Sinkhorn iterations (safety bound).
    tol: convergence tolerance on the max-norm change of the dual variables.
    temperature: controls sharpness of the output; values < 1 sharpen,
                 values > 1 soften the assignment.
    returns: doubly-stochastic-like matrix in probability space,
             shape (B, n+1, m+1).
    """
    tol = 1.0e-2
    B_size, np1, mp1 = log_A.shape
    n, m = np1 - 1, mp1 - 1

    log_A = log_A / temperature

    device, dtype = log_A.device, log_A.dtype

    # Log-space scaling vectors (log(x), log(y)), initialised to 0 (i.e. x=y=1)
    log_x = torch.zeros(B_size, np1, device=device, dtype=dtype)
    log_y = torch.zeros(B_size, mp1, device=device, dtype=dtype)

    # Exclude masked rows: set their dual variable to -inf so they
    # contribute nothing to column normalisation and produce zero output.
    if row_mask is not None:
        log_x = log_x.masked_fill(row_mask, -torch.inf)

    for idx in range(max_iter):
        log_x_prev = log_x
        log_y_prev = log_y

        # Row normalisation: log_x_i = -logsumexp_j(log_A[i,j] + log_y[j])
        log_x = -torch.logsumexp(log_A + log_y.unsqueeze(-2), dim=-1)
        log_x[:, n] = 0.0  # slack row stays unscaled
        if row_mask is not None:
            log_x = log_x.masked_fill(row_mask, -torch.inf)

        # Column normalisation: log_y_j = -logsumexp_i(log_A[i,j] + log_x[i])
        log_y = -torch.logsumexp(log_A + log_x.unsqueeze(-1), dim=-2)
        # log_y[:, m] = 0.0  # slack column stays unscaled

        ths = torch.max(
            (log_x - log_x_prev).abs().max(),
            (log_y - log_y_prev).abs().max(),
        )
        converged = ths < tol
        if converged:
            break

    # Result in probability space: exp(log_x_i + log_A[i,j] + log_y_j)
    return torch.exp(log_x.unsqueeze(-1) + log_A + log_y.unsqueeze(-2))


class SinkhornGraphEdit(nn.Module):
    """
    Self-contained graph-edit predictor using Sinkhorn-based slot selection.

    Given node embeddings, edge structure, and an action index per sample,
    this module:
      1. Builds the current dense adjacency from the sparse edge representation.
      2. Computes slot→node attention using learnable action-slot parameters.
      3. Applies Gumbel noise (during training) and Sinkhorn normalisation to
         obtain a soft permutation / selection matrix.
      4. Combines the selection with a learnable per-action add/delete tensor
         to produce a delta matrix.
      5. Returns the predicted next-state adjacency via an XOR-like update:
             A' = A + Δ - 2·A·Δ

    Args:
        number_objects (int): Maximum number of nodes per graph.
        object_dim (int): Dimensionality of node embeddings.
        num_relations (int): Number of edge / relation types.
        num_actions (int): Number of discrete actions.
        number_of_slots (int): Number of action-parameter slots (k).
        tau (float): Temperature for Gumbel noise and Sinkhorn sharpness.
    """

    def __init__(
        self,
        num_objects: int,
        arity_relations: list[int],
        num_actions: list[int],
        num_slots: list[int],
        object_dim: int,
        tau: float = 1.0,
        force_action_args: bool = False,
        per_action_slots: bool = False,
        learn_preconditions: bool = True,
        action_arities: list[int] | None = None,
        gumbel_sinkhorn: bool = False,
        gumbel_softmax: bool = False,
        gmean_offset: float = 0.0,
    ):
        super().__init__()

        self.num_objects = num_objects
        self.arity_relations = arity_relations
        self.num_actions = num_actions
        self.num_slots = num_slots
        self.object_dim = object_dim
        self.tau = tau
        self.force_action_args = force_action_args
        self.per_action_slots = per_action_slots
        self.learn_preconditions = learn_preconditions
        self.gumbel_sinkhorn = gumbel_sinkhorn
        self.gumbel_softmax = gumbel_softmax
        self.gmean_offset = gmean_offset

        # Register as buffer so it's saved / moved with the model
        _action_arities = (
            torch.tensor(action_arities, dtype=torch.long)
            if action_arities is not None
            else torch.full((num_actions,), num_slots, dtype=torch.long)
        )
        self.register_buffer("action_arities_t", _action_arities)

        self.num_relations = len(arity_relations)

        self.slot_encodings = 0.1 * torch.randn(num_actions, num_slots, object_dim)
        self.slot_encodings = nn.Parameter(self.slot_encodings)
        # nn.init.kaiming_uniform_(self.slot_encodings)

        # Learned slot weights: action-dependent probability for each slot.
        # Applied (via sigmoid) to the selection matrix when per_action_slots=False.
        self.slot_weight_logits = nn.Parameter(torch.zeros(num_actions, num_slots))

        # When per_action_slots is True, mask out slots beyond each action's arity
        if self.per_action_slots:
            with torch.no_grad():
                slot_indices = torch.arange(num_slots).unsqueeze(0)  # (1, k)
                unused_mask = slot_indices >= self.action_arities_t.unsqueeze(
                    1
                )  # (num_actions, k)
                self.slot_weight_logits.masked_fill_(unused_mask, -torch.inf)

        self.preconditions = 0.1 * torch.randn(
            num_actions, self.num_relations, num_slots, num_slots, 3
        )
        self.preconditions = nn.Parameter(self.preconditions)
        # nn.init.kaiming_uniform_(self.preconditions)

        self.effects = 0.1 * torch.randn(
            num_actions, self.num_relations, num_slots, num_slots, 3
        )
        self.effects = nn.Parameter(self.effects)
        # nn.init.kaiming_uniform_(self.effects)

        relation_is_unary = torch.tensor(
            [arity == 1 for arity in arity_relations], dtype=torch.bool
        )
        diag_slot_pairs = torch.eye(num_slots, dtype=torch.bool).view(
            1, 1, num_slots, num_slots, 1
        )
        unary_offdiag_mask = (
            relation_is_unary.view(1, self.num_relations, 1, 1, 1) & ~diag_slot_pairs
        )

        with torch.no_grad():
            self.preconditions.masked_fill_(unary_offdiag_mask, -torch.inf)
            self.preconditions[..., 0].masked_fill_(unary_offdiag_mask.squeeze(-1), 0)
            self.effects.masked_fill_(unary_offdiag_mask, -torch.inf)
            self.effects[..., 0].masked_fill_(unary_offdiag_mask.squeeze(-1), 0)

        # Cache for epoch-end logging (populated during training)
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
        tau: float = 1.0,
        sinkhorn_temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: (Total_Nodes, D) or (B, N, D) – node embeddings
                (e.g. from a dynamics GNN).
            edge_index: (2, Total_Edges)  – sparse edge index.
                Not required when ``adj_dense`` is provided.
            edge_type: (Total_Edges,)  – integer edge / relation types.
                Not required when ``adj_dense`` is provided.
            batch_idx: (Total_Nodes,)  – batch assignment vector.
                Not required when ``adj_dense`` is provided.
            y_action: (B,)  – action index for each sample in the batch.
            adj_dense: (B, R, n_nodes, n_nodes)  – optional pre-computed dense
                adjacency matrix.  When provided, ``edge_index``,
                ``edge_type`` and ``batch_idx`` are ignored and the sparse →
                dense conversion is skipped.
            gmean_temperature: temperature for the gmean function.
            tau: temperature for the gumbel softmax function.
            sinkhorn_temperature: temperature for the Sinkhorn normalisation;
                values < 1 sharpen the assignment, values > 1 soften it.
        Returns:
            tuple of:
                adj_out_probs: (B, R, n_nodes, n_nodes)  – predicted adjacency
                    probabilities for the next state (values in [0, 1]).
                adj_batch: (B, R, n_nodes, n_nodes)  – dense adjacency matrix
                    of the current (s1) state (binary, unchanged).
        """
        # ── Force float32 for the entire forward pass ──────────────────────
        # Sinkhorn iterations and the log-space arithmetic are numerically
        # sensitive and MUST NOT run in float16 (AMP).  Casting here prevents
        # float16 overflow (e.g. -1e6 → -inf, large attention scores → NaN)
        # that otherwise causes CUDA BCE assertions after many epochs.
        orig_dtype = node_embeddings.dtype
        node_embeddings = node_embeddings.float()

        B = y_action.size(0)
        n_nodes = self.num_objects
        k = self.num_slots

        # Build batched adjacency: (B, R, n_nodes, n_nodes)
        if adj_dense is not None:
            adj_batch = adj_dense.float()
        else:
            # All graphs have exactly n_nodes nodes, so local index = global - graph_id * n
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

        # Reshape embeddings to (B, n_nodes, D)
        embeddings_batch = node_embeddings.view(B, n_nodes, -1)

        # Batched attention: (B, k, n_nodes), scaled by sqrt(D)
        slot_encodings = self.slot_encodings[y_action]  # (B, k, D)
        attention_batch = torch.bmm(
            slot_encodings, embeddings_batch.transpose(1, 2)
        ) / (self.object_dim**0.5)

        # Gumbel noise + logits
        k = self.num_slots
        if self.training and self.gumbel_sinkhorn:
            gumbel_dist = torch.distributions.gumbel.Gumbel(0.0, 1.0)
            gumbel_noise = gumbel_dist.sample(attention_batch.shape).to(
                attention_batch.device
            )
            logits = (attention_batch + gumbel_noise) / self.tau
        else:
            logits = attention_batch / self.tau

        # Extended logit matrix: (B, k+1, n_nodes+1)
        # Slack row (last row) ↔ non-slack columns: 0  → unmatched nodes can route here
        # Non-slack rows ↔ slack column (last col): learnable slack bias
        # Slack ↔ slack corner: 0
        #
        # When per_action_slots is True, each action only uses its own arity
        # worth of slots (no slack column).  Unused slot rows are set to -inf
        # so Sinkhorn effectively ignores them.
        if self.per_action_slots:
            # Slack row but no slack column: matrix is (B, k+1, n_nodes)
            logits_ext = torch.zeros(
                (B, k + 1, n_nodes),
                device=attention_batch.device,
                dtype=torch.float32,
            )
            logits_ext[:, :-1, :] = logits

            # Build a row mask for unused slots (beyond each action's arity)
            # instead of writing -inf which breaks Sinkhorn normalisation.
            arity_per_sample = self.action_arities_t[y_action]  # (B,)
            slot_indices = torch.arange(k, device=logits.device).unsqueeze(0)  # (1, k)
            unused_mask = slot_indices >= arity_per_sample.unsqueeze(1)  # (B, k)
            # Extend to (B, k+1): slack row is never masked
            sinkhorn_row_mask = F.pad(unused_mask, (0, 1), value=False)  # (B, k+1)
        else:
            # Slack row but no slack column (slot weights replace the slack column)
            logits_ext = torch.zeros(
                (B, k + 1, n_nodes),
                device=attention_batch.device,
                dtype=torch.float32,
            )
            logits_ext[:, :-1, :] = logits
            sinkhorn_row_mask = None

        # Force selection of marked action arguments by hard-masking logits.
        # Known args (>= 0) in y_action_args stay at their original slot index
        # (e.g. a known arg at position 3 is forced into slot 3).  For each
        # forced (b, slot, object) triple:
        #   - entire slot row → -inf  (slot must pick that object)
        #   - entire object column → -inf  (other slots can't pick the same object)
        #   - the forced (slot, object) cell → 0  (restore the one valid entry)
        if self.force_action_args and y_action_args is not None:
            known_mask = y_action_args >= 0  # (B, k)
            if known_mask.any():
                b_idx, orig_idx = known_mask.nonzero(as_tuple=True)
                s_idx = orig_idx  # keep original slot position
                obj_idx = y_action_args[b_idx, orig_idx]  # object to force

                n_cols = logits_ext.size(2)

                # Row mask: zero out each forced slot row
                row_mask = torch.zeros(
                    B, logits_ext.size(1), 1, device=logits_ext.device, dtype=torch.bool
                )
                row_mask[b_idx, s_idx, 0] = True
                logits_ext.masked_fill_(row_mask, -torch.inf)

                # Column mask: zero out each forced object column
                col_mask = torch.zeros(
                    B, n_cols, device=logits_ext.device, dtype=torch.bool
                )
                col_mask[b_idx, obj_idx] = True
                logits_ext.masked_fill_(col_mask.unsqueeze(1), -torch.inf)

                # Restore the forced (slot, object) cells to 0
                logits_ext[b_idx, s_idx, obj_idx] = 0.0

        # Batched sinkhorn in log-space
        selection_all = _batched_sinkhorn_d1d2(
            logits_ext, temperature=sinkhorn_temperature, row_mask=sinkhorn_row_mask
        )

        if self.per_action_slots:
            # Strip slack row; zero out unused slot rows
            selection = selection_all[:, :-1, :].clone()
            selection[unused_mask] = 0.0
        else:
            # Strip slack row
            selection = selection_all[:, :-1, :]
            # Apply learned slot weights: sigmoid gives per-slot probability
            slot_weights = torch.sigmoid(self.slot_weight_logits[y_action])  # (B, k)
            # Keep weight = 1.0 for forced selections so they aren't downweighted
            if self.force_action_args and y_action_args is not None:
                forced_mask = y_action_args >= 0  # (B, k)
                if forced_mask.any():
                    # Pad forced_mask to match slot_weights if num_slots > action arity
                    if forced_mask.shape[1] < slot_weights.shape[1]:
                        forced_mask = F.pad(
                            forced_mask,
                            (0, slot_weights.shape[1] - forced_mask.shape[1]),
                        )
                    slot_weights = torch.where(
                        forced_mask, torch.ones_like(slot_weights), slot_weights
                    )
            selection = selection * slot_weights.unsqueeze(-1)  # (B, k, n_nodes)

        # Cache for epoch-end logging
        if self.training:
            self._last_selection = selection.detach().cpu()
            self._last_y_action = y_action.detach().cpu()

        # Gumbel softmax
        if self.training and self.gumbel_softmax:
            preconditions = F.gumbel_softmax(
                self.preconditions[y_action],
                hard=False,
                dim=-1,
            )
            effects = F.gumbel_softmax(
                self.effects[y_action],
                hard=False,
                dim=-1,
            )
        else:
            preconditions = F.softmax(
                self.preconditions[y_action],
                dim=-1,
            )
            effects = F.softmax(
                self.effects[y_action],
                dim=-1,
            )

        ignore_effects = effects[..., 0]
        add_effects = effects[..., 1]
        del_effects = effects[..., -1]
        ignore_precond = preconditions[..., 0]
        pos_precond = preconditions[..., 1]
        neg_precond = preconditions[..., -1]

        # project slot-based effects and preconditions into adjacency matrix according to selection
        add_effects_projected = torch.einsum(
            "bai,brac,bcj->brij", selection, add_effects, selection
        )
        del_effects_projected = torch.einsum(
            "bai,brac,bcj->brij", selection, del_effects, selection
        )
        pos_precond_projected = torch.einsum(
            "bai,brac,bcj->brij", selection, pos_precond, selection
        )
        neg_precond_projected = torch.einsum(
            "bai,brac,bcj->brij", selection, neg_precond, selection
        )

        # check precondition fulfillment
        precond_fulfilled = (
            (1 - pos_precond_projected * (1 - adj_batch))
            * (1 - neg_precond_projected * adj_batch)
        ).flatten(start_dim=1)
        pre = gmean(
            precond_fulfilled, dim=-1, temperature=gmean_temperature + self.gmean_offset
        )

        # Cache mean precondition fulfillment for logging
        self._last_pre_mean = pre.mean().detach()
        self._last_pre_prod_mean = precond_fulfilled.prod(dim=-1).mean().detach()

        delta = (
            1 - adj_batch
        ) * add_effects_projected - adj_batch * del_effects_projected

        # apply changes
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
