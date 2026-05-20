import queue
import sys
import threading
import traceback
from typing import Mapping, Optional

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch_geometric.utils import to_dense_adj

from dias.utils import LossWeightScheduleCollection

from .components.dynamics_rgcn_st import DynamicsModelRGCNST
from .components.sinkhorn_graph_edit import SinkhornGraphEdit


class _PlotWorker:
    def __init__(self):
        # Keep at most one pending plot job so rendering never backlogs behind
        # training and stale validation snapshots are dropped.
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="PlotWorker", daemon=True
        )
        self._thread.start()

    def submit(self, fn, *args, **kwargs) -> None:
        if self._stopping:
            return

        while True:
            try:
                old_item = self._q.get_nowait()
            except queue.Empty:
                break
            else:
                self._q.task_done()
                if old_item is None:
                    return

        try:
            self._q.put_nowait((fn, args, kwargs))
        except queue.Full:
            # A newer job raced with us; drop this submission.
            pass

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            fn, args, kwargs = item
            try:
                fn(*args, **kwargs)
            except Exception as e:
                print(f"[PlotWorker] error: {e}", file=sys.stderr)
                traceback.print_exc()
            finally:
                self._q.task_done()

    def shutdown(self, timeout: float = 60.0) -> None:
        self._stopping = True
        self._q.put(None)
        self._thread.join(timeout=timeout)


class DiasModel(L.LightningModule):
    def __init__(
        self,
        num_objects: int,
        arity_relations: list[int],
        num_actions: int,
        object_dim: int,
        num_slots: int | None = None,
        batch_size: int = 100,
        learning_rate: float = 1e-4,
        force_action_args: bool = False,
        per_action_slots: bool = False,
        learn_preconditions: bool = True,
        action_arities: list[int] | None = None,
        action_arg_positions: list[list[int]] | None = None,
        successor_relations: bool = False,
        gumbel_sinkhorn: bool = True,
        gumbel_softmax: bool = True,
        random_node_embeddings: bool = True,
        parameter_plotting: Optional[dict[str, bool]] = {
            "effect_and_precondition": False,
            "selection": False,
        },
        loss_weight_schedule: dict | None = None,
        pcgrad_dyn_clip: bool = True,
        gmean_offset: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_objects = num_objects
        self.arity_relations = arity_relations
        self.num_actions = num_actions
        self.object_dim = object_dim

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.parameter_plotting = parameter_plotting
        self.force_action_args = force_action_args
        self.per_action_slots = per_action_slots
        self.learn_preconditions = learn_preconditions
        self.action_arities = action_arities
        self.successor_relations = successor_relations
        self.gumbel_sinkhorn = gumbel_sinkhorn
        self.gumbel_softmax = gumbel_softmax
        self.random_node_embeddings = random_node_embeddings
        self.pcgrad_dyn_clip = pcgrad_dyn_clip
        self.gmean_offset = gmean_offset

        # When per_action_slots is True, derive num_slots from the domain.
        if self.per_action_slots:
            assert (
                action_arities is not None
            ), "action_arities must be provided when per_action_slots=True"
            self.num_slots = max(action_arities)
        else:
            assert (
                num_slots is not None
            ), "num_slots must be set explicitly when per_action_slots=False"
            if action_arities is not None:
                assert num_slots >= max(action_arities), (
                    f"num_slots ({num_slots}) must be >= max action arity "
                    f"({max(action_arities)})"
                )
            if num_slots > self.num_objects:
                import sys as _sys

                print(
                    f"[graph_learner] clamping num_slots {num_slots} → "
                    f"{self.num_objects} (num_objects in this instance).",
                    file=_sys.stderr,
                )
                num_slots = self.num_objects
                if action_arities is not None:
                    assert num_slots >= max(action_arities), (
                        f"Clamped num_slots ({num_slots}) is smaller than max "
                        f"action arity ({max(action_arities)}); problem instance "
                        f"has too few objects for this domain."
                    )
            self.num_slots = num_slots

        self.num_relations = len(arity_relations)

        self.graph_edit = SinkhornGraphEdit(
            num_objects=self.num_objects,
            arity_relations=self.arity_relations,
            num_actions=self.num_actions,
            num_slots=self.num_slots,
            object_dim=self.object_dim,
            force_action_args=self.force_action_args,
            per_action_slots=self.per_action_slots,
            learn_preconditions=self.learn_preconditions,
            action_arities=self.action_arities,
            gumbel_sinkhorn=self.gumbel_sinkhorn,
            gumbel_softmax=self.gumbel_softmax,
            gmean_offset=self.gmean_offset,
        )

        # Action self-loop relation types: one per (action, active_position) pair.
        # Build a lookup table (num_actions, max_arity) -> compact relation index
        # (-1 for unused positions).
        if action_arg_positions is None:
            action_arg_positions = [[] for _ in range(self.num_actions)]
        self.action_arg_positions = action_arg_positions

        max_arity = max(
            (max(pos) + 1 if pos else 0 for pos in action_arg_positions),
            default=0,
        )
        lookup = torch.full((self.num_actions, max(max_arity, 1)), -1, dtype=torch.long)
        idx = 0
        for a, positions in enumerate(action_arg_positions):
            for p in positions:
                lookup[a, p] = idx
                idx += 1
        self.num_action_relations = idx
        self.register_buffer("_action_rel_lookup", lookup)

        # State-change relations: delete/add plus reverse counterparts.
        self.num_successor_relations = (
            self.num_relations * 4 if self.successor_relations else 0
        )

        self.dynamics_model = DynamicsModelRGCNST(
            self.object_dim,
            self.num_relations * 2
            + self.num_action_relations
            + self.num_successor_relations,
        )

        # Enable manual optimization for gradient projection
        self.automatic_optimization = False

        # LOSS WEIGHT SCHEDULES
        _default_loss_weights = {
            "gmean_weight": {"type": "linear", "keypoints": {0: 1.0}},
            "learning_rate": {"type": "linear", "keypoints": {0: 0.01}},
            "tau": {"type": "exponential", "v0": 1.0, "vT": 1.0, "T": 1.0},
            "alpha": {"type": "exponential", "v0": 1.0, "vT": 1.0, "T": 1.0},
            "sinkhorn_temperature": {"type": "linear", "keypoints": {0: 1.0}},
        }
        # Override with any user-provided schedules
        if loss_weight_schedule:
            _default_loss_weights.update(loss_weight_schedule)
        self.loss_weight_schedule_collection = LossWeightScheduleCollection(
            schedules=_default_loss_weights
        )

        # Background worker that owns all plotting + wandb image uploads.
        self._plot_worker = _PlotWorker()
        self._last_plot_step_submitted: int | None = None

    def teardown(self, stage: str) -> None:
        # Flush any pending plots before the run exits.
        if hasattr(self, "_plot_worker"):
            self._plot_worker.shutdown()

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        return optimizer

    def _encode_relation_edges(self, edge_index, edge_type, num_nodes: int):
        if edge_type.numel() == 0:
            return edge_type.new_empty((0,), dtype=torch.long)

        src, dst = edge_index
        return (
            (src.to(torch.long) * num_nodes) + dst.to(torch.long)
        ) * self.num_relations + edge_type.to(torch.long)

    def _build_relation_change_edges(self, batch_s1, batch_s2):
        empty_edge_index = batch_s1.edge_index.new_empty((2, 0))
        empty_edge_type = batch_s1.edge_attr.new_empty((0,))

        if batch_s2 is None:
            return empty_edge_index, empty_edge_type

        total_nodes = batch_s1.batch.size(0)
        s1_edge_codes = self._encode_relation_edges(
            batch_s1.edge_index,
            batch_s1.edge_attr,
            total_nodes,
        )
        s2_edge_codes = self._encode_relation_edges(
            batch_s2.edge_index,
            batch_s2.edge_attr,
            total_nodes,
        )

        deleted_mask = ~torch.isin(s1_edge_codes, s2_edge_codes)
        added_mask = ~torch.isin(s2_edge_codes, s1_edge_codes)

        change_rel_base = 2 * self.num_relations + self.num_action_relations
        edge_index_parts = []
        edge_type_parts = []

        if deleted_mask.any():
            deleted_edge_index = batch_s1.edge_index[:, deleted_mask]
            deleted_edge_type = batch_s1.edge_attr[deleted_mask]
            edge_index_parts.append(deleted_edge_index)
            edge_type_parts.append(deleted_edge_type + change_rel_base)

            edge_index_parts.append(deleted_edge_index.flip(0))
            edge_type_parts.append(
                deleted_edge_type + change_rel_base + self.num_relations
            )

        if added_mask.any():
            added_edge_index = batch_s2.edge_index[:, added_mask]
            added_edge_type = batch_s2.edge_attr[added_mask]
            edge_index_parts.append(added_edge_index)
            edge_type_parts.append(
                added_edge_type + change_rel_base + 2 * self.num_relations
            )

            edge_index_parts.append(added_edge_index.flip(0))
            edge_type_parts.append(
                added_edge_type + change_rel_base + 3 * self.num_relations
            )

        if not edge_index_parts:
            return empty_edge_index, empty_edge_type

        return torch.cat(edge_index_parts, dim=1), torch.cat(edge_type_parts, dim=0)

    def forward(self, batch_s1, batch_s2=None):
        edge_index = batch_s1.edge_index
        edge_type = batch_s1.edge_attr
        y_action = batch_s1.y_action
        y_action_args = batch_s1.y_action_args  # (B, max_arity)
        batch_idx = batch_s1.batch

        total_nodes = batch_idx.size(0)

        if self.random_node_embeddings:
            half = self.object_dim // 2
            z_a_action = torch.zeros(
                total_nodes,
                self.object_dim,
                device=self.device,
                dtype=torch.float32,
            )
            z_a_action[:, :half] = 0.1 * torch.randn(
                total_nodes,
                half,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            z_a_action = torch.zeros(
                total_nodes,
                self.object_dim,
                device=self.device,
                dtype=torch.float32,
            )

        # Dynamics uses directed relations and their reverse counterparts as
        # separate edge types. Reverse relation ids are offset by number_relations.
        edge_index_rev = edge_index.flip(0)
        edge_type_rev = edge_type + self.num_relations

        # Build action self-loop edges using the compact lookup table.
        # For action a with arg obj at position p, add a self-loop on obj
        # with relation type 2*num_relations + lookup[a, p].
        B = y_action.size(0)
        args = y_action_args  # (B, max_arity)
        data_arity = args.size(1)
        lookup_arity = self._action_rel_lookup.size(1)
        if data_arity < lookup_arity:
            pad = torch.full(
                (B, lookup_arity - data_arity),
                -1,
                device=self.device,
                dtype=args.dtype,
            )
            args = torch.cat([args, pad], dim=1)  # (B, lookup_arity)
        elif data_arity > lookup_arity:
            args = args[:, :lookup_arity]

        valid_mask = args >= 0  # (B, lookup_arity)
        b_idx, p_idx = valid_mask.nonzero(as_tuple=True)

        # Look up compact relation indices; filter out any -1 (unused positions)
        compact_idx = self._action_rel_lookup[y_action[b_idx], p_idx]
        used = compact_idx >= 0
        b_idx = b_idx[used]
        p_idx = p_idx[used]
        compact_idx = compact_idx[used]

        graph_offsets = torch.arange(B, device=self.device) * self.num_objects
        global_node_idx = graph_offsets[b_idx] + args[b_idx, p_idx]

        action_rel_base = 2 * self.num_relations
        rel_types = action_rel_base + compact_idx

        self_loop_ei = torch.stack([global_node_idx, global_node_idx], dim=0)

        dyn_ei_parts = [edge_index, edge_index_rev, self_loop_ei]
        dyn_et_parts = [edge_type, edge_type_rev, rel_types]

        # Optionally add state-change relations derived from S -> S'.
        if self.successor_relations and batch_s2 is not None:
            change_edge_index, change_edge_type = self._build_relation_change_edges(
                batch_s1,
                batch_s2,
            )
            if change_edge_type.numel() > 0:
                dyn_ei_parts.append(change_edge_index)
                dyn_et_parts.append(change_edge_type)

        edge_index_dyn = torch.cat(dyn_ei_parts, dim=1)
        edge_type_dyn = torch.cat(dyn_et_parts, dim=0)

        z_b_dyn = self.dynamics_model(
            z_a_action,
            edge_index_dyn,
            edge_type_dyn,
        )

        (
            adj_out_probs,
            adj_s1,
            ignore_effects,
            add_effects,
            del_effects,
            ignore_precond,
            pos_precond,
            neg_precond,
            selection,
            logits,
            delta,
        ) = self.graph_edit(
            node_embeddings=z_b_dyn,
            edge_index=edge_index,
            edge_type=edge_type,
            batch_idx=batch_idx,
            y_action=y_action,
            y_action_args=y_action_args,
            gmean_temperature=self.loss_weight_schedule_collection["gmean_weight"],
            tau=self.loss_weight_schedule_collection["tau"],
            sinkhorn_temperature=self.loss_weight_schedule_collection[
                "sinkhorn_temperature"
            ],
        )

        # Expose cached selection for epoch-end logging
        self._last_selection = self.graph_edit._last_selection
        self._last_y_action = self.graph_edit._last_y_action

        return (
            adj_out_probs,
            adj_s1,
            ignore_effects,
            add_effects,
            del_effects,
            ignore_precond,
            pos_precond,
            neg_precond,
            selection,
            logits,
            delta,
        )

    def step(self, batch, batch_idx, validation: bool = False):
        batch_s1, batch_s2, indices, _, _, _, _ = batch

        (
            adj_out_probs,
            adj_s1,
            ignore_effects,
            add_effects,
            del_effects,
            ignore_precond,
            pos_precond,
            neg_precond,
            selection,
            logits,
            delta,
        ) = self.forward(batch_s1, batch_s2)

        target_adj_list = []

        for r_idx in range(self.num_relations):
            mask = batch_s2.edge_attr == r_idx

            subset_edge_index = batch_s2.edge_index[:, mask]
            target_adj_dense = to_dense_adj(
                subset_edge_index,
                batch_s2.batch,
                max_num_nodes=self.num_objects,
            )
            target_adj_list.append(target_adj_dense)

        target_adj_tensor = torch.stack(target_adj_list, dim=1)

        # Disable autocast so binary_cross_entropy is safe under AMP
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            # CUDA BCE asserts if inputs are outside (0, 1) or NaN; adj predictions
            # can slightly leave [0, 1] under numerical stress (tau schedule, gmean).
            _eps = 1e-7
            p_adj = adj_out_probs.float().clamp(_eps, 1.0 - _eps)
            p_add = add_effects.float().clamp(_eps, 1.0 - _eps)
            p_del = del_effects.float().clamp(_eps, 1.0 - _eps)
            p_pos = pos_precond.float().clamp(_eps, 1.0 - _eps)
            p_neg = neg_precond.float().clamp(_eps, 1.0 - _eps)
            tgt = target_adj_tensor.float().clamp(0.0, 1.0)

            # Total number of probability elements across all losses
            total_numel = (
                adj_out_probs.numel()
                + (self.num_slots ** torch.tensor(self.arity_relations)).sum()
                * len(batch_s1)
                * 2
            )

            loss_adjacency = (
                F.binary_cross_entropy(p_adj, tgt, reduction="sum") / total_numel
            )

            # Weight preconditions by slot weights so that maximising
            # preconditions also requires slot activation.
            slot_w = torch.sigmoid(
                self.graph_edit.slot_weight_logits[batch_s1.y_action]
            )  # (B, k)
            # Outer product over the two slot dims → (B, 1, k, k)
            sw_matrix = slot_w[:, None, :, None] * slot_w[:, None, None, :]
            one_minus_ignore_precond_weighted = (
                (1 - ignore_precond) * sw_matrix
            ).clamp(_eps, 1.0 - _eps)

            loss_aux = (
                F.binary_cross_entropy(
                    ignore_effects, torch.ones_like(ignore_effects), reduction="sum"
                )
                + F.binary_cross_entropy(
                    one_minus_ignore_precond_weighted,
                    torch.ones_like(one_minus_ignore_precond_weighted),
                    reduction="sum",
                )
            ) / total_numel

            loss_descent = (
                torch.sum(
                    p_add * (1 - p_add)
                    + p_del * (1 - p_del)
                    + p_pos * (1 - p_pos)
                    + p_neg * (1 - p_neg)
                )
                / total_numel
            )

        loss_mse = ((adj_out_probs > 0.5) ^ (target_adj_tensor > 0.5)).sum()

        return {
            "loss_adjacency": loss_adjacency,
            "loss_aux": loss_aux,
            "loss_descent": loss_descent,
            "loss_mse": loss_mse,
            "pre_mean": self.graph_edit._last_pre_mean,
            "pre_prod_mean": self.graph_edit._last_pre_prod_mean,
        }

    def on_train_batch_start(self, batch, batch_idx):
        # Keyframe x-axis is global_step (not current_epoch), so schedules
        # advance per optimizer step. This matters for unbounded IterableDataset
        # training where "epoch" is only a logging cycle defined by
        # trainer.limit_train_batches.
        self.loss_weight_schedule_collection.update_loss_weights(self.global_step)

        new_lr = self.loss_weight_schedule_collection["learning_rate"]
        opt = self.optimizers()
        optimizers = opt if isinstance(opt, list) else [opt]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

        self.log_dict(
            {
                f"scheduled_w/{name}": self.loss_weight_schedule_collection[name]
                for name in self.loss_weight_schedule_collection
            }
        )

    def on_validation_epoch_start(self):
        self.loss_weight_schedule_collection.update_loss_weights(self.global_step)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        batch_s1, *_ = batch
        if hasattr(batch_s1, "num_flips_added"):
            self.log(
                "noise/num_flips_added",
                batch_s1.num_flips_added.float().mean(),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "noise/num_flips_removed",
                batch_s1.num_flips_removed.float().mean(),
                on_step=True,
                on_epoch=False,
            )
        if hasattr(batch_s1, "num_flips_added_per_rel"):
            _, relation_names, _ = self._get_domain_names()
            added_mean = batch_s1.num_flips_added_per_rel.float().mean(dim=0)
            removed_mean = batch_s1.num_flips_removed_per_rel.float().mean(dim=0)
            per_rel_log: dict[str, torch.Tensor] = {}
            for r, name in enumerate(relation_names):
                per_rel_log[f"noise/{name}/added"] = added_mean[r]
                per_rel_log[f"noise/{name}/removed"] = removed_mean[r]
            self.log_dict(per_rel_log, on_step=True, on_epoch=False)
        if hasattr(batch_s1, "num_true_predicates"):
            self.log(
                "noise/num_true_predicates",
                batch_s1.num_true_predicates.float().mean(),
                on_step=True,
                on_epoch=False,
            )

        results = self.step(batch, batch_idx)

        loss_adj = results["loss_adjacency"]
        loss_aux = results["loss_aux"]
        loss_descent = results["loss_descent"]

        # --- Gradient projection ---
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        # Compute gradients for the main adjacency loss
        grads_adj = torch.autograd.grad(
            loss_adj, trainable_params, retain_graph=True, allow_unused=True
        )
        # Compute gradients for the auxiliary loss
        grads_aux = torch.autograd.grad(
            self.loss_weight_schedule_collection["alpha"] * loss_aux,
            trainable_params,
            retain_graph=True,
            allow_unused=True,
        )

        # Project auxiliary gradients: remove component conflicting with adjacency
        for p, g_adj, g_aux in zip(trainable_params, grads_adj, grads_aux):
            if g_adj is None and g_aux is None:
                continue
            if g_adj is None:
                p.grad = g_aux.clone()
                continue
            if g_aux is None:
                p.grad = g_adj.clone()
                continue

            g_adj_flat = g_adj.reshape(-1)
            g_aux_flat = g_aux.reshape(-1)
            dot = torch.dot(g_adj_flat, g_aux_flat)

            # 1. Determine the direction (Projection)
            if dot < 0:
                # Conflict: project g_aux onto the plane orthogonal to g_adj
                g_adj_norm_sq = torch.dot(g_adj_flat, g_adj_flat)
                if g_adj_norm_sq > 1e-12:
                    g_aux_final = g_aux - (dot / g_adj_norm_sq) * g_adj
                else:
                    g_aux_final = g_aux
            else:
                # No conflict: keep original direction
                g_aux_final = g_aux

            # 2. Determine the magnitude (Dynamic Weighting / Relative Clipping)
            if self.pcgrad_dyn_clip:
                norm_adj = torch.norm(g_adj)
                norm_aux = torch.norm(g_aux_final)

                # If the aux gradient is larger than the main gradient, scale it down
                if norm_aux > norm_adj and norm_aux > 1e-12:
                    g_aux_final = g_aux_final * (norm_adj / norm_aux)

            # 3. Combine safely
            p.grad = (g_adj + g_aux_final).clone()

        opt.step()

        # Log all components
        combined_loss = (
            loss_adj
            + self.loss_weight_schedule_collection["alpha"] * loss_aux
            + self.loss_weight_schedule_collection["gamma"] * loss_descent
        )
        self.log_results(
            "train",
            {
                "loss": combined_loss.detach(),
                "loss_adjacency": loss_adj.detach(),
                "loss_aux": loss_aux.detach(),
                "loss_descent": loss_descent.detach(),
                "loss_mse": results["loss_mse"],
                "pre_mean": results["pre_mean"],
                "pre_prod_mean": results["pre_prod_mean"],
            },
        )

        # Return None so Lightning skips its own backward pass;
        # we handle grads manually. The optimizer step still runs.
        self.manual_backward_done = True
        return None

    def validation_step(self, batch, batch_idx):
        results = self.step(batch, batch_idx, validation=True)

        combined_loss = (
            results["loss_adjacency"]
            + self.loss_weight_schedule_collection["alpha"] * results["loss_aux"]
            + self.loss_weight_schedule_collection["gamma"] * results["loss_descent"]
        )
        self.log_results(
            "valid",
            {
                "loss": combined_loss,
                "loss_adjacency": results["loss_adjacency"],
                "loss_aux": results["loss_aux"],
                "loss_descent": results["loss_descent"].detach(),
                "loss_mse": results["loss_mse"],
                "pre_mean": results["pre_mean"],
                "pre_prod_mean": results["pre_prod_mean"],
            },
        )

        return combined_loss

    def test_step(self, batch, batch_idx):
        results = self.step(batch, batch_idx, validation=True)

        combined_loss = (
            results["loss_adjacency"]
            + self.loss_weight_schedule_collection["alpha"] * results["loss_aux"]
            + self.loss_weight_schedule_collection["gamma"] * results["loss_descent"]
        )
        self.log_results(
            "test",
            {
                "loss": combined_loss,
                "loss_adjacency": results["loss_adjacency"],
                "loss_aux": results["loss_aux"],
                "loss_descent": results["loss_descent"],
                "loss_mse": results["loss_mse"],
                "pre_mean": results["pre_mean"],
                "pre_prod_mean": results["pre_prod_mean"],
            },
        )

        return combined_loss

    def on_validation_epoch_end(self):
        if self.parameter_plotting:
            self._queue_validation_heatmaps()

    def _get_domain_names(self):
        """Retrieve action, relation, and object names from the datamodule.

        Falls back to generic numbered labels when the datamodule does not
        provide the names (e.g. when using a non-PDDL data source).
        """
        dm = getattr(self.trainer, "datamodule", None)

        def _normalise_names(names, target_len, prefix):
            names_list = [] if names is None else list(names)
            if len(names_list) < target_len:
                names_list.extend(
                    [f"{prefix} {i}" for i in range(len(names_list), target_len)]
                )
            else:
                names_list = names_list[:target_len]
            return names_list

        action_names = getattr(dm, "action_names", None)
        action_names = _normalise_names(action_names, self.num_actions, "Action")

        relation_names = getattr(dm, "relation_names", None)
        relation_names = _normalise_names(relation_names, self.num_relations, "Rel")

        # object_names is a dict {idx_str: name} – convert to ordered list
        obj_dict = getattr(dm, "object_names", None)
        if obj_dict is not None:
            object_names = [
                obj_dict.get(str(i), f"Node {i}") for i in range(self.num_objects)
            ]
        else:
            object_names = [f"Node {n}" for n in range(self.num_objects)]

        return action_names, relation_names, object_names

    @staticmethod
    def _render_action_relation_heatmap(
        values: np.ndarray,
        action_names: list,
        relation_names: list,
        title: str,
        epoch: int,
    ) -> wandb.Image:
        """Worker-thread body: build a heatmap image for one action/relation tensor."""
        num_actions, num_relations, k, _ = values.shape
        slot_labels = [f"Slot {s}" for s in range(k)]

        fig, axes = plt.subplots(
            num_actions,
            num_relations,
            figsize=(3 * num_relations + 1, 3 * num_actions),
            squeeze=False,
            constrained_layout=True,
        )

        im = None
        for a in range(num_actions):
            for r in range(num_relations):
                ax = axes[a][r]
                im = ax.imshow(values[a, r], vmin=0, vmax=1, cmap="RdYlBu_r")
                ax.set_xticks(range(k))
                ax.set_xticklabels(slot_labels, fontsize=7)
                ax.set_yticks(range(k))
                ax.set_yticklabels(slot_labels, fontsize=7)
                if a == 0:
                    relation_label = (
                        relation_names[r] if r < len(relation_names) else f"Rel {r}"
                    )
                    ax.set_title(relation_label, fontsize=9)
                if r == 0:
                    action_label = (
                        action_names[a] if a < len(action_names) else f"Action {a}"
                    )
                    ax.set_ylabel(action_label, fontsize=9)

                for i in range(k):
                    for j in range(k):
                        val = values[a, r, i, j]
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if abs(val - 0.5) > 0.3 else "black",
                        )

        fig.suptitle(f"{title} (epoch {epoch})", fontsize=12)
        if im is not None:
            fig.colorbar(im, ax=axes, shrink=0.6, label="probability")

        image = wandb.Image(fig)
        plt.close(fig)
        return image

    @staticmethod
    def _render_selection_heatmap(
        samples: dict,
        slot_weights: dict,
        action_names: list,
        object_names: list,
        epoch: int,
    ) -> wandb.Image:
        sorted_actions = sorted(samples.keys())
        num_actions = len(sorted_actions)
        k, n_nodes = next(iter(samples.values())).shape
        slot_labels = [f"Slot {s}" for s in range(k)]
        node_labels = object_names[:n_nodes]

        fig, axes = plt.subplots(
            num_actions,
            2,
            figsize=(max(4, 0.5 * n_nodes) + 2, 2 * num_actions),
            squeeze=False,
            constrained_layout=True,
            gridspec_kw={"width_ratios": [n_nodes, 1]},
        )

        im = None
        im_sw = None
        for row, a in enumerate(sorted_actions):
            ax = axes[row][0]
            mat = samples[a]
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(n_nodes))
            ax.set_xticklabels(node_labels, fontsize=6, rotation=45)
            ax.set_yticks(range(k))
            ax.set_yticklabels(slot_labels, fontsize=7)
            ax.set_ylabel(action_names[a], fontsize=9)
            if row == 0:
                ax.set_title("Slot → Node Selection", fontsize=9)

            for i in range(k):
                for j in range(n_nodes):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if mat[i, j] < 0.5 else "black",
                    )

            # Second column: slot activation probabilities (k, 1)
            ax_sw = axes[row][1]
            sw = slot_weights.get(a, np.zeros(k))  # (k,)
            sw_mat = sw.reshape(k, 1)
            im_sw = ax_sw.imshow(sw_mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax_sw.set_xticks([])
            ax_sw.set_yticks(range(k))
            ax_sw.set_yticklabels(slot_labels, fontsize=7)
            if row == 0:
                ax_sw.set_title("Slot\nWeight", fontsize=9)

            for i in range(k):
                val = float(sw[i])
                ax_sw.text(
                    0,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val < 0.5 else "black",
                )

        fig.suptitle(f"Selection Matrices (epoch {epoch})", fontsize=12)
        if im is not None:
            fig.colorbar(im, ax=axes, shrink=0.6, label="weight")

        image = wandb.Image(fig)
        plt.close(fig)
        return image

    def _queue_validation_heatmaps(self):
        """Snapshot validation visuals once and enqueue one media log job."""
        if not isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            return
        step = int(self.global_step)
        if self._last_plot_step_submitted == step:
            return

        action_names, relation_names, object_names = self._get_domain_names()
        effect_and_precondition = bool(
            self.parameter_plotting.get("effect_and_precondition", False)
        )
        selection_enabled = bool(self.parameter_plotting.get("selection", False))

        if not effect_and_precondition and not selection_enabled:
            return

        payload: dict[str, object] = {
            "epoch": int(self.current_epoch),
            "step": step,
            "action_names": action_names,
            "relation_names": relation_names,
            "object_names": object_names,
        }

        if effect_and_precondition:
            with torch.no_grad():
                precond_probs = F.softmax(
                    self.graph_edit.preconditions.detach(), dim=-1
                )
                effect_probs = F.softmax(self.graph_edit.effects.detach(), dim=-1)
                payload["components"] = [
                    (
                        effect_probs[..., 1].cpu().numpy(),
                        "Add Effects",
                        "add_effects_heatmap",
                    ),
                    (
                        effect_probs[..., -1].cpu().numpy(),
                        "Delete Effects",
                        "delete_effects_heatmap",
                    ),
                    (
                        precond_probs[..., 1].cpu().numpy(),
                        "Positive Preconditions",
                        "positive_preconditions_heatmap",
                    ),
                    (
                        precond_probs[..., -1].cpu().numpy(),
                        "Negative Preconditions",
                        "negative_preconditions_heatmap",
                    ),
                ]

        if (
            selection_enabled
            and self._last_selection is not None
            and self._last_y_action is not None
        ):
            selection = self._last_selection.numpy()
            y_action = self._last_y_action.numpy()

            samples: dict = {}
            for idx in range(len(y_action)):
                action_idx = int(y_action[idx])
                if action_idx not in samples:
                    samples[action_idx] = selection[idx].copy()

            if samples:
                with torch.no_grad():
                    all_slot_weights = (
                        torch.sigmoid(self.graph_edit.slot_weight_logits.detach())
                        .cpu()
                        .numpy()
                    )
                payload["selection_samples"] = samples
                payload["selection_slot_weights"] = {
                    action_idx: all_slot_weights[action_idx] for action_idx in samples
                }

        if len(payload) == 5:
            return

        self._last_plot_step_submitted = step
        self._plot_worker.submit(
            self._render_and_log_validation_heatmaps,
            payload,
            self.logger.experiment,
        )

    @staticmethod
    def _render_and_log_validation_heatmaps(
        payload: dict[str, object], wandb_run
    ) -> None:
        media_payload = {}

        components = payload.get("components", [])
        for values, title, log_key in components:
            media_payload[log_key] = (
                DiasModel._render_action_relation_heatmap(
                    values,
                    payload["action_names"],
                    payload["relation_names"],
                    title,
                    payload["epoch"],
                )
            )

        samples = payload.get("selection_samples")
        slot_weights = payload.get("selection_slot_weights")
        if samples and slot_weights:
            media_payload["selection_heatmap"] = (
                DiasModel._render_selection_heatmap(
                    samples,
                    slot_weights,
                    payload["action_names"],
                    payload["object_names"],
                    payload["epoch"],
                )
            )

        if media_payload:
            wandb_run.log(media_payload, step=payload["step"])

    def log_results(self, prefix: str, results: Mapping[str, torch.Tensor | int]):
        is_train = prefix == "train"
        for key, value in results.items():
            if isinstance(value, torch.Tensor) and value.isnan():
                continue

            # Log timing metrics to separate tab
            if key.startswith("timing_"):
                # Remove timing_ prefix and log to timing tab
                timing_key = key[7:]  # Remove 'timing_' prefix
                self.log(
                    f"timing/{timing_key}",
                    value,
                    prog_bar=False,
                    on_step=is_train,
                    on_epoch=not is_train,
                )
            else:
                # Log regular metrics to train/valid tabs
                self.log(
                    f"{prefix}/{key}",
                    value,
                    prog_bar=(key == "loss"),
                    on_step=is_train,
                    on_epoch=not is_train,
                )
