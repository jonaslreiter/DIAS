import sys
from typing import Mapping, Optional

import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from dias.utils import LossWeightScheduleCollection, resolve_label_names
from dias.utils.plotting import PlotWorker, queue_validation_heatmaps

from .gnn import GNNModule
from .prediction import PredictionModule
from .utils import build_relation_change_edges


class DIAS(L.LightningModule):
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
        self.random_node_embeddings = random_node_embeddings
        self.pcgrad_dyn_clip = pcgrad_dyn_clip
        self.gmean_offset = gmean_offset

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
                print(
                    f"[graph_learner] clamping num_slots {num_slots} → "
                    f"{self.num_objects} (num_objects in this instance).",
                    file=sys.stderr,
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

        self.graph_edit = PredictionModule(
            num_objects=self.num_objects,
            arity_relations=self.arity_relations,
            num_actions=self.num_actions,
            num_slots=self.num_slots,
            object_dim=self.object_dim,
            force_action_args=self.force_action_args,
            per_action_slots=self.per_action_slots,
            learn_preconditions=self.learn_preconditions,
            action_arities=self.action_arities,
            gmean_offset=self.gmean_offset,
        )

        # Action self-loop relation types: one per (action, active-position) pair.
        if action_arg_positions is None:
            action_arg_positions = [[] for _ in range(self.num_actions)]
        self.action_arg_positions = action_arg_positions

        max_arity = max(
            (max(pos) + 1 if pos else 0 for pos in action_arg_positions), default=0
        )
        lookup = torch.full((self.num_actions, max(max_arity, 1)), -1, dtype=torch.long)
        idx = 0
        for a, positions in enumerate(action_arg_positions):
            for p in positions:
                lookup[a, p] = idx
                idx += 1
        self.num_action_relations = idx
        self.register_buffer("_action_rel_lookup", lookup)

        # State-change relations (S→S'): delete+add, plus reverse counterparts.
        self.num_successor_relations = (
            self.num_relations * 4 if self.successor_relations else 0
        )

        self.dynamics_model = GNNModule(
            self.object_dim,
            self.num_relations * 2
            + self.num_action_relations
            + self.num_successor_relations,
        )

        self.automatic_optimization = False

        defaults = {
            "gmean_temperature": {"type": "linear", "keypoints": {0: 1.0}},
            "learning_rate": {"type": "linear", "keypoints": {0: 0.01}},
            "aux_loss_weight": {"type": "exponential", "v0": 1.0, "vT": 1.0, "T": 1.0},
        }
        if loss_weight_schedule:
            defaults.update(loss_weight_schedule)
        self.loss_weight_schedule_collection = LossWeightScheduleCollection(
            schedules=defaults
        )

        self._plot_worker = PlotWorker()
        self._last_plot_step_submitted: int | None = None
        self._label_names: tuple[list[str], list[str], list[str]] | None = None

    def teardown(self, stage: str) -> None:
        if hasattr(self, "_plot_worker"):
            self._plot_worker.shutdown()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.learning_rate
        )

    def forward(self, batch_s1, batch_s2=None):
        edge_index = batch_s1.edge_index
        edge_type = batch_s1.edge_attr
        y_action = batch_s1.y_action
        y_action_args = batch_s1.y_action_args  # (B, max_arity)
        total_nodes = batch_s1.batch.size(0)

        z_a_action = torch.zeros(
            total_nodes,
            self.object_dim,
            device=self.device,
            dtype=torch.float32,
        )
        if self.random_node_embeddings:
            half = self.object_dim // 2
            z_a_action[:, :half] = 0.1 * torch.randn(
                total_nodes, half, device=self.device, dtype=torch.float32
            )

        # Reverse-edge relations are offset by num_relations.
        edge_index_rev = edge_index.flip(0)
        edge_type_rev = edge_type + self.num_relations

        # Action self-loops: arg at position p of action a -> self-loop on object
        # with relation type 2*num_relations + lookup[a, p].
        B = y_action.size(0)
        args = y_action_args
        lookup_arity = self._action_rel_lookup.size(1)
        if args.size(1) < lookup_arity:
            args = F.pad(args, (0, lookup_arity - args.size(1)), value=-1)
        elif args.size(1) > lookup_arity:
            args = args[:, :lookup_arity]

        b_idx, p_idx = (args >= 0).nonzero(as_tuple=True)
        compact_idx = self._action_rel_lookup[y_action[b_idx], p_idx]
        used = compact_idx >= 0
        b_idx, p_idx, compact_idx = b_idx[used], p_idx[used], compact_idx[used]

        global_node_idx = (
            torch.arange(B, device=self.device)[b_idx] * self.num_objects
            + args[b_idx, p_idx]
        )
        self_loop_ei = torch.stack([global_node_idx, global_node_idx], dim=0)
        rel_types = 2 * self.num_relations + compact_idx

        dyn_ei_parts = [edge_index, edge_index_rev, self_loop_ei]
        dyn_et_parts = [edge_type, edge_type_rev, rel_types]

        if self.successor_relations and batch_s2 is not None:
            ch_ei, ch_et = build_relation_change_edges(
                batch_s1,
                batch_s2,
                num_relations=self.num_relations,
                num_action_relations=self.num_action_relations,
            )
            if ch_et.numel() > 0:
                dyn_ei_parts.append(ch_ei)
                dyn_et_parts.append(ch_et)

        z_b_dyn = self.dynamics_model(
            z_a_action,
            torch.cat(dyn_ei_parts, dim=1),
            torch.cat(dyn_et_parts, dim=0),
        )

        out = self.graph_edit(
            node_embeddings=z_b_dyn,
            edge_index=edge_index,
            edge_type=edge_type,
            batch_idx=batch_s1.batch,
            y_action=y_action,
            y_action_args=y_action_args,
            gmean_temperature=self.loss_weight_schedule_collection["gmean_temperature"],
        )

        self._last_selection = self.graph_edit._last_selection
        self._last_y_action = self.graph_edit._last_y_action
        return out

    def step(self, batch, batch_idx, validation: bool = False):
        batch_s1, batch_s2, *_ = batch
        (
            adj_out_probs,
            _adj_s1,
            ignore_effects,
            _add_effects,
            _del_effects,
            ignore_precond,
            _pos_precond,
            _neg_precond,
            _selection,
            _logits,
            _delta,
        ) = self.forward(batch_s1, batch_s2)

        target_adj = torch.stack(
            [
                to_dense_adj(
                    batch_s2.edge_index[:, batch_s2.edge_attr == r],
                    batch_s2.batch,
                    max_num_nodes=self.num_objects,
                )
                for r in range(self.num_relations)
            ],
            dim=1,
        )

        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            eps = 1e-7
            p_adj = adj_out_probs.float().clamp(eps, 1 - eps)
            tgt = target_adj.float().clamp(0.0, 1.0)

            total_numel = (
                adj_out_probs.numel()
                + (self.num_slots ** torch.tensor(self.arity_relations)).sum()
                * len(batch_s1)
                * 2
            )

            loss_adjacency = (
                F.binary_cross_entropy(p_adj, tgt, reduction="sum") / total_numel
            )

            # Weight preconditions by slot weights so maximising preconditions
            # also requires slot activation.
            slot_w = torch.sigmoid(
                self.graph_edit.slot_weight_logits[batch_s1.y_action]
            )
            sw_matrix = slot_w[:, None, :, None] * slot_w[:, None, None, :]
            one_minus_ip_w = ((1 - ignore_precond) * sw_matrix).clamp(eps, 1 - eps)

            loss_aux = (
                F.binary_cross_entropy(
                    ignore_effects, torch.ones_like(ignore_effects), reduction="sum"
                )
                + F.binary_cross_entropy(
                    one_minus_ip_w, torch.ones_like(one_minus_ip_w), reduction="sum"
                )
            ) / total_numel

        loss_mse = ((adj_out_probs > 0.5) ^ (target_adj > 0.5)).sum()

        return {
            "loss_adjacency": loss_adjacency,
            "loss_aux": loss_aux,
            "loss_mse": loss_mse,
            "pre_mean": self.graph_edit._last_pre_mean,
            "pre_prod_mean": self.graph_edit._last_pre_prod_mean,
        }

    def _combined_loss(self, results: Mapping[str, torch.Tensor]) -> torch.Tensor:
        w = self.loss_weight_schedule_collection
        return results["loss_adjacency"] + w["aux_loss_weight"] * results["loss_aux"]

    def on_train_batch_start(self, batch, batch_idx):
        # x-axis is global_step (not epoch) so schedules advance per optimizer step.
        self.loss_weight_schedule_collection.update_loss_weights(self.global_step)
        new_lr = self.loss_weight_schedule_collection["learning_rate"]
        opt = self.optimizers()
        for optimizer in (opt if isinstance(opt, list) else [opt]):
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        self.log_dict(
            {
                f"scheduled_w/{name}": self.loss_weight_schedule_collection[name]
                for name in self.loss_weight_schedule_collection
            }
        )

    def on_validation_epoch_start(self):
        self.loss_weight_schedule_collection.update_loss_weights(self.global_step)

    def _log_noise(self, batch_s1) -> None:
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
            _, relation_names, _ = self.label_names
            added = batch_s1.num_flips_added_per_rel.float().mean(dim=0)
            removed = batch_s1.num_flips_removed_per_rel.float().mean(dim=0)
            self.log_dict(
                {f"noise/{n}/added": added[r] for r, n in enumerate(relation_names)}
                | {
                    f"noise/{n}/removed": removed[r]
                    for r, n in enumerate(relation_names)
                },
                on_step=True,
                on_epoch=False,
            )
        if hasattr(batch_s1, "num_true_predicates"):
            self.log(
                "noise/num_true_predicates",
                batch_s1.num_true_predicates.float().mean(),
                on_step=True,
                on_epoch=False,
            )

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        self._log_noise(batch[0])

        results = self.step(batch, batch_idx)
        loss_adj = results["loss_adjacency"]
        loss_aux = results["loss_aux"]
        aux_w = self.loss_weight_schedule_collection["aux_loss_weight"]

        # PCGrad-style projection: remove the component of loss_aux's gradient
        # that conflicts with loss_adj's gradient.
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        grads_adj = torch.autograd.grad(
            loss_adj, trainable_params, retain_graph=True, allow_unused=True
        )
        grads_aux = torch.autograd.grad(
            aux_w * loss_aux, trainable_params, retain_graph=True, allow_unused=True
        )

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
            if dot < 0:
                g_adj_norm_sq = torch.dot(g_adj_flat, g_adj_flat)
                g_aux_final = (
                    g_aux - (dot / g_adj_norm_sq) * g_adj
                    if g_adj_norm_sq > 1e-12
                    else g_aux
                )
            else:
                g_aux_final = g_aux

            if self.pcgrad_dyn_clip:
                norm_adj = torch.norm(g_adj)
                norm_aux = torch.norm(g_aux_final)
                if norm_aux > norm_adj and norm_aux > 1e-12:
                    g_aux_final = g_aux_final * (norm_adj / norm_aux)

            p.grad = (g_adj + g_aux_final).clone()

        opt.step()

        combined = loss_adj + aux_w * loss_aux
        self.log_results(
            "train",
            {
                "loss": combined.detach(),
                "loss_adjacency": loss_adj.detach(),
                "loss_aux": loss_aux.detach(),
                "loss_mse": results["loss_mse"],
                "pre_mean": results["pre_mean"],
                "pre_prod_mean": results["pre_prod_mean"],
            },
        )
        # Skip Lightning's auto backward — we set p.grad manually above.
        self.manual_backward_done = True
        return None

    def _eval_step(self, prefix: str, batch, batch_idx) -> torch.Tensor:
        results = self.step(batch, batch_idx, validation=True)
        combined = self._combined_loss(results)
        self.log_results(
            prefix,
            {
                "loss": combined,
                "loss_adjacency": results["loss_adjacency"],
                "loss_aux": results["loss_aux"],
                "loss_mse": results["loss_mse"],
                "pre_mean": results["pre_mean"],
                "pre_prod_mean": results["pre_prod_mean"],
            },
        )
        return combined

    def validation_step(self, batch, batch_idx):
        return self._eval_step("valid", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._eval_step("test", batch, batch_idx)

    def on_validation_epoch_end(self):
        if self.parameter_plotting:
            queue_validation_heatmaps(self)

    @property
    def label_names(self):
        """Cached (action, relation, object) display names; resolved on first access."""
        if self._label_names is None:
            self._label_names = resolve_label_names(
                getattr(self.trainer, "datamodule", None),
                self.num_actions,
                self.num_relations,
                self.num_objects,
            )
        return self._label_names

    def log_results(self, prefix: str, results: Mapping[str, torch.Tensor | int]):
        is_train = prefix == "train"
        for key, value in results.items():
            if isinstance(value, torch.Tensor) and value.isnan():
                continue
            if key.startswith("timing_"):
                self.log(
                    f"timing/{key[7:]}",
                    value,
                    prog_bar=False,
                    on_step=is_train,
                    on_epoch=not is_train,
                )
            else:
                self.log(
                    f"{prefix}/{key}",
                    value,
                    prog_bar=(key == "loss"),
                    on_step=is_train,
                    on_epoch=not is_train,
                )
