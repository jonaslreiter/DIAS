import io
import logging
import sys
from typing import Mapping, Optional

import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from dias.reporting.labels import resolve_label_names
from dias.reporting.plotting import PlotWorker, queue_validation_heatmaps
from dias.schedules import Schedule, as_schedule, schedule_config

from .checkpoints import CHECKPOINT_VERSION, VERSION_KEY, migrate
from .gnn import DEFAULT_HIDDEN_DIMS, GNNModule
from .prediction import PredictionModule

logger = logging.getLogger(__name__)
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
        learning_rate: float = 0.01,
        aux_loss_weight: float = 1.0,
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
        gmean_temperature: Schedule | Mapping | None = None,
        pcgrad_dyn_clip: bool = True,
        state_dependent_effects: bool = False,
        use_slot_weights: bool = True,
        use_precondition_loss: bool = True,
        use_effect_loss: bool = True,
        use_pcgrad: bool = True,
        encoder_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.gmean_temperature = as_schedule(gmean_temperature)
        # Hyperparameters must stay primitive: torch.load defaults to
        # weights_only=True since 2.6 and refuses to unpickle a dataclass,
        # which would make the whole checkpoint unloadable.
        self.hparams["gmean_temperature"] = schedule_config(self.gmean_temperature)

        self.num_objects = num_objects
        self.arity_relations = arity_relations
        self.num_actions = num_actions
        self.object_dim = object_dim
        self.learning_rate = learning_rate
        self.aux_loss_weight = aux_loss_weight
        self.parameter_plotting = parameter_plotting
        self.force_action_args = force_action_args
        self.per_action_slots = per_action_slots
        self.learn_preconditions = learn_preconditions
        self.action_arities = action_arities
        self.successor_relations = successor_relations
        self.random_node_embeddings = random_node_embeddings
        self.pcgrad_dyn_clip = pcgrad_dyn_clip
        self.state_dependent_effects = state_dependent_effects
        self.use_slot_weights = use_slot_weights
        self.use_precondition_loss = use_precondition_loss
        self.use_effect_loss = use_effect_loss
        self.use_pcgrad = use_pcgrad

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
            state_dependent_effects=self.state_dependent_effects,
            use_slot_weights=self.use_slot_weights,
        )

        # Action self-loop relation types: one per (action, active-position) pair.
        if action_arg_positions is None:
            action_arg_positions = [[] for _ in range(self.num_actions)]

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
            hidden_dims=encoder_hidden_dims or DEFAULT_HIDDEN_DIMS,
        )

        self.automatic_optimization = False

        self._plot_worker = PlotWorker()
        self._last_plot_step_submitted: int | None = None
        self._label_names: tuple[list[str], list[str], list[str]] | None = None

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint[VERSION_KEY] = CHECKPOINT_VERSION

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        """Load a checkpoint, bringing an older layout up to date on the way in.

        The version stamp decides, not a failed load: Lightning drops
        hyper-parameters it does not recognise instead of raising, so a renamed
        one would silently fall back to its default and quietly change what the
        archived run means.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        version = checkpoint.get(VERSION_KEY, 1)
        if version < CHECKPOINT_VERSION:
            logger.info("Migrating checkpoint from version %d.", version)
            checkpoint = migrate(checkpoint)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        return super().load_from_checkpoint(buffer, *args, **kwargs)

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

        initial_node_embeddings = torch.zeros(
            total_nodes,
            self.object_dim,
            device=self.device,
            dtype=torch.float32,
        )
        if self.random_node_embeddings:
            half = self.object_dim // 2
            initial_node_embeddings[:, :half] = 0.1 * torch.randn(
                total_nodes, half, device=self.device, dtype=torch.float32
            )

        # Reverse-edge relations are offset by num_relations.
        reverse_edge_index = edge_index.flip(0)
        reverse_edge_type = edge_type + self.num_relations

        # Action self-loops: arg at position p of action a -> self-loop on object
        # with relation type 2*num_relations + lookup[a, p].
        batch_size = y_action.size(0)
        action_arguments = y_action_args
        lookup_arity = self._action_rel_lookup.size(1)
        if action_arguments.size(1) < lookup_arity:
            action_arguments = F.pad(
                action_arguments,
                (0, lookup_arity - action_arguments.size(1)),
                value=-1,
            )
        elif action_arguments.size(1) > lookup_arity:
            action_arguments = action_arguments[:, :lookup_arity]

        batch_indices, argument_positions = (action_arguments >= 0).nonzero(
            as_tuple=True
        )
        action_relation_indices = self._action_rel_lookup[
            y_action[batch_indices], argument_positions
        ]
        used_positions = action_relation_indices >= 0
        batch_indices = batch_indices[used_positions]
        argument_positions = argument_positions[used_positions]
        action_relation_indices = action_relation_indices[used_positions]

        action_argument_nodes = (
            torch.arange(batch_size, device=self.device)[batch_indices]
            * self.num_objects
            + action_arguments[batch_indices, argument_positions]
        )
        action_edge_index = torch.stack(
            [action_argument_nodes, action_argument_nodes], dim=0
        )
        action_edge_type = 2 * self.num_relations + action_relation_indices

        dynamics_edge_indices = [edge_index, reverse_edge_index, action_edge_index]
        dynamics_edge_types = [edge_type, reverse_edge_type, action_edge_type]

        if self.successor_relations and batch_s2 is not None:
            ch_ei, ch_et = build_relation_change_edges(
                batch_s1,
                batch_s2,
                num_relations=self.num_relations,
                num_action_relations=self.num_action_relations,
            )
            if ch_et.numel() > 0:
                dynamics_edge_indices.append(ch_ei)
                dynamics_edge_types.append(ch_et)

        encoded_node_embeddings = self.dynamics_model(
            initial_node_embeddings,
            torch.cat(dynamics_edge_indices, dim=1),
            torch.cat(dynamics_edge_types, dim=0),
        )

        prediction = self.graph_edit(
            node_embeddings=encoded_node_embeddings,
            edge_index=edge_index,
            edge_type=edge_type,
            batch_idx=batch_s1.batch,
            y_action=y_action,
            y_action_args=y_action_args,
            gmean_temperature=self.gmean_temperature(self.global_step),
        )

        self._last_selection = self.graph_edit._last_selection
        self._last_y_action = self.graph_edit._last_y_action
        return prediction

    def step(self, batch):
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
            # also requires slot activation. With use_slot_weights=False the
            # weights are forced to 1 (all slots active).
            if self.use_slot_weights:
                slot_w = torch.sigmoid(
                    self.graph_edit.slot_weight_logits[batch_s1.y_action]
                )
            else:
                slot_w = torch.ones_like(
                    self.graph_edit.slot_weight_logits[batch_s1.y_action]
                )
            sw_matrix = slot_w[:, None, :, None] * slot_w[:, None, None, :]
            one_minus_ip_w = ((1 - ignore_precond) * sw_matrix).clamp(eps, 1 - eps)

            # L_eff encourages minimal effects; L_pre encourages restrictive
            # preconditions. Each term can be ablated independently.
            loss_eff = F.binary_cross_entropy(
                ignore_effects, torch.ones_like(ignore_effects), reduction="sum"
            )
            loss_pre = F.binary_cross_entropy(
                one_minus_ip_w, torch.ones_like(one_minus_ip_w), reduction="sum"
            )
            aux_terms = torch.zeros((), device=p_adj.device, dtype=p_adj.dtype)
            if self.use_effect_loss:
                aux_terms = aux_terms + loss_eff
            if self.use_precondition_loss:
                aux_terms = aux_terms + loss_pre
            loss_aux = aux_terms / total_numel

        loss_mse = ((adj_out_probs > 0.5) ^ (target_adj > 0.5)).sum()

        return {
            "loss_adjacency": loss_adjacency,
            "loss_aux": loss_aux,
            "loss_mse": loss_mse,
            "pre_mean": self.graph_edit._last_pre_mean,
            "pre_prod_mean": self.graph_edit._last_pre_prod_mean,
        }

    def on_train_batch_start(self, batch, batch_idx):
        # Scheduled on global_step, not epoch, so it advances per optimizer step.
        self.log(
            "scheduled_w/gmean_temperature", self.gmean_temperature(self.global_step)
        )

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

    def _aux_active(self) -> bool:
        """Whether the auxiliary objective contributes any gradient this run."""
        return (self.aux_loss_weight != 0.0) and (
            self.use_effect_loss or self.use_precondition_loss
        )

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        self._log_noise(batch[0])

        results = self.step(batch)
        loss_adj = results["loss_adjacency"]
        loss_aux = results["loss_aux"]
        aux_w = self.aux_loss_weight

        if not self._aux_active():
            # Ablation: no auxiliary objective (alpha = 0 / L_pre and L_eff off).
            loss_adj.backward()
        elif not self.use_pcgrad:
            # Ablation: plain weighted sum instead of PCGrad projection.
            (loss_adj + aux_w * loss_aux).backward()
        else:
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

    def _eval_step(self, prefix: str, batch) -> torch.Tensor:
        results = self.step(batch)
        combined = (
            results["loss_adjacency"] + self.aux_loss_weight * results["loss_aux"]
        )
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
        return self._eval_step("valid", batch)

    def test_step(self, batch, batch_idx):
        return self._eval_step("test", batch)

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
            self.log(
                f"{prefix}/{key}",
                value,
                prog_bar=(key == "loss"),
                on_step=is_train,
                on_epoch=not is_train,
            )
