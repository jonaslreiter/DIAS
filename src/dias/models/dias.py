"""The DIAS model (paper, Sec. 4)."""

import io
import logging
from typing import Mapping

import lightning as L
import torch

from dias.adjacency import dense_from_edges
from dias.hparams import to_mapping
from dias.schedules import Schedule, as_schedule

from .checkpoints import CHECKPOINT_VERSION, VERSION_KEY, migrate
from .edges import EdgeTypeLayout
from .encoder import DEFAULT_HIDDEN_DIMS, RelationalEncoder
from .objectives import DiasObjective, Losses
from .outputs import SchemaLogits
from .pcgrad import gradient_combiner
from .prediction import SuccessorPredictor
from .selection import resolve_num_slots

logger = logging.getLogger(__name__)


class DIAS(L.LightningModule):
    """Differentiable Induction of Action Schemas. The defaults are the paper's model."""

    def __init__(
        self,
        num_objects: int,
        relation_arities: list[int],
        num_actions: int,
        object_dim: int = 32,
        num_slots: int | None = None,
        learning_rate: float = 0.005,
        aux_loss_alpha: float = 0.1,
        force_action_args: bool = False,
        per_action_slots: bool = False,
        gate_preconditions: bool = True,
        action_arities: list[int] | None = None,
        action_arg_positions: list[list[int]] | None = None,
        random_node_features: bool = True,
        precondition_tau: Schedule | Mapping | None = None,
        mlp_effects: bool = False,
        use_slot_weights: bool = True,
        use_precondition_loss: bool = True,
        use_applicability_loss: bool = False,
        observation_noise: float = 0.0,
        use_effect_loss: bool = True,
        use_pcgrad: bool = True,
        encoder_hidden_dims: list[int] | None = None,
        derivation_threshold: float = 0.5,
    ):
        """
        Args:
            num_objects: Objects in the instance. (linked from the datamodule)
            relation_arities: Arity (1 or 2) of every relation, in channel order. (linked from the datamodule)
            num_actions: Actions of the domain, one lifted schema each. (linked from the datamodule)
            object_dim: Width d of the object embeddings.
            num_slots: Slots M per action, capped at num_objects. Required unless per_action_slots.
            learning_rate: AdamW learning rate.
            aux_loss_alpha: Weight alpha of L_aux.
            force_action_args: Bind the slots of the arguments an action label names to those objects.
            per_action_slots: Give every action one slot per argument instead of num_slots.
            gate_preconditions: Apply the effects only as far as the learned preconditions hold (p_pre in Eq. 5).
            action_arities: Arity of every action. (linked from the datamodule)
            action_arg_positions: Argument positions the action labels name, per action. (linked from the datamodule)
            random_node_features: Draw the first half of the initial node features at random.
            precondition_tau: Schedule of tau, the root degree of the relaxed precondition product. Unset holds it at 1.
            mlp_effects: Predict the effects from the state with an MLP instead of learning P_eff.
            use_slot_weights: Learn the slot weights w, which give the arity.
            use_precondition_loss: Include L_pre in L_aux.
            use_applicability_loss: Add to L_main a cross-entropy asking every transition's preconditions to hold at its bound objects.
            observation_noise: Probability with which L_adj takes each observed atom to be flipped.
            use_effect_loss: Include L_eff in L_aux.
            use_pcgrad: Combine the gradients of L_main and L_aux with PCGrad instead of adding them.
            encoder_hidden_dims: Widths of the R-GCN's hidden layers.
            derivation_threshold: Threshold of the schema derivation for the learned domain written beside each checkpoint.
        """
        super().__init__()
        self.save_hyperparameters()
        self.precondition_tau = as_schedule(precondition_tau)
        self.hparams["precondition_tau"] = to_mapping(self.precondition_tau)

        self.num_objects = num_objects
        self.num_actions = num_actions
        self.num_relations = len(relation_arities)
        self.num_slots = resolve_num_slots(
            num_slots,
            num_objects=num_objects,
            action_arities=action_arities,
            per_action_slots=per_action_slots,
        )
        self.force_action_args = force_action_args
        self.learning_rate = learning_rate
        self.aux_loss_alpha = aux_loss_alpha

        self.edges = EdgeTypeLayout(
            num_relations=self.num_relations,
            num_actions=num_actions,
            action_arg_positions=action_arg_positions,
        )
        self.encoder = RelationalEncoder(
            object_dim,
            self.edges.num_edge_types,
            hidden_dims=encoder_hidden_dims or DEFAULT_HIDDEN_DIMS,
            random_node_features=random_node_features,
        )
        self.predictor = SuccessorPredictor(
            relation_arities=relation_arities,
            num_actions=num_actions,
            num_slots=self.num_slots,
            object_dim=object_dim,
            force_action_args=force_action_args,
            per_action_slots=per_action_slots,
            gate_preconditions=gate_preconditions,
            action_arities=action_arities,
            mlp_effects=mlp_effects,
            use_slot_weights=use_slot_weights,
        )
        self.objective = DiasObjective(
            relation_arities=relation_arities,
            num_slots=self.num_slots,
            use_effect_loss=use_effect_loss,
            use_precondition_loss=use_precondition_loss,
            use_applicability_loss=use_applicability_loss,
            observation_noise=observation_noise,
        )
        self.gradients = gradient_combiner(
            use_pcgrad=use_pcgrad,
            alpha=aux_loss_alpha,
            aux_active=aux_loss_alpha != 0.0
            and (use_effect_loss or use_precondition_loss),
        )
        self.automatic_optimization = False

        aux_terms = [
            name
            for name, active in (
                ("L_eff", use_effect_loss),
                ("L_pre", use_precondition_loss),
            )
            if active
        ]
        logger.info(
            "DIAS: %d slots over %d objects, %d relations, %d actions, %s parameters.",
            self.num_slots,
            num_objects,
            self.num_relations,
            num_actions,
            f"{sum(p.numel() for p in self.parameters()):,}",
        )
        logger.info(
            "Effects %s; L_aux %s at alpha %g, combined by %s.",
            "from an MLP" if mlp_effects else "lifted",
            " + ".join(aux_terms) if aux_terms else "off",
            aux_loss_alpha,
            type(self.gradients).__name__,
        )
        if use_applicability_loss:
            logger.info("L_main includes the applicability term.")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint[VERSION_KEY] = CHECKPOINT_VERSION

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        """Load a checkpoint, in an earlier format too."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get(VERSION_KEY) != CHECKPOINT_VERSION:
            checkpoint = migrate(checkpoint)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        return super().load_from_checkpoint(buffer, *args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.trainable_parameters(), lr=self.learning_rate)

    def forward(self, batch_state, batch_next_state=None):
        edge_index, edge_type = self.edges(batch_state, batch_next_state)
        node_embeddings = self.encoder(
            self.encoder.initial_features(batch_state.batch.size(0), self.device),
            edge_index,
            edge_type,
        )

        prediction = self.predictor(
            node_embeddings=node_embeddings,
            adjacency=self.dense(batch_state),
            y_action=batch_state.y_action,
            y_action_args=batch_state.y_action_args,
            tau=self.precondition_tau(self.global_step),
        )
        return prediction

    def schema_logits(self) -> SchemaLogits:
        """The learned schema as the forward pass applies it: arity-masked, with the slot weights it uses, on the CPU."""
        with torch.no_grad():
            preconditions = self.predictor.preconditions.action_logits()
            effects = self.predictor.effects.action_logits()
            slot_weights = self.predictor.selector.applied_slot_weight_logits()
            return SchemaLogits(
                preconditions=preconditions.detach().cpu(),
                effects=None if effects is None else effects.detach().cpu(),
                slot_weight_logits=slot_weights.detach().cpu(),
            )

    def apply_schema(
        self,
        adjacency: torch.Tensor,
        selection: torch.Tensor,
        y_action: torch.Tensor,
    ) -> torch.Tensor:
        """The successor state for a known slot binding, preconditions not consulted."""
        return self.predictor.apply(adjacency, selection, y_action)

    def retarget(self, num_objects: int) -> None:
        """Apply the model to an instance with `num_objects` objects."""
        if num_objects < self.num_slots:
            raise ValueError(
                f"This instance has {num_objects} objects, fewer than the model's {self.num_slots} slots; it cannot bind them all."
            )
        self.num_objects = num_objects

    def dense(self, batch) -> torch.Tensor:
        """A graph batch as the ``[B, R, O, O]`` adjacency tensor."""
        return dense_from_edges(
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            self.num_relations,
            self.num_objects,
        )

    def step(self, batch) -> Losses:
        batch_state, batch_next_state, *_ = batch
        prediction = self.forward(batch_state, batch_next_state)
        return self.objective(
            prediction,
            self.dense(batch_next_state),
            self.predictor.selector.slot_weights(batch_state.y_action),
        )

    def on_train_batch_start(self, batch, batch_idx):
        self.log("schedule/precondition_tau", self.precondition_tau(self.global_step))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        losses = self.step(batch)
        self.gradients.backward(losses.main, losses.aux, self.trainable_parameters())
        optimizer.step()

        self.log_results("train", losses.logged(self.aux_loss_alpha))
        return None

    def _eval_step(self, prefix: str, batch) -> torch.Tensor:
        losses = self.step(batch)
        self.log_results(prefix, losses.logged(self.aux_loss_alpha))
        return losses.total(self.aux_loss_alpha)

    def validation_step(self, batch, batch_idx):
        return self._eval_step("valid", batch)

    def test_step(self, batch, batch_idx):
        return self._eval_step("test", batch)

    def log_results(self, prefix: str, results: Mapping[str, torch.Tensor | int]):
        is_train = prefix == "train"
        for key, value in results.items():
            self.log(
                f"{prefix}/{key}",
                value,
                prog_bar=(key == "loss"),
                on_step=is_train,
                on_epoch=not is_train,
            )
