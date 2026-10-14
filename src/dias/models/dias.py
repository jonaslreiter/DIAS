import io
import logging
from typing import Literal, Mapping

import lightning as L
import torch

from dias.adjacency import dense_from_edges
from dias.hparams import to_mapping
from dias.schedules import Schedule, as_schedule

from .checkpoints import CHECKPOINT_VERSION, UNVERSIONED, VERSION_KEY, migrate
from .edges import EdgeTypeLayout
from .encoder import DEFAULT_HIDDEN_DIMS, RelationalEncoder
from .objectives import DiasObjective, Losses
from .outputs import Channels, SchemaLogits
from .pcgrad import gradient_combiner
from .prediction import SuccessorPredictor
from .rejection import admitted_fraction
from .selection import resolve_num_slots

logger = logging.getLogger(__name__)


class DIAS(L.LightningModule):
    """A relational encoder, a slot assignment, and one lifted schema per action. Defaults are the paper's full model."""

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
        learn_preconditions: bool = True,
        action_arities: list[int] | None = None,
        action_arg_positions: list[list[int]] | None = None,
        successor_relations: bool = True,
        random_node_embeddings: bool = True,
        precondition_tau: Schedule | Mapping | None = None,
        pcgrad_norm_cap: bool = True,
        state_dependent_effects: bool = False,
        use_slot_weights: bool = True,
        use_precondition_loss: bool = True,
        precondition_loss: Literal["restrict", "reject"] = "restrict",
        rejection_samples: int = 16,
        use_effect_loss: bool = True,
        use_pcgrad: bool = True,
        encoder_hidden_dims: list[int] | None = None,
        schema_threshold: float = 0.5,
    ):
        """
        Args:
            num_objects: Objects in the instance. (linked from the datamodule)
            relation_arities: Arity (1 or 2) of every relation, in channel order. (linked from the datamodule)
            num_actions: Action schemas to learn, one per action of the domain. (linked from the datamodule)
            object_dim: Width of the object embeddings.
            num_slots: Parameter slots per schema, clamped to num_objects. Required unless per_action_slots.
            learning_rate: Adam step size.
            aux_loss_alpha: Alpha, the weight of the auxiliary objective against the adjacency loss.
            force_action_args: Bind the slots of observed action arguments instead of assigning them.
            per_action_slots: Give every action its own arity in slots, rather than num_slots each.
            learn_preconditions: Gate the predicted change on the learned preconditions.
            action_arities: Arity of every action. (linked from the datamodule)
            action_arg_positions: Argument positions the labels observe, per action. (linked from the datamodule)
            successor_relations: Show the encoder what the transition changed.
            random_node_embeddings: Start each object from noise, so symmetric objects differ.
            precondition_tau: Schedule for tau, the root degree of the relaxed precondition product. None holds it at 1, the geometric mean.
            pcgrad_norm_cap: Cap the projected auxiliary gradient at the primary one's norm.
            state_dependent_effects: Predict effects from the state with an MLP instead of learning one lifted tensor per action.
            use_slot_weights: Learn which slots an action uses, which is its recovered arity.
            use_precondition_loss: Include the precondition auxiliary term.
            precondition_loss: Which precondition term: restrict asks every precondition cell to require or forbid its atom (the paper's); reject asks the preconditions to reject as many random ground actions of the batch's states as they can.
            rejection_samples: Random ground actions the rejection loss scores per batch state and action.
            use_effect_loss: Include the minimal-effects auxiliary term.
            use_pcgrad: Project the auxiliary gradient rather than adding a weighted sum.
            encoder_hidden_dims: Widths of the relational convolutions.
            schema_threshold: Probability at which the learned domain written beside each checkpoint binarizes slots, effects and preconditions (Eq. 14).
        """
        super().__init__()
        self.save_hyperparameters()
        self.precondition_tau = as_schedule(precondition_tau)
        # Hyper-parameters must stay primitive; see dias.hparams.
        self.hparams["precondition_tau"] = to_mapping(self.precondition_tau)

        # Kept here because something reads them back; the rest goes to the module that owns it.
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
        self.rejection_samples = rejection_samples

        self.edges = EdgeTypeLayout(
            num_relations=self.num_relations,
            num_actions=num_actions,
            action_arg_positions=action_arg_positions,
            successor_relations=successor_relations,
        )
        self.encoder = RelationalEncoder(
            object_dim,
            self.edges.num_edge_types,
            hidden_dims=encoder_hidden_dims or DEFAULT_HIDDEN_DIMS,
            random_node_embeddings=random_node_embeddings,
        )

        self.predictor = SuccessorPredictor(
            relation_arities=relation_arities,
            num_actions=num_actions,
            num_slots=self.num_slots,
            object_dim=object_dim,
            force_action_args=force_action_args,
            per_action_slots=per_action_slots,
            learn_preconditions=learn_preconditions,
            action_arities=action_arities,
            state_dependent_effects=state_dependent_effects,
            use_slot_weights=use_slot_weights,
        )

        self.objective = DiasObjective(
            relation_arities=relation_arities,
            num_slots=self.num_slots,
            use_effect_loss=use_effect_loss,
            use_precondition_loss=use_precondition_loss,
            precondition_loss=precondition_loss,
        )
        self.gradients = gradient_combiner(
            use_pcgrad=use_pcgrad,
            alpha=aux_loss_alpha,
            auxiliary_active=aux_loss_alpha != 0.0
            and (use_effect_loss or use_precondition_loss),
            norm_cap=pcgrad_norm_cap,
        )

        self.automatic_optimization = False

        auxiliary = [
            name
            for name, active in (
                ("effects", use_effect_loss),
                (f"preconditions ({precondition_loss})", use_precondition_loss),
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
            "Effects are %s; auxiliary losses %s at weight %g, combined by %s.",
            (
                "predicted from the state"
                if state_dependent_effects
                else "a lifted tensor"
            ),
            ", ".join(auxiliary) if auxiliary else "off",
            aux_loss_alpha,
            type(self.gradients).__name__,
        )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint[VERSION_KEY] = CHECKPOINT_VERSION

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        """Load a checkpoint, bringing an older layout up to date on the way in."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        version = checkpoint.get(VERSION_KEY, UNVERSIONED)
        if version != CHECKPOINT_VERSION:
            logger.info("Reading a checkpoint written in format %d.", version)
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
        """The learned schema as the model applies it, arity-masked, on the CPU.

        The supported way to read a schema off a trained model: the raw parameters hold noise at masked cells, and the stored slot-weight logits are not what the forward pass applies when the weights are not learned.
        """
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
        """The successor state predicted for a known slot binding, preconditions not consulted."""
        return self.predictor.apply(adjacency, selection, y_action)

    def retarget(self, num_objects: int) -> None:
        """Point the model at an instance with a different number of objects."""
        if num_objects < self.num_slots:
            raise ValueError(
                f"This instance has {num_objects} objects, fewer than the model's {self.num_slots} slots; it cannot bind them all."
            )
        self.num_objects = num_objects

    def dense(self, batch) -> torch.Tensor:
        """One graph batch as the ``[B, R, O, O]`` tensor the schema acts on."""
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
        successor = self.dense(batch_next_state)
        return self.objective(
            prediction,
            successor,
            self.predictor.selector.slot_weights(batch_state.y_action),
            admitted=self.admitted(batch_state, prediction.input_adjacency, successor),
        )

    def admitted(
        self, batch_state, state: torch.Tensor, successor: torch.Tensor
    ) -> torch.Tensor | None:
        """What the rejection loss scores, ``(2B, A)``: for every state of the batch, successors included, and every action, the fraction of random ground actions the preconditions admit. None when the objective does not use it.

        Every action is scored on every state, not only on those it was taken in: a state where an action is inapplicable is where its preconditions have the most to reject.
        """
        if not self.objective.rejects:
            return None
        weights = self.predictor.selector.action_slot_weights(
            batch_state.y_action, batch_state.y_action_args
        )
        with torch.amp.autocast(device_type=state.device.type, enabled=False):
            return admitted_fraction(
                torch.cat([state, successor]).float(),
                Channels.from_logits(self.predictor.preconditions.action_logits()),
                weights.float(),
                samples=self.rejection_samples,
                tau=self.precondition_tau(self.global_step),
            )

    def on_train_batch_start(self, batch, batch_idx):
        self.log("schedule/precondition_tau", self.precondition_tau(self.global_step))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        losses = self.step(batch)
        self.gradients.backward(
            losses.adjacency, losses.auxiliary, self.trainable_parameters()
        )
        optimizer.step()

        self.log_results("train", losses.logged(self.aux_loss_alpha))
        # Automatic optimization is off: `gradients` set p.grad itself.
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
