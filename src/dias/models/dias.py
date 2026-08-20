import io
import logging
from typing import Mapping, Optional

import lightning as L
import torch

from dias.adjacency import dense_from_edges
from dias.hparams import to_mapping
from dias.reporting.labels import resolve_label_names
from dias.reporting.plotting import PlotWorker, queue_validation_heatmaps
from dias.schedules import Schedule, as_schedule

from .checkpoints import CHECKPOINT_VERSION, VERSION_KEY, migrate
from .edges import EdgeTypeLayout
from .encoder import DEFAULT_HIDDEN_DIMS, RelationalEncoder
from .objectives import DiasObjective, Losses
from .outputs import SchemaLogits
from .pcgrad import gradient_combiner
from .prediction import PredictionModule
from .selection import resolve_num_slots

logger = logging.getLogger(__name__)


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
        self.hparams["gmean_temperature"] = to_mapping(self.gmean_temperature)

        # Everything else is handed to the module that owns it; what is kept
        # here is what something reads back later -- the losses, the optimizer,
        # the plotting callback, and the validators.
        self.num_objects = num_objects
        self.num_actions = num_actions
        self.num_relations = len(arity_relations)
        self.num_slots = resolve_num_slots(
            num_slots,
            num_objects=num_objects,
            action_arities=action_arities,
            per_action_slots=per_action_slots,
        )
        self.force_action_args = force_action_args
        self.learning_rate = learning_rate
        self.aux_loss_weight = aux_loss_weight
        self.parameter_plotting = parameter_plotting

        self.graph_edit = PredictionModule(
            arity_relations=arity_relations,
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

        self.edges = EdgeTypeLayout(
            num_relations=self.num_relations,
            num_actions=num_actions,
            action_arg_positions=action_arg_positions,
            successor_relations=successor_relations,
        )
        self.state_encoder = RelationalEncoder(
            object_dim,
            self.edges.num_edge_types,
            hidden_dims=encoder_hidden_dims or DEFAULT_HIDDEN_DIMS,
            random_node_embeddings=random_node_embeddings,
        )

        self.objective = DiasObjective(
            arity_relations=arity_relations,
            num_slots=self.num_slots,
            use_effect_loss=use_effect_loss,
            use_precondition_loss=use_precondition_loss,
        )
        # Whether the auxiliary objective contributes any gradient is fixed at
        # construction: alpha = 0, or both of its terms ablated away.
        self.gradients = gradient_combiner(
            use_pcgrad=use_pcgrad,
            auxiliary_weight=aux_loss_weight,
            auxiliary_active=aux_loss_weight != 0.0
            and (use_effect_loss or use_precondition_loss),
            dynamic_clip=pcgrad_dyn_clip,
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
        if version != CHECKPOINT_VERSION:
            logger.info("Reading a checkpoint written in format %d.", version)
            checkpoint = migrate(checkpoint)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        return super().load_from_checkpoint(buffer, *args, **kwargs)

    def teardown(self, stage: str) -> None:
        if hasattr(self, "_plot_worker"):
            self._plot_worker.shutdown()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.trainable_parameters(), lr=self.learning_rate)

    def forward(self, batch_s1, batch_s2=None):
        edge_index, edge_type = self.edges(batch_s1, batch_s2)
        node_embeddings = self.state_encoder(
            self.state_encoder.initial_features(batch_s1.batch.size(0), self.device),
            edge_index,
            edge_type,
        )

        prediction = self.graph_edit(
            node_embeddings=node_embeddings,
            adjacency=self.dense(batch_s1),
            y_action=batch_s1.y_action,
            y_action_args=batch_s1.y_action_args,
            gmean_temperature=self.gmean_temperature(self.global_step),
        )

        self._last_selection = self.graph_edit._last_selection
        self._last_y_action = self.graph_edit._last_y_action
        return prediction

    def schema_logits(self) -> SchemaLogits:
        """The learned schema as the model would apply it, on the CPU.

        The supported way to read a schema off a trained model. The arity mask
        is applied here, so the raw parameters are not a faithful record of what
        the model can express -- reading them directly finds effects on cells
        the model never uses.
        """
        with torch.no_grad():
            preconditions = self.graph_edit.preconditions.action_logits()
            effects = self.graph_edit.effects.action_logits()
            slot_weights = self.graph_edit.selector.slot_weight_logits
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
        """The successor state this model predicts for a known slot binding.

        For a model whose effects are predicted from the state, this is the
        only way to see them: `schema_logits` reports None for it, because
        there is no lifted effect tensor to report.
        """
        return self.graph_edit.apply(adjacency, selection, y_action)

    def retarget(self, num_objects: int) -> None:
        """Point the model at an instance with a different number of objects.

        A learned schema is lifted, so it transfers; the recorded instance size
        does not, and the losses and the target adjacency are built from it.
        An instance with fewer objects than the model has slots cannot bind
        them all, so it is refused rather than silently mis-scored.
        """
        if num_objects < self.num_slots:
            raise ValueError(
                f"This instance has {num_objects} objects, fewer than the "
                f"model's {self.num_slots} slots; it cannot bind them all."
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
        batch_s1, batch_s2, *_ = batch
        prediction = self.forward(batch_s1, batch_s2)
        return self.objective(
            prediction,
            self.dense(batch_s2),
            self.graph_edit.selector.slot_weights(batch_s1.y_action),
        )

    def on_train_batch_start(self, batch, batch_idx):
        # Scheduled on global_step, not epoch, so it advances per optimizer step.
        self.log(
            "scheduled_w/gmean_temperature", self.gmean_temperature(self.global_step)
        )

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

        self.log_results("train", losses.logged(self.aux_loss_weight))
        # Lightning's automatic backward is off: `gradients` set p.grad itself.
        return None

    def _eval_step(self, prefix: str, batch) -> torch.Tensor:
        losses = self.step(batch)
        self.log_results(prefix, losses.logged(self.aux_loss_weight))
        return losses.total(self.aux_loss_weight)

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
