"""Reporting a run's learned schema to W&B while it trains."""

from __future__ import annotations

import lightning as L
import torch

from dias.models.outputs import Channels
from dias.reporting import charts
from dias.reporting.labels import resolve_label_names
from dias.reporting.vega import SCHEMA, SELECTION, SLOT_WEIGHTS

GRID_COLUMNS = ("action", "relation", "row slot", "column slot", "probability")
GRID_FIELDS = {
    "row_facet": "action",
    "col_facet": "relation",
    "x": "column slot",
    "y": "row slot",
    "value": "probability",
}


def _slots(count: int) -> list[int]:
    """Slot numbers, as numbers.

    The column names say these are slots, so spelling it again in every cell cost a third of the table. It also sorts them correctly: a chart of ten slots ordered "slot 1", "slot 10", "slot 2".
    """
    return list(range(count))


class SchemaChartCallback(L.Callback):
    """Log the learned schema to W&B as interactive heatmaps.

    Six panels: add and delete effects, positive and negative preconditions, `selection_samples` slot-to-object assignments per action from one validation batch, and the slot weights. Each is drawn in the browser from the numbers it is made of, by a Vega specification registered in the run's own entity.

    Charts go out every `log_every_n_validations` validation epochs and once more when training ends. Without a W&B logger nothing is logged at all.
    """

    def __init__(
        self,
        effect_and_precondition: bool = True,
        selection: bool = True,
        selection_samples: int = 3,
        log_every_n_validations: int = 20,
    ):
        if log_every_n_validations < 1:
            raise ValueError(
                f"Expected log_every_n_validations >= 1, got {log_every_n_validations}"
            )
        if selection_samples < 1:
            raise ValueError(
                f"Expected selection_samples >= 1, got {selection_samples}"
            )
        self.effect_and_precondition = effect_and_precondition
        self.selection = selection
        self.selection_samples = selection_samples
        self.log_every_n_validations = log_every_n_validations

        self._validations = 0
        self._first_step: int | None = None
        self._selection: tuple[torch.Tensor, torch.Tensor] | None = None
        self._spec_ids: dict[str, str] | None = None

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._validations += 1

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if batch_idx == 0 and self.selection and self._due(trainer):
            self._capture_selection(pl_module, batch)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self._due(trainer):
            self.log_charts(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module) -> None:
        self.log_charts(trainer, pl_module)

    def _due(self, trainer) -> bool:
        return (
            not trainer.sanity_checking
            and self._validations % self.log_every_n_validations == 0
        )

    def _capture_selection(self, pl_module, batch) -> None:
        """The slot assignment the model makes on one validation batch.

        Re-running the forward pass is what keeps the model free of a side-channel written for a plotter. The initial node features are random, though, so it runs on a forked generator: a diagnostic must not shift the stream the run would otherwise have drawn from.
        """
        devices = [pl_module.device] if pl_module.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            prediction = pl_module(batch[0], batch[1])
        self._selection = (
            prediction.selection.detach().cpu(),
            batch[0].y_action.detach().cpu(),
        )

    def log_charts(self, trainer, pl_module) -> None:
        """Send the current schema to the run, if it is a W&B one."""
        if not isinstance(trainer.logger, L.pytorch.loggers.WandbLogger):
            return
        run = trainer.logger.experiment
        if self._spec_ids is None:
            self._spec_ids = charts.register(run.entity)

        media = self._charts(
            trainer, pl_module, self._spec_ids, self._panel_step(trainer)
        )
        if media:
            run.log(media, step=int(trainer.global_step))

    def _panel_step(self, trainer) -> int:
        """The step each panel's history query names, fixed at the first log.

        W&B rewrites a panel's configuration every time the chart is logged, so naming the newest step here would keep pointing the page at a table that is still uploading, which is what made a live run's panels blink in and out. Written once, the configuration stops moving; the slider opens on the first logged schema and reaches the rest.
        """
        if self._first_step is None:
            self._first_step = int(trainer.global_step)
        return self._first_step

    def _charts(self, trainer, pl_module, specs: dict[str, str], step) -> dict:
        schema = pl_module.schema_logits()
        actions, relations, objects = resolve_label_names(
            getattr(trainer, "datamodule", None),
            pl_module.num_actions,
            pl_module.num_relations,
            pl_module.num_objects,
        )

        media = {}
        if self.effect_and_precondition:
            media |= self._schema_charts(schema, actions, relations, specs, step)
        if self.selection and self._selection is not None:
            media |= self._selection_charts(schema, actions, objects, specs, step)
        return media

    def _schema_charts(self, schema, actions, relations, specs, step) -> dict:
        panels = []
        # A model whose effects come from the state has no lifted effects to show; its preconditions are still a schema.
        if schema.effects is not None:
            effects = Channels.from_logits(schema.effects)
            panels += [
                ("add_effects", "Add effects", effects.positive),
                ("delete_effects", "Delete effects", effects.negative),
            ]
        preconditions = Channels.from_logits(schema.preconditions)
        panels += [
            (
                "positive_preconditions",
                "Positive preconditions",
                preconditions.positive,
            ),
            (
                "negative_preconditions",
                "Negative preconditions",
                preconditions.negative,
            ),
        ]

        slots = _slots(panels[0][2].shape[-1])
        return {
            key: charts.heatmap(
                specs[SCHEMA],
                title,
                values.numpy(),
                columns=GRID_COLUMNS,
                axis_labels=(actions, relations, slots, slots),
                fields=GRID_FIELDS,
                step=step,
            )
            for key, title, values in panels
        }

    def _selection_charts(self, schema, actions, objects, specs, step) -> dict:
        selection, y_action = self._selection
        by_action: dict[int, list[torch.Tensor]] = {}
        for index, action in enumerate(y_action.tolist()):
            by_action.setdefault(int(action), []).append(selection[index])

        order = sorted(by_action)
        # One column per sample. A batch does not hold every action equally often and the grid has to be rectangular, so the rarest one sets the width: a batch short of samples shows fewer, not a ragged chart.
        count = min(self.selection_samples, *(len(by_action[a]) for a in order))
        assignments = torch.stack(
            [torch.stack(by_action[action][:count]) for action in order]
        )
        weights = torch.sigmoid(schema.slot_weight_logits)

        return {
            "selection": charts.heatmap(
                specs[SELECTION],
                "Object selection",
                assignments.numpy(),
                columns=("action", "sample", "slot", "object", "weight"),
                axis_labels=(
                    [actions[action] for action in order],
                    list(range(count)),
                    _slots(assignments.shape[2]),
                    objects,
                ),
                fields={
                    "row_facet": "action",
                    "col_facet": "sample",
                    "x": "object",
                    "y": "slot",
                    "value": "weight",
                },
                step=step,
            ),
            "slot_weights": charts.heatmap(
                specs[SLOT_WEIGHTS],
                "Slot weights",
                weights.numpy(),
                columns=("action", "slot", "weight"),
                axis_labels=(actions, _slots(weights.shape[-1])),
                # One strip per action; the panel has no second axis to place.
                fields={"row_facet": "action", "x": "slot", "value": "weight"},
                step=step,
            ),
        }
