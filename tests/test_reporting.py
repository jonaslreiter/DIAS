"""Tests for the charts a run reports its learned schema through."""

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import dias
from checkpoints import build_model
from dias.reporting.callback import SchemaChartCallback
from dias.reporting.charts import cells
from dias.reporting.vega import CELL, SCHEMA, SELECTION, SLOT_WEIGHTS, SPECS

SPEC_IDS = {name: f"someone/{name}" for name in SPECS}


class CellsTest(unittest.TestCase):
    def test_every_cell_becomes_a_labelled_row(self):
        values = np.array([[0.0, 0.25], [0.5, 1.0]])
        self.assertEqual(
            cells(values, ["a", "b"], ["x", "y"]),
            [["a", "x", 0.0], ["a", "y", 0.25], ["b", "x", 0.5], ["b", "y", 1.0]],
        )

    def test_the_shape_of_the_tensor_decides_the_shape_of_the_table(self):
        rows = cells(np.zeros((2, 3, 4, 4)), *([["l"] * 4] * 4))
        self.assertEqual(len(rows), 2 * 3 * 4 * 4)
        self.assertEqual(len(rows[0]), 5)


class VegaSpecTest(unittest.TestCase):
    def test_each_spec_is_shaped_for_the_chart_it_draws(self):
        self.assertEqual(set(SPECS[SCHEMA]["facet"]), {"row", "column"})
        self.assertEqual(set(SPECS[SELECTION]["facet"]), {"row", "column"})
        # Slot weights are one strip per action, so the panel has no y axis.
        self.assertEqual(set(SPECS[SLOT_WEIGHTS]["facet"]), {"row"})
        self.assertNotIn("y", SPECS[SLOT_WEIGHTS]["spec"]["layer"][0]["encoding"])

    def test_a_value_cell_is_square_and_the_same_size_everywhere(self):
        for name, spec in SPECS.items():
            panel = spec.get("spec", spec)
            with self.subTest(spec=name):
                self.assertEqual(panel["width"], {"step": CELL})
                self.assertEqual(
                    panel["height"],
                    {"step": CELL} if "y" in panel["layer"][0]["encoding"] else CELL,
                )

    def test_nothing_names_the_axis_it_sits_on(self):
        """The values say what they are, so a title would repeat it per panel."""
        for name, spec in SPECS.items():
            panel = spec.get("spec", spec)
            titled = [
                channel
                for channel, encoding in panel["layer"][0]["encoding"].items()
                if channel in ("x", "y")
                and encoding.get("title", "missing") is not None
            ] + [
                side
                for side, encoding in spec.get("facet", {}).items()
                if encoding.get("title", "missing") is not None
            ]
            with self.subTest(spec=name):
                self.assertEqual(titled, [])

    def test_no_panel_carries_a_colour_bar(self):
        """Every cell is labelled with its value, so a legend would repeat it.

        `false`, not the schema's `null`, which W&B draws a bar for anyway.
        """
        for name, spec in SPECS.items():
            panel = spec.get("spec", spec)
            for layer in panel["layer"]:
                with self.subTest(spec=name, mark=layer["mark"]["type"]):
                    self.assertIs(layer["encoding"]["color"]["legend"], False)


class ChartTest(unittest.TestCase):
    def test_a_heatmap_binds_its_table_to_the_specification(self):
        from dias.reporting.charts import heatmap

        chart = heatmap(
            "someone/dias-selection-heatmap",
            "Add effects",
            np.zeros((2, 2)),
            columns=("row", "column", "probability"),
            axis_labels=(["a", "b"], ["x", "y"]),
            fields={"x": "column", "y": "row", "value": "probability"},
        )
        self.assertEqual(chart.table.columns, ["row", "column", "probability"])
        self.assertEqual(len(chart.table.data), 4)


def trainer(logger=None, sanity_checking=False, global_step=7, datamodule=None):
    return SimpleNamespace(
        logger=logger,
        sanity_checking=sanity_checking,
        global_step=global_step,
        datamodule=datamodule,
    )


class CadenceTest(unittest.TestCase):
    def test_charts_go_out_every_nth_validation(self):
        callback = SchemaChartCallback(log_every_n_validations=3)
        due = []
        for _ in range(6):
            callback.on_validation_epoch_start(trainer(), None)
            due.append(callback._due(trainer()))
        self.assertEqual(due, [False, False, True, False, False, True])

    def test_the_sanity_check_epoch_does_not_count(self):
        callback = SchemaChartCallback(log_every_n_validations=1)
        callback.on_validation_epoch_start(trainer(sanity_checking=True), None)
        self.assertEqual(callback._validations, 0)
        self.assertFalse(callback._due(trainer(sanity_checking=True)))

    def test_an_interval_below_one_is_refused(self):
        with self.assertRaisesRegex(ValueError, "log_every_n_validations >= 1"):
            SchemaChartCallback(log_every_n_validations=0)


class ChartSetTest(unittest.TestCase):
    """What a logging pass produces, without a W&B run to send it to."""

    def charts(self, callback, **overrides):
        model, _ = build_model(**overrides)
        return callback._charts(trainer(), model, SPEC_IDS)

    def test_the_full_model_reports_effects_preconditions_and_selection(self):
        callback = SchemaChartCallback()
        callback._selection = (torch.rand(2, 3, 4), torch.tensor([0, 1]))
        self.assertEqual(
            set(self.charts(callback)),
            {
                "add_effects",
                "delete_effects",
                "positive_preconditions",
                "negative_preconditions",
                "selection",
                "slot_weights",
            },
        )

    def test_mlp_effects_have_no_lifted_panel_to_draw(self):
        charts = self.charts(SchemaChartCallback(), mlp_effects=True)
        self.assertEqual(
            set(charts), {"positive_preconditions", "negative_preconditions"}
        )

    def test_the_selection_grid_holds_the_requested_number_of_samples(self):
        callback = SchemaChartCallback(selection_samples=2)
        # Three samples of action 0, two of action 1, four slots, five objects.
        callback._selection = (torch.rand(5, 4, 5), torch.tensor([0, 1, 0, 1, 0]))
        chart = self.charts(callback)["selection"]
        self.assertEqual(
            chart.table.columns, ["action", "sample", "slot", "object", "weight"]
        )
        self.assertEqual(len(chart.table.data), 2 * 2 * 4 * 5)

    def test_an_action_short_of_samples_narrows_the_grid(self):
        """The grid is rectangular, so the rarest action sets its width."""
        callback = SchemaChartCallback(selection_samples=3)
        callback._selection = (torch.rand(3, 4, 5), torch.tensor([0, 0, 1]))
        chart = self.charts(callback)["selection"]
        self.assertEqual(len(chart.table.data), 2 * 1 * 4 * 5)

    def test_each_switch_drops_its_own_panels(self):
        callback = SchemaChartCallback(effect_and_precondition=False)
        callback._selection = (torch.rand(1, 3, 4), torch.tensor([0]))
        self.assertEqual(set(self.charts(callback)), {"selection", "slot_weights"})

    def test_a_table_holds_one_row_per_cell_of_the_tensor(self):
        model, _ = build_model()
        charts = SchemaChartCallback()._charts(trainer(), model, SPEC_IDS)
        chart = charts["add_effects"]
        schema = model.schema_logits()
        self.assertEqual(len(chart.table.data), schema.effects[..., 0].numel())
        self.assertEqual(
            chart.table.columns,
            ["action", "relation", "row slot", "column slot", "probability"],
        )


class LoggingTest(unittest.TestCase):
    def test_nothing_is_logged_without_a_wandb_logger(self):
        callback = SchemaChartCallback()
        model, _ = build_model()
        callback.log_charts(trainer(logger=None), model)
        callback.log_charts(trainer(logger=SimpleNamespace()), model)
        self.assertIsNone(callback._spec_ids)


class _RandomForwardModule:
    """Stands in for a model whose forward draws from the global generator.

    `RelationalEncoder.initial_features` seeds every object with noise, so a forward pass really does spend the stream this test is about.
    """

    device = torch.device("cpu")

    def __call__(self, *_args):
        torch.randn(16)
        return SimpleNamespace(selection=torch.zeros(1, 2, 2))


class SelectionCaptureTest(unittest.TestCase):
    def test_capturing_a_selection_leaves_the_random_stream_alone(self):
        """A diagnostic must not shift the draws the run would have made."""
        callback = SchemaChartCallback()
        batch = (SimpleNamespace(y_action=torch.tensor([0])), None)

        torch.manual_seed(0)
        undisturbed = torch.randn(3)

        torch.manual_seed(0)
        callback._capture_selection(_RandomForwardModule(), batch)
        self.assertTrue(torch.equal(torch.randn(3), undisturbed))
        self.assertEqual(callback._selection[1].tolist(), [0])


class ImportLayeringTest(unittest.TestCase):
    def test_importing_the_model_does_not_require_wandb(self):
        """Reporting is the only part that talks to W&B, and it imports it late.

        Only wandb is blocked here: matplotlib arrives through torchmetrics, which lightning imports, so its absence cannot be arranged from inside this package. `test_matplotlib_left_the_package` covers dias itself.
        """
        code = "import sys;sys.modules['wandb'] = None;import dias.models.dias, dias.reporting.callback, dias.reporting.charts"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_matplotlib_left_the_package(self):
        """The charts are drawn in the browser."""
        root = Path(dias.__file__).parent
        offenders = [
            str(path.relative_to(root))
            for path in root.rglob("*.py")
            if "matplotlib" in path.read_text()
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
