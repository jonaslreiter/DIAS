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
from dias.reporting.vega import SPECS

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
    def test_each_spec_facets_the_way_its_name_says(self):
        self.assertNotIn("facet", SPECS["dias-heatmap"])
        self.assertEqual(set(SPECS["dias-heatmap-rows"]["facet"]), {"row"})
        self.assertEqual(set(SPECS["dias-heatmap-grid"]["facet"]), {"row", "column"})

    def test_no_panel_carries_a_colour_bar(self):
        """Every cell is labelled with its value; a legend repeats it.

        `false`, not the schema's `null`, which W&B draws a bar for anyway.
        """
        for name, spec in SPECS.items():
            panel = spec.get("spec", spec)
            for layer in panel["layer"]:
                with self.subTest(spec=name, mark=layer["mark"]["type"]):
                    self.assertIs(layer["encoding"]["color"]["legend"], False)


class ChartQueryTest(unittest.TestCase):
    def test_the_panel_reads_the_table_from_history_at_a_step(self):
        """The two together are what put a working step slider under the panel.

        wandb.plot_table binds to summaryTable, which holds only the last table
        logged under a key. historyTable keeps every step, but without `index`
        the panel has no step to draw and comes up empty.
        """
        from dias.reporting.charts import heatmap

        chart = heatmap(
            "someone/dias-heatmap",
            "Add effects",
            np.zeros((2, 2)),
            columns=("row", "column", "probability"),
            axis_labels=(["a", "b"], ["x", "y"]),
            fields={"x": "column", "y": "row", "value": "probability"},
            step=150,
        )
        chart.set_key("add_effects")
        panel = chart.spec.config_value["panel_config"]
        query = panel["userQuery"]
        fields = query["queryFields"][0]["fields"]
        names = [f["name"] for f in fields]

        self.assertIn("historyTable", names)
        self.assertNotIn("summaryTable", names)

        (history,) = [f for f in fields if f["name"] == "historyTable"]
        self.assertEqual(
            history["args"],
            [
                {"name": "extraKeys", "value": []},
                {"name": "tableKey", "value": "add_effects_table"},
                {"name": "index", "value": 150},
            ],
        )
        self.assertIs(panel["showStepSelector"], True)
        # Deliberately absent: naming an opening position the table has not
        # finished uploading to made a live run's panels blink in and out.
        self.assertNotIn("defaultViewedStepIndex", panel)


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


class PanelStepTest(unittest.TestCase):
    def test_the_query_names_one_step_for_the_whole_run(self):
        """A configuration that keeps moving points the page at a table that

        has not finished uploading, which is what made live panels blink.
        """
        callback = SchemaChartCallback()
        named = [callback._panel_step(trainer(global_step=s)) for s in (50, 100, 150)]
        self.assertEqual(named, [50, 50, 50])


class ChartSetTest(unittest.TestCase):
    """What a logging pass produces, without a W&B run to send it to."""

    def charts(self, callback, **overrides):
        model, _ = build_model(**overrides)
        return callback._charts(trainer(), model, SPEC_IDS, 0)

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

    def test_state_dependent_effects_have_no_lifted_panel_to_draw(self):
        """The MLP baseline: preconditions are a schema, its effects are not."""
        charts = self.charts(SchemaChartCallback(), state_dependent_effects=True)
        self.assertEqual(
            set(charts), {"positive_preconditions", "negative_preconditions"}
        )

    def test_each_switch_drops_its_own_panels(self):
        callback = SchemaChartCallback(effect_and_precondition=False)
        callback._selection = (torch.rand(1, 3, 4), torch.tensor([0]))
        self.assertEqual(set(self.charts(callback)), {"selection", "slot_weights"})

    def test_a_table_holds_one_row_per_cell_of_the_tensor(self):
        model, _ = build_model()
        charts = SchemaChartCallback()._charts(trainer(), model, SPEC_IDS, 0)
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

    `RelationalEncoder.initial_features` seeds every object with noise, so a
    forward pass really does spend the stream this test is about.
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

        Only wandb is blocked here: matplotlib arrives through torchmetrics,
        which lightning imports, so its absence cannot be arranged from inside
        this package. `test_matplotlib_left_the_package` covers dias itself.
        """
        code = (
            "import sys;"
            "sys.modules['wandb'] = None;"
            "import dias.models.dias, dias.reporting.callback, dias.reporting.charts"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_matplotlib_left_the_package(self):
        """It went with the rendered heatmaps; the charts are drawn in the browser."""
        root = Path(dias.__file__).parent
        offenders = [
            str(path.relative_to(root))
            for path in root.rglob("*.py")
            if "matplotlib" in path.read_text()
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
