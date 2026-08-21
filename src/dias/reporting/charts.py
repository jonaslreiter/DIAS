"""The learned schema as W&B custom charts.

The numbers behind one panel are few -- an add-effect panel is actions x
relations x slots x slots floats -- so they are logged as the numbers
themselves and drawn in the browser, rather than rendered here into an image
two orders of magnitude larger than the data it shows.

`wandb` is imported inside the functions. Importing this module must not
require it, and nothing here runs unless a run actually logs charts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

from dias.reporting.vega import SPECS

logger = logging.getLogger(__name__)


# These tables are read as a picture, and full float repr costs three times
# what it is worth: a probability written out to 17 digits is 19 bytes of a
# ~45 byte row. The chart labels cells to two decimals and tooltips to four.
DIGITS = 4

# What W&B calls a Vega-Lite specification, and who may see the preset. The
# SDK takes the access level in lower case and upper-cases it for the backend;
# the mutation below talks to the backend directly and has to do that itself.
SPEC_TYPE = "vega2"
ACCESS = "private"


def cells(values: np.ndarray, *axis_labels: Sequence[str]) -> list[list]:
    """Every cell of `values` as one table row: its labels, then the number.

    `axis_labels` names the ticks of each axis, outermost first, so the shape
    of the tensor decides the shape of the table.
    """
    return [
        [
            *(labels[i] for labels, i in zip(axis_labels, index)),
            round(float(values[index]), DIGITS),
        ]
        for index in np.ndindex(values.shape)
    ]


_LIST_CHARTS = """
query DiasCharts($entity: String!, $type: String!) {
  customCharts(entity: $entity, type: $type, first: 100) {
    edges { node { name spec } }
  }
}
"""

_UPDATE_CHART = """
mutation UpdateDiasChart(
  $entity: String!, $name: String!, $displayName: String!,
  $type: String!, $access: String!, $spec: JSONString!
) {
  updateCustomChart(input: {
    entity: $entity, name: $name, displayName: $displayName,
    type: $type, access: $access, spec: $spec
  }) { chart { id } }
}
"""


def _registered(api, entity: str) -> dict[str, dict]:
    """The Vega specifications this entity already holds, by name."""
    from wandb_gql import gql

    charts = api.client.execute(
        gql(_LIST_CHARTS), {"entity": entity, "type": SPEC_TYPE}
    )
    return {
        edge["node"]["name"]: json.loads(edge["node"]["spec"])
        for edge in charts["customCharts"]["edges"]
    }


def _write(api, entity: str, name: str, spec: dict, exists: bool) -> None:
    """Create the preset, or update the one that is already there.

    The Python SDK wraps only creation, and creating a preset that exists
    fails, so a specification edited here would otherwise never reach W&B --
    which is how a fixed chart can go on being drawn the old way forever.
    """
    from wandb_gql import gql

    if not exists:
        api.create_custom_chart(
            entity=entity,
            name=name,
            display_name=name,
            spec_type=SPEC_TYPE,
            access=ACCESS,
            spec=spec,
        )
        logger.info("Registered the W&B chart preset %s/%s.", entity, name)
        return

    api.client.execute(
        gql(_UPDATE_CHART),
        {
            "entity": entity,
            "name": name,
            "displayName": name,
            "type": SPEC_TYPE,
            "access": ACCESS.upper(),
            "spec": json.dumps(spec),
        },
    )
    logger.info("Updated the W&B chart preset %s/%s.", entity, name)


def register(entity: str) -> dict[str, str]:
    """The id of each chart specification within `entity`, bringing it up to date.

    A preset that is already there and already matches is left alone, so the
    ordinary run makes one read and no writes. If W&B cannot be reached the
    ids still follow from the names, and only the drawn panel suffers -- the
    tables are logged either way.
    """
    import wandb

    ids = {name: f"{entity}/{name}" for name in SPECS}
    try:
        api = wandb.Api()
        existing = _registered(api, entity)
    except Exception as error:
        logger.warning("Could not read this entity's chart presets: %s", error)
        return ids

    for name, spec in SPECS.items():
        if existing.get(name) == spec:
            continue
        try:
            _write(api, entity, name, spec, exists=name in existing)
        except Exception as error:
            logger.warning("Could not write the chart preset %s: %s", name, error)
    return ids


@lru_cache(maxsize=1)
def _spec_class():
    """`CustomChartSpec`, but reading the table from history, not the summary.

    `wandb.plot_table` binds a panel to `summaryTable`, which holds only the
    last table logged under a key -- so a panel logged every N validations
    showed the newest schema and nothing before it, where the images it
    replaced got a step slider for free.

    Reproducing what the editor writes when a panel's query is switched by
    hand takes three things it does not: `historyTable` in place of
    `summaryTable`, its `extraKeys` and `index` arguments -- without `index`
    the panel has no step to draw and comes up empty -- and `showStepSelector`,
    which puts the slider on screen.

    The editor also writes `defaultViewedStepIndex`, the slider's opening
    position. It is left out on purpose: the panel's configuration is rewritten
    on every log, and naming a position that the table for it has not finished
    uploading to is what made a live run's panels blink in and out.

    Built on first use: defining the subclass needs wandb imported, and
    importing this module must not.
    """
    from wandb.plot.custom_chart import CustomChartSpec

    @dataclass
    class HistoryChartSpec(CustomChartSpec):
        step: int = 0

        @property
        def config_value(self) -> dict:
            config = super().config_value
            config["panel_config"]["showStepSelector"] = True
            queries = config["panel_config"]["userQuery"]["queryFields"]
            summary = [
                field
                for query in queries
                for field in query["fields"]
                if field["name"] == "summaryTable"
            ]
            if not summary:
                logger.warning(
                    "This wandb does not query summaryTable, so the panel's step "
                    "slider could not be set up; it will show the last logged "
                    "schema only."
                )
            for field in summary:
                field["name"] = "historyTable"
                field["args"] = [
                    {"name": "extraKeys", "value": []},
                    {"name": "tableKey", "value": self.table_key},
                    {"name": "index", "value": self.step},
                ]
            return config

    return HistoryChartSpec


def heatmap(
    spec_id: str,
    title: str,
    values: np.ndarray,
    columns: Sequence[str],
    axis_labels: Sequence[Sequence[str]],
    fields: Mapping[str, str],
    step: int,
):
    """One custom chart: `values` as a table, bound to a registered spec.

    `columns` names the table's columns -- one per axis of `values`, then the
    value itself -- and `fields` maps the specification's placeholders onto
    them. `step` names the step the panel's history query asks for; see
    `SchemaChartCallback._panel_step` for why it stays put.
    """
    import wandb
    from wandb.plot.custom_chart import CustomChart

    return CustomChart(
        table=wandb.Table(columns=list(columns), data=cells(values, *axis_labels)),
        spec=_spec_class()(
            spec_name=spec_id,
            fields=dict(fields),
            string_fields={"title": title},
            step=step,
        ),
    )
