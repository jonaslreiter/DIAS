"""The learned schema as W&B custom charts.

One panel is few numbers (actions x relations x slots x slots), so they are logged as numbers and drawn in the browser rather than rendered here. `wandb` is imported inside the functions, so importing this module does not require it.
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


# The chart labels cells to two decimals and tooltips to four; a full float repr would be most of every row for nothing.
DIGITS = 4

# What W&B calls a Vega-Lite specification, and who may see the preset. The SDK upper-cases the access level; the mutation below talks to the backend directly and must do it itself.
SPEC_TYPE = "vega2"
ACCESS = "private"


def cells(values: np.ndarray, *axis_labels: Sequence[str]) -> list[list]:
    """Every cell of `values` as one table row: its labels, then the number. `axis_labels` names each axis' ticks, outermost first."""
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
    """Create the preset, or update the one already there. The SDK wraps only creation, and creating a preset that exists fails, so an edited specification would otherwise never reach W&B."""
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

    A preset that already matches is left alone, so the ordinary run makes one read and no writes. If W&B is unreachable the ids still follow from the names and only the drawn panel suffers.
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

    Built on first use: defining the subclass needs wandb imported, and importing this module must not.
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
                    "This wandb does not query summaryTable, so the panel's step slider could not be set up; it will show the last logged schema only."
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

    `columns` names one column per axis of `values` plus the value itself, and `fields` maps the specification's placeholders onto them.
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
