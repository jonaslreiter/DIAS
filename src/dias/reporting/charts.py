"""The learned schema as W&B custom charts: tables of numbers drawn by the Vega-Lite specifications in `vega`."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np

from dias.reporting.vega import SPECS

logger = logging.getLogger(__name__)

DIGITS = 4


def cells(values: np.ndarray, *axis_labels: Sequence[str]) -> list[list]:
    """Every cell of `values` as a table row: its label on each axis, outermost first, then the value."""
    return [
        [
            *(labels[i] for labels, i in zip(axis_labels, index)),
            round(float(values[index]), DIGITS),
        ]
        for index in np.ndindex(values.shape)
    ]


def register(entity: str) -> dict[str, str]:
    """The id of each specification in `entity`, creating those it lacks."""
    import wandb

    ids = {}
    for name, spec in SPECS.items():
        ids[name] = f"{entity}/{name}"
        try:
            wandb.Api().create_custom_chart(
                entity=entity,
                name=name,
                display_name=name,
                spec_type="vega2",
                access="private",
                spec=spec,
            )
            logger.info("Created the W&B chart preset %s.", ids[name])
        except Exception as error:  # noqa: BLE001
            logger.debug("Did not create %s: %s", ids[name], error)
    return ids


def heatmap(
    spec_id: str,
    title: str,
    values: np.ndarray,
    columns: Sequence[str],
    axis_labels: Sequence[Sequence[str]],
    fields: Mapping[str, str],
):
    """`values` as a table drawn by the specification `spec_id`; `columns` names its axes and the value, and `fields` binds the specification's fields to them."""
    import wandb

    return wandb.plot_table(
        spec_id,
        wandb.Table(columns=list(columns), data=cells(values, *axis_labels)),
        dict(fields),
        {"title": title},
    )
