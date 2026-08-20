"""The learned schema as W&B custom charts.

The numbers behind one panel are few -- an add-effect panel is actions x
relations x slots x slots floats -- so they are logged as the numbers
themselves and drawn in the browser, rather than rendered here into an image
two orders of magnitude larger than the data it shows.

`wandb` is imported inside the functions. Importing this module must not
require it, and nothing here runs unless a run actually logs charts.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np

from dias.reporting.vega import SPECS

logger = logging.getLogger(__name__)


# These tables are read as a picture, and full float repr costs three times
# what it is worth: a probability written out to 17 digits is 19 bytes of a
# ~45 byte row. The chart labels cells to two decimals and tooltips to four.
DIGITS = 4


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


def register(entity: str) -> dict[str, str]:
    """The id of each chart specification within `entity`, creating them once.

    The id follows from the name, so a preset that already exists -- the
    ordinary case from an entity's second run onward -- or an API that cannot
    be reached costs nothing but the drawn panel on a first run. The tables are
    logged either way.
    """
    import wandb

    ids = {name: f"{entity}/{name}" for name in SPECS}
    try:
        api = wandb.Api()
    except Exception as error:
        logger.debug("No W&B API to register chart presets with (%s)", error)
        return ids

    for name, spec in SPECS.items():
        try:
            ids[name] = api.create_custom_chart(
                entity=entity,
                name=name,
                display_name=name,
                spec_type="vega2",
                access="private",
                spec=spec,
            )
            logger.info("Registered the W&B chart preset %s.", ids[name])
        except Exception as error:
            # W&B reports this on its own terminal as an error. It is the
            # ordinary case -- the preset is already there -- so say what it
            # means rather than leaving that line to stand on its own.
            logger.info(
                "Chart preset %s is already registered or not writable; "
                "using %s (%s)",
                name,
                ids[name],
                error,
            )
    return ids


def heatmap(
    spec_id: str,
    title: str,
    values: np.ndarray,
    columns: Sequence[str],
    axis_labels: Sequence[Sequence[str]],
    fields: Mapping[str, str],
):
    """One custom chart: `values` as a table, bound to a registered spec.

    `columns` names the table's columns -- one per axis of `values`, then the
    value itself -- and `fields` maps the specification's placeholders onto
    them.
    """
    import wandb

    table = wandb.Table(columns=list(columns), data=cells(values, *axis_labels))
    return wandb.plot_table(
        vega_spec_name=spec_id,
        data_table=table,
        fields=dict(fields),
        string_fields={"title": title},
    )
