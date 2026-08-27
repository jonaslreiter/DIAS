"""Vega-Lite specifications for the charts a run logs.

W&B has no built-in heatmap preset, and a custom one is stored per entity and referenced by id. `dias.reporting.charts` registers these in the entity that owns the run, so a clone of this repository draws its own charts instead of depending on a specification saved in somebody else's account.

`${field:...}` placeholders are bound to table columns at log time and `${string:title}` to the panel's title. Nothing is titled: what an axis holds is clear from the values along it, and every string here would be repeated on every panel of every chart.

Selection and slot weights are separate charts, where the figures these replaced drew them side by side per action. A `facet` takes only a unit or layer spec, never a concat, so the two cannot share one panel without smuggling the weights in as an extra object column: a different quantity on an axis that says it holds objects.
"""

from __future__ import annotations

# The edge of one value cell, in pixels. Every chart states it, so a cell is square and the same size whichever chart it is in.
CELL = 18

# Where the ink has to switch from light to dark to stay readable on the cell beneath it. A threshold scale takes one boundary fewer than it has colours.
_TEXT_ON = {
    # Dark at both ends of the scale, pale through the middle.
    "redyellowblue": {"domain": [0.2, 0.8], "range": ["white", "black", "white"]},
    # Dark at zero, bright at one.
    "viridis": {"domain": [0.55], "range": ["white", "black"]},
}


def _cells(scheme: str, reverse: bool, stacked: bool) -> list[dict]:
    """A row of coloured cells with the value written into each one.

    `stacked` gives them a second axis to spread over; without it the panel is the single row a slot-weight strip needs.
    """
    position = {
        "x": {
            "field": "${field:x}",
            "type": "nominal",
            "title": None,
            "axis": {"labelAngle": 0},
        }
    }
    if stacked:
        position["y"] = {"field": "${field:y}", "type": "nominal", "title": None}

    tooltip = [{"field": f"${{field:{axis}}}", "type": "nominal"} for axis in position]
    return [
        {
            "mark": {"type": "rect"},
            "encoding": position
            | {
                "color": {
                    "field": "${field:value}",
                    "type": "quantitative",
                    "scale": {
                        "scheme": scheme,
                        "reverse": reverse,
                        "domain": [0.0, 1.0],
                    },
                    # Every cell carries its own number, so a colour bar beside each panel would only say it again. `false` rather than the schema's `null`: W&B draws the bar anyway for `null`, and for a spec-wide `config.legend.disable`.
                    "legend": False,
                },
                "tooltip": tooltip
                + [
                    {
                        "field": "${field:value}",
                        "type": "quantitative",
                        "format": ".4f",
                    }
                ],
            },
        },
        {
            "mark": {"type": "text", "fontSize": 8},
            "encoding": position
            | {
                "text": {
                    "field": "${field:value}",
                    "type": "quantitative",
                    "format": ".2f",
                },
                "color": {
                    "field": "${field:value}",
                    "type": "quantitative",
                    "scale": {"type": "threshold", **_TEXT_ON[scheme]},
                    "legend": False,
                },
            },
        },
    ]


def _heatmap(
    scheme: str,
    reverse: bool = False,
    *,
    rows: bool = False,
    columns: bool = False,
    stacked: bool = True,
) -> dict:
    """One heatmap, optionally repeated over a row and/or a column field."""
    panel: dict = {
        "layer": _cells(scheme, reverse, stacked),
        "width": {"step": CELL},
        "height": {"step": CELL} if stacked else CELL,
    }
    facet = {
        side: {
            "field": f"${{field:{side[:3]}_facet}}",
            "type": "nominal",
            "title": None,
        }
        for side, wanted in (("row", rows), ("column", columns))
        if wanted
    }

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "${string:title}",
        "data": {"name": "wandb"},
        "resolve": {"scale": {"color": "shared"}},
        **({"facet": facet, "spec": panel} if facet else panel),
    }


# The name each specification is registered under, within the run's entity, and which chart it draws.
SCHEMA = "dias-schema-heatmap"
SELECTION = "dias-selection-heatmap"
SLOT_WEIGHTS = "dias-slot-weights-heatmap"

SPECS = {
    SCHEMA: _heatmap("redyellowblue", reverse=True, rows=True, columns=True),
    # One strip per action, rather than one matrix with the actions down its side.
    SLOT_WEIGHTS: _heatmap("viridis", rows=True, stacked=False),
    SELECTION: _heatmap("viridis", rows=True, columns=True),
}
