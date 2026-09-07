"""Vega-Lite specifications for the charts a run logs.

W&B has no built-in heatmap preset, so `dias.reporting.charts` registers these per entity, in whichever one owns the run. `${field:...}` placeholders bind to table columns at log time and `${string:title}` to the panel title; no axis is titled.

Selection and slot weights are separate charts because a `facet` takes only a unit or layer spec, never a concat.
"""

from __future__ import annotations

# The edge of one value cell, in pixels. Every chart states it, so a cell is square and the same size in all of them.
CELL = 18

# Where the ink switches to stay readable on the cell beneath. A threshold scale takes one boundary fewer than it has colours.
_TEXT_ON = {
    # Dark at both ends of the scale, pale through the middle.
    "redyellowblue": {"domain": [0.2, 0.8], "range": ["white", "black", "white"]},
    # Dark at zero, bright at one.
    "viridis": {"domain": [0.55], "range": ["white", "black"]},
}


def _cells(scheme: str, reverse: bool, stacked: bool) -> list[dict]:
    """A row of coloured cells with the value written into each one. `stacked` gives them a second axis; without it the panel is the single row a slot-weight strip needs."""
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
                    # Every cell carries its number, so a colour bar would repeat it. `false` rather than the schema's `null`, which W&B ignores.
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


# The name each specification is registered under, within the run's entity.
SCHEMA = "dias-schema-heatmap"
SELECTION = "dias-selection-heatmap"
SLOT_WEIGHTS = "dias-slot-weights-heatmap"

SPECS = {
    SCHEMA: _heatmap("redyellowblue", reverse=True, rows=True, columns=True),
    # One strip per action, rather than one matrix with the actions down its side.
    SLOT_WEIGHTS: _heatmap("viridis", rows=True, stacked=False),
    SELECTION: _heatmap("viridis", rows=True, columns=True),
}
