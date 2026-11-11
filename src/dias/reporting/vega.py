"""Vega-Lite heatmap specifications for the charts a run logs. ``${field:...}`` binds to a table column and ``${string:title}`` to the title."""

from __future__ import annotations

# The edge of a cell, in pixels.
CELL = 18

# The text colour on each colour scheme.
_TEXT_ON = {
    "redyellowblue": {"domain": [0.2, 0.8], "range": ["white", "black", "white"]},
    "viridis": {"domain": [0.55], "range": ["white", "black"]},
}


def _cells(scheme: str, reverse: bool, stacked: bool) -> list[dict]:
    """Coloured cells with their values, in a grid if `stacked`, in a row otherwise."""
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
    """A heatmap, faceted by rows and columns if asked."""
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


SCHEMA = "dias-schema-heatmap"
SELECTION = "dias-selection-heatmap"
SLOT_WEIGHTS = "dias-slot-weights-heatmap"

SPECS = {
    SCHEMA: _heatmap("redyellowblue", reverse=True, rows=True, columns=True),
    SLOT_WEIGHTS: _heatmap("viridis", rows=True, stacked=False),
    SELECTION: _heatmap("viridis", rows=True, columns=True),
}
