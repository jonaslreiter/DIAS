"""Vega-Lite specifications for the charts a run logs.

W&B has no built-in heatmap preset, and a custom one is stored per entity and
referenced by id. `dias.reporting.charts` registers these in the entity that
owns the run, so a clone of this repository draws its own charts instead of
depending on a specification saved in somebody else's account.

`${field:...}` placeholders are bound to table columns at log time and
`${string:title}` to the panel's title. Axes are deliberately left untitled:
Vega-Lite then labels them with the bound column names, which say what they
are without a second set of strings to keep in step.
"""

from __future__ import annotations

# Where the ink has to switch from light to dark to stay readable on the cell
# beneath it. A threshold scale takes one boundary fewer than it has colours.
_TEXT_ON = {
    # Dark at both ends of the scale, pale through the middle.
    "redyellowblue": {"domain": [0.2, 0.8], "range": ["white", "black", "white"]},
    # Dark at zero, bright at one.
    "viridis": {"domain": [0.55], "range": ["white", "black"]},
}


def _cells(scheme: str, reverse: bool) -> list[dict]:
    """A grid of coloured cells with the value written into each one."""
    position = {
        "x": {"field": "${field:x}", "type": "nominal", "axis": {"labelAngle": 0}},
        "y": {"field": "${field:y}", "type": "nominal"},
    }
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
                    # Every cell carries its own number; a colour bar beside
                    # each panel would say the same thing again.
                    "legend": None,
                },
                "tooltip": [
                    {"field": "${field:x}", "type": "nominal"},
                    {"field": "${field:y}", "type": "nominal"},
                    {
                        "field": "${field:value}",
                        "type": "quantitative",
                        "format": ".4f",
                    },
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
                    "legend": None,
                },
            },
        },
    ]


def _heatmap(scheme: str, reverse: bool = False, *, rows=False, columns=False) -> dict:
    """One heatmap, optionally repeated over a row and/or a column field."""
    panel: dict = {"layer": _cells(scheme, reverse)}
    facet = {}
    if rows:
        facet["row"] = {"field": "${field:row_facet}", "type": "nominal"}
    if columns:
        facet["column"] = {"field": "${field:col_facet}", "type": "nominal"}

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "${string:title}",
        "data": {"name": "wandb"},
        "resolve": {"scale": {"color": "shared"}},
        **({"facet": facet, "spec": panel} if facet else panel),
    }


# The name each specification is registered under, within the run's entity.
SPECS = {
    "dias-heatmap": _heatmap("viridis"),
    "dias-heatmap-rows": _heatmap("viridis", rows=True),
    "dias-heatmap-grid": _heatmap(
        "redyellowblue", reverse=True, rows=True, columns=True
    ),
}
