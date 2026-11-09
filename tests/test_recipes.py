"""Golden of every training recipe in experiments/README.md: the model and data arguments each (recipe, domain) run resolves to.

Regenerate after an intended change with ``DIAS_REGENERATE_GOLDEN=1 pytest tests/test_recipes.py`` and review the fixture diff.
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import launch
import pipeline
from test_configs import MAIN, ParseOnlyCLI

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "recipes.json"
REGENERATE = os.environ.get("DIAS_REGENERATE_GOLDEN") == "1"

CURVE_DOMAINS = ["blocks-3", "delivery", "gripper", "hanoi", "logistics"]
ABLATIONS = sorted(launch.ABLATION_OVERLAYS)

BASE = "dias-names"
ABSENT = "<absent>"
RECIPES = [
    ["--experiment", BASE],
    ["--experiment", "dias-strips", "--labels", "strips"],
    ["--experiment", "dias-strips-plus", "--labels", "strips-plus"],
    ["--experiment", "mlp-names", "--experiment-type", "mlp"],
    ["--experiment", "noise", "--experiment-type", "noise", "--sweep", "noise"],
    *(["--experiment", f"ablation-{a}", "--ablation", a] for a in ABLATIONS),
    ["--experiment", "alpha", "--sweep", "aux-loss-alpha"],
    ["--experiment", "dias-names", "--sweep", "num-slots", "--sweep-values", "7", "9"],
    [
        "--experiment",
        "dias-names",
        "--sweep",
        "tau",
        "--sweep-values",
        "0",
        "100",
        "1000",
        "5000",
    ],
    ["--experiment", "curve", "--sweep", "dataset-size", "--domains", *CURVE_DOMAINS],
    [
        "--experiment",
        "curve-f2",
        "--experiment-type",
        "noise",
        "--sweep",
        "dataset-size",
        "--set",
        "data.init_args.noise.added_atoms=0.5",
        "--set",
        "data.init_args.noise.removed_atoms=0.5",
        "--domains",
        *CURVE_DOMAINS,
    ],
]


def flatten(node, prefix=""):
    if isinstance(node, dict):
        return {
            key: value
            for name, child in node.items()
            for key, value in flatten(child, f"{prefix}{name}.").items()
        }
    return {prefix[:-1]: node}


def resolve_all() -> dict:
    """``{"base": {domain: args}, "recipes": {experiment: {"validator": ..., "domains": {domain: args differing from base}}}}``."""
    parser = ParseOnlyCLI(
        args=[
            "--config",
            str(MAIN),
            "--config",
            str(ROOT / "pddl/gripper/config.yaml"),
        ],
        run=False,
    ).parser

    def resolve(args):
        config = parser.parse_args(args).as_dict()
        resolved = {}
        for section in ("model", "data"):
            resolved.update(flatten(config[section]["init_args"], f"{section}."))
        for key in ("data.domain_path", "data.problem_path"):
            resolved[key] = str(Path(resolved[key]).relative_to(ROOT))
        return json.loads(json.dumps(resolved, default=str))

    base = {}
    recipes = {}
    for recipe in RECIPES:
        plan = launch.build_plan(
            launch.parse_args([*recipe, "--backend", "local", "--no-wandb"])
        )
        for point in plan.points:
            for domain in plan.domains:
                args = resolve(plan.train_args(point, domain))
                if point.experiment == BASE:
                    base[domain] = args
                    continue
                entry = recipes.setdefault(
                    point.experiment, {"validator": plan.validator, "domains": {}}
                )
                entry["domains"][domain] = {
                    key: args.get(key, ABSENT)
                    for key in {*args, *base[domain]}
                    if base[domain].get(key, ABSENT) != args.get(key, ABSENT)
                }
    return {"base": base, "recipes": recipes}


class RecipeTest(unittest.TestCase):
    def test_every_recipe_resolves_as_recorded(self):
        resolved = resolve_all()
        if REGENERATE:
            FIXTURE.write_text(json.dumps(resolved, indent=1, sort_keys=True) + "\n")
        expected = json.loads(FIXTURE.read_text())
        self.assertEqual(sorted(resolved["base"]), pipeline.available_domains())
        for domain, args in expected["base"].items():
            with self.subTest(base=domain):
                self.assertEqual(resolved["base"][domain], args)
        self.assertEqual(sorted(resolved["recipes"]), sorted(expected["recipes"]))
        for name, entry in expected["recipes"].items():
            with self.subTest(recipe=name):
                self.assertEqual(resolved["recipes"][name], entry)


if __name__ == "__main__":
    unittest.main()
