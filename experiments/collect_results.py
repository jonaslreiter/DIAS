"""Collect validation results into tables of P, R, N_S, N_C, N_SC and n per domain.

A seed's precision and recall are micro-averaged over its test problems. P and R are means over seeds, shown to two decimals and never rounded up to 1.00; N_S, N_C and N_SC count the sound, complete, and sound and complete seeds, and n the seeds. A seed that did not score every problem is counted over the problems it scored, or left out with ``--exclude-partial``, and listed after the tables either way.

``--experiment`` takes several experiments, one table each. ``--sweep AXIS`` instead collects the experiments ``launch.py --sweep`` wrote into one table with a row per domain and value; a value without an experiment of its own is read from the runs of ``--experiment`` that trained at it.

Examples:
    python experiments/collect_results.py --experiment dias-strips dias-strips-plus dias-names l1 --markdown experiments/results/main.md
    python experiments/collect_results.py --experiment alpha --sweep aux-loss-alpha --markdown experiments/results/alpha-sweep.md
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import SWEEP_AXES, sweep_experiment_name  # noqa: E402
from pipeline import configure_logging  # noqa: E402
from pipeline import (
    LOGS_DIR,
    add_logging_arguments,
    available_domains,
    domain_config,
    format_array,
    load_yaml,
    parse_seeds,
)

logger = logging.getLogger("dias.collect")

COLUMNS = ("P", "R", "N_S", "N_C", "N_SC", "n")


def seed_precision_recall(per_problem: dict) -> tuple[float, float] | None:
    """A seed's precision and recall over the problems it scored, or None if none."""
    scored = [
        r for r in per_problem.values() if isinstance(r, dict) and "error" not in r
    ]
    if not scored:
        return None
    tp, fp, fn = (sum(int(r[key]) for r in scored) for key in ("tp", "fp", "fn"))
    return (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0)


def _results(experiment: str, logs_dir: Path, seeds: set[int] | None):
    """(domain, payload) of every result of `experiment` among `seeds`."""
    root = logs_dir / "validation" / experiment
    for seed_json in sorted(root.glob("*/seed*.json")):
        payload = json.loads(seed_json.read_text())
        if seeds is None or payload["seed"] in seeds:
            yield payload["domain"], payload


def _scored_problems(payload: dict) -> tuple[str, ...]:
    return tuple(
        sorted(
            stem
            for stem, result in payload.get("per_problem", {}).items()
            if "error" not in result
        )
    )


def collect(
    experiment: str,
    logs_dir: Path,
    seeds: set[int] | None = None,
    exclude_partial: bool = False,
) -> dict[str, list[tuple[float, float]]]:
    """Per domain, a (precision, recall) per seed."""
    by_domain: dict[str, list[tuple[float, float]]] = {}
    for domain, payload in _results(experiment, logs_dir, seeds):
        if exclude_partial and payload["status"] != "ok":
            continue
        pr = seed_precision_recall(payload.get("per_problem", {}))
        if pr is not None:
            by_domain.setdefault(domain, []).append(pr)
    return by_domain


def coverage(
    experiment: str, logs_dir: Path, seeds: set[int] | None = None
) -> dict[str, collections.Counter]:
    """Per domain, the seeds that did not score every problem, counted by the problems they scored."""
    counts: dict[str, collections.Counter] = {}
    for domain, payload in _results(experiment, logs_dir, seeds):
        if payload["status"] != "ok":
            counts.setdefault(domain, collections.Counter())[
                _scored_problems(payload)
            ] += 1
    return counts


def aggregate(prs: list[tuple[float, float]]) -> dict:
    ps = [p for p, _ in prs]
    rs = [r for _, r in prs]
    return {
        "P": sum(ps) / len(prs),
        "R": sum(rs) / len(prs),
        "N_S": sum(p == 1.0 for p in ps),
        "N_C": sum(r == 1.0 for r in rs),
        "N_SC": sum(p == 1.0 and r == 1.0 for p, r in prs),
        "n": len(prs),
    }


def rate(value: float) -> str:
    """Two decimals, never rounded up to 1.00."""
    shown = f"{value:.2f}"
    return "0.99" if shown == "1.00" and value < 1.0 else shown


def cells(prs: list[tuple[float, float]] | None) -> list[str]:
    if not prs:
        return ["-"] * len(COLUMNS)
    values = aggregate(prs)
    return [rate(values[c]) if c in ("P", "R") else str(values[c]) for c in COLUMNS]


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def plain_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(x)) for x in column) for column in zip(header, *rows)]
    return "\n".join(
        "  ".join(
            str(x).rjust(w) if i else str(x).ljust(w)
            for i, (x, w) in enumerate(zip(line, widths))
        )
        for line in [header, *rows]
    )


def coverage_rows(
    cells_: list[tuple[str, str]], args
) -> tuple[list[str], list[list[str]]]:
    """A table of the seeds that did not score every problem, per (experiment, domain) in `cells_`."""
    found = {}
    for experiment, domain in cells_:
        counts = coverage(experiment, args.log_dir, args.seeds).get(domain)
        if counts and (not args.domains or domain in args.domains):
            found[(experiment, domain)] = counts
    problems = sorted(
        {key for counts in found.values() for key in counts},
        key=lambda key: (-len(key), key),
    )
    header = [
        "experiment",
        "domain",
        *(" and ".join(p) + " only" if p else "none" for p in problems),
    ]
    rows = [
        [e, d, *(str(counts.get(p, 0)) for p in problems)]
        for (e, d), counts in found.items()
    ]
    return header, rows


def report(
    title: str, tables: list[tuple[str, list[str], list[list[str]]]], args
) -> None:
    """Print `tables`, each (heading, header, rows), and write them to ``--markdown``."""
    for heading, header, rows in tables:
        print(f"\n{heading}\n{plain_table(header, rows)}")
    if not args.markdown:
        return
    lines = [f"# {title}", ""]
    for heading, header, rows in tables:
        lines += [f"## {heading}", "", *markdown_table(header, rows), ""]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines))
    print(f"\nwrote {args.markdown}")


def configured_values(param: str, domains: list[str]) -> dict[str, float]:
    """Per domain, the value its config.yaml gives the dotted `param`, if any."""
    configured = {}
    for domain in domains:
        node = load_yaml(domain_config(domain))
        for key in param.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            configured[domain] = float(node)
    return configured


def trained_values(experiment: str, param: str, logs_dir: Path) -> dict[str, float]:
    """Per domain, the value of the dotted `param` the runs of `experiment` trained with, from their saved config.yaml."""
    values: dict[str, float] = {}
    for domain in available_domains():
        pattern = re.compile(rf"{re.escape(experiment)}-{re.escape(domain)}-s\d+")
        for config in sorted(
            (logs_dir / "dias").glob(f"{experiment}-{domain}-s*/config.yaml")
        ):
            if not pattern.fullmatch(config.parent.name):
                continue
            node = load_yaml(config)
            for key in param.split("."):
                if isinstance(node, dict) and key in node:
                    node = node[key]
                elif key != "init_args":
                    # A dataclass-typed argument is saved without init_args.
                    node = None
                    break
            if isinstance(node, (int, float)) and not isinstance(node, bool):
                values[domain] = float(node)
            break
    return values


def run_sweep(args, axis_name: str, values: list[float]) -> None:
    axis = SWEEP_AXES[axis_name]
    names = {v: sweep_experiment_name(args.experiment, axis, v) for v in values}
    results = {v: collect_for(names[v], args) for v in values}
    if axis.linked is None:
        trained = {
            domain: value / axis.scale
            for domain, value in trained_values(
                args.experiment, axis.param, args.log_dir
            ).items()
        }
        base = None
        for value in values:
            at_value = {d for d, v in trained.items() if math.isclose(v, value)}
            if not results[value] and at_value:
                base = base or collect_for(args.experiment, args)
                results[value] = {d: prs for d, prs in base.items() if d in at_value}
                names[value] = args.experiment
    domains = sorted(
        {d for by_domain in results.values() for d in by_domain}
        | {
            d
            for v in values
            for d in coverage(names[v], args.log_dir, args.seeds)
            if not args.domains or d in args.domains
        }
    )
    configured = configured_values(axis.param, domains) if axis.linked is None else {}
    rows = []
    for domain in domains:
        for i, value in enumerate(values):
            shown = f"{value:g}"
            if domain in configured and math.isclose(configured[domain], value):
                shown = f"**{shown}**"
            rows.append(
                [domain if i == 0 else "", shown, *cells(results[value].get(domain))]
            )
    pooled = [
        ["ALL", f"{v:g}", *cells([pr for d in domains for pr in results[v].get(d, [])])]
        for v in values
    ]
    tables = [
        (
            f"{args.experiment}, {axis_name}",
            ["domain", axis_name, *COLUMNS],
            rows + pooled,
        )
    ]
    header, missing = coverage_rows(
        [(names[v], d) for v in values for d in domains], args
    )
    if missing:
        tables.append(("Seeds not scored on every problem", header, missing))
    report(f"{axis_name} sweep{seeds_note(args.seeds)}", tables, args)


def collect_for(experiment: str, args) -> dict[str, list[tuple[float, float]]]:
    by_domain = collect(experiment, args.log_dir, args.seeds, args.exclude_partial)
    if args.domains:
        by_domain = {d: v for d, v in by_domain.items() if d in args.domains}
    return by_domain


def seeds_note(seeds: set[int] | None) -> str:
    return f", seeds {format_array(sorted(seeds))}" if seeds else ""


def run_experiments(args, experiments: list[str]) -> None:
    results = {e: collect_for(e, args) for e in experiments}
    header, missing = coverage_rows(
        [(e, d) for e in experiments for d in available_domains()], args
    )
    domains = sorted(
        {d for by_domain in results.values() for d in by_domain}
        | {row[1] for row in missing}
    )
    tables = [
        (
            "Summary",
            ["experiment", *COLUMNS],
            [
                [e, *cells([pr for d in domains for pr in results[e].get(d, [])])]
                for e in experiments
            ],
        )
    ]
    for experiment in experiments:
        rows = [[d, *cells(results[experiment].get(d))] for d in domains]
        rows.append(
            [
                "ALL",
                *cells([pr for d in domains for pr in results[experiment].get(d, [])]),
            ]
        )
        tables.append((experiment, ["domain", *COLUMNS], rows))
    if missing:
        tables.append(("Seeds not scored on every problem", header, missing))
    report(f"Validation results{seeds_note(args.seeds)}", tables, args)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        required=True,
        nargs="+",
        metavar="NAME",
        help="Experiments; one with --sweep.",
    )
    p.add_argument("--domains", nargs="+", default=None, help="Only these domains.")
    p.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        metavar="SEED",
        help="Only these seeds, e.g. '--seeds 0-4'.",
    )
    p.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    p.add_argument(
        "--sweep",
        default=None,
        choices=sorted(SWEEP_AXES),
        help="Collect the experiments launch.py --sweep wrote for --experiment.",
    )
    p.add_argument(
        "--sweep-values",
        nargs="+",
        type=float,
        default=None,
        metavar="V",
        help="Values of --sweep instead of its defaults.",
    )
    p.add_argument(
        "--exclude-partial",
        action="store_true",
        help="Leave out seeds that did not score every problem.",
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Also write the tables to this file.",
    )
    add_logging_arguments(p)
    args = p.parse_args()
    configure_logging(args)
    args.seeds = set(parse_seeds(args.seeds)) if args.seeds is not None else None

    experiments = list(dict.fromkeys(args.experiment))
    if args.sweep is None:
        if args.sweep_values:
            raise SystemExit("--sweep-values requires --sweep")
        run_experiments(args, experiments)
        return 0
    if len(experiments) > 1:
        raise SystemExit("--sweep takes one --experiment")
    args.experiment = experiments[0]
    run_sweep(
        args, args.sweep, list(args.sweep_values or SWEEP_AXES[args.sweep].values)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
