"""Aggregate validation results for an experiment into a paper-style table.

Reads per-run results (written by ``experiments/validate_runs.py``) from local JSON
or the W&B API, keyed by the ``--experiment`` name (= W&B group). For each
domain it aggregates over seeds and reports:

  P     mean precision over seeds
  R     mean recall over seeds
  N_S   #seeds that are sound     (precision == 1)
  N_C   #seeds that are complete  (recall == 1)
  N_SC  #seeds sound *and* complete
  n     #seeds aggregated

Per seed, the seed's validation problems are reduced to a single precision /
recall by micro-averaging the per-action tp/fp/fn in ``action_stats`` (so a seed
is sound iff every problem has zero false positives); when tp/fp/fn are absent
(e.g. W&B summaries) it falls back to the mean of per-problem precision/recall.

With ``--sweep AXIS`` it instead reads the sub-experiments written by
``experiments/launch.py --sweep AXIS`` (``<experiment>-<tag><value>``) and prints one
domain x value matrix per metric; ``--markdown`` writes the same matrices as
markdown tables. ``--reuse-default EXPERIMENT`` mirrors the launcher flag: the
axis-default column is read from that existing experiment.

``--threshold-sweep`` does the same for the *validation* binarization threshold
(sub-experiments ``<experiment>-thr<T>`` written by
``experiments/launch_threshold_validation.py``). Those columns share one set of
trained checkpoints, so they isolate the effect of the threshold alone.

Without a sweep, ``--experiment`` takes any number of unrelated experiments (an
ablation set, say): each gets its own table, and ``--markdown`` writes them into
one document with a cross-experiment summary on top.

Usage:
    python experiments/collect_results.py --experiment paper-main
    python experiments/collect_results.py --experiment paper-main --source wandb
    python experiments/collect_results.py --experiment noise-a2r2 \
        --domains gripper hanoi
    python experiments/collect_results.py --experiment dias-full abl-no-pcgrad \
        --markdown experiments/results/ablations-summary.md
    python experiments/collect_results.py --experiment dias-full --threshold-sweep \
        --markdown experiments/results/threshold-sweep.md
    python experiments/collect_results.py --experiment aux-sweep \
        --sweep aux-loss-weight \
        --markdown experiments/results/aux-sweep.md
    python experiments/collect_results.py --experiment dias --sweep num-slots \
        --reuse-default dias-full --markdown experiments/results/num-slots-sweep.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import launch`
from launch import SWEEP_AXES, sweep_experiment_name  # noqa: E402
from launch_threshold_validation import (  # noqa: E402
    THRESHOLD_TAG,
    THRESHOLD_VALUES,
    threshold_experiment_name,
)
from pipeline import configure_logging  # noqa: E402
from pipeline import (
    LOGS_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    add_logging_arguments,
)

ONE = 1.0 - 1e-9


def _seed_precision_recall(per_problem: dict) -> tuple[float, float] | None:
    """Reduce one seed's per-problem results to a single (precision, recall).

    Micro-averaged: the per-problem tp/fp/fn are summed, so a problem with many
    reachable successors counts for more than a small one. Those counts are
    per-action -- a successor reached under the wrong operator is an error --
    which is why summing `action_stats` gives the same answer and is what
    results written before the totals existed are read through. The mean of
    per-problem precisions is a last resort for older W&B runs whose summaries
    carry nothing else. The counts arrive as floats from JSON.
    """

    def counts(result: dict) -> tuple[int, int, int] | None:
        """The tp/fp/fn of one problem, however that result was written."""
        values = [result.get(k) for k in ("tp", "fp", "fn")]
        # JSON round-trips these as floats; they are counts either way.
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return tuple(int(v) for v in values)
        stats = result.get("action_stats")
        if isinstance(stats, dict):
            return tuple(
                sum(int(a.get(k, 0)) for a in stats.values())
                for k in ("tp", "fp", "fn")
            )
        return None

    tp = fp = fn = 0
    counted = False
    precisions: list[float] = []
    recalls: list[float] = []
    any_ok = False
    for result in per_problem.values():
        if not isinstance(result, dict) or "error" in result:
            continue
        any_ok = True
        pooled = counts(result)
        if pooled is not None:
            counted = True
            tp += pooled[0]
            fp += pooled[1]
            fn += pooled[2]
        if isinstance(result.get("precision"), (int, float)):
            precisions.append(float(result["precision"]))
        if isinstance(result.get("recall"), (int, float)):
            recalls.append(float(result["recall"]))
    if not any_ok:
        return None
    if counted:
        return (
            tp / (tp + fp) if tp + fp else 0.0,
            tp / (tp + fn) if tp + fn else 0.0,
        )
    return (
        sum(precisions) / len(precisions) if precisions else 0.0,
        sum(recalls) / len(recalls) if recalls else 0.0,
    )


def collect_local(
    experiment: str, logs_dir: Path
) -> dict[str, list[tuple[float, float]]]:
    root = logs_dir / "validation" / experiment
    by_domain: dict[str, list[tuple[float, float]]] = {}
    for seed_json in sorted(root.glob("*/seed*.json")):
        payload = json.loads(seed_json.read_text())
        domain = payload.get("domain", seed_json.parent.name)
        pr = _seed_precision_recall(payload.get("per_problem", {}))
        if pr is not None:
            by_domain.setdefault(domain, []).append(pr)
    return by_domain


def collect_wandb(
    experiment: str, project: str, entity: str | None
) -> dict[str, list[tuple[float, float]]]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    by_domain: dict[str, list[tuple[float, float]]] = {}
    key_re = re.compile(
        r"^validation/(?P<stem>.+)/(?P<metric>precision|recall|tp|fp|fn)$"
    )
    for run in api.runs(path, filters={"group": experiment}):
        domain = run.config.get("domain")
        if domain is None:
            continue
        per_problem: dict[str, dict] = {}
        for key, val in run.summary.items():
            m = key_re.match(str(key))
            if m and isinstance(val, (int, float)):
                metric = m["metric"]
                per_problem.setdefault(m["stem"], {})[metric] = (
                    int(val) if metric in ("tp", "fp", "fn") else float(val)
                )
        pr = _seed_precision_recall(per_problem)
        if pr is not None:
            by_domain.setdefault(domain, []).append(pr)
    return by_domain


def _aggregate(prs: list[tuple[float, float]]) -> dict:
    n = len(prs)
    ps = [p for p, _ in prs]
    rs = [r for _, r in prs]
    return {
        "P": sum(ps) / n,
        "R": sum(rs) / n,
        "N_S": sum(p >= ONE for p in ps),
        "N_C": sum(r >= ONE for r in rs),
        "N_SC": sum(p >= ONE and r >= ONE for p, r in prs),
        "n": n,
    }


METRICS = ("P", "R", "N_S", "N_C", "N_SC")
METRIC_LABELS = {
    "P": "mean precision over seeds",
    "R": "mean recall over seeds",
    "N_S": "#seeds sound (precision == 1)",
    "N_C": "#seeds complete (recall == 1)",
    "N_SC": "#seeds sound and complete",
    "n": "#seeds aggregated (< requested means runs are missing)",
}


def _collect(
    experiment: str, args: argparse.Namespace
) -> dict[str, list[tuple[float, float]]]:
    """Per-domain (precision, recall) pairs of an experiment, from the chosen source."""
    if args.source == "local":
        by_domain = collect_local(experiment, args.log_dir)
    else:
        by_domain = collect_wandb(experiment, args.wandb_project, args.wandb_entity)
    if args.domains:
        by_domain = {d: v for d, v in by_domain.items() if d in args.domains}
    return by_domain


def _cell_value(
    metric: str, prs: list[tuple[float, float]] | None, missing: str = "-"
) -> str:
    """One matrix cell: 3-decimal rate, integer count, or `missing` if no runs."""
    if not prs:
        return missing
    value = _aggregate(prs)[metric]
    return f"{value:.3f}" if metric in ("P", "R") else f"{value:d}"


def print_sweep_matrix(
    metric: str,
    values: list[float],
    domains: list[str],
    cells: dict[tuple[float, str], list[tuple[float, float]]],
) -> None:
    """Print one domain x value matrix for a single metric."""

    def cell(prs: list[tuple[float, float]] | None) -> str:
        return f"{_cell_value(metric, prs):>8}"

    header = f"{'domain':<16}" + "".join(f"{f'{v:g}':>8}" for v in values)
    print(f"\n{metric}  ({METRIC_LABELS[metric]})")
    print(header)
    print("-" * len(header))
    for domain in domains:
        row = "".join(cell(cells.get((v, domain))) for v in values)
        print(f"{domain:<16}{row}")
    print("-" * len(header))
    pooled = "".join(
        cell([pr for d in domains for pr in cells.get((v, d), [])]) for v in values
    )
    print(f"{'ALL':<16}{pooled}")


def write_sweep_markdown(
    path: Path,
    axis_name: str,
    param: str,
    values: list[float],
    names: dict[float, str],
    domains: list[str],
    cells: dict[tuple[float, str], list[tuple[float, float]]],
    source: str,
) -> None:
    """Write one markdown table per metric (domains x value, raw values)."""
    cols = [f"{v:g}" for v in values]
    lines = [
        f"# {axis_name} sweep",
        "",
        f"Rows are domains, columns are `{param}` (source: {source}); each cell "
        "aggregates over the seeds of that (domain, value) pair. `ALL` pools "
        "every seed of every domain at that value. Columns come from the "
        "sub-experiments " + ", ".join(f"`{v:g}` → `{names[v]}`" for v in values) + ".",
        "",
    ]
    for metric in (*METRICS, "n"):
        lines += [
            f"## {metric} — {METRIC_LABELS[metric]}",
            "",
            "| domain | " + " | ".join(cols) + " |",
            "|" + "---|" * (len(cols) + 1),
        ]
        for domain in domains:
            row = [_cell_value(metric, cells.get((v, domain))) for v in values]
            lines.append(f"| {domain} | " + " | ".join(row) + " |")
        pooled = [
            _cell_value(metric, [pr for d in domains for pr in cells.get((v, d), [])])
            for v in values
        ]
        lines += ["| **ALL** | " + " | ".join(pooled) + " |", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")


def run_sweep(args: argparse.Namespace, axis_name: str, values: list[float]) -> None:
    """Collect one sub-experiment per value and print/write the domain x value grid."""
    axis = SWEEP_AXES[axis_name]
    if args.reuse_default and axis.default is None:
        raise SystemExit(
            f"--reuse-default: the {axis_name} axis has no reusable default "
            "(the value is tuned per domain)"
        )
    # Same name derivation as the launcher, so the reused default column points
    # at whatever experiment was passed to `launch.py --reuse-default`.
    names = {
        v: sweep_experiment_name(args.experiment, axis, v, args.reuse_default)
        for v in values
    }
    _report_matrix(args, axis_name, axis.param, values, names)


def run_threshold_sweep(args: argparse.Namespace, values: list[float]) -> None:
    """Collect the re-validations written by launch_threshold_validation.py.

    Unlike --sweep this varies a *validation* parameter, so every column reads
    the same trained checkpoints — differences between columns are purely the
    effect of how the learned schema is binarized.
    """
    names = {v: threshold_experiment_name(args.experiment, v) for v in values}
    _report_matrix(args, "threshold", "binarization threshold", values, names)


def _report_matrix(
    args: argparse.Namespace,
    axis_name: str,
    param: str,
    values: list[float],
    names: dict[float, str],
) -> None:
    """Print (and optionally write) the domain x value grid for named experiments."""
    cells: dict[tuple[float, str], list[tuple[float, float]]] = {}
    domains: set[str] = set()
    missing: list[str] = []
    for value in values:
        by_domain = _collect(names[value], args)
        if not by_domain:
            missing.append(names[value])
        for domain, prs in by_domain.items():
            cells[(value, domain)] = prs
            domains.add(domain)

    print(f"\nexperiment: {args.experiment}  ({axis_name} sweep, {param})")
    print(f"values    : {', '.join(f'{v:g} -> {names[v]}' for v in values)}")
    if missing:
        print(f"no results: {', '.join(missing)}")
    if not domains:
        print("(no results found)")
        return

    ordered = sorted(domains)
    for metric in (*METRICS, "n"):
        print_sweep_matrix(metric, values, ordered, cells)
    if args.markdown:
        write_sweep_markdown(
            args.markdown,
            axis_name,
            param,
            values,
            names,
            ordered,
            cells,
            args.source,
        )


def print_table(
    by_domain: dict[str, list[tuple[float, float]]], experiment: str
) -> None:
    header = (
        f"{'domain':<16} {'P':>6} {'R':>6} {'N_S':>5} {'N_C':>5} {'N_SC':>5} {'n':>4}"
    )
    print(f"\nexperiment: {experiment}")
    print(header)
    print("-" * len(header))
    pooled: list[tuple[float, float]] = []
    for domain in sorted(by_domain):
        prs = by_domain[domain]
        pooled += prs
        a = _aggregate(prs)
        print(
            f"{domain:<16} {a['P']:>6.3f} {a['R']:>6.3f} "
            f"{a['N_S']:>5} {a['N_C']:>5} {a['N_SC']:>5} {a['n']:>4}"
        )
    if pooled:
        print("-" * len(header))
        a = _aggregate(pooled)
        print(
            f"{'ALL':<16} {a['P']:>6.3f} {a['R']:>6.3f} "
            f"{a['N_S']:>5} {a['N_C']:>5} {a['N_SC']:>5} {a['n']:>4}"
        )
    else:
        print("(no results found)")


def write_experiments_markdown(
    path: Path,
    experiments: list[str],
    results: dict[str, dict[str, list[tuple[float, float]]]],
    domains: list[str],
    source: str,
) -> None:
    """Write one domain table per experiment, plus a cross-experiment summary.

    Every table spans the union of the domains, so an experiment that is missing
    a domain entirely shows a row of ``-`` instead of silently dropping it.
    """
    columns = (*METRICS, "n")
    # Per domain, the seed count of the most complete experiment in the set:
    # the yardstick for "is this experiment fully collected?".
    expected = sum(
        max(len(results[e].get(d, [])) for e in experiments) for d in domains
    )

    def pooled_of(experiment: str) -> list[tuple[float, float]]:
        return [pr for d in domains for pr in results[experiment].get(d, [])]

    lines = [
        "# Validation results",
        "",
        "Experiments: " + ", ".join(f"`{e}`" for e in experiments) + ".",
        "",
        f"Rows are domains (source: {source}); each cell aggregates over the "
        "seeds of that (experiment, domain) pair. `ALL` pools every seed of "
        "every domain of that experiment, `-` means no results.",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        *(f"| {c} | {METRIC_LABELS[c]} |" for c in columns),
        "",
        "## Summary",
        "",
        f"`n` is compared against {expected} seeds — the most complete "
        "experiment of this set, per domain.",
        "",
        "| experiment | " + " | ".join(columns) + " |",
        "|" + "---|" * (len(columns) + 1),
    ]
    for experiment in experiments:
        pooled = pooled_of(experiment)
        cells = [_cell_value(c, pooled) for c in METRICS]
        lines.append(
            f"| `{experiment}` | "
            + " | ".join(cells)
            + f" | {len(pooled)}/{expected} |"
        )
    lines.append("")

    for experiment in experiments:
        lines += [
            f"## `{experiment}`",
            "",
            "| domain | " + " | ".join(columns) + " |",
            "|" + "---|" * (len(columns) + 1),
        ]
        for domain in domains:
            row = [_cell_value(c, results[experiment].get(domain)) for c in columns]
            lines.append(f"| {domain} | " + " | ".join(row) + " |")
        pooled = pooled_of(experiment)
        lines += [
            "| **ALL** | " + " | ".join(_cell_value(c, pooled) for c in columns) + " |",
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")


def run_experiments(args: argparse.Namespace, experiments: list[str]) -> None:
    """Print one table per experiment and optionally write them as markdown."""
    results = {e: _collect(e, args) for e in experiments}
    for experiment in experiments:
        print_table(results[experiment], experiment)
    if not args.markdown:
        return
    domains = sorted({d for by_domain in results.values() for d in by_domain})
    if not domains:
        raise SystemExit("--markdown: no results found, nothing written")
    write_experiments_markdown(
        args.markdown, experiments, results, domains, args.source
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        required=True,
        nargs="+",
        metavar="NAME",
        help="Experiment name(s) (= W&B group). Several names are allowed "
        "without a sweep: each gets its own table, and --markdown writes them "
        "into one document. A sweep takes exactly one (its base name).",
    )
    p.add_argument("--source", default="local", choices=["local", "wandb"])
    p.add_argument(
        "--domains", nargs="+", default=None, help="Restrict to these domains."
    )
    p.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    p.add_argument("--wandb-project", default=WANDB_PROJECT)
    p.add_argument("--wandb-entity", default=WANDB_ENTITY)
    p.add_argument(
        "--sweep",
        default=None,
        choices=sorted(SWEEP_AXES),
        help="Aggregate the sweep launched under --experiment (sub-experiments "
        "<experiment>-<tag><value>) into a domain x value matrix per metric. "
        "Default grids — "
        + "; ".join(
            f"{name}: {', '.join(f'{v:g}' for v in axis.values)}"
            for name, axis in SWEEP_AXES.items()
        )
        + ".",
    )
    p.add_argument(
        "--sweep-values",
        nargs="+",
        type=float,
        default=None,
        metavar="V",
        help="Values for --sweep instead of that axis' default grid.",
    )
    p.add_argument(
        "--reuse-default",
        default=None,
        metavar="EXPERIMENT",
        help="Read the axis-default column from this existing experiment "
        "(mirrors 'launch.py --reuse-default').",
    )
    p.add_argument(
        "--threshold-sweep",
        nargs="*",
        type=float,
        default=None,
        metavar="T",
        help="Aggregate the re-validations written by "
        "experiments/launch_threshold_validation.py (sub-experiments "
        f"<experiment>-{THRESHOLD_TAG}<T>) into a domain x threshold matrix per "
        "metric. "
        "The bare flag uses the default grid "
        + ", ".join(f"{t:g}" for t in THRESHOLD_VALUES)
        + ".",
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Also write the tables to this markdown file: one table per metric "
        "with --sweep / --threshold-sweep, one table per experiment otherwise.",
    )
    add_logging_arguments(p)
    args = p.parse_args()
    configure_logging(args)

    experiments = list(dict.fromkeys(args.experiment))
    sweeping = args.sweep is not None or args.threshold_sweep is not None
    if len(experiments) > 1 and sweeping:
        raise SystemExit(
            "--sweep / --threshold-sweep read the sub-experiments of a single "
            f"base experiment; got {len(experiments)}: {', '.join(experiments)}"
        )
    # The sweep paths derive sub-experiment names from one base name.
    args.experiment = experiments[0]

    if args.threshold_sweep is not None:
        if args.sweep is not None:
            raise SystemExit(
                "--threshold-sweep varies a validation parameter and cannot be "
                "combined with the training-parameter sweeps"
            )
        if args.reuse_default:
            raise SystemExit(
                "--reuse-default does not apply to --threshold-sweep: the 0.5 "
                "column is re-validated into its own sub-experiment"
            )
        run_threshold_sweep(args, list(args.threshold_sweep) or list(THRESHOLD_VALUES))
        return 0

    axis_name = args.sweep
    values = list(args.sweep_values) if args.sweep_values else None
    if axis_name is not None:
        run_sweep(args, axis_name, values or list(SWEEP_AXES[axis_name].values))
        return 0
    if args.sweep_values:
        raise SystemExit("--sweep-values requires --sweep")
    if args.reuse_default:
        raise SystemExit("--reuse-default requires --sweep")

    run_experiments(args, experiments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
