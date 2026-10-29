"""Aggregate validation results for an experiment into a paper-style table.

Reads per-run results (written by ``experiments/validate_runs.py``) from local JSON or the W&B API, keyed by the ``--experiment`` name (= W&B group). For each domain it aggregates over seeds and reports:

  P     mean precision over seeds
  R     mean recall over seeds
  N_S   #seeds that are sound     (precision == 1)
  N_C   #seeds that are complete  (recall == 1)
  N_SC  #seeds sound *and* complete
  n     #seeds aggregated

A seed's problems are reduced to one precision / recall by micro-averaging their tp/fp/fn, falling back to the mean of per-problem rates when the counts are absent.

P and R are shown to two decimals, a rate below 1 as at most 0.99.

``--sweep AXIS`` instead reads the sub-experiments ``launch.py --sweep`` wrote and prints one domain x value matrix per metric (``--markdown``: one table, a row per domain and value); a value with no sub-experiment is read from the base experiment's own runs where they trained at it, so a sweep can leave out the value its base already ran. ``--threshold-sweep`` does the same over the thresholds of ``launch_threshold_validation.py``, whose columns share one set of checkpoints.

A seed whose validation did not score every held-out problem is left out, or with ``--include-partial`` counted over the problems it scored completely; either way, the seeds concerned are listed per cell after the tables. Without a sweep, ``--experiment`` takes any number of unrelated experiments, each with its own table. ``--seeds`` restricts every experiment to those seeds, so experiments run with different seed counts are compared on the same ones.

Usage:
    python experiments/collect_results.py --experiment dias-names [--source wandb] [--domains gripper hanoi]
    python experiments/collect_results.py --experiment dias-names ablation-no-pcgrad --seeds 0-4 --markdown experiments/results/ablations.md
    python experiments/collect_results.py --experiment alpha --sweep aux-loss-alpha --markdown experiments/results/alpha-sweep.md
    python experiments/collect_results.py --experiment dias-names --threshold-sweep --markdown experiments/results/threshold.md
    python experiments/collect_results.py --experiment dias-names --sweep num-slots --seeds 0-4 --markdown experiments/results/slots.md
    python experiments/collect_results.py --experiment noise --sweep noise --include-partial --markdown experiments/results/noise.md
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
    available_domains,
    domain_config,
    format_array,
    load_yaml,
    parse_seeds,
)

logger = logging.getLogger("dias.collect")


def _seed_precision_recall(per_problem: dict) -> tuple[float, float] | None:
    """Reduce one seed's per-problem results to a single (precision, recall).

    Micro-averaged, so a problem with many reachable successors counts for more. Summing `action_stats` is the same answer, for a result written before the totals existed; the mean of per-problem rates is a last resort for W&B summaries carrying nothing else.
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


def _local_results(experiment: str, logs_dir: Path, seeds: set[int] | None):
    """(domain, seed, payload) of every result JSON of `experiment`, restricted to `seeds`."""
    root = logs_dir / "validation" / experiment
    for seed_json in sorted(root.glob("*/seed*.json")):
        payload = json.loads(seed_json.read_text())
        seed = payload.get("seed", int(seed_json.stem.removeprefix("seed")))
        if seeds is not None and seed not in seeds:
            continue
        yield payload.get("domain", seed_json.parent.name), seed, payload


def _scored_problems(payload: dict) -> list[str]:
    """The problems a result scored completely, in order; a problem cut short leaves only an error behind."""
    return sorted(
        stem
        for stem, result in payload.get("per_problem", {}).items()
        if isinstance(result, dict) and "error" not in result
    )


def collect_local(
    experiment: str,
    logs_dir: Path,
    seeds: set[int] | None = None,
    include_partial: bool = False,
) -> dict[str, list[tuple[float, float]]]:
    """Per domain, one (precision, recall) per seed. A seed that did not score every problem is left out, or with `include_partial` counted over the problems it scored completely."""
    by_domain: dict[str, list[tuple[float, float]]] = {}
    for domain, _, payload in _local_results(experiment, logs_dir, seeds):
        status = payload.get("status", "ok")
        if status != "ok" and not include_partial:
            logger.warning(
                "Skipping %s seed %s: %s.", domain, payload.get("seed"), status
            )
            continue
        pr = _seed_precision_recall(payload.get("per_problem", {}))
        if pr is not None:
            by_domain.setdefault(domain, []).append(pr)
    return by_domain


def coverage_local(
    experiment: str, logs_dir: Path, seeds: set[int] | None = None
) -> dict[str, collections.Counter]:
    """Per domain, how many seeds that did not score every problem scored which problems completely (an empty tuple: none)."""
    coverage: dict[str, collections.Counter] = {}
    for domain, _, payload in _local_results(experiment, logs_dir, seeds):
        if payload.get("status", "ok") != "ok":
            coverage.setdefault(domain, collections.Counter())[
                tuple(_scored_problems(payload))
            ] += 1
    return coverage


def collect_wandb(
    experiment: str, project: str, entity: str | None, seeds: set[int] | None = None
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
        if seeds is not None and run.config.get("seed") not in seeds:
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
        "N_S": sum(p == 1.0 for p in ps),
        "N_C": sum(r == 1.0 for r in rs),
        "N_SC": sum(p == 1.0 and r == 1.0 for p, r in prs),
        "n": n,
    }


METRICS = ("P", "R", "N_S", "N_C", "N_SC")
METRIC_LABELS = {
    "P": "mean precision over seeds (below 1 shown as at most 0.99)",
    "R": "mean recall over seeds (below 1 shown as at most 0.99)",
    "N_S": "#seeds sound (precision == 1)",
    "N_C": "#seeds complete (recall == 1)",
    "N_SC": "#seeds sound and complete",
    "n": "#seeds aggregated (< requested means runs are missing)",
}


def _collect(
    experiment: str, args: argparse.Namespace
) -> dict[str, list[tuple[float, float]]]:
    """Per-domain (precision, recall) pairs, from the chosen source."""
    if args.source == "local":
        by_domain = collect_local(
            experiment, args.log_dir, args.seeds, args.include_partial
        )
    else:
        by_domain = collect_wandb(
            experiment, args.wandb_project, args.wandb_entity, args.seeds
        )
    if args.domains:
        by_domain = {d: v for d, v in by_domain.items() if d in args.domains}
    return by_domain


def _coverage(
    cells: list[tuple[str, str]], args: argparse.Namespace
) -> list[tuple[str, str, collections.Counter]]:
    """For each (experiment, domain) in `cells`, the seeds that did not score every problem, by which problems they did score; local results only, since W&B keeps no status."""
    if args.source != "local":
        return []
    rows = []
    by_experiment: dict[str, dict[str, collections.Counter]] = {}
    for experiment, domain in cells:
        if experiment not in by_experiment:
            by_experiment[experiment] = coverage_local(
                experiment, args.log_dir, args.seeds
            )
        if args.domains and domain not in args.domains:
            continue
        counts = by_experiment[experiment].get(domain)
        if counts:
            rows.append((experiment, domain, counts))
    return rows


def _scored_label(problems: tuple[str, ...]) -> str:
    """'test-1 and test-2 only', 'test-1 only' or 'none' for the problems a seed scored."""
    return " and ".join(problems) + " only" if problems else "none"


def _coverage_columns(rows) -> list[tuple[str, ...]]:
    """The scored-problem sets that occur, most problems first."""
    present = {key for _, _, counts in rows for key in counts}
    return sorted(present, key=lambda key: (-len(key), key))


def coverage_markdown(rows, include_partial: bool) -> list[str]:
    """A section listing the cells with seeds that did not score every problem, or nothing if there are none."""
    if not rows:
        return []
    columns = _coverage_columns(rows)
    use = (
        "are counted over the problems they scored completely; a problem scored only partly is left out"
        if include_partial
        else "are left out of the tables above"
    )
    lines = [
        "## Seeds not scored on every problem",
        "",
        f"Seeds that did not score every held-out problem (out of memory, stalled or stopped) {use}. Per cell, how many such seeds scored which problems completely:",
        "",
        "| experiment | domain | "
        + " | ".join(_scored_label(c) for c in columns)
        + " |",
        "|---|---|" + "--:|" * len(columns),
    ]
    for experiment, domain, counts in rows:
        lines.append(
            f"| `{experiment}` | {domain} | "
            + " | ".join(str(counts.get(c, 0)) for c in columns)
            + " |"
        )
    return lines + [""]


def print_coverage(rows, include_partial: bool) -> None:
    """The coverage section, as plain text."""
    if not rows:
        return
    columns = _coverage_columns(rows)
    use = "counted over their scored problems" if include_partial else "left out"
    print(f"\nseeds not scored on every problem ({use}):")
    print(
        f"{'experiment':<24} {'domain':<16} "
        + "  ".join(_scored_label(c) for c in columns)
    )
    for experiment, domain, counts in rows:
        print(
            f"{experiment:<24} {domain:<16} "
            + "  ".join(f"{counts.get(c, 0):>{len(_scored_label(c))}}" for c in columns)
        )


def _seeds_note(seeds: set[int] | None) -> str:
    """A sentence naming the seeds a table is restricted to, if it is."""
    return f" Seeds {format_array(sorted(seeds))} only." if seeds else ""


def _rate(value: float) -> str:
    """Two decimals, never rounded up to 1.00."""
    shown = f"{value:.2f}"
    return "0.99" if shown == "1.00" and value < 1.0 else shown


def _cell_value(
    metric: str, prs: list[tuple[float, float]] | None, missing: str = "-"
) -> str:
    """One cell: a rate, an integer count, or `missing` if no runs."""
    if not prs:
        return missing
    value = _aggregate(prs)[metric]
    return _rate(value) if metric in ("P", "R") else f"{value:d}"


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
    flag: str,
    values: list[float],
    names: dict[float, str],
    domains: list[str],
    cells: dict[tuple[float, str], list[tuple[float, float]]],
    configured: dict[str, float],
    seeds: set[int] | None = None,
    appendix: list[str] | None = None,
) -> None:
    """Write one markdown table, a row per (domain, value); the value a domain's config.yaml sets is in bold. `appendix` lines follow the table."""
    columns = (*METRICS, "n")
    lines = [
        f"# {axis_name} sweep",
        "",
        f"`{flag}`: "
        + ", ".join(f"{v:g} from `{names[v]}`" for v in values)
        + "."
        + _seeds_note(seeds)
        + (" Bold: the value in the domain's `config.yaml`." if configured else ""),
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        *(f"| {c} | {METRIC_LABELS[c]} |" for c in columns),
        "",
        f"| domain | {axis_name} | " + " | ".join(columns) + " |",
        "|---|" + "--:|" * (len(columns) + 1),
    ]
    for domain in domains:
        for i, value in enumerate(values):
            shown = f"{value:g}"
            if domain in configured and math.isclose(configured[domain], value):
                shown = f"**{shown}**"
            row = [_cell_value(c, cells.get((value, domain))) for c in columns]
            lines.append(
                f"| {domain if i == 0 else ''} | {shown} | " + " | ".join(row) + " |"
            )
    if appendix:
        lines += ["", *appendix]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


def configured_values(param: str, domains: list[str]) -> dict[str, float]:
    """Per domain, the value its config.yaml sets for the dotted `param`, where it sets one."""
    configured = {}
    for domain in domains:
        node = (
            load_yaml(domain_config(domain)) if domain_config(domain).is_file() else {}
        )
        for key in param.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            configured[domain] = float(node)
    return configured


def trained_values(experiment: str, param: str, logs_dir: Path) -> dict[str, float]:
    """Per domain, the value of the dotted `param` the runs of `experiment` trained with, read off their saved config.yaml."""
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
                    # A dataclass-typed argument is saved without its class_path / init_args wrapper.
                    node = None
                    break
            if isinstance(node, (int, float)) and not isinstance(node, bool):
                values[domain] = float(node)
            break
    return values


def run_sweep(args: argparse.Namespace, axis_name: str, values: list[float]) -> None:
    """Collect one sub-experiment per value and print/write the domain x value grid. A value with no sub-experiment is read from the base experiment's own runs where those trained at it, so `launch.py --sweep` can leave out the value its base already ran."""
    axis = SWEEP_AXES[axis_name]
    names = {v: sweep_experiment_name(args.experiment, axis, v) for v in values}
    base = None
    # An axis that moves other parameters with it has no single configured value.
    if axis.linked is None and args.source == "local":
        trained = {
            domain: value / axis.scale
            for domain, value in trained_values(
                args.experiment, axis.param, args.log_dir
            ).items()
        }
        base = (args.experiment, trained) if trained else None
    _report_matrix(
        args, axis_name, axis.param, values, names, axis.linked is None, base
    )


def run_threshold_sweep(args: argparse.Namespace, values: list[float]) -> None:
    """Collect the re-validations written by launch_threshold_validation.py, whose columns all read the same checkpoints."""
    names = {v: threshold_experiment_name(args.experiment, v) for v in values}
    _report_matrix(args, "threshold", "binarization threshold", values, names, False)


def _report_matrix(
    args: argparse.Namespace,
    axis_name: str,
    param: str,
    values: list[float],
    names: dict[float, str],
    mark_configured: bool,
    base: tuple[str, dict[str, float]] | None = None,
) -> None:
    """Print (and optionally write) the domain x value grid for named experiments. `base` is the base experiment and, per domain, the value its runs trained at; a value with no sub-experiment of its own is read from there."""
    names = dict(names)
    cells: dict[tuple[float, str], list[tuple[float, float]]] = {}
    domains: set[str] = set()
    missing: list[str] = []
    base_results = None
    for value in values:
        by_domain = _collect(names[value], args)
        if not by_domain and base is not None:
            base_experiment, trained = base
            at_value = {d for d, v in trained.items() if math.isclose(v, value)}
            if at_value:
                if base_results is None:
                    base_results = _collect(base_experiment, args)
                by_domain = {d: prs for d, prs in base_results.items() if d in at_value}
                names[value] = base_experiment
        if not by_domain:
            missing.append(names[value])
        for domain, prs in by_domain.items():
            cells[(value, domain)] = prs
            domains.add(domain)
        if args.source == "local":
            # A domain no seed of which scored a problem still belongs in the table.
            domains.update(
                d
                for d in coverage_local(names[value], args.log_dir, args.seeds)
                if not args.domains or d in args.domains
            )

    print(f"\nexperiment: {args.experiment}  ({axis_name} sweep, {param})")
    print(f"values    : {', '.join(f'{v:g} -> {names[v]}' for v in values)}")
    if args.seeds:
        print(f"seeds     : {format_array(sorted(args.seeds))}")
    if missing:
        print(f"no results: {', '.join(missing)}")
    if not domains:
        print("(no results found)")
        return

    ordered = sorted(domains)
    for metric in (*METRICS, "n"):
        print_sweep_matrix(metric, values, ordered, cells)
    coverage = _coverage([(names[v], d) for v in values for d in ordered], args)
    print_coverage(coverage, args.include_partial)
    if args.markdown:
        write_sweep_markdown(
            args.markdown,
            axis_name,
            "--threshold-sweep" if axis_name == "threshold" else f"--sweep {axis_name}",
            values,
            names,
            ordered,
            cells,
            configured_values(param, ordered) if mark_configured else {},
            args.seeds,
            coverage_markdown(coverage, args.include_partial),
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
            f"{domain:<16} {_rate(a['P']):>6} {_rate(a['R']):>6} {a['N_S']:>5} {a['N_C']:>5} {a['N_SC']:>5} {a['n']:>4}"
        )
    if pooled:
        print("-" * len(header))
        a = _aggregate(pooled)
        print(
            f"{'ALL':<16} {_rate(a['P']):>6} {_rate(a['R']):>6} {a['N_S']:>5} {a['N_C']:>5} {a['N_SC']:>5} {a['n']:>4}"
        )
    else:
        print("(no results found)")


def write_experiments_markdown(
    path: Path,
    experiments: list[str],
    results: dict[str, dict[str, list[tuple[float, float]]]],
    domains: list[str],
    source: str,
    seeds: set[int] | None = None,
    appendix: list[str] | None = None,
) -> None:
    """Write one domain table per experiment, plus a cross-experiment summary. Every table spans the union of the domains, so a missing one shows ``-`` rather than vanishing. `appendix` lines close the document."""
    columns = (*METRICS, "n")
    # Per domain, the seed count of the most complete experiment in the set.
    expected = sum(
        max(len(results[e].get(d, [])) for e in experiments) for d in domains
    )

    def pooled_of(experiment: str) -> list[tuple[float, float]]:
        return [pr for d in domains for pr in results[experiment].get(d, [])]

    lines = [
        "# Validation results",
        "",
        "Experiments: "
        + ", ".join(f"`{e}`" for e in experiments)
        + "."
        + _seeds_note(seeds),
        "",
        f"Rows are domains (source: {source}); each cell aggregates over the seeds of that (experiment, domain) pair. `ALL` pools every seed of every domain of that experiment, `-` means no results.",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        *(f"| {c} | {METRIC_LABELS[c]} |" for c in columns),
        "",
        "## Summary",
        "",
        f"`n` is compared against {expected} seeds, the most complete experiment of this set, per domain.",
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
    lines += appendix or []

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")


def run_experiments(args: argparse.Namespace, experiments: list[str]) -> None:
    """Print one table per experiment and optionally write them as markdown."""
    results = {e: _collect(e, args) for e in experiments}
    if args.seeds:
        print(f"seeds: {format_array(sorted(args.seeds))}")
    for experiment in experiments:
        print_table(results[experiment], experiment)
    domains = sorted({d for by_domain in results.values() for d in by_domain})
    coverage = _coverage(
        [(e, d) for e in experiments for d in available_domains()], args
    )
    domains = sorted(set(domains) | {d for _, d, _ in coverage})
    print_coverage(coverage, args.include_partial)
    if not args.markdown:
        return
    if not domains:
        raise SystemExit("--markdown: no results found, nothing written")
    write_experiments_markdown(
        args.markdown,
        experiments,
        results,
        domains,
        args.source,
        args.seeds,
        coverage_markdown(coverage, args.include_partial),
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
        help="Experiment name(s) (= W&B group). Several names are allowed without a sweep: each gets its own table, and --markdown writes them into one document. A sweep takes exactly one (its base name).",
    )
    p.add_argument("--source", default="local", choices=["local", "wandb"])
    p.add_argument(
        "--domains", nargs="+", default=None, help="Restrict to these domains."
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        metavar="SEED",
        help="Restrict every experiment to these seeds, as single values and/or inclusive ranges (e.g. '--seeds 0-4'). Default: every seed found.",
    )
    p.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    p.add_argument("--wandb-project", default=WANDB_PROJECT)
    p.add_argument("--wandb-entity", default=WANDB_ENTITY)
    p.add_argument(
        "--sweep",
        default=None,
        choices=sorted(SWEEP_AXES),
        help="Aggregate the sweep launched under --experiment (sub-experiments <experiment>-<tag><value>) into a domain x value matrix per metric. Default grids: "
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
        "--threshold-sweep",
        nargs="*",
        type=float,
        default=None,
        metavar="T",
        help=f"Aggregate the re-validations written by experiments/launch_threshold_validation.py (sub-experiments <experiment>-{THRESHOLD_TAG}<T>) into a domain x threshold matrix per metric. The bare flag uses the default grid "
        + ", ".join(f"{t:g}" for t in THRESHOLD_VALUES)
        + ".",
    )
    p.add_argument(
        "--include-partial",
        action="store_true",
        help="Count a seed that did not score every problem (out of memory, stalled or stopped) over the problems it scored completely, instead of leaving it out. Either way, the seeds concerned are listed per cell after the tables.",
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Also write the tables to this markdown file: one table with --sweep / --threshold-sweep, one per experiment otherwise.",
    )
    add_logging_arguments(p)
    args = p.parse_args()
    configure_logging(args)
    args.seeds = set(parse_seeds(args.seeds)) if args.seeds is not None else None

    experiments = list(dict.fromkeys(args.experiment))
    sweeping = args.sweep is not None or args.threshold_sweep is not None
    if len(experiments) > 1 and sweeping:
        raise SystemExit(
            f"--sweep / --threshold-sweep read the sub-experiments of a single base experiment; got {len(experiments)}: {', '.join(experiments)}"
        )
    args.experiment = experiments[0]

    if args.threshold_sweep is not None:
        if args.sweep is not None:
            raise SystemExit(
                "--threshold-sweep varies a validation parameter and cannot be combined with the training-parameter sweeps"
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

    run_experiments(args, experiments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
