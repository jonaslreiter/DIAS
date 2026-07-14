"""Aggregate per-seed validation JSON files into a paper-style table.

Reads the tree produced by ``scripts/run_all_domains.py``::

    logs/validation/<group>/<domain>/seed<n>.json

and emits a table of precision / recall / accuracy averaged over all
seeds and validation problems for each domain.

Usage::

    # Summary table (default)
    python scripts/aggregate_results.py --group paper-main-2026

    # Choose a different log root
    python scripts/aggregate_results.py --group paper-main-2026 --log-dir logs

    # Only selected domains
    python scripts/aggregate_results.py --group paper-main-2026 \\
        --domains satellite hanoi

    # Machine-readable CSV
    python scripts/aggregate_results.py --group paper-main-2026 --format csv

    # Write to file
    python scripts/aggregate_results.py --group paper-main-2026 \\
        --out results/paper-main-2026.csv --format csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Metrics extracted from per-problem result dicts.
NUMERIC_METRICS = ("precision", "recall", "accuracy", "matches", "total")


def _load_seed_jsons(group_dir: Path) -> dict[str, list[dict]]:
    """Return {domain: [payload, ...]} for every seed JSON found."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for seed_file in sorted(group_dir.rglob("seed*.json")):
        domain = seed_file.parent.name
        try:
            payload = json.loads(seed_file.read_text())
        except Exception as exc:
            print(f"[warn] could not read {seed_file}: {exc}", file=sys.stderr)
            continue
        by_domain[domain].append(payload)
    return dict(by_domain)


def _aggregate_domain(payloads: list[dict]) -> dict[str, Any]:
    """Compute per-domain aggregates over seeds × problems."""
    seed_precision: list[float] = []
    seed_recall: list[float] = []
    seed_accuracy: list[float] = []
    total_matches = 0
    total_states = 0
    ok_seeds = 0
    failed_seeds = 0

    for payload in payloads:
        if payload.get("status") not in ("ok",):
            failed_seeds += 1
            continue
        ok_seeds += 1
        per_problem: dict = payload.get("per_problem", {})
        if not per_problem:
            continue

        problem_precisions: list[float] = []
        problem_recalls: list[float] = []
        problem_accuracies: list[float] = []

        for stem, result in per_problem.items():
            if "error" in result:
                continue
            if "precision" in result:
                problem_precisions.append(float(result["precision"]))
            if "recall" in result:
                problem_recalls.append(float(result["recall"]))
            if "accuracy" in result:
                problem_accuracies.append(float(result["accuracy"]))
            if "matches" in result:
                total_matches += int(result["matches"])
            if "total" in result:
                total_states += int(result["total"])

        if problem_precisions:
            seed_precision.append(sum(problem_precisions) / len(problem_precisions))
        if problem_recalls:
            seed_recall.append(sum(problem_recalls) / len(problem_recalls))
        if problem_accuracies:
            seed_accuracy.append(sum(problem_accuracies) / len(problem_accuracies))

    def _stats(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return float("nan"), float("nan")
        mean = sum(vals) / len(vals)
        if len(vals) == 1:
            return mean, float("nan")
        variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return mean, math.sqrt(variance)

    p_mean, p_std = _stats(seed_precision)
    r_mean, r_std = _stats(seed_recall)
    a_mean, a_std = _stats(seed_accuracy)

    return {
        "ok_seeds": ok_seeds,
        "failed_seeds": failed_seeds,
        "precision_mean": p_mean,
        "precision_std": p_std,
        "recall_mean": r_mean,
        "recall_std": r_std,
        "accuracy_mean": a_mean,
        "accuracy_std": a_std,
        "total_matches": total_matches,
        "total_states": total_states,
    }


def _fmt_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    if math.isnan(mean):
        return "n/a"
    m_str = f"{mean:.{decimals}f}"
    if math.isnan(std):
        return m_str
    return f"{m_str} ± {std:.{decimals}f}"


def print_table(rows: list[dict[str, Any]], domains: list[str]) -> None:
    col_domain = max(max((len(d) for d in domains), default=6), 6)
    col_val = 18  # "0.123 ± 0.045" fits

    header = (
        f"{'Domain':<{col_domain}}  "
        f"{'Precision':>{col_val}}  "
        f"{'Recall':>{col_val}}  "
        f"{'Accuracy':>{col_val}}  "
        f"{'Seeds':>6}  "
        f"{'States':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        d = row["domain"]
        agg = row["agg"]
        print(
            f"{d:<{col_domain}}  "
            f"{_fmt_mean_std(agg['precision_mean'], agg['precision_std']):>{col_val}}  "
            f"{_fmt_mean_std(agg['recall_mean'], agg['recall_std']):>{col_val}}  "
            f"{_fmt_mean_std(agg['accuracy_mean'], agg['accuracy_std']):>{col_val}}  "
            f"{agg['ok_seeds']:>6}  "
            f"{agg['total_states']:>8}"
        )
    print(sep)


def write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    fieldnames = [
        "domain",
        "ok_seeds",
        "failed_seeds",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "accuracy_mean",
        "accuracy_std",
        "total_matches",
        "total_states",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            entry = {"domain": row["domain"]}
            entry.update(row["agg"])
            writer.writerow(entry)
    print(f"Wrote {out}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Aggregate per-seed validation JSONs into a summary table."
    )
    p.add_argument("--group", required=True, help="W&B / run group name.")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "logs",
        help="Root log directory (default: logs/).",
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Restrict to these domains (default: all found).",
    )
    p.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write output to this file (default: stdout for table/json, required for csv).",
    )
    args = p.parse_args()

    group_dir = args.log_dir / "validation" / args.group
    if not group_dir.exists():
        print(
            f"error: validation directory not found: {group_dir}",
            file=sys.stderr,
        )
        return 1

    by_domain = _load_seed_jsons(group_dir)
    if not by_domain:
        print(f"error: no seed*.json files found under {group_dir}", file=sys.stderr)
        return 1

    if args.domains:
        missing = [d for d in args.domains if d not in by_domain]
        if missing:
            print(
                f"warn: domains not found in results: {missing}", file=sys.stderr
            )
        selected = args.domains
    else:
        selected = sorted(by_domain)

    rows: list[dict] = []
    for domain in selected:
        if domain not in by_domain:
            continue
        agg = _aggregate_domain(by_domain[domain])
        rows.append({"domain": domain, "agg": agg})

    if not rows:
        print("error: no data to aggregate.", file=sys.stderr)
        return 1

    if args.format == "table":
        print_table(rows, selected)
        if args.out:
            # Redirect stdout to file by writing directly.
            lines: list[str] = []
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_table(rows, selected)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(buf.getvalue())
            print(f"Wrote {args.out}", file=sys.stderr)
    elif args.format == "csv":
        if args.out is None:
            # Write CSV to stdout.
            import io
            buf = io.StringIO()
            fieldnames = [
                "domain",
                "ok_seeds",
                "failed_seeds",
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
                "accuracy_mean",
                "accuracy_std",
                "total_matches",
                "total_states",
            ]
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                entry = {"domain": row["domain"]}
                entry.update(row["agg"])
                writer.writerow(entry)
            print(buf.getvalue(), end="")
        else:
            write_csv(rows, args.out)
    elif args.format == "json":
        import json as _json
        result = [{"domain": r["domain"], **r["agg"]} for r in rows]
        text = _json.dumps(result, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print(f"Wrote {args.out}", file=sys.stderr)
        else:
            print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
