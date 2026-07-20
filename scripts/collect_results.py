"""Aggregate validation results for an experiment into a paper-style table.

Reads per-run results (written by ``scripts/validate_runs.py``) from local JSON
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

Usage:
    python scripts/collect_results.py --experiment paper-main
    python scripts/collect_results.py --experiment paper-main --source wandb
    python scripts/collect_results.py --experiment noise-a2r2 --domains gripper hanoi
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WANDB_PROJECT = "dias"
ONE = 1.0 - 1e-9


def _seed_precision_recall(per_problem: dict) -> tuple[float, float] | None:
    """Reduce one seed's per-problem results to a single (precision, recall)."""
    tp = fp = fn = 0
    precisions: list[float] = []
    recalls: list[float] = []
    saw_stats = False
    any_ok = False
    for result in per_problem.values():
        if not isinstance(result, dict) or "error" in result:
            continue
        any_ok = True
        stats = result.get("action_stats")
        if isinstance(stats, dict):
            saw_stats = True
            for a in stats.values():
                tp += int(a.get("tp", 0))
                fp += int(a.get("fp", 0))
                fn += int(a.get("fn", 0))
        if isinstance(result.get("precision"), (int, float)):
            precisions.append(float(result["precision"]))
        if isinstance(result.get("recall"), (int, float)):
            recalls.append(float(result["recall"]))
    if not any_ok:
        return None
    if saw_stats:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        p = sum(precisions) / len(precisions) if precisions else 0.0
        r = sum(recalls) / len(recalls) if recalls else 0.0
    return p, r


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
    key_re = re.compile(r"^validation/(?P<stem>.+)/(?P<metric>precision|recall)$")
    for run in api.runs(path, filters={"group": experiment}):
        domain = run.config.get("domain")
        if domain is None:
            continue
        per_problem: dict[str, dict] = {}
        for key, val in run.summary.items():
            m = key_re.match(str(key))
            if m and isinstance(val, (int, float)):
                per_problem.setdefault(m["stem"], {})[m["metric"]] = float(val)
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


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", required=True, help="Experiment name (= W&B group).")
    p.add_argument("--source", default="local", choices=["local", "wandb"])
    p.add_argument(
        "--domains", nargs="+", default=None, help="Restrict to these domains."
    )
    p.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs")
    p.add_argument("--wandb-project", default=WANDB_PROJECT)
    p.add_argument("--wandb-entity", default=None)
    args = p.parse_args()

    if args.source == "local":
        by_domain = collect_local(args.experiment, args.logs_dir)
    else:
        by_domain = collect_wandb(
            args.experiment, args.wandb_project, args.wandb_entity
        )

    if args.domains:
        by_domain = {d: v for d, v in by_domain.items() if d in args.domains}

    print_table(by_domain, args.experiment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
