#!/usr/bin/env python
"""Launch the instance-size scaling study: one Slurm array per domain, each task walking that domain's whole size ladder on one exclusive GPU. No validation array; this study produces timings, not schemas.

Unlike ``experiments/launch.py`` the arrays are not one-task-per-run: the sizes of one (domain, seed) share a task so the curve is measured on one card, and the ``slurm:`` block of ``sizes.yaml`` asks for a whole GPU on a pinned partition rather than a shard.

Results land in ``logs/scaling/<experiment>/<domain>/<instance>-s<seed>.json``; aggregate with ``experiments/scaling/collect_scaling.py``.

Examples
--------
    # Full grid (3 domains x 6 sizes x 3 seeds) on Slurm, or one domain, printed only.
    python experiments/scaling/launch_scaling.py --experiment scaling-a10
    python experiments/scaling/launch_scaling.py --domains delivery --dry-run

    # Re-run the seeds that died, keeping the finished points; or run here.
    python experiments/scaling/launch_scaling.py --experiment scaling-a10 --domains blocks-3 --seeds 1 2 --skip-existing
    python experiments/scaling/launch_scaling.py --backend local --domains blocks-3 --seeds 0 --experiment smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SCALING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCALING_DIR.parents[1]
SLURM_SCRIPT = SCALING_DIR / "slurm" / "benchmark.sh"

# Appended, not prepended, so the sibling imports below are not shadowed.
sys.path.append(str(SCALING_DIR.parent))
from benchmark import DEFAULT_RESULTS_DIR, SIZES_CONFIG  # noqa: E402

# Reused, so this study's sbatch flags mean what the main pipeline's mean.
from pipeline import (  # noqa: E402
    format_array,
    prepare_slurm_logs,
    run_command,
    sbatch_output_opt,
    sbatch_resource_opts,
)
from run_scaling import instances_for, run_ladder  # noqa: E402


def _submit_domain(
    domain: str,
    seeds: list[int],
    resources: dict,
    args: argparse.Namespace,
) -> None:
    export = ",".join(
        [
            "ALL",
            f"DIAS_ROOT={PROJECT_ROOT}",
            f"EXPERIMENT={args.experiment}",
            f"DOMAIN={domain}",
            f"RESULTS_DIR={args.results_dir}",
            f"SIZES_CONFIG={args.sizes_config}",
            f"SKIP_EXISTING={1 if args.skip_existing else 0}",
        ]
    )
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=dias-scaling-{domain}",
        f"--array={format_array(seeds)}",
        sbatch_output_opt(PROJECT_ROOT / "logs"),
        *sbatch_resource_opts(resources),
        f"--export={export}",
        str(SLURM_SCRIPT),
    ]
    print(f"[{domain}] submit scaling array ({len(seeds)} seeds)")
    run_command(cmd, args.dry_run)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", default="scaling", help="Results group name.")
    p.add_argument(
        "--domains", nargs="+", default=None, help="Default: all in sizes.yaml."
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="Default: sizes.yaml seeds."
    )
    p.add_argument("--backend", choices=["slurm", "local"], default="slurm")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--sizes-config", type=Path, default=SIZES_CONFIG)
    p.add_argument("--time", default=None, help="Override the Slurm --time.")
    p.add_argument("--mem", default=None, help="Override the Slurm --mem.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (domain, size, seed) points that already have a result JSON.",
    )
    p.add_argument("--accelerator", default=None, help="local backend only, e.g. cpu.")
    p.add_argument("--dry-run", action="store_true", help="Print, do not submit/run.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.sizes_config.read_text())
    domains = args.domains or list(cfg["domains"])
    unknown = [d for d in domains if d not in cfg["domains"]]
    if unknown:
        raise SystemExit(f"Unknown domain(s): {unknown}. Known: {list(cfg['domains'])}")
    seeds = sorted(args.seeds if args.seeds is not None else cfg["seeds"])

    resources = dict(cfg.get("slurm", {}))
    if args.time:
        resources["time"] = args.time
    if args.mem:
        resources["mem"] = args.mem

    missing = [
        f"{domain}/{instance}"
        for domain in domains
        for instance in instances_for(domain, cfg)
        if not (SCALING_DIR / "instances" / domain / f"{instance}.pddl").exists()
    ]
    if missing:
        raise SystemExit(
            "Missing instance file(s): "
            + ", ".join(missing)
            + "\nRegenerate them with experiments/scaling/generate_instances.py --generators-dir DIR (see experiments/scaling/README.md)."
        )

    total = sum(len(instances_for(d, cfg)) for d in domains) * len(seeds)
    print(
        f"experiment={args.experiment}  backend={args.backend}  domains={domains}  seeds={seeds}  points={total}\n"
    )

    if args.backend == "slurm":
        if not args.dry_run:
            prepare_slurm_logs(PROJECT_ROOT / "logs")
        for domain in domains:
            _submit_domain(domain, seeds, resources, args)
        return 0

    failures = 0
    for domain in domains:
        for seed in seeds:
            if args.dry_run:
                print(f"[{domain} seed={seed}] would run {instances_for(domain, cfg)}")
                continue
            failures += run_ladder(
                domain=domain,
                seed=seed,
                experiment=args.experiment,
                cfg=cfg,
                results_dir=args.results_dir,
                skip_existing=args.skip_existing,
                accelerator=args.accelerator,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
