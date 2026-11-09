#!/usr/bin/env python
"""Launch the scaling study: per domain, a Slurm array with one task per seed, which benchmarks the domain's instances on a whole GPU. ``--seeds``, ``--num-runs``, ``--log-dir``, ``--backend`` and ``--dry-run`` work as for ``experiments/launch.py``.

Examples:
    python experiments/scaling/launch_scaling.py --experiment scaling --seeds 0-4
    python experiments/scaling/launch_scaling.py --experiment smoke --backend local --domains blocks-3 --seeds 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SCALING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCALING_DIR.parents[1]
SLURM_SCRIPT = SCALING_DIR / "slurm" / "benchmark.sh"

sys.path.append(str(SCALING_DIR.parent))
from benchmark import SIZES_CONFIG, results_dir  # noqa: E402
from pipeline import (  # noqa: E402
    LOGS_DIR,
    add_seed_arguments,
    format_array,
    prepare_slurm_logs,
    resolve_seeds,
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
            f"LOGS_DIR={args.log_dir}",
            f"SIZES_CONFIG={args.sizes_config}",
            f"SKIP_EXISTING={1 if args.skip_existing else 0}",
        ]
    )
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=dias-scaling-{domain}",
        f"--array={format_array(seeds)}",
        sbatch_output_opt(args.log_dir),
        *sbatch_resource_opts(resources),
        f"--export={export}",
        str(SLURM_SCRIPT),
    ]
    print(f"[{domain}] submit scaling array ({len(seeds)} seeds {format_array(seeds)})")
    run_command(cmd, args.dry_run)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", default="scaling", help="Experiment name.")
    p.add_argument(
        "--domains", nargs="+", default=None, help="Default: all in sizes.yaml."
    )
    add_seed_arguments(p, "the seeds sizes.yaml lists")
    p.add_argument("--backend", choices=["slurm", "local"], default="slurm")
    p.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    p.add_argument("--sizes-config", type=Path, default=SIZES_CONFIG)
    p.add_argument("--time", default=None, help="Override the Slurm --time.")
    p.add_argument("--mem", default=None, help="Override the Slurm --mem.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip the points that have a result.",
    )
    p.add_argument("--accelerator", default=None, help="local backend only, e.g. cpu.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(args.sizes_config.read_text())
    domains = args.domains or list(cfg["domains"])
    unknown = [d for d in domains if d not in cfg["domains"]]
    if unknown:
        raise SystemExit(f"Unknown domain(s): {unknown}. Known: {list(cfg['domains'])}")
    seeds = resolve_seeds(args, default=cfg["seeds"])

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
        raise SystemExit("Missing instance file(s): " + ", ".join(missing))

    total = sum(len(instances_for(d, cfg)) for d in domains) * len(seeds)
    print(f"experiment : {args.experiment}")
    print(
        f"domains    : {len(domains)} x {len(seeds)} runs  (seeds={format_array(seeds)}, backend={args.backend})"
    )
    print(f"points     : {total}")
    print(f"results    : {results_dir(args.log_dir, args.experiment)}/\n")

    if args.backend == "slurm":
        if not args.dry_run:
            prepare_slurm_logs(args.log_dir)
        for domain in domains:
            _submit_domain(domain, seeds, resources, args)
        print("\nDone submitting.")
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
                log_dir=args.log_dir,
                skip_existing=args.skip_existing,
                accelerator=args.accelerator,
            )
    print("\nDone.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
