#!/usr/bin/env python
"""Launch the instance-size scaling study: one Slurm array per domain.

Per selected domain this submits one array over the seeds (array task id =
seed); each task walks that domain's whole size ladder on one exclusive GPU via
``scaling/slurm/benchmark.sh`` -> ``scaling/run_scaling.py``. There is no second
(validation) array: this study produces timings, not schemas.

Unlike ``scripts/launch.py`` the arrays are *not* one-task-per-run — the sizes of
one (domain, seed) deliberately share a task so the whole curve is measured on
the same card. Resources come from the ``slurm:`` block of
``scaling/configs/sizes.yaml``, which asks for a whole GPU on a pinned partition
(a shard would put co-tenant load into the throughput numbers, and an unpinned
partition would mix GPU models within one curve).

Results land in ``logs/scaling/<experiment>/<domain>/<instance>-s<seed>.json``;
aggregate them with ``scaling/collect_scaling.py``.

Examples
--------
    # Full grid (3 domains x 6 sizes x 3 seeds) on Slurm:
    python scaling/launch_scaling.py --experiment scaling-a10

    # One domain, see what would be submitted:
    python scaling/launch_scaling.py --domains delivery --dry-run

    # Re-run only the seeds that died, keeping the finished points:
    python scaling/launch_scaling.py --experiment scaling-a10 --domains blocks-3 \
        --seeds 1 2 --skip-existing

    # Run here instead of on Slurm (sequential; good for a smoke test):
    python scaling/launch_scaling.py --backend local --domains blocks-3 --seeds 0 \
        --experiment smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALING_DIR = PROJECT_ROOT / "scaling"
SLURM_SCRIPT = SCALING_DIR / "slurm" / "benchmark.sh"

# Appended, not prepended: this directory must stay ahead of scripts/ so the
# sibling imports below cannot be shadowed.
sys.path.append(str(PROJECT_ROOT / "scripts"))
from benchmark import DEFAULT_RESULTS_DIR, SIZES_CONFIG  # noqa: E402

# Reused rather than re-derived so this study's sbatch flags mean exactly what
# the main pipeline's mean (see AGENTS.md on keeping launchers in sync).
from pipeline import format_array, run_command, sbatch_resource_opts  # noqa: E402
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
            f"EXPERIMENT={args.experiment}",
            f"DOMAIN={domain}",
            f"RESULTS_DIR={args.results_dir}",
            f"SIZES_CONFIG={args.sizes_config}",
            f"DOMAIN_SETTINGS={args.domain_settings}",
            f"SKIP_EXISTING={1 if args.skip_existing else 0}",
        ]
    )
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=dias-scaling-{domain}",
        f"--array={format_array(seeds)}",
        *sbatch_resource_opts(resources),
        f"--export={export}",
        str(SLURM_SCRIPT.relative_to(PROJECT_ROOT)),
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
    p.add_argument(
        "--domain-settings",
        type=Path,
        default=PROJECT_ROOT / "pddl/domain-settings.yaml",
    )
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
            + "\nRegenerate them with scaling/generate_instances.py "
            "--generators-dir DIR (see scaling/README.md)."
        )

    total = sum(len(instances_for(d, cfg)) for d in domains) * len(seeds)
    print(
        f"experiment={args.experiment}  backend={args.backend}  "
        f"domains={domains}  seeds={seeds}  points={total}\n"
    )

    if args.backend == "slurm":
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
                domain_settings=args.domain_settings,
                skip_existing=args.skip_existing,
                accelerator=args.accelerator,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
