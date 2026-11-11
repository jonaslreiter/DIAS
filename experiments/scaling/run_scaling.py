#!/usr/bin/env python
"""Benchmark one domain's instances for one seed, each in a ``benchmark.py`` subprocess, so that an out-of-memory error costs one point. It is the task of a scaling array.

Example:
    python experiments/scaling/run_scaling.py --domain blocks-3 --seed 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from benchmark import SIZES_CONFIG, instance_path, result_path, results_dir
from pipeline import LOGS_DIR

SCALING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCALING_DIR.parents[1]


def ladder_for(domain: str, cfg: dict) -> list[tuple[int, str]]:
    """A domain's (target number of objects, instance), smallest first."""
    return sorted(
        (int(target), instance) for target, instance in cfg["domains"][domain].items()
    )


def instances_for(domain: str, cfg: dict) -> list[str]:
    """A domain's instances, smallest first."""
    return [instance for _, instance in ladder_for(domain, cfg)]


def run_ladder(
    domain: str,
    seed: int,
    experiment: str,
    cfg: dict,
    log_dir: Path,
    skip_existing: bool,
    accelerator: str | None = None,
) -> int:
    instances = instances_for(domain, cfg)
    print(
        f"[{domain} seed={seed}] ladder: {', '.join(instances)} -> {results_dir(log_dir, experiment) / domain}",
        flush=True,
    )
    failures = 0
    for instance in instances:
        out = result_path(log_dir, experiment, domain, instance, seed)
        if skip_existing and out.exists():
            print(f"[{domain} seed={seed}] {instance}: exists, skipping", flush=True)
            continue
        if not instance_path(domain, instance).exists():
            print(
                f"[{domain} seed={seed}] {instance}: no instance file",
                file=sys.stderr,
                flush=True,
            )
            failures += 1
            continue

        cmd = [
            sys.executable,
            str(SCALING_DIR / "benchmark.py"),
            "--domain", domain,
            "--instance", instance,
            "--seed", str(seed),
            "--experiment", experiment,
            "--warmup", str(cfg["warmup_steps"]),
            "--measure", str(cfg["measure_steps"]),
            "--log-dir", str(log_dir),
        ]  # fmt: skip
        if accelerator:
            cmd += ["--accelerator", accelerator]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            failures += 1
            print(
                f"[{domain} seed={seed}] {instance}: benchmark failed (exit {proc.returncode}, {elapsed:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
    return failures


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--domain", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--experiment", default="scaling")
    p.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    p.add_argument("--sizes-config", type=Path, default=SIZES_CONFIG)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip the points that have a result.",
    )
    p.add_argument("--accelerator", default=None, help="e.g. cpu.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.sizes_config.read_text())
    if args.domain not in cfg["domains"]:
        raise SystemExit(
            f"Unknown domain {args.domain!r}. Known: {list(cfg['domains'])}"
        )

    failures = run_ladder(
        domain=args.domain,
        seed=args.seed,
        experiment=args.experiment,
        cfg=cfg,
        log_dir=args.log_dir,
        skip_existing=args.skip_existing,
        accelerator=args.accelerator,
    )

    written = sorted(
        (results_dir(args.log_dir, args.experiment) / args.domain).glob(
            f"*-s{args.seed}.json"
        )
    )
    print(f"\n[{args.domain} seed={args.seed}] {len(written)} result(s):", flush=True)
    for path in written:
        rec = json.loads(path.read_text())
        its = rec.get("iterations_per_second")
        print(
            f"  {rec['instance']} n={rec.get('num_objects', '?'):>4} {rec['status']:5s} {('%.2f it/s' % its) if its else '        -'} {rec.get('gpu_peak_allocated_mib', 0):8.0f} MiB allocated",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
