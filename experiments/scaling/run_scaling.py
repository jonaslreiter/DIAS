#!/usr/bin/env python
"""Walk one domain's size ladder for one seed (compute-side driver).

This is what a Slurm array task runs (``experiments/scaling/slurm/benchmark.sh``, array task id = seed) and what ``experiments/scaling/launch_scaling.py --backend local`` calls directly. It is the analogue of ``experiments/validate_runs.py`` in the main pipeline.

Every size runs as its own ``experiments/scaling/benchmark.py`` subprocess. That is the point of the indirection: a CUDA OOM (expected at the top of the ladder) leaves the process's allocator in a state not worth trusting for the next measurement, and one dead subprocess costs one point instead of the whole ladder. The subprocess still records the OOM to JSON, so the ladder keeps its data.

All sizes of one (domain, seed) run in the same task, hence on the same physical GPU, so a curve is never stitched together across cards.

Usage
-----
    python experiments/scaling/run_scaling.py --domain blocks-3 --seed 0
    python experiments/scaling/run_scaling.py --domain delivery --seed 2 --experiment scaling-a10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from benchmark import DEFAULT_RESULTS_DIR, SIZES_CONFIG, instance_path, result_path
from generate_instances import OBJECT_COUNT, instance_id

SCALING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCALING_DIR.parents[1]


def ladder_for(domain: str, cfg: dict) -> list[tuple[int, str]]:
    """(target rung, instance id) of a domain's ladder, smallest first.

    The rung is what makes the three domains comparable: they scale in different units and cannot all land on the same object count (see sizes.yaml).
    """
    spec = cfg["domains"][domain]
    generator = spec["generator"]
    rungs = [
        (int(size["target"]), instance_id(OBJECT_COUNT[generator](size)))
        for size in spec["sizes"]
    ]
    targets = [t for t, _ in rungs]
    if len(set(targets)) != len(targets):
        raise SystemExit(f"{domain}: duplicate `target` in sizes.yaml ({targets}).")
    return rungs


def instances_for(domain: str, cfg: dict) -> list[str]:
    """Instance ids of a domain's ladder, smallest first."""
    return [instance for _, instance in ladder_for(domain, cfg)]


def run_ladder(
    domain: str,
    seed: int,
    experiment: str,
    cfg: dict,
    results_dir: Path,
    skip_existing: bool,
    accelerator: str | None = None,
) -> int:
    instances = instances_for(domain, cfg)
    print(
        f"[{domain} seed={seed}] ladder: {', '.join(instances)} -> {results_dir / experiment / domain}",
        flush=True,
    )
    failures = 0
    for instance in instances:
        out = result_path(results_dir, experiment, domain, instance, seed)
        if skip_existing and out.exists():
            print(f"[{domain} seed={seed}] {instance}: exists, skipping", flush=True)
            continue
        if not instance_path(domain, instance).exists():
            print(
                f"[{domain} seed={seed}] {instance}: missing instance file, run experiments/scaling/generate_instances.py first",
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
            "--results-dir", str(results_dir),
        ]  # fmt: skip
        if accelerator:
            cmd += ["--accelerator", accelerator]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            # An OOM exits 0 with status="oom" recorded; a non-zero exit is a real failure (crash, missing config, killed task).
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
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--sizes-config", type=Path, default=SIZES_CONFIG)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave points that already have a result JSON alone (resume a task).",
    )
    p.add_argument("--accelerator", default=None, help="Passed through (e.g. cpu).")
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
        results_dir=args.results_dir,
        skip_existing=args.skip_existing,
        accelerator=args.accelerator,
    )

    written = sorted(
        (args.results_dir / args.experiment / args.domain).glob(f"*-s{args.seed}.json")
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
