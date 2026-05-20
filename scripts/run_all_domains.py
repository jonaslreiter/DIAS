"""Run training + validation for every (domain, seed) pair.

For each (domain, seed):
  1. Pre-generates a wandb run id.
  2. Spawns `dias-train fit` with the right domain/problem/seed/data overrides,
     wired to that run id and tagged with the experiment group.
  3. Re-opens the wandb run with resume="must" and runs
     `dias.validation.dias_validate.ModelValidation` on the next
     `num_problems` problems from the manifest, logging metrics under
     `validation/<problem_stem>/...`.
  4. Writes a per-(domain, seed) JSON to
     logs/validation/<group>/<domain>/seed<seed>.json.

Slurm-array usage (one task per (domain, seed)):
    sbatch slurm/run_all_domains.sh

Local usage:
    python scripts/run_all_domains.py                       # all domains × all seeds
    python scripts/run_all_domains.py --domain satellite    # one domain, all seeds
    python scripts/run_all_domains.py --domain satellite --seed 3
    python scripts/run_all_domains.py --domains satellite hanoi --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb
import yaml

from dias.validation.dias_validate import ModelValidation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WANDB_PROJECT = "dias"


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def load_validation_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def select_pairs(
    manifest: dict,
    val_cfg: dict,
    only_domain: str | None,
    only_domains: list[str] | None,
    only_seed: int | None,
    only_seeds: list[int] | None,
) -> list[tuple[str, int]]:
    if only_domain:
        domains = [only_domain]
    elif only_domains:
        domains = only_domains
    else:
        domains = sorted(manifest.keys())

    if only_seed is not None:
        seeds = [only_seed]
    elif only_seeds:
        seeds = only_seeds
    else:
        seeds = list(val_cfg["seeds"])

    for d in domains:
        if d not in manifest:
            raise SystemExit(f"domain '{d}' not in manifest")

    return [(d, s) for d in domains for s in seeds]


def flatten_overrides(overrides: dict, prefix: str) -> list[str]:
    args: list[str] = []
    for k, v in overrides.items():
        if v is None:
            continue  # Lightning CLI rejects `null` on the command line; rely on yaml default.
        args += [f"--{prefix}{k}", str(v)]
    return args


def train_one(
    domain: str,
    seed: int,
    spec: dict,
    run_id: str,
    run_name: str,
    group: str,
    train_config: Path,
    log_dir: Path,
) -> None:
    """Spawn `dias-train fit` for one (domain, seed) pair."""
    cmd = [
        "dias-train", "fit",
        "--config", str(train_config),
        "--seed_everything", str(seed),
        "--experiment", f"{group}/{domain}/seed{seed}",
        "--data.init_args.domain_path", str(PROJECT_ROOT / spec["domain"]),
        "--data.init_args.problem_path", str(PROJECT_ROOT / spec["problems"][0]),
        "--data.init_args.seed", str(seed),
        "--trainer.logger.init_args.id", run_id,
        "--trainer.logger.init_args.name", run_name,
        "--trainer.logger.init_args.group", group,
        "--trainer.logger.init_args.save_dir", str(log_dir),
    ]
    cmd += flatten_overrides(spec.get("data_overrides", {}), prefix="data.init_args.")

    print(f"\n=== train  {domain}  seed={seed}  run_id={run_id} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def validate_one(
    domain: str,
    seed: int,
    spec: dict,
    run_id: str,
    group: str,
    val_cfg: dict,
    log_dir: Path,
) -> dict:
    """Resume the wandb run, validate next-N problems, log + return results."""
    offset = int(val_cfg["problem_offset"])
    n = int(val_cfg["num_problems"])
    val_problems = spec["problems"][offset : offset + n]

    wandb.init(
        id=run_id,
        resume="must",
        project=WANDB_PROJECT,
        group=group,
        dir=str(log_dir),
    )

    per_problem: dict[str, dict[str, Any]] = {}
    for problem in val_problems:
        problem_abs = PROJECT_ROOT / problem
        stem = Path(problem).stem
        print(f"--- validate  {domain}  seed={seed}  problem={stem} ---", flush=True)
        try:
            v = ModelValidation(
                model_id=run_id,
                logs_path_abs=str(log_dir),
                problem_path=str(problem_abs),
            )
            v.setup(
                num_samples=int(val_cfg["num_samples"]),
                sample_population=int(val_cfg["sample_population"]),
            )
            result = v.validate()
        except Exception as e:
            traceback.print_exc()
            per_problem[stem] = {"error": str(e)}
            continue

        numeric = {k: float(v_) for k, v_ in result.items()
                   if isinstance(v_, (int, float)) and not isinstance(v_, bool)}
        wandb.log({f"validation/{stem}/{k}": v_ for k, v_ in numeric.items()})
        per_problem[stem] = {k: (float(v_) if isinstance(v_, (int, float)) else str(v_))
                             for k, v_ in result.items()}

    wandb.finish()
    return per_problem


def write_local_json(
    out_path: Path,
    domain: str,
    seed: int,
    run_id: str,
    group: str,
    status: str,
    per_problem: dict | None = None,
    error: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "domain": domain,
        "seed": seed,
        "run_id": run_id,
        "group": group,
        "status": status,
        "per_problem": per_problem or {},
    }
    if error is not None:
        payload["error"] = error
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=PROJECT_ROOT / "pddl_files" / "domain_manifest.json")
    p.add_argument("--train-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "pddl_graph_learner.yaml")
    p.add_argument("--validation-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "validation.yaml")
    p.add_argument("--group", type=str,
                   default=os.environ.get(
                       "DIAS_GROUP",
                       f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                   help="W&B group name. All (domain, seed) runs share this. "
                        "Defaults to a timestamp.")
    p.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs")

    # Single-pair (for slurm array).
    p.add_argument("--domain", type=str, default=None,
                   help="Run this single domain only.")
    p.add_argument("--seed", type=int, default=None,
                   help="Run this single seed only.")

    # Subset (for local).
    p.add_argument("--domains", nargs="+", default=None,
                   help="Subset of domains (alternative to --domain).")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Subset of seeds (alternative to --seed). "
                        "Defaults to validation.yaml `seeds`.")

    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    val_cfg = load_validation_config(args.validation_config)

    pairs = select_pairs(manifest, val_cfg,
                         args.domain, args.domains, args.seed, args.seeds)

    print(f"group   : {args.group}")
    print(f"pairs   : {len(pairs)} = {len(set(d for d,_ in pairs))} domains × "
          f"{len(set(s for _,s in pairs))} seeds")
    print(f"log_dir : {args.log_dir}")

    results_dir = args.log_dir / "validation" / args.group
    failures: list[tuple[str, int, str]] = []

    for domain, seed in pairs:
        spec = manifest[domain]
        run_id = wandb.util.generate_id()
        run_name = f"{domain}-s{seed}"
        out_json = results_dir / domain / f"seed{seed}.json"

        try:
            train_one(
                domain=domain, seed=seed, spec=spec,
                run_id=run_id, run_name=run_name, group=args.group,
                train_config=args.train_config, log_dir=args.log_dir,
            )
        except subprocess.CalledProcessError as e:
            msg = f"training failed (exit {e.returncode})"
            print(f"[skip] {domain} seed={seed}: {msg}", flush=True)
            failures.append((domain, seed, msg))
            write_local_json(out_json, domain, seed, run_id, args.group,
                             status="train_failed", error=msg)
            continue

        try:
            per_problem = validate_one(
                domain=domain, seed=seed, spec=spec,
                run_id=run_id, group=args.group,
                val_cfg=val_cfg, log_dir=args.log_dir,
            )
            write_local_json(out_json, domain, seed, run_id, args.group,
                             status="ok", per_problem=per_problem)
        except Exception as e:
            traceback.print_exc()
            failures.append((domain, seed, f"validation failed: {e}"))
            write_local_json(out_json, domain, seed, run_id, args.group,
                             status="validation_failed", error=str(e))

    print(f"\n=== done. {len(pairs) - len(failures)}/{len(pairs)} ok ===")
    for d, s, msg in failures:
        print(f"  FAILED  {d} seed={s}: {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
