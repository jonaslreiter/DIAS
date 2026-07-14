"""Run MLP-baseline validation for every (domain, seed) pair.

This script mirrors ``scripts/run_all_domains.py`` but runs the MLP-effects
baseline validation (``dias.validation.mlp_baseline.ModelValidation``) instead
of the DIAS structural validator.  It does **not** train new models: it reuses
the checkpoints produced by a prior DIAS training sweep.

Run-id discovery
----------------
The script needs the W&B run-id produced for each (domain, seed) during DIAS
training.  It reads these from the per-seed JSONs written by
``run_all_domains.py``:

    logs/validation/<dias-group>/<domain>/seed<seed>.json

Pass the name of that group via ``--dias-group``.

Alternatively, supply a hand-crafted JSON mapping::

    {"satellite": {"0": "run_abc123", "3": "run_def456", ...}, ...}

via ``--run-ids-file``.

Output
------
Results are written to::

    logs/validation/<group>/<domain>/seed<seed>.json

where ``<group>`` defaults to ``<dias-group>-mlp``.

Usage::

    # All domains × all seeds from a prior DIAS sweep
    python scripts/run_all_domains_mlp.py --dias-group paper-main-2026

    # Override the output group name
    python scripts/run_all_domains_mlp.py \\
        --dias-group paper-main-2026 \\
        --group paper-main-2026-mlp

    # Subset
    python scripts/run_all_domains_mlp.py \\
        --dias-group paper-main-2026 \\
        --domains satellite hanoi --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dias.validation.mlp_baseline import ModelValidation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_domain_settings(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_validation_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _discover_run_ids(
    dias_group_dir: Path,
    domains: list[str],
    seeds: list[int],
) -> dict[str, dict[int, str]]:
    """Read run-ids from DIAS sweep JSONs.

    Returns {domain: {seed: run_id}}.
    """
    run_ids: dict[str, dict[int, str]] = {}
    for domain in domains:
        domain_dir = dias_group_dir / domain
        if not domain_dir.exists():
            continue
        run_ids.setdefault(domain, {})
        for seed in seeds:
            seed_file = domain_dir / f"seed{seed}.json"
            if not seed_file.exists():
                continue
            try:
                payload = json.loads(seed_file.read_text())
                rid = payload.get("run_id")
                if rid:
                    run_ids[domain][seed] = rid
            except Exception as exc:
                print(f"[warn] could not read {seed_file}: {exc}")
    return run_ids


def validate_mlp_one(
    domain: str,
    seed: int,
    run_id: str,
    spec: dict,
    val_cfg: dict,
    log_dir: Path,
    num_workers: int,
    batch_size: int,
) -> dict:
    """Run MLP validation for one (domain, seed) across val problems."""
    offset = int(val_cfg["problem_offset"])
    n = int(val_cfg["num_problems"])
    val_problems = spec["problems"][offset : offset + n]

    per_problem: dict[str, dict[str, Any]] = {}
    for problem in val_problems:
        problem_abs = PROJECT_ROOT / problem
        stem = Path(problem).stem
        print(f"--- mlp validate  {domain}  seed={seed}  problem={stem} ---", flush=True)
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
            result = v.validate(num_workers=num_workers, batch_size=batch_size)
        except Exception as e:
            traceback.print_exc()
            per_problem[stem] = {"error": str(e)}
            continue

        per_problem[stem] = {
            k: (float(v_) if isinstance(v_, (int, float)) else str(v_))
            for k, v_ in result.items()
        }

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
    payload: dict[str, Any] = {
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
    import sys

    p = argparse.ArgumentParser(
        description="MLP-baseline validation sweep over all (domain, seed) pairs."
    )
    p.add_argument(
        "--dias-group",
        required=True,
        help="Name of the prior DIAS training sweep group used to look up run-ids.",
    )
    p.add_argument(
        "--group",
        type=str,
        default=None,
        help="Output group name for this MLP sweep (default: <dias-group>-mlp).",
    )
    p.add_argument(
        "--domain-settings",
        type=Path,
        default=PROJECT_ROOT / "pddl_files" / "domain-settings.yaml",
    )
    p.add_argument(
        "--validation-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "validation.yaml",
    )
    p.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs")

    # Override run-id discovery.
    p.add_argument(
        "--run-ids-file",
        type=Path,
        default=None,
        help="JSON file mapping {domain: {seed: run_id}} (overrides DIAS sweep JSONs).",
    )

    # Subset filters.
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--domains", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)

    # MLP-specific knobs.
    p.add_argument("--num-workers", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)

    args = p.parse_args()

    output_group = args.group or f"{args.dias_group}-mlp"

    domain_settings = load_domain_settings(args.domain_settings)
    val_cfg = load_validation_config(args.validation_config)

    # Determine domain/seed list.
    if args.domain:
        domains = [args.domain]
    elif args.domains:
        domains = args.domains
    else:
        domains = sorted(domain_settings.keys())

    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = list(val_cfg["seeds"])

    for d in domains:
        if d not in domain_settings:
            print(f"error: domain '{d}' not in domain settings", file=sys.stderr)
            return 1

    # Discover run-ids.
    if args.run_ids_file:
        raw = json.loads(args.run_ids_file.read_text())
        run_ids: dict[str, dict[int, str]] = {
            dom: {int(s): rid for s, rid in seed_map.items()}
            for dom, seed_map in raw.items()
        }
    else:
        dias_group_dir = args.log_dir / "validation" / args.dias_group
        if not dias_group_dir.exists():
            print(
                f"error: DIAS validation dir not found: {dias_group_dir}\n"
                "Run 'python scripts/run_all_domains.py --group <dias-group>' first,\n"
                "or pass --run-ids-file to supply run-ids manually.",
                file=sys.stderr,
            )
            return 1
        run_ids = _discover_run_ids(dias_group_dir, domains, seeds)

    results_dir = args.log_dir / "validation" / output_group
    failures: list[tuple[str, int, str]] = []
    pairs = [(d, s) for d in domains for s in seeds]

    print(f"dias-group : {args.dias_group}")
    print(f"out-group  : {output_group}")
    print(f"pairs      : {len(pairs)}")
    print(f"log_dir    : {args.log_dir}")

    for domain, seed in pairs:
        spec = domain_settings[domain]
        out_json = results_dir / domain / f"seed{seed}.json"

        rid = run_ids.get(domain, {}).get(seed)
        if rid is None:
            msg = "run_id not found (check --dias-group or --run-ids-file)"
            print(f"[skip] {domain} seed={seed}: {msg}", flush=True)
            failures.append((domain, seed, msg))
            write_local_json(
                out_json, domain, seed, "", output_group,
                status="skipped", error=msg,
            )
            continue

        print(f"\n=== mlp validate  {domain}  seed={seed}  run_id={rid} ===", flush=True)
        try:
            per_problem = validate_mlp_one(
                domain=domain,
                seed=seed,
                run_id=rid,
                spec=spec,
                val_cfg=val_cfg,
                log_dir=args.log_dir,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
            )
            write_local_json(
                out_json, domain, seed, rid, output_group,
                status="ok", per_problem=per_problem,
            )
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            failures.append((domain, seed, f"validation failed: {msg}"))
            write_local_json(
                out_json, domain, seed, rid, output_group,
                status="validation_failed", error=msg,
            )

    print(f"\n=== done. {len(pairs) - len(failures)}/{len(pairs)} ok ===")
    for d, s, msg in failures:
        print(f"  FAILED  {d} seed={s}: {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
