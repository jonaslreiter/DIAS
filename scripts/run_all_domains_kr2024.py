"""Run KR-2024 baseline validation for every (domain, seed) pair.

Unlike the DIAS / MLP sweeps, the KR-2024 baseline does **not** produce a
PyTorch checkpoint — it outputs a *symbolic PDDL domain file* directly.
This script therefore expects a directory of pre-generated KR-2024 learned
domain files and validates each against the ground-truth domain.

Expected layout of ``--learned-domains-dir``::

    <learned-domains-dir>/
        <domain>/
            seed<seed>.pddl      # one file per (domain, seed)

Output::

    logs/validation/<group>/<domain>/seed<seed>.json

Usage::

    # All domains × all seeds (domain files must exist)
    python scripts/run_all_domains_kr2024.py \\
        --learned-domains-dir path/to/kr2024_output \\
        --group paper-main-2026-kr2024

    # Subset
    python scripts/run_all_domains_kr2024.py \\
        --learned-domains-dir path/to/kr2024_output \\
        --group paper-main-2026-kr2024 \\
        --domains satellite hanoi --seeds 0 1 2

    # Summarise afterwards
    python scripts/aggregate_results.py --group paper-main-2026-kr2024
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import yaml

from dias.validation.kr2024 import ModelValidation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_domain_settings(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_validation_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def validate_kr2024_one(
    domain: str,
    seed: int,
    original_domain_path: Path,
    learned_domain_path: Path,
    spec: dict,
    val_cfg: dict,
    log_dir: Path,
) -> dict:
    """Run KR-2024 validation for one (domain, seed) across validation problems."""
    offset = int(val_cfg["problem_offset"])
    n = int(val_cfg["num_problems"])
    val_problems = spec["problems"][offset : offset + n]

    per_problem: dict[str, dict[str, Any]] = {}
    for problem in val_problems:
        problem_abs = PROJECT_ROOT / problem
        stem = Path(problem).stem
        print(
            f"--- kr2024 validate  {domain}  seed={seed}  problem={stem} ---",
            flush=True,
        )
        try:
            v = ModelValidation(
                original_domain_path=str(original_domain_path),
                learned_domain_path=str(learned_domain_path),
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

        per_problem[stem] = {
            k: (float(v_) if isinstance(v_, (int, float)) else str(v_))
            for k, v_ in result.items()
        }

    return per_problem


def write_local_json(
    out_path: Path,
    domain: str,
    seed: int,
    learned_domain_path: str,
    group: str,
    status: str,
    per_problem: dict | None = None,
    error: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "domain": domain,
        "seed": seed,
        "learned_domain": learned_domain_path,
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
        description="KR-2024 baseline validation sweep over all (domain, seed) pairs."
    )
    p.add_argument(
        "--learned-domains-dir",
        type=Path,
        required=True,
        help=(
            "Root directory of KR-2024 learned PDDL files.  "
            "Expected structure: <dir>/<domain>/seed<n>.pddl"
        ),
    )
    p.add_argument(
        "--group",
        type=str,
        required=True,
        help="Output group name for this sweep.",
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

    # Subset filters.
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--domains", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)

    args = p.parse_args()

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

    results_dir = args.log_dir / "validation" / args.group
    failures: list[tuple[str, int, str]] = []
    pairs = [(d, s) for d in domains for s in seeds]

    print(f"learned-domains-dir : {args.learned_domains_dir}")
    print(f"group               : {args.group}")
    print(f"pairs               : {len(pairs)}")
    print(f"log_dir             : {args.log_dir}")

    for domain, seed in pairs:
        spec = domain_settings[domain]
        original_domain_path = PROJECT_ROOT / spec["domain"]
        learned_domain_path = args.learned_domains_dir / domain / f"seed{seed}.pddl"
        out_json = results_dir / domain / f"seed{seed}.json"

        if not learned_domain_path.exists():
            msg = f"learned domain file not found: {learned_domain_path}"
            print(f"[skip] {domain} seed={seed}: {msg}", flush=True)
            failures.append((domain, seed, msg))
            write_local_json(
                out_json, domain, seed, str(learned_domain_path), args.group,
                status="skipped", error=msg,
            )
            continue

        print(
            f"\n=== kr2024 validate  {domain}  seed={seed} ===",
            flush=True,
        )
        try:
            per_problem = validate_kr2024_one(
                domain=domain,
                seed=seed,
                original_domain_path=original_domain_path,
                learned_domain_path=learned_domain_path,
                spec=spec,
                val_cfg=val_cfg,
                log_dir=args.log_dir,
            )
            write_local_json(
                out_json, domain, seed, str(learned_domain_path), args.group,
                status="ok", per_problem=per_problem,
            )
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            failures.append((domain, seed, f"validation failed: {msg}"))
            write_local_json(
                out_json, domain, seed, str(learned_domain_path), args.group,
                status="validation_failed", error=msg,
            )

    print(f"\n=== done. {len(pairs) - len(failures)}/{len(pairs)} ok ===")
    for d, s, msg in failures:
        print(f"  FAILED  {d} seed={s}: {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
