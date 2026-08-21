"""Validate one ``(experiment, domain, seed)`` run and write its results.

This is the compute-side validation step. It is invoked once per seed by
``slurm/validate.sh`` (as an array task) and directly by
``scripts/launch.py --backend local``. The validator is chosen by experiment
type:

  dias / noise -> dias.validation.dias_validate.ModelValidation  (checkpoint)
  mlp          -> dias.validation.mlp_baseline.ModelValidation   (checkpoint)

(The L1 baseline validates an externally-produced symbolic domain and has no
training checkpoint; use ``scripts/validation/run_l1.py`` for it.)

The training run is located by a *deterministic* run id
(``<experiment>-<domain>-s<seed>``) so no job-to-job id handoff is needed:
its checkpoint lives at ``<logs>/dias/<run_id>/checkpoints/best-*.ckpt``.

Validation problems come from ``pddl/domain-settings.yaml``
(``problems[problem_offset : problem_offset + num_problems]``); the sampling
knobs come from ``configs/validation.yaml``. Results are written to
``<logs>/validation/<experiment>/<domain>/seed<seed>.json`` and, when W&B is
reachable, also logged to the resumed run under ``validation/<stem>/<metric>``.

``--threshold`` re-reads the same checkpoint at a different binarization
threshold (DIAS validator only; 0.5 is what the paper reports), and
``--results-experiment`` redirects only the *output* directory (and W&B group)
while the checkpoint is still located via ``--experiment``. Together they let a
finished experiment be re-validated into a sibling results directory without
touching its original JSONs — see ``scripts/launch_threshold_validation.py``.
"""

from __future__ import annotations

import argparse
import json
import numbers
import re
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import pipeline`

from pipeline import (  # noqa: E402
    EXPERIMENT_TYPES,
    EXPERIMENT_VALIDATOR,
    PROJECT_ROOT,
    VALIDATORS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    add_path_arguments,
    load_yaml,
    select_domains,
    validator_class,
)


def run_id_for(experiment: str, domain: str, seed: int) -> str:
    """Deterministic, filesystem/W&B-safe run id shared by train and validate."""
    raw = f"{experiment}-{domain}-s{seed}"
    return re.sub(r"[^0-9A-Za-z_-]", "-", raw)


def validate_run(
    experiment: str,
    domain: str,
    seed: int,
    experiment_type: str,
    domain_settings: dict,
    val_cfg: dict,
    logs_dir: Path,
    run_id: str | None = None,
    log_wandb: bool = True,
    threshold: float = 0.5,
    results_experiment: str | None = None,
) -> dict:
    """Validate a single run over its held-out problems; write + return results.

    The checkpoint is always found via *experiment*; *results_experiment*
    (default: *experiment*) only picks the directory the JSON is written to, so
    a re-validation at another *threshold* can be kept apart from the original.
    """
    run_id = run_id or run_id_for(experiment, domain, seed)
    results_experiment = results_experiment or experiment
    spec = domain_settings[domain]
    offset = int(val_cfg["problem_offset"])
    n = int(val_cfg["num_problems"])
    val_problems = spec["problems"][offset : offset + n]

    validator = VALIDATORS[EXPERIMENT_VALIDATOR[experiment_type]]
    ModelValidation = validator_class(validator.name)
    if not validator.binarizes_schema and threshold != 0.5:
        raise SystemExit(
            f"--threshold applies to a validator that reads a lifted schema; "
            f"{validator.summary}"
        )
    num_samples = int(val_cfg["num_samples"])
    sample_population = int(val_cfg["sample_population"])
    configured_seed = val_cfg.get("sample_seed")
    sample_seed = None if configured_seed is None else int(configured_seed)
    extra_kwargs = {"threshold": threshold} if validator.binarizes_schema else {}

    per_problem: dict[str, dict[str, Any]] = {}
    for problem in val_problems:
        stem = Path(problem).stem
        print(f"--- validate {domain} seed={seed} problem={stem} ---", flush=True)
        try:
            v = ModelValidation(
                model_id=run_id,
                logs_path_abs=str(logs_dir.resolve()),
                problem_path=str((PROJECT_ROOT / problem).resolve()),
                **extra_kwargs,
            )
            v.setup(
                num_samples=num_samples,
                sample_population=sample_population,
                seed=sample_seed,
            )
            result = v.validate()
        except Exception as e:  # noqa: BLE001 - one bad problem shouldn't sink the rest
            traceback.print_exc()
            per_problem[stem] = {"error": str(e)}
            continue
        per_problem[stem] = {k: _plain(val) for k, val in result.items()}

    _write_json(
        logs_dir,
        results_experiment,
        domain,
        seed,
        run_id,
        per_problem,
        threshold,
        sample_seed,
    )
    if log_wandb:
        _log_to_wandb(results_experiment, domain, seed, run_id, logs_dir, per_problem)
    return per_problem


def _plain(value: Any) -> Any:
    """Coerce numpy scalars to Python numbers, keeping counts integral.

    _write_json falls back to str() for anything it cannot serialise, so a
    numpy int would otherwise land in the JSON as a string.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value


def _write_json(
    logs_dir: Path,
    experiment: str,
    domain: str,
    seed: int,
    run_id: str,
    per_problem: dict,
    threshold: float = 0.5,
    sample_seed: int | None = None,
) -> None:
    out = logs_dir / "validation" / experiment / domain / f"seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = any("error" not in v for v in per_problem.values())
    out.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "domain": domain,
                "seed": seed,
                "run_id": run_id,
                "threshold": threshold,
                # null means the states were drawn unseeded, so these numbers
                # are one sample and not reproducible.
                "sample_seed": sample_seed,
                "status": "ok" if ok else "validation_failed",
                "per_problem": per_problem,
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {out}", flush=True)


def _log_to_wandb(
    experiment: str,
    domain: str,
    seed: int,
    run_id: str,
    logs_dir: Path,
    per_problem: dict,
) -> None:
    """Log numeric metrics to the resumed W&B run (best-effort)."""
    try:
        import wandb

        wandb.init(
            id=run_id,
            resume="allow",
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            group=experiment,
            dir=str(logs_dir),
        )
        # Tag with domain/seed so scripts/collect_results.py can group by domain.
        wandb.config.update({"domain": domain, "seed": seed}, allow_val_change=True)
        for stem, result in per_problem.items():
            for k, val in result.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    wandb.log({f"validation/{stem}/{k}": float(val)})
        wandb.finish()
    except (
        Exception
    ) as e:  # noqa: BLE001 - W&B is optional; JSON is the source of truth
        print(f"[warn] W&B logging skipped: {e}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, help="Experiment name (= W&B group).")
    p.add_argument(
        "--domain", required=True, help="Domain key in domain-settings.yaml."
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--experiment-type", default="dias", choices=sorted(EXPERIMENT_TYPES)
    )
    p.add_argument(
        "--run-id", default=None, help="Defaults to <experiment>-<domain>-s<seed>."
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold for slot activations and the learned "
        "precondition/effect tensors (DIAS validator only). Default 0.5.",
    )
    p.add_argument(
        "--results-experiment",
        default=None,
        help="Write results under logs/validation/<this>/ instead of "
        "<experiment>/. The checkpoint is still located via --experiment, so a "
        "re-validation at another --threshold leaves the original JSONs alone.",
    )
    add_path_arguments(p)
    p.add_argument("--no-wandb", action="store_true", help="Skip W&B logging.")
    args = p.parse_args()

    domain_settings = load_yaml(args.domain_settings)
    select_domains(domain_settings, [args.domain])

    per_problem = validate_run(
        experiment=args.experiment,
        domain=args.domain,
        seed=args.seed,
        experiment_type=args.experiment_type,
        domain_settings=domain_settings,
        val_cfg=load_yaml(args.validation_config),
        logs_dir=args.log_dir,
        run_id=args.run_id,
        log_wandb=not args.no_wandb,
        threshold=args.threshold,
        results_experiment=args.results_experiment,
    )
    failed = all("error" in v for v in per_problem.values()) if per_problem else True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
