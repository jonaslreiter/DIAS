"""Validate one ``(experiment, domain, seed)`` run and write its results.

The compute-side validation step, invoked once per seed by ``slurm/validate.sh`` and by ``launch.py --backend local``. The experiment type picks the validator: dias / noise read a DIAS checkpoint, mlp an MLP-effects one. (L1 has no checkpoint and no sweep iterates over it; reach it through ``validate_checkpoint.py --validator l1``.)

The run is located by its deterministic id, its problems are the domain folder's ``test-{1,2,3}.pddl`` and its sampling knobs come from ``experiments/configs/validation.yaml``. Results go to ``<logs>/validation/<experiment>/<domain>/seed<seed>.json``, and to the resumed W&B run when it is reachable.

``--threshold`` re-reads the same checkpoint at another binarization threshold and ``--results-experiment`` redirects only the output directory, which together re-validate a finished experiment without touching its JSONs; see ``experiments/launch_threshold_validation.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import numbers
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import pipeline`

from pipeline import configure_logging  # noqa: E402
from pipeline import (
    EXPERIMENT_TYPES,
    EXPERIMENT_VALIDATOR,
    VALIDATORS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    add_logging_arguments,
    add_path_arguments,
    load_yaml,
    select_domains,
    validation_problems,
    validator_class,
)

logger = logging.getLogger("dias.validate")


def run_id_for(experiment: str, domain: str, seed: int) -> str:
    """Deterministic, filesystem/W&B-safe run id shared by train and validate."""
    raw = f"{experiment}-{domain}-s{seed}"
    return re.sub(r"[^0-9A-Za-z_-]", "-", raw)


def validate_run(
    experiment: str,
    domain: str,
    seed: int,
    experiment_type: str,
    val_cfg: dict,
    logs_dir: Path,
    run_id: str | None = None,
    log_wandb: bool = True,
    threshold: float = 0.5,
    results_experiment: str | None = None,
) -> dict:
    """Validate a single run over its held-out problems; write and return the results.

    The checkpoint is always found via *experiment*; *results_experiment* only picks the directory the JSON goes to.
    """
    # Imported here, not at module scope: submitting jobs must not need torch or pymimir.
    from dias.validation.successors import resolve_sample_seed

    run_id = run_id or run_id_for(experiment, domain, seed)
    results_experiment = results_experiment or experiment
    val_problems = validation_problems(domain)

    validator = VALIDATORS[EXPERIMENT_VALIDATOR[experiment_type]]
    validation = validator_class(validator.name)
    if not validator.binarizes_schema and threshold != 0.5:
        raise SystemExit(
            f"--threshold applies to a validator that reads a lifted schema; {validator.summary}"
        )
    num_samples = int(val_cfg["num_samples"])
    sample_population = int(val_cfg["sample_population"])
    configured_seed = val_cfg.get("sample_seed")
    # Resolved once, so every problem of a seed shares one draw.
    sample_seed = resolve_sample_seed(
        None if configured_seed is None else int(configured_seed)
    )
    extra_kwargs = {"threshold": threshold} if validator.binarizes_schema else {}

    per_problem: dict[str, dict[str, Any]] = {}
    for problem in val_problems:
        stem = problem.stem
        logger.info(
            "Validating %s seed %d on %s (%d states of at most %d reachable).",
            domain,
            seed,
            stem,
            num_samples,
            sample_population,
        )
        try:
            v = validation(
                model_id=run_id,
                logs_path=str(logs_dir.resolve()),
                problem_path=str(problem.resolve()),
                **extra_kwargs,
            )
            v.setup(
                num_samples=num_samples,
                sample_population=sample_population,
                seed=sample_seed,
            )
            result = v.validate()
        except Exception as e:  # noqa: BLE001 - one bad problem shouldn't sink the rest
            logger.exception("Validation of %s failed.", stem)
            per_problem[stem] = {"error": str(e)}
            continue
        per_problem[stem] = {k: _plain(val) for k, val in result.items()}
        logger.info(
            "%s: precision %.3f, recall %.3f (tp=%s fp=%s fn=%s).",
            stem,
            result.get("precision", float("nan")),
            result.get("recall", float("nan")),
            result.get("tp", "?"),
            result.get("fp", "?"),
            result.get("fn", "?"),
        )

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
    """Coerce numpy scalars to Python numbers: _write_json falls back to str(), so a numpy int would land in the JSON as a string."""
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
                # Passing this back as validation.yaml's sample_seed reproduces these exact states.
                "sample_seed": sample_seed,
                "status": "ok" if ok else "validation_failed",
                "per_problem": per_problem,
            },
            indent=2,
            default=str,
        )
    )
    logger.info("Wrote %s.", out)


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
        # Tagged with domain/seed so collect_results.py can group by domain.
        wandb.config.update({"domain": domain, "seed": seed}, allow_val_change=True)
        for stem, result in per_problem.items():
            for k, val in result.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    wandb.log({f"validation/{stem}/{k}": float(val)})
        wandb.finish()
    except (
        Exception
    ) as e:  # noqa: BLE001 - W&B is optional; JSON is the source of truth
        logger.warning("W&B logging skipped: %s", e)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, help="Experiment name (= W&B group).")
    p.add_argument("--domain", required=True, help="Domain folder under pddl/.")
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
        help="Binarization threshold for slot activations and the learned precondition/effect tensors (DIAS validator only). Default 0.5.",
    )
    p.add_argument(
        "--results-experiment",
        default=None,
        help="Write results under logs/validation/<this>/ instead of <experiment>/. The checkpoint is still located via --experiment, so a re-validation at another --threshold leaves the original JSONs alone.",
    )
    add_path_arguments(p)
    add_logging_arguments(p)
    p.add_argument("--no-wandb", action="store_true", help="Skip W&B logging.")
    args = p.parse_args()
    configure_logging(args)

    select_domains([args.domain])

    per_problem = validate_run(
        experiment=args.experiment,
        domain=args.domain,
        seed=args.seed,
        experiment_type=args.experiment_type,
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
