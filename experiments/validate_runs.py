"""Validate one ``(experiment, domain, seed)`` run and write its results.

The compute-side validation step, invoked once per seed by ``slurm/validate.sh`` and by ``launch.py --backend local``. The experiment type picks the validator: dias / noise read a DIAS checkpoint, mlp an MLP-effects one. (L1 has no checkpoint and no sweep iterates over it; reach it through ``validate_checkpoint.py --validator l1``.)

The run is located by its deterministic id, its problems are the domain folder's ``test-{1,2,3}.pddl`` and its sampling knobs come from ``experiments/configs/validation.yaml``. Results go to ``<logs>/validation/<experiment>/<domain>/seed<seed>.json``, and to the resumed W&B run when it is reachable.

``--threshold`` re-reads the same checkpoint at another binarization threshold and ``--results-experiment`` redirects only the output directory, which together re-validate a finished experiment without touching its JSONs; see ``experiments/launch_threshold_validation.py``.

The MLP-effects validator grounds its sampled states across ``--num-workers`` processes, defaulting to the CPUs the job was allocated, and ``--resume`` continues a seed from what an earlier run on the same checkpoint scored.
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
    allocated_cpus,
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
    num_workers: int | None = None,
    states_per_worker: int | None = None,
    resume: bool = False,
) -> dict:
    """Validate a single run over its held-out problems; write and return the results.

    The checkpoint is always found via *experiment*; *results_experiment* only picks the directory the JSON goes to. With *resume*, problems an earlier run on the same checkpoint scored are kept, and the rest use its sample seed.
    """
    # Imported here, not at module scope: submitting jobs must not need torch or pymimir.
    from dias.validation.checkpoint import resolve_best_checkpoint
    from dias.validation.successors import resolve_sample_seed

    run_id = run_id or run_id_for(experiment, domain, seed)
    results_experiment = results_experiment or experiment
    val_problems = validation_problems(domain)
    out = _result_path(logs_dir, results_experiment, domain, seed)
    checkpoint = resolve_best_checkpoint(logs_dir / "dias" / run_id / "checkpoints")

    validator = VALIDATORS[EXPERIMENT_VALIDATOR[experiment_type]]
    validation = validator_class(validator.name)
    if not validator.binarizes_schema and threshold != 0.5:
        raise SystemExit(
            f"--threshold applies to a validator that reads a lifted schema; {validator.summary}"
        )
    num_samples = int(val_cfg["num_samples"])
    sample_population = int(val_cfg["sample_population"])
    # Only a checkpoint's schema is decoded here; L1's domain is another system's output. Written into the result, since it changes which schema is scored.
    injective = bool(val_cfg.get("injective", True)) and validator.from_checkpoint

    per_problem: dict[str, dict[str, Any]] = {}
    if resume and out.exists():
        previous = json.loads(out.read_text())
        refusal = _resume_refusal(previous, out, checkpoint, threshold, injective)
        if refusal is not None:
            raise SystemExit(f"--resume {out}: {refusal}")
        sample_seed = previous["sample_seed"]
        per_problem = {
            stem: result
            for stem, result in previous["per_problem"].items()
            if "error" not in result
        }
    else:
        # Resolved once, so every problem of a seed shares one draw.
        configured_seed = val_cfg.get("sample_seed")
        sample_seed = resolve_sample_seed(
            None if configured_seed is None else int(configured_seed)
        )

    extra_kwargs = {"threshold": threshold} if validator.binarizes_schema else {}
    if validator.from_checkpoint:
        extra_kwargs["injective"] = injective
    if validator.grounds_in_parallel:
        workers = allocated_cpus() if num_workers is None else num_workers
        logger.info("Grounding states across %d worker processes.", workers)
        extra_kwargs["num_workers"] = workers
        if states_per_worker is not None:
            extra_kwargs["states_per_worker"] = states_per_worker

    for problem in val_problems:
        stem = problem.stem
        if stem in per_problem:
            logger.info("%s is already scored.", stem)
            continue
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
            if validator.grounds_in_parallel:
                result = v.validate(out.with_name(f"seed{seed}.{stem}.states.jsonl"))
            else:
                result = v.validate()
        except Exception as e:  # noqa: BLE001 - one bad problem shouldn't sink the rest
            logger.exception("Validation of %s failed.", stem)
            per_problem[stem] = {"error": str(e)}
        else:
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
        # After every problem, so a job killed partway keeps what it finished.
        _write_json(
            logs_dir,
            results_experiment,
            domain,
            seed,
            run_id,
            {
                p.stem: per_problem[p.stem]
                for p in val_problems
                if p.stem in per_problem
            },
            threshold,
            sample_seed,
            expected=[p.stem for p in val_problems],
            checkpoint=checkpoint,
            injective=injective,
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


def _result_path(logs_dir: Path, experiment: str, domain: str, seed: int) -> Path:
    return logs_dir / "validation" / experiment / domain / f"seed{seed}.json"


def _resume_refusal(
    previous: dict, path: Path, checkpoint: Path, threshold: float, injective: bool
) -> str | None:
    """Why the results in `path` cannot be resumed, or None if they can."""
    if previous["threshold"] != threshold:
        return f"scored at threshold {previous['threshold']}, not {threshold}"
    # Results written before the key existed were all decoded as learned.
    if previous.get("injective", False) != injective:
        return (
            f"scored with injective {previous.get('injective', False)}, not {injective}"
        )
    if "checkpoint" not in previous:
        # Written before results recorded their checkpoint.
        if checkpoint.stat().st_mtime >= path.stat().st_mtime:
            return f"{checkpoint} is newer than the results"
    elif (previous["checkpoint"], previous["checkpoint_mtime"]) != (
        str(checkpoint),
        checkpoint.stat().st_mtime,
    ):
        return f"scored on {previous['checkpoint']}, not {checkpoint}"
    return None


def _write_json(
    logs_dir: Path,
    experiment: str,
    domain: str,
    seed: int,
    run_id: str,
    per_problem: dict,
    threshold: float = 0.5,
    sample_seed: int | None = None,
    expected: list[str] | None = None,
    checkpoint: Path | None = None,
    injective: bool = False,
) -> None:
    out = _result_path(logs_dir, experiment, domain, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A seed is one number per held-out problem: `collect_results` aggregates `ok` only.
    expected = list(per_problem) if expected is None else expected
    scored = sum(
        stem in per_problem and "error" not in per_problem[stem] for stem in expected
    )
    status = (
        "ok"
        if expected and scored == len(expected)
        else "partial" if scored else "validation_failed"
    )
    payload = {
        "experiment": experiment,
        "domain": domain,
        "seed": seed,
        "run_id": run_id,
        "threshold": threshold,
        # Passing this back as validation.yaml's sample_seed reproduces these exact states.
        "sample_seed": sample_seed,
        "status": status,
        "injective": injective,
        "per_problem": per_problem,
    }
    if checkpoint is not None:
        payload["checkpoint"] = str(checkpoint)
        payload["checkpoint_mtime"] = checkpoint.stat().st_mtime
    staged = out.with_name(out.name + ".writing")
    staged.write_text(json.dumps(payload, indent=2, default=str))
    staged.replace(out)
    logger.info("Wrote %s (%s).", out, status)


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
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Processes the MLP-effects validator grounds states across. Default: the CPUs this job was allocated. Ignored by the other validators, which ground nothing in parallel.",
    )
    p.add_argument(
        "--states-per-worker",
        type=int,
        default=None,
        help="States an MLP-effects grounding worker handles before it is replaced, releasing the ground actions pymimir keeps until a process exits.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Keep the problems an earlier run on the same checkpoint scored, and continue the rest with its sample seed from their last scored state.",
    )
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
        num_workers=args.num_workers,
        states_per_worker=args.states_per_worker,
        resume=args.resume,
    )
    failed = all("error" in v for v in per_problem.values()) if per_problem else True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
