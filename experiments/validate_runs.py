"""Validate one ``(experiment, domain, seed)`` run on the domain's test problems. It is the validation step of experiments/launch.py.

The experiment type picks the validator. Results go to ``<logs>/validation/<experiment>/<domain>/seed<seed>.json``, the learned domain they scored beside it as ``seed<seed>.pddl``, and the metrics to the run's W&B run. The JSON records the sample seed the states were drawn with and is written after every problem. Each state's score is recorded as it finishes, and ``--resume`` continues from there. A state that does not fit in the job's memory ends the seed as ``out_of_memory``.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import configure_logging  # noqa: E402
from pipeline import (
    CHECKPOINTS,
    EXPERIMENT_TYPES,
    EXPERIMENT_VALIDATOR,
    VALIDATORS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    add_logging_arguments,
    add_path_arguments,
    domain_pddl,
    l1_domain,
    load_yaml,
    select_domains,
    validation_problems,
    validator_class,
)

logger = logging.getLogger("dias.validate")


def run_id_for(experiment: str, domain: str, seed: int) -> str:
    """The id of a run, as experiments/slurm/train.sh writes it."""
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
    threshold: float | None = None,
    results_experiment: str | None = None,
    num_workers: int | None = None,
    resume: bool = False,
    checkpoint: str = "final",
) -> dict:
    """Validate the run of `experiment` on its domain's test problems and write the results under `results_experiment`. With `resume`, the problems already scored on the same checkpoint are kept, and the others drawn with the same sample seed."""
    from dias.validation.checkpoint import resolve_checkpoint
    from dias.validation.successors import resolve_sample_seed, sampling_from_record
    from dias.validation.workers import StateDoesNotFit, available_cpus

    run_id = run_id or run_id_for(experiment, domain, seed)
    results_experiment = results_experiment or experiment
    val_problems = validation_problems(domain)
    out = _result_path(logs_dir, results_experiment, domain, seed)

    validator = VALIDATORS[EXPERIMENT_VALIDATOR[experiment_type]]
    validation = validator_class(validator.name)
    if validator.from_checkpoint:
        checkpoint_path = resolve_checkpoint(
            logs_dir / "dias" / run_id / "checkpoints", checkpoint
        )
        learned = {
            "model_id": run_id,
            "logs_path": str(logs_dir.resolve()),
            "threshold": threshold,
            "checkpoint": checkpoint,
        }
    else:
        if threshold is not None or checkpoint != "final":
            raise SystemExit(
                f"--threshold and --checkpoint pick a training run's learned domain; {validator.summary}"
            )
        checkpoint_path = None
        log_wandb = False
        learned = {
            "original_domain_path": str(domain_pddl(domain)),
            "learned_domain_path": str(l1_domain(domain)),
        }
    learned["num_workers"] = available_cpus() if num_workers is None else num_workers
    sampling = sampling_from_record(val_cfg["sampling"])
    force_injective = bool(val_cfg.get("force_injective", False))

    per_problem: dict[str, dict[str, Any]] = {}
    scored_threshold = threshold
    if resume and out.exists():
        previous = json.loads(out.read_text())
        refusal = _resume_refusal(
            previous, out, checkpoint_path, threshold, force_injective, sampling
        )
        if refusal is not None:
            raise SystemExit(f"--resume {out}: {refusal}")
        sample_seed = previous["sample_seed"]
        scored_threshold = previous["threshold"]
        per_problem = {
            stem: result
            for stem, result in previous["per_problem"].items()
            if "error" not in result
        }
    else:
        configured_seed = val_cfg.get("sample_seed")
        sample_seed = resolve_sample_seed(
            None if configured_seed is None else int(configured_seed)
        )

    def record() -> None:
        """Write the results so far."""
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
            scored_threshold,
            sample_seed,
            expected=[p.stem for p in val_problems],
            checkpoint=checkpoint_path,
            force_injective=force_injective,
            sampling=sampling.record(),
        )

    record()
    for problem in val_problems:
        stem = problem.stem
        if stem in per_problem:
            logger.info("%s is already scored.", stem)
            continue
        logger.info(
            "Validating %s seed %d on %s, sampling %s.",
            domain,
            seed,
            stem,
            sampling.record(),
        )
        try:
            v = validation(
                problem_path=str(problem.resolve()),
                force_injective=force_injective,
                **learned,
            )
            scored_threshold = v.threshold
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".pddl").write_text(v.learned_domain_text)
            v.setup(sampling, seed=sample_seed)
            result = v.validate(out.with_name(f"seed{seed}.{stem}.states.jsonl"))
        except StateDoesNotFit as e:
            logger.error("%s: %s, so the seed is left out.", stem, e)
            per_problem[stem] = {"error": str(e), "out_of_memory": True}
        except Exception as e:  # noqa: BLE001
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
        record()
        if per_problem[stem].get("out_of_memory"):
            break

    if log_wandb:
        _log_to_wandb(results_experiment, domain, seed, run_id, logs_dir, per_problem)
    return per_problem


def _plain(value: Any) -> Any:
    """`value`, a numpy number as a Python one."""
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
    previous: dict,
    path: Path,
    checkpoint: Path | None,
    threshold: float | None,
    force_injective: bool,
    sampling,
) -> str | None:
    """Why the results in `path` cannot be resumed, or None if they can."""
    from dias.validation.successors import sampling_from_record

    if threshold is not None and previous["threshold"] != threshold:
        return f"scored at threshold {previous['threshold']}, not {threshold}"
    if previous.get("force_injective") != force_injective:
        return f"scored with force_injective {previous.get('force_injective')}, not {force_injective}"
    previous_sampling = previous["sampling"]
    if sampling_from_record(previous_sampling) != sampling:
        return f"states sampled as {previous_sampling}, not {sampling.record()}"
    if checkpoint is None:
        return None
    if (previous["checkpoint"], previous["checkpoint_mtime"]) != (
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
    threshold: float | None = None,
    sample_seed: int | None = None,
    expected: list[str] | None = None,
    checkpoint: Path | None = None,
    force_injective: bool = False,
    sampling: dict | None = None,
) -> None:
    out = _result_path(logs_dir, experiment, domain, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    expected = list(per_problem) if expected is None else expected
    scored = sum(
        stem in per_problem and "error" not in per_problem[stem] for stem in expected
    )
    failed = any("error" in result for result in per_problem.values())
    if any(result.get("out_of_memory") for result in per_problem.values()):
        status = "out_of_memory"
    elif expected and scored == len(expected):
        status = "ok"
    else:
        status = "partial" if scored or not failed else "validation_failed"
    payload = {
        "experiment": experiment,
        "domain": domain,
        "seed": seed,
        "run_id": run_id,
        "threshold": threshold,
        "sample_seed": sample_seed,
        "status": status,
        "force_injective": force_injective,
        "sampling": sampling,
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
    """Log the numeric results to the run's W&B run, if it can be reached."""
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
        wandb.config.update({"domain": domain, "seed": seed}, allow_val_change=True)
        for stem, result in per_problem.items():
            for k, val in result.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    wandb.log({f"validation/{stem}/{k}": float(val)})
        wandb.finish()
    except Exception as e:  # noqa: BLE001
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
        default=None,
        help="Derive the learned domain at this threshold instead of scoring the one beside the checkpoint (dias and mlp).",
    )
    p.add_argument(
        "--checkpoint",
        default="final",
        choices=CHECKPOINTS,
        help="The checkpoint to score: final (at the end of training, default) or best (at the lowest validation loss).",
    )
    p.add_argument(
        "--results-experiment",
        default=None,
        help="Write the results under logs/validation/<this>/ instead of <experiment>/.",
    )
    add_path_arguments(p)
    add_logging_arguments(p)
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes. Default: one per CPU.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the problems and states already scored.",
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
        resume=args.resume,
        checkpoint=args.checkpoint,
    )
    failed = [
        v for v in per_problem.values() if "error" in v and not v.get("out_of_memory")
    ]
    return 1 if not per_problem or len(failed) == len(per_problem) else 0


if __name__ == "__main__":
    import safe_slurm  # noqa: F401

    sys.exit(main())
