"""Validate one ``(experiment, domain, seed)`` run and write its results.

The compute-side validation step, invoked once per seed by ``slurm/validate.sh`` and by ``launch.py --backend local``. The experiment type picks the validator: dias / noise score a DIAS run's learned domain, mlp an MLP-effects run's, and l1 the domain L1 learned, in ``experiments/l1_pddl/``, which has no run behind it.

A run is located by its deterministic id, its problems are the domain folder's ``test-{1,2,3}.pddl`` and its sampling knobs come from ``experiments/configs/validation.yaml``. Results go to ``<logs>/validation/<experiment>/<domain>/seed<seed>.json``, the learned domain they scored beside it as ``seed<seed>.pddl``, and the metrics to the resumed W&B run when it is reachable.

``--threshold`` extracts a run's learned domain anew at another binarization threshold, and ``--results-experiment`` redirects only the output directory, which together re-validate a finished experiment without touching its JSONs; see ``experiments/launch_threshold_validation.py``.

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
    CHECKPOINTS,
    EXPERIMENT_TYPES,
    EXPERIMENT_VALIDATOR,
    VALIDATORS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    add_logging_arguments,
    add_path_arguments,
    allocated_cpus,
    domain_pddl,
    l1_domain,
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
    threshold: float | None = None,
    results_experiment: str | None = None,
    num_workers: int | None = None,
    states_per_worker: int | None = None,
    resume: bool = False,
    checkpoint: str = "final",
) -> dict:
    """Validate a single run over its held-out problems; write and return the results.

    The checkpoint is always found via *experiment*; *results_experiment* only picks the directory the JSON goes to. With *resume*, problems an earlier run on the same checkpoint scored are kept, and the rest use its sample seed.
    """
    # Imported here, not at module scope: submitting jobs must not need torch or pymimir.
    from dias.validation.checkpoint import resolve_checkpoint
    from dias.validation.successors import resolve_sample_seed, sampling_from_record

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
        # There is no training run to mirror the metrics to.
        log_wandb = False
        learned = {
            "original_domain_path": str(domain_pddl(domain)),
            "learned_domain_path": str(l1_domain(domain)),
        }
    if validator.grounds_in_parallel:
        workers = allocated_cpus() if num_workers is None else num_workers
        logger.info("Grounding states across %d worker processes.", workers)
        learned["num_workers"] = workers
        if states_per_worker is not None:
            learned["states_per_worker"] = states_per_worker
    # Written into the result, since a resumed run must draw its remaining problems' states the same way.
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
        # Resolved once, so every problem of a seed shares one draw.
        configured_seed = val_cfg.get("sample_seed")
        sample_seed = resolve_sample_seed(
            None if configured_seed is None else int(configured_seed)
        )

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
            scored_threshold,
            sample_seed,
            expected=[p.stem for p in val_problems],
            checkpoint=checkpoint_path,
            force_injective=force_injective,
            sampling=sampling.record(),
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
    previous: dict,
    path: Path,
    checkpoint: Path | None,
    threshold: float | None,
    force_injective: bool,
    sampling,
) -> str | None:
    """Why the results in `path` cannot be resumed, or None if they can."""
    from dias.validation.successors import UNRECORDED_SAMPLING, sampling_from_record

    if threshold is not None and previous["threshold"] != threshold:
        return f"scored at threshold {previous['threshold']}, not {threshold}"
    if previous.get("force_injective") != force_injective:
        return f"scored with force_injective {previous.get('force_injective')}, not {force_injective}"
    # Results written before the key existed drew their states breadth first.
    previous_sampling = previous.get("sampling", UNRECORDED_SAMPLING.record())
    if sampling_from_record(previous_sampling) != sampling:
        return f"states sampled as {previous_sampling}, not {sampling.record()}"
    if checkpoint is None:
        return None
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
    threshold: float | None = None,
    sample_seed: int | None = None,
    expected: list[str] | None = None,
    checkpoint: Path | None = None,
    force_injective: bool = False,
    sampling: dict | None = None,
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
        default=None,
        help="Extract the learned domain anew at this binarization threshold, instead of scoring the one written beside the checkpoint at the run's schema_threshold (dias and mlp).",
    )
    p.add_argument(
        "--checkpoint",
        default="final",
        choices=CHECKPOINTS,
        help="Which of a run's checkpoints to score: final, the model as training left it, or best, the one with the lowest validation loss. Default: final.",
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
        checkpoint=args.checkpoint,
    )
    failed = all("error" in v for v in per_problem.values()) if per_problem else True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
