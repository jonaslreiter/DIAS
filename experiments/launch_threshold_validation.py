"""Re-validate the checkpoints of a finished experiment at several thresholds, measuring how much P / R / N_S / N_C / N_SC depend on that choice.

Per (domain, threshold) this submits one array over the seeds through the normal ``experiments/slurm/validate.sh``, no training, over runs that trained to ``max_steps`` (``launch.finished_training``). Checkpoints come from the source experiment's run ids; results go to a sibling ``logs/validation/<experiment>-thr<t>/``, so the originals are untouched. W&B is always off here: ``validate_runs`` mirrors metrics by resuming the training run, and a second threshold would overwrite the first's summary.

Aggregate with::

    python experiments/collect_results.py --experiment dias-names --threshold-sweep

Examples
--------
    # The default grid over all domains and seeds 0 to 4, then one domain at two thresholds, printed only.
    python experiments/launch_threshold_validation.py --experiment dias-names --seeds 0-4
    python experiments/launch_threshold_validation.py --experiment dias-names --domains gripper --thresholds 0.1 0.9 --seeds 0-4 --dry-run

    # Continue validations that ran out of time.
    python experiments/launch_threshold_validation.py --experiment dias-names --domains driverlog --seeds 3 --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import pipeline`

from launch import unfinished_runs  # noqa: E402
from pipeline import configure_logging  # noqa: E402
from pipeline import (
    CHECKPOINTS,
    EXPERIMENT_VALIDATOR,
    PROJECT_ROOT,
    SLURM_DIR,
    VALIDATORS,
    add_domain_arguments,
    add_logging_arguments,
    add_path_arguments,
    format_array,
    load_yaml,
    prepare_slurm_logs,
    resolve_resources,
    resolve_seeds,
    run_command,
    sbatch_output_opt,
    sbatch_resource_opts,
    select_domains,
    validation_resources_file,
)

# Grid of the bare --thresholds flag. 0.5 is re-run rather than reused, so every column comes from one code path.
THRESHOLD_VALUES = (0.1, 0.25, 0.5, 0.75, 0.9)
THRESHOLD_TAG = "thr"


def threshold_experiment_name(experiment: str, threshold: float) -> str:
    """Results directory / group for one threshold; ``collect_results.py --threshold-sweep`` imports it to rebuild these."""
    return f"{experiment}-{THRESHOLD_TAG}{threshold:g}"


def _submit_slurm(
    experiment: str,
    results_experiment: str,
    domain: str,
    threshold: float,
    seeds: list[int],
    val_res: dict,
    logs_dir: Path,
    args: argparse.Namespace,
) -> None:
    array = format_array(seeds)
    export = ",".join(
        [
            "ALL",
            f"DIAS_ROOT={PROJECT_ROOT}",
            f"EXPERIMENT={experiment}",
            f"RESULTS_EXPERIMENT={results_experiment}",
            f"THRESHOLD={threshold:g}",
            f"CHECKPOINT={args.checkpoint}",
            "NO_WANDB=1",
            f"DOMAIN={domain}",
            f"EXP_TYPE={args.experiment_type}",
            f"LOGS_DIR={logs_dir}",
            f"VALIDATION_CONFIG={args.validation_config}",
            *(
                [f"NUM_WORKERS={args.num_workers}"]
                if args.num_workers is not None
                else []
            ),
            *(["RESUME=1"] if args.resume else []),
        ]
    )
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=dias-val-{THRESHOLD_TAG}{threshold:g}-{domain}",
        f"--array={array}",
        sbatch_output_opt(logs_dir),
        *sbatch_resource_opts(val_res),
        f"--export={export}",
        str(SLURM_DIR / "validate.sh"),
    ]
    print(f"[{domain} @ {threshold:g}] submit validation array ({len(seeds)} seeds)")
    run_command(cmd, args.dry_run)


def _run_local(
    experiment: str,
    results_experiment: str,
    domain: str,
    threshold: float,
    seeds: list[int],
    logs_dir: Path,
    args: argparse.Namespace,
) -> None:
    # Imported here so Slurm submission never needs torch/pymimir.
    from validate_runs import validate_run

    val_cfg = load_yaml(args.validation_config)
    for seed in seeds:
        print(f"\n[{domain} seed={seed} @ {threshold:g}] validate", flush=True)
        if args.dry_run:
            continue
        validate_run(
            experiment=experiment,
            domain=domain,
            seed=seed,
            experiment_type=args.experiment_type,
            val_cfg=val_cfg,
            logs_dir=logs_dir,
            log_wandb=False,
            threshold=threshold,
            checkpoint=args.checkpoint,
            num_workers=args.num_workers,
            resume=args.resume,
            results_experiment=results_experiment,
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        required=True,
        help="Existing experiment whose checkpoints are re-validated (e.g. dias-names).",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        metavar="T",
        help="Thresholds to validate at. Default: "
        + ", ".join(f"{t:g}" for t in THRESHOLD_VALUES)
        + ".",
    )
    p.add_argument(
        "--experiment-type",
        default="dias",
        choices=[
            kind
            for kind, validator in EXPERIMENT_VALIDATOR.items()
            if VALIDATORS[validator].from_checkpoint
        ],
        help="The source experiment's type, which picks the validator.",
    )
    add_domain_arguments(p)
    p.add_argument(
        "--checkpoint",
        default="final",
        choices=CHECKPOINTS,
        help="Which of each run's checkpoints to re-validate: final, the model as training left it, or best, the one with the lowest validation loss. Default: final.",
    )
    p.add_argument("--backend", default="slurm", choices=["slurm", "local"])
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes each validation task scores states in. Default: the CPUs it is allocated.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Validation keeps the problems an earlier run on the same checkpoint scored, and continues the rest with its sample seed from their last scored state.",
    )
    # Applied to every array, over slurm-resources-validation.yaml: the sweep is several times the usual volume, so smaller CPU-friendly asks are usually worth it.
    p.add_argument(
        "--time", default=None, help="Override --time for every array (HH:MM:SS)."
    )
    p.add_argument("--mem", default=None, help="Override --mem for every array.")
    p.add_argument("--cpus", default=None, help="Override --cpus-per-task.")
    p.add_argument(
        "--partition",
        default=None,
        help="Override the partition list. Validation is CPU-only and never requests a --gres; pinning CPU partitions keeps the arrays off the GPU nodes.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print commands without submitting."
    )
    p.add_argument(
        "--allow-unfinished",
        action="store_true",
        help="Submit even for runs that stopped short of max_steps.",
    )
    add_path_arguments(p)
    add_logging_arguments(p)
    p.add_argument(
        "--slurm-resources-validation",
        type=Path,
        default=None,
        help="Default: experiments/configs/slurm-resources-validation-mlp.yaml for the MLP-effects validator, slurm-resources-validation.yaml for the others.",
    )
    args = p.parse_args()
    configure_logging(args)

    domains = select_domains(args.domains)
    seeds = resolve_seeds(args)

    thresholds = list(args.thresholds or THRESHOLD_VALUES)
    for t in thresholds:
        if not 0.0 < t < 1.0:
            raise SystemExit(
                f"--thresholds: expected a probability in (0,1), got {t:g}"
            )

    if not args.allow_unfinished:
        unfinished = unfinished_runs([args.experiment], domains, seeds, args.log_dir)
        if unfinished:
            listed = ", ".join(unfinished[:5])
            more = f" (and {len(unfinished) - 5} more)" if len(unfinished) > 5 else ""
            raise SystemExit(
                f"{len(unfinished)} of {len(domains) * len(seeds)} runs did not train to max_steps: {listed}{more}. Train them first, or pass --allow-unfinished to validate them as they are."
            )

    print(f"source     : {args.experiment}  (validator={args.experiment_type})")
    print(
        f"domains    : {len(domains)} x {len(seeds)} seeds x {len(thresholds)} thresholds = {len(domains) * len(seeds) * len(thresholds)} validations (seeds={format_array(seeds)}, backend={args.backend})"
    )
    print(f"thresholds : {', '.join(f'{t:g}' for t in thresholds)}")
    print(
        "results    : "
        + ", ".join(
            f"{args.log_dir / 'validation' / threshold_experiment_name(args.experiment, t)}/"
            for t in thresholds[:2]
        )
        + (" ..." if len(thresholds) > 2 else "")
    )

    if args.backend == "slurm" and not args.dry_run:
        prepare_slurm_logs(args.log_dir)
    val_slurm_cfg = (
        load_yaml(
            args.slurm_resources_validation
            or validation_resources_file(EXPERIMENT_VALIDATOR[args.experiment_type])
        )
        if args.backend == "slurm"
        else {}
    )

    for threshold in thresholds:
        results_experiment = threshold_experiment_name(args.experiment, threshold)
        print(f"\n=== threshold {threshold:g} -> {results_experiment} ===")
        for domain in domains:
            if args.backend == "slurm":
                res = resolve_resources(val_slurm_cfg, domain)
                if args.time:
                    res["time"] = args.time
                if args.mem:
                    res["mem"] = args.mem
                if args.cpus:
                    res["cpus"] = args.cpus
                if args.partition:
                    res["sbatch_extra"] = [
                        *res.get("sbatch_extra", []),
                        f"--partition={args.partition}",
                    ]
                _submit_slurm(
                    args.experiment,
                    results_experiment,
                    domain,
                    threshold,
                    seeds,
                    res,
                    args.log_dir,
                    args,
                )
            else:
                _run_local(
                    args.experiment,
                    results_experiment,
                    domain,
                    threshold,
                    seeds,
                    args.log_dir,
                    args,
                )

    print("\nDone submitting." if args.backend == "slurm" else "\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
