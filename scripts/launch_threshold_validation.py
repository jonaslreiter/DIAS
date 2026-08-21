"""Re-validate the checkpoints of a finished experiment at several thresholds.

Validation reads a learned schema off the trained model by binarizing the slot
activations and the precondition/effect probabilities; the number reported in
the paper uses a threshold of 0.5 for all of them. This script re-runs *only*
the validation step of an existing experiment (no training) once per threshold,
so the sensitivity of P / R / N_S / N_C / N_SC to that choice can be measured on
exactly the runs the paper reports.

Per (domain, threshold) it submits one Slurm array over the seeds — the same
``scripts/slurm/validate.sh`` the normal pipeline uses, with ``THRESHOLD`` and
``RESULTS_EXPERIMENT`` exported. Checkpoints are located via the *source*
experiment's run id ``<experiment>-<domain>-s<seed>``, while results are written
to a sibling directory ``logs/validation/<experiment>-thr<t>/<domain>/``, so the
original ``logs/validation/<experiment>/`` JSONs are never touched.

W&B logging is always disabled here: ``validate_runs`` mirrors metrics by
*resuming* the training run, so logging a second threshold would overwrite the
0.5 numbers in that run's summary. The JSONs are the source of truth.

Validation is CPU-only (the DIAS validator runs the model on CPU and spends its
time in pymimir successor generation), so resources come from
``configs/slurm-resources-validation.yaml``, which requests no GPU.

Aggregate the result with::

    python scripts/collect_results.py --experiment dias-full --threshold-sweep

Examples
--------
    # Default grid (0.1, 0.25, 0.5, 0.75, 0.9) over all domains x 3 seeds:
    python scripts/launch_threshold_validation.py --experiment dias-full --num-runs 3

    # One domain, two thresholds, printed but not submitted:
    python scripts/launch_threshold_validation.py --experiment dias-full \
        --domains gripper --thresholds 0.1 0.9 --num-runs 3 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import pipeline`

from pipeline import (  # noqa: E402
    PROJECT_ROOT,
    SLURM_RESOURCES_VALIDATION,
    add_domain_arguments,
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
)

# Grid of the bare --thresholds flag; 0.5 is the default the paper reports and
# is re-run here too, so every column of the sweep comes from the same code path.
THRESHOLD_VALUES = (0.1, 0.25, 0.5, 0.75, 0.9)
THRESHOLD_TAG = "thr"


def threshold_experiment_name(experiment: str, threshold: float) -> str:
    """Results directory / group for one threshold of the sweep.

    Also used by ``scripts/collect_results.py --threshold-sweep`` to rebuild the
    per-threshold names, so keep the two in sync by importing this function
    rather than re-deriving it.
    """
    return f"{experiment}-{THRESHOLD_TAG}{threshold:g}"


def _checkpoint_dir(logs_dir: Path, run_id: str) -> Path:
    return logs_dir / "dias" / run_id / "checkpoints"


def _missing_checkpoints(
    experiment: str, domains: list[str], seeds: list[int], logs_dir: Path
) -> list[str]:
    """Run ids of the (domain, seed) pairs that have no trained checkpoint."""
    from validate_runs import run_id_for

    missing = []
    for domain in domains:
        for seed in seeds:
            run_id = run_id_for(experiment, domain, seed)
            if not list(_checkpoint_dir(logs_dir, run_id).glob("best-epoch*.ckpt")):
                missing.append(run_id)
    return missing


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
    # Array task id == seed, exactly as in the normal validation array, so a
    # seed subset becomes a sparse array spec.
    array = format_array(seeds)
    export = ",".join(
        [
            "ALL",
            f"DIAS_ROOT={PROJECT_ROOT}",
            f"EXPERIMENT={experiment}",
            f"RESULTS_EXPERIMENT={results_experiment}",
            f"THRESHOLD={threshold:g}",
            "NO_WANDB=1",
            f"DOMAIN={domain}",
            f"EXP_TYPE={args.experiment_type}",
            f"LOGS_DIR={logs_dir}",
            f"DOMAIN_SETTINGS={args.domain_settings}",
            f"VALIDATION_CONFIG={args.validation_config}",
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
        str(PROJECT_ROOT / "scripts" / "slurm" / "validate.sh"),
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
    # Imported here so Slurm submission never needs torch/pymimir installed.
    from validate_runs import validate_run

    domain_settings = load_yaml(args.domain_settings)
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
            domain_settings=domain_settings,
            val_cfg=val_cfg,
            logs_dir=logs_dir,
            log_wandb=False,
            threshold=threshold,
            results_experiment=results_experiment,
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        required=True,
        help="Existing experiment whose checkpoints are re-validated (e.g. dias-full).",
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
        choices=["dias", "noise"],
        help="Validator to use. The MLP baseline has no schema to binarize.",
    )
    add_domain_arguments(p)
    p.add_argument("--backend", default="slurm", choices=["slurm", "local"])
    # Overrides for configs/slurm-resources-validation.yaml, applied to every
    # array: the sweep is 5x the usual validation volume, so it is usually worth
    # trading the per-domain defaults for smaller, CPU-partition-friendly asks.
    p.add_argument(
        "--time", default=None, help="Override --time for every array (HH:MM:SS)."
    )
    p.add_argument("--mem", default=None, help="Override --mem for every array.")
    p.add_argument("--cpus", default=None, help="Override --cpus-per-task.")
    p.add_argument(
        "--partition",
        default=None,
        help="Override the partition list. Validation is CPU-only (the DIAS "
        "validator runs the model on CPU and spends its time in pymimir), so "
        "no --gres is ever requested; pinning CPU partitions here just keeps "
        "the arrays off the GPU nodes.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print commands without submitting."
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Submit even when some (domain, seed) has no checkpoint yet.",
    )
    add_path_arguments(p)
    p.add_argument(
        "--slurm-resources-validation",
        type=Path,
        default=SLURM_RESOURCES_VALIDATION,
    )
    args = p.parse_args()

    domain_settings = load_yaml(args.domain_settings)
    domains = select_domains(domain_settings, args.domains)
    seeds = resolve_seeds(args)

    thresholds = list(args.thresholds or THRESHOLD_VALUES)
    for t in thresholds:
        if not 0.0 < t < 1.0:
            raise SystemExit(
                f"--thresholds: expected a probability in (0,1), got {t:g}"
            )

    missing = _missing_checkpoints(args.experiment, domains, seeds, args.log_dir)
    if missing:
        head = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        if not args.allow_missing:
            raise SystemExit(
                f"{len(missing)} of {len(domains) * len(seeds)} runs have no "
                f"best-epoch*.ckpt under {args.log_dir}/dias/: {head}\n"
                "Re-validation needs trained checkpoints; pass --allow-missing "
                "to submit anyway (those tasks will fail)."
            )
        print(f"[warn] {len(missing)} runs have no checkpoint: {head}")

    print(f"source     : {args.experiment}  (validator={args.experiment_type})")
    print(
        f"domains    : {len(domains)} x {len(seeds)} seeds x "
        f"{len(thresholds)} thresholds = "
        f"{len(domains) * len(seeds) * len(thresholds)} validations "
        f"(seeds={format_array(seeds)}, backend={args.backend})"
    )
    print(f"thresholds : {', '.join(f'{t:g}' for t in thresholds)}")
    print(
        "results    : "
        + ", ".join(
            f"logs/validation/{threshold_experiment_name(args.experiment, t)}/"
            for t in thresholds[:2]
        )
        + (" ..." if len(thresholds) > 2 else "")
    )

    if args.backend == "slurm" and not args.dry_run:
        prepare_slurm_logs(args.log_dir)
    val_slurm_cfg = (
        load_yaml(args.slurm_resources_validation) if args.backend == "slurm" else {}
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
