"""Launch DIAS paper experiments: one training array + a dependent validation array per domain.

For each selected domain this submits (Slurm backend):

  1. a training array job of ``--num-runs`` tasks (task i = seed i), and
  2. a validation array job of the same size with
     ``--dependency=aftercorr:<train_jobid>`` so validation task i runs only
     after training task i succeeds.

Runs are identified by a deterministic id ``<experiment>-<domain>-s<seed>`` (see
``validate_runs.run_id_for``), so the two array jobs need no id handoff: the
checkpoint of task i lands at ``logs/dias/<run_id>/checkpoints/best-*.ckpt`` and
validation recomputes the same id.

Config composition (in order): ``configs/main.yaml`` -> ``<domain>/config.yaml``
-> optional experiment overlay (``configs/experiments/{mlp,noise}.yaml``) ->
optional ablation overlay (``configs/experiments/ablation-*.yaml``, selected by
``--ablation``) -> launcher ``--set`` overrides. The experiment type also selects
the validator (``mlp`` uses the MLP baseline; ``dias``/``noise`` use the DIAS
validator); ablations always use the DIAS validator.

Per-domain Slurm resources come from ``configs/slurm-resources.yaml``.

Examples
--------
    # Full grid on Slurm (all domains, 10 seeds each), grouped as "paper-main":
    python scripts/launch.py --experiment paper-main

    # MLP baseline on two domains:
    python scripts/launch.py --experiment-type mlp --domains gripper satellite \
        --experiment mlp-main

    # Noise sweep point (expected 2 added / 2 removed atoms):
    python scripts/launch.py --experiment-type noise --experiment noise-a2r2 \
        --set data.init_args.expected_added_atoms=2 \
        --set data.init_args.expected_removed_atoms=2

    # Ablation (no PCGrad) on the default DIAS model:
    python scripts/launch.py --ablation no-pcgrad --experiment abl-no-pcgrad

    # See exactly what would be submitted, without submitting:
    python scripts/launch.py --domains gripper --num-runs 3 --dry-run

    # Run locally (no Slurm), sequentially — good for smoke tests:
    python scripts/launch.py --backend local --domains gripper --num-runs 2 \
        --experiment smoke --no-wandb
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "configs" / "experiments"
EXPERIMENT_OVERLAYS = {
    "mlp": EXPERIMENTS_DIR / "mlp.yaml",
    "noise": EXPERIMENTS_DIR / "noise.yaml",
}
# Which validator each experiment type uses (noise still uses the DIAS validator).
VALIDATOR_TYPE = {"dias": "dias", "noise": "dias", "mlp": "mlp"}
# Model ablation overlays, orthogonal to --experiment-type (all use the DIAS
# validator). Composed after the experiment overlay.
ABLATION_OVERLAYS = {
    "no-slot-weights": EXPERIMENTS_DIR / "ablation-no-slot-weights.yaml",
    "no-precondition-loss": EXPERIMENTS_DIR / "ablation-no-precondition-loss.yaml",
    "no-effect-loss": EXPERIMENTS_DIR / "ablation-no-effect-loss.yaml",
    "no-aux-loss": EXPERIMENTS_DIR / "ablation-no-aux-loss.yaml",
    "no-pcgrad": EXPERIMENTS_DIR / "ablation-no-pcgrad.yaml",
    "no-random-embeddings": EXPERIMENTS_DIR / "ablation-no-random-embeddings.yaml",
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _resolve_resources(slurm_cfg: dict, domain: str) -> dict:
    merged = dict(slurm_cfg.get("defaults", {}))
    merged.update(slurm_cfg.get(domain) or {})
    return merged


def _sbatch_resource_opts(res: dict) -> list[str]:
    opts: list[str] = []
    if "time" in res:
        opts.append(f"--time={res['time']}")
    if "mem" in res:
        opts.append(f"--mem={res['mem']}")
    if "cpus" in res:
        opts.append(f"--cpus-per-task={res['cpus']}")
    if "gres" in res:
        opts.append(f"--gres={res['gres']}")
    opts += list(res.get("sbatch_extra", []))
    return opts


def _base_train_args(
    experiment: str,
    domain: str,
    spec: dict,
    train_config: Path,
    domain_config: Path,
    overlays: list[Path],
    logs_dir: Path,
    extra: list[str],
) -> list[str]:
    """Seed-independent Lightning args shared by all runs of a domain."""
    args = ["--config", str(train_config), "--config", str(domain_config)]
    for overlay in overlays:
        args += ["--config", str(overlay)]
    args += [
        "--experiment",
        experiment,
        "--data.init_args.domain_path",
        str(PROJECT_ROOT / spec["domain"]),
        "--data.init_args.problem_path",
        str(PROJECT_ROOT / spec["problems"][0]),
        "--trainer.logger.init_args.group",
        experiment,
        "--trainer.logger.init_args.save_dir",
        str(logs_dir),
    ]
    return args + extra


def _run(cmd: list[str], dry_run: bool) -> str | None:
    print("  " + " ".join(cmd), flush=True)
    if dry_run:
        return None
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def _submit_domain_slurm(
    experiment: str,
    domain: str,
    spec: dict,
    base_args: list[str],
    val_type: str,
    num_runs: int,
    train_res: dict,
    val_res: dict,
    logs_dir: Path,
    args: argparse.Namespace,
) -> None:
    array = f"0-{num_runs - 1}"
    train_res_opts = _sbatch_resource_opts(train_res)
    val_res_opts = _sbatch_resource_opts(val_res)

    # EXPERIMENT/DOMAIN go through --export (single tokens, needed to build the
    # run id); the Lightning config chain is passed as positional args so sbatch
    # preserves argv exactly (no fragile string splitting).
    train_export = ",".join(["ALL", f"EXPERIMENT={experiment}", f"DOMAIN={domain}"])
    train_cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=dias-train-{domain}",
        f"--array={array}",
        *train_res_opts,
        f"--export={train_export}",
        "scripts/slurm/train.sh",
        *base_args,
    ]
    print(f"[{domain}] submit training array ({num_runs} runs)")
    train_jobid = _run(train_cmd, args.dry_run)

    val_export = ",".join(
        [
            "ALL",
            f"EXPERIMENT={experiment}",
            f"DOMAIN={domain}",
            f"EXP_TYPE={val_type}",
            f"LOGS_DIR={logs_dir}",
            f"DOMAIN_SETTINGS={args.domain_settings}",
            f"VALIDATION_CONFIG={args.validation_config}",
        ]
    )
    dep = train_jobid or "<train_jobid>"
    val_cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=dias-val-{domain}",
        f"--array={array}",
        f"--dependency=aftercorr:{dep}",
        *val_res_opts,
        f"--export={val_export}",
        "scripts/slurm/validate.sh",
    ]
    print(f"[{domain}] submit validation array (aftercorr:{dep})")
    _run(val_cmd, args.dry_run)


def _run_domain_local(
    experiment: str,
    domain: str,
    spec: dict,
    base_args: list[str],
    val_type: str,
    num_runs: int,
    logs_dir: Path,
    args: argparse.Namespace,
) -> None:
    # Imported here so Slurm submission never needs torch/pymimir installed.
    from validate_runs import run_id_for, validate_run

    domain_settings = _load_yaml(args.domain_settings)
    val_cfg = _load_yaml(args.validation_config)

    for seed in range(num_runs):
        run_id = run_id_for(experiment, domain, seed)
        train_cmd = [
            "dias-train",
            "fit",
            *base_args,
            "--seed_everything",
            str(seed),
            "--data.init_args.seed",
            str(seed),
            "--trainer.logger.init_args.id",
            run_id,
            "--trainer.logger.init_args.name",
            run_id,
        ]
        print(f"\n[{domain} seed={seed}] train  (run_id={run_id})", flush=True)
        print("  " + " ".join(train_cmd), flush=True)
        if args.dry_run:
            continue
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[skip] {domain} seed={seed}: training failed ({e})", flush=True)
            continue
        validate_run(
            experiment=experiment,
            domain=domain,
            seed=seed,
            experiment_type=val_type,
            domain_settings=domain_settings,
            val_cfg=val_cfg,
            logs_dir=logs_dir,
            run_id=run_id,
            log_wandb=not args.no_wandb,
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        default=f"run-{datetime.now():%Y%m%d-%H%M%S}",
        help="Experiment name = W&B group = results dir. Defaults to a timestamp.",
    )
    p.add_argument(
        "--experiment-type", default="dias", choices=["dias", "mlp", "noise"]
    )
    p.add_argument(
        "--ablation",
        default=None,
        choices=sorted(ABLATION_OVERLAYS),
        help="Model ablation overlay, composed on top of the experiment type "
        "(uses the DIAS validator). Default: none (full model).",
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domain keys to run (default: all in the manifest).",
    )
    p.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Runs (seeds 0..N-1) per domain. Default: len(validation.yaml seeds).",
    )
    p.add_argument("--backend", default="slurm", choices=["slurm", "local"])
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="Extra Lightning override, repeatable (no commas in value).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without submitting/running.",
    )
    p.add_argument(
        "--no-wandb", action="store_true", help="Local backend: skip W&B logging."
    )
    p.add_argument(
        "--domain-settings",
        type=Path,
        default=PROJECT_ROOT / "pddl" / "domain-settings.yaml",
    )
    p.add_argument(
        "--train-config", type=Path, default=PROJECT_ROOT / "configs" / "main.yaml"
    )
    p.add_argument(
        "--validation-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "validation.yaml",
    )
    p.add_argument(
        "--slurm-resources",
        type=Path,
        default=PROJECT_ROOT / "configs" / "slurm-resources.yaml",
    )
    p.add_argument(
        "--slurm-resources-validation",
        type=Path,
        default=PROJECT_ROOT / "configs" / "slurm-resources-validation.yaml",
    )
    p.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs")
    args = p.parse_args()

    domain_settings = _load_yaml(args.domain_settings)
    domains = args.domains or list(domain_settings.keys())
    for d in domains:
        if d not in domain_settings:
            raise SystemExit(f"domain '{d}' not in {args.domain_settings}")

    num_runs = args.num_runs or len(_load_yaml(args.validation_config)["seeds"])
    overlays = []
    exp_overlay = EXPERIMENT_OVERLAYS.get(args.experiment_type)
    if exp_overlay is not None:
        overlays.append(exp_overlay)
    if args.ablation is not None:
        overlays.append(ABLATION_OVERLAYS[args.ablation])
    val_type = VALIDATOR_TYPE[args.experiment_type]
    extra = []
    for o in args.overrides:
        if "=" not in o:
            raise SystemExit(f"--set expects key=value, got {o!r}")
        if "," in o:
            raise SystemExit(f"--set value must not contain commas: {o!r}")
        extra.append(f"--{o}")

    print(
        f"experiment : {args.experiment}  (type={args.experiment_type}, "
        f"ablation={args.ablation or 'none'}, validator={val_type})"
    )
    print(f"domains    : {len(domains)} x {num_runs} runs  (backend={args.backend})")
    for overlay in overlays:
        print(f"overlay    : {overlay}")

    if args.backend == "slurm" and not args.dry_run:
        (PROJECT_ROOT / "logs" / "slurm").mkdir(parents=True, exist_ok=True)

    slurm_cfg = _load_yaml(args.slurm_resources) if args.backend == "slurm" else {}
    val_slurm_cfg = (
        _load_yaml(args.slurm_resources_validation) if args.backend == "slurm" else {}
    )

    for domain in domains:
        spec = domain_settings[domain]
        domain_config = (PROJECT_ROOT / spec["domain"]).parent / "config.yaml"
        base_args = _base_train_args(
            args.experiment,
            domain,
            spec,
            args.train_config,
            domain_config,
            overlays,
            args.log_dir,
            extra,
        )
        if args.backend == "slurm":
            _submit_domain_slurm(
                args.experiment,
                domain,
                spec,
                base_args,
                val_type,
                num_runs,
                _resolve_resources(slurm_cfg, domain),
                _resolve_resources(val_slurm_cfg, domain),
                args.log_dir,
                args,
            )
        else:
            _run_domain_local(
                args.experiment,
                domain,
                spec,
                base_args,
                val_type,
                num_runs,
                args.log_dir,
                args,
            )

    print("\nDone submitting." if args.backend == "slurm" else "\nDone.")
    return 0


if __name__ == "__main__":
    sys.path.insert(
        0, str(Path(__file__).resolve().parent)
    )  # for `import validate_runs`
    sys.exit(main())
