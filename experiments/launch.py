"""Launch DIAS experiments: per domain, a training array plus a validation array that depends on it.

Array task id = seed, so ``--dependency=aftercorr`` pairs training task i with validation task i, and ``--seeds`` becomes a sparse spec such as ``--array=3,7-8``. Runs are identified by a deterministic ``<experiment>-<domain>-s<seed>`` (``validate_runs.run_id_for``), so the two arrays need no id handoff.

Config chain: ``configs/main.yaml`` -> ``pddl/<domain>/config.yaml`` -> ``--experiment-type`` overlay -> ``--labels`` overlay -> ``--ablation`` overlay -> ``--set`` overrides. The experiment type also picks the validator. ``--sweep AXIS`` submits the whole grid once per value of that axis, each into its own sub-experiment. ``--validate-only`` submits the validation array alone, over runs that already trained to ``max_steps``; ``--experiment-type l1`` does nothing else, since L1's learned domains have no training run.

Examples
--------
    # All domains, 10 seeds each: action names, then full STRIPS actions.
    python experiments/launch.py --experiment dias-names
    python experiments/launch.py --experiment dias-strips --labels strips

    # 8 flips per transition.
    python experiments/launch.py --experiment noise-f8 --experiment-type noise --set data.init_args.noise.added_atoms=2 --set data.init_args.noise.removed_atoms=2

    # An ablation, and a sweep into the experiments alpha-aux0.01 .. alpha-aux1.
    python experiments/launch.py --experiment ablation-no-pcgrad --ablation no-pcgrad
    python experiments/launch.py --experiment alpha --sweep aux-loss-alpha --seeds 0-4

    # The L1 baseline.
    python experiments/launch.py --experiment l1 --experiment-type l1

    # Re-run seeds into the same experiment, or only their validation.
    python experiments/launch.py --experiment dias-names --domains gripper --seeds 3 7-8
    python experiments/launch.py --experiment dias-names --domains visitall --validate-only

    # Print without submitting, or run here instead of on Slurm.
    python experiments/launch.py --experiment dias-names --domains gripper --dry-run
    python experiments/launch.py --experiment smoke --backend local --domains gripper --num-runs 1 --no-wandb
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import pipeline`

from pipeline import configure_logging  # noqa: E402
from pipeline import (
    CHECKPOINTS,
    CONFIGS_DIR,
    EXPERIMENT_TYPES,
    EXPERIMENT_VALIDATOR,
    PROJECT_ROOT,
    SLURM_DIR,
    SLURM_RESOURCES,
    VALIDATORS,
    add_domain_arguments,
    add_logging_arguments,
    add_path_arguments,
    domain_config,
    domain_pddl,
    format_array,
    l1_domains,
    load_yaml,
    prepare_slurm_logs,
    resolve_resources,
    resolve_seeds,
    run_command,
    sbatch_output_opt,
    sbatch_resource_opts,
    select_domains,
    train_problem,
    validation_resources_file,
)

EXPERIMENT_OVERLAYS = {
    "mlp": CONFIGS_DIR / "experiment-mlp.yaml",
    "noise": CONFIGS_DIR / "experiment-noise.yaml",
}
# What a transition is labelled with; action names need no overlay.
LABEL_OVERLAYS = {
    "strips": CONFIGS_DIR / "labels-strips.yaml",
    "strips-plus": CONFIGS_DIR / "labels-strips-plus.yaml",
}
LABELS = ("names", *LABEL_OVERLAYS)
# Orthogonal to --experiment-type, composed after it; all use the DIAS validator.
ABLATION_OVERLAYS = {
    "no-slot-weights": CONFIGS_DIR / "ablation-no-slot-weights.yaml",
    "no-precondition-loss": CONFIGS_DIR / "ablation-no-precondition-loss.yaml",
    "no-effect-loss": CONFIGS_DIR / "ablation-no-effect-loss.yaml",
    "no-aux-loss": CONFIGS_DIR / "ablation-no-aux-loss.yaml",
    "no-pcgrad": CONFIGS_DIR / "ablation-no-pcgrad.yaml",
    "no-random-embeddings": CONFIGS_DIR / "ablation-no-random-embeddings.yaml",
}
# The tau schedule is a typed dataclass, so its fields are ordinary dotted keys.
SCHEDULE_KEY = "model.init_args.precondition_tau"

TRAIN_SCRIPT = SLURM_DIR / "train.sh"
VALIDATE_SCRIPT = SLURM_DIR / "validate.sh"


@dataclass(frozen=True)
class SweepAxis:
    """One hyperparameter axis that ``--sweep`` varies over the full grid.

    ``values`` is the bare flag's default grid and ``tag`` the sub-experiment infix. A value sets ``param`` to ``value * scale``, and ``linked`` gives further overrides that move with it. All are appended last, so they beat the domain config and the overlays, and all are rejected as ``--set`` targets.
    """

    param: str
    tag: str
    values: tuple[float, ...]
    scale: float = 1.0
    linked: Callable[[float], dict[str, float]] | None = None


def _dataset_size_overrides(value: float) -> dict[str, int]:
    """Sampling knobs that move with ``min_per_action_train`` on the size axis.

    Pinning the caps to the same N makes the dataset size exact rather than a lower bound, and pinning the ``*_floor`` knobs keeps any floor a run sets from raising it.
    """
    n = int(value)
    half = max(1, n // 2)
    return {
        "data.init_args.max_per_action_train": n,
        "data.init_args.min_per_action_valid": half,
        "data.init_args.max_per_action_valid": half,
        "data.init_args.min_per_action_train_floor": n,
        "data.init_args.min_per_action_valid_floor": half,
    }


SWEEP_AXES: dict[str, SweepAxis] = {
    # Grid covers every value currently tuned in pddl/<domain>/config.yaml.
    "aux-loss-alpha": SweepAxis(
        param="model.init_args.aux_loss_alpha",
        tag="aux",
        values=(0.01, 0.03, 0.1, 0.3, 1.0),
    ),
    "num-slots": SweepAxis(
        param="model.init_args.num_slots",
        tag="slots",
        values=(5, 7, 9),
    ),
    # 0 means tau sits at its floor from step 0; with the default floor of 0 that is the exact precondition product rather than a relaxation. Exponential handles the degenerate case.
    "tau": SweepAxis(
        param=f"{SCHEDULE_KEY}.init_args.every",
        tag="tau",
        values=(0, 100, 500, 1000, 5000),
    ),
    # Training-set size, in transitions per action schema (N train + N/2 valid). The grid brackets every tuned per-domain minimum, so each domain is scaled both down and up relative to its own default.
    "dataset-size": SweepAxis(
        param="data.init_args.min_per_action_train",
        tag="n",
        values=(20, 50, 100, 200, 500, 1000, 2000),
        linked=_dataset_size_overrides,
    ),
    # Flips per transition. Both states of a transition are corrupted, each by a quarter of the flips as added and a quarter as removed atoms; needs --experiment-type noise.
    "noise": SweepAxis(
        param="data.init_args.noise.added_atoms",
        tag="f",
        values=(1, 2, 4, 8),
        scale=0.25,
        linked=lambda flips: {"data.init_args.noise.removed_atoms": flips / 4},
    ),
}


def _axis_params(axis: SweepAxis) -> list[str]:
    """Every Lightning key an axis pins (the axis parameter plus its linked ones)."""
    linked = axis.linked(axis.values[0]) if axis.linked else {}
    return [axis.param, *linked]


def sweep_experiment_name(experiment: str, axis: SweepAxis, value: float) -> str:
    """Sub-experiment name for one point of a sweep; ``collect_results.py`` imports it to rebuild these."""
    return f"{experiment}-{axis.tag}{value:g}"


def _axis_grid_help() -> str:
    return "; ".join(
        f"{name}: {', '.join(f'{v:g}' for v in axis.values)}"
        for name, axis in SWEEP_AXES.items()
    )


def _sweep_override_args(axis: SweepAxis, value: float) -> list[str]:
    """Lightning args pinning ``axis`` to ``value`` (appended last, so they win)."""
    extra = axis.linked(value) if axis.linked else {}
    linked = [f"--{k}={v:g}" for k, v in extra.items()]
    return [f"--{axis.param}={value * axis.scale:g}", *linked]


# ── the launch, resolved ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SweepPoint:
    """One value of the swept axis: its sub-experiment and the args pinning it. Without a sweep, one point carrying the launch's own name."""

    experiment: str
    overrides: list[str]


@dataclass(frozen=True)
class LaunchPlan:
    """Everything a launch does, decided before anything is submitted: every ``SystemExit`` lives in `build_plan`, so an invalid launch fails before the first sbatch."""

    experiment: str
    experiment_type: str
    labels: str
    ablation: str | None
    validator: str
    threshold: float | None
    checkpoint: str
    domains: list[str]
    seeds: list[int]
    overlays: list[Path]
    points: list[SweepPoint]
    axis: SweepAxis | None
    axis_name: str | None
    values: list[float]
    backend: str
    validate_only: bool
    num_workers: int | None
    resume: bool
    dry_run: bool
    log_wandb: bool
    log_dir: Path
    train_config: Path
    validation_config: Path
    train_resources: dict
    validation_resources: dict

    def train_args(self, point: SweepPoint, domain: str) -> list[str]:
        """Seed-independent Lightning args shared by all runs of a domain."""
        args = ["--config", str(self.train_config)]
        for path in [domain_config(domain), *self.overlays]:
            args += ["--config", str(path)]
        # Absolute and appended last: a Slurm job does not run from the repository root.
        args += [
            "--data.init_args.domain_path",
            str(domain_pddl(domain)),
            "--data.init_args.problem_path",
            str(train_problem(domain)),
        ]
        if self.log_wandb:
            args += [
                "--trainer.logger.init_args.group",
                point.experiment,
                "--trainer.logger.init_args.save_dir",
                str(self.log_dir),
            ]
        else:
            # Without a logger the run directory comes from default_root_dir; see run_directory_args.
            args += ["--trainer.logger=false"]
        return args + point.overrides


def run_directory_args(run_id: str, logs_dir: Path, log_wandb: bool) -> list[str]:
    """Per-seed args putting this run's checkpoints where validation looks.

    ``load_run`` resolves ``<logs>/dias/<run_id>/checkpoints``: the logger's own path with W&B, and ``default_root_dir`` without one. Mirrored in experiments/slurm/train.sh.
    """
    if not log_wandb:
        return ["--trainer.default_root_dir", str(logs_dir / "dias" / run_id)]
    return [
        "--trainer.logger.init_args.id",
        run_id,
        "--trainer.logger.init_args.name",
        run_id,
    ]


def _sweep_points(args: argparse.Namespace) -> tuple:
    """``(points, axis, axis_name, values)``; without ``--sweep``, the single point named by ``--experiment``."""
    overrides = []
    for override in args.overrides:
        if "=" not in override:
            raise SystemExit(f"--set expects key=value, got {override!r}")
        if "," in override:
            raise SystemExit(f"--set value must not contain commas: {override!r}")
        overrides.append(f"--{override}")

    if args.sweep is None:
        if args.sweep_values:
            raise SystemExit("--sweep-values requires --sweep")
        return [SweepPoint(args.experiment, overrides)], None, None, []

    axis_name = args.sweep
    axis = SWEEP_AXES[axis_name]
    values = list(args.sweep_values) if args.sweep_values else list(axis.values)
    for param in _axis_params(axis):
        if any(o.startswith(f"{param}=") for o in args.overrides):
            raise SystemExit(f"--set {param}=... conflicts with --sweep {axis_name}")
    if axis_name == "noise" and args.experiment_type != "noise":
        raise SystemExit("--sweep noise needs --experiment-type noise")
    if axis_name == "aux-loss-alpha" and args.ablation == "no-aux-loss":
        raise SystemExit(
            "--ablation no-aux-loss disables the auxiliary losses, so sweeping their alpha has no effect"
        )
    # The override is appended last so it beats the per-domain config.
    points = [
        SweepPoint(
            sweep_experiment_name(args.experiment, axis, value),
            overrides + _sweep_override_args(axis, value),
        )
        for value in values
    ]
    return points, axis, axis_name, values


def _last_step(run_dir: Path) -> int | None:
    """The step the ``final-`` checkpoint was last written at, or None if there is none."""
    steps = [
        int(match.group(1))
        for path in (run_dir / "checkpoints").glob("final-epoch*.ckpt")
        if (match := re.search(r"step=(\d+)", path.name))
    ]
    return max(steps) if steps else None


def finished_training(run_dir: Path) -> bool:
    """Whether this run reached the ``max_steps`` its own saved config resolved to.

    Neither checkpoint name says on its own that a run finished: ``main.py`` registers ``final-`` unmonitored and ``best-`` on ``valid/loss_adjacency``, so both are rewritten throughout the run and a run killed at half its steps leaves both behind. What does say it is how far the ``final-`` one got, against the ``max_steps`` in ``<run_dir>/config.yaml``, which is per-run and so already accounts for the config chain, a sweep and any ``--set``.
    """
    config = run_dir / "config.yaml"
    if not config.exists():
        return False
    max_steps = (load_yaml(config).get("trainer") or {}).get("max_steps")
    step = _last_step(run_dir)
    if not isinstance(max_steps, int) or max_steps <= 0 or step is None:
        return False
    return step >= max_steps


def unfinished_runs(
    experiments: list[str], domains: list[str], seeds: list[int], logs_dir: Path
) -> list[str]:
    """Run ids among these runs that have no finished training run to validate."""
    # Imported here, not at module scope, so that submission still needs neither torch nor pymimir.
    from validate_runs import run_id_for

    return [
        run_id
        for experiment in experiments
        for domain in domains
        for seed in seeds
        if not finished_training(
            logs_dir / "dias" / (run_id := run_id_for(experiment, domain, seed))
        )
    ]


def build_plan(args: argparse.Namespace) -> LaunchPlan:
    """Resolve command-line arguments into a launch, or refuse them."""
    domains = select_domains(args.domains)
    trains = VALIDATORS[EXPERIMENT_VALIDATOR[args.experiment_type]].from_checkpoint
    if not trains:
        refused = [
            flag
            for flag, given in (
                ("--labels", args.labels != "names"),
                ("--ablation", args.ablation is not None),
                ("--sweep", args.sweep is not None),
                ("--set", bool(args.overrides)),
                ("--validation-threshold", args.validation_threshold is not None),
                ("--validation-checkpoint", args.validation_checkpoint != "final"),
            )
            if given
        ]
        if refused:
            raise SystemExit(
                f"--experiment-type {args.experiment_type} validates a learned domain with no training run behind it, so {', '.join(refused)} has nothing to act on"
            )
        learned = l1_domains()
        unlearned = [domain for domain in domains if domain not in learned]
        if args.domains is None:
            domains = learned
        elif unlearned:
            raise SystemExit(
                f"L1 learned no domain for {', '.join(unlearned)}; experiments/l1_pddl/ holds {', '.join(learned)}"
            )
    # One learned domain per domain: a seed only draws another sample of states.
    seeds = (
        resolve_seeds(args)
        if trains or args.seeds is not None or args.num_runs is not None
        else [0]
    )

    overlays = []
    exp_overlay = EXPERIMENT_OVERLAYS.get(args.experiment_type)
    if exp_overlay is not None:
        overlays.append(exp_overlay)
    if args.labels in LABEL_OVERLAYS:
        overlays.append(LABEL_OVERLAYS[args.labels])
    if args.ablation is not None:
        overlays.append(ABLATION_OVERLAYS[args.ablation])

    threshold = args.validation_threshold
    if threshold is not None and not 0.0 < threshold < 1.0:
        raise SystemExit(
            f"--validation-threshold: expected a probability in (0,1), got {threshold:g}"
        )

    points, axis, axis_name, values = _sweep_points(args)

    validate_only = args.validate_only or not trains
    if trains and validate_only and not args.allow_unfinished:
        unfinished = unfinished_runs(
            [point.experiment for point in points], domains, seeds, args.log_dir
        )
        if unfinished:
            listed = ", ".join(unfinished[:5])
            more = f" (and {len(unfinished) - 5} more)" if len(unfinished) > 5 else ""
            raise SystemExit(
                f"--validate-only: {len(unfinished)} of {len(points) * len(domains) * len(seeds)} runs did not train to max_steps: {listed}{more}. Train them first, or pass --allow-unfinished to validate them as they are."
            )

    on_slurm = args.backend == "slurm"
    return LaunchPlan(
        experiment=args.experiment,
        experiment_type=args.experiment_type,
        labels=args.labels,
        ablation=args.ablation,
        validator=EXPERIMENT_VALIDATOR[args.experiment_type],
        threshold=threshold,
        checkpoint=args.validation_checkpoint,
        domains=domains,
        seeds=seeds,
        overlays=overlays,
        points=points,
        axis=axis,
        axis_name=axis_name,
        values=values,
        backend=args.backend,
        validate_only=validate_only,
        num_workers=args.num_workers,
        resume=args.resume,
        dry_run=args.dry_run,
        log_wandb=not args.no_wandb,
        log_dir=args.log_dir,
        train_config=args.train_config,
        validation_config=args.validation_config,
        train_resources=load_yaml(args.slurm_resources) if on_slurm else {},
        validation_resources=(
            load_yaml(
                args.slurm_resources_validation
                or validation_resources_file(EXPERIMENT_VALIDATOR[args.experiment_type])
            )
            if on_slurm
            else {}
        ),
    )


def describe(plan: LaunchPlan) -> None:
    """Print what is about to happen, before it happens."""
    print(
        f"experiment : {plan.experiment}  (type={plan.experiment_type}, labels={plan.labels}, ablation={plan.ablation or 'none'}, validator={plan.validator}, threshold={'default' if plan.threshold is None else f'{plan.threshold:g}'})"
    )
    print(
        f"domains    : {len(plan.domains)} x {len(plan.seeds)} runs  (seeds={format_array(plan.seeds)}, backend={plan.backend})"
    )
    if plan.validate_only:
        print("steps      : validation only")
    if plan.axis is not None:
        runs = len(plan.points) * len(plan.domains) * len(plan.seeds)
        values = ", ".join(f"{v:g}" for v in plan.values)
        print(f"sweep      : {plan.axis.param} in {{{values}}}  -> {runs} runs total")
    for overlay in plan.overlays:
        print(f"overlay    : {overlay}")


# ── backends ──────────────────────────────────────────────────────────────────


def submit_domain(plan: LaunchPlan, point: SweepPoint, domain: str) -> None:
    """Submit one domain's training array plus its dependent validation array, or under `--validate-only` the second alone."""
    array = format_array(plan.seeds)

    # Single tokens via --export; the Lightning config chain goes positionally, so sbatch preserves argv exactly.
    train_vars = [
        "ALL",
        f"DIAS_ROOT={PROJECT_ROOT}",
        f"EXPERIMENT={point.experiment}",
        f"DOMAIN={domain}",
        f"LOGS_DIR={plan.log_dir}",
    ]
    val_vars = [
        "ALL",
        f"DIAS_ROOT={PROJECT_ROOT}",
        f"EXPERIMENT={point.experiment}",
        f"DOMAIN={domain}",
        f"EXP_TYPE={plan.validator}",
        f"LOGS_DIR={plan.log_dir}",
        f"VALIDATION_CONFIG={plan.validation_config}",
    ]
    if not plan.log_wandb:
        train_vars.append("NO_WANDB=1")
        val_vars.append("NO_WANDB=1")
    if plan.threshold is not None:
        val_vars.append(f"THRESHOLD={plan.threshold:g}")
    if plan.checkpoint != "final":
        val_vars.append(f"CHECKPOINT={plan.checkpoint}")
    if plan.num_workers is not None:
        val_vars.append(f"NUM_WORKERS={plan.num_workers}")
    if plan.resume:
        val_vars.append("RESUME=1")

    dep = None
    if not plan.validate_only:
        train_cmd = [
            "sbatch",
            "--parsable",
            f"--job-name=dias-train-{domain}",
            f"--array={array}",
            sbatch_output_opt(plan.log_dir),
            *sbatch_resource_opts(resolve_resources(plan.train_resources, domain)),
            f"--export={','.join(train_vars)}",
            str(TRAIN_SCRIPT),
            *plan.train_args(point, domain),
        ]
        print(
            f"[{domain}] submit training array ({len(plan.seeds)} runs, seeds {array})"
        )
        dep = run_command(train_cmd, plan.dry_run) or "<train_jobid>"

    val_cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=dias-val-{domain}",
        f"--array={array}",
        # Nothing to wait for when the checkpoints already exist.
        *([f"--dependency=aftercorr:{dep}"] if dep is not None else []),
        sbatch_output_opt(plan.log_dir),
        *sbatch_resource_opts(resolve_resources(plan.validation_resources, domain)),
        f"--export={','.join(val_vars)}",
        str(VALIDATE_SCRIPT),
    ]
    waits = f"aftercorr:{dep}" if dep is not None else "no dependency"
    print(f"[{domain}] submit validation array ({waits})")
    run_command(val_cmd, plan.dry_run)


def run_domain(plan: LaunchPlan, point: SweepPoint, domain: str) -> None:
    """Train and validate one domain's seeds here, one after another; under `--validate-only`, only validate."""
    # Imported here so Slurm submission never needs torch/pymimir.
    from validate_runs import run_id_for, validate_run

    val_cfg = load_yaml(plan.validation_config)
    for seed in plan.seeds:
        run_id = run_id_for(point.experiment, domain, seed)
        if not plan.validate_only:
            train_cmd = [
                "dias",
                "fit",
                *plan.train_args(point, domain),
                # The datamodule's seed follows this one unless a config gave it one.
                "--seed_everything",
                str(seed),
                *run_directory_args(run_id, plan.log_dir, plan.log_wandb),
            ]
            print(f"\n[{domain} seed={seed}] train  (run_id={run_id})", flush=True)
            try:
                # Not captured: this step takes hours, and its output is the only sign of progress.
                run_command(train_cmd, plan.dry_run, capture=False)
            except subprocess.CalledProcessError as error:
                print(
                    f"[skip] {domain} seed={seed}: training failed ({error})",
                    flush=True,
                )
                continue
        print(f"\n[{domain} seed={seed}] validate  (run_id={run_id})", flush=True)
        if plan.dry_run:
            continue
        validate_run(
            experiment=point.experiment,
            domain=domain,
            seed=seed,
            experiment_type=plan.experiment_type,
            val_cfg=val_cfg,
            logs_dir=plan.log_dir,
            run_id=run_id,
            log_wandb=plan.log_wandb,
            threshold=plan.threshold,
            checkpoint=plan.checkpoint,
            num_workers=plan.num_workers,
            resume=plan.resume,
        )


def execute(plan: LaunchPlan) -> None:
    if plan.backend == "slurm" and not plan.dry_run:
        prepare_slurm_logs(plan.log_dir)
    backend = submit_domain if plan.backend == "slurm" else run_domain

    for point in plan.points:
        if len(plan.points) > 1:
            print(f"\n=== sub-experiment {point.experiment} ===")
        for domain in plan.domains:
            backend(plan, point, domain)


# ── command line ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        default=f"run-{datetime.now():%Y%m%d-%H%M%S}",
        help="Experiment name = W&B group = results dir. Defaults to a timestamp.",
    )
    p.add_argument(
        "--experiment-type", default="dias", choices=sorted(EXPERIMENT_TYPES)
    )
    p.add_argument(
        "--labels",
        default="names",
        choices=LABELS,
        help="What each transition is labelled with: the action name (default), full STRIPS actions, or STRIPS+ actions.",
    )
    p.add_argument(
        "--ablation",
        default=None,
        choices=sorted(ABLATION_OVERLAYS),
        help="Model ablation overlay, composed on top of the experiment type (uses the DIAS validator). Default: none (full model).",
    )
    add_domain_arguments(p)
    p.add_argument(
        "--sweep",
        default=None,
        choices=sorted(SWEEP_AXES),
        help="Sweep this hyperparameter: submit the full domain x seed grid once per value, each into sub-experiment <experiment>-<tag><value>. Default grids: "
        + _axis_grid_help()
        + ".",
    )
    p.add_argument(
        "--sweep-values",
        nargs="+",
        type=float,
        default=None,
        metavar="V",
        help="Values for --sweep instead of that axis' default grid.",
    )
    p.add_argument(
        "--validation-threshold",
        type=float,
        default=None,
        metavar="T",
        help="Extract each run's learned domain anew at this schema-derivation threshold for validation. Default: score the one written beside the checkpoint, at the run's schema_threshold.",
    )
    p.add_argument(
        "--validation-checkpoint",
        default="final",
        choices=CHECKPOINTS,
        help="Which of each run's checkpoints validation scores: final, the model as training left it, or best, the one with the lowest validation loss. Default: final.",
    )
    p.add_argument("--backend", default="slurm", choices=["slurm", "local"])
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip training and only (re-)run the validation step, over the checkpoints of runs that already trained to max_steps.",
    )
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
    p.add_argument(
        "--allow-unfinished",
        action="store_true",
        help="With --validate-only, submit even for runs that stopped short of max_steps.",
    )
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
        "--no-wandb",
        action="store_true",
        help="Log neither training nor validation to W&B (both backends).",
    )
    add_path_arguments(p, train_config=True)
    add_logging_arguments(p)
    p.add_argument("--slurm-resources", type=Path, default=SLURM_RESOURCES)
    p.add_argument(
        "--slurm-resources-validation",
        type=Path,
        default=None,
        help="Default: experiments/configs/slurm-resources-validation-mlp.yaml for the MLP-effects validator, slurm-resources-validation.yaml for the others.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The local backend runs both steps in this process, so their log lines need a handler.
    configure_logging(args)
    plan = build_plan(args)
    describe(plan)
    execute(plan)
    print("\nDone submitting." if plan.backend == "slurm" else "\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
