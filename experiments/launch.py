"""Launch an experiment: per domain, a Slurm array training one run per seed, and a validation array whose task i waits for training task i.

A run's config is ``configs/main.yaml``, ``pddl/<domain>/config.yaml``, the ``--experiment-type``, ``--labels`` and ``--ablation`` overlays and the ``--set`` and sweep overrides, in this order; its id is ``<experiment>-<domain>-s<seed>``. ``--sweep AXIS`` launches every value of an axis as an experiment ``<experiment>-<tag><value>``; the ``threshold`` axis validates the existing runs of ``--experiment`` at each derivation threshold. ``--validate-only`` validates runs that finished training, and ``--backend local`` runs everything here.

Examples:
    python experiments/launch.py --experiment dias-names
    python experiments/launch.py --experiment dias-strips --labels strips
    python experiments/launch.py --experiment alpha --sweep aux-loss-alpha --seeds 0-4
    python experiments/launch.py --experiment dias-names --sweep threshold --seeds 0-4
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import configure_logging  # noqa: E402
from pipeline import (
    CHECKPOINTS,
    CONFIGS_DIR,
    EXPERIMENT_TYPES,
    EXPERIMENT_VALIDATOR,
    PROJECT_ROOT,
    SLURM_CONFIG,
    SLURM_DIR,
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
    training_resources,
    validation_resources,
)

EXPERIMENT_OVERLAYS = {
    "mlp": CONFIGS_DIR / "experiment-mlp.yaml",
    "noise": CONFIGS_DIR / "experiment-noise.yaml",
}
LABEL_OVERLAYS = {
    "strips": CONFIGS_DIR / "labels-strips.yaml",
    "strips-plus": CONFIGS_DIR / "labels-strips-plus.yaml",
}
LABELS = ("names", *LABEL_OVERLAYS)
ABLATION_OVERLAYS = {
    "no-slot-weights": CONFIGS_DIR / "ablation-no-slot-weights.yaml",
    "no-precondition-loss": CONFIGS_DIR / "ablation-no-precondition-loss.yaml",
    "no-effect-loss": CONFIGS_DIR / "ablation-no-effect-loss.yaml",
    "no-aux-loss": CONFIGS_DIR / "ablation-no-aux-loss.yaml",
    "no-pcgrad": CONFIGS_DIR / "ablation-no-pcgrad.yaml",
    "no-random-features": CONFIGS_DIR / "ablation-no-random-features.yaml",
}
SCHEDULE_KEY = "model.init_args.precondition_tau"

TRAIN_SCRIPT = SLURM_DIR / "train.sh"
VALIDATE_SCRIPT = SLURM_DIR / "validate.sh"


@dataclass(frozen=True)
class SweepAxis:
    """A hyperparameter ``--sweep`` varies: each value sets `param` to ``value * scale``, `linked` the further overrides that move with it, into the experiment ``<experiment>-<tag><value>``. A `validation` axis is the derivation threshold, which validates existing runs."""

    param: str
    tag: str
    values: tuple[float, ...]
    scale: float = 1.0
    linked: Callable[[float], dict[str, float]] | None = None
    validation: bool = False


def _dataset_size_overrides(value: float) -> dict[str, int]:
    """Exactly N training and N/2 validation transitions per action."""
    n = int(value)
    half = max(1, n // 2)
    return {
        "data.init_args.max_per_action_train": n,
        "data.init_args.min_per_action_valid": half,
        "data.init_args.max_per_action_valid": half,
    }


SWEEP_AXES: dict[str, SweepAxis] = {
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
    # T in tau = 0.1^(t/T); T = 0 is the plain product from the start.
    "tau": SweepAxis(
        param=f"{SCHEDULE_KEY}.init_args.every",
        tag="tau",
        values=(0, 100, 500, 1000, 5000),
    ),
    # N training and N/2 validation transitions per action.
    "dataset-size": SweepAxis(
        param="data.init_args.min_per_action_train",
        tag="n",
        values=(20, 50, 100, 200, 500, 1000, 2000),
        linked=_dataset_size_overrides,
    ),
    # Flips per transition: a quarter added and a quarter removed in each state.
    "noise": SweepAxis(
        param="data.init_args.noise.added_atoms",
        tag="f",
        values=(1, 2, 4, 8),
        scale=0.25,
        linked=lambda flips: {"data.init_args.noise.removed_atoms": flips / 4},
    ),
    "threshold": SweepAxis(
        param="model.init_args.derivation_threshold",
        tag="thr",
        values=(0.1, 0.25, 0.5, 0.75, 0.9),
        validation=True,
    ),
}


def _axis_params(axis: SweepAxis) -> list[str]:
    """The keys an axis sets."""
    linked = axis.linked(axis.values[0]) if axis.linked else {}
    return [axis.param, *linked]


def sweep_experiment_name(experiment: str, axis: SweepAxis, value: float) -> str:
    """The experiment of one value of a sweep."""
    return f"{experiment}-{axis.tag}{value:g}"


def _axis_grid_help() -> str:
    return "; ".join(
        f"{name}: {', '.join(f'{v:g}' for v in axis.values)}"
        for name, axis in SWEEP_AXES.items()
    )


def _sweep_override_args(axis: SweepAxis, value: float) -> list[str]:
    """The overrides setting `axis` to `value`."""
    extra = axis.linked(value) if axis.linked else {}
    linked = [f"--{k}={v:g}" for k, v in extra.items()]
    return [f"--{axis.param}={value * axis.scale:g}", *linked]


@dataclass(frozen=True)
class SweepPoint:
    """One experiment of a launch: the runs it trains or validates under `experiment`, with `overrides`, validated into `results` at derivation `threshold` (None: the run's own)."""

    experiment: str
    results: str
    overrides: list[str]
    threshold: float | None = None


@dataclass(frozen=True)
class LaunchPlan:
    """Everything a launch does, resolved and checked before anything runs."""

    experiment: str
    experiment_type: str
    labels: str
    ablation: str | None
    validator: str
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
        """The Lightning arguments of every run of `domain` in `point`."""
        args = ["--config", str(self.train_config)]
        for path in [domain_config(domain), *self.overlays]:
            args += ["--config", str(path)]
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
            args += ["--trainer.logger=false"]
        return args + point.overrides


def run_directory_args(run_id: str, logs_dir: Path, log_wandb: bool) -> list[str]:
    """The arguments that put a run's checkpoints into ``<logs>/dias/<run_id>/checkpoints``, as experiments/slurm/train.sh does."""
    if not log_wandb:
        return ["--trainer.default_root_dir", str(logs_dir / "dias" / run_id)]
    return [
        "--trainer.logger.init_args.id",
        run_id,
        "--trainer.logger.init_args.name",
        run_id,
    ]


def _sweep_points(args: argparse.Namespace) -> tuple:
    """``(points, axis, axis_name, values)``; without ``--sweep``, the one point of ``--experiment``."""
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
        return [SweepPoint(args.experiment, args.experiment, overrides)], None, None, []

    axis_name = args.sweep
    axis = SWEEP_AXES[axis_name]
    values = list(args.sweep_values) if args.sweep_values else list(axis.values)
    for param in _axis_params(axis):
        if any(o.startswith(f"{param}=") for o in args.overrides):
            raise SystemExit(f"--set {param}=... conflicts with --sweep {axis_name}")
    if axis.validation:
        if overrides:
            raise SystemExit(
                f"--sweep {axis_name} validates existing runs; --set has nothing to act on"
            )
        points = [
            SweepPoint(
                args.experiment,
                sweep_experiment_name(args.experiment, axis, value),
                [],
                value,
            )
            for value in values
        ]
        return points, axis, axis_name, values
    if axis_name == "noise" and args.experiment_type != "noise":
        raise SystemExit("--sweep noise needs --experiment-type noise")
    if axis_name == "aux-loss-alpha" and args.ablation == "no-aux-loss":
        raise SystemExit(
            "--ablation no-aux-loss disables the auxiliary losses, so sweeping their alpha has no effect"
        )
    points = []
    for value in values:
        name = sweep_experiment_name(args.experiment, axis, value)
        points.append(
            SweepPoint(name, name, overrides + _sweep_override_args(axis, value))
        )
    return points, axis, axis_name, values


def _last_step(run_dir: Path) -> int | None:
    """The step of the ``final-`` checkpoint, or None."""
    steps = [
        int(match.group(1))
        for path in (run_dir / "checkpoints").glob("final-epoch*.ckpt")
        if (match := re.search(r"step=(\d+)", path.name))
    ]
    return max(steps) if steps else None


def finished_training(run_dir: Path) -> bool:
    """Whether the run's ``final-`` checkpoint reached the ``max_steps`` of its saved config."""
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
    """The ids of these runs that did not finish training."""
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
    """The launch the arguments describe, or `SystemExit` if they are inconsistent."""
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
    # L1 has one learned domain per domain; a seed only draws other test states.
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

    points, axis, axis_name, values = _sweep_points(args)

    validate_only = (
        args.validate_only or not trains or (axis is not None and axis.validation)
    )
    if trains and validate_only and not args.allow_unfinished:
        unfinished = unfinished_runs(
            list(dict.fromkeys(point.experiment for point in points)),
            domains,
            seeds,
            args.log_dir,
        )
        if unfinished:
            listed = ", ".join(unfinished[:5])
            more = f" (and {len(unfinished) - 5} more)" if len(unfinished) > 5 else ""
            raise SystemExit(
                f"--validate-only: {len(unfinished)} of {len(points) * len(domains) * len(seeds)} runs did not train to max_steps: {listed}{more}. Train them first, or pass --allow-unfinished to validate them as they are."
            )

    slurm_config = load_yaml(args.slurm_config) if args.backend == "slurm" else None
    return LaunchPlan(
        experiment=args.experiment,
        experiment_type=args.experiment_type,
        labels=args.labels,
        ablation=args.ablation,
        validator=EXPERIMENT_VALIDATOR[args.experiment_type],
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
        train_resources=(
            training_resources(slurm_config, args.experiment_type)
            if slurm_config
            else {}
        ),
        validation_resources=(
            validation_resources(slurm_config, args.experiment_type)
            if slurm_config
            else {}
        ),
    )


def describe(plan: LaunchPlan) -> None:
    print(
        f"experiment : {plan.experiment}  (type={plan.experiment_type}, labels={plan.labels}, ablation={plan.ablation or 'none'}, validator={plan.validator})"
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


def submit_domain(plan: LaunchPlan, point: SweepPoint, domain: str) -> None:
    """Submit one domain's training array and its validation array, or the latter alone."""
    array = format_array(plan.seeds)
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
    if not plan.log_wandb or point.results != point.experiment:
        val_vars.append("NO_WANDB=1")
    if point.results != point.experiment:
        val_vars.append(f"RESULTS_EXPERIMENT={point.results}")
    if point.threshold is not None:
        val_vars.append(f"THRESHOLD={point.threshold:g}")
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
    """Train and validate one domain's seeds here, one after another, or only validate."""
    from validate_runs import run_id_for, validate_run

    val_cfg = load_yaml(plan.validation_config)
    for seed in plan.seeds:
        run_id = run_id_for(point.experiment, domain, seed)
        if not plan.validate_only:
            train_cmd = [
                "dias",
                "fit",
                *plan.train_args(point, domain),
                "--seed_everything",
                str(seed),
                *run_directory_args(run_id, plan.log_dir, plan.log_wandb),
            ]
            print(f"\n[{domain} seed={seed}] train  (run_id={run_id})", flush=True)
            try:
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
            log_wandb=plan.log_wandb and point.results == point.experiment,
            threshold=point.threshold,
            results_experiment=point.results,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        default=f"run-{datetime.now():%Y%m%d-%H%M%S}",
        help="Experiment name, also the W&B group. Default: a timestamp.",
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
        help="An ablation of the model. Default: none.",
    )
    add_domain_arguments(p)
    p.add_argument(
        "--sweep",
        default=None,
        choices=sorted(SWEEP_AXES),
        help="Launch every value of this axis as the experiment <experiment>-<tag><value>; threshold validates the existing runs of --experiment. Default values: "
        + _axis_grid_help()
        + ".",
    )
    p.add_argument(
        "--sweep-values",
        nargs="+",
        type=float,
        default=None,
        metavar="V",
        help="Values for --sweep instead of its defaults.",
    )
    p.add_argument(
        "--validation-checkpoint",
        default="final",
        choices=CHECKPOINTS,
        help="The checkpoint validation scores: final (at the end of training, default) or best (at the lowest validation loss).",
    )
    p.add_argument("--backend", default="slurm", choices=["slurm", "local"])
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate runs that finished training.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes per validation. Default: one per CPU.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue validations from the problems and states they scored.",
    )
    p.add_argument(
        "--allow-unfinished",
        action="store_true",
        help="Validate runs that did not finish training too.",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="A Lightning override, repeatable; no commas in the value.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Log nothing to W&B.",
    )
    add_path_arguments(p, train_config=True)
    add_logging_arguments(p)
    p.add_argument(
        "--slurm-config",
        type=Path,
        default=SLURM_CONFIG,
        help="Slurm resources per job kind and domain.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args)
    plan = build_plan(args)
    describe(plan)
    execute(plan)
    print("\nDone submitting." if plan.backend == "slurm" else "\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
