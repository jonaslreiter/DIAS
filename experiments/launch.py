"""Launch DIAS paper experiments: one training array + a dependent validation array per domain.

For each selected domain this submits (Slurm backend):

  1. a training array job with one task per seed (array task id = seed), and
  2. a validation array job over the same task ids with ``--dependency=aftercorr:<train_jobid>`` so validation task i runs only after training task i succeeds.

Seeds default to ``0 .. --num-runs-1``; ``--seeds`` selects an explicit subset (e.g. to re-run only the seeds that failed), which becomes a sparse sbatch array spec such as ``--array=3,7-8``.

Runs are identified by a deterministic id ``<experiment>-<domain>-s<seed>`` (see ``validate_runs.run_id_for``), so the two array jobs need no id handoff: the checkpoint of task i lands at ``logs/dias/<run_id>/checkpoints/best-*.ckpt`` and validation recomputes the same id.

Config composition (in order): ``configs/main.yaml`` -> ``<domain>/config.yaml`` -> optional experiment overlay (``configs/experiments/{mlp,noise}.yaml``) -> optional ablation overlay (``configs/experiments/ablation-*.yaml``, selected by ``--ablation``) -> launcher ``--set`` overrides. The experiment type also selects the validator (``mlp`` uses the MLP baseline; ``dias``/``noise`` use the DIAS validator) and its schema-derivation threshold (0.5, but 0.75 for ``noise``, as in the paper; override with ``--validation-threshold``); ablations always use the DIAS validator.

``--sweep AXIS`` turns one launch into a hyperparameter sweep: the whole domain x seed grid is submitted once per value of that axis (see ``SWEEP_AXES``), each into its own sub-experiment ``<experiment>-<tag><value>`` (see ``sweep_experiment_name``), with the swept parameter overridden last so it wins over per-domain values in ``pddl/<domain>/config.yaml``. ``--reuse-default EXPERIMENT`` skips the point that equals the axis default and takes its results from an existing experiment instead.

Per-domain Slurm resources come from ``experiments/configs/slurm-resources.yaml``.

Examples
--------
    # Full grid on Slurm (all domains, 10 seeds each), grouped as "paper-main":
    python experiments/launch.py --experiment paper-main

    # MLP baseline on two domains:
    python experiments/launch.py --experiment-type mlp --domains gripper satellite --experiment mlp-main

    # Noise sweep point (expected 2 added / 2 removed atoms):
    python experiments/launch.py --experiment-type noise --experiment noise-a2r2 --set data.init_args.noise.added_atoms=2 --set data.init_args.noise.removed_atoms=2

    # Ablation (no PCGrad) on the default DIAS model:
    python experiments/launch.py --ablation no-pcgrad --experiment abl-no-pcgrad

    # aux_loss_weight sweep: all domains x {0.01,0.03,0.1,0.3,1.0} x 3 seeds,
    # submitted as sub-experiments aux-sweep-aux0.01 ... aux-sweep-aux1:
    python experiments/launch.py --experiment aux-sweep --sweep aux-loss-weight --num-runs 3

    # num_slots sweep, reusing the finished dias-full runs for the default 5
    # (only <experiment>-slots7 and <experiment>-slots9 are submitted):
    python experiments/launch.py --experiment dias --sweep num-slots --reuse-default dias-full --num-runs 3

    # Dataset-size sweep, clean and noisy (2 atom flips per transition), on the
    # six-domain subset; sub-experiments <experiment>-n20 ... <experiment>-n2000:
    python experiments/launch.py --experiment datasize-clean --sweep dataset-size --domains blocks-3 delivery gripper hanoi logistics miconic --num-runs 3
    python experiments/launch.py --experiment datasize-noise --sweep dataset-size --experiment-type noise --num-runs 3 --domains blocks-3 delivery gripper hanoi logistics miconic --set data.init_args.noise.removed_atoms=0.5 --set data.init_args.noise.added_atoms=0.5

    # Re-run just the seeds that failed, into the same experiment/group:
    python experiments/launch.py --experiment paper-main --domains gripper --seeds 3 7-8

    # See exactly what would be submitted, without submitting:
    python experiments/launch.py --domains gripper --num-runs 3 --dry-run

    # Run locally (no Slurm), sequentially, good for smoke tests:
    python experiments/launch.py --backend local --domains gripper --num-runs 2 --experiment smoke --no-wandb
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import pipeline`

from pipeline import configure_logging  # noqa: E402
from pipeline import (
    EXPERIMENT_TYPES,
    EXPERIMENT_VALIDATOR,
    OVERLAYS_DIR,
    PROJECT_ROOT,
    SLURM_DIR,
    SLURM_RESOURCES,
    SLURM_RESOURCES_VALIDATION,
    add_domain_arguments,
    add_logging_arguments,
    add_path_arguments,
    domain_config,
    domain_pddl,
    format_array,
    load_yaml,
    prepare_slurm_logs,
    resolve_resources,
    resolve_seeds,
    run_command,
    sbatch_output_opt,
    sbatch_resource_opts,
    select_domains,
    train_problem,
)

EXPERIMENT_OVERLAYS = {
    "mlp": OVERLAYS_DIR / "mlp.yaml",
    "noise": OVERLAYS_DIR / "noise.yaml",
}
# Schema-derivation threshold (Eq. 14) per experiment type; the paper raises it to 0.75 under noise. Override per launch with --validation-threshold.
DEFAULT_VALIDATION_THRESHOLD = 0.5
VALIDATION_THRESHOLD = {"noise": 0.75}
# Model ablation overlays, orthogonal to --experiment-type (all use the DIAS validator). Composed after the experiment overlay.
ABLATION_OVERLAYS = {
    "no-slot-weights": OVERLAYS_DIR / "ablation-no-slot-weights.yaml",
    "no-precondition-loss": OVERLAYS_DIR / "ablation-no-precondition-loss.yaml",
    "no-effect-loss": OVERLAYS_DIR / "ablation-no-effect-loss.yaml",
    "no-aux-loss": OVERLAYS_DIR / "ablation-no-aux-loss.yaml",
    "no-pcgrad": OVERLAYS_DIR / "ablation-no-pcgrad.yaml",
    "no-random-embeddings": OVERLAYS_DIR / "ablation-no-random-embeddings.yaml",
}
# The tau schedule is a typed dataclass, so its fields are ordinary dotted keys.
SCHEDULE_KEY = "model.init_args.gmean_temperature"

TRAIN_SCRIPT = SLURM_DIR / "train.sh"
VALIDATE_SCRIPT = SLURM_DIR / "validate.sh"


@dataclass(frozen=True)
class SweepAxis:
    """One hyperparameter axis that ``--sweep`` varies over the full grid.

    ``values`` is the default grid of the bare flag, ``tag`` the sub-experiment infix, and ``default`` the value the base config chain already realizes: the point ``--reuse-default`` takes from an existing experiment instead of re-running it. ``default = None`` means there is no reusable default (the value is tuned per domain), so ``--reuse-default`` is rejected.

    ``linked`` maps one axis value to *further* Lightning overrides that have to move with it (see ``_dataset_size_overrides``); ``param`` alone is what the axis is named and reported by. All of them are appended last on the command line, so they beat both ``pddl/<domain>/config.yaml`` and the overlays, and all of them are rejected as ``--set`` targets.
    """

    param: str
    tag: str
    values: tuple[float, ...]
    default: float | None
    cast: Callable[[float], float]
    linked: Callable[[float], dict[str, float]] | None = None


def _dataset_size_overrides(value: float) -> dict[str, int]:
    """Sampling knobs that move with ``min_per_label_train`` on the size axis.

    Pinning ``max_per_label_train`` to the same N (and both ``*_valid`` to N/2) makes the dataset size *exact* rather than a lower bound: sampling stops once every action has 1.5N transitions and the cap trims every action back to 1.5N, so the pool is exactly ``num_actions * 1.5N``. The 2:1 train:valid weight then splits it into ~N train / ~N/2 valid per action (the ratio is enforced on the pool as a whole, so individual actions drift a few percent). The per-domain configs instead pair a tuned minimum with a loose cap, so no point of this sweep reproduces a domain's default run exactly, which is why the axis has no reusable ``default``.

    The ``*_floor`` overrides exist for ``--experiment-type noise``: that overlay floors the minima at 200/100, which is composed after the domain config and would otherwise silently clamp the 20/50/100 points back up. Setting each floor to the value it floors makes it a no-op.
    """
    n = int(value)
    half = max(1, n // 2)
    return {
        "data.init_args.max_per_label_train": n,
        "data.init_args.min_per_label_valid": half,
        "data.init_args.max_per_label_valid": half,
        "data.init_args.min_per_label_train_floor": n,
        "data.init_args.min_per_label_valid_floor": half,
    }


SWEEP_AXES: dict[str, SweepAxis] = {
    # Grid covers every value currently tuned in pddl/<domain>/config.yaml; tuned per domain, hence no single default point to reuse.
    "aux-loss-weight": SweepAxis(
        param="model.init_args.aux_loss_weight",
        tag="aux",
        values=(0.01, 0.03, 0.1, 0.3, 1.0),
        default=None,
        cast=float,
    ),
    "num-slots": SweepAxis(
        param="model.init_args.num_slots",
        tag="slots",
        values=(5, 7, 9),
        default=5,
        cast=int,
    ),
    # 0 means tau sits at its floor from step 0; with the default floor of 0 that is the exact precondition product rather than a relaxation. Exponential handles the degenerate case.
    "gmean-temperature": SweepAxis(
        param=f"{SCHEDULE_KEY}.init_args.every",
        tag="gmeanT",
        values=(0, 100, 500, 1000, 5000),
        default=500,
        cast=int,
    ),
    # Training-set size, in transitions per action schema (N train + N/2 valid). The grid brackets every tuned per-domain minimum, so each domain is scaled both down and up relative to its own default.
    "dataset-size": SweepAxis(
        param="data.init_args.min_per_label_train",
        tag="n",
        values=(20, 50, 100, 200, 500, 1000, 2000),
        default=None,
        cast=int,
        linked=_dataset_size_overrides,
    ),
}


def _axis_params(axis: SweepAxis) -> list[str]:
    """Every Lightning key an axis pins (the axis parameter plus its linked ones)."""
    linked = axis.linked(axis.values[0]) if axis.linked else {}
    return [axis.param, *linked]


def sweep_experiment_name(
    experiment: str,
    axis: SweepAxis,
    value: float,
    reuse_default: str | None = None,
) -> str:
    """Sub-experiment name for one point of a sweep.

    With ``reuse_default`` the axis-default point maps to that existing experiment instead of a new sub-experiment. Also used by ``experiments/collect_results.py`` to rebuild the per-value experiment names, so keep the two in sync by importing this function rather than re-deriving it.
    """
    if reuse_default and axis.default is not None and axis.cast(value) == axis.default:
        return reuse_default
    return f"{experiment}-{axis.tag}{value:g}"


def _axis_grid_help() -> str:
    return "; ".join(
        f"{name}: {', '.join(f'{v:g}' for v in axis.values)}"
        for name, axis in SWEEP_AXES.items()
    )


def _config_value(cfg: dict, dotted_key: str):
    """Look up a dotted key in a nested config dict; ``None`` if absent."""
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _sweep_override_args(axis: SweepAxis, value: float) -> list[str]:
    """Lightning args pinning ``axis`` to ``value`` (appended last, so they win)."""
    extra = axis.linked(value) if axis.linked else {}
    linked = [f"--{k}={v:g}" for k, v in extra.items()]
    return [f"--{axis.param}={value:g}", *linked]


def _check_sweep_config(
    axis: SweepAxis,
    axis_name: str,
    domains: list[str],
    train_cfg: dict,
    train_config: Path,
    overlays: list[Path],
    reuse_default: str | None,
) -> None:
    """Fail early when the swept parameter is not where the sweep assumes it is."""
    base = _config_value(train_cfg, axis.param)
    if base is None:
        raise SystemExit(
            f"--sweep {axis_name}: {axis.param} not found in {train_config}"
        )
    if reuse_default and axis.cast(base) != axis.default:
        raise SystemExit(
            f"--reuse-default: {train_config} sets {axis.param}={base}, but the {axis_name} axis default is {axis.default:g}; '{reuse_default}' therefore did not run that point"
        )
    paths = [domain_config(domain) for domain in domains]
    # Overlays are composed after the domain config and can carry the same parameters (the noise overlay replaces the whole schedule, for instance).
    for path in paths + overlays:
        cfg = load_yaml(path) or {}
        if reuse_default and _config_value(cfg, axis.param) is not None:
            raise SystemExit(
                f"--reuse-default: {path} overrides {axis.param}, so '{reuse_default}' did not run it at the axis default {axis.default:g}"
            )


# ── the launch, resolved ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SweepPoint:
    """One value of the swept axis: its sub-experiment and the args pinning it.

    Without a sweep there is exactly one point, carrying the launch's own name and only the ``--set`` overrides. ``reused`` marks the axis-default point that ``--reuse-default`` maps onto an existing experiment; it is described but never submitted.
    """

    experiment: str
    overrides: list[str]
    reused: bool = False


@dataclass(frozen=True)
class LaunchPlan:
    """Everything a launch does, decided before anything is submitted.

    Building this is where every ``SystemExit`` lives, so an invalid launch fails before the first sbatch rather than half way through the grid.
    """

    experiment: str
    experiment_type: str
    ablation: str | None
    validator: str
    threshold: float
    domains: list[str]
    seeds: list[int]
    overlays: list[Path]
    points: list[SweepPoint]
    axis: SweepAxis | None
    axis_name: str | None
    values: list[float]
    reuse_default: str | None
    backend: str
    dry_run: bool
    log_wandb: bool
    log_dir: Path
    train_config: Path
    validation_config: Path
    train_resources: dict
    validation_resources: dict

    @property
    def submitted(self) -> list[SweepPoint]:
        return [point for point in self.points if not point.reused]

    def train_args(self, point: SweepPoint, domain: str) -> list[str]:
        """Seed-independent Lightning args shared by all runs of a domain."""
        args = ["--config", str(self.train_config)]
        for path in [domain_config(domain), *self.overlays]:
            args += ["--config", str(path)]
        # Absolute, and appended after the config chain: a Slurm job does not run from the repository root, where the configs' relative paths resolve.
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
            # Without a logger the run directory comes from default_root_dir, which run_directory_args sets to the path validation looks the checkpoint up under.
            args += ["--trainer.logger=false"]
        return args + point.overrides


def run_directory_args(run_id: str, logs_dir: Path, log_wandb: bool) -> list[str]:
    """Per-seed args putting this run's checkpoints where validation looks.

    ``validation.checkpoint.load_run`` resolves ``<logs>/dias/<run_id>/ checkpoints``. With W&B that path is the logger's own (project / run id); without one, ModelCheckpoint falls back to ``default_root_dir``, so the same path has to be named explicitly; otherwise a --no-wandb run writes its checkpoint to ./checkpoints and validation cannot find it. Mirrored in experiments/slurm/train.sh, which builds the same args for an array task.
    """
    if not log_wandb:
        return ["--trainer.default_root_dir", str(logs_dir / "dias" / run_id)]
    return [
        "--trainer.logger.init_args.id",
        run_id,
        "--trainer.logger.init_args.name",
        run_id,
    ]


def _sweep_points(args: argparse.Namespace, plan_context: dict) -> tuple:
    """The sweep points of a launch, and the axis they came from.

    Returns ``(points, axis, axis_name, values)``; without ``--sweep`` that is the single point named by ``--experiment``.
    """
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
        if args.reuse_default:
            raise SystemExit("--reuse-default requires --sweep")
        return [SweepPoint(args.experiment, overrides)], None, None, []

    axis_name = args.sweep
    axis = SWEEP_AXES[axis_name]
    values = list(args.sweep_values) if args.sweep_values else list(axis.values)
    for param in _axis_params(axis):
        if any(o.startswith(f"{param}=") for o in args.overrides):
            raise SystemExit(f"--set {param}=... conflicts with --sweep {axis_name}")
    if axis_name == "aux-loss-weight" and args.ablation == "no-aux-loss":
        raise SystemExit(
            "--ablation no-aux-loss disables the auxiliary losses, so sweeping their weight has no effect"
        )
    if args.reuse_default and axis.default is None:
        raise SystemExit(
            f"--reuse-default: the {axis_name} axis has no reusable default (the value is tuned per domain)"
        )
    _check_sweep_config(
        axis,
        axis_name,
        plan_context["domains"],
        load_yaml(args.train_config),
        args.train_config,
        plan_context["overlays"],
        args.reuse_default,
    )
    # The override is appended last so it beats the per-domain config.
    points = [
        SweepPoint(
            name := sweep_experiment_name(
                args.experiment, axis, value, args.reuse_default
            ),
            overrides + _sweep_override_args(axis, value),
            reused=name == args.reuse_default,
        )
        for value in values
    ]
    return points, axis, axis_name, values


def build_plan(args: argparse.Namespace) -> LaunchPlan:
    """Resolve command-line arguments into a launch, or refuse them."""
    domains = select_domains(args.domains)
    seeds = resolve_seeds(args)

    overlays = []
    exp_overlay = EXPERIMENT_OVERLAYS.get(args.experiment_type)
    if exp_overlay is not None:
        overlays.append(exp_overlay)
    if args.ablation is not None:
        overlays.append(ABLATION_OVERLAYS[args.ablation])

    threshold = (
        args.validation_threshold
        if args.validation_threshold is not None
        else VALIDATION_THRESHOLD.get(
            args.experiment_type, DEFAULT_VALIDATION_THRESHOLD
        )
    )
    if not 0.0 < threshold < 1.0:
        raise SystemExit(
            f"--validation-threshold: expected a probability in (0,1), got {threshold:g}"
        )

    points, axis, axis_name, values = _sweep_points(
        args,
        {"domains": domains, "overlays": overlays},
    )

    on_slurm = args.backend == "slurm"
    return LaunchPlan(
        experiment=args.experiment,
        experiment_type=args.experiment_type,
        ablation=args.ablation,
        validator=EXPERIMENT_VALIDATOR[args.experiment_type],
        threshold=threshold,
        domains=domains,
        seeds=seeds,
        overlays=overlays,
        points=points,
        axis=axis,
        axis_name=axis_name,
        values=values,
        reuse_default=args.reuse_default,
        backend=args.backend,
        dry_run=args.dry_run,
        log_wandb=not args.no_wandb,
        log_dir=args.log_dir,
        train_config=args.train_config,
        validation_config=args.validation_config,
        train_resources=load_yaml(args.slurm_resources) if on_slurm else {},
        validation_resources=(
            load_yaml(args.slurm_resources_validation) if on_slurm else {}
        ),
    )


def describe(plan: LaunchPlan) -> None:
    """Print what is about to happen, before it happens."""
    print(
        f"experiment : {plan.experiment}  (type={plan.experiment_type}, ablation={plan.ablation or 'none'}, validator={plan.validator}, threshold={plan.threshold:g})"
    )
    print(
        f"domains    : {len(plan.domains)} x {len(plan.seeds)} runs  (seeds={format_array(plan.seeds)}, backend={plan.backend})"
    )
    if plan.axis is not None:
        runs = len(plan.submitted) * len(plan.domains) * len(plan.seeds)
        values = ", ".join(f"{v:g}" for v in plan.values)
        print(f"sweep      : {plan.axis.param} in {{{values}}}  -> {runs} runs total")
        if len(plan.submitted) < len(plan.points):
            print(
                f"reuse      : {plan.axis.default:g} not submitted; results come from experiment '{plan.reuse_default}'"
            )
    for overlay in plan.overlays:
        print(f"overlay    : {overlay}")


# ── backends ──────────────────────────────────────────────────────────────────


def submit_domain(plan: LaunchPlan, point: SweepPoint, domain: str) -> None:
    """Submit one domain's training array plus its dependent validation array."""
    # Array task id == seed, so a seed subset becomes a sparse array spec and aftercorr still pairs train/validate tasks of the same seed.
    array = format_array(plan.seeds)

    # EXPERIMENT/DOMAIN go through --export (single tokens, needed to build the run id); the Lightning config chain is passed as positional args so sbatch preserves argv exactly (no fragile string splitting).
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
    # validate.sh only forwards --threshold when THRESHOLD is set, so leave it unset at the default and keep the plain pipeline unchanged.
    if plan.threshold != DEFAULT_VALIDATION_THRESHOLD:
        val_vars.append(f"THRESHOLD={plan.threshold:g}")

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
    print(f"[{domain}] submit training array ({len(plan.seeds)} runs, seeds {array})")
    train_jobid = run_command(train_cmd, plan.dry_run)

    dep = train_jobid or "<train_jobid>"
    val_cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=dias-val-{domain}",
        f"--array={array}",
        f"--dependency=aftercorr:{dep}",
        sbatch_output_opt(plan.log_dir),
        *sbatch_resource_opts(resolve_resources(plan.validation_resources, domain)),
        f"--export={','.join(val_vars)}",
        str(VALIDATE_SCRIPT),
    ]
    print(f"[{domain}] submit validation array (aftercorr:{dep})")
    run_command(val_cmd, plan.dry_run)


def run_domain(plan: LaunchPlan, point: SweepPoint, domain: str) -> None:
    """Train and validate one domain's seeds here, one after another."""
    # Imported here so Slurm submission never needs torch/pymimir installed.
    from validate_runs import run_id_for, validate_run

    val_cfg = load_yaml(plan.validation_config)
    for seed in plan.seeds:
        run_id = run_id_for(point.experiment, domain, seed)
        train_cmd = [
            "dias",
            "fit",
            *plan.train_args(point, domain),
            "--seed_everything",
            str(seed),
            "--data.init_args.seed",
            str(seed),
            *run_directory_args(run_id, plan.log_dir, plan.log_wandb),
        ]
        print(f"\n[{domain} seed={seed}] train  (run_id={run_id})", flush=True)
        try:
            # Not captured: this is the step that takes hours, and watching it is the only way to see where it is.
            run_command(train_cmd, plan.dry_run, capture=False)
        except subprocess.CalledProcessError as error:
            print(f"[skip] {domain} seed={seed}: training failed ({error})", flush=True)
            continue
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
        )


def execute(plan: LaunchPlan) -> None:
    if plan.backend == "slurm" and not plan.dry_run:
        prepare_slurm_logs(plan.log_dir)
    backend = submit_domain if plan.backend == "slurm" else run_domain

    for point in plan.points:
        if point.reused:
            print(f"\n=== {point.experiment} (reused, nothing submitted) ===")
            continue
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
        "--reuse-default",
        default=None,
        metavar="EXPERIMENT",
        help="Do not submit the sweep point equal to the axis default; take its results from this existing experiment (e.g. dias-full). Read back by 'collect_results.py --reuse-default'.",
    )
    p.add_argument(
        "--validation-threshold",
        type=float,
        default=None,
        metavar="T",
        help=f"Schema-derivation threshold used by the validation step. Default: {DEFAULT_VALIDATION_THRESHOLD:g}, except "
        + ", ".join(
            f"{t:g} for --experiment-type {n}" for n, t in VALIDATION_THRESHOLD.items()
        )
        + ".",
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
        default=SLURM_RESOURCES_VALIDATION,
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The local backend runs training and validation in this process, so their log lines need a handler; the Slurm backend only prints what it submits.
    configure_logging(args)
    plan = build_plan(args)
    describe(plan)
    execute(plan)
    print("\nDone submitting." if plan.backend == "slurm" else "\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
