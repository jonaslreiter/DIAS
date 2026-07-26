"""Launch DIAS paper experiments: one training array + a dependent validation array per domain.

For each selected domain this submits (Slurm backend):

  1. a training array job with one task per seed (array task id = seed), and
  2. a validation array job over the same task ids with
     ``--dependency=aftercorr:<train_jobid>`` so validation task i runs only
     after training task i succeeds.

Seeds default to ``0 .. --num-runs-1``; ``--seeds`` selects an explicit subset
(e.g. to re-run only the seeds that failed), which becomes a sparse sbatch
array spec such as ``--array=3,7-8``.

Runs are identified by a deterministic id ``<experiment>-<domain>-s<seed>`` (see
``validate_runs.run_id_for``), so the two array jobs need no id handoff: the
checkpoint of task i lands at ``logs/dias/<run_id>/checkpoints/best-*.ckpt`` and
validation recomputes the same id.

Config composition (in order): ``configs/main.yaml`` -> ``<domain>/config.yaml``
-> optional experiment overlay (``configs/experiments/{mlp,noise}.yaml``) ->
optional ablation overlay (``configs/experiments/ablation-*.yaml``, selected by
``--ablation``) -> launcher ``--set`` overrides. The experiment type also selects
the validator (``mlp`` uses the MLP baseline; ``dias``/``noise`` use the DIAS
validator) and its schema-derivation threshold (0.5, but 0.75 for ``noise``, as
in the paper; override with ``--validation-threshold``); ablations always use
the DIAS validator.

``--sweep AXIS`` turns one launch into a hyperparameter sweep: the whole domain
x seed grid is submitted once per value of that axis (see ``SWEEP_AXES``), each
into its own sub-experiment ``<experiment>-<tag><value>`` (see
``sweep_experiment_name``), with the swept parameter overridden last so it wins
over per-domain values in ``pddl/<domain>/config.yaml``. ``--reuse-default
EXPERIMENT`` skips the point that equals the axis default and takes its results
from an existing experiment instead.

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

    # aux_loss_weight sweep: all domains x {0.01,0.03,0.1,0.3,1.0} x 3 seeds,
    # submitted as sub-experiments aux-sweep-aux0.01 ... aux-sweep-aux1:
    python scripts/launch.py --experiment aux-sweep --sweep aux-loss-weight \
        --num-runs 3

    # num_slots sweep, reusing the finished dias-full runs for the default 5
    # (only <experiment>-slots7 and <experiment>-slots9 are submitted):
    python scripts/launch.py --experiment dias --sweep num-slots \
        --reuse-default dias-full --num-runs 3

    # Re-run just the seeds that failed, into the same experiment/group:
    python scripts/launch.py --experiment paper-main --domains gripper --seeds 3 7-8

    # See exactly what would be submitted, without submitting:
    python scripts/launch.py --domains gripper --num-runs 3 --dry-run

    # Run locally (no Slurm), sequentially — good for smoke tests:
    python scripts/launch.py --backend local --domains gripper --num-runs 2 \
        --experiment smoke --no-wandb
"""

from __future__ import annotations

import argparse
import copy
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "configs" / "experiments"
EXPERIMENT_OVERLAYS = {
    "mlp": EXPERIMENTS_DIR / "mlp.yaml",
    "noise": EXPERIMENTS_DIR / "noise.yaml",
}
# Which validator each experiment type uses (noise still uses the DIAS validator).
VALIDATOR_TYPE = {"dias": "dias", "noise": "dias", "mlp": "mlp"}
# Schema-derivation threshold (Eq. 14) per experiment type; the paper raises it
# to 0.75 under noise. Override per launch with --validation-threshold.
DEFAULT_VALIDATION_THRESHOLD = 0.5
VALIDATION_THRESHOLD = {"noise": 0.75}
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
# Lightning key of the whole loss-weight schedule dict; the gmean-temperature
# axis lives inside it (see _sweep_override_args).
SCHEDULE_KEY = "model.init_args.loss_weight_schedule"


@dataclass(frozen=True)
class SweepAxis:
    """One hyperparameter axis that ``--sweep`` varies over the full grid.

    ``values`` is the default grid of the bare flag, ``tag`` the sub-experiment
    infix, and ``default`` the value the base config chain already realizes —
    the point ``--reuse-default`` takes from an existing experiment instead of
    re-running it. ``default = None`` means there is no reusable default (the
    value is tuned per domain), so ``--reuse-default`` is rejected.
    """

    param: str
    tag: str
    values: tuple[float, ...]
    default: float | None
    cast: Callable[[float], float]


SWEEP_AXES: dict[str, SweepAxis] = {
    # Grid covers every value currently tuned in pddl/<domain>/config.yaml;
    # tuned per domain, hence no single default point to reuse.
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
    # T = 0 is a special point: constant tau = 0 from step 0, i.e. the exact
    # precondition product instead of a relaxation (see _sweep_override_args).
    "gmean-temperature": SweepAxis(
        param=f"{SCHEDULE_KEY}.gmean_temperature.T",
        tag="gmeanT",
        values=(0, 100, 500, 1000, 5000),
        default=500,
        cast=int,
    ),
}


def sweep_experiment_name(
    experiment: str,
    axis: SweepAxis,
    value: float,
    reuse_default: str | None = None,
) -> str:
    """Sub-experiment name for one point of a sweep.

    With ``reuse_default`` the axis-default point maps to that existing
    experiment instead of a new sub-experiment. Also used by
    ``scripts/collect_results.py`` to rebuild the per-value experiment names, so
    keep the two in sync by importing this function rather than re-deriving it.
    """
    if reuse_default and axis.default is not None and axis.cast(value) == axis.default:
        return reuse_default
    return f"{experiment}-{axis.tag}{value:g}"


def _axis_grid_help() -> str:
    return "; ".join(
        f"{name}: {', '.join(f'{v:g}' for v in axis.values)}"
        for name, axis in SWEEP_AXES.items()
    )


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _config_value(cfg: dict, dotted_key: str):
    """Look up a dotted key in a nested config dict; ``None`` if absent."""
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _sweep_override_args(axis: SweepAxis, value: float, train_cfg: dict) -> list[str]:
    """Lightning args pinning ``axis`` to ``value`` (appended last, so they win)."""
    if axis.param.startswith(f"{SCHEDULE_KEY}."):
        # jsonargparse rebuilds a dict-typed argument from scratch when a
        # sub-key is set on the command line, so passing
        # `--...loss_weight_schedule.gmean_temperature.T=100` would silently
        # drop type/v0/vT. Send the whole schedule with the value substituted.
        schedule = copy.deepcopy(_config_value(train_cfg, SCHEDULE_KEY))
        *path, leaf = axis.param[len(SCHEDULE_KEY) + 1 :].split(".")
        node = schedule
        for part in path:
            node = node[part]
        if leaf == "T" and axis.cast(value) == 0:
            # T = 0 is the limit of the exponential decay: the scheduled value
            # is 0 from step 0 on (v0 * (vT/v0)**(step/T) -> 0 as T -> 0+), and
            # gmean's divisor becomes 1, i.e. the exact precondition product
            # instead of a relaxation. Expressed as a constant schedule because
            # the exponential form would divide by T.
            node.clear()
            node.update({"type": "linear", "keypoints": {0: 0.0}})
        else:
            node[leaf] = axis.cast(value)
        # YAML flow style, not JSON: `keypoints` is keyed by int step, and JSON
        # would stringify those keys (LossWeightSchedule then compares int to str).
        dumped = yaml.safe_dump(schedule, default_flow_style=True, width=10**9)
        return [f"--{SCHEDULE_KEY}={dumped.strip()}"]
    return [f"--{axis.param}={value:g}"]


def _check_sweep_config(
    axis: SweepAxis,
    axis_name: str,
    domains: list[str],
    domain_settings: dict,
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
            f"--reuse-default: {train_config} sets {axis.param}={base}, but the "
            f"{axis_name} axis default is {axis.default:g}; '{reuse_default}' "
            "therefore did not run that point"
        )
    paths = [
        (PROJECT_ROOT / domain_settings[domain]["domain"]).parent / "config.yaml"
        for domain in domains
    ]
    # Overlays are composed after the domain config and can carry the same
    # parameters (the noise overlay replaces the whole schedule, for instance).
    for path in paths + overlays:
        cfg = _load_yaml(path) or {}
        in_schedule = axis.param.startswith(f"{SCHEDULE_KEY}.")
        if in_schedule and _config_value(cfg, SCHEDULE_KEY):
            raise SystemExit(
                f"--sweep {axis_name}: {path} overrides {SCHEDULE_KEY}, which "
                "the sweep replaces wholesale"
            )
        if reuse_default and _config_value(cfg, axis.param) is not None:
            raise SystemExit(
                f"--reuse-default: {path} overrides {axis.param}, so "
                f"'{reuse_default}' did not run it at the axis default "
                f"{axis.default:g}"
            )


def _parse_seeds(specs: list[str]) -> list[int]:
    """Parse ``--seeds`` tokens into a sorted, deduplicated seed list.

    Accepts single seeds and inclusive ranges, space- or comma-separated:
    ``--seeds 3 7-9`` and ``--seeds 3,7-9`` both give ``[3, 7, 8, 9]``.
    """
    seeds: set[int] = set()
    for spec in specs:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            lo_s, sep, hi_s = part.partition("-")
            try:
                lo = int(lo_s)
                hi = int(hi_s) if sep else lo
            except ValueError:
                raise SystemExit(f"--seeds: expected N or N-M, got {part!r}")
            if lo < 0 or hi < lo:
                raise SystemExit(f"--seeds: invalid range {part!r}")
            seeds.update(range(lo, hi + 1))
    if not seeds:
        raise SystemExit("--seeds: no seeds given")
    return sorted(seeds)


def _format_array(seeds: list[int]) -> str:
    """Compress a sorted seed list into an sbatch ``--array`` spec (``3,7-9``)."""
    parts: list[str] = []
    start = prev = seeds[0]
    for seed in seeds[1:]:
        if seed == prev + 1:
            prev = seed
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        start = prev = seed
    parts.append(f"{start}-{prev}" if prev > start else str(start))
    return ",".join(parts)


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
    # shlex.join keeps the printed command copy-pasteable: sweep overrides can
    # carry a JSON value, whose quotes a plain join would lose.
    print("  " + shlex.join(cmd), flush=True)
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
    threshold: float,
    seeds: list[int],
    train_res: dict,
    val_res: dict,
    logs_dir: Path,
    args: argparse.Namespace,
) -> None:
    # Array task id == seed, so a seed subset becomes a sparse array spec and
    # aftercorr still pairs train/validate tasks of the same seed.
    array = _format_array(seeds)
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
    print(f"[{domain}] submit training array ({len(seeds)} runs, seeds {array})")
    train_jobid = _run(train_cmd, args.dry_run)

    val_vars = [
        "ALL",
        f"EXPERIMENT={experiment}",
        f"DOMAIN={domain}",
        f"EXP_TYPE={val_type}",
        f"LOGS_DIR={logs_dir}",
        f"DOMAIN_SETTINGS={args.domain_settings}",
        f"VALIDATION_CONFIG={args.validation_config}",
    ]
    # validate.sh only forwards --threshold when THRESHOLD is set, so leave it
    # unset at the default and keep the plain pipeline unchanged.
    if threshold != DEFAULT_VALIDATION_THRESHOLD:
        val_vars.append(f"THRESHOLD={threshold:g}")
    val_export = ",".join(val_vars)
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
    threshold: float,
    seeds: list[int],
    logs_dir: Path,
    args: argparse.Namespace,
) -> None:
    # Imported here so Slurm submission never needs torch/pymimir installed.
    from validate_runs import run_id_for, validate_run

    domain_settings = _load_yaml(args.domain_settings)
    val_cfg = _load_yaml(args.validation_config)

    for seed in seeds:
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
        print("  " + shlex.join(train_cmd), flush=True)
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
            threshold=threshold,
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
    p.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        metavar="SEED",
        help="Explicit seeds instead of 0..num-runs-1, as single values and/or "
        "inclusive ranges (e.g. '--seeds 3 7-9'). Use to re-run individual "
        "failed seeds. Mutually exclusive with --num-runs.",
    )
    p.add_argument(
        "--sweep",
        default=None,
        choices=sorted(SWEEP_AXES),
        help="Sweep this hyperparameter: submit the full domain x seed grid "
        "once per value, each into sub-experiment <experiment>-<tag><value>. "
        "Default grids — " + _axis_grid_help() + ".",
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
        "--sweep-aux-loss-weight",
        nargs="*",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Alias for '--sweep aux-loss-weight [--sweep-values ALPHA ...]'.",
    )
    p.add_argument(
        "--reuse-default",
        default=None,
        metavar="EXPERIMENT",
        help="Do not submit the sweep point equal to the axis default; take its "
        "results from this existing experiment (e.g. dias-full). Read back by "
        "'collect_results.py --reuse-default'.",
    )
    p.add_argument(
        "--validation-threshold",
        type=float,
        default=None,
        metavar="T",
        help="Schema-derivation threshold used by the validation step. Default: "
        f"{DEFAULT_VALIDATION_THRESHOLD:g}, except "
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

    if args.seeds is not None:
        if args.num_runs is not None:
            raise SystemExit("--seeds and --num-runs are mutually exclusive")
        seeds = _parse_seeds(args.seeds)
    else:
        num_runs = args.num_runs or len(_load_yaml(args.validation_config)["seeds"])
        seeds = list(range(num_runs))
    overlays = []
    exp_overlay = EXPERIMENT_OVERLAYS.get(args.experiment_type)
    if exp_overlay is not None:
        overlays.append(exp_overlay)
    if args.ablation is not None:
        overlays.append(ABLATION_OVERLAYS[args.ablation])
    val_type = VALIDATOR_TYPE[args.experiment_type]
    threshold = (
        args.validation_threshold
        if args.validation_threshold is not None
        else VALIDATION_THRESHOLD.get(
            args.experiment_type, DEFAULT_VALIDATION_THRESHOLD
        )
    )
    if not 0.0 < threshold < 1.0:
        raise SystemExit(
            f"--validation-threshold: expected a probability in (0,1), got "
            f"{threshold:g}"
        )
    extra = []
    for o in args.overrides:
        if "=" not in o:
            raise SystemExit(f"--set expects key=value, got {o!r}")
        if "," in o:
            raise SystemExit(f"--set value must not contain commas: {o!r}")
        extra.append(f"--{o}")

    axis_name = args.sweep
    values = list(args.sweep_values) if args.sweep_values else None
    if args.sweep_aux_loss_weight is not None:
        if axis_name is not None:
            raise SystemExit(
                "--sweep-aux-loss-weight is an alias for --sweep aux-loss-weight; "
                "pass only one of them"
            )
        axis_name = "aux-loss-weight"
        values = list(args.sweep_aux_loss_weight) or None

    # Sweep points: (sub-experiment name, extra Lightning args, reused?). Without
    # a sweep this is the single point (args.experiment, extra, False).
    if axis_name is None:
        if args.sweep_values:
            raise SystemExit("--sweep-values requires --sweep")
        if args.reuse_default:
            raise SystemExit("--reuse-default requires --sweep")
        points = [(args.experiment, extra, False)]
        axis = None
    else:
        axis = SWEEP_AXES[axis_name]
        values = values or list(axis.values)
        if any(o.startswith(f"{axis.param}=") for o in args.overrides):
            raise SystemExit(
                f"--set {axis.param}=... conflicts with --sweep {axis_name}"
            )
        if axis_name == "aux-loss-weight" and args.ablation == "no-aux-loss":
            raise SystemExit(
                "--ablation no-aux-loss disables the auxiliary losses, so "
                "sweeping their weight has no effect"
            )
        if args.reuse_default and axis.default is None:
            raise SystemExit(
                f"--reuse-default: the {axis_name} axis has no reusable default "
                "(the value is tuned per domain)"
            )
        train_cfg = _load_yaml(args.train_config)
        _check_sweep_config(
            axis,
            axis_name,
            domains,
            domain_settings,
            train_cfg,
            args.train_config,
            overlays,
            args.reuse_default,
        )
        # The override is appended last so it beats the per-domain config.
        points = []
        for v in values:
            name = sweep_experiment_name(args.experiment, axis, v, args.reuse_default)
            points.append(
                (
                    name,
                    extra + _sweep_override_args(axis, v, train_cfg),
                    name == args.reuse_default,
                )
            )

    submitted = [p for p in points if not p[2]]
    print(
        f"experiment : {args.experiment}  (type={args.experiment_type}, "
        f"ablation={args.ablation or 'none'}, validator={val_type}, "
        f"threshold={threshold:g})"
    )
    print(
        f"domains    : {len(domains)} x {len(seeds)} runs  "
        f"(seeds={_format_array(seeds)}, backend={args.backend})"
    )
    if axis is not None:
        print(
            f"sweep      : {axis.param} in "
            f"{{{', '.join(f'{v:g}' for v in values)}}}  "
            f"-> {len(submitted) * len(domains) * len(seeds)} runs total"
        )
        if len(submitted) < len(points):
            print(
                f"reuse      : {axis.default:g} not submitted; results come from "
                f"experiment '{args.reuse_default}'"
            )
    for overlay in overlays:
        print(f"overlay    : {overlay}")

    if args.backend == "slurm" and not args.dry_run:
        (PROJECT_ROOT / "logs" / "slurm").mkdir(parents=True, exist_ok=True)

    slurm_cfg = _load_yaml(args.slurm_resources) if args.backend == "slurm" else {}
    val_slurm_cfg = (
        _load_yaml(args.slurm_resources_validation) if args.backend == "slurm" else {}
    )

    for experiment, point_extra, reused in points:
        if reused:
            print(f"\n=== {experiment} (reused, nothing submitted) ===")
            continue
        if len(points) > 1:
            print(f"\n=== sub-experiment {experiment} ===")
        for domain in domains:
            spec = domain_settings[domain]
            domain_config = (PROJECT_ROOT / spec["domain"]).parent / "config.yaml"
            base_args = _base_train_args(
                experiment,
                domain,
                spec,
                args.train_config,
                domain_config,
                overlays,
                args.log_dir,
                point_extra,
            )
            if args.backend == "slurm":
                _submit_domain_slurm(
                    experiment,
                    domain,
                    spec,
                    base_args,
                    val_type,
                    threshold,
                    seeds,
                    _resolve_resources(slurm_cfg, domain),
                    _resolve_resources(val_slurm_cfg, domain),
                    args.log_dir,
                    args,
                )
            else:
                _run_domain_local(
                    experiment,
                    domain,
                    spec,
                    base_args,
                    val_type,
                    threshold,
                    seeds,
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
