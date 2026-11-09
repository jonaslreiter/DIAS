"""What the experiment scripts share: paths, the domain folder layout, validators, seeds and Slurm options. Imports neither torch nor pymimir, so jobs can be submitted without them."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

EXPERIMENTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_ROOT.parent

SLURM_DIR = EXPERIMENTS_ROOT / "slurm"
PIPELINE_CONFIGS = EXPERIMENTS_ROOT / "configs"
RESULTS_DIR = EXPERIMENTS_ROOT / "results"
L1_PDDL_DIR = EXPERIMENTS_ROOT / "l1_pddl"

CONFIGS_DIR = PROJECT_ROOT / "configs"
PDDL_DIR = PROJECT_ROOT / "pddl"
LOGS_DIR = PROJECT_ROOT / "logs"

TRAIN_CONFIG = CONFIGS_DIR / "main.yaml"
VALIDATION_CONFIG = PIPELINE_CONFIGS / "validation.yaml"
SLURM_CONFIG = PIPELINE_CONFIGS / "slurm.yaml"

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "dias")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


# A domain is a folder under pddl/ holding these files.
DOMAIN_FILE = "domain.pddl"
TRAIN_PROBLEM = "train.pddl"
TEST_PROBLEMS = ("test-1.pddl", "test-2.pddl", "test-3.pddl")
DOMAIN_CONFIG = "config.yaml"


def domain_dir(domain: str) -> Path:
    return PDDL_DIR / domain


def domain_pddl(domain: str) -> Path:
    return domain_dir(domain) / DOMAIN_FILE


def train_problem(domain: str) -> Path:
    return domain_dir(domain) / TRAIN_PROBLEM


def validation_problems(domain: str) -> list[Path]:
    return [domain_dir(domain) / name for name in TEST_PROBLEMS]


def domain_config(domain: str) -> Path:
    return domain_dir(domain) / DOMAIN_CONFIG


def available_domains() -> list[str]:
    """Every domain folder under pddl/, in name order."""
    return sorted(
        path.name
        for path in PDDL_DIR.iterdir()
        if path.is_dir() and (path / DOMAIN_FILE).is_file()
    )


def select_domains(requested: list[str] | None) -> list[str]:
    """`requested`, all of them known domains, or every domain."""
    available = available_domains()
    domains = requested or available
    unknown = [d for d in domains if d not in available]
    if unknown:
        raise SystemExit(
            f"unknown domain(s) {unknown}; {PDDL_DIR} holds {', '.join(available)}"
        )
    return domains


def l1_domain(domain: str) -> Path:
    """The domain L1 learned."""
    return L1_PDDL_DIR / f"{domain}.pddl"


def l1_domains() -> list[str]:
    """The domains L1 learned one for."""
    return [domain for domain in available_domains() if l1_domain(domain).is_file()]


def run_command(cmd: list[str], dry_run: bool, capture: bool = True) -> str | None:
    """Print `cmd` and run it unless `dry_run`; with `capture`, return its output."""
    print("  " + shlex.join(cmd), flush=True)
    if dry_run:
        return None
    if not capture:
        subprocess.run(cmd, check=True)
        return None
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


@dataclass(frozen=True)
class Validator:
    """A way of scoring a learned domain; `from_checkpoint` if it is a training run's."""

    name: str
    module: str
    class_name: str
    summary: str
    from_checkpoint: bool


VALIDATORS: dict[str, Validator] = {
    "dias": Validator(
        name="dias",
        module="dias.validation.dias",
        class_name="DiasValidator",
        summary="DIAS: the learned domain of a training run",
        from_checkpoint=True,
    ),
    "mlp": Validator(
        name="mlp",
        module="dias.validation.mlp_effects",
        class_name="MlpEffectsValidator",
        summary="MLP effects: learned preconditions, effects predicted from the state",
        from_checkpoint=True,
    ),
    "l1": Validator(
        name="l1",
        module="dias.validation.pddl",
        class_name="PddlValidator",
        summary="L1: the learned domain in experiments/l1_pddl/",
        from_checkpoint=False,
    ),
}

# The checkpoints of a run: at the end of training, and at the lowest validation loss.
CHECKPOINTS = ("final", "best")

EXPERIMENT_VALIDATOR = {"dias": "dias", "noise": "dias", "mlp": "mlp", "l1": "l1"}
EXPERIMENT_TYPES = tuple(EXPERIMENT_VALIDATOR)


def validator_class(name: str):
    from importlib import import_module

    spec = VALIDATORS[name]
    return getattr(import_module(spec.module), spec.class_name)


def training_resources(slurm_config: dict, experiment_type: str) -> dict:
    """The section of `slurm_config` a training job of `experiment_type` takes."""
    return slurm_config["train-noise" if experiment_type == "noise" else "train"]


def validation_resources(slurm_config: dict, experiment_type: str) -> dict:
    """The section of `slurm_config` a validation job of `experiment_type` takes."""
    section = {"mlp": "validate-mlp", "noise": "validate-noise"}
    return slurm_config[section.get(experiment_type, "validate")]


def resolve_resources(resources: dict, domain: str) -> dict:
    """A domain's resources: the ``defaults`` with its own entry on top."""
    merged = dict(resources.get("defaults", {}))
    merged.update(resources.get(domain) or {})
    return merged


def sbatch_resource_opts(res: dict) -> list[str]:
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


def sbatch_output_opt(log_dir: Path) -> str:
    return f"--output={Path(log_dir).resolve()}/slurm/%x-%A_%a.out"


def prepare_slurm_logs(log_dir: Path) -> None:
    (Path(log_dir) / "slurm").mkdir(parents=True, exist_ok=True)


def parse_seeds(specs: list[str]) -> list[int]:
    """``--seeds`` as a sorted list: single values and inclusive ranges, e.g. ``3 7-9``."""
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


def resolve_seeds(
    args: argparse.Namespace, default: list[int] | None = None
) -> list[int]:
    """``--seeds``, ``0 .. --num-runs - 1``, or `default`, itself defaulting to validation.yaml's seeds."""
    if args.seeds is not None:
        if args.num_runs is not None:
            raise SystemExit("--seeds and --num-runs are mutually exclusive")
        return parse_seeds(args.seeds)
    if args.num_runs is not None:
        if args.num_runs < 1:
            raise SystemExit(f"--num-runs: expected at least 1, got {args.num_runs}")
        return list(range(args.num_runs))
    if default is None:
        return list(range(len(load_yaml(args.validation_config)["seeds"])))
    return sorted(default)


def format_array(seeds: list[int]) -> str:
    """Seeds as an sbatch ``--array`` spec, e.g. ``3,7-9``."""
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


def add_domain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domain folders under pddl/ (default: all).",
    )
    add_seed_arguments(parser, "the seeds validation.yaml lists")


def add_seed_arguments(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help=f"Seeds 0 to N-1. Default: {default}.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        metavar="SEED",
        help="Seeds as single values and inclusive ranges, e.g. '--seeds 0-4' or '--seeds 3 7-9'.",
    )


def add_path_arguments(
    parser: argparse.ArgumentParser, *, train_config: bool = False
) -> None:
    parser.add_argument("--validation-config", type=Path, default=VALIDATION_CONFIG)
    parser.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    if train_config:
        parser.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="Log details.")
    group.add_argument(
        "-q", "--quiet", action="store_true", help="Log warnings and errors only."
    )


def configure_logging(args: argparse.Namespace | None = None) -> None:
    """Log at the level ``-v`` / ``-q`` choose, or ``DIAS_LOG_LEVEL`` names."""
    level = logging.INFO
    if getattr(args, "verbose", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    level = os.environ.get("DIAS_LOG_LEVEL", "").upper() or level
    logging.basicConfig(level=level, format="%(name)s: %(message)s")
