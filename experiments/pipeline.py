"""What every script in the DIAS experiment pipeline shares: the paths, the domain-folder layout, the sbatch flag builders, the seed parsing and the validator table.

Nothing here imports torch or pymimir: submitting jobs must work on a login node with nothing but PyYAML.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

# ── where everything is ───────────────────────────────────────────────────────
#
# The launchers, Slurm scripts, orchestration configs and collected tables are this file's siblings. At the repository root sits what is not about running a sweep: the Lightning configs `dias fit` reads (CONFIGS_DIR) and the domains (PDDL_DIR).

EXPERIMENTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_ROOT.parent

SLURM_DIR = EXPERIMENTS_ROOT / "slurm"
PIPELINE_CONFIGS = EXPERIMENTS_ROOT / "configs"
RESULTS_DIR = EXPERIMENTS_ROOT / "results"
L1_PDDL_DIR = EXPERIMENTS_ROOT / "l1_pddl"

CONFIGS_DIR = PROJECT_ROOT / "configs"
PDDL_DIR = PROJECT_ROOT / "pddl"

TRAIN_CONFIG = CONFIGS_DIR / "main.yaml"
VALIDATION_CONFIG = PIPELINE_CONFIGS / "validation.yaml"
SLURM_RESOURCES = PIPELINE_CONFIGS / "slurm-resources.yaml"
SLURM_RESOURCES_VALIDATION = PIPELINE_CONFIGS / "slurm-resources-validation.yaml"
SLURM_RESOURCES_VALIDATION_MLP = (
    PIPELINE_CONFIGS / "slurm-resources-validation-mlp.yaml"
)
LOGS_DIR = PROJECT_ROOT / "logs"

# An unset entity means whichever account is logged in, so a clone needs no edit here.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "dias")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


# ── where a domain's files are ────────────────────────────────────────────────
#
# A domain is a folder under pddl/, named for the domain. There is no manifest; these names are the layout.

DOMAIN_FILE = "domain.pddl"
TRAIN_PROBLEM = "train.pddl"
# Held out from training, and validated on in this order.
TEST_PROBLEMS = ("test-1.pddl", "test-2.pddl", "test-3.pddl")
DOMAIN_CONFIG = "config.yaml"


def domain_dir(domain: str) -> Path:
    return PDDL_DIR / domain


def domain_pddl(domain: str) -> Path:
    return domain_dir(domain) / DOMAIN_FILE


def train_problem(domain: str) -> Path:
    """The single instance a domain's transitions are sampled from."""
    return domain_dir(domain) / TRAIN_PROBLEM


def validation_problems(domain: str) -> list[Path]:
    """The held-out instances every validator scores against."""
    return [domain_dir(domain) / name for name in TEST_PROBLEMS]


def domain_config(domain: str) -> Path:
    """The Lightning partial config composed after configs/main.yaml."""
    return domain_dir(domain) / DOMAIN_CONFIG


def available_domains() -> list[str]:
    """Every domain folder under pddl/, in name order."""
    return sorted(
        path.name
        for path in PDDL_DIR.iterdir()
        if path.is_dir() and (path / DOMAIN_FILE).is_file()
    )


def run_command(cmd: list[str], dry_run: bool, capture: bool = True) -> str | None:
    """Print a command and run it, or only print it.

    sbatch needs ``capture``, for the job id on stdout; a training run wants the opposite, so its progress appears while it happens.
    """
    # shlex.join keeps the printed command copy-pasteable.
    print("  " + shlex.join(cmd), flush=True)
    if dry_run:
        return None
    if not capture:
        subprocess.run(cmd, check=True)
        return None
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


# ── validators ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Validator:
    """One way of scoring a learned action model against the real domain.

    ``from_checkpoint``: the learned domain is a training run's, so it can be extracted anew at another threshold; L1's is a PDDL file. ``grounds_in_parallel``: whether the class takes ``num_workers``, which only the MLP-effects one does, its per-state grounding being the one that pays for a pool.
    """

    name: str
    module: str
    class_name: str
    summary: str
    from_checkpoint: bool
    grounds_in_parallel: bool = False


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
        grounds_in_parallel=True,
    ),
    "l1": Validator(
        name="l1",
        module="dias.validation.pddl",
        class_name="PddlValidator",
        summary="L1 baseline: the learned domain in experiments/l1_pddl/",
        from_checkpoint=False,
    ),
}

# The checkpoints main.py saves, by name: the model as training left it, and the one with the lowest validation loss. Validation reads the first unless told otherwise.
CHECKPOINTS = ("final", "best")

# Noise is a training-side setting, so its runs are still DIAS models.
EXPERIMENT_VALIDATOR = {"dias": "dias", "noise": "dias", "mlp": "mlp", "l1": "l1"}
EXPERIMENT_TYPES = tuple(EXPERIMENT_VALIDATOR)


def validation_resources_file(validator: str) -> Path:
    """The resources a validator's array asks for unless told otherwise: the MLP-effects validator grounds every learned precondition, which costs far more than scoring a learned domain, so it has a file of its own."""
    return (
        SLURM_RESOURCES_VALIDATION_MLP
        if validator == "mlp"
        else SLURM_RESOURCES_VALIDATION
    )


def l1_domain(domain: str) -> Path:
    """The domain L1 learned for the paper's comparison."""
    return L1_PDDL_DIR / f"{domain}.pddl"


def validator_class(name: str):
    """The validator class one entry names, imported on demand."""
    from importlib import import_module

    spec = VALIDATORS[name]
    return getattr(import_module(spec.module), spec.class_name)


# ── seeds and Slurm ───────────────────────────────────────────────────────────


def parse_seeds(specs: list[str]) -> list[int]:
    """Parse ``--seeds`` into a sorted, deduplicated list. Single values and inclusive ranges, space- or comma-separated: ``3 7-9`` gives ``[3, 7, 8, 9]``."""
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


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    """The seeds a launch covers: an explicit subset, or ``0 .. num_runs-1``, counted from validation.yaml."""
    if args.seeds is not None:
        if args.num_runs is not None:
            raise SystemExit("--seeds and --num-runs are mutually exclusive")
        return parse_seeds(args.seeds)
    num_runs = args.num_runs or len(load_yaml(args.validation_config)["seeds"])
    return list(range(num_runs))


def format_array(seeds: list[int]) -> str:
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


def resolve_resources(slurm_cfg: dict, domain: str) -> dict:
    """A domain's Slurm resources: the ``defaults`` block plus its own entry."""
    merged = dict(slurm_cfg.get("defaults", {}))
    merged.update(slurm_cfg.get(domain) or {})
    return merged


def sbatch_resource_opts(res: dict) -> list[str]:
    """The sbatch flags a resolved resource block asks for. This is what the fields of the two slurm-resources files mean."""
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


def allocated_cpus() -> int:
    """How many CPUs this process may actually use: the affinity mask a Slurm job is pinned to, or the machine's count off the cluster."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def sbatch_output_opt(log_dir: Path) -> str:
    """Where an array task's stdout goes, absolute: the scripts' own ``#SBATCH --output`` is relative and would only resolve from the repository root."""
    return f"--output={Path(log_dir).resolve()}/slurm/%x-%A_%a.out"


def prepare_slurm_logs(log_dir: Path) -> None:
    """Create the directory ``#SBATCH --output`` writes into, under ``--log-dir``: sbatch refuses a job whose output directory is missing."""
    (Path(log_dir) / "slurm").mkdir(parents=True, exist_ok=True)


# ── shared arguments ──────────────────────────────────────────────────────────


def add_domain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domain folders under pddl/ to run (default: all of them).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Runs (seeds 0..N-1) per domain. Default: len(validation.yaml seeds).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        metavar="SEED",
        help="Explicit seeds instead of 0..num-runs-1, as single values and/or inclusive ranges (e.g. '--seeds 3 7-9'). Use to re-run individual failed seeds. Mutually exclusive with --num-runs.",
    )


def add_path_arguments(
    parser: argparse.ArgumentParser, *, train_config: bool = False
) -> None:
    """The config / logs paths every pipeline script accepts."""
    parser.add_argument("--validation-config", type=Path, default=VALIDATION_CONFIG)
    parser.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    if train_config:
        parser.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """``-v`` / ``-q``, which every script's entry point resolves the same way."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Report per-step detail as well as state changes.",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Report only warnings and errors.",
    )


def configure_logging(args: argparse.Namespace | None = None) -> None:
    """Install the one log handler a script gets. ``DIAS_LOG_LEVEL`` overrides the flags, for a script whose command line is fixed by a Slurm job."""
    level = logging.INFO
    if getattr(args, "verbose", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    level = os.environ.get("DIAS_LOG_LEVEL", "").upper() or level
    logging.basicConfig(level=level, format="%(name)s: %(message)s")


def select_domains(requested: list[str] | None) -> list[str]:
    """Validate ``--domains`` against pddl/; default to every domain there."""
    available = available_domains()
    domains = requested or available
    unknown = [d for d in domains if d not in available]
    if unknown:
        raise SystemExit(
            f"unknown domain(s) {unknown}; {PDDL_DIR} holds {', '.join(available)}"
        )
    return domains
