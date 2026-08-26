"""What every script in the DIAS experiment pipeline shares.

The launchers, the per-run validation step, the result collector and the
scaling study all need the same handful of things: where the repository is,
how to read a YAML config, how to turn resource settings into sbatch flags,
which seeds a run covers, and which validator a name refers to. Each of those
is declared here once, so the scripts cannot drift apart on any of them --
which they had, down to spelling the same option ``--log-dir`` in one place and
``--logs-dir`` in another.

Nothing here imports torch or pymimir: submitting jobs must work on a login
node with nothing but PyYAML installed.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"
PDDL_DIR = PROJECT_ROOT / "pddl"

TRAIN_CONFIG = CONFIGS_DIR / "main.yaml"
VALIDATION_CONFIG = CONFIGS_DIR / "validation.yaml"
SLURM_RESOURCES = CONFIGS_DIR / "slurm-resources.yaml"
SLURM_RESOURCES_VALIDATION = CONFIGS_DIR / "slurm-resources-validation.yaml"
LOGS_DIR = PROJECT_ROOT / "logs"

# Which W&B account and project the runs live in. An unset entity means
# "whichever account is logged in", so a clone needs no edit here.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "dias")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


# ── where a domain's files are ────────────────────────────────────────────────
#
# A domain is a folder under pddl/ with standardized filenames in it, and the
# folder name is the domain name. There is no manifest: a list repeating what
# the layout already says is a second source of truth, and the two drift.

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

    ``capture`` is what an sbatch call needs -- the job id comes back on
    stdout. A long-running child such as a training run wants the opposite:
    with ``capture=False`` it inherits this terminal, so its progress appears
    while it happens rather than being swallowed and thrown away at the end.
    """
    # shlex.join keeps the printed command copy-pasteable: sweep overrides can
    # carry a JSON value, whose quotes a plain join would lose.
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

    ``binarizes_schema`` says whether a derivation threshold applies: only a
    model that emits lifted precondition/effect tensors has anything to
    binarize. ``from_checkpoint`` separates the two that read a trained model
    from L1, which is handed a learned PDDL domain someone else produced.
    """

    name: str
    module: str
    summary: str
    binarizes_schema: bool
    from_checkpoint: bool


VALIDATORS: dict[str, Validator] = {
    "dias": Validator(
        name="dias",
        module="dias.validation.dias_validate",
        summary="DIAS: lifted schemas read off a trained checkpoint",
        binarizes_schema=True,
        from_checkpoint=True,
    ),
    "mlp": Validator(
        name="mlp",
        module="dias.validation.mlp_baseline",
        summary="MLP baseline: effects predicted from the state, no lifted schema",
        binarizes_schema=False,
        from_checkpoint=True,
    ),
    "l1": Validator(
        name="l1",
        module="dias.validation.l1",
        summary="L1 baseline: a learned PDDL domain produced outside this repo",
        binarizes_schema=False,
        from_checkpoint=False,
    ),
}

# Which validator each experiment type scores with. Noise is a training-side
# setting, so its runs are still DIAS models.
EXPERIMENT_VALIDATOR = {"dias": "dias", "noise": "dias", "mlp": "mlp"}
EXPERIMENT_TYPES = tuple(EXPERIMENT_VALIDATOR)


def validator_class(name: str):
    """The ``ModelValidation`` class of one validator, imported on demand."""
    from importlib import import_module

    return import_module(VALIDATORS[name].module).ModelValidation


# ── seeds and Slurm ───────────────────────────────────────────────────────────


def parse_seeds(specs: list[str]) -> list[int]:
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


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    """The seeds a launch covers: an explicit subset, or ``0 .. num_runs-1``.

    Without either flag the count comes from ``configs/validation.yaml``, which
    is what the paper's ten seeds per domain are declared in.
    """
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
    """Where an array task's stdout goes, as an absolute path.

    The ``#SBATCH --output`` line inside each script is relative, so it only
    resolves when sbatch is invoked from the repository root. Naming it on the
    command line instead makes a submission from anywhere work, and makes
    ``--log-dir`` apply to the job output too.
    """
    return f"--output={Path(log_dir).resolve()}/slurm/%x-%A_%a.out"


def prepare_slurm_logs(log_dir: Path) -> None:
    """Create the directory the ``#SBATCH --output`` patterns write into.

    Under ``--log-dir``, not the repository's own ``logs/``: sbatch refuses a
    job whose output directory does not exist, and it used to be created in the
    wrong place whenever the logs were redirected elsewhere.
    """
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
        help="Explicit seeds instead of 0..num-runs-1, as single values and/or "
        "inclusive ranges (e.g. '--seeds 3 7-9'). Use to re-run individual "
        "failed seeds. Mutually exclusive with --num-runs.",
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
    """Install the one log handler a script gets.

    Only entry points may do this -- the library reports through ``logging``
    and configures nothing, which is what keeps ``dias`` importable inside
    someone else's application. ``DIAS_LOG_LEVEL`` overrides the default for a
    script whose CLI has no flag for it, such as a Slurm job.
    """
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
