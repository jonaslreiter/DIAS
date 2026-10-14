"""Validate one learned action model on one PDDL problem and print the result, where ``experiments/validate_runs.py`` walks a whole grid.

``--validator`` picks what is scored: ``dias`` and ``mlp`` the learned domain of a training run, given as ``--model-id``, which ``--threshold`` extracts anew; ``l1`` a learned PDDL domain from another system, given as ``--original-domain`` / ``--learned-domain``. Sampling defaults come from ``experiments/configs/validation.yaml``.

Examples
--------
    python experiments/validate_checkpoint.py --validator dias --model-id <run-id> --problem pddl/satellite/test-1.pddl
    python experiments/validate_checkpoint.py --validator l1 --original-domain pddl/satellite/domain.pddl --learned-domain experiments/l1_pddl/satellite.pddl --problem pddl/satellite/test-1.pddl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import pipeline`

from pipeline import configure_logging  # noqa: E402
from pipeline import (
    CHECKPOINTS,
    LOGS_DIR,
    VALIDATION_CONFIG,
    VALIDATORS,
    add_logging_arguments,
    load_yaml,
    validator_class,
)


def _validator_help() -> str:
    return "  ".join(f"{v.name}: {v.summary}" for v in VALIDATORS.values())


def _construction_kwargs(args: argparse.Namespace) -> dict:
    """The arguments that differ between a training run and a learned domain."""
    validator = VALIDATORS[args.validator]
    common = {
        "problem_path": str(args.problem.resolve()),
        "force_injective": args.force_injective,
    }
    if not validator.from_checkpoint:
        if not (args.original_domain and args.learned_domain):
            raise SystemExit(
                f"--validator {args.validator} scores a learned PDDL domain: pass --original-domain and --learned-domain"
            )
        if args.threshold is not None or args.checkpoint != "final":
            raise SystemExit(
                f"--threshold and --checkpoint pick a training run's learned domain; {validator.summary}"
            )
        return {
            **common,
            "original_domain_path": str(args.original_domain.resolve()),
            "learned_domain_path": str(args.learned_domain.resolve()),
        }
    if not args.model_id:
        raise SystemExit(f"--validator {args.validator} needs --model-id")
    kwargs = {
        **common,
        "model_id": args.model_id,
        "logs_path": str(args.log_dir.resolve()),
        "threshold": args.threshold,
        "checkpoint": args.checkpoint,
    }
    if validator.grounds_in_parallel:
        kwargs |= {"num_workers": args.num_workers, "batch_size": args.batch_size}
    return kwargs


# Command-line settings of either sampling strategy; one the chosen strategy lacks is refused.
SAMPLING_SETTINGS = ("num_samples", "walk_length", "min_walks", "sample_population")


def _sampling(args: argparse.Namespace, configured: dict):
    """validation.yaml's sampling, or the defaults of the strategy `--sampling` names instead, with the settings given on the command line."""
    from dias.validation.successors import sampling_from_record

    record = dict(configured)
    if args.sampling not in (None, record["strategy"]):
        record = {"strategy": args.sampling}
    for setting in SAMPLING_SETTINGS:
        if getattr(args, setting) is not None:
            record[setting] = getattr(args, setting)
    try:
        return sampling_from_record(record)
    except ValueError as error:
        raise SystemExit(str(error))


def main() -> int:
    defaults = load_yaml(VALIDATION_CONFIG)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--validator",
        default="dias",
        choices=sorted(VALIDATORS),
        help=_validator_help(),
    )
    p.add_argument("--problem", type=Path, required=True, help="The PDDL problem.")
    p.add_argument(
        "--model-id",
        default=None,
        help="Run id locating the checkpoint under <log-dir>/dias/<id>/checkpoints.",
    )
    p.add_argument("--original-domain", type=Path, default=None, help="l1 only.")
    p.add_argument("--learned-domain", type=Path, default=None, help="l1 only.")
    p.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    # Unset, these are validation.yaml's, so a single-problem check samples what the sweep does.
    p.add_argument(
        "--sampling",
        choices=("walks", "bfs"),
        default=None,
        help="How states are drawn: random walks (validation.yaml's default), or bfs, the paper's draw from the first states a breadth-first search reaches. Naming the other strategy starts from its own defaults.",
    )
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--walk-length", type=int, default=None, help="walks only.")
    p.add_argument("--min-walks", type=int, default=None, help="walks only.")
    p.add_argument("--sample-population", type=int, default=None, help="bfs only.")
    p.add_argument(
        "--sample-seed",
        type=int,
        default=defaults.get("sample_seed"),
        help="Fix the state draw. Unset draws freshly; either way the seed is reported and reproduces the states.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Extract the run's learned domain anew at this binarization threshold, instead of scoring the one written beside its checkpoint (dias and mlp).",
    )
    p.add_argument(
        "--checkpoint",
        default="final",
        choices=CHECKPOINTS,
        help="Which of the run's checkpoints to score: final, the model as training left it, or best, the one with the lowest validation loss (dias and mlp).",
    )
    p.add_argument(
        "--force-injective",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("force_injective", False)),
        help="Replace whatever the learned domain says about equality with distinct parameters in every schema. Default from validation.yaml.",
    )
    p.add_argument("--num-workers", type=int, default=1, help="mlp only.")
    p.add_argument("--batch-size", type=int, default=4096, help="mlp only.")
    p.add_argument("--out", type=Path, default=None, help="Write the result JSON here.")
    add_logging_arguments(p)
    args = p.parse_args()
    configure_logging(args)

    sampling = _sampling(args, defaults["sampling"])
    validation = validator_class(args.validator)(**_construction_kwargs(args))
    validation.setup(sampling, seed=args.sample_seed)
    result = validation.validate()
    # Passing this back as --sample-seed, with the same sampling, scores the next run on these exact states.
    result = {
        "sample_seed": validation.sample_seed,
        "sampling": sampling.record(),
        "threshold": validation.threshold,
        "force_injective": args.force_injective,
        **result,
    }

    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
