"""Score one learned domain on one problem and print the result: a training run's (``--validator dias`` or ``mlp``, given by ``--model-id``) or any learned PDDL domain (``--validator l1``). The test states are drawn as validation.yaml says.

Examples:
    python experiments/validate_checkpoint.py --validator dias --model-id <run-id> --problem pddl/satellite/test-1.pddl
    python experiments/validate_checkpoint.py --validator l1 --original-domain pddl/satellite/domain.pddl --learned-domain experiments/l1_pddl/satellite.pddl --problem pddl/satellite/test-1.pddl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    """The validator's arguments."""
    validator = VALIDATORS[args.validator]
    common = {
        "problem_path": str(args.problem.resolve()),
        "force_injective": args.force_injective,
        "num_workers": args.num_workers,
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
    if args.validator == "mlp":
        kwargs["batch_size"] = args.batch_size
    return kwargs


SAMPLING_SETTINGS = ("num_samples", "walk_length", "min_walks")


def _sampling(args: argparse.Namespace, configured: dict):
    """validation.yaml's sampling with the settings given on the command line."""
    from dias.validation.successors import sampling_from_record

    record = dict(configured)
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
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--walk-length", type=int, default=None)
    p.add_argument("--min-walks", type=int, default=None)
    p.add_argument(
        "--sample-seed",
        type=int,
        default=defaults.get("sample_seed"),
        help="The seed of the state draw. Default: drawn and reported.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Derive the learned domain at this threshold instead of scoring the one beside the checkpoint (dias and mlp).",
    )
    p.add_argument(
        "--checkpoint",
        default="final",
        choices=CHECKPOINTS,
        help="The checkpoint to score: final (default) or best (dias and mlp).",
    )
    p.add_argument(
        "--force-injective",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("force_injective", False)),
        help="Replace what the learned domain says about equality with the distinctness of every two parameters. Default: validation.yaml's.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Worker processes to score states in.",
    )
    p.add_argument("--batch-size", type=int, default=4096, help="mlp only.")
    p.add_argument("--out", type=Path, default=None, help="Write the result JSON here.")
    add_logging_arguments(p)
    args = p.parse_args()
    configure_logging(args)

    sampling = _sampling(args, defaults["sampling"])
    validation = validator_class(args.validator)(**_construction_kwargs(args))
    validation.setup(sampling, seed=args.sample_seed)
    result = validation.validate()
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
