"""Validate a single DIAS checkpoint on a single PDDL problem.

Usage:
    python scripts/validation/run_dias.py \
        --model-id <wandb-run-id> \
        --logs-dir logs \
        --problem pddl_files/satellite/train.pddl

Outputs precision/recall/accuracy on the action-grounding task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dias.validation.dias_validate import ModelValidation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-id",
        required=True,
        help="Wandb run id (used to locate the checkpoint under <logs-dir>).",
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing wandb-style run subdirs.",
    )
    p.add_argument(
        "--problem", type=Path, required=True, help="Path to the PDDL problem file."
    )
    p.add_argument("--num-samples", type=int, default=250)
    p.add_argument("--sample-population", type=int, default=500)
    p.add_argument(
        "--out", type=Path, default=None, help="Optional path to write the result JSON."
    )
    args = p.parse_args()

    v = ModelValidation(
        model_id=args.model_id,
        logs_path_abs=str(args.logs_dir.resolve()),
        problem_path=str(args.problem.resolve()),
    )
    v.setup(num_samples=args.num_samples, sample_population=args.sample_population)
    result = v.validate()
    print(json.dumps(result, indent=2, default=str))

    if args.out:
        args.out.write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
