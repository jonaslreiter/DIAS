"""Validate a single MLP-baseline checkpoint on a single PDDL problem.

Usage:
    python scripts/validation/run_mlp_baseline.py \
        --model-id <wandb-run-id> \
        --logs-dir logs \
        --problem pddl_files/satellite/train.pddl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dias.validation.mlp_baseline import ModelValidation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--problem", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=250)
    p.add_argument("--sample-population", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    v = ModelValidation(
        model_id=args.model_id,
        logs_path_abs=str(args.logs_dir.resolve()),
        problem_path=str(args.problem.resolve()),
    )
    v.setup(num_samples=args.num_samples, sample_population=args.sample_population)
    result = v.validate(num_workers=args.num_workers, batch_size=args.batch_size)
    print(json.dumps(result, indent=2, default=str))

    if args.out:
        args.out.write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
