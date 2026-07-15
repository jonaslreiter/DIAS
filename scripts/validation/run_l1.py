"""Validate an L1 learned PDDL domain on a problem.

Unlike the DIAS / MLP runners this one does not load a torch checkpoint —
L1 produces a *symbolic* PDDL domain file directly. We
compare it to the ground-truth (original) domain on a given problem.

Usage:
    python scripts/validation/run_l1.py \
        --original-domain pddl_files/satellite/domain.pddl \
        --learned-domain  path/to/l1_learned/satellite.pddl \
        --problem         pddl_files/satellite/train.pddl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dias.validation.l1 import ModelValidation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original-domain", type=Path, required=True)
    p.add_argument("--learned-domain", type=Path, required=True)
    p.add_argument("--problem", type=Path, required=True)
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Used for any auxiliary outputs the validator writes.",
    )
    p.add_argument("--num-samples", type=int, default=250)
    p.add_argument("--sample-population", type=int, default=500)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    v = ModelValidation(
        original_domain_path=str(args.original_domain.resolve()),
        learned_domain_path=str(args.learned_domain.resolve()),
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
