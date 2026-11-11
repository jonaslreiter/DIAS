"""Compare the arity, effect count and precondition count of an experiment's learned action schemas with the original ones, per domain and action, beside the experiment's precision and recall.

Schemas are read as in ``precondition_categories.py``, equalities left out on both sides. The match counts are over (domain, action, seed).

Example:
    python experiments/schema_sizes.py --experiment dias-names --markdown experiments/results/schema-sizes.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_results import seed_precision_recall  # noqa: E402
from dias.pddl import text as pddl_text
from pipeline import (  # noqa: E402
    CHECKPOINTS,
    LOGS_DIR,
    VALIDATION_CONFIG,
    add_domain_arguments,
    add_logging_arguments,
    configure_logging,
    domain_pddl,
    resolve_seeds,
    select_domains,
)
from precondition_categories import (  # noqa: E402
    EQUALITY_NAMES,
    atoms_of,
    section,
    strip_comments,
)
from validate_runs import run_id_for  # noqa: E402

logger = logging.getLogger(__name__)

SIZES = ("arity", "effects", "preconditions")


def parameter_count(block: str) -> int:
    """The number of parameters of an action block."""
    index = block.find(":parameters")
    if index == -1:
        return 0
    rest = block[index + len(":parameters") :]
    expr, _ = pddl_text.find_balanced_expr(rest, rest.index("("))
    return sum(token.startswith("?") for token in expr.strip("()").split())


def ground_truth_sizes(domain: str) -> dict[str, dict[str, int]]:
    """Per action, the original arity, effect count and precondition count."""
    text = strip_comments(domain_pddl(domain).read_text())
    sizes = {}
    for block in pddl_text.extract_action_blocks(text):
        action = block.split()[1].strip("()").lower()
        preconditions = {
            atom
            for atom in atoms_of(section(block, ":precondition"))
            if atom[0] not in EQUALITY_NAMES
        }
        sizes[action] = {
            "arity": parameter_count(block),
            "effects": len(set(atoms_of(section(block, ":effect")))),
            "preconditions": len(preconditions),
        }
    return sizes


def learned_sizes(
    run_id: str, log_dir: Path, threshold: float, checkpoint: str
) -> dict[str, dict[str, int]]:
    """Per action, a run's learned arity, effect count and precondition count."""
    from dias.schema.export import read_learned_schemas
    from dias.validation.checkpoint import load_run

    run = load_run(run_id, str(log_dir), checkpoint)
    transpiler = read_learned_schemas(
        run.model, run.schema, run.datamodule_hparams, threshold
    )
    return {
        schema.name.lower(): {
            "arity": len(schema.parameters),
            "effects": len(schema.effects),
            "preconditions": sum(
                atom.predicate_name not in EQUALITY_NAMES
                for atom in schema.preconditions
            ),
        }
        for schema in transpiler.action_schemas
    }


def precision_recall(
    experiment: str, domain: str, seed: int, log_dir: Path
) -> tuple[float, float] | None:
    """A seed's precision and recall, if its validation scored every problem."""
    path = log_dir / "validation" / experiment / domain / f"seed{seed}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload["status"] != "ok":
        return None
    return seed_precision_recall(payload["per_problem"])


def measure_domain(domain: str, seeds: list[int], args: argparse.Namespace) -> dict:
    """A domain's original sizes and, per seed, its learned sizes, precision and recall."""
    truth = ground_truth_sizes(domain)
    learned, rates = {}, {}
    for seed in seeds:
        run_id = run_id_for(args.experiment, domain, seed)
        try:
            learned[seed] = learned_sizes(
                run_id, args.log_dir, args.threshold, args.checkpoint
            )
        except FileNotFoundError as error:
            logger.warning("%s: %s", run_id, error)
            continue
        missing = set(truth) ^ set(learned[seed])
        if missing:
            logger.warning(
                "%s: actions differ from the ground truth: %s", run_id, sorted(missing)
            )
        pr = precision_recall(args.experiment, domain, seed, args.log_dir)
        if pr is not None:
            rates[seed] = pr
    return {"truth": truth, "learned": learned, "rates": rates}


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def action_rows(result: dict) -> list[tuple[str, dict[str, float], dict[str, float]]]:
    """Per action, the original sizes and the learned ones averaged over seeds."""
    rows = []
    for action, truth in result["truth"].items():
        runs = [
            sizes[action] for sizes in result["learned"].values() if action in sizes
        ]
        learned = {key: mean(run[key] for run in runs) for key in SIZES}
        rows.append((action, truth, learned))
    return rows


def domain_row(result: dict) -> dict:
    """A domain's sizes averaged over its actions, and its mean precision and recall."""
    rows = action_rows(result)
    rates = list(result["rates"].values())
    return {
        "actions": len(rows),
        "seeds": len(result["learned"]),
        "truth": {key: mean(t[key] for _, t, _ in rows) for key in SIZES},
        "learned": {key: mean(l[key] for _, _, l in rows) for key in SIZES},
        "P": mean(p for p, _ in rates),
        "R": mean(r for _, r in rates),
        "rated": len(rates),
    }


def matches(results: dict) -> dict[str, tuple[int, int]]:
    """How many (domain, action, seed) have the original arity and effect count, out of how many."""
    counts = {"arity": [0, 0], "effects": [0, 0]}
    for result in results.values():
        for sizes in result["learned"].values():
            for action, truth in result["truth"].items():
                if action not in sizes:
                    continue
                for key in counts:
                    counts[key][0] += sizes[action][key] == truth[key]
                    counts[key][1] += 1
    return {key: tuple(value) for key, value in counts.items()}


SIZE_HEADER = [f"{key} {side}" for key in SIZES for side in ("GT", "L")]


def tables(results: dict) -> list[tuple[str, list[str], list[list[str]]]]:
    """Per domain, per action, and the match counts, each as (heading, header, rows)."""
    domains = []
    actions = []
    for domain, result in results.items():
        row = domain_row(result)
        sizes = [f"{row[side][k]:.3g}" for k in SIZES for side in ("truth", "learned")]
        domains.append(
            [
                domain,
                str(row["actions"]),
                str(row["seeds"]),
                *sizes,
                f"{row['P']:.3f}",
                f"{row['R']:.3f}",
            ]
        )
        for i, (action, truth, learned) in enumerate(action_rows(result)):
            sizes = [f"{v[k]:.3g}" for k in SIZES for v in (truth, learned)]
            actions.append([domain if i == 0 else "", action, *sizes])
    hits = matches(results)
    return [
        ("Per domain", ["domain", "actions", "n", *SIZE_HEADER, "P", "R"], domains),
        ("Per action", ["domain", "action", *SIZE_HEADER], actions),
        (
            "Learned as the original",
            ["", "matching", "total"],
            [[key, str(n), str(total)] for key, (n, total) in hits.items()],
        ),
    ]


def write_markdown(path: Path, experiment: str, tables_) -> None:
    lines = [f"# Schema sizes, {experiment}", ""]
    for heading, header, rows in tables_:
        lines += [
            f"## {heading}",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
        ]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        required=True,
        help="Experiment whose runs to read.",
    )
    add_domain_arguments(p)
    p.add_argument(
        "--validation-config",
        type=Path,
        default=VALIDATION_CONFIG,
        help="Its seeds are the default.",
    )
    p.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Derivation threshold (default: 0.5).",
    )
    p.add_argument(
        "--checkpoint",
        default="final",
        choices=CHECKPOINTS,
        help="The checkpoint to read: final (default) or best.",
    )
    p.add_argument(
        "--markdown", type=Path, default=None, help="Also write the tables here."
    )
    add_logging_arguments(p)
    args = p.parse_args()
    configure_logging(args)

    seeds = resolve_seeds(args)
    results = {}
    for domain in select_domains(args.domains):
        result = measure_domain(domain, seeds, args)
        if not result["learned"]:
            logger.warning("%s: no checkpoint read, skipped", domain)
            continue
        results[domain] = result
    if not results:
        raise SystemExit(f"no runs of {args.experiment} found under {args.log_dir}")
    tables_ = tables(results)
    for heading, header, rows in tables_:
        print(f"\n{heading}")
        for row in [header, *rows]:
            print("\t".join(row))
    if args.markdown:
        write_markdown(args.markdown, args.experiment, tables_)
    return 0


if __name__ == "__main__":
    sys.exit(main())
