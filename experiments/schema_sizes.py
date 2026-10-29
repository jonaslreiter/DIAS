"""Compare the size of an experiment's learned action schemas with the ground truth: arity, effects and preconditions per action, beside the experiment's precision and recall.

Learned schemas are read as ``precondition_categories.py`` reads them, through ``dias.schema.export.read_learned_schemas``; ground truth is parsed off ``pddl/<domain>/domain.pddl`` with its helpers. Counts are distinct lifted atoms, ``(not (p ...))`` counting as one. Equalities between parameters are left out on both sides: binding is injective by assumption, so a learned domain states the distinctness of its parameters wholesale rather than as learned literals. A type in a ground-truth ``:parameters`` is not a precondition and is not counted, while DIAS has a channel for the type's unary predicate and its learned preconditions include it.

Arity is the number of parameters of a schema, learned or ground truth. The summary counts the ``(domain, action, seed)`` triples whose learned arity, or effect count, equals the ground truth's.

Usage:
    python experiments/schema_sizes.py --experiment dias-names --markdown experiments/results/schema-sizes.md
    python experiments/schema_sizes.py --experiment dias-names --domains satellite --seeds 0-2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # for `import precondition_categories`
from collect_results import _seed_precision_recall  # noqa: E402
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
    """The number of parameters an action block declares, types left out."""
    index = block.find(":parameters")
    if index == -1:
        return 0
    rest = block[index + len(":parameters") :]
    expr, _ = pddl_text.find_balanced_expr(rest, rest.index("("))
    return sum(token.startswith("?") for token in expr.strip("()").split())


def ground_truth_sizes(domain: str) -> dict[str, dict[str, int]]:
    """Per action, as lower-case name, the ground-truth arity, effect count and precondition count."""
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
    """Per action, as lower-case name, one run's learned arity, effect count and precondition count. Raises FileNotFoundError if the run has no checkpoint."""
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
    """One seed's validated precision and recall, if its validation scored every problem."""
    path = log_dir / "validation" / experiment / domain / f"seed{seed}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("status", "ok") != "ok":
        return None
    return _seed_precision_recall(payload.get("per_problem", {}))


def measure_domain(domain: str, seeds: list[int], args: argparse.Namespace) -> dict:
    """One domain's ground truth and, per seed read, its learned sizes and validated precision and recall."""
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
    """Per action: ground truth, and the learned sizes averaged over the seeds read."""
    rows = []
    for action, truth in result["truth"].items():
        runs = [
            sizes[action] for sizes in result["learned"].values() if action in sizes
        ]
        learned = {key: mean(run[key] for run in runs) for key in SIZES}
        rows.append((action, truth, learned))
    return rows


def domain_row(result: dict) -> dict:
    """A domain averaged over its actions: ground truth, learned sizes, mean precision and recall over its seeds."""
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
    """Over every (domain, action, seed) triple, how many learned arities and effect counts equal the ground truth's."""
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


def size_cell(truth: float, learned: float) -> str:
    return f"{truth:.3g} → {learned:.3g}"


def print_report(results: dict) -> None:
    print(
        f"\n{'domain':<16} {'act':>3} {'n':>3}  {'arity GT→L':>14}  {'#eff GT→L':>14}  {'#pre GT→L':>14}  {'P':>6} {'R':>6}"
    )
    for domain, result in results.items():
        row = domain_row(result)
        cells = "  ".join(
            f"{size_cell(row['truth'][k], row['learned'][k]):>14}" for k in SIZES
        )
        print(
            f"{domain:<16} {row['actions']:>3} {row['seeds']:>3}  {cells}  {row['P']:>6.3f} {row['R']:>6.3f}"
        )
    hits = matches(results)
    print(
        f"learned arity equals the ground truth in {hits['arity'][0]} of {hits['arity'][1]} (domain, action, seed) triples, the effect count in {hits['effects'][0]} of {hits['effects'][1]}"
    )


def write_markdown(path: Path, results: dict, args: argparse.Namespace) -> None:
    hits = matches(results)
    lines = [
        f"# Learned vs. ground-truth schema sizes (`{args.experiment}`)",
        "",
        f"Per domain, the arity, effect count and precondition count of its action schemas, averaged over its actions; learned values are also averaged over the seeds read (`n`). `P` and `R` are the mean validated precision and recall over the seeds whose validation scored every problem. Learned arity equals the ground truth in {hits['arity'][0]} of {hits['arity'][1]} (domain, action, seed) triples, and the learned effect count in {hits['effects'][0]} of {hits['effects'][1]}.",
        "",
        "## How the numbers were produced",
        "",
        f"- `python experiments/schema_sizes.py --experiment {args.experiment} --markdown {path}`.",
        f"- Learned schemas were read from the `{args.checkpoint}` checkpoints through `dias.schema.export.read_learned_schemas` at threshold {args.threshold:g}, as `precondition_categories.py` reads them; ground truth was parsed from `pddl/<domain>/domain.pddl`.",
        "- Distinct lifted atoms, `(not (p ...))` counting as one. Equalities between parameters are left out on both sides, since binding is injective by assumption. A type in a ground-truth `:parameters` is not counted as a precondition, while DIAS learns the type's unary predicate as one.",
        "",
        "| domain | actions | n | arity (GT → learned) | #eff (GT → learned) | #pre (GT → learned) | P | R |",
        "|---|--:|--:|---|---|---|--:|--:|",
    ]
    for domain, result in results.items():
        row = domain_row(result)
        cells = " | ".join(size_cell(row["truth"][k], row["learned"][k]) for k in SIZES)
        lines.append(
            f"| {domain} | {row['actions']} | {row['seeds']} | {cells} | {row['P']:.3f} | {row['R']:.3f} |"
        )
    lines += [
        "",
        "## Per action",
        "",
        "| domain | action | arity (GT → learned) | #eff (GT → learned) | #pre (GT → learned) |",
        "|---|---|---|---|---|",
    ]
    for domain, result in results.items():
        for i, (action, truth, learned) in enumerate(action_rows(result)):
            cells = " | ".join(size_cell(truth[k], learned[k]) for k in SIZES)
            lines.append(f"| {domain if i == 0 else ''} | {action} | {cells} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        required=True,
        help="Experiment whose runs to read, the prefix of the run ids.",
    )
    add_domain_arguments(p)
    p.add_argument(
        "--validation-config",
        type=Path,
        default=VALIDATION_CONFIG,
        help="The validation.yaml whose seeds are read without --seeds or --num-runs.",
    )
    p.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold handed to derive_schema (default: 0.5, schema_threshold's default).",
    )
    p.add_argument(
        "--checkpoint",
        default="final",
        choices=CHECKPOINTS,
        help="Which of each run's checkpoints to read (default: final, the one validation scores).",
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
    print_report(results)
    if args.markdown:
        write_markdown(args.markdown, results, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
