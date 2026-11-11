"""Count the precondition literals of an experiment's learned schemas and of the original schemas by class, per domain: equalities, unary and binary static literals, and positive and negative fluent literals.

Learned schemas are derived from each run's checkpoint, without the distinctness of parameters the learned domain adds. The original ones are parsed off ``pddl/<domain>/domain.pddl``, without the parameter types. A static predicate is one no action changes. Literals are counted as distinct lifted atoms, learned ones as a mean over seeds.

Example:
    python experiments/precondition_categories.py --experiment dias-names --markdown experiments/results/precondition-categories.md
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from dias.pddl import text as pddl_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
from validate_runs import run_id_for  # noqa: E402

logger = logging.getLogger(__name__)

EQUALITY_NAMES = frozenset({"=", *pddl_text.EQUALITY_PREDICATE_NAMES})

KINDS = ("eq", "stat1", "stat2", "flu")
KEYS = tuple(f"{kind}{sign}" for kind in KINDS for sign in "+-")
# The reported classes: equalities, unary and binary static literals, positive and negative fluent literals.
CLASSES = {
    "eq": ("eq+", "eq-"),
    "stat1": ("stat1+", "stat1-"),
    "stat2": ("stat2+", "stat2-"),
    "pos": ("flu+",),
    "neg": ("flu-",),
}


def classify(predicate: str, arity: int, positive: bool, statics: set[str]) -> str:
    """The ``(class, polarity)`` of an atom."""
    sign = "+" if positive else "-"
    if predicate in EQUALITY_NAMES:
        return f"eq{sign}"
    if predicate in statics:
        return f"stat1{sign}" if arity == 1 else f"stat2{sign}"
    return f"flu{sign}"


def regroup(counts, grouping: dict) -> dict[str, float]:
    """The counts summed into the classes of `grouping`."""
    return {
        column: sum(counts[key] for key in keys) for column, keys in grouping.items()
    }


def strip_comments(text: str) -> str:
    """`text` without PDDL comments."""
    return "\n".join(line.split(";")[0] for line in text.splitlines())


def parse_sexpr(expr: str):
    """A PDDL expression as nested lists of tokens."""
    tokens = re.findall(r"[()]|[^\s()]+", expr)
    position = 0

    def walk():
        nonlocal position
        token = tokens[position]
        position += 1
        if token != "(":
            return token
        items = []
        while tokens[position] != ")":
            items.append(walk())
        position += 1
        return items

    return walk() if tokens else []


def atoms_of(expr: str) -> list[tuple[str, tuple[str, ...], bool]]:
    """The ``(predicate, arguments, positive)`` atoms of a conjunction."""
    if not expr.strip():
        return []
    found: list[tuple[str, tuple[str, ...], bool]] = []

    def walk(node, positive: bool):
        if isinstance(node, str) or not node:
            return
        head = node[0]
        if head == "and":
            for child in node[1:]:
                walk(child, positive)
        elif head == "not":
            for child in node[1:]:
                walk(child, not positive)
        else:
            found.append((head, tuple(node[1:]), positive))

    walk(parse_sexpr(expr), True)
    return found


def section(block: str, keyword: str) -> str:
    """The expression following ``:precondition`` or ``:effect`` in one action block."""
    index = block.find(keyword)
    if index == -1:
        return ""
    rest = block[index + len(keyword) :]
    expr, _ = pddl_text.find_balanced_expr(rest, rest.index("("))
    return expr


def declared_predicates(domain_text: str) -> dict[str, int]:
    """The declared predicates with their arities."""
    expr, _ = pddl_text.find_balanced_expr(
        domain_text, domain_text.index("(:predicates")
    )
    inner = expr[len("(:predicates") : -1]
    predicates: dict[str, int] = {}
    index = 0
    while index < len(inner):
        if inner[index] != "(":
            index += 1
            continue
        atom, index = pddl_text.find_balanced_expr(inner, index)
        name, *arguments = atom[1:-1].split()
        arity, skip = 0, False
        for token in arguments:
            if token == "-":
                skip = True
            elif skip:
                skip = False
            else:
                arity += 1
        predicates[name] = arity
    return predicates


def format_atom(atom) -> str:
    predicate, arguments, positive = atom
    inner = " ".join((predicate, *arguments))
    return f"({inner})" if positive else f"(not ({inner}))"


def ground_truth(domain: str) -> tuple[Counter, int, set[str]]:
    """The original schemas' counts per (class, polarity), their number, and the static predicates."""
    text = strip_comments(domain_pddl(domain).read_text())
    blocks = pddl_text.extract_action_blocks(text)
    declared = declared_predicates(text)
    in_effects = {a[0] for b in blocks for a in atoms_of(section(b, ":effect"))}
    statics = {name for name in declared if name not in in_effects}

    counts = Counter()
    for block in blocks:
        action = block.split()[1].strip("()")
        atoms = atoms_of(section(block, ":precondition"))
        for atom, written in Counter(atoms).items():
            if written > 1:
                logger.info(
                    "%s/%s writes %s %d times, counted once",
                    domain,
                    action,
                    format_atom(atom),
                    written,
                )
            predicate, arguments, positive = atom
            counts[classify(predicate, len(arguments), positive, statics)] += 1
    return counts, len(blocks), statics


def pymimir_statics(domain: str) -> set[str]:
    """The predicates pymimir reports static, the types included."""
    import pymimir

    parsed = pymimir.Domain(domain_pddl(domain))
    return {p.get_name() for p in parsed.get_predicates() if p.is_static()}


def learned_counts(
    domain: str, run_id: str, log_dir: Path, threshold: float, checkpoint: str
) -> Counter:
    """A run's precondition counts per (class, polarity)."""
    from dias.schema.export import read_learned_schemas
    from dias.validation.checkpoint import load_run

    run = load_run(run_id, str(log_dir), checkpoint)
    transpiler = read_learned_schemas(
        run.model, run.schema, run.datamodule_hparams, threshold
    )
    statics = pymimir_statics(domain)

    counts = Counter()
    for schema in transpiler.action_schemas:
        for atom in schema.preconditions:
            counts[
                classify(atom.predicate_name, atom.arity, atom.positive, statics)
            ] += 1
    logger.debug(
        "%s: %d schemas, %d atoms",
        run_id,
        len(transpiler.action_schemas),
        sum(counts.values()),
    )
    return counts


def count_domain(
    domain: str, seeds: list[int], args: argparse.Namespace
) -> dict | None:
    """A domain's learned counts as means over seeds and its original counts, or None without runs."""
    gt, num_actions, text_statics = ground_truth(domain)

    declared = set(declared_predicates(strip_comments(domain_pddl(domain).read_text())))
    disagree = (text_statics ^ pymimir_statics(domain)) & declared
    if disagree:
        logger.warning(
            "%s: 'occurs in no effect' and pymimir's is_static() disagree on %s",
            domain,
            sorted(disagree),
        )

    pooled = Counter()
    read = 0
    for seed in seeds:
        run_id = run_id_for(args.experiment, domain, seed)
        try:
            pooled += learned_counts(
                domain, run_id, args.log_dir, args.threshold, args.checkpoint
            )
        except FileNotFoundError as error:
            logger.warning("%s: %s", run_id, error)
            continue
        read += 1
    if read == 0:
        return None
    return {
        "learned": {key: pooled[key] / read for key in KEYS},
        "gt": {key: gt[key] for key in KEYS},
        "actions": num_actions,
        "seeds": read,
    }


def rows(results: dict) -> list[tuple[str, dict]]:
    """One row per domain and an ALL row summing them."""
    pooled = {
        "learned": {key: 0.0 for key in KEYS},
        "gt": {key: 0 for key in KEYS},
        "actions": 0,
        "seeds": 0,
    }
    out = []
    for domain, result in results.items():
        for side in ("learned", "gt"):
            for key in KEYS:
                pooled[side][key] += result[side][key]
        pooled["actions"] += result["actions"]
        pooled["seeds"] = max(pooled["seeds"], result["seeds"])
        out.append((domain, result))
    out.append(("ALL", pooled))
    return out


def table(results: dict) -> tuple[list[str], list[list[str]]]:
    """Per domain, its actions, the seeds read, and per class the learned (L) and original (GT) count."""
    header = [
        "domain",
        "actions",
        "n",
        *(f"{c} {side}" for c in CLASSES for side in ("L", "GT")),
    ]
    body = []
    for label, result in rows(results):
        learned = regroup(result["learned"], CLASSES)
        gt = regroup(result["gt"], CLASSES)
        values = [v for c in CLASSES for v in (f"{learned[c]:.1f}", f"{gt[c]:.0f}")]
        body.append([label, str(result["actions"]), str(result["seeds"]), *values])
    return header, body


def write_markdown(path: Path, experiment: str, header, body) -> None:
    lines = [
        f"# Precondition literals, {experiment}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", required=True, help="Experiment whose runs to read.")
    add_domain_arguments(p)
    p.add_argument("--validation-config", type=Path, default=VALIDATION_CONFIG)
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
        "--markdown",
        type=Path,
        default=None,
        help="Also write the table to this file.",
    )
    add_logging_arguments(p)
    args = p.parse_args()
    configure_logging(args)

    seeds = resolve_seeds(args)
    results = {}
    for domain in select_domains(args.domains):
        result = count_domain(domain, seeds, args)
        if result is None:
            logger.warning("%s: no checkpoint read, skipped", domain)
            continue
        results[domain] = result
    if not results:
        raise SystemExit(f"no runs of {args.experiment} found under {args.log_dir}")

    header, body = table(results)
    print("\t".join(header))
    for row in body:
        print("\t".join(row))
    if args.markdown:
        write_markdown(args.markdown, args.experiment, header, body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
