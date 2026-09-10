"""Sort the precondition atoms of an experiment's learned schemas, and of the ground-truth domains, into categories.

Every ``(domain, seed)`` checkpoint is read through ``dias.validation.dias.read_learned_schemas``, the function ``DiasValidator`` itself uses (``load_run``, then ``DIAS.schema_logits()``, then ``derive_schema`` at ``--threshold``, then ``PDDLTranspiler``), so what is counted here is exactly what a validator would score. Ground truth is parsed out of the ``:precondition`` conjunctions of ``pddl/<domain>/domain.pddl``.

Each atom lands in exactly one class:

  eq      equalities between action parameters: the built-in ``=`` under :equality, or the ordinary reflexive predicate a domain declares in its place (``eq``)
  stat1   unary literals over static predicates, either polarity
  stat2   binary literals over static predicates, either polarity
  pos     positive fluent literals
  neg     negative fluent literals

A static predicate is one that occurs in no action's effect, which is also what pymimir's ``Predicate.is_static()`` reports and therefore the flag DIAS itself is given; the two are compared per domain and a disagreement is a warning.

The paper and the rebuttals quote a four-way split with no class for ``stat2``, folding those literals into ``pos`` / ``neg`` by polarity. Both groupings are always reported, since the four-way one is what the text quotes and the five-way one is what the class names mean.

Counting follows ``experiments/results/schema-complexity-dias-full.md``, so the two are comparable: distinct lifted atoms after de-duplication, and ``(not (p ...))`` is one atom. Type-derived unary predicates are excluded from the ground-truth counts, since a ``?t - truck`` in ``:parameters`` is not a precondition. DIAS does get a channel for them (pymimir materialises them, and ``include_statics`` defaults to true), which is what the ``stat1`` column measures.

Usage:
    python experiments/precondition_categories.py --experiment dias-full
    python experiments/precondition_categories.py --experiment dias-full --domains gripper satellite -v
    python experiments/precondition_categories.py --experiment dias-full --markdown experiments/results/precondition-categories-dias-full.md
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from dias.pddl import text as pddl_text
from dias.pddl.untype import EQUALITY_PREDICATE

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import validate_runs`
from pipeline import (  # noqa: E402
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

# Equality reaches the schemas under two spellings: the built-in "=", which read_learned_schemas renames, and the ordinary predicate a domain declares in its place.
EQUALITY_NAMES = frozenset({"=", EQUALITY_PREDICATE})

# Atoms are counted per (class, polarity), because the two reported groupings cut across the pair differently.
KINDS = ("eq", "stat1", "stat2", "flu")
KEYS = tuple(f"{kind}{sign}" for kind in KINDS for sign in "+-")

# What each grouping's columns hold, in report order.
FIVE_WAY = {
    "eq": ("eq+", "eq-"),
    "stat1": ("stat1+", "stat1-"),
    "stat2": ("stat2+", "stat2-"),
    "pos": ("flu+",),
    "neg": ("flu-",),
}
FOUR_WAY = {
    "eq": ("eq+", "eq-"),
    "stat1": ("stat1+", "stat1-"),
    "pos": ("flu+", "stat2+"),
    "neg": ("flu-", "stat2-"),
}
LABELS = {
    "eq": "equalities between action parameters",
    "stat1": "unary literals over static predicates",
    "stat2": "binary literals over static predicates",
    "pos": "positive fluent literals",
    "neg": "negative fluent literals",
}
FOUR_WAY_LABELS = {
    **LABELS,
    "pos": "positive fluent literals, plus positive binary static literals",
    "neg": "negative fluent literals, plus negative binary static literals",
}


def classify(predicate: str, arity: int, positive: bool, statics: set[str]) -> str:
    """The ``(class, polarity)`` key one atom belongs to. `statics` also names the equality predicate, so the first test wins."""
    sign = "+" if positive else "-"
    if predicate in EQUALITY_NAMES:
        return f"eq{sign}"
    if predicate in statics:
        return f"stat1{sign}" if arity == 1 else f"stat2{sign}"
    return f"flu{sign}"


def regroup(counts, grouping: dict) -> dict[str, float]:
    """Sum the per-(class, polarity) counts into one grouping's columns."""
    return {
        column: sum(counts[key] for key in keys) for column, keys in grouping.items()
    }


# ── the ground-truth domain ───────────────────────────────────────────────────
#
# Parsed from the PDDL text rather than through pymimir, because what is wanted is the atoms the domain *writes*: pymimir has already turned the `:parameters` types into unary predicates, and those are exactly what the counting convention excludes.


def strip_comments(text: str) -> str:
    """PDDL line comments, dropped before anything counts parentheses: a comment may hold an unbalanced one."""
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
    """Flatten a conjunction into ``(predicate, arguments, positive)`` triples, in the order written."""
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
    """``name -> arity`` from the ``(:predicates ...)`` block, type annotations removed."""
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
    """One domain's ground-truth class counts, its action count, and its static predicates."""
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
    """The predicates pymimir reports static, which is the map DIAS is given. Includes the ones materialised out of ``:types``."""
    import pymimir

    parsed = pymimir.Domain(domain_pddl(domain))
    return {p.get_name() for p in parsed.get_predicates() if p.is_static()}


# ── the learned schemas ───────────────────────────────────────────────────────


def learned_counts(
    domain: str, run_id: str, log_dir: Path, threshold: float
) -> Counter:
    """One run's precondition atoms, classified. Raises FileNotFoundError if the run has no checkpoint."""
    from dias.validation.checkpoint import load_run
    from dias.validation.dias import read_learned_schemas

    run = load_run(run_id, str(log_dir))
    has_equality = pddl_text.domain_uses_equality(Path(run.domain_path).read_text())
    transpiler = read_learned_schemas(run, has_equality, threshold)
    # read_learned_schemas renames the "=" channel, so the static map must use the same name.
    statics = {
        EQUALITY_PREDICATE if name == "=" else name for name in pymimir_statics(domain)
    }

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
    """One domain's per-seed learned means and ground-truth counts, or None if no run was readable."""
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
            pooled += learned_counts(domain, run_id, args.log_dir, args.threshold)
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


# ── reporting ─────────────────────────────────────────────────────────────────


def rows(results: dict) -> list[tuple[str, dict]]:
    """One row per domain, then a pooled ALL row. Learned values are already per-seed means, so pooling is a sum."""
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


def print_table(results: dict, experiment: str, grouping: dict, title: str) -> None:
    columns = list(grouping)
    header = f"{'domain':<16} {'act':>4} {'n':>3} " + " ".join(
        f"{c + ' L':>9} {c + ' GT':>8}" for c in columns
    )
    print(f"\n{title} ({experiment}; L = per-seed mean, GT = ground truth)")
    print(header)
    print("-" * len(header))
    for label, result in rows(results):
        if label == "ALL":
            print("-" * len(header))
        learned = regroup(result["learned"], grouping)
        gt = regroup(result["gt"], grouping)
        cells = " ".join(f"{learned[c]:>9.1f} {gt[c]:>8.0f}" for c in columns)
        print(f"{label:<16} {result['actions']:>4} {result['seeds']:>3} {cells}")


def markdown_table(results: dict, grouping: dict) -> list[str]:
    columns = list(grouping)
    lines = [
        "| domain | act | n | " + " | ".join(f"{c} L | {c} GT" for c in columns) + " |",
        "|" + "---|" * (2 * len(columns) + 3),
    ]
    for label, result in rows(results):
        learned = regroup(result["learned"], grouping)
        gt = regroup(result["gt"], grouping)
        values = [v for c in columns for v in (f"{learned[c]:.1f}", f"{gt[c]:.0f}")]
        if label == "ALL":
            values = [f"**{v}**" for v in values]
            lines.append(
                f"| **ALL** | **{result['actions']}** | **{result['seeds']}** | "
                + " | ".join(values)
                + " |"
            )
        else:
            lines.append(
                f"| {label} | {result['actions']} | {result['seeds']} | "
                + " | ".join(values)
                + " |"
            )
    lines.append("")
    return lines


def write_markdown(
    path: Path, results: dict, experiment: str, threshold: float
) -> None:
    lines = [
        f"# Learned vs. ground-truth precondition categories (`{experiment}`)",
        "",
        f"Every precondition atom of every `{experiment}` action schema, learned and ground truth, sorted into classes. `act` is the domain's action count and `n` the seeds read; `L` columns are per-seed means and `GT` columns count each action's schema once, which is the unit on which the two are comparable.",
        "",
        "## How the numbers were produced",
        "",
        f"- `python experiments/precondition_categories.py --experiment {experiment} --markdown {path}`.",
        f"- Learned schemas were read from `logs/dias/{experiment}-<domain>-s<seed>/checkpoints/best-epoch*.ckpt` through `dias.validation.dias.read_learned_schemas`, the function `DiasValidator` uses: `load_run`, then `DIAS.schema_logits()`, then `derive_schema(threshold={threshold:g})`, then `PDDLTranspiler`.",
        "- Ground-truth schemas were parsed from the `:precondition` conjunction of `pddl/<domain>/domain.pddl`.",
        "- Distinct lifted atoms after de-duplication; `(not (p ...))` counts as one atom; type-derived unary predicates are excluded from the ground-truth counts, since a `?t - truck` in `:parameters` is not a precondition. This matches `schema-complexity-dias-full.md`, so the two files are comparable.",
        "- A static predicate is one that occurs in no action's effect, which is also what `pymimir`'s `Predicate.is_static()` reports and so the flag DIAS is given. `pymimir` additionally marks static the unary predicates it materialises out of a `:types` declaration, and `include_statics` defaults to true, so DIAS does get a channel for `(truck ?x)` and its kind while a ground-truth schema states that guard in `:parameters` instead. That asymmetry is what the `stat1` column measures.",
        "",
        "## Five classes",
        "",
        "| Class | Meaning |",
        "| --- | --- |",
        *(f"| {c} | {LABELS[c]} |" for c in FIVE_WAY),
        "",
        *markdown_table(results, FIVE_WAY),
        "## Four classes",
        "",
        "The grouping the paper and the rebuttals quote. It has no class for binary static literals (`link`, `adjacent`, `above`, `loc`, `on_board` and their kind), which land in `pos` or `neg` by their polarity, so these two columns are not what their names say; the five-class table above splits them out.",
        "",
        "| Class | Meaning |",
        "| --- | --- |",
        *(f"| {c} | {FOUR_WAY_LABELS[c]} |" for c in FOUR_WAY),
        "",
        *markdown_table(results, FOUR_WAY),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        default="dias-full",
        help="Experiment whose runs to read, the prefix of the run ids (default: dias-full).",
    )
    add_domain_arguments(p)
    p.add_argument("--validation-config", type=Path, default=VALIDATION_CONFIG)
    p.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold handed to derive_schema (default: 0.5, what launch.py uses for --experiment-type dias; it uses 0.75 for noise).",
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Also write both tables to this markdown file.",
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

    print_table(results, args.experiment, FIVE_WAY, "five classes")
    print_table(results, args.experiment, FOUR_WAY, "four classes, as the paper quotes")
    if args.markdown:
        write_markdown(args.markdown, results, args.experiment, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
