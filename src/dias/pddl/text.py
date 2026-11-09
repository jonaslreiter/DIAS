"""Parsing and rewriting PDDL text."""

from __future__ import annotations

import re
from itertools import combinations

EQUALITY_REQUIREMENT = ":equality"
TYPES_PATTERN = re.compile(r"\(:types(.*?)\)", re.DOTALL)
REQUIREMENTS_PATTERN = re.compile(r"\(:requirements([^)]*)\)", re.IGNORECASE)
COMMENT_PATTERN = re.compile(r";[^\n]*")
# Equality as an ordinary predicate, which validation writes it as.
EQUALITY_PREDICATE = "equal"
EQUALITY_PREDICATE_NAMES = frozenset({"eq", EQUALITY_PREDICATE})
EQUALITY_LITERAL_PATTERN = re.compile(
    rf"\s*\(\s*not\s*\(\s*{EQUALITY_PREDICATE}\s[^()]*\)\s*\)|\s*\(\s*{EQUALITY_PREDICATE}\s[^()]*\)"
)


def find_balanced_expr(text: str, start_idx: int) -> tuple[str, int]:
    """The first parenthesised expression at or after `start_idx`, and the index past it."""
    left = text.find("(", start_idx)
    if left == -1:
        raise ValueError("Expected balanced expression but found no '('")

    depth = 0
    for i in range(left, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[left : i + 1], i + 1

    raise ValueError("Unbalanced parentheses while parsing expression")


def extract_blocks(text: str, opener: str) -> list[str]:
    """Every balanced block starting with `opener`, e.g. ``(:action``."""
    blocks: list[str] = []
    index = 0
    while True:
        start = text.find(opener, index)
        if start == -1:
            return blocks
        block, index = find_balanced_expr(text, start)
        blocks.append(block.strip())


def extract_action_blocks(domain_text: str) -> list[str]:
    return extract_blocks(domain_text, "(:action")


def parse_typed_list(body: str) -> list[tuple[list[str], str | None]]:
    """A PDDL typed list (``a b - t c``) as ``(names, type)`` groups, in order; names after the last type get ``None``."""
    groups: list[tuple[list[str], str | None]] = []
    pending: list[str] = []
    tokens = body.split()
    index = 0
    while index < len(tokens):
        if tokens[index] == "-" and index + 1 < len(tokens):
            groups.append((pending, tokens[index + 1]))
            pending = []
            index += 2
        else:
            pending.append(tokens[index])
            index += 1
    if pending:
        groups.append((pending, None))
    return groups


def parse_typed_parameters(
    param_expr: str,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split a ``(:parameters ...)`` body into all variables and the typed ones."""
    groups = parse_typed_list(param_expr.strip()[1:-1])
    variables = [name for names, _ in groups for name in names]
    typed = [
        (name, type_name)
        for names, type_name in groups
        if type_name is not None
        for name in names
    ]
    return variables, typed


def parse_type_groups(domain_text: str) -> list[tuple[list[str], str]]:
    """The domain's ``(:types ...)`` as ``(children, parent)`` groups in order, ``object`` the parent where none is given."""
    match = TYPES_PATTERN.search(domain_text)
    if match is None:
        return []
    return [
        (children, parent or "object")
        for children, parent in parse_typed_list(match.group(1))
        if children
    ]


def type_parents(domain_text: str) -> dict[str, str]:
    """Each declared type's parent."""
    return {
        child: parent
        for children, parent in parse_type_groups(domain_text)
        for child in children
    }


def domain_uses_equality(domain_text: str) -> bool:
    match = REQUIREMENTS_PATTERN.search(domain_text)
    return bool(match and EQUALITY_REQUIREMENT in match.group(1).lower())


def declared_predicates(domain_text: str) -> dict[str, int]:
    """The predicates the domain declares, with their arities."""
    text = COMMENT_PATTERN.sub("", domain_text)
    start = text.find("(:predicates")
    if start == -1:
        return {}
    block, _ = find_balanced_expr(text, start)
    return {
        match.group(1): len(re.findall(r"\?", match.group(2)))
        for match in re.finditer(r"\(\s*([^\s()]+)([^()]*)\)", block[1:-1])
    }


def is_equality_predicate(name: str, arity: int) -> bool:
    """PDDL's `=`, or a binary predicate named `eq` or `equal`."""
    return name == "=" or (name.lower() in EQUALITY_PREDICATE_NAMES and arity == 2)


def equality_predicates(domain_text: str) -> list[str]:
    """The predicates a domain declares in place of `=`."""
    return [
        name
        for name, arity in declared_predicates(domain_text).items()
        if is_equality_predicate(name, arity)
    ]


def equality_as_predicate(domain_text: str) -> str:
    """The domain with `=`, `eq` or `equal` written as the declared predicate `equal`."""
    declared = equality_predicates(domain_text)
    start = domain_text.find("(:predicates")
    block, end = find_balanced_expr(domain_text, start)
    for name in declared:
        block = re.sub(
            rf"\s*\(\s*{re.escape(name)}\s[^()]*\)", "", block, flags=re.IGNORECASE
        )
    block = block[:-1].rstrip() + f"\n    ({EQUALITY_PREDICATE} ?x ?y)\n  )"
    text = domain_text[:start] + block + domain_text[end:]
    names = "|".join(re.escape(name) for name in ["=", *declared])
    text = re.sub(
        rf"\(\s*(?:{names})\s+([^\s()]+)\s+([^\s()]+)\s*\)",
        rf"({EQUALITY_PREDICATE} \1 \2)",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"(\(:requirements[^)]*?)\s*:equality", r"\1", text, flags=re.IGNORECASE
    )


def force_distinct_parameters(domain_text: str) -> str:
    """The domain, equality written as `equal`, with every action's `equal` literals replaced by the distinctness of every two parameters."""
    text = domain_text
    for block in extract_action_blocks(domain_text):
        variables, _ = parse_typed_parameters(
            find_balanced_expr(block, block.find(":parameters"))[0]
        )
        distinct = [
            f"(not ({EQUALITY_PREDICATE} {a} {b}))"
            for a, b in combinations(variables, 2)
        ]
        rewritten = block
        for part in (":precondition", ":effect"):
            start = rewritten.find(part)
            expression, end = find_balanced_expr(rewritten, start)
            kept = EQUALITY_LITERAL_PATTERN.sub("", expression) or "()"
            if part == ":precondition":
                kept = inject_extra_preconditions(kept, distinct)
            rewritten = (
                rewritten[: rewritten.index("(", start)] + kept + rewritten[end:]
            )
        text = text.replace(block, rewritten, 1)
    return text


def require_equality(domain_text: str) -> str:
    """The domain with :equality among its requirements."""
    if domain_uses_equality(domain_text):
        return domain_text
    requirements = REQUIREMENTS_PATTERN.search(domain_text)
    if requirements is not None:
        return (
            domain_text[: requirements.end(1)]
            + f" {EQUALITY_REQUIREMENT}"
            + domain_text[requirements.end(1) :]
        )
    definition = re.search(r"\(define\s*\(domain\s+[^\s()]+\s*\)", domain_text)
    return (
        domain_text[: definition.end()]
        + f"\n  (:requirements {EQUALITY_REQUIREMENT})"
        + domain_text[definition.end() :]
    )


def inject_extra_preconditions(precondition_expr: str, extras: list[str]) -> str:
    """Conjoin `extras` onto a precondition, flattening a top-level ``and``."""
    precondition = precondition_expr.strip()
    if not extras:
        return precondition

    joined = " ".join(extras)
    if precondition in {"()", "(and)"}:
        return f"(and {joined})"

    if precondition.startswith("(and") and precondition.endswith(")"):
        inner = precondition[4:-1].strip()
        return f"(and {joined} {inner})" if inner else f"(and {joined})"

    return f"(and {joined} {precondition})"
