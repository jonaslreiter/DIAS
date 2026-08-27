"""Parsing and rewriting of PDDL source text.

Strings in, strings out: nothing here touches the filesystem or pymimir, so it is all unit-testable without either.
"""

from __future__ import annotations

import re

EQUALITY_REQUIREMENT = ":equality"
TYPES_PATTERN = re.compile(r"\(:types(.*?)\)", re.DOTALL)
TYPE_GROUP_PATTERN = re.compile(r"([\w\s]+)-\s+(\w+)")
REQUIREMENTS_PATTERN = re.compile(r"\(:requirements([^)]*)\)", re.IGNORECASE)


def find_balanced_expr(text: str, start_idx: int) -> tuple[str, int]:
    """Return the first parenthesised expression at or after `start_idx`.

    Yields the expression and the index just past its closing paren.
    """
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
    """Return every balanced block starting with `opener`, e.g. ``(:action``."""
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


def parse_typed_parameters(
    param_expr: str,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split a ``(:parameters ...)`` body into all variables and the typed ones."""
    tokens = param_expr.strip()[1:-1].split()
    variables: list[str] = []
    typed: list[tuple[str, str]] = []

    pending: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-" and index + 1 < len(tokens):
            type_name = tokens[index + 1]
            for name in pending:
                variables.append(name)
                typed.append((name, type_name))
            pending = []
            index += 2
        else:
            pending.append(token)
            index += 1

    variables.extend(pending)
    return variables, typed


def parse_type_groups(domain_text: str) -> list[tuple[list[str], str]]:
    """The domain's ``(:types ...)`` as ``(children, parent)`` groups, in order.

    Grouping is kept rather than flattened to a child -> parent map because callers that emit text depend on children and their parent staying adjacent.
    """
    match = TYPES_PATTERN.search(domain_text)
    if match is None:
        return []
    return [
        ([child.strip() for child in children.split()], parent.strip())
        for children, parent in TYPE_GROUP_PATTERN.findall(match.group(1))
    ]


def type_parents(domain_text: str) -> dict[str, str]:
    """Map each declared child type to its parent."""
    return {
        child: parent
        for children, parent in parse_type_groups(domain_text)
        for child in children
    }


def domain_uses_equality(domain_text: str) -> bool:
    match = REQUIREMENTS_PATTERN.search(domain_text)
    return bool(match and EQUALITY_REQUIREMENT in match.group(1).lower())


def replace_equality_with_predicate(expr: str, predicate_name: str) -> str:
    """Rewrite ``(= ?a ?b)`` as an ordinary predicate, which untyped PDDL needs."""
    return re.sub(
        r"\(\s*=\s+([^\s()]+)\s+([^\s()]+)\s*\)",
        rf"({predicate_name} \1 \2)",
        expr,
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
