"""Parsing and rewriting of PDDL source text. Strings in, strings out: nothing here touches the filesystem or pymimir."""

from __future__ import annotations

import re

EQUALITY_REQUIREMENT = ":equality"
TYPES_PATTERN = re.compile(r"\(:types(.*?)\)", re.DOTALL)
REQUIREMENTS_PATTERN = re.compile(r"\(:requirements([^)]*)\)", re.IGNORECASE)
COMMENT_PATTERN = re.compile(r";[^\n]*")
# What a domain written without :equality calls the static predicate it spells `=` with.
EQUALITY_PREDICATE_NAMES = frozenset({"eq", "equal"})


def find_balanced_expr(text: str, start_idx: int) -> tuple[str, int]:
    """The first parenthesised expression at or after `start_idx`, and the index past its closing paren."""
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


def parse_typed_list(body: str) -> list[tuple[list[str], str | None]]:
    """A PDDL typed list (``a b - t c``) as ``(names, type)`` groups, in order. Names after the last type have none, which is ``None`` here: what that means is the caller's to say."""
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
    """The domain's ``(:types ...)`` as ``(children, parent)`` groups, in order, a type declared without a parent being a child of ``object`` (satellite declares all four of its types that way). Kept grouped rather than flattened: callers that emit text need children and parent adjacent."""
    match = TYPES_PATTERN.search(domain_text)
    if match is None:
        return []
    return [
        (children, parent or "object")
        for children, parent in parse_typed_list(match.group(1))
        if children
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


def declared_predicates(domain_text: str) -> dict[str, int]:
    """Each predicate the domain's ``(:predicates ...)`` declares, with its arity, as written."""
    text = COMMENT_PATTERN.sub("", domain_text)
    start = text.find("(:predicates")
    if start == -1:
        return {}
    block, _ = find_balanced_expr(text, start)
    return {
        match.group(1): len(re.findall(r"\?", match.group(2)))
        for match in re.finditer(r"\(\s*([^\s()]+)([^()]*)\)", block[1:-1])
    }


def equality_predicates(domain_text: str) -> list[str]:
    """The predicates a domain spells `=` with: binary, named for equality (`eq`, `equal`) and in no action's effect.

    A domain without :equality has to write parameter distinctness this way, asserting ``(eq o o)`` in each problem; L1's output does, as blocks-3 and gripper here once did. The facts are trusted to be reflexive, which ``drop_equality_facts`` checks.
    """
    effects = " ".join(
        find_balanced_expr(block, block.find(":effect"))[0]
        for block in extract_action_blocks(COMMENT_PATTERN.sub("", domain_text))
        if ":effect" in block
    )
    return [
        name
        for name, arity in declared_predicates(domain_text).items()
        if name.lower() in EQUALITY_PREDICATE_NAMES
        and arity == 2
        and not re.search(rf"\(\s*{re.escape(name)}\s", effects, re.IGNORECASE)
    ]


def write_equality_as_builtin(domain_text: str, names: list[str]) -> str:
    """The domain with the predicates in `names` written as `=`: their atoms rewritten, their declarations dropped, and :equality required."""
    if not names:
        return domain_text
    alternatives = "|".join(re.escape(name) for name in names)
    start = domain_text.find("(:predicates")
    block, end = find_balanced_expr(domain_text, start)
    block = re.sub(
        rf"\s*\(\s*(?:{alternatives})\s[^()]*\)", "", block, flags=re.IGNORECASE
    )
    text = domain_text[:start] + block + domain_text[end:]
    text = re.sub(
        rf"\(\s*(?:{alternatives})\s+([^\s()]+)\s+([^\s()]+)\s*\)",
        r"(= \1 \2)",
        text,
        flags=re.IGNORECASE,
    )
    if domain_uses_equality(text):
        return text
    requirements = REQUIREMENTS_PATTERN.search(text)
    if requirements is not None:
        return (
            text[: requirements.end(1)]
            + f" {EQUALITY_REQUIREMENT}"
            + text[requirements.end(1) :]
        )
    definition = re.search(r"\(define\s*\(domain\s+[^\s()]+\s*\)", text)
    return (
        text[: definition.end()]
        + f"\n  (:requirements {EQUALITY_REQUIREMENT})"
        + text[definition.end() :]
    )


def drop_equality_facts(problem_text: str, names: list[str]) -> str:
    """The problem without its atoms over the predicates in `names`, which `=` makes redundant once ``write_equality_as_builtin`` has run. One relating two different objects is refused: that predicate is not equality."""
    if not names:
        return problem_text
    alternatives = "|".join(re.escape(name) for name in names)
    pattern = re.compile(
        rf"\s*\(\s*(?:{alternatives})\s+([^\s()]+)\s+([^\s()]+)\s*\)", re.IGNORECASE
    )
    for match in pattern.finditer(problem_text):
        if match.group(1).lower() != match.group(2).lower():
            raise ValueError(
                f"{match.group(0).strip()} relates two different objects, so the predicate is not equality"
            )
    return pattern.sub("", problem_text)


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
