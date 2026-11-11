"""Rewriting typed PDDL as untyped PDDL, the types as unary predicates."""

from __future__ import annotations

import re

from . import text as pddl_text


def strip_problem_types(
    problem_text: str,
    domain_text: str | None = None,
) -> str:
    """The problem without types in ``:objects``, and with `domain_text` each object's types and their ancestors asserted in ``:init``."""
    type_atoms: list[str] = []

    if domain_text is not None:
        parents = pddl_text.type_parents(domain_text)
        for names, type_name in pddl_text.parse_typed_list(_objects_body(problem_text)):
            for name in names:
                ancestor = type_name
                while ancestor and ancestor != "object":
                    type_atoms.append(f"({ancestor} {name})")
                    ancestor = parents.get(ancestor)

    text = re.sub(r"\s+-\s+[^\s()]+", "", problem_text)

    return _extend_init(text, type_atoms) if type_atoms else text


def with_equality_facts(problem_text: str) -> str:
    """The problem asserting ``(equal o o)`` for every object in place of the equality facts it had, which must each relate an object to itself."""
    pattern = re.compile(
        rf"\s*\(\s*(?:{'|'.join(pddl_text.EQUALITY_PREDICATE_NAMES)})\s+([^\s()]+)\s+([^\s()]+)\s*\)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(problem_text):
        if match.group(1).lower() != match.group(2).lower():
            raise ValueError(
                f"{match.group(0).strip()} relates two different objects, so it is not equality"
            )
    return _extend_init(
        pattern.sub("", problem_text),
        [
            f"({pddl_text.EQUALITY_PREDICATE} {obj} {obj})"
            for obj in problem_object_names(problem_text)
        ],
    )


def _extend_init(text: str, atoms: list[str]) -> str:
    """`text` with `atoms` added to ``:init``."""
    goal_pos = text.find("(:goal")
    init_close = text.rfind(")", 0, goal_pos) if goal_pos != -1 else text.rfind(")")
    if init_close == -1:
        return text
    extra = "\n        " + "\n        ".join(sorted(set(atoms)))
    return text[:init_close] + extra + "\n    " + text[init_close:]


def extract_domain_header(
    domain_text: str,
    predicate_names: list[str] | None = None,
    predicate_arities: list[int] | None = None,
) -> str:
    """Everything before the first ``:action``, untyped, requiring ``:negative-preconditions`` and ``:equality`` if the original did. Given the predicates, ``:predicates`` declares them, the types and the original's predicates."""
    action_start = domain_text.find("(:action")
    if action_start == -1:
        header = domain_text.rstrip().rstrip(")")
    else:
        header = domain_text[:action_start].rstrip()

    orig_reqs_match = re.search(r"\(:requirements([^)]*)\)", domain_text)
    needs_equality = bool(orig_reqs_match and ":equality" in orig_reqs_match.group(1))
    reqs = ":negative-preconditions"
    if needs_equality:
        reqs += " :equality"
    header = re.sub(r"\(:requirements[^)]*\)\s*", "", header)
    m_def = re.search(r"\(define\s*\(domain\s+\S+\s*\)", header)
    if m_def is not None:
        pos = m_def.end()
        header = header[:pos] + f"\n  (:requirements {reqs})" + header[pos:]

    header = re.sub(r"\(:types[^)]*\)\s*", "", header)
    header = re.sub(r":typing\s*", "", header)
    header = re.sub(r"(\?[^\s()]+)\s+-\s+[^\s()]+", r"\1", header)

    if predicate_names is None or predicate_arities is None:
        return header

    pred_start = header.find("(:predicates")
    if pred_start != -1:
        block, _ = pddl_text.find_balanced_expr(header, pred_start)
        header = header[:pred_start] + header[pred_start + len(block) :].lstrip("\n")

    declared: set[str] = set()
    decls: list[str] = []

    def declare(name: str, arity: int) -> None:
        if name in declared or name == "=":
            return
        declared.add(name)
        if arity == 0:
            decls.append(f"({name})")
        elif arity == 1:
            decls.append(f"({name} ?x)")
        else:
            decls.append(f"({name} " + " ".join(f"?x{i}" for i in range(arity)) + ")")

    for name, arity in zip(predicate_names, predicate_arities):
        declare(name, arity)

    for children, parent in pddl_text.parse_type_groups(domain_text):
        for child in children:
            if child:
                declare(child, 1)
        if parent and parent != "object":
            declare(parent, 1)

    for name, arity in pddl_text.declared_predicates(domain_text).items():
        declare(name, arity)

    preds_block = "    (:predicates\n        " + "\n        ".join(decls) + "\n    )"
    return header.rstrip() + "\n\n" + preds_block


def _objects_body(problem_text: str) -> str:
    match = re.search(r"\(:objects(.*?)\)", problem_text, re.DOTALL)
    return match.group(1) if match else ""


def problem_object_names(problem_text: str) -> list[str]:
    """The objects a problem declares."""
    return [
        name
        for names, _ in pddl_text.parse_typed_list(_objects_body(problem_text))
        for name in names
    ]


def untype_action(
    action_block: str,
    name_prefix: str = "",
    ignored_types: frozenset[str] = frozenset(),
) -> str:
    """An ``(:action ...)`` block without types, each parameter type but `ignored_types` a unary precondition."""
    match = re.match(r"\(\s*:action\s+(\S+)", action_block)
    if match is None:
        raise ValueError("Failed to parse action name")
    action_name = f"{name_prefix}{match.group(1)}"

    indices = [
        action_block.find(part) for part in (":parameters", ":precondition", ":effect")
    ]
    if -1 in indices:
        raise ValueError(
            f"Action '{action_name}' is missing parameters/precondition/effect"
        )
    param_expr, pre_expr, eff_expr = (
        pddl_text.find_balanced_expr(action_block, index)[0] for index in indices
    )

    params, typed_params = pddl_text.parse_typed_parameters(param_expr)
    type_preconditions = [
        f"({type_name} {var_name})"
        for var_name, type_name in typed_params
        if type_name.lower() not in ignored_types
    ]

    pre_expr = pddl_text.inject_extra_preconditions(pre_expr, type_preconditions)

    return f"(:action {action_name}\n  :parameters ({' '.join(params)})\n  :precondition {pre_expr}\n  :effect {eff_expr}\n)"
