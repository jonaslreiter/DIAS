"""Rewriting typed PDDL as untyped PDDL, which is what validating an untyped learned schema against a typed domain needs: the type hierarchy becomes unary static predicates in ``:init``, and ``(= ?x ?y)`` an ordinary predicate."""

from __future__ import annotations

import re

from . import text as pddl_text

# The predicate an untyped domain uses in place of the built-in `=`.
EQUALITY_PREDICATE = "eq"


def strip_problem_types(
    problem_text: str,
    domain_text: str | None = None,
    equality_predicate_name: str | None = None,
    reflexive_predicate_names: list[str] | None = None,
) -> str:
    """Remove the type annotations from a problem's ``:objects``.

    With *domain_text*, type-derived unary atoms (``(cell c_0_0)``) go into ``:init`` so the untyped domain can use them as static preconditions. Named reflexive predicates get ``(name o o)`` per object, which is how ``(= ?x ?y)`` survives.
    """
    type_atoms: list[str] = []

    if domain_text is not None:
        parents = pddl_text.type_parents(domain_text)
        # Tokenize rather than split on '-', so object names like "p1-0" stay intact.
        match = re.search(r"\(:objects(.*?)\)", problem_text, re.DOTALL)
        tokens = match.group(1).split() if match else []
        pending: list[str] = []
        index = 0
        while index < len(tokens):
            if tokens[index] == "-" and index + 1 < len(tokens):
                for name in pending:
                    # Assert the object's type and every ancestor of it.
                    type_name = tokens[index + 1]
                    while type_name and type_name != "object":
                        type_atoms.append(f"({type_name} {name})")
                        type_name = parents.get(type_name)
                pending = []
                index += 2
            else:
                pending.append(tokens[index])
                index += 1

    text = re.sub(r"\s+-\s+\w+", "", problem_text)

    reflexive_names = [equality_predicate_name] if equality_predicate_name else []
    reflexive_names += reflexive_predicate_names or []
    objects = problem_object_names(problem_text) if reflexive_names else []
    reflexive_atoms = [
        f"({name} {obj} {obj})" for name in reflexive_names for obj in objects
    ]

    for atoms in (type_atoms, reflexive_atoms):
        if atoms:
            text = _extend_init(text, atoms)

    return text


def _extend_init(text: str, atoms: list[str]) -> str:
    """Insert *atoms* just before the closing paren of ``:init``."""
    goal_pos = text.find("(:goal")
    init_close = text.rfind(")", 0, goal_pos) if goal_pos != -1 else text.rfind(")")
    if init_close == -1:
        return text
    extra = "\n        " + "\n        ".join(sorted(set(atoms)))
    return text[:init_close] + extra + "\n    " + text[init_close:]


def extract_domain_header(
    domain_text: str,
    rel_names: list[str] | None = None,
    predicate_arities: list[int] | None = None,
    keep_equality_requirement: bool = True,
) -> str:
    """Return everything before the first ``:action``, untyped.

    ``:requirements`` gains ``:negative-preconditions``, and keeps ``:equality`` only if the original declared it, so pymimir does not materialise reflexive ``(= x x)`` atoms. Given *rel_names* and *predicate_arities*, ``:predicates`` is rebuilt so the generated domain is self-contained.
    """
    action_start = domain_text.find("(:action")
    if action_start == -1:
        header = domain_text.rstrip().rstrip(")")
    else:
        header = domain_text[:action_start].rstrip()

    # Rewrite :requirements in place, after the (define (domain ...)) line.
    orig_reqs_match = re.search(r"\(:requirements([^)]*)\)", domain_text)
    needs_equality = bool(orig_reqs_match and ":equality" in orig_reqs_match.group(1))
    reqs = ":negative-preconditions"
    if needs_equality and keep_equality_requirement:
        reqs += " :equality"
    header = re.sub(r"\(:requirements[^)]*\)\s*", "", header)
    m_def = re.search(r"\(define\s*\(domain\s+\S+\s*\)", header)
    if m_def is not None:
        pos = m_def.end()
        header = header[:pos] + f"\n  (:requirements {reqs})" + header[pos:]

    # Drop the type declarations and every typed parameter annotation.
    header = re.sub(r"\(:types[^)]*\)\s*", "", header)
    header = re.sub(r":typing\s*", "", header)
    header = re.sub(r"(\?\w+)\s+-\s+\w+", r"\1", header)

    if rel_names is None or predicate_arities is None:
        return header

    # Rebuild :predicates from the learned relations, the type names and whatever the original declared beyond both.
    pred_start = header.find("(:predicates")
    if pred_start != -1:
        block, _ = pddl_text.find_balanced_expr(header, pred_start)
        header = header[:pred_start] + header[pred_start + len(block) :].lstrip("\n")

    declared: set[str] = set()
    decls: list[str] = []

    def declare(name: str, arity: int) -> None:
        # "=" is a built-in enabled by :equality, not a user predicate.
        if name in declared or name == "=":
            return
        declared.add(name)
        if arity == 0:
            decls.append(f"({name})")
        elif arity == 1:
            decls.append(f"({name} ?x)")
        else:
            decls.append(f"({name} " + " ".join(f"?x{i}" for i in range(arity)) + ")")

    for name, arity in zip(rel_names, predicate_arities):
        declare(name, arity)

    for children, parent in pddl_text.parse_type_groups(domain_text):
        for child in children:
            if child:
                declare(child, 1)
        if parent and parent != "object":
            declare(parent, 1)

    if domain_text.find("(:predicates") != -1:
        block, _ = pddl_text.find_balanced_expr(
            domain_text, domain_text.find("(:predicates")
        )
        for match in re.finditer(
            r"\((\w+)((?:\s+\?\w+(?:\s+-\s+\w+)?)*)\)", block[len("(:predicates") : -1]
        ):
            declare(match.group(1), len(re.findall(r"\?\w+", match.group(2))))

    preds_block = "    (:predicates\n        " + "\n        ".join(decls) + "\n    )"
    return header.rstrip() + "\n\n" + preds_block


def problem_object_names(problem_text: str) -> list[str]:
    """Every object a problem declares, with any type annotations dropped."""
    match = re.search(r"\(:objects(.*?)\)", problem_text, re.DOTALL)
    if not match:
        return []
    tokens = match.group(1).split()
    names: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] == "-" and index + 1 < len(tokens):
            index += 2
        else:
            names.append(tokens[index])
            index += 1
    return names


def inject_reflexive_atoms(problem_text: str, predicate_names: list[str]) -> str:
    """Assert ``(p o o)`` in :init for each named predicate and every object, which a domain with its own equality-like predicate expects and a foreign problem file does not supply."""
    objects = problem_object_names(problem_text) if predicate_names else []
    if not objects:
        return problem_text
    return _extend_init(
        problem_text,
        [f"({name} {obj} {obj})" for name in predicate_names for obj in objects],
    )


def untype_action(
    action_block: str,
    name_prefix: str = "",
    equality_predicate_name: str | None = None,
    ignored_types: frozenset[str] = frozenset(),
) -> str:
    """Rewrite one ``(:action ...)`` block without types: each typed parameter becomes a unary precondition, except the uninformative ones in *ignored_types*."""
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

    if equality_predicate_name is not None:
        pre_expr = pddl_text.replace_equality_with_predicate(
            pre_expr, equality_predicate_name
        )
        eff_expr = pddl_text.replace_equality_with_predicate(
            eff_expr, equality_predicate_name
        )
    pre_expr = pddl_text.inject_extra_preconditions(pre_expr, type_preconditions)

    return f"(:action {action_name}\n  :parameters ({' '.join(params)})\n  :precondition {pre_expr}\n  :effect {eff_expr}\n)"
