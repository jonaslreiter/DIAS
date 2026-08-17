"""Rewriting typed PDDL as untyped PDDL.

Learned schemas are untyped, so validating them against a typed domain means
compiling the types away: the type hierarchy becomes unary static predicates
asserted in ``:init``, and ``(= ?x ?y)`` becomes an ordinary predicate. Both
functions take and return PDDL source.
"""

from __future__ import annotations

import re

from . import text as pddl_text


def strip_problem_types(
    problem_text: str,
    domain_text: str | None = None,
    equality_predicate_name: str | None = None,
    reflexive_predicate_names: list[str] | None = None,
) -> str:
    """Remove the type annotations from a problem's ``:objects``.

    With *domain_text*, the type hierarchy is read from the domain and the
    type-derived unary atoms (``(cell c_0_0)``, ``(locatable p1)``) are added to
    ``:init``, so the untyped learned domain can use them as static
    preconditions. Named reflexive predicates get ``(name o o)`` for every
    object, which is how ``(= ?x ?y)`` survives the translation.
    """
    text = problem_text

    type_atoms: list[str] = []
    object_names: list[str] = []

    if domain_text is not None:
        type_parents = pddl_text.type_parents(domain_text)

        # Tokenize "obj1 obj2 - typename" rather than splitting on '-', so
        # object names containing a hyphen (e.g. "p1-0") stay intact.
        objects_match = re.search(r"\(:objects(.*?)\)", text, re.DOTALL)
        if objects_match:
            tokens = objects_match.group(1).split()
            pending: list[str] = []
            i = 0
            while i < len(tokens):
                if tokens[i] == "-" and i + 1 < len(tokens):
                    type_name = tokens[i + 1]
                    for obj_name in pending:
                        object_names.append(obj_name)
                        t = type_name
                        while t and t != "object":
                            type_atoms.append(f"({t} {obj_name})")
                            t = type_parents.get(t)
                    pending = []
                    i += 2
                else:
                    pending.append(tokens[i])
                    i += 1

    if not object_names:
        objects_match = re.search(r"\(:objects(.*?)\)", text, re.DOTALL)
        if objects_match:
            tokens = objects_match.group(1).split()
            i = 0
            while i < len(tokens):
                if tokens[i] == "-" and i + 1 < len(tokens):
                    i += 2
                else:
                    object_names.append(tokens[i])
                    i += 1

    text = re.sub(r"\s+-\s+\w+", "", text)

    reflexive_names = [equality_predicate_name] if equality_predicate_name else []
    reflexive_names += reflexive_predicate_names or []
    reflexive_atoms = [
        f"({name} {obj} {obj})" for name in reflexive_names for obj in object_names
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

    ``:requirements`` is rewritten to guarantee ``:negative-preconditions``, and
    ``:equality`` is kept only if the original declared it, so pymimir does not
    materialise reflexive ``(= x x)`` atoms the original domain would not have.

    Given *rel_names* and *predicate_arities*, the ``:predicates`` block is
    rebuilt from scratch: the learned relations, then the type-derived
    predicates, then anything the original declared that neither covers. That
    makes the generated domain self-contained.
    """
    action_start = domain_text.find("(:action")
    if action_start == -1:
        header = domain_text.rstrip().rstrip(")")
    else:
        header = domain_text[:action_start].rstrip()

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

    header = re.sub(r"\(:types[^)]*\)\s*", "", header)
    header = re.sub(r":typing\s*", "", header)
    header = re.sub(r"(\?\w+)\s+-\s+\w+", r"\1", header)

    if rel_names is None or predicate_arities is None:
        return header

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
