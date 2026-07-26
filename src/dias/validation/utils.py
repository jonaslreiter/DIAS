import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F

from dias.main import CLI


class CheckpointCLI(CLI):
    def add_arguments_to_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--ckpt_path", type=Optional[Path], required=False, default=None
        )
        parser.add_argument(
            "--experiment", type=Optional[str], required=False, default=None
        )


def resolve_best_checkpoint(ckpt_dir: Path) -> Path:
    """Return the ``best-epoch*.ckpt`` most recently written in ``ckpt_dir``.

    Re-running a seed (``launch.py --seeds``) reuses the run id and therefore the
    checkpoint directory, and Lightning does not delete the previous attempt's
    file: it side-steps the name clash with a ``-v1`` suffix. Selecting by mtime
    picks the current attempt's checkpoint; plain glob order is arbitrary and
    could silently validate the stale one.
    """
    matches = list(ckpt_dir.glob("best-epoch*.ckpt"))
    if not matches:
        raise FileNotFoundError(
            f"No best-epoch*.ckpt found in {ckpt_dir} (run did not finish?)"
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def process_model_logits(
    preconditions_logits: torch.tensor,
    effect_logits: torch.tensor,
    slot_weight_logits: torch.tensor,
    force_action_args: bool,
    action_arguments: bool,
    strips_plus_actions: bool,
    free_pos_map: dict,
    action_names,
    action_arities,
    num_slots,
    random_flip: bool = False,
    threshold: float = 0.5,
):
    """Binarize the learned schema tensors at *threshold*.

    ``threshold`` is applied uniformly to every probability the schema is read
    off from: slot activation and the four add/delete/positive/negative
    tensors. Lowering it yields more active slots (higher learned arity) and
    both more effects and more preconditions; raising it yields fewer of each.
    0.5 is the default the paper's numbers use.
    """
    effect_probs = F.softmax(preconditions_logits, dim=-1)
    add_effects_probs = effect_probs[..., 1]
    del_effects_probs = effect_probs[..., -1]

    precond_probs = F.softmax(effect_logits, dim=-1)
    pos_precond_probs = precond_probs[..., 1]
    neg_precond_probs = precond_probs[..., -1]

    slot_probs = F.sigmoid(slot_weight_logits)

    # When force_action_args is enabled, compute a mask of known argument
    # slots per action and force those slots to be active in slot_probs.
    # - strips_plus: only the positions listed in strips+.yaml are known
    # - otherwise:   all argument positions (up to action arity) are known
    if action_arguments and force_action_args:
        if strips_plus_actions:
            forced_slot_mask = torch.zeros(
                len(action_names), num_slots, dtype=torch.bool
            )
            for a_idx, name in enumerate(action_names):
                for pos in free_pos_map.get(name, []):
                    if pos < num_slots:
                        forced_slot_mask[a_idx, pos] = True
        else:
            # All positions up to the action arity are known
            slot_indices = torch.arange(num_slots).unsqueeze(0)
            arity_t = torch.tensor(action_arities, dtype=torch.long).unsqueeze(1)
            forced_slot_mask = slot_indices < arity_t  # (num_actions, num_slots)
        slot_probs[forced_slot_mask] = 1.0

    active_slots = np.array(slot_probs) > threshold  # shape: (num_actions, num_slots)

    # Zero out rows and columns for inactive slots in effects/preconditions
    # Shapes: (num_actions, num_relations, num_slots, num_slots)
    for a in range(active_slots.shape[0]):
        inactive = ~active_slots[a]  # (num_slots,)
        add_effects_probs[a, :, inactive, :] = 0
        add_effects_probs[a, :, :, inactive] = 0
        del_effects_probs[a, :, inactive, :] = 0
        del_effects_probs[a, :, :, inactive] = 0
        pos_precond_probs[a, :, inactive, :] = 0
        pos_precond_probs[a, :, :, inactive] = 0
        neg_precond_probs[a, :, inactive, :] = 0
        neg_precond_probs[a, :, :, inactive] = 0

    # Threshold after masking
    add_eff = np.where(add_effects_probs > threshold, 1, 0)
    del_eff = np.where(del_effects_probs > threshold, 1, 0)
    pos_prec = np.where(pos_precond_probs > threshold, 1, 0)
    neg_prec = np.where(neg_precond_probs > threshold, 1, 0)

    if random_flip:
        for arr in (add_eff, del_eff, pos_prec, neg_prec):
            idx = tuple(np.random.randint(0, s) for s in arr.shape)
            arr[idx] = 1 - arr[idx]

    return add_eff, del_eff, pos_prec, neg_prec, active_slots


def strip_problem_types(
    problem_pddl_path: str,
    domain_pddl_path: str | None = None,
    equality_predicate_name: str | None = None,
    reflexive_predicate_names: list[str] | None = None,
) -> str:
    """Return the problem PDDL text with type annotations removed from :objects.

    If *domain_pddl_path* is given, the type hierarchy is read from the domain file and
    type-derived unary predicates (e.g. ``(cell c_0_0)``, ``(locatable p1)``) are injected
    into :init so that the untyped learned domain can use them as static preconditions.
    """
    import re

    text = Path(problem_pddl_path).read_text()

    type_atoms: list[str] = []
    object_names: list[str] = []

    if domain_pddl_path is not None:
        # Parse the type hierarchy from the domain: "child - parent"
        domain_text = Path(domain_pddl_path).read_text()
        types_match = re.search(r"\(:types(.*?)\)", domain_text, re.DOTALL)
        type_parents: dict[str, str] = {}  # child → parent
        if types_match:
            for m in re.finditer(r"([\w\s]+)-\s+(\w+)", types_match.group(1)):
                parent = m.group(2).strip()
                for child in m.group(1).split():
                    type_parents[child.strip()] = parent

        # Parse object-type assignments from the problem: "obj1 obj2 - typename".
        # Tokenize so object names containing '-' (e.g. "p1-0") are not split.
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

    # Remove type annotations: "obj1 obj2 - typename" → "obj1 obj2"
    text = re.sub(r"\s+-\s+\w+", "", text)

    if type_atoms:
        # Insert type atoms just before the closing ')' of :init
        goal_pos = text.find("(:goal")
        init_close = text.rfind(")", 0, goal_pos) if goal_pos != -1 else text.rfind(")")
        if init_close != -1:
            extra = "\n        " + "\n        ".join(sorted(set(type_atoms)))
            text = text[:init_close] + extra + "\n    " + text[init_close:]

    reflexive_names: list[str] = []
    if equality_predicate_name is not None:
        reflexive_names.append(equality_predicate_name)
    if reflexive_predicate_names:
        reflexive_names.extend(reflexive_predicate_names)

    if reflexive_names:
        eq_atoms = [
            f"({name} {obj} {obj})" for name in reflexive_names for obj in object_names
        ]
        if eq_atoms:
            goal_pos = text.find("(:goal")
            init_close = (
                text.rfind(")", 0, goal_pos) if goal_pos != -1 else text.rfind(")")
            )
            if init_close != -1:
                extra = "\n        " + "\n        ".join(sorted(set(eq_atoms)))
                text = text[:init_close] + extra + "\n    " + text[init_close:]

    return text


def extract_domain_header(
    domain_pddl_path: str,
    rel_names: list[str] | None = None,
    predicate_arities: list[int] | None = None,
    keep_equality_requirement: bool = True,
) -> tuple[str, str]:
    """Return (header, domain_name) where header is everything up to (but not including)
    the first :action block, with :negative-preconditions added to :requirements if missing.

    If *rel_names* and *predicate_arities* are given, any relation that appears in the
    learned actions but is not declared in the original :predicates block (e.g. type-
    derived static predicates like ``cell``, ``locatable``) will be appended to the
    :predicates section so that the generated domain file is self-contained.
    """
    text = Path(domain_pddl_path).read_text()

    # Find where the first action starts
    action_start = text.find("(:action")
    if action_start == -1:
        # No actions — use everything up to the closing paren of the define block
        header = text.rstrip().rstrip(")")
    else:
        header = text[:action_start].rstrip()

    # Force a :requirements block that includes :negative-preconditions.
    # We drop any existing :requirements block and re-inject a clean one right
    # after "(define (domain <name>)", so the learned domain is always valid.
    # Only add :equality if the *original* domain declared it, so pymimir does
    # not auto-materialise reflexive (= x x) atoms where the original wouldn't.
    import re as _re

    orig_reqs_match = _re.search(r"\(:requirements([^)]*)\)", text)
    needs_equality = bool(orig_reqs_match and ":equality" in orig_reqs_match.group(1))
    reqs = ":negative-preconditions"
    if needs_equality and keep_equality_requirement:
        reqs += " :equality"
    header = _re.sub(r"\(:requirements[^)]*\)\s*", "", header)
    m_def = _re.search(r"\(define\s*\(domain\s+\S+\s*\)", header)
    if m_def is not None:
        pos = m_def.end()
        header = header[:pos] + f"\n  (:requirements {reqs})" + header[pos:]

    import re

    # --- remove typing: :types block, :typing requirement, type annotations ---
    header = re.sub(r"\(:types[^)]*\)\s*", "", header)
    header = re.sub(r":typing\s*", "", header)
    # Strip type annotations from predicate params: "?x1 - locatable" → "?x1"
    header = re.sub(r"(\?\w+)\s+-\s+\w+", r"\1", header)

    # --- rebuild :predicates from rel_names (complete & untyped) --------
    if rel_names is not None and predicate_arities is not None:
        # Remove existing :predicates block (it has typed params we don't want)
        pred_start = header.find("(:predicates")
        if pred_start != -1:
            depth = 0
            for i in range(pred_start, len(header)):
                if header[i] == "(":
                    depth += 1
                elif header[i] == ")":
                    depth -= 1
                    if depth == 0:
                        # Remove the block and any trailing whitespace
                        header = header[:pred_start] + header[i + 1 :].lstrip("\n")
                        break

        # Build fresh untyped predicate declarations from all relation names
        # Skip "=" — it is a built-in enabled by :equality, not a user predicate
        declared: set[str] = set()
        decls: list[str] = []
        for name, arity in zip(rel_names, predicate_arities):
            if name == "=":
                continue
            declared.add(name)
            if arity == 0:
                decls.append(f"({name})")
            elif arity == 1:
                decls.append(f"({name} ?x)")
            else:
                params = " ".join(f"?x{i}" for i in range(arity))
                decls.append(f"({name} {params})")

        # Also declare type-derived predicates (unary) and any original-domain
        # predicates not already covered by rel_names (e.g. "object")
        types_match = re.search(r"\(:types(.*?)\)", text, re.DOTALL)
        if types_match:
            for m2 in re.finditer(r"([\w\s]+)-\s+(\w+)", types_match.group(1)):
                parent = m2.group(2).strip()
                for child in m2.group(1).split():
                    child = child.strip()
                    if child and child not in declared:
                        declared.add(child)
                        decls.append(f"({child} ?x)")
                if parent and parent != "object" and parent not in declared:
                    declared.add(parent)
                    decls.append(f"({parent} ?x)")

        # Extract the original :predicates block with proper paren balancing
        # (non-greedy regex stops at the first ')' which breaks for multi-line blocks).
        orig_pred_body = None
        pp = text.find("(:predicates")
        if pp != -1:
            depth = 0
            start = None
            for i in range(pp, len(text)):
                if text[i] == "(":
                    if depth == 0:
                        start = i
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        orig_pred_body = text[start + len("(:predicates") : i]
                        break
        if orig_pred_body is not None:
            for m2 in re.finditer(
                r"\((\w+)((?:\s+\?\w+(?:\s+-\s+\w+)?)*)\)", orig_pred_body
            ):
                name = m2.group(1)
                if name not in declared and name != "=":
                    arity = len(re.findall(r"\?\w+", m2.group(2)))
                    declared.add(name)
                    if arity == 0:
                        decls.append(f"({name})")
                    elif arity == 1:
                        decls.append(f"({name} ?x)")
                    else:
                        params = " ".join(f"?x{i}" for i in range(arity))
                        decls.append(f"({name} {params})")

        preds_block = (
            "    (:predicates\n        " + "\n        ".join(decls) + "\n    )"
        )
        header = header.rstrip() + "\n\n" + preds_block

    # Extract domain name
    m = re.search(r"\(define\s+\(domain\s+(\S+)\)", header)
    domain_name = m.group(1) if m else "learned-domain"

    return header, domain_name
