"""Scoring a learned PDDL domain produced outside this repository, against the original domain it was learned from."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pymimir

from dias.pddl import text as pddl_text
from dias.pddl.untype import (
    EQUALITY_PREDICATE,
    extract_domain_header,
    strip_problem_types,
    untype_action,
)

from .artifacts import PddlWorkspace
from .successors import (
    LEARNED_ACTION_PREFIX,
    compare_successors,
    resolve_sample_seed,
    sample_reachable_states,
    transfer_states,
)

logger = logging.getLogger(__name__)

# A learned schema's parameters carry these when it has no real type for them.
UNINFORMATIVE_TYPES = frozenset({"untyped", "object"})


class L1Validator:
    def __init__(
        self,
        original_domain_path,
        learned_domain_path,
        logs_path,
        problem_path=None,
    ):
        self.logs_path = logs_path

        # Equality is `=` from here on, however each side spells it: L1 writes it as the static `eq` or `equal` its input domain had, the domains in pddl/ as `=`.
        original_domain_text = Path(original_domain_path).read_text()
        original_equalities = pddl_text.equality_predicates(original_domain_text)
        original_domain_text = pddl_text.write_equality_as_builtin(
            original_domain_text, original_equalities
        )
        problem_text = pddl_text.drop_equality_facts(
            Path(problem_path).read_text(), original_equalities
        )
        learned_domain_text = Path(learned_domain_path).read_text()
        learned_domain_text = pddl_text.write_equality_as_builtin(
            learned_domain_text, pddl_text.equality_predicates(learned_domain_text)
        )
        self._check_learned_types(original_domain_text, learned_domain_text)

        # The merged domain drops :equality, so `=` on either side is compiled to the same reflexive predicate.
        has_equality = pddl_text.domain_uses_equality(
            original_domain_text
        ) or pddl_text.domain_uses_equality(learned_domain_text)
        equality_predicate_name = EQUALITY_PREDICATE if has_equality else None

        # A merged untyped domain: transformed original + learned actions.
        original_actions_text = "\n\n".join(
            untype_action(action_block, equality_predicate_name=equality_predicate_name)
            for action_block in pddl_text.extract_action_blocks(original_domain_text)
        )
        learned_actions_text = "\n\n".join(
            untype_action(
                action_block,
                name_prefix=LEARNED_ACTION_PREFIX,
                ignored_types=UNINFORMATIVE_TYPES,
                equality_predicate_name=equality_predicate_name,
            )
            for action_block in pddl_text.extract_action_blocks(learned_domain_text)
        )
        header = self._patch_missing_predicates(
            extract_domain_header(
                original_domain_text, keep_equality_requirement=False
            ),
            original_actions_text + "\n\n" + learned_actions_text,
            original_domain_text,
            has_equality=has_equality,
        )
        merged_domain_text = (
            header
            + "\n\n"
            + original_actions_text
            + "\n\n"
            + learned_actions_text
            + "\n)"
        )

        self.workspace = PddlWorkspace()
        self.merged_domain_filename = self.workspace.write(
            "merged-domain.pddl", merged_domain_text + "\n\n"
        )
        # An untyped problem, with the type and equality facts in :init.
        self.merged_problem_filename = self.workspace.write(
            "merged-problem.pddl",
            strip_problem_types(
                problem_text,
                original_domain_text,
                equality_predicate_name=equality_predicate_name,
            ),
        )

        self.merged_domain = pymimir.Domain(Path(self.merged_domain_filename))
        self.merged_problem = pymimir.Problem(
            self.merged_domain, Path(self.merged_problem_filename)
        )

        # The original as respelled, which is the file as written unless it spelled equality as a predicate.
        self.original_domain = pymimir.Domain(
            Path(self.workspace.write("original-domain.pddl", original_domain_text))
        )
        self.original_problem = pymimir.Problem(
            self.original_domain,
            Path(self.workspace.write("original-problem.pddl", problem_text)),
        )

        self.state_set = None

    @staticmethod
    def _check_learned_types(
        original_domain_text: str, learned_domain_text: str
    ) -> None:
        """Refuse a learned parameter type the original cannot supply. Types are given rather than learned, so a parameter may carry one of the original's types, or a unary predicate an untyped original writes its types as (logistics' `truck`); any other type would become a precondition no problem asserts, and its schema would silently never apply."""
        known = {
            name.lower()
            for children, parent in pddl_text.parse_type_groups(original_domain_text)
            for name in (*children, parent)
        }
        known |= {
            name.lower()
            for name, arity in pddl_text.declared_predicates(
                original_domain_text
            ).items()
            if arity == 1
        }
        used = {
            type_name.lower()
            for block in pddl_text.extract_action_blocks(learned_domain_text)
            for _, type_name in pddl_text.parse_typed_parameters(
                pddl_text.find_balanced_expr(block, block.find(":parameters"))[0]
            )[1]
        }
        unknown = sorted(used - known - UNINFORMATIVE_TYPES)
        if unknown:
            raise ValueError(
                f"The learned domain types parameters as {', '.join(unknown)}, which the original declares neither as a type nor as a unary predicate"
            )

    def _patch_missing_predicates(
        self,
        header: str,
        actions_text: str,
        original_domain_text: str = "",
        has_equality: bool = False,
    ) -> str:
        """Declare every predicate the merged domain uses but does not define: ones an action body references, original or learned, the `eq` replacement for `=`, and the type names, which became unary atoms."""
        defined_words = {
            m.group(1).lower() for m in re.finditer(r"\(\s*([^\s()]+)", header)
        }
        # `eq` is absent on purpose: here it is a user predicate, not a keyword.
        keywords = {"and", "or", "not", "imply", "when", "forall", "exists", "=", "-"}

        missing_preds: dict[str, int] = {}

        for match in re.finditer(r"\(\s*([^\s()]+)([^()]*)\)", actions_text):
            name = match.group(1).lower()
            if name.startswith(":") or name.startswith("?"):
                continue
            if name not in keywords and name not in defined_words:
                arity = len([w for w in match.group(2).split() if w.startswith("?")])
                missing_preds.setdefault(name, arity)

        if has_equality:
            eq_name = EQUALITY_PREDICATE.lower()
            if eq_name not in defined_words and eq_name not in missing_preds:
                missing_preds[eq_name] = 2

        for children, parent in pddl_text.parse_type_groups(original_domain_text):
            for name in (*children, parent):
                name = name.lower()
                if (
                    name != "object"
                    and name not in keywords
                    and name not in defined_words
                    and name not in missing_preds
                ):
                    missing_preds[name] = 1

        if missing_preds:
            declarations = "".join(
                "\n    ({} {})".format(name, " ".join(f"?v{i}" for i in range(arity)))
                for name, arity in missing_preds.items()
            )
            header = re.sub(
                r"(\(:predicates)", rf"\1{declarations}", header, flags=re.IGNORECASE
            )
            logger.debug("Declared missing predicates: %s", sorted(missing_preds))

        return header

    def setup(
        self,
        num_samples: int = 250,
        sample_population: int = 500,
        seed: int | None = None,
    ):
        self.sample_seed = resolve_sample_seed(seed)
        states = sample_reachable_states(
            self.original_problem, num_samples, sample_population, self.sample_seed
        )
        self.state_set = transfer_states(
            states, self.merged_problem, self.original_domain, self.merged_domain
        )

    def validate(self):
        if self.state_set is None:
            raise RuntimeError("Call setup() before validate().")
        return compare_successors(self.state_set, LEARNED_ACTION_PREFIX)
