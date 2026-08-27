from __future__ import annotations

import logging
import re
from pathlib import Path

import pymimir

from dias.pddl import text as pddl_text
from dias.pddl.untype import (
    EQUALITY_PREDICATE,
    extract_domain_header,
    inject_reflexive_atoms,
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


class ModelValidation:
    def __init__(
        self,
        original_domain_path,
        learned_domain_path,
        logs_path_abs,
        problem_path=None,
    ):
        self.logs_path_abs = logs_path_abs

        # Read both files
        original_domain_text = Path(original_domain_path).read_text()
        learned_domain_text = Path(learned_domain_path).read_text()

        has_equality = pddl_text.domain_uses_equality(original_domain_text)
        fluent_equal_predicates = self._fluent_equal_predicates(original_domain_text)

        # A merged untyped domain: transformed original + learned actions.
        original_actions_text = "\n\n".join(
            untype_action(
                action_block,
                equality_predicate_name=(EQUALITY_PREDICATE if has_equality else None),
            )
            for action_block in pddl_text.extract_action_blocks(original_domain_text)
        )
        learned_actions_text = "\n\n".join(
            untype_action(
                action_block,
                name_prefix=LEARNED_ACTION_PREFIX,
                ignored_types=UNINFORMATIVE_TYPES,
            )
            for action_block in pddl_text.extract_action_blocks(learned_domain_text)
        )
        header = self._patch_missing_predicates(
            extract_domain_header(
                original_domain_text, keep_equality_requirement=False
            ),
            original_actions_text,
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
                Path(problem_path).read_text(),
                original_domain_text,
                equality_predicate_name=(EQUALITY_PREDICATE if has_equality else None),
                reflexive_predicate_names=fluent_equal_predicates or None,
            ),
        )

        self.merged_domain = pymimir.Domain(Path(self.merged_domain_filename))
        self.merged_problem = pymimir.Problem(
            self.merged_domain, Path(self.merged_problem_filename)
        )

        self.original_domain = pymimir.Domain(Path(original_domain_path))

        original_problem_path_to_use = problem_path
        if fluent_equal_predicates:
            # The original domain needs the same reflexive atoms the stripped problem got, or the two would not enumerate the same actions.
            original_problem_path_to_use = self.workspace.write(
                "original-problem.pddl",
                inject_reflexive_atoms(
                    Path(problem_path).read_text(), fluent_equal_predicates
                ),
            )

        self.original_problem = pymimir.Problem(
            self.original_domain, Path(original_problem_path_to_use)
        )

        self.state_set = None

    def _patch_missing_predicates(
        self,
        header: str,
        actions_text: str,
        original_domain_text: str = "",
        has_equality: bool = False,
    ) -> str:
        """Declare every predicate the merged domain uses but does not define.

        Three sources go missing: predicates an action body references that the header never declared, the `eq` replacement for `=` (strip_problem_types puts `(eq o o)` in :init even when no action mentions equality), and the type names, which become unary atoms like `(locatable bob)` for every parent in the chain.
        """
        defined_words = {
            m.group(1).lower() for m in re.finditer(r"\(\s*([a-zA-Z0-9_\-]+)", header)
        }
        # `eq` is deliberately absent: it is a user-defined predicate here, not a keyword, and must be declared.
        keywords = {"and", "or", "not", "imply", "when", "forall", "exists", "=", "-"}

        missing_preds: dict[str, int] = {}

        for match in re.finditer(r"\(\s*([a-zA-Z0-9_\-]+)\s+([^()]*)\)", actions_text):
            name = match.group(1).lower()
            if name not in keywords and name not in defined_words:
                arity = len([w for w in match.group(2).split() if w.startswith("?")])
                missing_preds.setdefault(name, arity)

        if has_equality:
            eq_name = EQUALITY_PREDICATE.lower()
            if eq_name not in defined_words and eq_name not in missing_preds:
                missing_preds[eq_name] = 2

        types_match = re.search(r"\(:types(.*?)\)", original_domain_text, re.DOTALL)
        if types_match:
            type_names: set[str] = set()
            for m2 in re.finditer(r"([\w\s]+)-\s+(\w+)", types_match.group(1)):
                parent = m2.group(2).strip().lower()
                for child in m2.group(1).split():
                    type_names.add(child.strip().lower())
                if parent and parent != "object":
                    type_names.add(parent)
            for name in type_names:
                if (
                    name
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
        # Recorded, so a result drawn without a fixed seed can still be redrawn.
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

    @staticmethod
    def _fluent_equal_predicates(domain_text: str) -> list[str]:
        """Return names of binary fluent predicates that look like reflexive equality (e.g. `equal`, `eq`) declared in the domain's :predicates block. The kr_2024 problem.pddl files initialize these as `(equal X X)` for every object; new problem files don't, so we need to bootstrap them on the merged problem."""
        preds_match = re.search(r"\(:predicates(.*?)\n\s*\)", domain_text, re.DOTALL)
        if not preds_match:
            return []
        block = preds_match.group(1)
        results: list[str] = []
        for m in re.finditer(r"\(\s*([a-zA-Z0-9_\-]+)([^()]*)\)", block):
            name = m.group(1).lower()
            if name not in {"equal", "eq"}:
                continue
            args = m.group(2)
            var_count = len([w for w in args.split() if w.startswith("?")])
            if var_count == 2:
                results.append(m.group(1))
        return results
