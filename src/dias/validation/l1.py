from __future__ import annotations

import re
from pathlib import Path

import pymimir

from dias.pddl import text as pddl_text
from dias.pddl.untype import (
    extract_domain_header,
    inject_reflexive_atoms,
    strip_problem_types,
    untype_action,
)

from .artifacts import PddlWorkspace
from .successors import (
    compare_successors,
    sample_reachable_states,
    transfer_states,
)

# A learned schema's parameters carry these when it has no real type for them.
UNINFORMATIVE_TYPES = frozenset({"untyped", "object"})


class ModelValidation:
    LEARNED_ACTION_PREFIX = "learned_"
    EQUALITY_PREDICATE_NAME = "eq"

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
                equality_predicate_name=(
                    self.EQUALITY_PREDICATE_NAME if has_equality else None
                ),
            )
            for action_block in pddl_text.extract_action_blocks(original_domain_text)
        )
        learned_actions_text = "\n\n".join(
            untype_action(
                action_block,
                name_prefix=self.LEARNED_ACTION_PREFIX,
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
                equality_predicate_name=(
                    self.EQUALITY_PREDICATE_NAME if has_equality else None
                ),
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
            # The original domain needs the same reflexive atoms the stripped
            # problem got, or the two would not enumerate the same actions.
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
        """
        Detects undeclared predicates in the actions and injects them
        into the (:predicates ...) block of the header. Also injects all
        type names from the original domain's :types hierarchy as unary
        predicates, since strip_problem_types emits them in :init for
        every parent in the chain (e.g. (locatable bob)). When the original
        domain uses :equality, declares the binary 'eq' replacement
        predicate even if no action body references it.
        """
        import re

        # 1. Collect everything that looks like a word after an opening parenthesis in the header
        # This will safely catch 'at', 'carrying', 'link', etc., as well as PDDL keywords.
        defined_words = set(
            [m.group(1).lower() for m in re.finditer(r"\(\s*([a-zA-Z0-9_\-]+)", header)]
        )

        # Standard keywords to ignore. Note: 'eq' is intentionally NOT here —
        # it is a user-defined replacement for '=' that must be declared.
        keywords = {"and", "or", "not", "imply", "when", "forall", "exists", "=", "-"}

        missing_preds = {}

        # 2. Find all predicate usages in the original actions text
        # Looks for (predicate_name ?arg1 ?arg2 ...)
        for match in re.finditer(r"\(\s*([a-zA-Z0-9_\-]+)\s+([^()]*)\)", actions_text):
            name = match.group(1).lower()

            # If it's not a keyword and not in the header, it's missing!
            if name not in keywords and name not in defined_words:
                if name not in missing_preds:
                    # Count how many variables (words starting with '?') it uses
                    args_str = match.group(2)
                    vars_count = len([w for w in args_str.split() if w.startswith("?")])
                    missing_preds[name] = vars_count

        # 2a. If the original domain uses :equality, strip_problem_types will
        # inject (eq obj obj) atoms into :init for every object — even when no
        # action body references equality. Declare the binary eq predicate so
        # those atoms parse.
        if has_equality:
            eq_name = self.EQUALITY_PREDICATE_NAME.lower()
            if eq_name not in defined_words and eq_name not in missing_preds:
                missing_preds[eq_name] = 2

        # 2b. Pull every type name from the original domain's :types hierarchy and
        # declare them as unary predicates. strip_problem_types injects atoms for
        # parent types (e.g. (locatable bob)) that never appear in any action body.
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

        # 3. Inject the missing predicates right after the '(:predicates' keyword
        if missing_preds:
            missing_str = ""
            for name, arity in missing_preds.items():
                args = " ".join([f"?v{i}" for i in range(arity)])
                missing_str += f"\n    ({name} {args})"

            # Use regex substitution to patch the header
            header = re.sub(
                r"(\(:predicates)", rf"\1{missing_str}", header, flags=re.IGNORECASE
            )
            print(f"Patched missing predicates: {list(missing_preds.keys())}")

        return header

    @classmethod
    def setup(self, num_samples: int = 250, sample_population: int = 500):
        states = sample_reachable_states(
            self.original_problem, num_samples, sample_population
        )
        self.state_set = transfer_states(
            states, self.merged_problem, self.original_domain, self.merged_domain
        )

    def validate(self):
        if self.state_set is None:
            raise RuntimeError("Call setup() before validate().")
        return compare_successors(self.state_set, self.LEARNED_ACTION_PREFIX)

    @staticmethod
    def _fluent_equal_predicates(domain_text: str) -> list[str]:
        """Return names of binary fluent predicates that look like reflexive equality
        (e.g. `equal`, `eq`) declared in the domain's :predicates block. The kr_2024
        problem.pddl files initialize these as `(equal X X)` for every object; new
        problem files don't, so we need to bootstrap them on the merged problem.
        """
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
