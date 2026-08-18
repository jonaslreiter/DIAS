from __future__ import annotations

import random
import re
from pathlib import Path

import pymimir
from tqdm import tqdm

from dias.data.generate_samples import generate_bfs
from dias.pddl import text as pddl_text
from dias.pddl.states import state_fluents, transfer_state
from dias.pddl.untype import (
    extract_domain_header,
    inject_reflexive_atoms,
    strip_problem_types,
    untype_action,
)

from .artifacts import PddlWorkspace

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
        self._state_id_cache: dict[int, int] = {}

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

        if num_samples > sample_population:
            num_samples = sample_population
            print(
                f"\033[93m[WARNING] num_samples ({num_samples}) exceeds population, clamping to {sample_population}.\033[0m"
            )

        self.state_set = []

        bfs_samples = generate_bfs(self.original_problem, sample_population)
        seen = set()
        population_set = []
        for s in bfs_samples:
            st = s["_pymimir_state"]
            sid = st.get_index()
            if sid not in seen:
                seen.add(sid)
                population_set.append(st)
            if len(population_set) >= sample_population:
                break

        if num_samples > len(population_set):
            print(
                f"\033[93m[WARNING] num_samples ({num_samples}) exceeds reachable "
                f"states ({len(population_set)}), clamping.\033[0m"
            )
            num_samples = len(population_set)

        self.state_set = random.sample(population_set, num_samples)

        shared_preds = {p.get_name() for p in self.original_domain.get_predicates()} & {
            p.get_name() for p in self.merged_domain.get_predicates()
        }

        merged_state_set = []
        for state in self.state_set:
            merged_state = transfer_state(state, self.merged_problem)
            orig_shared = state_fluents(state, shared_preds)
            merged_shared = state_fluents(merged_state, shared_preds)
            assert orig_shared == merged_shared, (
                f"State transfer mismatch (comparison restricted to predicates "
                f"present in both domains):\n"
                f"  shared predicates: {sorted(shared_preds)}\n"
                f"  Len. original: {len(orig_shared)}\n"
                f"  Len. merged:   {len(merged_shared)}\n"
                f"  original: {orig_shared}\n"
                f"  merged:   {merged_shared}\n"
                f"  only in original: {orig_shared - merged_shared}\n"
                f"  only in merged:   {merged_shared - orig_shared}"
            )
            merged_state_set.append(merged_state)

        self.state_set = merged_state_set

    def validate(self):
        if self.state_set is None:
            raise RuntimeError("\033[91mCall setup() before validate().\033[0m")

        matches = 0
        tp_total = 0
        fp_total = 0
        fn_total = 0

        from collections import defaultdict

        action_tp = defaultdict(int)
        action_fp = defaultdict(int)
        action_fn = defaultdict(int)

        prefix_len = len(self.LEARNED_ACTION_PREFIX)
        state_id = self.state_id

        for state in tqdm(
            self.state_set, total=len(self.state_set), desc="Validating states"
        ):
            orig_by_action = defaultdict(set)
            learned_by_action = defaultdict(set)
            for ga in state.generate_applicable_actions():
                succ_id = state_id(ga.apply(state))
                action_name = ga.get_action().get_name()
                if action_name.startswith(self.LEARNED_ACTION_PREFIX):
                    learned_by_action[action_name[prefix_len:]].add(succ_id)
                else:
                    orig_by_action[action_name].add(succ_id)

            orig_succs = (
                set().union(*orig_by_action.values()) if orig_by_action else set()
            )
            learned_succs = (
                set().union(*learned_by_action.values()) if learned_by_action else set()
            )

            tp = len(orig_succs & learned_succs)
            fp = len(learned_succs) - tp
            fn = len(orig_succs) - tp

            tp_total += tp
            fp_total += fp
            fn_total += fn

            if orig_succs == learned_succs:
                matches += 1

            for name in set(orig_by_action) | set(learned_by_action):
                o = orig_by_action.get(name, set())
                lrn = learned_by_action.get(name, set())
                a_tp = len(o & lrn)
                action_tp[name] += a_tp
                action_fp[name] += len(lrn) - a_tp
                action_fn[name] += len(o) - a_tp

        precision = (
            tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        )
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

        action_stats = {
            name: {
                "tp": action_tp[name],
                "fp": action_fp[name],
                "fn": action_fn[name],
                "conflicts": action_fp[name] + action_fn[name],
            }
            for name in set(action_tp) | set(action_fp) | set(action_fn)
        }

        return {
            "matches": matches,
            "total": len(self.state_set),
            "accuracy": matches / len(self.state_set),
            "precision": precision,
            "recall": recall,
            "action_stats": action_stats,
        }

    def state_id(self, state):
        sid = state.get_index()
        cid = self._state_id_cache.get(sid)
        if cid is None:
            cid = sid
            self._state_id_cache[sid] = cid
        return cid

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
