from __future__ import annotations

import random
import re
import tempfile
from pathlib import Path

import pymimir
import torch
from pymimir.advanced.formalism import FluentGroundAtomList, ObjectList
from pymimir.wrapper_formalism import State
from tqdm import tqdm

from dias.data.datamodule import PddlGraphDataModule
from dias.data.generate_samples import generate_bfs
from dias.models.dias import DIAS

from .transpiler import PDDLTranspiler
from .utils import extract_domain_header, process_model_logits, strip_problem_types


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

        has_equality = self._domain_uses_equality(original_domain_text)
        fluent_equal_predicates = self._fluent_equal_predicates(original_domain_text)

        # Create a merged untyped domain: transformed original actions + transformed learned actions.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pddl", dir="/tmp", delete=False
        ) as f:

            # Note: Removed pddl_transpiler kwargs. If extract_domain_header requires
            # them, you may need to adjust that utility function to just parse the original file.
            header, _ = extract_domain_header(
                original_domain_path,
                keep_equality_requirement=False,
            )

            # Process Original Actions
            original_actions = self._extract_action_blocks(original_domain_text)
            transformed_original_actions = [
                self._transform_original_action(
                    action_block, replace_equality=has_equality
                )
                for action_block in original_actions
            ]

            # Process Learned Actions
            learned_actions = self._extract_action_blocks(learned_domain_text)
            transformed_learned_actions = [
                self._transform_learned_action(action_block)
                for action_block in learned_actions
            ]

            # --- NEW: Patch missing predicates before merging ---
            all_original_actions_text = "\n\n".join(transformed_original_actions)
            header = self._patch_missing_predicates(
                header,
                all_original_actions_text,
                original_domain_text,
                has_equality=has_equality,
            )
            # ----------------------------------------------------

            # Merge everything into a single PDDL string
            pddl_str = (
                header
                + "\n\n"
                + all_original_actions_text
                + "\n\n"
                + "\n\n".join(transformed_learned_actions)
                + "\n)"
            )
            self.merged_domain_filename = f.name
            print(f"Merged domain file: {self.merged_domain_filename}")
            f.write(pddl_str + "\n\n")

        # Create an untyped problem with type/equality facts in :init.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pddl", dir="/tmp", delete=False
        ) as pf:
            self.merged_problem_filename = pf.name
            pf.write(
                strip_problem_types(
                    problem_path,
                    original_domain_path,
                    equality_predicate_name=(
                        self.EQUALITY_PREDICATE_NAME if has_equality else None
                    ),
                    reflexive_predicate_names=fluent_equal_predicates or None,
                )
            )
            print(f"Merged problem file: {self.merged_problem_filename}")

        self.merged_domain = pymimir.Domain(Path(self.merged_domain_filename))
        self.merged_problem = pymimir.Problem(
            self.merged_domain, Path(self.merged_problem_filename)
        )

        # Note: datamodule_hparams seems to be a global or outer-scope variable in your script.
        self.original_domain = pymimir.Domain(Path(original_domain_path))

        original_problem_path_to_use = problem_path
        if fluent_equal_predicates:
            patched_text = self._inject_reflexive_atoms(
                Path(problem_path).read_text(), fluent_equal_predicates
            )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pddl", dir="/tmp", delete=False
            ) as opf:
                opf.write(patched_text)
                original_problem_path_to_use = opf.name
            print(
                f"Patched original problem with reflexive {fluent_equal_predicates}: "
                f"{original_problem_path_to_use}"
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
    def _transform_learned_action(cls, action_text: str) -> str:
        """
        Transforms a learned action block by prefixing its name to prevent
        collisions, stripping its parameter types, and injecting the dropped
        types as unary predicate preconditions so the action remains typed
        in the untyped merged domain.
        """
        m = re.match(r"\(\s*:action\s+(\S+)", action_text)
        if m is None:
            raise ValueError("Failed to parse learned action name")
        action_name = f"{cls.LEARNED_ACTION_PREFIX}{m.group(1)}"

        params_idx = action_text.find(":parameters")
        pre_idx = action_text.find(":precondition")
        eff_idx = action_text.find(":effect")

        if params_idx == -1 or pre_idx == -1 or eff_idx == -1:
            raise ValueError(
                f"Learned action '{action_name}' is missing parameters/precondition/effect"
            )

        param_expr, _ = cls._find_balanced_expr(action_text, params_idx)
        pre_expr, _ = cls._find_balanced_expr(action_text, pre_idx)
        eff_expr, _ = cls._find_balanced_expr(action_text, eff_idx)

        params, typed_params = cls._parse_typed_parameters(param_expr)
        typed_preconditions = [
            f"({type_name} {var_name})"
            for var_name, type_name in typed_params
            if type_name.lower() not in {"untyped", "object"}
        ]
        pre_expr = cls._inject_extra_preconditions(pre_expr, typed_preconditions)

        params_str = " ".join(params)
        return (
            f"(:action {action_name}\n"
            f"  :parameters ({params_str})\n"
            f"  :precondition {pre_expr}\n"
            f"  :effect {eff_expr}\n"
            f")"
        )

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
            merged_state = self.transfer_state(state, self.merged_problem)
            orig_shared = self._state_fluents_on_predicates(state, shared_preds)
            merged_shared = self._state_fluents_on_predicates(
                merged_state, shared_preds
            )
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
    def state_fluents(state):
        return frozenset(
            str(atom)
            for atom in state.get_atoms(ignore_static=False, ignore_derived=True)
        )

    @staticmethod
    def state_dynamic_fluents(state):
        return frozenset(
            str(atom)
            for atom in state.get_atoms(ignore_static=True, ignore_derived=True)
        )

    @staticmethod
    def _state_fluents_on_predicates(state, predicate_names):
        return frozenset(
            str(atom)
            for atom in state.get_atoms(ignore_static=False, ignore_derived=True)
            if atom.get_predicate().get_name() in predicate_names
        )

    @staticmethod
    def transfer_state(source_state, target_problem):
        """Reconstruct a pymimir State on target_problem that matches source_state's atoms.

        Both problems must share the same object and predicate names (e.g. same problem file,
        different domain file). Static predicates in the target domain are kept from the
        target problem definition and are not copied from source_state.
        """

        adv_problem = target_problem._advanced_problem

        fluent_pred_by_name = {
            p.get_name(): p for p in adv_problem.get_domain().get_fluent_predicates()
        }
        obj_by_name = {o.get_name(): o for o in adv_problem.get_objects()}

        target_adv_atoms = []
        for atom in source_state.get_atoms(ignore_static=False, ignore_derived=True):
            pred_name = atom.get_predicate().get_name()
            fluent_pred = fluent_pred_by_name.get(pred_name)
            if fluent_pred is None:
                continue  # static in target (shouldn't happen with dummy action)

            obj_names = [t.get_name() for t in atom.get_terms()]
            objs = ObjectList([obj_by_name[n] for n in obj_names])
            adv_atom = adv_problem.get_or_create_ground_atom(fluent_pred, objs)
            target_adv_atoms.append(adv_atom)

        fluent_list = FluentGroundAtomList(target_adv_atoms)
        (
            adv_state,
            _,
        ) = target_problem._search_context.get_state_repository().get_or_create_state(
            fluent_list, []
        )

        return State(adv_state, target_problem)

    @staticmethod
    def _extract_action_blocks(domain_text: str) -> list[str]:
        blocks: list[str] = []
        i = 0
        while True:
            start = domain_text.find("(:action", i)
            if start == -1:
                break

            depth = 0
            end = None
            for j in range(start, len(domain_text)):
                ch = domain_text[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break

            if end is None:
                raise ValueError("Unbalanced parentheses while parsing :action block")

            blocks.append(domain_text[start:end].strip())
            i = end

        return blocks

    @staticmethod
    def _find_balanced_expr(text: str, start_idx: int) -> tuple[str, int]:
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

    @staticmethod
    def _parse_typed_parameters(
        param_expr: str,
    ) -> tuple[list[str], list[tuple[str, str]]]:
        tokens = param_expr.strip()[1:-1].split()
        vars_out: list[str] = []
        typed_vars: list[tuple[str, str]] = []

        pending: list[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "-" and i + 1 < len(tokens):
                type_name = tokens[i + 1]
                for var_name in pending:
                    vars_out.append(var_name)
                    typed_vars.append((var_name, type_name))
                pending = []
                i += 2
            else:
                pending.append(tok)
                i += 1

        for var_name in pending:
            vars_out.append(var_name)

        return vars_out, typed_vars

    @classmethod
    def _replace_equality_with_predicate(cls, expr: str) -> str:
        return re.sub(
            r"\(\s*=\s+([^\s()]+)\s+([^\s()]+)\s*\)",
            rf"({cls.EQUALITY_PREDICATE_NAME} \1 \2)",
            expr,
        )

    @staticmethod
    def _domain_uses_equality(domain_text: str) -> bool:
        req_match = re.search(r"\(:requirements([^)]*)\)", domain_text, re.IGNORECASE)
        return bool(req_match and ":equality" in req_match.group(1).lower())

    @staticmethod
    def _inject_reflexive_atoms(problem_text: str, predicate_names: list[str]) -> str:
        """Inject `(p X X)` atoms for every predicate name in *predicate_names* and
        every object in the problem's :objects block, into the :init section.
        Works for both typed and untyped :objects.
        """
        if not predicate_names:
            return problem_text
        objects_match = re.search(r"\(:objects(.*?)\)", problem_text, re.DOTALL)
        if not objects_match:
            return problem_text
        tokens = objects_match.group(1).split()
        object_names: list[str] = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "-" and i + 1 < len(tokens):
                i += 2
            else:
                object_names.append(tokens[i])
                i += 1
        if not object_names:
            return problem_text
        eq_atoms = sorted(
            {
                f"({name} {obj} {obj})"
                for name in predicate_names
                for obj in object_names
            }
        )
        goal_pos = problem_text.find("(:goal")
        init_close = (
            problem_text.rfind(")", 0, goal_pos)
            if goal_pos != -1
            else problem_text.rfind(")")
        )
        if init_close == -1:
            return problem_text
        extra = "\n        " + "\n        ".join(eq_atoms)
        return problem_text[:init_close] + extra + "\n    " + problem_text[init_close:]

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

    @staticmethod
    def _inject_extra_preconditions(precondition_expr: str, extras: list[str]) -> str:
        pre = precondition_expr.strip()
        if not extras:
            return pre

        if pre in {"()", "(and)"}:
            return "(and " + " ".join(extras) + ")"

        if pre.startswith("(and") and pre.endswith(")"):
            inner = pre[4:-1].strip()
            if inner:
                return "(and " + " ".join(extras) + " " + inner + ")"
            return "(and " + " ".join(extras) + ")"

        return "(and " + " ".join(extras) + " " + pre + ")"

    @classmethod
    def _transform_original_action(
        cls, action_block: str, replace_equality: bool = True
    ) -> str:
        m = re.match(r"\(\s*:action\s+(\S+)", action_block)
        if m is None:
            raise ValueError("Failed to parse action name")
        action_name = m.group(1)

        params_idx = action_block.find(":parameters")
        pre_idx = action_block.find(":precondition")
        eff_idx = action_block.find(":effect")

        if params_idx == -1 or pre_idx == -1 or eff_idx == -1:
            raise ValueError(
                f"Action '{action_name}' is missing parameters/precondition/effect"
            )

        param_expr, _ = cls._find_balanced_expr(action_block, params_idx)
        pre_expr, _ = cls._find_balanced_expr(action_block, pre_idx)
        eff_expr, _ = cls._find_balanced_expr(action_block, eff_idx)

        params, typed_params = cls._parse_typed_parameters(param_expr)
        typed_preconditions = [
            f"({type_name} {var_name})" for var_name, type_name in typed_params
        ]

        if replace_equality:
            pre_expr = cls._replace_equality_with_predicate(pre_expr)
            eff_expr = cls._replace_equality_with_predicate(eff_expr)
        pre_expr = cls._inject_extra_preconditions(pre_expr, typed_preconditions)

        params_str = " ".join(params)
        return (
            f"(:action {action_name}\n"
            f"  :parameters ({params_str})\n"
            f"  :precondition {pre_expr}\n"
            f"  :effect {eff_expr}\n"
            f")"
        )
