from __future__ import annotations

import random
import re
import tempfile
from pathlib import Path

import pymimir
import torch
from tqdm import tqdm

from dias.data.datamodule import PddlGraphDataModule
from dias.data.generate_samples import generate_bfs
from dias.models.dias import DIAS
from dias.pddl.states import state_fluents, transfer_state
from dias.schema import DomainSchema

from .transpiler import PDDLTranspiler
from .utils import (
    extract_domain_header,
    process_model_logits,
    resolve_best_checkpoint,
    strip_problem_types,
)


class ModelValidation:
    LEARNED_ACTION_PREFIX = "learned_"
    EQUALITY_PREDICATE_NAME = "eq"

    def __init__(self, model_id, logs_path_abs, problem_path=None, threshold=0.5):
        self.model_id = model_id
        self.logs_path_abs = logs_path_abs
        self.threshold = threshold

        ckpt_dir = Path(f"{logs_path_abs}/dias/{model_id}/checkpoints")
        self.ckpt_file = str(resolve_best_checkpoint(ckpt_dir))

        ckpt = torch.load(self.ckpt_file, map_location="cpu")
        # Drop legacy hparams no longer accepted by PddlGraphDataModule so old
        # checkpoints (pre-sampling-refactor) still load.
        _legacy_keys = {
            "method",
            "num_val",
            "num_test",
            "balanced_min_per_label",
            "balanced_max_steps",
            "inapplicable_actions_ratio",
        }
        datamodule_hparams = {
            k: v
            for k, v in ckpt["datamodule_hyper_parameters"].items()
            if not k.startswith("_") and k not in _legacy_keys
        }
        # Relative paths in a checkpoint were relative to the working directory
        # the run was launched from, which is the repository root.
        for key in ("domain_path", "problem_path", "pt_file"):
            if datamodule_hparams.get(key) is not None:
                datamodule_hparams[key] = str(Path(datamodule_hparams[key]).resolve())

        self.schema = DomainSchema.from_pddl(
            datamodule_hparams["domain_path"],
            datamodule_hparams["problem_path"],
            include_statics=datamodule_hparams.get("include_statics", False),
            default_statics=datamodule_hparams.get("default_statics"),
            action_arguments=datamodule_hparams.get("action_arguments", False),
            strips_plus_actions=datamodule_hparams.get("strips_plus_actions", False),
        )

        self.model = DIAS.load_from_checkpoint(self.ckpt_file, map_location="cpu")
        self.model.to("cpu")
        self.model.eval()

        free_pos_map = PddlGraphDataModule._load_strips_plus_yaml(
            datamodule_hparams["domain_path"],
        )
        action_arguments = datamodule_hparams.get("action_arguments", False)
        strips_plus_actions = datamodule_hparams.get("strips_plus_actions", False)

        precondition_logits = self.model.graph_edit.effects.detach().cpu()
        effect_logits = self.model.graph_edit.preconditions.detach().cpu()
        slot_weights = self.model.graph_edit.slot_weight_logits.detach().cpu()

        add_eff, del_eff, pos_prec, neg_prec, active_slots = process_model_logits(
            preconditions_logits=precondition_logits,
            effect_logits=effect_logits,
            slot_weight_logits=slot_weights,
            force_action_args=self.model.force_action_args,
            action_arguments=action_arguments,
            strips_plus_actions=strips_plus_actions,
            free_pos_map=free_pos_map,
            action_names=list(self.schema.action_names),
            action_arities=list(self.schema.action_arities),
            num_slots=self.model.num_slots,
            random_flip=False,
            threshold=threshold,
        )

        original_domain_path = datamodule_hparams["domain_path"]
        original_domain_text = Path(original_domain_path).read_text()
        has_equality = self._domain_uses_equality(original_domain_text)

        relation_names = list(self.schema.relation_names)
        relation_arities = list(self.schema.relation_arities)
        if not has_equality:
            keep = [i for i, n in enumerate(relation_names) if n != "="]
            add_eff = add_eff[:, keep, :, :]
            del_eff = del_eff[:, keep, :, :]
            pos_prec = pos_prec[:, keep, :, :]
            neg_prec = neg_prec[:, keep, :, :]
            relation_names = [relation_names[i] for i in keep]
            relation_arities = [relation_arities[i] for i in keep]

        transpiler_predicate_names = [
            self.EQUALITY_PREDICATE_NAME if has_equality and n == "=" else n
            for n in relation_names
        ]

        pddl_transpiler = PDDLTranspiler(
            add_effects=add_eff,
            delete_effects=del_eff,
            positive_preconditions=pos_prec,
            negative_preconditions=neg_prec,
            predicate_names=transpiler_predicate_names,
            predicate_arities=relation_arities,
            action_names=list(self.schema.action_names),
        )
        pddl_transpiler.generate_action_schemas()

        for schema in pddl_transpiler.action_schemas:
            schema.name = f"{self.LEARNED_ACTION_PREFIX}{schema.name}"

        # Create a merged untyped domain: transformed original actions + learned actions.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pddl", dir="/tmp", delete=False
        ) as f:
            header, _ = extract_domain_header(
                original_domain_path,
                rel_names=pddl_transpiler.predicate_names,
                predicate_arities=pddl_transpiler.predicate_arities,
                keep_equality_requirement=False,
            )
            original_actions = self._extract_action_blocks(original_domain_text)
            transformed_original_actions = [
                self._transform_original_action(
                    action_block, replace_equality=has_equality
                )
                for action_block in original_actions
            ]
            action_schemas_str = [
                str(action_schema) for action_schema in pddl_transpiler.action_schemas
            ]
            pddl_str = (
                header
                + "\n\n"
                + "\n\n".join(transformed_original_actions)
                + "\n\n"
                + "".join(action_schemas_str)
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
                )
            )
            print(f"Merged problem file: {self.merged_problem_filename}")

        self.merged_domain = pymimir.Domain(Path(self.merged_domain_filename))
        self.merged_problem = pymimir.Problem(
            self.merged_domain, Path(self.merged_problem_filename)
        )

        self.original_domain = pymimir.Domain(Path(datamodule_hparams["domain_path"]))
        self.original_problem = pymimir.Problem(
            self.original_domain, Path(problem_path)
        )

        self.state_set = None
        self._state_id_cache: dict[int, int] = {}

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
