from __future__ import annotations

from pathlib import Path

import pymimir
import torch

from dias.data.datamodule import PddlGraphDataModule
from dias.models.dias import DIAS
from dias.pddl import text as pddl_text
from dias.pddl.transpiler import PDDLTranspiler
from dias.pddl.untype import extract_domain_header, strip_problem_types, untype_action
from dias.schema import DomainSchema

from .artifacts import PddlWorkspace
from .successors import (
    compare_successors,
    sample_reachable_states,
    transfer_states,
)
from .utils import process_model_logits, resolve_best_checkpoint


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
        has_equality = pddl_text.domain_uses_equality(original_domain_text)

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

        # A merged untyped domain: transformed original actions + learned actions.
        header = extract_domain_header(
            original_domain_text,
            rel_names=pddl_transpiler.predicate_names,
            predicate_arities=pddl_transpiler.predicate_arities,
            keep_equality_requirement=False,
        )
        equality_predicate = self.EQUALITY_PREDICATE_NAME if has_equality else None
        transformed_original_actions = [
            untype_action(action_block, equality_predicate_name=equality_predicate)
            for action_block in pddl_text.extract_action_blocks(original_domain_text)
        ]
        merged_domain_text = (
            header
            + "\n\n"
            + "\n\n".join(transformed_original_actions)
            + "\n\n"
            + "".join(str(schema) for schema in pddl_transpiler.action_schemas)
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
            ),
        )

        self.merged_domain = pymimir.Domain(Path(self.merged_domain_filename))
        self.merged_problem = pymimir.Problem(
            self.merged_domain, Path(self.merged_problem_filename)
        )

        self.original_domain = pymimir.Domain(Path(datamodule_hparams["domain_path"]))
        self.original_problem = pymimir.Problem(
            self.original_domain, Path(problem_path)
        )

        self.state_set = None

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
