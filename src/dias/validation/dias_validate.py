from __future__ import annotations

import logging
from pathlib import Path

import pymimir

from dias.pddl import text as pddl_text
from dias.pddl.transpiler import PDDLTranspiler
from dias.pddl.untype import (
    EQUALITY_PREDICATE,
    extract_domain_header,
    strip_problem_types,
    untype_action,
)
from dias.schema.domain import strips_plus_arguments

from .artifacts import PddlWorkspace
from .checkpoint import load_run
from .decoding import process_model_logits
from .successors import (
    LEARNED_ACTION_PREFIX,
    compare_successors,
    resolve_sample_seed,
    sample_reachable_states,
    transfer_states,
)

logger = logging.getLogger(__name__)


class ModelValidation:
    def __init__(self, model_id, logs_path_abs, problem_path=None, threshold=0.5):
        self.model_id = model_id
        self.threshold = threshold

        run = load_run(model_id, logs_path_abs)
        self.schema = run.schema
        self.model = run.model

        domain_text = Path(run.domain_path).read_text()
        has_equality = pddl_text.domain_uses_equality(domain_text)
        transpiler = self._read_learned_schemas(run, has_equality, threshold)

        self.workspace = PddlWorkspace()
        self.merged_domain_filename = self.workspace.write(
            "merged-domain.pddl",
            self._merge_domains(domain_text, has_equality, transpiler) + "\n\n",
        )
        self.merged_problem_filename = self.workspace.write(
            "merged-problem.pddl",
            strip_problem_types(
                Path(problem_path).read_text(),
                domain_text,
                equality_predicate_name=EQUALITY_PREDICATE if has_equality else None,
            ),
        )

        self.merged_domain = pymimir.Domain(Path(self.merged_domain_filename))
        self.merged_problem = pymimir.Problem(
            self.merged_domain, Path(self.merged_problem_filename)
        )
        self.original_domain = pymimir.Domain(Path(run.domain_path))
        self.original_problem = pymimir.Problem(
            self.original_domain, Path(problem_path)
        )
        self.state_set = None

    def _read_learned_schemas(self, run, has_equality: bool, threshold: float):
        """Binarize the model's logits and transpile them into action schemas.

        Returns the transpiler rather than the schemas, because it also carries
        the predicate names and arities the merged domain has to declare.
        """
        hparams = run.datamodule_hparams
        schema = self.model.schema_logits()
        add_eff, del_eff, pos_prec, neg_prec, _ = process_model_logits(
            effect_logits=schema.effects,
            precondition_logits=schema.preconditions,
            slot_weight_logits=schema.slot_weight_logits,
            force_action_args=self.model.force_action_args,
            action_arguments=hparams.get("action_arguments", False),
            strips_plus_actions=hparams.get("strips_plus_actions", False),
            free_pos_map=(
                strips_plus_arguments(run.domain_path)
                if hparams.get("strips_plus_actions", False)
                else {}
            ),
            action_names=list(self.schema.action_names),
            action_arities=list(self.schema.action_arities),
            num_slots=self.model.num_slots,
            random_flip=False,
            threshold=threshold,
        )

        names = list(self.schema.relation_names)
        arities = list(self.schema.relation_arities)
        if not has_equality:
            # Without :equality the "=" channel has no predicate to write to.
            keep = [i for i, name in enumerate(names) if name != "="]
            add_eff, del_eff, pos_prec, neg_prec = (
                array[:, keep, :, :] for array in (add_eff, del_eff, pos_prec, neg_prec)
            )
            names = [names[i] for i in keep]
            arities = [arities[i] for i in keep]

        transpiler = PDDLTranspiler(
            add_effects=add_eff,
            delete_effects=del_eff,
            positive_preconditions=pos_prec,
            negative_preconditions=neg_prec,
            predicate_names=[
                EQUALITY_PREDICATE if has_equality and n == "=" else n for n in names
            ],
            predicate_arities=arities,
            action_names=list(self.schema.action_names),
        )
        transpiler.generate_action_schemas()
        for schema in transpiler.action_schemas:
            schema.name = f"{LEARNED_ACTION_PREFIX}{schema.name}"
        logger.info(
            "Read %d schemas off the checkpoint at threshold %g: %s.",
            len(transpiler.action_schemas),
            threshold,
            ", ".join(
                f"{schema.name} ({len(schema.preconditions)} preconditions, "
                f"{len(schema.effects)} effects)"
                for schema in transpiler.action_schemas
            ),
        )
        return transpiler

    @staticmethod
    def _merge_domains(domain_text: str, has_equality: bool, transpiler) -> str:
        """One untyped domain holding the original actions and the learned ones."""
        header = extract_domain_header(
            domain_text,
            rel_names=transpiler.predicate_names,
            predicate_arities=transpiler.predicate_arities,
            keep_equality_requirement=False,
        )
        original_actions = [
            untype_action(
                block,
                equality_predicate_name=EQUALITY_PREDICATE if has_equality else None,
            )
            for block in pddl_text.extract_action_blocks(domain_text)
        ]
        return (
            header
            + "\n\n"
            + "\n\n".join(original_actions)
            + "\n\n"
            + "".join(str(schema) for schema in transpiler.action_schemas)
            + "\n)"
        )

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
