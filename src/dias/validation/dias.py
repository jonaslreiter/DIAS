"""Scoring a DIAS checkpoint: the lifted schemas read off it are merged into the original domain, and the two sets of actions are compared successor by successor."""

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
from .decoding import derive_schema
from .successors import (
    LEARNED_ACTION_PREFIX,
    compare_successors,
    resolve_sample_seed,
    sample_reachable_states,
    transfer_states,
)

logger = logging.getLogger(__name__)


def read_learned_schemas(
    run, has_equality: bool, threshold: float, injective: bool = True
) -> PDDLTranspiler:
    """Binarize a trained run's logits into action schemas.

    Returns the transpiler, which also carries the predicate names and arities a generated domain must declare. This is the one path from a checkpoint to lifted schemas: `DiasValidator` merges what it returns into the original domain, and an analysis script reads the same schemas without building a validator.

    With `injective`, every schema also requires its parameters to be distinct, written with `EQUALITY_PREDICATE`, which the transpiler then declares and a validator must give its reflexive facts. That is how the model binds them: it is trained on injective bindings only, so nothing it learns says what a binding that repeats an object does, while in PDDL such a binding is applicable unless a precondition rules it out. No domain in `pddl/` has an applicable grounding that repeats an object, so this never removes a successor a correct schema has. Off, the schema is read as learned, which is what a count of its literals wants.
    """
    hparams = run.datamodule_hparams
    logits = run.model.schema_logits()
    (
        add_effects,
        delete_effects,
        positive_preconditions,
        negative_preconditions,
        _,
    ) = derive_schema(
        effect_logits=logits.effects,
        precondition_logits=logits.preconditions,
        slot_weight_logits=logits.slot_weight_logits,
        force_action_args=run.model.force_action_args,
        action_arguments=hparams.get("action_arguments", False),
        strips_plus_actions=hparams.get("strips_plus_actions", False),
        strips_plus_positions=(
            strips_plus_arguments(run.domain_path)
            if hparams.get("strips_plus_actions", False)
            else {}
        ),
        action_names=list(run.schema.action_names),
        action_arities=list(run.schema.action_arities),
        num_slots=run.model.num_slots,
        random_flip=False,
        threshold=threshold,
    )

    names = list(run.schema.relation_names)
    arities = list(run.schema.relation_arities)
    if not has_equality:
        # Without :equality the "=" channel has no predicate to write to.
        keep = [i for i, name in enumerate(names) if name != "="]
        (
            add_effects,
            delete_effects,
            positive_preconditions,
            negative_preconditions,
        ) = (
            array[:, keep, :, :]
            for array in (
                add_effects,
                delete_effects,
                positive_preconditions,
                negative_preconditions,
            )
        )
        names = [names[i] for i in keep]
        arities = [arities[i] for i in keep]

    transpiler = PDDLTranspiler(
        add_effects=add_effects,
        delete_effects=delete_effects,
        positive_preconditions=positive_preconditions,
        negative_preconditions=negative_preconditions,
        predicate_names=[
            EQUALITY_PREDICATE if has_equality and n == "=" else n for n in names
        ],
        predicate_arities=arities,
        action_names=list(run.schema.action_names),
    )
    transpiler.generate_action_schemas()
    if has_equality:
        # The original actions are compiled to it, whether or not the model had a channel for `=`.
        transpiler.declare_predicate(EQUALITY_PREDICATE, 2)
    if injective:
        transpiler.require_distinct(EQUALITY_PREDICATE)
    return transpiler


class DiasValidator:
    def __init__(
        self, model_id, logs_path, problem_path=None, threshold=0.5, injective=True
    ):
        self.model_id = model_id
        self.threshold = threshold
        self.injective = injective

        run = load_run(model_id, logs_path)
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
                # The distinctness `injective` writes needs the reflexive facts whether or not the domain has :equality.
                equality_predicate_name=(
                    EQUALITY_PREDICATE if has_equality or injective else None
                ),
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
        """The learned schemas under the prefix that tells them apart from the originals in the merged domain."""
        transpiler = read_learned_schemas(
            run, has_equality, threshold, injective=self.injective
        )
        for schema in transpiler.action_schemas:
            schema.name = f"{LEARNED_ACTION_PREFIX}{schema.name}"
        logger.info(
            "Read %d schemas off the checkpoint at threshold %g: %s.",
            len(transpiler.action_schemas),
            threshold,
            ", ".join(
                f"{schema.name} ({len(schema.preconditions)} preconditions, {len(schema.effects)} effects)"
                for schema in transpiler.action_schemas
            ),
        )
        return transpiler

    @staticmethod
    def _merge_domains(domain_text: str, has_equality: bool, transpiler) -> str:
        """One untyped domain holding the original actions and the learned ones."""
        header = extract_domain_header(
            domain_text,
            predicate_names=transpiler.predicate_names,
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
