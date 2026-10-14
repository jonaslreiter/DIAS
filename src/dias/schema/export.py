"""A trained model's learned domain as PDDL: written beside each checkpoint, and what validation scores."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from dias.pddl import text as pddl_text
from dias.pddl.transpiler import PDDLTranspiler
from dias.pddl.untype import extract_domain_header

from .decoding import derive_schema
from .domain import (
    EQUALITY,
    EQUALITY_BEFORE_ARGUMENT,
    DomainSchema,
    strips_plus_arguments,
)


def read_learned_schemas(
    model, schema: DomainSchema, datamodule_hparams: Mapping, threshold: float
) -> PDDLTranspiler:
    """The model's lifted schemas as learned, binarized at `threshold`. A model with state-dependent effects has preconditions only, and every active slot as a parameter."""
    logits = model.schema_logits()
    strips_plus = datamodule_hparams.get("strips_plus_actions", False)
    add, delete, positive, negative, active_slots = derive_schema(
        effect_logits=logits.effects,
        precondition_logits=logits.preconditions,
        slot_weight_logits=logits.slot_weight_logits,
        force_action_args=model.force_action_args,
        action_arguments=datamodule_hparams.get("action_arguments", False),
        strips_plus_actions=strips_plus,
        strips_plus_positions=(
            strips_plus_arguments(datamodule_hparams["domain_path"])
            if strips_plus
            else {}
        ),
        action_names=list(schema.action_names),
        action_arities=list(schema.action_arities),
        num_slots=model.num_slots,
        threshold=threshold,
    )
    state_dependent = add is None
    if state_dependent:
        add = delete = np.zeros_like(positive)
    transpiler = PDDLTranspiler(
        add_effects=add,
        delete_effects=delete,
        positive_preconditions=positive,
        negative_preconditions=negative,
        predicate_names=list(schema.relation_names),
        predicate_arities=list(schema.relation_arities),
        action_names=list(schema.action_names),
    )
    transpiler.generate_action_schemas()
    if state_dependent:
        for action_schema, active in zip(transpiler.action_schemas, active_slots):
            action_schema.parameters = [int(i) for i in np.flatnonzero(active)]
    return transpiler


def learned_domain(
    model, schema: DomainSchema, datamodule_hparams: Mapping, threshold: float
) -> str:
    """The learned schemas as an untyped PDDL domain over the original's predicates. Unless the model learned equality, every schema requires its parameters to be distinct."""
    learned_equality = datamodule_hparams.get(
        "include_equality", EQUALITY_BEFORE_ARGUMENT
    )
    transpiler = read_learned_schemas(model, schema, datamodule_hparams, threshold)
    if not learned_equality:
        transpiler.require_distinct(EQUALITY)
    header = extract_domain_header(
        Path(datamodule_hparams["domain_path"]).read_text(),
        predicate_names=transpiler.predicate_names,
        predicate_arities=transpiler.predicate_arities,
    )
    domain = (
        header
        + "\n\n"
        + "".join(str(action_schema) for action_schema in transpiler.action_schemas)
        + ")\n"
    )
    return domain if learned_equality else pddl_text.require_equality(domain)
