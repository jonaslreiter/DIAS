"""A model's learned domain as PDDL."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from dias.pddl import text as pddl_text
from dias.pddl.transpiler import PDDLTranspiler
from dias.pddl.untype import extract_domain_header

from .derivation import derive_schema
from .domain import EQUALITY, DomainSchema, strips_plus_arguments


def read_learned_schemas(
    model, schema: DomainSchema, datamodule_hparams: Mapping, threshold: float
) -> PDDLTranspiler:
    """The model's lifted schemas derived at `threshold`, with equality written as `equal`. Those of an MLP-effects model have preconditions only and every active slot as a parameter."""
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
    mlp_effects = add is None
    if mlp_effects:
        add = delete = np.zeros_like(positive)
    transpiler = PDDLTranspiler(
        add_effects=add,
        delete_effects=delete,
        positive_preconditions=positive,
        negative_preconditions=negative,
        predicate_names=[
            (
                pddl_text.EQUALITY_PREDICATE
                if pddl_text.is_equality_predicate(name, arity)
                else name
            )
            for name, arity in zip(schema.relation_names, schema.relation_arities)
        ],
        predicate_arities=list(schema.relation_arities),
        action_names=list(schema.action_names),
    )
    transpiler.generate_action_schemas()
    if mlp_effects:
        for action_schema, active in zip(transpiler.action_schemas, active_slots):
            action_schema.parameters = [int(i) for i in np.flatnonzero(active)]
    return transpiler


def learned_domain(
    model, schema: DomainSchema, datamodule_hparams: Mapping, threshold: float
) -> str:
    """The learned schemas as an untyped PDDL domain over the original's predicates, the types as unary predicates. Unless equality had a channel, every schema requires its parameters to be distinct."""
    learned_equality = datamodule_hparams.get("include_equality", False)
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
