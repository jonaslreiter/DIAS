"""The relations, actions and objects of a (domain, problem) pair."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from dias.pddl.text import is_equality_predicate

# Unary predicates pymimir gives every object; they get no channel.
OMITTED_PREDICATES = ("number", "object")

# PDDL's equality, as pymimir names it.
EQUALITY = "="


@dataclass(frozen=True)
class DomainSchema:
    """The metadata of one (domain, problem) pair. ``relation_*`` lists the predicates that are channels of the adjacency tensor, and ``action_arg_positions`` the argument positions the action labels name."""

    relation_names: tuple[str, ...]
    relation_arities: tuple[int, ...]
    action_names: tuple[str, ...]
    action_arities: tuple[int, ...]
    action_arg_positions: tuple[tuple[int, ...], ...]
    object_names: Mapping[str, str] = field(default_factory=dict)
    object_name_to_idx: Mapping[str, int] = field(default_factory=dict)
    domain_name: str = ""

    @property
    def num_relations(self) -> int:
        return len(self.relation_arities)

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    @property
    def num_objects(self) -> int:
        return len(self.object_names)

    @property
    def max_action_arity(self) -> int:
        return max(self.action_arities, default=0)

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping,
        *,
        include_equality: bool = False,
        action_arg_positions: Sequence[Sequence[int]] | None = None,
    ) -> "DomainSchema":
        """From what `read_metadata` returns."""
        relations = tensor_relations(
            metadata["predicates"], include_equality=include_equality
        )
        actions = metadata["actions"]
        if action_arg_positions is None:
            action_arg_positions = [() for _ in actions]
        return cls(
            relation_names=tuple(p["name"] for p in relations),
            relation_arities=tuple(p["arity"] for p in relations),
            action_names=tuple(a["name"] for a in actions),
            action_arities=tuple(a["arity"] for a in actions),
            action_arg_positions=tuple(tuple(p) for p in action_arg_positions),
            object_names=dict(metadata["objects"]),
            object_name_to_idx=dict(metadata["object_name_to_idx"]),
            domain_name=metadata.get("domain_name", ""),
        )

    @classmethod
    def from_pddl(
        cls,
        domain_path: str | Path,
        problem_path: str | Path,
        *,
        include_equality: bool = False,
        action_arguments: bool = False,
        strips_plus_actions: bool = False,
    ) -> "DomainSchema":
        import pymimir

        domain = pymimir.Domain(Path(domain_path))
        problem = pymimir.Problem(domain, Path(problem_path))
        metadata = read_metadata(domain, problem)
        return cls.from_metadata(
            metadata,
            include_equality=include_equality,
            action_arg_positions=argument_positions(
                metadata["actions"],
                domain_path=domain_path,
                action_arguments=action_arguments,
                strips_plus_actions=strips_plus_actions,
            ),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping) -> "DomainSchema":
        """From the datamodule hyper-parameters a checkpoint stores."""
        return cls.from_run(checkpoint["datamodule_hyper_parameters"])

    @classmethod
    def from_run(
        cls, hparams: Mapping, relation_arities: Sequence[int] | None = None
    ) -> "DomainSchema":
        """From a run's datamodule hyper-parameters; given the model's `relation_arities`, refused if the domain file no longer gives them."""
        schema = cls.from_pddl(
            hparams["domain_path"],
            hparams["problem_path"],
            include_equality=hparams.get("include_equality", False),
            action_arguments=hparams.get("action_arguments", False),
            strips_plus_actions=hparams.get("strips_plus_actions", False),
        )
        if relation_arities is not None and list(schema.relation_arities) != list(
            relation_arities
        ):
            raise ValueError(
                f"The run was trained on relations of arities {list(relation_arities)}, but {hparams['domain_path']} gives {dict(zip(schema.relation_names, schema.relation_arities))}."
            )
        return schema


def argument_positions(
    actions: Sequence[Mapping],
    *,
    domain_path: str | Path,
    action_arguments: bool,
    strips_plus_actions: bool,
) -> list[tuple[int, ...]]:
    """Per action, the argument positions its labels name."""
    if not action_arguments:
        return [() for _ in actions]
    if strips_plus_actions:
        free = strips_plus_arguments(domain_path)
        return [tuple(sorted(free.get(a["name"], []))) for a in actions]
    return [tuple(range(a["arity"])) for a in actions]


def read_metadata(domain, problem) -> dict:
    """The objects, predicates and actions of a pymimir domain and problem."""
    objects = {str(o.get_index()): o.get_name() for o in problem.get_objects()}
    object_name_to_idx = {o.get_name(): o.get_index() for o in problem.get_objects()}
    predicates = sorted(
        (
            {"name": p.get_name(), "arity": p.get_arity(), "static": p.is_static()}
            for p in domain.get_predicates()
        ),
        key=lambda p: p["name"],
    )
    actions = [
        {"name": a.get_name(), "arity": a.get_arity()} for a in domain.get_actions()
    ]
    return {
        "domain_name": domain.get_name(),
        "objects": objects,
        "object_name_to_idx": object_name_to_idx,
        "num_objects": len(objects),
        "predicates": predicates,
        "actions": actions,
    }


def tensor_relations(
    predicates: Sequence[Mapping], *, include_equality: bool = False
) -> list[Mapping]:
    """The predicates that are channels of the adjacency tensor: arity at most two, not in `OMITTED_PREDICATES`, and equality only if included."""
    return [
        p
        for p in predicates
        if p["arity"] <= 2
        and not (p["static"] and p["name"] in OMITTED_PREDICATES)
        and (include_equality or not is_equality_predicate(p["name"], p["arity"]))
    ]


def strips_plus_arguments(domain_path: str | Path) -> dict[str, list[int]]:
    """Per action, the argument positions STRIPS+ labels name, from the domain's strips+.yaml."""
    import yaml

    yaml_path = Path(domain_path).parent / "strips+.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"No strips+.yaml file found in {Path(domain_path).parent}. Required when strips_plus_actions=True."
        )
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f) or {}
    return {name: (lst[0] if lst else []) for name, lst in yaml_data.items()}
