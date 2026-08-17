"""The shape of a planning domain: its relations, actions, and objects.

Everything downstream -- tensor layout, model dimensions, schema transpilation,
validation -- is a function of these lists. Keeping them in one immutable value
means a consumer that only needs metadata does not have to build a datamodule,
sample transitions, or touch a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

DEFAULT_STATICS = ("number", "object")


@dataclass(frozen=True)
class DomainSchema:
    """Relation, action and object metadata for one (domain, problem) pair.

    ``relation_*`` covers only the predicates represented as tensor channels:
    arity <= 2, after the static filter. ``action_arg_positions`` is the
    supervision projection -- empty lists when action arguments are unobserved.
    """

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
        include_statics: bool = False,
        default_statics: Sequence[str] | None = None,
        action_arg_positions: Sequence[Sequence[int]] | None = None,
    ) -> "DomainSchema":
        """Build from the raw dict `generate_samples.get_metadata` returns."""
        from dias.data.generate_samples import filter_predicates_for_tensor_relations

        relations = filter_predicates_for_tensor_relations(
            metadata["predicates"],
            include_statics=include_statics,
            default_statics=list(default_statics or DEFAULT_STATICS),
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
        include_statics: bool = False,
        default_statics: Sequence[str] | None = None,
        action_arguments: bool = False,
        strips_plus_actions: bool = False,
    ) -> "DomainSchema":
        import pymimir

        from dias.data.generate_samples import get_metadata

        domain = pymimir.Domain(Path(domain_path))
        problem = pymimir.Problem(domain, Path(problem_path))
        metadata = get_metadata(domain, problem)
        return cls.from_metadata(
            metadata,
            include_statics=include_statics,
            default_statics=default_statics,
            action_arg_positions=_argument_positions(
                metadata["actions"],
                domain_path=domain_path,
                action_arguments=action_arguments,
                strips_plus_actions=strips_plus_actions,
            ),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping) -> "DomainSchema":
        """Rebuild from the datamodule hyper-parameters stored in a checkpoint."""
        hparams = checkpoint["datamodule_hyper_parameters"]
        return cls.from_pddl(
            hparams["domain_path"],
            hparams["problem_path"],
            include_statics=hparams.get("include_statics", False),
            default_statics=hparams.get("default_statics"),
            action_arguments=hparams.get("action_arguments", False),
            strips_plus_actions=hparams.get("strips_plus_actions", False),
        )


def _argument_positions(
    actions: Sequence[Mapping],
    *,
    domain_path: str | Path,
    action_arguments: bool,
    strips_plus_actions: bool,
) -> list[tuple[int, ...]]:
    """Which argument positions of each action are observed."""
    if not action_arguments:
        return [() for _ in actions]
    if strips_plus_actions:
        from dias.data.datamodule import PddlGraphDataModule

        free = PddlGraphDataModule._load_strips_plus_yaml(domain_path)
        return [tuple(sorted(free.get(a["name"], []))) for a in actions]
    return [tuple(range(a["arity"])) for a in actions]
