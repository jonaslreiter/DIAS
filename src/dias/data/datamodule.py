"""The training data: transitions sampled from one problem instance (paper, App. C.2)."""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightning as L
import pymimir
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from dias.data.datasets import PddlTransitionDataset, StratifiedActionIterableDataset
from dias.data.noise import ExpectedCount, NoiseModel, Ratio, RelationStats
from dias.data.splitting import (
    cap_per_action,
    check_every_action_survives,
    stratified_split,
)
from dias.hparams import from_mapping, to_mapping
from dias.schema import DomainSchema
from dias.schema.domain import (
    argument_positions,
    read_metadata,
    strips_plus_arguments,
)

from .generate_samples import generate_random_walk, samples_to_tensors

logger = logging.getLogger(__name__)


class PddlGraphDataModule(L.LightningDataModule):
    """Transitions of one problem instance, sampled by a random walk until every action has its quota, split by action into training and validation, and drawn in action-balanced batches without end.

    The training length is therefore ``trainer.max_steps``.
    """

    def __init__(
        self,
        domain_path: str,
        problem_path: str,
        seed: Optional[int] = None,
        include_equality: bool = False,
        action_arguments: bool = False,
        strips_plus_actions: bool = False,
        min_per_action_train: Optional[int] = None,
        min_per_action_valid: Optional[int] = None,
        max_per_action_train: Optional[int] = None,
        max_per_action_valid: Optional[int] = None,
        num_train: Optional[int] = None,
        num_valid: Optional[int] = None,
        max_walk_steps: int = 1_000_000,
        batch_size: int = 200,
        num_workers: int = 0,
        noise: NoiseModel | None = None,
    ):
        """
        Args:
            domain_path: PDDL domain whose action schemas are learned.
            problem_path: Instance the transitions are sampled from.
            seed: Seed of the walk and the split. Unset takes the run's seed.
            include_equality: Give equality (`=`, or a binary `eq` or `equal`) a channel. Without it, the learned domain requires every schema's parameters to be distinct.
            action_arguments: Label a transition with the action's arguments too, not only its name.
            strips_plus_actions: Label it with the arguments at the STRIPS+ positions of strips+.yaml only.
            min_per_action_train: N_min^tr, the training transitions per action the walk samples at least.
            min_per_action_valid: N_min^val, the same for validation. The ratio of the two splits the transitions.
            max_per_action_train: N_max^tr, the training transitions per action kept at most.
            max_per_action_valid: N_max^val, the same for validation. Set both maxima or neither.
            num_train: Training transitions the walk samples at least, in total.
            num_valid: The same for validation.
            max_walk_steps: Steps after which the walk stops, met quota or not.
            batch_size: Transitions per batch.
            num_workers: Data loader worker processes.
            noise: Corruption of the training states. Unset trains on clean transitions.
        """
        super().__init__()
        self.save_hyperparameters()

        self.domain_path = domain_path
        self.problem_path = problem_path
        self.seed = torch.initial_seed() if seed is None else seed
        self.hparams["seed"] = self.seed
        self.max_walk_steps = max_walk_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.action_arguments = action_arguments
        self.strips_plus_actions = strips_plus_actions
        self.include_equality = include_equality

        self.quota = SampleQuota(
            min_per_action_train=min_per_action_train,
            min_per_action_valid=min_per_action_valid,
            max_per_action_train=max_per_action_train,
            max_per_action_valid=max_per_action_valid,
            num_train=num_train,
            num_valid=num_valid,
        )
        self.noise = (
            None if noise is None else from_mapping(noise, (Ratio, ExpectedCount))
        )
        self.hparams["noise"] = None if self.noise is None else to_mapping(self.noise)

        logger.info("Loading domain %s, problem %s", domain_path, problem_path)
        self.domain = pymimir.Domain(Path(domain_path))
        self.problem = pymimir.Problem(self.domain, Path(problem_path))
        self.metadata = read_metadata(self.domain, self.problem)
        self.schema = DomainSchema.from_metadata(
            self.metadata,
            include_equality=include_equality,
            action_arg_positions=argument_positions(
                self.metadata["actions"],
                domain_path=domain_path,
                action_arguments=action_arguments,
                strips_plus_actions=strips_plus_actions,
            ),
        )
        self.relation_arities: list[int] = list(self.schema.relation_arities)

        self._base: Optional[PddlTransitionDataset] = None
        self._train_ds = None
        self._valid_ds = None
        self._train_idx: list[int] = []
        self._valid_idx: list[int] = []

    @property
    def base(self) -> PddlTransitionDataset:
        assert self._base is not None, "Call .setup() first."
        return self._base

    @property
    def num_relations(self) -> int:
        return self.schema.num_relations

    @property
    def num_objects(self) -> int:
        return self.schema.num_objects

    @property
    def num_actions(self) -> int:
        return self.schema.num_actions

    @property
    def max_action_arity(self) -> int:
        return self.schema.max_action_arity

    @property
    def action_arities(self) -> list[int]:
        return list(self.schema.action_arities)

    @property
    def action_arg_positions(self) -> list[list[int]]:
        return [list(p) for p in self.schema.action_arg_positions]

    @property
    def relation_names(self) -> list[str]:
        return list(self.schema.relation_names)

    @property
    def action_names(self) -> list[str]:
        return list(self.schema.action_names)

    @property
    def object_names(self) -> dict:
        return dict(self.schema.object_names)

    @property
    def dataset_size(self) -> int:
        return len(self.base)

    def setup(self, stage: Optional[str] = None) -> None:
        samples = self._sample_transitions()
        self._label_actions(samples)
        data = samples_to_tensors(
            samples, self.metadata, include_equality=self.include_equality
        )
        self._base = PddlTransitionDataset(data)

        total = len(self._base)
        self._train_idx, self._valid_idx = stratified_split(
            self._base.action_indices, *self.quota.split_weights, self.seed
        )
        self.relation_stats = self._measure_relations()
        if total > 0:
            self._log_action_distribution()

        train_noisy_s, train_noisy_ns = self._build_train_noise(self._train_idx)
        check_every_action_survives(
            self._base.action_indices,
            self._base.action_names,
            {"train": self._train_idx, "valid": self._valid_idx},
        )
        self._train_ds = self._make_stratified(
            self._train_idx, 2, train_noisy_s, train_noisy_ns
        )
        self._valid_ds = self._make_stratified(self._valid_idx, 3)
        logger.info(
            "%d samples: train=%d, valid=%d.",
            total,
            len(self._train_idx),
            len(self._valid_idx),
        )

    def _sample_transitions(self) -> list[dict]:
        random.seed(self.seed)
        logger.info(
            "Domain %s: %d objects, actions %s.",
            self.metadata["domain_name"],
            self.metadata["num_objects"],
            [a["name"] for a in self.metadata["actions"]],
        )
        logger.info(
            "Sampling transitions: at least %d per action%s, walking at most %d steps.",
            self.quota.min_total,
            f" and {self.quota.num_total} in total" if self.quota.num_total else "",
            self.max_walk_steps,
        )
        samples = generate_random_walk(
            problem=self.problem,
            min_total_samples=self.quota.num_total,
            min_per_action=self.quota.min_total,
            max_steps=self.max_walk_steps,
        )
        if self.quota.cap is None:
            return samples
        return cap_per_action(samples, self.quota.cap)

    def _label_actions(self, samples: list[dict]) -> None:
        """Hide the arguments the action labels do not name, as -1."""
        if self.action_arguments and self.strips_plus_actions:
            named = strips_plus_arguments(self.domain_path)
            for s in samples:
                allowed = named.get(s["action"], [])
                s["action_args"] = [
                    a if p in allowed else -1 for p, a in enumerate(s["action_args"])
                ]
        elif not self.action_arguments:
            for s in samples:
                s["action_args"] = [-1] * len(s["action_args"])

    def _build_train_noise(
        self, train_idx: list[int]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Noisy copies of the training states, or None for clean training."""
        if self.noise is None or not self.noise.enabled or not train_idx:
            return None, None
        logger.info("Train split gets %s", self.noise.describe(self.relation_stats))

        generator = torch.Generator().manual_seed(self.seed + 1)
        return tuple(
            self.noise.corrupt(source[train_idx], self.relation_stats, generator)
            for source in (self._base.states, self._base.next_states)
        )

    def _measure_relations(self) -> RelationStats:
        base = self._base
        static_names = {p["name"] for p in self.metadata["predicates"] if p["static"]}
        is_fluent = [name not in static_names for name in base.relation_names]
        if len(base) == 0:
            return RelationStats(
                self.relation_arities, [0.0] * len(base.relation_names), is_fluent
            )

        stats = RelationStats.measure(
            torch.cat([base.states, base.next_states], dim=0),
            self.relation_arities,
            is_fluent,
        )
        logger.info(
            "Per-relation true/false ratios: %s",
            {
                f"{name}/{arity}": f"{ratio:.4g}"
                for name, arity, ratio in zip(
                    base.relation_names, stats.arities, stats.true_ratio
                )
            },
        )
        return stats

    def _log_action_distribution(self) -> None:
        ds = self._base
        counts = torch.bincount(ds.action_indices, minlength=len(ds.action_names))
        logger.info(
            "Action frequencies: %s",
            {
                name: f"{count} ({100.0 * count / len(ds):.1f}%)"
                for name, count in zip(ds.action_names, counts.tolist())
            },
        )

    def _make_stratified(
        self,
        indices: list[int],
        seed_offset: int,
        noisy_states: Optional[torch.Tensor] = None,
        noisy_next_states: Optional[torch.Tensor] = None,
    ):
        if not indices:
            return []
        return StratifiedActionIterableDataset(
            base=self._base,
            indices=indices,
            batch_size=self.batch_size,
            seed=self.seed + seed_offset,
            noisy_states=noisy_states,
            noisy_next_states=noisy_next_states,
        )

    def _make_loader(self, dataset):
        return PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._make_loader(self._train_ds)

    def val_dataloader(self):
        return self._make_loader(self._valid_ds)


@dataclass(frozen=True)
class SampleQuota:
    """How many transitions the walk samples: at least ``min_per_action_*`` per action, or ``num_*`` in total, or both with the same train-to-validation ratio. At most ``max_per_action_*`` per action are kept."""

    min_per_action_train: int | None = None
    min_per_action_valid: int | None = None
    max_per_action_train: int | None = None
    max_per_action_valid: int | None = None
    num_train: int | None = None
    num_valid: int | None = None

    def __post_init__(self):
        if not self.has_min and not self.has_num:
            raise ValueError(
                "Nothing says how much to sample: set min_per_action_train and min_per_action_valid, or num_train and num_valid, or both."
            )
        if (
            self.has_min
            and min(self.min_per_action_train, self.min_per_action_valid) <= 0
        ):
            raise ValueError(
                "min_per_action_train and min_per_action_valid must be > 0."
            )
        if self.has_num and min(self.num_train, self.num_valid) <= 0:
            raise ValueError("num_train and num_valid must be > 0.")
        if (self.max_per_action_train is None) != (self.max_per_action_valid is None):
            raise ValueError(
                "max_per_action_train and max_per_action_valid must be set together."
            )
        if (
            self.has_max
            and min(self.max_per_action_train, self.max_per_action_valid) <= 0
        ):
            raise ValueError(
                "max_per_action_train and max_per_action_valid must be > 0."
            )
        if self.has_max and self.has_min and self.cap < self.min_total:
            raise ValueError(
                f"max_per_action_* must be >= min_per_action_*: max=({self.max_per_action_train},{self.max_per_action_valid}) vs min=({self.min_per_action_train},{self.min_per_action_valid})."
            )
        if (
            self.has_min
            and self.has_num
            and (
                self.min_per_action_train * self.num_valid
                != self.num_train * self.min_per_action_valid
            )
        ):
            raise ValueError(
                f"One pool cannot satisfy both ratios: min_per_action_train:min_per_action_valid = {self.min_per_action_train}:{self.min_per_action_valid}, but num_train:num_valid = {self.num_train}:{self.num_valid}."
            )

    @property
    def has_min(self) -> bool:
        return None not in (self.min_per_action_train, self.min_per_action_valid)

    @property
    def has_num(self) -> bool:
        return None not in (self.num_train, self.num_valid)

    @property
    def has_max(self) -> bool:
        return None not in (self.max_per_action_train, self.max_per_action_valid)

    @property
    def min_total(self) -> int:
        """Transitions per action sampled at least, both splits together."""
        return (
            self.min_per_action_train + self.min_per_action_valid if self.has_min else 0
        )

    @property
    def num_total(self) -> int:
        """Transitions sampled at least, both splits together."""
        return self.num_train + self.num_valid if self.has_num else 0

    @property
    def cap(self) -> int | None:
        """Transitions per action kept at most, both splits together."""
        if not self.has_max:
            return None
        return self.max_per_action_train + self.max_per_action_valid

    @property
    def split_weights(self) -> tuple[int, int]:
        """The train-to-validation ratio of the split."""
        if self.has_num:
            return self.num_train, self.num_valid
        return self.min_per_action_train, self.min_per_action_valid
