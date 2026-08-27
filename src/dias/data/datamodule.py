import logging
import random
from pathlib import Path
from typing import Optional

import lightning as L
import pymimir
import torch
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from dias.data.datasets import (
    PddlGraphDataset,
    PddlTransitionDataset,
    StratifiedActionIterableDataset,
)
from dias.data.noise import ExpectedCount, NoiseModel, Ratio, RelationStats
from dias.data.quota import SampleQuota
from dias.data.splitting import (
    cap_per_label,
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
    """One problem instance, as the transitions a training run reads.

    A random walk draws transitions until `SampleQuota` is met, they are split by action, and the train half is optionally corrupted by a `NoiseModel`. `batch_sampling_strategy` is ``"random"`` (a shuffled pass over the split) or ``"stratified"`` (an unbounded action-balanced stream, in which case training length comes from ``trainer.max_steps``).

    The sampling arguments stay flat here rather than nesting into the quota, because they are what configs, sweeps and archived hyper-parameters name.
    """

    def __init__(
        self,
        domain_path: Optional[str] = None,
        problem_path: Optional[str] = None,
        seed: int = 42,
        include_statics: bool = False,
        default_statics: list[str] = ["number", "object"],
        action_arguments: bool = False,
        strips_plus_actions: bool = False,
        pt_file: Optional[str] = None,
        min_per_label_train: Optional[int] = None,
        min_per_label_valid: Optional[int] = None,
        min_per_label_train_floor: Optional[int] = None,
        min_per_label_valid_floor: Optional[int] = None,
        max_per_label_train: Optional[int] = None,
        max_per_label_valid: Optional[int] = None,
        num_train: Optional[int] = None,
        num_valid: Optional[int] = None,
        max_steps: int = 1_000_000,
        batch_size: int = 32,
        batch_sampling_strategy: str = "random",
        shuffle_train: bool = True,
        num_workers: int = 0,
        noise: NoiseModel | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.domain_path = domain_path
        self.problem_path = problem_path
        self.seed = seed
        self.pt_file = pt_file
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.batch_sampling_strategy = batch_sampling_strategy
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.action_arguments = action_arguments
        self.strips_plus_actions = strips_plus_actions
        self.include_statics = include_statics
        self.default_statics = default_statics

        if batch_sampling_strategy not in ("random", "stratified"):
            raise ValueError(
                f"batch_sampling_strategy must be 'random' or 'stratified', got {batch_sampling_strategy!r}."
            )
        # The floors mutate the minima, so the quota is derived rather than read back off `hparams`, which keep the per-domain values as written.
        self.quota = SampleQuota.with_floors(
            min_per_label_train=min_per_label_train,
            min_per_label_valid=min_per_label_valid,
            min_per_label_train_floor=min_per_label_train_floor,
            min_per_label_valid_floor=min_per_label_valid_floor,
            max_per_label_train=max_per_label_train,
            max_per_label_valid=max_per_label_valid,
            num_train=num_train,
            num_valid=num_valid,
        )
        self.noise = (
            None if noise is None else from_mapping(noise, (Ratio, ExpectedCount))
        )
        # Hyper-parameters must stay primitive; see dias.hparams.
        self.hparams["noise"] = None if self.noise is None else to_mapping(self.noise)

        logger.info("Loading domain %s, problem %s", domain_path, problem_path)
        self.domain = pymimir.Domain(Path(domain_path))
        self.problem = pymimir.Problem(self.domain, Path(problem_path))
        self.metadata = read_metadata(self.domain, self.problem)
        self.schema = DomainSchema.from_metadata(
            self.metadata,
            include_statics=include_statics,
            default_statics=default_statics,
            action_arg_positions=argument_positions(
                self.metadata["actions"],
                domain_path=domain_path,
                action_arguments=action_arguments,
                strips_plus_actions=strips_plus_actions,
            ),
        )
        self.arity_relations: list[int] = list(self.schema.relation_arities)

        self._base: Optional[PddlTransitionDataset] = None
        self._train_ds = None
        self._valid_ds = None
        self._train_idx: list[int] = []
        self._valid_idx: list[int] = []

    @property
    def base(self) -> PddlTransitionDataset:
        assert self._base is not None, "Call .setup() first."
        return self._base

    # Metadata comes from the schema, so it is available before setup().
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
        if self.pt_file is not None:
            data = torch.load(self.pt_file, weights_only=False)
        else:
            samples = self._sample_transitions()
            self._postprocess_action_args(samples)
            data = samples_to_tensors(
                samples,
                self.metadata,
                include_statics=self.include_statics,
                default_statics=self.default_statics,
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

        if self.batch_sampling_strategy == "stratified":
            check_every_action_survives(
                self._base.action_indices,
                self._base.action_names,
                {"train": self._train_idx, "valid": self._valid_idx},
            )
            self._train_ds = self._make_stratified(
                self._train_idx, 2, train_noisy_s, train_noisy_ns
            )
            self._valid_ds = self._make_stratified(self._valid_idx, 3)
        else:
            self._train_ds = PddlGraphDataset(
                self._base, self._train_idx, train_noisy_s, train_noisy_ns
            )
            self._valid_ds = PddlGraphDataset(self._base, self._valid_idx)

        logger.info(
            "%d samples: train=%d, valid=%d, batched %s.",
            total,
            len(self._train_idx),
            len(self._valid_idx),
            self.batch_sampling_strategy,
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
            "Sampling transitions: at least %d per action%s, walking at most %d steps. This is the slow part on a large instance.",
            self.quota.min_total,
            f" and {self.quota.num_total} in total" if self.quota.num_total else "",
            self.max_steps,
        )
        samples = generate_random_walk(
            problem=self.problem,
            min_num_samples=self.quota.num_total,
            min_per_label=self.quota.min_total,
            only_unseen_states=False,
            max_steps=self.max_steps,
            include_statics=self.include_statics,
        )
        if self.quota.cap is None:
            return samples
        return cap_per_label(samples, self.quota.cap)

    def _postprocess_action_args(self, samples: list[dict]) -> None:
        if self.action_arguments and self.strips_plus_actions:
            free = strips_plus_arguments(self.domain_path)
            for s in samples:
                allowed = free.get(s["action"], [])
                s["action_args"] = [
                    a if p in allowed else -1 for p, a in enumerate(s["action_args"])
                ]
        elif not self.action_arguments:
            for s in samples:
                s["action_args"] = [-1] * len(s["action_args"])

    def _build_train_noise(
        self, train_idx: list[int]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Noisy copies of the train split's states, or None for a clean run."""
        if self.noise is None or not self.noise.enabled or not train_idx:
            return None, None
        logger.info("Train split gets %s", self.noise.describe(self.relation_stats))

        generator = torch.Generator().manual_seed(self.seed + 1)
        return tuple(
            self.noise.corrupt(source[train_idx], self.relation_stats, generator)
            for source in (self._base.states, self._base.next_states)
        )

    def _measure_relations(self) -> RelationStats:
        """How often each relation holds, and which of them an action can change."""
        base = self._base
        static_names = {p["name"] for p in self.metadata["predicates"] if p["static"]}
        is_fluent = [name not in static_names for name in base.relation_names]
        if len(base) == 0:
            return RelationStats(
                self.arity_relations, [0.0] * len(base.relation_names), is_fluent
            )

        stats = RelationStats.measure(
            torch.cat([base.states, base.next_states], dim=0),
            self.arity_relations,
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
            return PddlGraphDataset(self._base, [])
        return StratifiedActionIterableDataset(
            base=self._base,
            indices=indices,
            batch_size=self.batch_size,
            seed=self.seed + seed_offset,
            noisy_states=noisy_states,
            noisy_next_states=noisy_next_states,
        )

    def _make_loader(self, dataset, shuffle: bool):
        return PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            # A stream decides its own order; the loader may not shuffle it.
            shuffle=shuffle and not isinstance(dataset, IterableDataset),
            num_workers=self.num_workers,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._make_loader(self._train_ds, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self._make_loader(self._valid_ds, shuffle=False)
