import logging
import random
from pathlib import Path
from typing import Optional

import lightning as L
import pymimir
import torch
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from dias.adjacency import adj_to_pyg_data
from dias.data.noise import add_adj_noise, add_adj_noise_expected_count
from dias.schema import DomainSchema

from ..schema.domain import _argument_positions
from .generate_samples import generate_random_walk, get_metadata, samples_to_tensors

logger = logging.getLogger(__name__)


def _flip_stats(state: torch.Tensor, clean: torch.Tensor):
    diff = state - clean
    pos = diff > 0
    neg = diff < 0
    return (
        int(pos.sum().item()),
        int(neg.sum().item()),
        pos.sum(dim=(1, 2)).long(),
        neg.sum(dim=(1, 2)).long(),
    )


def _build_sample(
    base: "PddlTransitionDataset",
    real_idx: int,
    noisy_state: Optional[torch.Tensor],
    noisy_next: Optional[torch.Tensor],
    idx_tensor: torch.Tensor,
):
    sample = base[real_idx]
    clean_s, clean_ns = sample["state"], sample["next_state"]
    R = clean_s.shape[0]
    zeros = torch.zeros(R, dtype=torch.long)

    if noisy_state is not None:
        state = noisy_state
        added_s, removed_s, added_s_pr, removed_s_pr = _flip_stats(state, clean_s)
    else:
        state, added_s, removed_s, added_s_pr, removed_s_pr = (
            clean_s,
            0,
            0,
            zeros,
            zeros,
        )
    if noisy_next is not None:
        next_state = noisy_next
        added_ns, removed_ns, added_ns_pr, removed_ns_pr = _flip_stats(
            next_state, clean_ns
        )
    else:
        next_state, added_ns, removed_ns, added_ns_pr, removed_ns_pr = (
            clean_ns,
            0,
            0,
            zeros,
            zeros,
        )

    action = sample["action_index"]
    action_args = sample["action_args"]
    data_s1 = adj_to_pyg_data(state, base.num_objects)
    data_s2 = adj_to_pyg_data(next_state, base.num_objects)

    data_s1.y_action = action
    data_s1.y_action_args = action_args.unsqueeze(0)
    data_s1.num_flips_added = torch.tensor([added_s + added_ns])
    data_s1.num_flips_removed = torch.tensor([removed_s + removed_ns])
    data_s1.num_flips_added_per_rel = (added_s_pr + added_ns_pr)[None, :]
    data_s1.num_flips_removed_per_rel = (removed_s_pr + removed_ns_pr)[None, :]
    data_s1.num_true_predicates = torch.tensor([int(state.sum().item())])
    data_s2.num_true_predicates = torch.tensor([int(next_state.sum().item())])

    return data_s1, data_s2, idx_tensor, state, next_state, action, action_args


class StratifiedActionIterableDataset(IterableDataset):
    """Infinite stream of action-stratified samples drawn with replacement."""

    def __init__(
        self,
        base: "PddlTransitionDataset",
        indices: list[int],
        batch_size: int,
        seed: int = 0,
        noisy_states: Optional[torch.Tensor] = None,
        noisy_next_states: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if not indices:
            raise ValueError("StratifiedActionIterableDataset: empty data pool.")
        self.base = base
        self.indices = list(indices)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.noisy_states = noisy_states
        self.noisy_next_states = noisy_next_states

        labels = base.action_indices[torch.tensor(self.indices)].tolist()
        self.per_action: dict[int, list[int]] = {}
        for pos, a in enumerate(labels):
            self.per_action.setdefault(a, []).append(pos)
        self.actions = sorted(self.per_action.keys())

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        gen = torch.Generator().manual_seed(self.seed + worker_id + 1)
        K = len(self.actions)
        base_cnt, remainder = divmod(self.batch_size, K)

        while True:
            order = [self.actions[i] for i in torch.randperm(K, generator=gen).tolist()]
            for i, a in enumerate(order):
                n = base_cnt + (1 if i < remainder else 0)
                if n == 0:
                    continue
                pool = self.per_action[a]
                picks = torch.randint(len(pool), (n,), generator=gen).tolist()
                for p in picks:
                    yield _build_sample(
                        self.base,
                        self.indices[pool[p]],
                        (
                            self.noisy_states[pool[p]]
                            if self.noisy_states is not None
                            else None
                        ),
                        (
                            self.noisy_next_states[pool[p]]
                            if self.noisy_next_states is not None
                            else None
                        ),
                        torch.tensor(self.indices[pool[p]]),
                    )


class PddlTransitionDataset(Dataset):
    """Pre-computed adjacency tensors as a map-style Dataset."""

    def __init__(self, data: dict):
        self.states: torch.Tensor = data["states"]
        self.next_states: torch.Tensor = data["next_states"]
        self.action_indices: torch.Tensor = data["action_indices"]
        self.action_args: torch.Tensor = data["action_args"]
        self.relation_names: list[str] = data["relation_names"]
        self.relation_arities: list[int] = data["relation_arities"]
        self.object_names: dict = data["object_names"]
        self.action_names: list[str] = data["action_names"]
        self.action_arities: list[int] = data["action_arities"]
        self.num_objects: int = data["num_objects"]
        self.num_relations: int = data["num_relations"]

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "state": self.states[idx],
            "next_state": self.next_states[idx],
            "action_index": self.action_indices[idx],
            "action_args": self.action_args[idx],
        }


class PddlGraphDataset(Dataset):
    """Map-style dataset returning PyG ``Data`` tuples."""

    def __init__(
        self,
        base_dataset: PddlTransitionDataset,
        indices: list[int],
        noisy_states: Optional[torch.Tensor] = None,
        noisy_next_states: Optional[torch.Tensor] = None,
    ):
        self.base = base_dataset
        self.indices = indices
        self.noisy_states = noisy_states
        self.noisy_next_states = noisy_next_states

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        return _build_sample(
            self.base,
            real_idx,
            self.noisy_states[idx] if self.noisy_states is not None else None,
            self.noisy_next_states[idx] if self.noisy_next_states is not None else None,
            torch.tensor([real_idx]),
        )


class PddlGraphDataModule(L.LightningDataModule):
    """Random-walk PDDL data pipeline with configurable stopping / batching.

    Stopping is driven by ``(min_per_label_train, min_per_label_valid)`` and/or
    ``(num_train, num_valid)``. If both are set, their train:valid ratios must
    match. Optional ``max_per_label_*`` cap each label after sampling, and
    optional ``min_per_label_*_floor`` raise (never lower) the corresponding
    minimum. ``batch_sampling_strategy`` is ``"random"`` (shuffled DataLoader)
    or ``"stratified"`` (unbounded action-balanced sampler).
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
        noise_model: str = "ratio",
        edge_remove_ratio: float = 0.0,
        edge_add_ratio: float = 0.0,
        expected_removed_atoms: float = 0.0,
        expected_added_atoms: float = 0.0,
        noise_on_fluents_only: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.domain_path = domain_path
        self.problem_path = problem_path
        self.seed = seed
        self.include_statics = include_statics
        self.default_statics = default_statics
        self.action_arguments = action_arguments
        self.strips_plus_actions = strips_plus_actions
        self.pt_file = pt_file
        self.min_per_label_train = min_per_label_train
        self.min_per_label_valid = min_per_label_valid
        self.min_per_label_train_floor = min_per_label_train_floor
        self.min_per_label_valid_floor = min_per_label_valid_floor
        self.max_per_label_train = max_per_label_train
        self.max_per_label_valid = max_per_label_valid
        self.num_train = num_train
        self.num_valid = num_valid
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.batch_sampling_strategy = batch_sampling_strategy
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.noise_model = noise_model
        self.edge_remove_ratio = edge_remove_ratio
        self.edge_add_ratio = edge_add_ratio
        self.expected_removed_atoms = expected_removed_atoms
        self.expected_added_atoms = expected_added_atoms
        self.noise_on_fluents_only = noise_on_fluents_only

        if batch_sampling_strategy not in ("random", "stratified"):
            raise ValueError(
                f"batch_sampling_strategy must be 'random' or 'stratified', "
                f"got {batch_sampling_strategy!r}."
            )
        if noise_model not in ("ratio", "expected_count"):
            raise ValueError(
                f"noise_model must be 'ratio' or 'expected_count', got {noise_model!r}."
            )
        self._apply_min_per_label_floors()
        self._validate_sampling_config()

        logger.info("Loading domain %s, problem %s", domain_path, problem_path)
        self.domain = pymimir.Domain(Path(domain_path))
        self.problem = pymimir.Problem(self.domain, Path(problem_path))
        self.metadata = get_metadata(self.domain, self.problem)
        self.schema = DomainSchema.from_metadata(
            self.metadata,
            include_statics=self.include_statics,
            default_statics=self.default_statics,
            action_arg_positions=_argument_positions(
                self.metadata["actions"],
                domain_path=domain_path,
                action_arguments=self.action_arguments,
                strips_plus_actions=self.strips_plus_actions,
            ),
        )
        self.arity_relations: list[int] = list(self.schema.relation_arities)

        self._base: Optional[PddlTransitionDataset] = None
        self._train_ds = None
        self._valid_ds = None
        self._train_idx: list[int] = []
        self._valid_idx: list[int] = []

    def _apply_min_per_label_floors(self) -> None:
        """Raise ``min_per_label_*`` to a shared lower bound, never lower it.

        Experiment overlays are composed *after* the per-domain config, so a
        plain ``min_per_label_train`` there would also shrink the domains that
        sample more. The floors only ever increase the per-domain value — that
        is what the paper's noise setup asks for: the *smallest* per-action
        minimum sample count is raised to Nmin = 200, the larger ones stay.
        """
        for name, floor in (
            ("min_per_label_train", self.min_per_label_train_floor),
            ("min_per_label_valid", self.min_per_label_valid_floor),
        ):
            current = getattr(self, name)
            if floor is None or current is None or current >= floor:
                continue
            logger.info("Raising %s from %s to its floor of %s.", name, current, floor)
            setattr(self, name, floor)

    @property
    def _has_min(self) -> bool:
        return (
            self.min_per_label_train is not None
            and self.min_per_label_valid is not None
        )

    @property
    def _has_num(self) -> bool:
        return self.num_train is not None and self.num_valid is not None

    @property
    def _has_max(self) -> bool:
        return (
            self.max_per_label_train is not None
            and self.max_per_label_valid is not None
        )

    def _validate_sampling_config(self) -> None:
        if self.pt_file is not None:
            return
        if not self._has_min and not self._has_num:
            raise ValueError(
                "PddlGraphDataModule: at least one of "
                "(min_per_label_train, min_per_label_valid) or "
                "(num_train, num_valid) must be set."
            )
        if self._has_min and (
            self.min_per_label_train <= 0 or self.min_per_label_valid <= 0
        ):
            raise ValueError("min_per_label_train and min_per_label_valid must be > 0.")
        if self._has_num and (self.num_train <= 0 or self.num_valid <= 0):
            raise ValueError("num_train and num_valid must be > 0.")
        if (self.max_per_label_train is None) != (self.max_per_label_valid is None):
            raise ValueError(
                "max_per_label_train and max_per_label_valid must be set together."
            )
        if self._has_max and (
            self.max_per_label_train <= 0 or self.max_per_label_valid <= 0
        ):
            raise ValueError("max_per_label_train and max_per_label_valid must be > 0.")
        if (
            self._has_max
            and self._has_min
            and (
                self.max_per_label_train < self.min_per_label_train
                or self.max_per_label_valid < self.min_per_label_valid
            )
        ):
            raise ValueError(
                f"max_per_label_* must be >= min_per_label_*: "
                f"max=({self.max_per_label_train},{self.max_per_label_valid}) "
                f"vs min=({self.min_per_label_train},{self.min_per_label_valid})."
            )
        if (
            self._has_min
            and self._has_num
            and (
                self.min_per_label_train * self.num_valid
                != self.num_train * self.min_per_label_valid
            )
        ):
            raise ValueError(
                f"Ratio mismatch: "
                f"min_per_label_train:min_per_label_valid="
                f"{self.min_per_label_train}:{self.min_per_label_valid} "
                f"vs num_train:num_valid={self.num_train}:{self.num_valid}."
            )

        # Each noise model owns its own knobs; the other pair must stay at 0.
        if self.noise_model == "ratio":
            if self.expected_removed_atoms != 0.0 or self.expected_added_atoms != 0.0:
                raise ValueError(
                    "expected_removed_atoms / expected_added_atoms must be 0.0 "
                    "when noise_model='ratio'."
                )
            if self.edge_remove_ratio < 0.0 or self.edge_add_ratio < 0.0:
                raise ValueError("edge_remove_ratio and edge_add_ratio must be >= 0.")
        else:
            if self.edge_remove_ratio != 0.0 or self.edge_add_ratio != 0.0:
                raise ValueError(
                    "edge_remove_ratio / edge_add_ratio must be 0.0 "
                    "when noise_model='expected_count'."
                )
            if self.expected_removed_atoms < 0.0 or self.expected_added_atoms < 0.0:
                raise ValueError(
                    "expected_removed_atoms and expected_added_atoms must be >= 0."
                )

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
        self._train_idx, self._valid_idx = self._make_split(total)
        self.relation_true_ratio = self._compute_relation_true_ratio()
        if total > 0:
            self._log_action_distribution()

        train_noisy_s, train_noisy_ns = self._build_train_noise(self._train_idx)

        if self.batch_sampling_strategy == "stratified":
            self._check_coverage()
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
        min_total = (
            self.min_per_label_train + self.min_per_label_valid if self._has_min else 0
        )
        num_total = self.num_train + self.num_valid if self._has_num else 0
        samples = generate_random_walk(
            problem=self.problem,
            min_num_samples=num_total,
            min_per_label=min_total,
            only_unseen_states=False,
            max_steps=self.max_steps,
            include_statics=self.include_statics,
        )
        if self._has_max:
            samples = self._cap_per_label(samples)
        return samples

    def _cap_per_label(self, samples: list[dict]) -> list[dict]:
        cap = self.max_per_label_train + self.max_per_label_valid
        kept_count: dict[str, int] = {}
        dropped: dict[str, int] = {}
        kept: list[dict] = []
        for s in samples:
            a = s["action"]
            if kept_count.get(a, 0) < cap:
                kept.append(s)
                kept_count[a] = kept_count.get(a, 0) + 1
            else:
                dropped[a] = dropped.get(a, 0) + 1
        if dropped:
            logger.info(
                "Capped per-label samples to %d (train=%s + valid=%s); "
                "dropped per action: %s",
                cap,
                self.max_per_label_train,
                self.max_per_label_valid,
                dropped,
            )
        return kept

    def _postprocess_action_args(self, samples: list[dict]) -> None:
        if self.action_arguments and self.strips_plus_actions:
            free = self._load_strips_plus_yaml(self.domain_path)
            for s in samples:
                allowed = free.get(s["action"], [])
                s["action_args"] = [
                    a if p in allowed else -1 for p, a in enumerate(s["action_args"])
                ]
        elif not self.action_arguments:
            for s in samples:
                s["action_args"] = [-1] * len(s["action_args"])

    def _make_split(self, total: int) -> tuple[list[int], list[int]]:
        if total == 0:
            return [], []
        train_w = self.num_train if self._has_num else self.min_per_label_train
        valid_w = self.num_valid if self._has_num else self.min_per_label_valid
        n_train_target = int(round(total * train_w / (train_w + valid_w)))
        n_train_target = max(1, min(total - 1, n_train_target))

        g = torch.Generator().manual_seed(self.seed)
        assert self._base is not None
        by_action: dict[int, list[int]] = {}
        for i, a in enumerate(self._base.action_indices.tolist()):
            by_action.setdefault(a, []).append(i)

        train_idx, valid_idx, pool, singletons = [], [], [], []
        for idxs in by_action.values():
            perm = torch.randperm(len(idxs), generator=g).tolist()
            shuffled = [idxs[p] for p in perm]
            if len(shuffled) >= 2:
                train_idx.append(shuffled[0])
                valid_idx.append(shuffled[1])
                pool.extend(shuffled[2:])
            else:
                singletons.extend(shuffled)
        train_idx.extend(singletons)

        n_extra = min(max(0, n_train_target - len(train_idx)), len(pool))
        perm = torch.randperm(len(pool), generator=g).tolist()
        train_idx.extend(pool[p] for p in perm[:n_extra])
        valid_idx.extend(pool[p] for p in perm[n_extra:])

        logger.info(
            "Split %d samples into train=%d, valid=%d at ratio %d:%d, "
            "reserving one per action per split where there are at least two.",
            total,
            len(train_idx),
            len(valid_idx),
            train_w,
            valid_w,
        )
        return train_idx, valid_idx

    def _check_coverage(self) -> None:
        assert self._base is not None
        pool_actions = set(self._base.action_indices.tolist())
        names = self._base.action_names
        for split_name, idxs in (
            ("train", self._train_idx),
            ("valid", self._valid_idx),
        ):
            if not idxs:
                continue
            missing = pool_actions - set(self._base.action_indices[idxs].tolist())
            if missing:
                raise ValueError(
                    f"Stratified '{split_name}' split is missing action(s) "
                    f"{[names[m] for m in sorted(missing)]} that appear in the full pool. "
                    f"Increase min_per_label_* / num_* or relax the train:valid ratio."
                )

    def _build_train_noise(
        self, train_idx: list[int]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self._noise_is_enabled() or not train_idx:
            return None, None
        gen = torch.Generator().manual_seed(self.seed + 1)
        if self.noise_model == "expected_count":
            return self._build_expected_count_train_noise(train_idx, gen)
        return self._build_ratio_train_noise(train_idx, gen)

    def _noise_is_enabled(self) -> bool:
        if self.noise_model == "expected_count":
            return self.expected_removed_atoms > 0.0 or self.expected_added_atoms > 0.0
        return self.edge_remove_ratio > 0.0 or self.edge_add_ratio > 0.0

    def _enabled_relation_mask(self) -> list[bool]:
        """Per-relation True/False; statics are disabled when noise_on_fluents_only."""
        static_relations = set()
        if self.noise_on_fluents_only:
            static_relations = {
                p["name"] for p in self.metadata["predicates"] if p["static"]
            }
        return [name not in static_relations for name in self.base.relation_names]

    def _build_ratio_train_noise(
        self, train_idx: list[int], gen: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        add_ratio_per_rel = [self.edge_add_ratio * r for r in self.relation_true_ratio]
        logger.info(
            "Ratio edge noise on the train split: remove=%s, add=%s, "
            "scaled add per relation=%s",
            self.edge_remove_ratio,
            self.edge_add_ratio,
            [f"{x:.4g}" for x in add_ratio_per_rel],
        )

        def noisy(src: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                [
                    add_adj_noise(
                        src[i],
                        self.arity_relations,
                        self.edge_remove_ratio,
                        add_ratio_per_rel,
                        gen,
                    )
                    for i in train_idx
                ]
            )

        return noisy(self._base.states), noisy(self._base.next_states)

    def _build_expected_count_train_noise(
        self, train_idx: list[int], gen: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation_mask = self._enabled_relation_mask()
        logger.info(
            "Expected-count edge noise on the train split: remove=%s, add=%s "
            "atoms per adjacency tensor.",
            self.expected_removed_atoms,
            self.expected_added_atoms,
        )

        def noisy(src: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                [
                    add_adj_noise_expected_count(
                        src[i],
                        self.arity_relations,
                        expected_remove_count=self.expected_removed_atoms,
                        expected_add_count=self.expected_added_atoms,
                        generator=gen,
                        relation_mask=relation_mask,
                    )
                    for i in train_idx
                ]
            )

        return noisy(self._base.states), noisy(self._base.next_states)

    def _compute_relation_true_ratio(self) -> list[float]:
        ds = self._base
        if ds is None or len(ds) == 0:
            return [0.0 for _ in self.arity_relations]
        all_adj = torch.cat([ds.states, ds.next_states], dim=0)
        ratios: list[float] = []
        for r, ar in enumerate(self.arity_relations):
            if ar == 2:
                region = all_adj[:, r].reshape(-1)
            elif ar == 1:
                region = all_adj[:, r].diagonal(dim1=-2, dim2=-1).reshape(-1)
            else:
                raise ValueError(f"Arity should be 1 or 2. Got {ar}.")
            ones = int(region.sum().item())
            zeros = region.numel() - ones
            ratios.append(ones / zeros if zeros > 0 else 0.0)

        logger.info(
            "Per-relation true/false ratios: %s",
            {
                f"{name}/{arity}": f"{ratio:.4g}"
                for name, arity, ratio in zip(
                    ds.relation_names, self.arity_relations, ratios
                )
            },
        )
        return ratios

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

    def _make_loader(self, ds, shuffle: bool):
        return PyGDataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle and not isinstance(ds, IterableDataset),
            num_workers=self.num_workers,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._make_loader(self._train_ds, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self._make_loader(self._valid_ds, shuffle=False)

    @staticmethod
    def _load_strips_plus_yaml(domain_path) -> dict[str, list[int]]:
        """Action name -> list of argument positions free to vary."""
        import yaml

        yaml_path = Path(domain_path).parent / "strips+.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"No strips+.yaml file found in {Path(domain_path).parent}. "
                f"Required when strips_plus_actions=True."
            )
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        return {name: (lst[0] if lst else []) for name, lst in yaml_data.items()}
