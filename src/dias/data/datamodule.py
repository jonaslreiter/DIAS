import random
import sys
from pathlib import Path
from typing import Optional

import lightning as L
import pymimir
import torch
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from dias.utils import add_adj_noise, adj_to_pyg_data

from .generate_samples import generate_random_walk, get_metadata, samples_to_tensors


class StratifiedActionIterableDataset(IterableDataset):
    """Infinite stream of action-stratified PyG samples.

    Given a disjoint sub-pool of transitions this dataset yields samples in
    chunks of ``batch_size`` where every action label present in the
    sub-pool contributes ~``batch_size // K`` samples per chunk — plus a
    round-robin remainder when ``batch_size % K != 0``. Within each
    action's pool, samples are drawn uniformly **with replacement**.

    Pair with a ``DataLoader(batch_size=B, shuffle=False)`` so every ``B``
    consecutive items forms one stratified batch.
    """

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
        self.base = base
        self.indices = list(indices)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.noisy_states = noisy_states
        self.noisy_next_states = noisy_next_states

        if not self.indices:
            raise ValueError("StratifiedActionIterableDataset: empty data pool.")
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
        base_cnt = self.batch_size // K
        remainder = self.batch_size - base_cnt * K

        while True:
            order = [self.actions[i] for i in torch.randperm(K, generator=gen).tolist()]
            for i, a in enumerate(order):
                n = base_cnt + (1 if i < remainder else 0)
                if n == 0:
                    continue
                pool = self.per_action[a]
                picks = torch.randint(len(pool), (n,), generator=gen).tolist()
                for p in picks:
                    yield self._build_item(pool[p])

    def _build_item(self, pos: int):
        real_idx = self.indices[pos]
        sample = self.base[real_idx]
        clean_state = sample["state"]
        clean_next = sample["next_state"]
        R = clean_state.shape[0]

        if self.noisy_states is not None:
            state = self.noisy_states[pos]
            diff = state - clean_state
            added_s = int((diff > 0).sum().item())
            removed_s = int((diff < 0).sum().item())
            added_s_per_rel = (diff > 0).sum(dim=(1, 2)).long()
            removed_s_per_rel = (diff < 0).sum(dim=(1, 2)).long()
        else:
            state, added_s, removed_s = clean_state, 0, 0
            added_s_per_rel = torch.zeros(R, dtype=torch.long)
            removed_s_per_rel = torch.zeros(R, dtype=torch.long)
        if self.noisy_next_states is not None:
            next_state = self.noisy_next_states[pos]
            diff = next_state - clean_next
            added_ns = int((diff > 0).sum().item())
            removed_ns = int((diff < 0).sum().item())
            added_ns_per_rel = (diff > 0).sum(dim=(1, 2)).long()
            removed_ns_per_rel = (diff < 0).sum(dim=(1, 2)).long()
        else:
            next_state, added_ns, removed_ns = clean_next, 0, 0
            added_ns_per_rel = torch.zeros(R, dtype=torch.long)
            removed_ns_per_rel = torch.zeros(R, dtype=torch.long)

        action = sample["action_index"]
        action_args = sample["action_args"]

        data_s1 = adj_to_pyg_data(state, self.base.num_objects)
        data_s2 = adj_to_pyg_data(next_state, self.base.num_objects)

        data_s1.y_action = action
        data_s1.y_action_args = action_args.unsqueeze(0)
        data_s1.num_flips_added = torch.tensor([added_s + added_ns])
        data_s1.num_flips_removed = torch.tensor([removed_s + removed_ns])
        data_s1.num_flips_added_per_rel = (added_s_per_rel + added_ns_per_rel)[None, :]
        data_s1.num_flips_removed_per_rel = (
            removed_s_per_rel + removed_ns_per_rel
        )[None, :]

        data_s1.num_true_predicates = torch.tensor([int(state.sum().item())])
        data_s2.num_true_predicates = torch.tensor([int(next_state.sum().item())])

        return (
            data_s1,
            data_s2,
            torch.tensor(real_idx),
            state,
            next_state,
            action,
            action_args,
        )


class PddlTransitionDataset(Dataset):
    """Wraps pre-computed adjacency-matrix tensors as a PyTorch Dataset.

    Each sample is a dict::

        {
            "state":        Tensor [R, O, O],
            "next_state":   Tensor [R, O, O],
            "action_index": Tensor scalar (long),
            "action_args":  Tensor [max_arity] (long, padded with -1),
        }
    """

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
    """Map-style dataset that returns PyG ``Data`` tuples.

    Each ``__getitem__`` returns
    ``(data_s1, data_s2, idx, state, next_state, action, action_args)``.
    """

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
        sample = self.base[real_idx]

        clean_state = sample["state"]
        clean_next = sample["next_state"]
        R = clean_state.shape[0]

        if self.noisy_states is not None:
            state = self.noisy_states[idx]
            diff = state - clean_state
            added_s = int((diff > 0).sum().item())
            removed_s = int((diff < 0).sum().item())
            added_s_per_rel = (diff > 0).sum(dim=(1, 2)).long()
            removed_s_per_rel = (diff < 0).sum(dim=(1, 2)).long()
        else:
            state, added_s, removed_s = clean_state, 0, 0
            added_s_per_rel = torch.zeros(R, dtype=torch.long)
            removed_s_per_rel = torch.zeros(R, dtype=torch.long)
        if self.noisy_next_states is not None:
            next_state = self.noisy_next_states[idx]
            diff = next_state - clean_next
            added_ns = int((diff > 0).sum().item())
            removed_ns = int((diff < 0).sum().item())
            added_ns_per_rel = (diff > 0).sum(dim=(1, 2)).long()
            removed_ns_per_rel = (diff < 0).sum(dim=(1, 2)).long()
        else:
            next_state, added_ns, removed_ns = clean_next, 0, 0
            added_ns_per_rel = torch.zeros(R, dtype=torch.long)
            removed_ns_per_rel = torch.zeros(R, dtype=torch.long)

        action = sample["action_index"]
        action_args = sample["action_args"]

        num_objects = self.base.num_objects
        data_s1 = adj_to_pyg_data(state, num_objects)
        data_s2 = adj_to_pyg_data(next_state, num_objects)

        data_s1.y_action = action
        data_s1.y_action_args = action_args.unsqueeze(0)
        data_s1.num_flips_added = torch.tensor([added_s + added_ns])
        data_s1.num_flips_removed = torch.tensor([removed_s + removed_ns])
        data_s1.num_flips_added_per_rel = (added_s_per_rel + added_ns_per_rel)[None, :]
        data_s1.num_flips_removed_per_rel = (
            removed_s_per_rel + removed_ns_per_rel
        )[None, :]

        data_s1.num_true_predicates = torch.tensor([int(state.sum().item())])
        data_s2.num_true_predicates = torch.tensor([int(next_state.sum().item())])

        return (
            data_s1,
            data_s2,
            torch.tensor([real_idx]),
            state,
            next_state,
            action,
            action_args,
        )


class PddlGraphDataModule(L.LightningDataModule):
    """Random-walk PDDL data pipeline with configurable stopping / batching.

    Exploration strategy is always a random walk. The stopping criterion is
    driven by four parameters:

    - ``min_per_label_train`` / ``min_per_label_valid``: require at least
      ``min_per_label_train + min_per_label_valid`` unique-state transitions
      per action label.
    - ``num_train`` / ``num_valid``: require at least
      ``num_train + num_valid`` total transitions.

    If both pairs are set, both criteria must hold and the train:valid
    ratios must match. If only one pair is set, only that criterion
    applies. If none are set, ``ValueError`` is raised. The ratio from
    the available pair is used to split the generated pool into disjoint
    train / valid index sets.

    Optional ``max_per_label_train`` / ``max_per_label_valid`` cap the
    pool *after* the random walk finishes: each action label is
    truncated to at most ``max_per_label_train + max_per_label_valid``
    samples (kept in walk order). The cap does not change the stopping
    criterion — it only trims the returned pool.

    Two batch sampling strategies are available via
    ``batch_sampling_strategy``:

    - ``"random"``: a plain map-style dataset with a shuffled DataLoader —
      one epoch iterates every train sample once in batches of
      ``batch_size``.
    - ``"stratified"``: an unbounded :class:`StratifiedActionIterableDataset`
      that yields action-balanced batches drawn with replacement. Bound
      length via the Lightning trainer (``max_steps`` /
      ``limit_train_batches`` / ``limit_val_batches``).
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
        max_per_label_train: Optional[int] = None,
        max_per_label_valid: Optional[int] = None,
        num_train: Optional[int] = None,
        num_valid: Optional[int] = None,
        max_steps: int = 1_000_000,
        batch_size: int = 32,
        batch_sampling_strategy: str = "random",
        shuffle_train: bool = True,
        num_workers: int = 0,
        edge_remove_ratio: float = 0.0,
        edge_add_ratio: float = 0.0,
        metadata_only: bool = False,
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
        self.max_per_label_train = max_per_label_train
        self.max_per_label_valid = max_per_label_valid
        self.num_train = num_train
        self.num_valid = num_valid
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.batch_sampling_strategy = batch_sampling_strategy
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.edge_remove_ratio = edge_remove_ratio
        self.edge_add_ratio = edge_add_ratio
        self.metadata_only = metadata_only

        if self.batch_sampling_strategy not in ("random", "stratified"):
            raise ValueError(
                f"batch_sampling_strategy must be 'random' or 'stratified', "
                f"got {self.batch_sampling_strategy!r}."
            )

        self._validate_sampling_config()

        print(f"Loading domain:  {self.domain_path}", file=sys.stderr)
        print(f"Loading problem: {self.problem_path}", file=sys.stderr)

        self.domain = pymimir.Domain(Path(self.domain_path))
        self.problem = pymimir.Problem(self.domain, Path(self.problem_path))
        self.metadata = get_metadata(self.domain, self.problem)

        self.arity_relations: list[int] = []
        for predicate in self.metadata["predicates"]:
            if (
                predicate["arity"] > 2
                or (not self.include_statics and predicate["static"])
                or (
                    self.include_statics
                    and predicate["static"]
                    and predicate["name"] in self.default_statics
                )
            ):
                continue
            self.arity_relations.append(predicate["arity"])

        self._base: Optional[PddlTransitionDataset] = None
        self._train_ds = None
        self._valid_ds = None
        self._train_idx: list[int] = []
        self._valid_idx: list[int] = []

    # ------------------------------------------------------------------
    #  config validation
    # ------------------------------------------------------------------

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
        if self.pt_file is not None or self.metadata_only:
            return
        if not self._has_min and not self._has_num:
            raise ValueError(
                "PddlGraphDataModule: at least one of "
                "(min_per_label_train, min_per_label_valid) or "
                "(num_train, num_valid) must be set. "
                "Pass metadata_only=True to build an empty dataset."
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
        if self._has_max and self._has_min and (
            self.max_per_label_train < self.min_per_label_train
            or self.max_per_label_valid < self.min_per_label_valid
        ):
            raise ValueError(
                f"max_per_label_* must be >= min_per_label_*: "
                f"max=({self.max_per_label_train},{self.max_per_label_valid}) "
                f"vs min=({self.min_per_label_train},{self.min_per_label_valid})."
            )
        if self._has_min and self._has_num:
            # Require equal train:valid ratio across the two pairs.
            if (
                self.min_per_label_train * self.num_valid
                != self.num_train * self.min_per_label_valid
            ):
                raise ValueError(
                    f"Ratio mismatch: "
                    f"min_per_label_train:min_per_label_valid = "
                    f"{self.min_per_label_train}:{self.min_per_label_valid} "
                    f"vs num_train:num_valid = "
                    f"{self.num_train}:{self.num_valid}."
                )

    # ------------------------------------------------------------------
    #  properties consumed by CLI link_arguments / model
    # ------------------------------------------------------------------

    @property
    def base(self) -> PddlTransitionDataset:
        assert self._base is not None, "Call .setup() first."
        return self._base

    @property
    def num_relations(self) -> int:
        return len(self.arity_relations)

    @property
    def num_objects(self) -> int:
        return len(self.metadata["objects"])

    @property
    def num_actions(self) -> int:
        return len(self.metadata["actions"])

    @property
    def max_action_arity(self) -> int:
        return max(action["arity"] for action in self.metadata["actions"])

    @property
    def action_arities(self) -> list[int]:
        return [action["arity"] for action in self.metadata["actions"]]

    @property
    def action_arg_positions(self) -> list[list[int]]:
        if not self.action_arguments:
            return [[] for _ in self.metadata["actions"]]
        if self.strips_plus_actions:
            free_pos_map = self._load_strips_plus_yaml(self.domain_path)
            return [
                sorted(free_pos_map.get(action["name"], []))
                for action in self.metadata["actions"]
            ]
        return [list(range(action["arity"])) for action in self.metadata["actions"]]

    @property
    def relation_names(self) -> list[str]:
        return self.base.relation_names

    @property
    def action_names(self) -> list[str]:
        return self.base.action_names

    @property
    def object_names(self) -> dict:
        return self.base.object_names

    @property
    def dataset_size(self) -> int:
        return len(self.base)

    # ------------------------------------------------------------------
    #  setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        if self.pt_file is not None:
            data = torch.load(self.pt_file, weights_only=False)
        elif self.metadata_only:
            data = self._empty_data()
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
                self._train_idx,
                seed_offset=2,
                noisy_states=train_noisy_s,
                noisy_next_states=train_noisy_ns,
            )
            self._valid_ds = self._make_stratified(self._valid_idx, seed_offset=3)
        else:
            self._train_ds = PddlGraphDataset(
                self._base,
                self._train_idx,
                noisy_states=train_noisy_s,
                noisy_next_states=train_noisy_ns,
            )
            self._valid_ds = PddlGraphDataset(self._base, self._valid_idx)

        print(
            f"PddlGraphDataModule: {total} total samples → "
            f"train={len(self._train_idx)}, valid={len(self._valid_idx)} "
            f"(batch_sampling_strategy={self.batch_sampling_strategy})",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    #  sampling
    # ------------------------------------------------------------------

    def _sample_transitions(self) -> list[dict]:
        random.seed(self.seed)

        print(
            f"Domain: {self.metadata['domain_name']},  "
            f"Objects: {self.metadata['num_objects']},  "
            f"Actions: {[a['name'] for a in self.metadata['actions']]}",
            file=sys.stderr,
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
        """Truncate each action label to ``max_per_label_train + max_per_label_valid``."""
        max_total = self.max_per_label_train + self.max_per_label_valid
        kept_count: dict[str, int] = {}
        dropped: dict[str, int] = {}
        kept: list[dict] = []
        for s in samples:
            a = s["action"]
            if kept_count.get(a, 0) < max_total:
                kept.append(s)
                kept_count[a] = kept_count.get(a, 0) + 1
            else:
                dropped[a] = dropped.get(a, 0) + 1
        if dropped:
            print(
                f"[info] Capped per-label samples to {max_total} "
                f"(train={self.max_per_label_train} + valid={self.max_per_label_valid}); "
                f"dropped per action: {dropped}",
                file=sys.stderr,
            )
        return kept

    def _postprocess_action_args(self, samples: list[dict]) -> None:
        """Mask action args according to action_arguments / strips_plus_actions."""
        if self.action_arguments and self.strips_plus_actions:
            free_pos_map = self._load_strips_plus_yaml(self.domain_path)
            for sample in samples:
                for p in range(len(sample["action_args"])):
                    if p not in free_pos_map.get(sample["action"], []):
                        sample["action_args"][p] = -1
        elif not self.action_arguments:
            for sample in samples:
                sample["action_args"] = [-1 for _ in range(len(sample["action_args"]))]

    # ------------------------------------------------------------------
    #  split
    # ------------------------------------------------------------------

    def _make_split(self, total: int) -> tuple[list[int], list[int]]:
        if total == 0:
            return [], []

        if self._has_num:
            train_w = self.num_train
            valid_w = self.num_valid
        else:
            train_w = self.min_per_label_train
            valid_w = self.min_per_label_valid

        denom = train_w + valid_w
        n_train_target = int(round(total * train_w / denom))
        n_train_target = max(1, min(total - 1, n_train_target))

        g = torch.Generator().manual_seed(self.seed)

        assert self._base is not None
        action_indices = self._base.action_indices.tolist()
        by_action: dict[int, list[int]] = {}
        for i, a in enumerate(action_indices):
            by_action.setdefault(a, []).append(i)

        train_idx: list[int] = []
        valid_idx: list[int] = []
        pool: list[int] = []
        singletons: list[int] = []
        for idxs in by_action.values():
            perm = torch.randperm(len(idxs), generator=g).tolist()
            shuffled = [idxs[p] for p in perm]
            if len(shuffled) >= 2:
                train_idx.append(shuffled[0])
                valid_idx.append(shuffled[1])
                pool.extend(shuffled[2:])
            else:
                singletons.extend(shuffled)

        # Singletons can't cover both splits; send them to train (the larger one).
        train_idx.extend(singletons)

        n_train_extra = max(0, n_train_target - len(train_idx))
        n_train_extra = min(n_train_extra, len(pool))
        perm = torch.randperm(len(pool), generator=g).tolist()
        train_idx.extend(pool[p] for p in perm[:n_train_extra])
        valid_idx.extend(pool[p] for p in perm[n_train_extra:])

        print(
            f"Split: {total} samples → train={len(train_idx)}, "
            f"valid={len(valid_idx)} (ratio {train_w}:{valid_w}, "
            f"stratified: 1 reserved per action per split when ≥2 samples).",
            file=sys.stderr,
        )
        return train_idx, valid_idx

    def _check_coverage(self) -> None:
        """Every non-empty split must cover all actions present in the pool."""
        assert self._base is not None
        pool_action_set = set(self._base.action_indices.tolist())
        action_names = self._base.action_names
        for name, idx_list in (
            ("train", self._train_idx),
            ("valid", self._valid_idx),
        ):
            if not idx_list:
                continue
            sub_actions = set(self._base.action_indices[idx_list].tolist())
            missing = pool_action_set - sub_actions
            if missing:
                missing_names = [action_names[m] for m in sorted(missing)]
                raise ValueError(
                    f"Stratified '{name}' split is missing action(s) "
                    f"{missing_names} that appear in the full pool. "
                    f"Increase min_per_label_* / num_* or relax the "
                    f"train:valid ratio."
                )

    # ------------------------------------------------------------------
    #  noise / diagnostics
    # ------------------------------------------------------------------

    def _build_train_noise(
        self, train_idx: list[int]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.edge_remove_ratio == 0.0 and self.edge_add_ratio == 0.0:
            return None, None
        if not train_idx:
            return None, None

        gen = torch.Generator().manual_seed(self.seed + 1)

        arities = self.arity_relations
        add_ratio_per_rel = [self.edge_add_ratio * r for r in self.relation_true_ratio]
        print(
            f"Applying edge noise to train split "
            f"(remove={self.edge_remove_ratio}, add={self.edge_add_ratio}, "
            f"scaled add per rel={[f'{x:.4g}' for x in add_ratio_per_rel]})",
            file=sys.stderr,
        )
        base = self._base
        noise_state = torch.stack(
            [
                add_adj_noise(
                    base.states[i],
                    arities,
                    self.edge_remove_ratio,
                    add_ratio_per_rel,
                    gen,
                )
                for i in train_idx
            ]
        )
        next_noise_state = torch.stack(
            [
                add_adj_noise(
                    base.next_states[i],
                    arities,
                    self.edge_remove_ratio,
                    add_ratio_per_rel,
                    gen,
                )
                for i in train_idx
            ]
        )
        return noise_state, next_noise_state

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
                raise ValueError(f"Arity should be one or two. Got {ar}.")

            ones = int(region.sum().item())
            zeros = region.numel() - ones
            ratios.append(ones / zeros if zeros > 0 else 0.0)

        print("Per-relation true/false ratios:", file=sys.stderr)
        for name, ar, r in zip(ds.relation_names, self.arity_relations, ratios):
            print(f"  {name} (arity {ar}): r={r:.4g}", file=sys.stderr)
        return ratios

    def _log_action_distribution(self) -> None:
        ds = self._base
        counts = torch.bincount(ds.action_indices, minlength=len(ds.action_names))
        print("Action frequency distribution:", file=sys.stderr)
        for name, count in zip(ds.action_names, counts.tolist()):
            pct = 100.0 * count / len(ds)
            print(f"  {name}: {count} ({pct:.1f}%)", file=sys.stderr)

    def _empty_data(self) -> dict:
        n_obj = len(self.metadata["objects"])
        n_rel = len(self.arity_relations)
        max_arity = max((a["arity"] for a in self.metadata["actions"]), default=0)
        relation_names: list[str] = []
        for predicate in self.metadata["predicates"]:
            if (
                predicate["arity"] > 2
                or (not self.include_statics and predicate["static"])
                or (
                    self.include_statics
                    and predicate["static"]
                    and predicate["name"] in self.default_statics
                )
            ):
                continue
            relation_names.append(predicate["name"])
        return {
            "states": torch.zeros(0, n_rel, n_obj, n_obj),
            "next_states": torch.zeros(0, n_rel, n_obj, n_obj),
            "action_indices": torch.zeros(0, dtype=torch.long),
            "action_args": torch.zeros(0, max_arity, dtype=torch.long),
            "relation_names": relation_names,
            "relation_arities": list(self.arity_relations),
            "object_names": self.metadata["objects"],
            "object_name_to_idx": self.metadata["object_name_to_idx"],
            "action_names": [a["name"] for a in self.metadata["actions"]],
            "action_arities": [a["arity"] for a in self.metadata["actions"]],
            "applicable_actions": [],
            "num_objects": n_obj,
            "num_relations": n_rel,
        }

    # ------------------------------------------------------------------
    #  loaders
    # ------------------------------------------------------------------

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
        is_iterable = isinstance(ds, IterableDataset)
        return PyGDataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle and not is_iterable,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._make_loader(self._train_ds, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self._make_loader(self._valid_ds, shuffle=False)

    # ------------------------------------------------------------------
    #  strips+ yaml
    # ------------------------------------------------------------------

    @staticmethod
    def _load_strips_plus_yaml(domain_path) -> dict[str, list[int]]:
        """Action name → list of argument positions free to vary."""
        import yaml

        domain_dir = Path(domain_path).parent
        yaml_files = list(domain_dir.glob("*strips+.yaml"))
        if not yaml_files:
            raise FileNotFoundError(
                f"No *strips+.yaml file found in {domain_dir}. "
                f"Required when strips_plus_actions=True."
            )
        yaml_path = yaml_files[0]
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f) or {}

        result: dict[str, list[int]] = {}
        for action_name, pos_lists in yaml_data.items():
            if pos_lists and len(pos_lists) > 0:
                result[action_name] = pos_lists[0]
            else:
                result[action_name] = []
        return result
