"""The three datasets one problem instance is read through.

`PddlTransitionDataset` holds the sampled transitions as dense tensors; `PddlGraphDataset` and `StratifiedActionIterableDataset` turn one of them into the pair of PyG graphs a training step consumes, the second by drawing an endless action-balanced stream rather than walking an index list.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from dias.adjacency import adj_to_pyg_data


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


def _build_sample(
    base: PddlTransitionDataset,
    real_idx: int,
    noisy_state: Optional[torch.Tensor],
    noisy_next: Optional[torch.Tensor],
    idx_tensor: torch.Tensor,
):
    sample = base[real_idx]
    transition = (
        sample["state"] if noisy_state is None else noisy_state,
        sample["next_state"] if noisy_next is None else noisy_next,
    )

    data_s1, data_s2 = [
        adj_to_pyg_data(state, base.num_objects) for state in transition
    ]
    data_s1.y_action = sample["action_index"]
    data_s1.y_action_args = sample["action_args"].unsqueeze(0)

    return (
        data_s1,
        data_s2,
        idx_tensor,
        *transition,
        sample["action_index"],
        sample["action_args"],
    )


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


class StratifiedActionIterableDataset(IterableDataset):
    """Infinite stream of action-stratified samples drawn with replacement."""

    def __init__(
        self,
        base: PddlTransitionDataset,
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
