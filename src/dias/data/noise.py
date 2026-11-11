"""Corrupting training states by flipping atoms (paper, App. C.4): `ExpectedCount` draws a number of flips per state, `Ratio` flips each atom with a probability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from dias.adjacency import valid_cells


@dataclass(frozen=True)
class RelationStats:
    """Per relation, its arity, the ratio of its true to its false atoms, and whether an action can change it."""

    arities: list[int] = field(default_factory=list)
    true_ratio: list[float] = field(default_factory=list)
    is_fluent: list[bool] = field(default_factory=list)

    @classmethod
    def measure(
        cls,
        states: torch.Tensor,
        arities: Sequence[int],
        is_fluent: Sequence[bool],
    ) -> "RelationStats":
        """The statistics of a ``[N, R, O, O]`` stack of states."""
        num_objects = states.size(-1)
        ratios = []
        for relation, arity in enumerate(arities):
            cells = valid_cells(arity, num_objects)
            if cells is None:
                raise ValueError(f"Relation arity should be 1 or 2, got {arity}.")
            region = states[:, relation].reshape(len(states), -1)[:, cells]
            ones = int(region.sum().item())
            zeros = region.numel() - ones
            ratios.append(ones / zeros if zeros > 0 else 0.0)
        return cls(list(arities), ratios, list(is_fluent))


def _partition(
    state: torch.Tensor, arity: int, num_objects: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The flat indices of a relation's true and false atoms."""
    cells = valid_cells(arity, num_objects)
    if cells is None:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    values = state.reshape(-1)[cells]
    return cells[values == 1], cells[values == 0]


@dataclass(frozen=True)
class Ratio:
    """Remove each true atom with probability `remove`, and add each false atom with probability `add` times the relation's ratio of true to false atoms."""

    remove: float = 0.0
    add: float = 0.0

    def __post_init__(self):
        if self.remove < 0.0 or self.add < 0.0:
            raise ValueError(f"Expected non-negative rates, got {self}")

    @property
    def enabled(self) -> bool:
        return self.remove > 0.0 or self.add > 0.0

    def describe(self, stats: RelationStats) -> str:
        scaled = [f"{self.add * ratio:.4g}" for ratio in stats.true_ratio]
        return f"{self}, adding per relation at {scaled}"

    def corrupt(self, states, stats, generator=None) -> torch.Tensor:
        return torch.stack(
            [self._corrupt_one(state, stats, generator) for state in states]
        )

    def _corrupt_one(self, state, stats, generator) -> torch.Tensor:
        noisy = state.clone()
        num_objects = state.size(-1)
        for relation, arity in enumerate(stats.arities):
            ones, zeros = _partition(noisy[relation], arity, num_objects)
            flat = noisy[relation].reshape(-1)
            for positions, rate, value in (
                (ones, self.remove, 0.0),
                (zeros, self.add * stats.true_ratio[relation], 1.0),
            ):
                if rate <= 0.0 or positions.numel() == 0:
                    continue
                flipped = torch.rand(positions.numel(), generator=generator) < rate
                flat[positions[flipped]] = value
        return noisy


@dataclass(frozen=True)
class ExpectedCount:
    """Remove and add Poisson-distributed numbers of atoms per state, `removed_atoms` and `added_atoms` in expectation, each relation's share proportional to ``min(num_true, num_false)``. With `fluents_only`, static relations stay clean."""

    removed_atoms: float = 0.0
    added_atoms: float = 0.0
    fluents_only: bool = False

    def __post_init__(self):
        if self.removed_atoms < 0.0 or self.added_atoms < 0.0:
            raise ValueError(f"Expected non-negative atom counts, got {self}")

    @property
    def enabled(self) -> bool:
        return self.removed_atoms > 0.0 or self.added_atoms > 0.0

    def describe(self, stats: RelationStats) -> str:
        return str(self)

    def corrupt(self, states, stats, generator=None) -> torch.Tensor:
        return torch.stack(
            [self._corrupt_one(state, stats, generator) for state in states]
        )

    def _corrupt_one(self, state, stats, generator) -> torch.Tensor:
        noisy = state.clone()
        num_objects = state.size(-1)

        partitions = []
        for relation, arity in enumerate(stats.arities):
            if self.fluents_only and not stats.is_fluent[relation]:
                empty = torch.empty(0, dtype=torch.long)
                partitions.append((empty, empty))
            else:
                partitions.append(_partition(noisy[relation], arity, num_objects))
        weights = [
            float(min(ones.numel(), zeros.numel())) for ones, zeros in partitions
        ]

        total = sum(weights)
        if total <= 0.0:
            return noisy

        for relation, ((ones, zeros), weight) in enumerate(zip(partitions, weights)):
            if weight <= 0.0:
                continue
            share = weight / total
            remove_count = _poisson(self.removed_atoms * share, generator)
            add_count = _poisson(self.added_atoms * share, generator)
            removed = _sample_without_replacement(ones, remove_count, generator)
            added = _sample_without_replacement(zeros, add_count, generator)

            flat = noisy[relation].reshape(-1)
            flat[removed] = 0.0
            flat[added] = 1.0
        return noisy


def _poisson(rate: float, generator) -> int:
    if rate <= 0.0:
        return 0
    return int(torch.poisson(torch.tensor([rate]), generator=generator).item())


def _sample_without_replacement(
    candidates: torch.Tensor, count: int, generator
) -> torch.Tensor:
    count = min(count, candidates.numel())
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    if count == candidates.numel():
        return candidates
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[:count]]


NoiseModel = Ratio | ExpectedCount
