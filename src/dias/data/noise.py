"""Corrupting observed states, for the noise ablations.

Both models flip cells of the adjacency tensor and differ in how the budget is chosen: `Ratio` per cell, `ExpectedCount` per tensor. Applied to the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from dias.adjacency import valid_cells


@dataclass(frozen=True)
class RelationStats:
    """Per relation: `true_ratio`, how many cells hold against how many do not within the region its arity allows, and `is_fluent`, false when no action can change it."""

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
        """Count how often each relation holds across a ``[N, R, O, O]`` stack."""
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
    """The flat indices of a relation's true and false cells, where it has any."""
    cells = valid_cells(arity, num_objects)
    if cells is None:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    values = state.reshape(-1)[cells]
    return cells[values == 1], cells[values == 0]


@dataclass(frozen=True)
class Ratio:
    """Flip every eligible cell independently. `add` is scaled by how often the relation holds, so a sparse one is not swamped by spurious atoms."""

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
    """Spend a Poisson budget of atom flips across the relations.

    Each relation's share is proportional to ``min(num_true, num_false)``, so the ones closest to an even split carry most of the noise. Being per tensor rather than per cell, the corruption does not grow with the instance.

    Static relations share the budget, as they did in the paper's runs. `fluents_only` leaves them clean and spends the whole budget on the fluents instead, which at the same counts is between 2 and 25 times more noise on what the effects have to explain, depending on the domain.
    """

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
            # Both counts are drawn before either is spent, so changing one budget does not shift the other's draws.
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
