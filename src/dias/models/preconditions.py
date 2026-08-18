"""Whether an action's preconditions hold, differentiably."""

from __future__ import annotations

import torch


def soft_conjunction(values: torch.Tensor, dim: int, temperature: float = 1.0):
    """A relaxed product of `values` along `dim`.

    Whether preconditions hold is a product over per-atom satisfactions, which
    collapses to zero as soon as one atom fails and carries almost no gradient
    back to the others. Dividing the sum of logs by ``1 + temperature*(n-1)``
    softens that: at temperature 1 it is the geometric mean, at 0 the exact
    product. Training anneals from one to the other.
    """
    # Clamped so log stays finite; zeros are common here, and log(0) gives an
    # unstable backward that can surface much later as a CUDA assert.
    logs = torch.log(values.clamp(min=1e-8, max=1.0))
    return torch.exp(logs.sum(dim=dim) / (1 + temperature * (values.shape[dim] - 1)))
