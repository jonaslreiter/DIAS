import torch
from torch_geometric.data import Data


class LossWeightSchedule:
    """Piecewise linear or exponential interpolation between epoch-weight keyframes."""

    def __init__(self, schedule_data: dict):
        self.type = schedule_data["type"]

        if self.type == "linear":
            self.keypoints = schedule_data["keypoints"]
            sorted_epochs = sorted(self.keypoints.keys())
            self._epochs = sorted_epochs
            self._weights = [self.keypoints[e] for e in sorted_epochs]
            self._current_weight = self._weights[0]
        elif self.type == "exponential":
            self.v0 = schedule_data["v0"]
            self.vT = schedule_data["vT"]
            self.T = schedule_data["T"]
            self._current_weight = self.v0
        else:
            raise ValueError(
                f'Wrong type {self.type}, expected one of ["linear", "exponential"]'
            )

    def __call__(self, step: int) -> float:
        if self.type == "linear":
            if step <= self._epochs[0]:
                self._current_weight = self._weights[0]
                return self._current_weight

            if step >= self._epochs[-1]:
                self._current_weight = self._weights[-1]
                return self._current_weight

            for i in range(len(self._epochs) - 1):
                if self._epochs[i] <= step < self._epochs[i + 1]:
                    t = (step - self._epochs[i]) / (
                        self._epochs[i + 1] - self._epochs[i]
                    )
                    self._current_weight = self._weights[i] + t * (
                        self._weights[i + 1] - self._weights[i]
                    )
                    return self._current_weight
        else:
            self._current_weight = self.v0 * (self.vT / self.v0) ** (step / self.T)
            return self._current_weight

        return self._current_weight

    def __repr__(self):
        kf = dict(zip(self._epochs, self._weights))
        return f"LossWeightSchedule({kf})"


class LossWeightScheduleCollection:
    def __init__(self, schedules: dict[str, dict[int, float] | float]):
        self.schedules = {k: LossWeightSchedule(v) for k, v in schedules.items()}

    def __getitem__(self, name: str) -> float:
        return self.schedules[name]._current_weight

    def __iter__(self):
        return iter(self.schedules)

    def update_loss_weights(self, epoch: int) -> dict[str, float]:
        """Recompute loss weights for the epoch and return the new weight map."""
        return {name: schedule(epoch) for name, schedule in self.schedules.items()}


def resolve_label_names(
    datamodule, num_actions: int, num_relations: int, num_objects: int
):
    """Resolve action / relation / object display names from a datamodule with numeric fallbacks."""

    def _normalise(names, target_len, prefix):
        lst = [] if names is None else list(names)
        if len(lst) < target_len:
            lst.extend(f"{prefix} {i}" for i in range(len(lst), target_len))
        return lst[:target_len]

    action_names = _normalise(
        getattr(datamodule, "action_names", None), num_actions, "Action"
    )
    relation_names = _normalise(
        getattr(datamodule, "relation_names", None), num_relations, "Rel"
    )

    obj_dict = getattr(datamodule, "object_names", None)
    if obj_dict is not None:
        object_names = [obj_dict.get(str(i), f"Node {i}") for i in range(num_objects)]
    else:
        object_names = [f"Node {n}" for n in range(num_objects)]

    return action_names, relation_names, object_names


def gmean(input_x, dim, temperature: float = 1.0):
    # Clamp so log is finite; zeros in precond products are common and log(0)
    # otherwise yields -inf / unstable backward (can surface as CUDA asserts later).
    x = input_x.clamp(min=1e-8, max=1.0)
    log_x = torch.log(x)
    return torch.exp(
        torch.sum(log_x, dim=dim) / (1 + temperature * (input_x.shape[dim] - 1))
    )


def adj_to_pyg_data(adj: torch.Tensor, num_objects: int) -> Data:
    """Convert a multi-relational adjacency tensor ``[R, O, O]`` to a PyG ``Data``.

    Each non-zero entry ``adj[r, i, j]`` becomes an edge ``(i → j)`` with
    ``edge_attr = r``.
    """
    rel, src, dst = adj.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    return Data(
        edge_index=edge_index,
        edge_attr=rel,
        num_nodes=num_objects,
    )


def add_adj_noise(
    adj: torch.Tensor,
    arities: list,
    remove_ratio: float,
    add_ratio: float | list[float] | torch.Tensor,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Return a noisy copy of a multi-relational adjacency tensor ``[R, O, O]``.

    For each relation slice the valid region depends on arity:
      - arity 2: the full ``[O, O]`` matrix.
      - arity 1: only diagonal entries ``(i, i)``.
      - other arities: skipped.

    Within the valid region, each existing 1-entry is independently flipped to
    0 with probability ``remove_ratio`` and each existing 0-entry is flipped to
    1 with probability ``add_ratio``. ``add_ratio`` may be a scalar or a
    per-relation sequence of length ``R``.
    """
    R, O, O2 = adj.shape
    assert O == O2, f"adj must be [R, O, O], got {adj.shape}"
    assert len(arities) == R, f"arities length {len(arities)} != R={R}"

    if isinstance(add_ratio, (int, float)):
        add_ratio_per_rel = [float(add_ratio)] * R
    else:
        add_ratio_per_rel = [float(x) for x in add_ratio]
        assert (
            len(add_ratio_per_rel) == R
        ), f"add_ratio length {len(add_ratio_per_rel)} != R={R}"

    noisy = adj.clone()
    for r, ar in enumerate(arities):
        if ar == 2:
            valid_idx = torch.arange(O * O)
        elif ar == 1:
            valid_idx = torch.arange(O) * (O + 1)
        else:
            continue

        flat = noisy[r].reshape(-1)
        vals = flat[valid_idx]
        ones_pos = valid_idx[vals == 1]
        zeros_pos = valid_idx[vals == 0]

        if remove_ratio > 0 and ones_pos.numel() > 0:
            mask = torch.rand(ones_pos.numel(), generator=generator) < remove_ratio
            flat[ones_pos[mask]] = 0
        if add_ratio_per_rel[r] > 0 and zeros_pos.numel() > 0:
            mask = (
                torch.rand(zeros_pos.numel(), generator=generator)
                < add_ratio_per_rel[r]
            )
            flat[zeros_pos[mask]] = 1

    return noisy


def add_adj_noise_expected_count(
    adj: torch.Tensor,
    arities: list[int],
    expected_remove_count: float,
    expected_add_count: float,
    generator: torch.Generator = None,
    relation_mask: list[bool] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a noisy copy using per-tensor Poisson flip budgets.

    For each relation, compute the eligible true and false atoms in the valid
    region. The balancing weight is ``min(num_true, num_false)``, so relations
    closest to a 50/50 true-false split receive the largest share of the noise
    budget. Removal and addition counts are sampled independently per relation
    from Poisson distributions whose rates are proportional to those weights,
    then clipped to the available eligible atoms and applied without replacement.
    ``relation_mask`` (length ``R``) disables noise on specific relations
    (e.g. statics) when an entry is ``False``.
    """
    R, O, O2 = adj.shape
    assert O == O2, f"adj must be [R, O, O], got {adj.shape}"
    assert len(arities) == R, f"arities length {len(arities)} != R={R}"

    if relation_mask is None:
        enabled_relations = [True] * R
    else:
        enabled_relations = [bool(x) for x in relation_mask]
        assert (
            len(enabled_relations) == R
        ), f"relation_mask length {len(enabled_relations)} != R={R}"

    noisy = adj.clone()
    ones_positions_per_rel: list[torch.Tensor] = []
    zeros_positions_per_rel: list[torch.Tensor] = []
    weights: list[float] = []

    for r, ar in enumerate(arities):
        if not enabled_relations[r] or ar not in (1, 2):
            ones_positions_per_rel.append(torch.empty(0, dtype=torch.long))
            zeros_positions_per_rel.append(torch.empty(0, dtype=torch.long))
            weights.append(0.0)
            continue

        valid_idx = torch.arange(O * O) if ar == 2 else torch.arange(O) * (O + 1)
        flat = noisy[r].reshape(-1)
        vals = flat[valid_idx]
        ones_pos = valid_idx[vals == 1]
        zeros_pos = valid_idx[vals == 0]
        ones_positions_per_rel.append(ones_pos)
        zeros_positions_per_rel.append(zeros_pos)
        weights.append(float(min(ones_pos.numel(), zeros_pos.numel())))

    total_weight = sum(weights)
    if total_weight <= 0.0:
        return noisy

    def _sample_poisson_count(rate: float) -> int:
        if rate <= 0.0:
            return 0
        return int(torch.poisson(torch.tensor([rate]), generator=generator).item())

    def _pick_positions(candidates: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0 or candidates.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        if count >= candidates.numel():
            return candidates
        perm = torch.randperm(candidates.numel(), generator=generator)
        return candidates[perm[:count]]

    for r in range(R):
        weight = weights[r]
        if weight <= 0.0:
            continue
        flat = noisy[r].reshape(-1)
        ones_pos = ones_positions_per_rel[r]
        zeros_pos = zeros_positions_per_rel[r]

        remove_count = min(
            _sample_poisson_count(expected_remove_count * weight / total_weight),
            ones_pos.numel(),
        )
        add_count = min(
            _sample_poisson_count(expected_add_count * weight / total_weight),
            zeros_pos.numel(),
        )
        remove_positions = _pick_positions(ones_pos, remove_count)
        add_positions = _pick_positions(zeros_pos, add_count)
        if remove_positions.numel() > 0:
            flat[remove_positions] = 0
        if add_positions.numel() > 0:
            flat[add_positions] = 1

    return noisy
