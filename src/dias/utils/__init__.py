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


def resolve_label_names(datamodule, num_actions: int, num_relations: int, num_objects: int):
    """Resolve action / relation / object display names from a datamodule with numeric fallbacks."""

    def _normalise(names, target_len, prefix):
        lst = [] if names is None else list(names)
        if len(lst) < target_len:
            lst.extend(f"{prefix} {i}" for i in range(len(lst), target_len))
        return lst[:target_len]

    action_names = _normalise(getattr(datamodule, "action_names", None), num_actions, "Action")
    relation_names = _normalise(getattr(datamodule, "relation_names", None), num_relations, "Rel")

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
        assert len(add_ratio_per_rel) == R, (
            f"add_ratio length {len(add_ratio_per_rel)} != R={R}"
        )

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
