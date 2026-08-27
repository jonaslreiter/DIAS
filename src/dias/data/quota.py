"""How many transitions to draw per action, and how to divide them."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleQuota:
    """The per-action sampling bounds one run draws under.

    Two ways to say how much to sample, either or both: a minimum per action per split (``min_per_action_*``) and a total per split (``num_*``); given both, their ratios must agree, since one pool serves both splits. ``max_per_action_*`` caps what a walk that overshoots keeps.
    """

    min_per_action_train: int | None = None
    min_per_action_valid: int | None = None
    max_per_action_train: int | None = None
    max_per_action_valid: int | None = None
    num_train: int | None = None
    num_valid: int | None = None

    @classmethod
    def with_floors(
        cls,
        min_per_action_train: int | None = None,
        min_per_action_valid: int | None = None,
        min_per_action_train_floor: int | None = None,
        min_per_action_valid_floor: int | None = None,
        **bounds,
    ) -> "SampleQuota":
        """Build a quota, raising the per-action minima to a shared lower bound.

        Overlays are composed after the per-domain config, so a plain ``min_per_action_train`` there would also shrink the domains that sample more.
        """
        return cls(
            min_per_action_train=_floored(
                "min_per_action_train", min_per_action_train, min_per_action_train_floor
            ),
            min_per_action_valid=_floored(
                "min_per_action_valid", min_per_action_valid, min_per_action_valid_floor
            ),
            **bounds,
        )

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
        """The smallest a per-action sample count may be, over both splits."""
        return (
            self.min_per_action_train + self.min_per_action_valid if self.has_min else 0
        )

    @property
    def num_total(self) -> int:
        """How many samples to draw in all, over both splits."""
        return self.num_train + self.num_valid if self.has_num else 0

    @property
    def cap(self) -> int | None:
        """The largest a per-action sample count may be, over both splits."""
        if not self.has_max:
            return None
        return self.max_per_action_train + self.max_per_action_valid

    @property
    def split_weights(self) -> tuple[int, int]:
        """How the sampled pool divides between the train and validation splits."""
        if self.has_num:
            return self.num_train, self.num_valid
        return self.min_per_action_train, self.min_per_action_valid


def _floored(name: str, value: int | None, floor: int | None) -> int | None:
    if floor is None or value is None or value >= floor:
        return value
    logger.info("Raising %s from %s to its floor of %s.", name, value, floor)
    return floor
