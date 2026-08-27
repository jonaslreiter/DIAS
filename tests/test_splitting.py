"""Tests for the action-stratified train/validation split."""

import unittest

import torch

from dias.data.splitting import (
    cap_per_action,
    check_every_action_survives,
    stratified_split,
)


def samples(*actions: str) -> list[dict]:
    return [{"action": action, "id": i} for i, action in enumerate(actions)]


class CapPerLabelTest(unittest.TestCase):
    def test_it_keeps_the_first_n_of_each_action(self):
        kept = cap_per_action(samples("a", "a", "a", "b"), cap=2)
        self.assertEqual([sample["id"] for sample in kept], [0, 1, 3])

    def test_a_cap_nothing_reaches_changes_nothing(self):
        drawn = samples("a", "b")
        self.assertEqual(cap_per_action(drawn, cap=5), drawn)


class StratifiedSplitTest(unittest.TestCase):
    def split(self, actions: list[int], train_weight=2, valid_weight=1):
        return stratified_split(torch.tensor(actions), train_weight, valid_weight, 0)

    def test_both_splits_get_every_action_that_has_two_samples(self):
        train, valid = self.split([0] * 6 + [1] * 6)
        for indices in (train, valid):
            self.assertEqual({0, 1}, {index // 6 for index in indices})

    def test_the_ratio_decides_what_is_left_over(self):
        train, valid = self.split(list(range(4)) * 30)
        self.assertEqual(len(train) + len(valid), 120)
        self.assertAlmostEqual(len(train) / len(valid), 2.0, delta=0.1)

    def test_an_action_with_one_sample_goes_to_the_train_split(self):
        """It can be learned from, but it cannot also be scored."""
        train, valid = self.split([0, 0, 1])
        self.assertIn(2, train)
        self.assertNotIn(2, valid)

    def test_neither_split_is_ever_empty(self):
        for weight in (1, 1000):
            train, valid = self.split([0, 0, 0, 0], train_weight=weight)
            self.assertTrue(train and valid)

    def test_an_empty_pool_splits_into_nothing(self):
        self.assertEqual(self.split([]), ([], []))

    def test_the_same_seed_gives_the_same_split(self):
        actions = torch.tensor([0, 0, 1, 1, 0, 1])
        self.assertEqual(
            stratified_split(actions, 2, 1, 7), stratified_split(actions, 2, 1, 7)
        )
        self.assertNotEqual(
            stratified_split(actions, 2, 1, 7), stratified_split(actions, 2, 1, 8)
        )


class CoverageTest(unittest.TestCase):
    ACTIONS = torch.tensor([0, 0, 1, 1])

    def test_a_split_holding_every_action_passes(self):
        check_every_action_survives(
            self.ACTIONS, ["move", "stack"], {"train": [0, 2], "valid": [1, 3]}
        )

    def test_a_missing_action_is_named_in_the_error(self):
        with self.assertRaisesRegex(ValueError, "valid.*'stack'"):
            check_every_action_survives(
                self.ACTIONS, ["move", "stack"], {"train": [0, 2], "valid": [1]}
            )

    def test_an_empty_split_is_not_checked(self):
        check_every_action_survives(
            self.ACTIONS, ["move", "stack"], {"train": [0, 2], "valid": []}
        )


if __name__ == "__main__":
    unittest.main()
