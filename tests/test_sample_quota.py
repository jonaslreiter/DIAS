"""Tests for the per-action sampling bounds a run draws under."""

import unittest

from dias.data.datamodule import SampleQuota


class QuotaValidationTest(unittest.TestCase):
    def test_something_has_to_say_how_much_to_sample(self):
        with self.assertRaisesRegex(ValueError, "Nothing says how much"):
            SampleQuota()

    def test_either_way_of_saying_it_is_enough_on_its_own(self):
        SampleQuota(min_per_action_train=2, min_per_action_valid=1)
        SampleQuota(num_train=200, num_valid=100)

    def test_a_bound_of_zero_would_leave_a_split_empty(self):
        with self.assertRaisesRegex(ValueError, "min_per_action.*> 0"):
            SampleQuota(min_per_action_train=0, min_per_action_valid=1)
        with self.assertRaisesRegex(ValueError, "num_train and num_valid must be > 0"):
            SampleQuota(num_train=1, num_valid=0)
        with self.assertRaisesRegex(ValueError, "max_per_action.*> 0"):
            SampleQuota(
                num_train=2, num_valid=1, max_per_action_train=0, max_per_action_valid=1
            )

    def test_a_cap_is_meaningless_on_one_split_alone(self):
        with self.assertRaisesRegex(ValueError, "set together"):
            SampleQuota(num_train=2, num_valid=1, max_per_action_train=5)

    def test_a_cap_below_the_minimum_could_never_be_met(self):
        with self.assertRaisesRegex(ValueError, "must be >= min_per_action"):
            SampleQuota(
                min_per_action_train=10,
                min_per_action_valid=10,
                max_per_action_train=1,
                max_per_action_valid=2,
            )

    def test_one_pool_cannot_satisfy_two_different_ratios(self):
        with self.assertRaisesRegex(ValueError, "both ratios"):
            SampleQuota(
                min_per_action_train=1,
                min_per_action_valid=1,
                num_train=3,
                num_valid=1,
            )
        # Agreeing ratios, stated at different scales, are fine.
        SampleQuota(
            min_per_action_train=2, min_per_action_valid=1, num_train=200, num_valid=100
        )


class QuotaArithmeticTest(unittest.TestCase):
    QUOTA = SampleQuota(
        min_per_action_train=100,
        min_per_action_valid=50,
        max_per_action_train=1000,
        max_per_action_valid=500,
    )

    def test_the_totals_span_both_splits(self):
        self.assertEqual(self.QUOTA.min_total, 150)
        self.assertEqual(self.QUOTA.cap, 1500)

    def test_what_is_unset_counts_as_nothing(self):
        self.assertEqual(self.QUOTA.num_total, 0)
        self.assertIsNone(SampleQuota(num_train=2, num_valid=1).cap)

    def test_num_decides_the_split_where_both_are_given(self):
        quota = SampleQuota(
            min_per_action_train=2, min_per_action_valid=1, num_train=200, num_valid=100
        )
        self.assertEqual(quota.split_weights, (200, 100))


if __name__ == "__main__":
    unittest.main()
