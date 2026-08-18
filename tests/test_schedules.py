"""Tests for the training-step schedules.

These replaced a stateful LossWeightSchedule whose __call__ mutated a private
field that __getitem__ then read, so the collection was only valid after
update_loss_weights had been called for the right step. A schedule is a pure
function of the step, and these pin that.
"""

import unittest

from dias.schedules import (
    Constant,
    Exponential,
    PiecewiseLinear,
    as_schedule,
    schedule_config,
)


class ScheduleValuesTest(unittest.TestCase):
    def test_constant_ignores_the_step(self):
        schedule = Constant(0.3)
        self.assertEqual([schedule(0), schedule(999)], [0.3, 0.3])

    def test_exponential_hits_its_endpoints_and_keeps_going(self):
        schedule = Exponential(v0=1.0, vT=0.1, T=500)
        self.assertAlmostEqual(schedule(0), 1.0)
        self.assertAlmostEqual(schedule(500), 0.1)
        # Not clamped at T: the paper's tau keeps decaying toward zero.
        self.assertAlmostEqual(schedule(1000), 0.01)

    def test_exponential_with_zero_period_decays_instantly(self):
        """The T -> 0 limit, and the sweep's tau = 0 grid point."""
        self.assertEqual(Exponential(v0=1.0, vT=0.1, T=0)(0), 0.0)
        self.assertEqual(Exponential(v0=1.0, vT=0.1, T=0)(500), 0.0)

    def test_piecewise_linear_interpolates_and_holds_flat_outside(self):
        schedule = PiecewiseLinear({0: 0.0, 10: 1.0})
        self.assertEqual([schedule(0), schedule(5), schedule(10)], [0.0, 0.5, 1.0])
        self.assertEqual(schedule(-5), 0.0)
        self.assertEqual(schedule(20), 1.0)

    def test_schedules_are_stateless(self):
        """Querying out of order must not change later answers."""
        schedule = Exponential(v0=1.0, vT=0.1, T=500)
        out_of_order = [schedule(s) for s in (500, 0, 250)]
        self.assertEqual(out_of_order, [schedule(500), schedule(0), schedule(250)])


class ScheduleConfigTest(unittest.TestCase):
    SCHEDULES = (
        Constant(0.0),
        Exponential(0.9, 0.09, 500),
        PiecewiseLinear({0: 1.0, 500: 0.1}),
    )

    def test_config_round_trip_is_lossless(self):
        for schedule in self.SCHEDULES:
            with self.subTest(schedule=schedule):
                self.assertEqual(as_schedule(schedule_config(schedule)), schedule)

    def test_config_holds_only_primitives(self):
        """A dataclass in the checkpoint would make torch.load refuse the file."""
        for schedule in self.SCHEDULES:
            config = schedule_config(schedule)
            self.assertEqual(set(config), {"class_path", "init_args"})
            self.assertIsInstance(config["class_path"], str)
            for value in config["init_args"].values():
                self.assertIsInstance(value, (int, float, dict))

    def test_int_keyed_keypoints_survive(self):
        """JSON would stringify them; the config form must not."""
        config = schedule_config(PiecewiseLinear({0: 1.0, 500: 0.1}))
        self.assertEqual(list(config["init_args"]["keypoints"]), [0, 500])

    def test_a_schedule_passes_through_unchanged(self):
        schedule = Constant(0.5)
        self.assertIs(as_schedule(schedule), schedule)

    def test_an_unknown_class_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown schedule"):
            as_schedule({"class_path": "dias.schedules.Nope", "init_args": {}})


if __name__ == "__main__":
    unittest.main()
