"""Tests for the training-step schedules, which are pure functions of the step and hold no state."""

import unittest

from dias.hparams import to_mapping
from dias.schedules import Constant, Exponential, PiecewiseLinear, as_schedule


class ScheduleValuesTest(unittest.TestCase):
    def test_constant_ignores_the_step(self):
        schedule = Constant(0.3)
        self.assertEqual([schedule(0), schedule(999)], [0.3, 0.3])

    def test_exponential_falls_by_factor_every_period(self):
        schedule = Exponential(start=1.0, floor=0.0, factor=0.1, every=500)
        self.assertAlmostEqual(schedule(0), 1.0)
        self.assertAlmostEqual(schedule(500), 0.1)
        # Not clamped at the period: the paper's tau keeps decaying.
        self.assertAlmostEqual(schedule(1000), 0.01)

    def test_the_floor_leaves_the_rate_alone(self):
        """The point of the parametrisation: raising the floor is one change."""
        clean = Exponential(start=1.0, floor=0.0, factor=0.1, every=500)
        floored = Exponential(start=1.0, floor=0.1, factor=0.1, every=500)
        for step in (0, 100, 500, 1000):
            self.assertAlmostEqual(floored(step) - clean(step) * 0.9, 0.1, places=12)
        self.assertAlmostEqual(floored(0), 1.0)
        self.assertAlmostEqual(floored(500), 0.19)
        self.assertAlmostEqual(floored(10**6), 0.1)

    def test_a_zero_period_decays_instantly_to_the_floor(self):
        """The every -> 0 limit, and the sweep's tau = 0 grid point."""
        self.assertEqual(Exponential(start=1.0, factor=0.1, every=0)(0), 0.0)
        self.assertEqual(Exponential(start=1.0, floor=0.1, factor=0.1, every=0)(9), 0.1)

    def test_endpoints_below_the_floor_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "start above floor"):
            Exponential(start=0.1, floor=0.5)
        with self.assertRaisesRegex(ValueError, "positive factor"):
            Exponential(factor=0.0)

    def test_piecewise_linear_interpolates_and_holds_flat_outside(self):
        schedule = PiecewiseLinear({0: 0.0, 10: 1.0})
        self.assertEqual([schedule(0), schedule(5), schedule(10)], [0.0, 0.5, 1.0])
        self.assertEqual(schedule(-5), 0.0)
        self.assertEqual(schedule(20), 1.0)

    def test_schedules_are_stateless(self):
        """Querying out of order must not change later answers."""
        schedule = Exponential(start=1.0, factor=0.1, every=500)
        out_of_order = [schedule(s) for s in (500, 0, 250)]
        self.assertEqual(out_of_order, [schedule(500), schedule(0), schedule(250)])


class ScheduleConfigTest(unittest.TestCase):
    SCHEDULES = (
        Constant(0.0),
        Exponential(start=1.0, floor=0.1, factor=0.1, every=500),
        PiecewiseLinear({0: 1.0, 500: 0.1}),
    )

    def test_config_round_trip_is_lossless(self):
        for schedule in self.SCHEDULES:
            with self.subTest(schedule=schedule):
                self.assertEqual(as_schedule(to_mapping(schedule)), schedule)

    def test_config_holds_only_primitives(self):
        """A dataclass in the checkpoint would make torch.load refuse the file."""
        for schedule in self.SCHEDULES:
            config = to_mapping(schedule)
            self.assertEqual(set(config), {"class_path", "init_args"})
            self.assertIsInstance(config["class_path"], str)
            for value in config["init_args"].values():
                self.assertIsInstance(value, (int, float, dict))

    def test_int_keyed_keypoints_survive(self):
        """JSON would stringify them, and the config form must not."""
        config = to_mapping(PiecewiseLinear({0: 1.0, 500: 0.1}))
        self.assertEqual(list(config["init_args"]["keypoints"]), [0, 500])

    def test_a_schedule_passes_through_unchanged(self):
        schedule = Constant(0.5)
        self.assertIs(as_schedule(schedule), schedule)

    def test_an_unknown_class_path_is_rejected_by_name(self):
        with self.assertRaisesRegex(ValueError, "Exponential.*dias.schedules.Nope"):
            as_schedule({"class_path": "dias.schedules.Nope", "init_args": {}})


if __name__ == "__main__":
    unittest.main()
