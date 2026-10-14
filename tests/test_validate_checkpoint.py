"""How `validate_checkpoint.py` resolves its sampling: validation.yaml's block, the other strategy's defaults when `--sampling` names it, and single settings over either."""

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from dias.validation.successors import BfsSampling, WalkSampling
from pipeline import VALIDATION_CONFIG, load_yaml
from validate_checkpoint import SAMPLING_SETTINGS, _sampling

CONFIGURED = {"strategy": "walks", "num_samples": 40, "walk_length": 30, "min_walks": 2}


def flags(**given) -> argparse.Namespace:
    return argparse.Namespace(
        **{"sampling": None, **dict.fromkeys(SAMPLING_SETTINGS), **given}
    )


class SamplingFlagsTest(unittest.TestCase):
    def test_unset_flags_sample_as_configured(self):
        self.assertEqual(
            _sampling(flags(), CONFIGURED),
            WalkSampling(num_samples=40, walk_length=30, min_walks=2),
        )

    def test_the_committed_configuration_reads(self):
        self.assertIsInstance(
            _sampling(flags(), load_yaml(VALIDATION_CONFIG)["sampling"]), WalkSampling
        )

    def test_naming_the_other_strategy_starts_from_its_defaults(self):
        self.assertEqual(_sampling(flags(sampling="bfs"), CONFIGURED), BfsSampling())
        self.assertEqual(
            _sampling(flags(sampling="bfs", num_samples=100), CONFIGURED),
            BfsSampling(num_samples=100),
        )

    def test_a_setting_overrides_the_configured_one(self):
        self.assertEqual(
            _sampling(flags(walk_length=7), CONFIGURED),
            WalkSampling(num_samples=40, walk_length=7, min_walks=2),
        )

    def test_a_setting_the_strategy_lacks_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "walk_length"):
            _sampling(flags(sampling="bfs", walk_length=7), CONFIGURED)
        with self.assertRaisesRegex(SystemExit, "sample_population"):
            _sampling(flags(sample_population=7), CONFIGURED)


if __name__ == "__main__":
    unittest.main()
