"""How `validate_checkpoint.py` resolves its sampling: validation.yaml's block, with the settings given on the command line over it."""

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from dias.validation.successors import WalkSampling
from pipeline import VALIDATION_CONFIG, load_yaml
from validate_checkpoint import SAMPLING_SETTINGS, _sampling

CONFIGURED = {"num_samples": 40, "walk_length": 30, "min_walks": 2}


def flags(**given) -> argparse.Namespace:
    return argparse.Namespace(**{**dict.fromkeys(SAMPLING_SETTINGS), **given})


class SamplingFlagsTest(unittest.TestCase):
    def test_unset_flags_sample_as_configured(self):
        self.assertEqual(_sampling(flags(), CONFIGURED), WalkSampling(**CONFIGURED))

    def test_the_committed_configuration_reads(self):
        self.assertIsInstance(
            _sampling(flags(), load_yaml(VALIDATION_CONFIG)["sampling"]), WalkSampling
        )

    def test_a_setting_overrides_the_configured_one(self):
        self.assertEqual(
            _sampling(flags(walk_length=7), CONFIGURED),
            WalkSampling(num_samples=40, walk_length=7, min_walks=2),
        )

    def test_results_recording_the_walk_strategy_still_read(self):
        self.assertEqual(
            _sampling(flags(), {"strategy": "walks", **CONFIGURED}),
            WalkSampling(**CONFIGURED),
        )


if __name__ == "__main__":
    unittest.main()
