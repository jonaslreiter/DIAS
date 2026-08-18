"""Tests for display-name resolution."""

import unittest

from dias.reporting.labels import resolve_label_names


class ResolveLabelNamesTest(unittest.TestCase):
    def test_missing_names_fall_back_to_numbered_placeholders(self):
        class DataModule:
            action_names = ["move"]
            relation_names = None
            object_names = {"0": "robot"}

        self.assertEqual(
            resolve_label_names(DataModule(), 2, 2, 2),
            (["move", "Action 1"], ["Rel 0", "Rel 1"], ["robot", "Node 1"]),
        )

    def test_no_datamodule_still_yields_a_full_set(self):
        self.assertEqual(
            resolve_label_names(None, 1, 1, 1),
            (["Action 0"], ["Rel 0"], ["Node 0"]),
        )


if __name__ == "__main__":
    unittest.main()
