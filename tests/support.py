"""Helpers the test modules share."""

import tempfile
import unittest
from pathlib import Path


def temp_dir(case: unittest.TestCase) -> Path:
    """A temporary directory, removed when `case` finishes.

    Not `TestCase.enterContext`, which reads better but arrived in Python 3.11 while this package supports 3.10, and a test suite that cannot run on the oldest supported version is not testing that version.
    """
    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    return Path(directory.name)
