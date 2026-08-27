"""Helpers the test modules share."""

import tempfile
import unittest
from pathlib import Path


def temp_dir(case: unittest.TestCase) -> Path:
    """A temporary directory, removed when `case` finishes. Not `TestCase.enterContext`, which is 3.11+ while this package supports 3.10."""
    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    return Path(directory.name)
