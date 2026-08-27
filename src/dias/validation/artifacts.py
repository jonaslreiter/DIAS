"""Somewhere to put the generated PDDL that pymimir has to read from disk."""

from __future__ import annotations

import tempfile
from pathlib import Path


class PddlWorkspace:
    """A self-deleting directory for one validator's generated PDDL files.

    pymimir parses from paths, so merged domains and stripped problems must exist as files for as long as the validator (and any worker process reading them) does. The directory is removed when the workspace is collected.
    """

    def __init__(self) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="dias-validation-")

    def write(self, name: str, text: str) -> str:
        path = Path(self._directory.name) / name
        path.write_text(text)
        return str(path)
