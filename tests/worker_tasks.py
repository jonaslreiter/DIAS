"""Tasks for tests/test_workers.py, which worker processes import by name."""

import os

_CONFIG: dict = {}


def init(config: dict) -> None:
    _CONFIG.update(config)


def offset(payload: int) -> int:
    return payload + _CONFIG["offset"]


def allocate(megabytes: int) -> int:
    return len(bytearray(megabytes * 2**20)) // 2**20


def abort(_) -> None:
    os.abort()


def fail(_) -> None:
    raise ValueError("boom")
