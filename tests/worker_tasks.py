"""Tasks for tests/test_workers.py, which worker processes import by name."""

import os
import resource
import time

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


def sized(payload: tuple[int, float]) -> tuple[int, int]:
    """Hold `megabytes` for `seconds`, then report them and this worker's memory limit."""
    megabytes, seconds = payload
    block = bytearray(megabytes * 2**20)
    time.sleep(seconds)
    return len(block) // 2**20, resource.getrlimit(resource.RLIMIT_AS)[0]
