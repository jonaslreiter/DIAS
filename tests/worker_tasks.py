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


def fill(megabytes: int) -> int:
    """Allocate in ever smaller pieces until `megabytes` are held or nothing more fits, so a MemoryError leaves no memory free while its traceback holds the pieces."""
    held, size, total = [], 2**20, 0
    while total < megabytes * 2**20:
        try:
            held.append(bytearray(size))
        except MemoryError:
            if size == 16:
                raise
            size //= 16
            continue
        total += size
    return megabytes


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
