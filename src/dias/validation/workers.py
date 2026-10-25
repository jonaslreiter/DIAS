"""Scoring sampled states in worker processes, within the memory this process may use.

One worker runs per CPU, each limited to an equal share of the memory the parent leaves. The states that do not fit their share are scored again once the others are done, in half as many workers with twice the memory each, until a state has all of it; one that does not fit then raises `StateDoesNotFit`. The limit is set before a worker imports anything, so the task's module must not import torch, whose address-space reservations alone exceed a share.
"""

from __future__ import annotations

import importlib
import logging
import os
import resource
import signal
import socket
import subprocess
import sys
import traceback
from collections import deque
from multiprocessing.connection import Connection, wait
from typing import Any, Callable, Iterator, Mapping

logger = logging.getLogger(__name__)

# Of the memory, the part kept for the parent beyond what it already uses: it receives every result.
PARENT_RESERVE = 0.1
# Fewer workers run when the memory would give each less than this.
MIN_WORKER_MEMORY = 2**30
# How a worker that ran out of memory in native code ends: std::bad_alloc escaping to std::terminate, or the kernel's out-of-memory killer.
_OUT_OF_MEMORY_SIGNALS = {signal.SIGABRT, signal.SIGKILL}


class StateDoesNotFit(MemoryError):
    """A state that ran out of memory even alone, with all of it."""

    def __init__(self, index: int, memory: int):
        super().__init__(f"state {index} does not fit in {memory / 2**30:.1f} GB")
        self.index = index
        self.memory = memory


def available_cpus() -> int:
    """The CPUs this process may use: the affinity mask a Slurm job is pinned to, or the machine's count."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def available_memory() -> int:
    """The bytes this process may use: the Slurm job's allocation, or what the machine has available."""
    if "SLURM_MEM_PER_NODE" in os.environ:
        return int(os.environ["SLURM_MEM_PER_NODE"]) * 2**20
    if "SLURM_MEM_PER_CPU" in os.environ:
        return int(os.environ["SLURM_MEM_PER_CPU"]) * 2**20 * available_cpus()
    return _meminfo("MemAvailable:")


def _meminfo(key: str) -> int:
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith(key):
                return int(line.split()[1]) * 2**10
    raise RuntimeError(f"no {key} in /proc/meminfo")


def _resident() -> int:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 2**10
    return 0


def _name(function: Callable) -> str:
    return f"{function.__module__}:{function.__qualname__}"


def _resolve(name: str) -> Callable:
    module, _, qualname = name.partition(":")
    return getattr(importlib.import_module(module), qualname)


class _Worker:
    """One worker process and the state it holds, if any."""

    def __init__(self, limit: int, task: str, initializer: str, config: Any):
        ours, theirs = socket.socketpair()
        # This process's import path, and one thread each: the workers are the parallelism, and every thread pool reserves address space.
        env = os.environ | {"PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        env |= dict.fromkeys(
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"
        )
        self.process = subprocess.Popen(
            [sys.executable, "-m", __name__, str(theirs.fileno()), str(limit)],
            pass_fds=(theirs.fileno(),),
            env=env,
        )
        theirs.close()
        self.connection = Connection(ours.detach())
        self.connection.send((task, initializer, config))
        self.index: int | None = None

    def give(self, index: int, payload: Any) -> None:
        self.index = index
        self.connection.send((index, payload))

    def receive(self) -> tuple[str, Any]:
        """``("done", result)``, ``("memory", None)`` if the state did not fit, or ``("error", text)``."""
        try:
            return self.connection.recv()
        except EOFError:
            code = self.process.wait()
            if -code in _OUT_OF_MEMORY_SIGNALS:
                return "memory", None
            return "error", f"the worker exited with code {code}"

    def stop(self) -> None:
        try:
            self.connection.send(None)
        except OSError:
            pass
        self.connection.close()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def kill(self) -> None:
        self.process.kill()
        self.process.wait()
        self.connection.close()


def score_states(
    task: Callable[[Any], Any],
    initializer: Callable[[Any], None],
    config: Any,
    payloads: Mapping[int, Any],
    workers: int | None = None,
    memory: int | None = None,
) -> Iterator[tuple[int, Any]]:
    """Yield ``(index, task(payload))`` for every payload, in the order they finish.

    Each worker runs `initializer(config)` once, then `task` on one payload after another. `workers` defaults to the CPUs available and `memory` to the bytes available; both functions must be importable by name, from modules that do not import torch.
    """
    if not payloads:
        return
    workers = available_cpus() if workers is None else workers
    memory = available_memory() if memory is None else memory
    budget = memory - _resident() - int(PARENT_RESERVE * memory)
    if budget <= 0:
        raise MemoryError(
            f"{memory / 2**30:.1f} GB leaves nothing for workers beside this process"
        )
    workers = max(1, min(workers, len(payloads), budget // MIN_WORKER_MEMORY))
    logger.info(
        "Scoring %d states in %d workers of %.1f GB each.",
        len(payloads),
        workers,
        budget / workers / 2**30,
    )
    names = (_name(task), _name(initializer), config)
    pending = list(payloads)
    while pending:
        failed: list[int] = []
        yield from _round(pending, payloads, names, workers, budget, failed)
        if failed:
            # Twice the memory each, until a state has all of it.
            workers = max(1, workers // 2)
            logger.warning(
                "%d %s did not fit; scoring again in %d %s of %.1f GB each.",
                len(failed),
                "state" if len(failed) == 1 else "states",
                workers,
                "worker" if workers == 1 else "workers",
                budget / workers / 2**30,
            )
        pending = failed


def _round(
    indices: list[int],
    payloads: Mapping[int, Any],
    names: tuple,
    workers: int,
    budget: int,
    failed: list[int],
) -> Iterator[tuple[int, Any]]:
    """Score `indices` in `workers` workers sharing `budget`, collecting in `failed` those that do not fit their share."""
    share = budget // workers
    queue = deque(indices)
    running: list[_Worker] = []
    try:
        while queue or running:
            while queue and len(running) < workers:
                worker = _Worker(share, *names)
                index = queue.popleft()
                worker.give(index, payloads[index])
                running.append(worker)
            for connection in wait([w.connection for w in running]):
                worker = next(w for w in running if w.connection is connection)
                index = worker.index
                kind, value = worker.receive()
                if kind == "error":
                    raise RuntimeError(f"Scoring state {index} failed: {value}")
                if kind == "memory":
                    if workers == 1:
                        raise StateDoesNotFit(index, budget)
                    logger.debug(
                        "State %d did not fit in %.1f GB.", index, share / 2**30
                    )
                    failed.append(index)
                    worker.kill()
                    running.remove(worker)
                    continue
                # The worker's next state first, so it scores while this result is used.
                if queue:
                    following = queue.popleft()
                    worker.give(following, payloads[following])
                else:
                    worker.stop()
                    running.remove(worker)
                yield index, value
    finally:
        for worker in running:
            worker.kill()


def _serve(connection: Connection) -> None:
    """A worker's life: initialize, then score payloads until told to stop. A state that does not fit ends the worker."""
    task, initializer, config = connection.recv()
    try:
        _resolve(initializer)(config)
        score = _resolve(task)
        ready = None
    except MemoryError:
        ready = "memory"
    while (message := connection.recv()) is not None:
        index, payload = message
        if ready is not None:
            connection.send((ready, None))
            return
        try:
            result = score(payload)
        except MemoryError:
            connection.send(("memory", None))
            return
        except (
            Exception
        ):  # noqa: BLE001 - the parent raises it with the worker's traceback
            connection.send(("error", traceback.format_exc()))
            return
        try:
            connection.send(("done", result))
        except MemoryError:
            del result
            connection.send(("memory", None))
            return


if __name__ == "__main__":
    descriptor, limit = int(sys.argv[1]), int(sys.argv[2])
    # A worker that runs out of memory may abort, and its core dump would be as large as its share.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    _serve(Connection(descriptor))
