"""Scoring states in worker processes within the memory this process may use.

At most one worker runs per CPU, under an address-space limit reserved out of the memory the parent leaves: an equal share of it, or a quarter more than the most a state has needed so far, whichever is larger. A state that does not fit is tried again with twice its reservation, up to all of the memory, and then raises `StateDoesNotFit`. The task's module must not import torch.
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
from multiprocessing.reduction import ForkingPickler
from typing import Any, Callable, Iterator, Mapping

logger = logging.getLogger(__name__)

# The share of the memory kept for the parent beyond what it uses.
PARENT_RESERVE = 0.1
# The least a worker reserves.
MIN_WORKER_MEMORY = 2**30
# What a state reserves beyond the most any state has needed so far.
HEADROOM = 1.25
# How a worker that ran out of memory in native code ends.
_OUT_OF_MEMORY_SIGNALS = {signal.SIGABRT, signal.SIGKILL}


class StateDoesNotFit(MemoryError):
    """A state that does not fit in all of the memory."""

    def __init__(self, index: int, memory: int):
        super().__init__(f"state {index} does not fit in {memory / 2**30:.1f} GB")
        self.index = index
        self.memory = memory


def available_cpus() -> int:
    """The CPUs this process may use."""
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
    """One worker process, its memory limit, and the state it holds, if any."""

    def __init__(self, limit: int, task: str, initializer: str, config: Any):
        self.limit = limit
        ours, theirs = socket.socketpair()
        # This process's import path, and one thread each.
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

    def receive(self) -> tuple[str, Any, int]:
        """``("done", result, peak bytes)``, ``("memory", None, 0)``, or ``("error", text, 0)``."""
        try:
            return self.connection.recv()
        except EOFError:
            code = self.process.wait()
            if -code in _OUT_OF_MEMORY_SIGNALS:
                return "memory", None, 0
            return "error", f"the worker exited with code {code}", 0

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
    """``(index, task(payload))`` for every payload, in the order they finish. Each worker runs `initializer(config)` once, and both functions are imported by name. `workers` and `memory` default to the CPUs and bytes available."""
    if not payloads:
        return
    workers = available_cpus() if workers is None else workers
    memory = available_memory() if memory is None else memory
    budget = memory - _resident() - int(PARENT_RESERVE * memory)
    if budget < MIN_WORKER_MEMORY:
        raise MemoryError(
            f"{memory / 2**30:.1f} GB leaves under {MIN_WORKER_MEMORY / 2**30:.0f} GB for workers beside this process"
        )
    workers = max(1, min(workers, len(payloads)))
    share = max(MIN_WORKER_MEMORY, budget // workers)
    logger.info(
        "Scoring %d states in up to %d workers of %.1f GB each.",
        len(payloads),
        workers,
        share / 2**30,
    )
    names = (_name(task), _name(initializer), config)
    queue = deque(payloads)
    needs: dict[int, int] = {}
    largest = 0

    def reservation(index: int) -> int:
        return min(budget, max(share, int(HEADROOM * largest), needs.get(index, 0)))

    free = budget
    running: list[_Worker] = []
    try:
        while queue or running:
            while queue and len(running) < workers and reservation(queue[0]) <= free:
                index = queue.popleft()
                worker = _Worker(reservation(index), *names)
                free -= worker.limit
                worker.give(index, payloads[index])
                running.append(worker)
            for connection in wait([w.connection for w in running]):
                worker = next(w for w in running if w.connection is connection)
                index = worker.index
                kind, value, peak = worker.receive()
                if kind == "error":
                    raise RuntimeError(f"Scoring state {index} failed: {value}")
                if kind == "memory":
                    if worker.limit >= budget:
                        raise StateDoesNotFit(index, budget)
                    needs[index] = min(budget, 2 * worker.limit)
                    logger.debug(
                        "State %d did not fit in %.1f GB.", index, worker.limit / 2**30
                    )
                    queue.appendleft(index)
                    worker.kill()
                    running.remove(worker)
                    free += worker.limit
                    continue
                if HEADROOM * peak > max(share, HEADROOM * largest):
                    logger.info(
                        "State %d needed %.1f GB, so states now reserve %.1f GB.",
                        index,
                        peak / 2**30,
                        min(budget, HEADROOM * peak) / 2**30,
                    )
                largest = max(largest, peak)
                if queue and reservation(queue[0]) == worker.limit:
                    following = queue.popleft()
                    worker.give(following, payloads[following])
                else:
                    worker.stop()
                    running.remove(worker)
                    free += worker.limit
                yield index, value
    finally:
        for worker in running:
            worker.kill()


def _serve(connection: Connection) -> None:
    """Initialize, then score payloads until told to stop. A state that does not fit ends the worker."""
    # Pickled while there is memory to pickle with.
    out_of_memory = ForkingPickler.dumps(("memory", None, 0))
    task, initializer, config = connection.recv()
    try:
        _resolve(initializer)(config)
        score = _resolve(task)
        fits = True
    except MemoryError:
        fits = False
    while (message := connection.recv()) is not None:
        index, payload = message
        if fits:
            try:
                result = score(payload)
            except MemoryError:
                fits = False
            except Exception:  # noqa: BLE001
                connection.send(("error", traceback.format_exc(), 0))
                return
        if fits:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 2**10
            try:
                connection.send(("done", result, peak))
                continue
            except MemoryError:
                del result
        # Outside the handler, whose traceback holds the state's memory.
        connection.send_bytes(out_of_memory)
        return


if __name__ == "__main__":
    descriptor, limit = int(sys.argv[1]), int(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    _serve(Connection(descriptor))
