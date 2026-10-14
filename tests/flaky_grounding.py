"""Stand-ins for `grounding.ground_state` that kill the worker process calling them, for the MLP-effects validator's recovery from a broken pool. They live in a module of their own because a spawned worker imports the function it runs by name."""

import os

from dias.validation.grounding import ground_state


def dies_once(payload):
    """Kill this worker the first time any worker calls it, then ground as usual. The marker file named by DIAS_TEST_DIE_ONCE is what the processes share."""
    try:
        os.close(os.open(os.environ["DIAS_TEST_DIE_ONCE"], os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return ground_state(payload)
    os._exit(1)


def always_dies(payload):
    os._exit(1)
