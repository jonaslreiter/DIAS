"""Scoring in worker processes within a memory budget."""

import unittest

import worker_tasks
from dias.validation.workers import (
    PARENT_RESERVE,
    StateDoesNotFit,
    _resident,
    score_states,
)

GB = 2**30


def memory_leaving(budget: int) -> int:
    """The memory to give `score_states` for its workers to share `budget` bytes."""
    return int((_resident() + budget) / (1 - PARENT_RESERVE))


def run(task, payloads, workers=2, budget=3 * GB):
    return dict(
        score_states(
            task,
            worker_tasks.init,
            {"offset": 10},
            payloads,
            workers=workers,
            memory=memory_leaving(budget),
        )
    )


class ScoreStatesTest(unittest.TestCase):
    def test_every_payload_is_scored_once(self):
        self.assertEqual(
            run(worker_tasks.offset, {i: i for i in range(7)}),
            {i: i + 10 for i in range(7)},
        )

    def test_a_state_over_its_share_is_tried_again_with_twice_the_memory(self):
        # Two workers share 3 GB, so 2 GB fits only in all of it.
        self.assertEqual(run(worker_tasks.allocate, {0: 2000, 1: 10}), {0: 2000, 1: 10})

    def test_states_reserve_a_quarter_more_than_the_largest_needed(self):
        # Two workers share 4 GB. State 0 needs 2.5 GB, so it runs alone once state 1 is done, and the states after it reserve 1.25 times its peak.
        with self.assertLogs("dias.validation.workers", "INFO") as logs:
            results = run(
                worker_tasks.sized,
                {0: (2500, 0), 1: (10, 1.0), 2: (10, 0), 3: (10, 0)},
                budget=4 * GB,
            )
        limits = {index: limit for index, (_, limit) in results.items()}
        self.assertAlmostEqual(limits[1] / GB, 2, delta=0.1)
        for later in (2, 3):
            self.assertGreater(limits[later], 1.25 * 2500 * 2**20)
            self.assertLess(limits[later], 4 * GB)
        self.assertTrue(any("State 0 needed" in line for line in logs.output))

    def test_a_state_that_does_not_fit_alone_is_reported(self):
        with self.assertRaises(StateDoesNotFit) as raised:
            run(worker_tasks.allocate, {0: 10, 1: 4000})
        self.assertEqual(raised.exception.index, 1)

    def test_a_single_worker_has_all_of_the_memory_from_the_start(self):
        with self.assertRaises(StateDoesNotFit):
            run(worker_tasks.allocate, {0: 4000}, workers=1)

    def test_an_aborting_worker_counts_as_out_of_memory(self):
        # What std::bad_alloc escaping pymimir does.
        with self.assertRaises(StateDoesNotFit):
            run(worker_tasks.abort, {0: None})

    def test_an_error_carries_the_workers_traceback(self):
        with self.assertRaisesRegex(RuntimeError, "ValueError: boom"):
            run(worker_tasks.fail, {0: None})


if __name__ == "__main__":
    unittest.main()
