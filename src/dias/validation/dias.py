"""Scoring a DIAS run by its learned domain."""

from __future__ import annotations

from .artifacts import PddlWorkspace
from .checkpoint import learned_pddl, load_run
from .pddl import PddlValidator


class DiasValidator(PddlValidator):
    def __init__(
        self,
        model_id,
        logs_path,
        problem_path,
        threshold=None,
        force_injective=False,
    ):
        run = load_run(model_id, logs_path)
        text, threshold = learned_pddl(run, threshold)
        self.learned = PddlWorkspace()
        super().__init__(
            run.domain_path,
            self.learned.write("learned-domain.pddl", text),
            problem_path,
            force_injective,
        )
        self.threshold = threshold
