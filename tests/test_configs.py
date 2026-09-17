"""The config chains a run is composed from, so a renamed ``init_args`` key fails here rather than on every task of a Slurm array.

Checked by parsing, not instantiating: the CLI's parser validates class paths, argument names and types, and skipping construction keeps the whole grid to a few seconds. The round trip is the one test that instantiates.
"""

import contextlib
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dias.main import CLI

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_CONFIGS = sorted((ROOT / "pddl").glob("*/config.yaml"))
MAIN = ROOT / "configs" / "main.yaml"
# configs/ is flat, and every file in it but the base is an overlay, so a new one is covered from the moment it is added.
OVERLAYS = sorted(path for path in (ROOT / "configs").glob("*.yaml") if path != MAIN)
# A domain to carry the overlays. They are domain-independent, so the whole cross product would parse the same files again for no extra cover.
A_DOMAIN = ROOT / "pddl" / "gripper" / "config.yaml"
SCALING_OVERLAY = ROOT / "experiments" / "scaling" / "configs" / "scaling.yaml"


class ParseOnlyCLI(CLI):
    """The real CLI, stopped after parsing.

    ``instantiate_classes`` is where the datamodule reads its PDDL and the argument links fire; neither is needed to know that a chain composes. The seed is left alone because these run beside tests that pin a trajectory, and seeding globally from a config file would move their random stream.
    """

    def instantiate_classes(self):
        pass

    def _set_seed(self):
        pass


def parse(*paths, overrides=()):
    args = []
    for path in paths:
        args += ["--config", str(path)]
    return ParseOnlyCLI(args=args + list(overrides), run=False).config


def parse_base():
    """The base config alone, with the domain it deliberately does not name.

    ``configs/main.yaml`` carries no ``domain_path`` / ``problem_path``, and neither does the datamodule, so parsing it on its own is an error by design (see ``RequiredValuesTest``). Supplying the two on the command line is how a test reads what the base config itself sets, without a domain config layered over it.
    """
    return parse(
        MAIN,
        overrides=[
            "--data.init_args.domain_path=pddl/gripper/domain.pddl",
            "--data.init_args.problem_path=pddl/gripper/train.pddl",
        ],
    )


@contextlib.contextmanager
def unseeded():
    """Undo the seeding a fully constructed CLI performs.

    ``seed_everything`` is global, and this module runs beside tests that pin a training trajectory; leaving their random stream where a config file put it would make them pass or fail depending on test order.
    """
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


class CompositionTest(unittest.TestCase):
    def test_every_domain_composes_onto_the_base_config(self):
        for config in DOMAIN_CONFIGS:
            with self.subTest(domain=config.parent.name):
                parse(MAIN, config)

    def test_every_experiment_overlay_composes_onto_a_domain(self):
        for overlay in OVERLAYS:
            with self.subTest(overlay=overlay.name):
                parse(MAIN, A_DOMAIN, overlay)

    def test_the_scaling_study_composes_the_same_way(self):
        parse(MAIN, A_DOMAIN, SCALING_OVERLAY)


class DomainSelectionTest(unittest.TestCase):
    """Each domain config names its own folder's files.

    A domain's files follow from its folder name, and ``pddl/<domain>/ config.yaml`` repeats two of them so that composing that config trains that domain. Repetition nothing checks is how the two drift apart, and the drift is invisible: the run trains the wrong domain under the right name.
    """

    def test_every_domain_config_selects_its_own_domain(self):
        for config in DOMAIN_CONFIGS:
            name = config.parent.name
            with self.subTest(domain=name):
                data = parse(MAIN, config).data.init_args
                self.assertEqual(data.domain_path, f"pddl/{name}/domain.pddl")
                self.assertEqual(data.problem_path, f"pddl/{name}/train.pddl")

    def test_every_domain_folder_has_a_config(self):
        folders = sorted(
            path.parent.name for path in (ROOT / "pddl").glob("*/domain.pddl")
        )
        self.assertEqual(folders, sorted(c.parent.name for c in DOMAIN_CONFIGS))


class RequiredValuesTest(unittest.TestCase):
    """What no default can stand in for is refused rather than guessed.

    Which transitions a run learns from is the run's own choice, so ``domain_path`` and ``problem_path`` have a default in neither the base config nor the datamodule: a base config that named a domain is how a launch trains gripper under satellite's name and nothing says so.
    """

    def test_the_base_config_alone_is_not_a_runnable_configuration(self):
        with self.assertRaises(SystemExit):
            parse(MAIN)

    def test_a_domain_config_completes_it(self):
        data = parse(MAIN, ROOT / "pddl/hanoi/config.yaml").data.init_args
        self.assertEqual(data.domain_path, "pddl/hanoi/domain.pddl")
        self.assertEqual(data.problem_path, "pddl/hanoi/train.pddl")


class SeedTest(unittest.TestCase):
    """One seed per run, not two.

    The walk's seed and the run's seed were always set to the same value by the launcher, in two places, so forgetting one was silent: the run sampled different transitions under the same name. ``data.seed`` now defaults to None, and ``dias.main.CLI`` fills it in from ``seed_everything`` before anything is constructed, so the config a run saves records the number it actually sampled under.
    """

    def test_the_walk_follows_the_runs_seed(self):
        config = parse(MAIN, A_DOMAIN, overrides=["--seed_everything=7"])
        self.assertEqual(config.data.init_args.seed, 7)

    def test_a_seed_of_its_own_wins(self):
        config = parse(
            MAIN,
            A_DOMAIN,
            overrides=["--seed_everything=7", "--data.init_args.seed=3"],
        )
        self.assertEqual(config.data.init_args.seed, 3)


class PrecedenceTest(unittest.TestCase):
    """Later layers win. This is what the launcher relies on and nothing states."""

    def test_a_domain_config_beats_the_base_config(self):
        # aux_loss_alpha is not in main.yaml at all: the base chain resolves it to the class default, and every domain tunes it.
        base = parse_base()
        tuned = parse(MAIN, A_DOMAIN)
        self.assertEqual(base.model.init_args.aux_loss_alpha, 0.1)
        self.assertEqual(tuned.model.init_args.aux_loss_alpha, 1.0)

    def test_an_overlay_beats_the_domain_config(self):
        config = parse(MAIN, A_DOMAIN, ROOT / "configs/experiment-mlp.yaml")
        self.assertTrue(config.model.init_args.state_dependent_effects)

    def test_the_noise_overlay_raises_the_per_action_floors(self):
        # A floor raises the smallest minimum without shrinking the domains that sample more.
        config = parse(MAIN, A_DOMAIN, ROOT / "configs/experiment-noise.yaml")
        self.assertEqual(config.data.init_args.min_per_action_train_floor, 200)
        self.assertEqual(config.data.init_args.min_per_action_valid_floor, 100)
        self.assertEqual(config.data.init_args.min_per_action_train, 200)
        self.assertEqual(config.model.init_args.precondition_tau.floor, 0.1)

    def test_a_command_line_override_beats_every_file(self):
        # How a sweep point reaches a run: appended last, one dotted key, and setting one field of the schedule leaves the others alone.
        config = parse(
            MAIN,
            A_DOMAIN,
            ROOT / "configs/experiment-noise.yaml",
            overrides=["--model.init_args.precondition_tau.every=100"],
        )
        schedule = config.model.init_args.precondition_tau
        self.assertEqual(schedule.every, 100)
        self.assertEqual(schedule.floor, 0.1)
        self.assertEqual(schedule.start, 1.0)


class SavedConfigTest(unittest.TestCase):
    def test_a_saved_config_reproduces_the_schedule_it_was_run_with(self):
        """The config a run saves is the record of what that run was.

        The parser resolves the schedule union structurally, so the dump carries its fields and no ``class_path``. That round-trips only while no two schedules share a field set.
        """
        # This is the one test that builds a Trainer, and main.yaml asks for a GPU and for W&B.
        plain = ["--trainer.accelerator=cpu", "--trainer.logger=false"]
        with unseeded():
            cli = CLI(
                args=["--config", str(MAIN), "--config", str(A_DOMAIN), *plain],
                run=False,
            )
            original = cli.model.precondition_tau

            with tempfile.TemporaryDirectory() as tmp:
                saved = Path(tmp) / "config.yaml"
                saved.write_text(cli.parser.dump(cli.config, skip_unset=False))
                reloaded = CLI(args=["--config", str(saved), *plain], run=False).model

        self.assertIs(type(reloaded.precondition_tau), type(original))
        self.assertEqual(reloaded.precondition_tau, original)


if __name__ == "__main__":
    unittest.main()
