"""The config chains a run is composed from, checked by parsing."""

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
OVERLAYS = sorted(path for path in (ROOT / "configs").glob("*.yaml") if path != MAIN)
A_DOMAIN = ROOT / "pddl" / "gripper" / "config.yaml"
SCALING_OVERLAY = ROOT / "experiments" / "scaling" / "configs" / "scaling.yaml"


class ParseOnlyCLI(CLI):
    """The CLI, stopped after parsing and without seeding."""

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
    """The base config with a domain given on the command line."""
    return parse(
        MAIN,
        overrides=[
            "--data.init_args.domain_path=pddl/gripper/domain.pddl",
            "--data.init_args.problem_path=pddl/gripper/train.pddl",
        ],
    )


@contextlib.contextmanager
def unseeded():
    """Restore the random state a fully constructed CLI seeds."""
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
    """Each domain config names its own folder's files."""

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
    """The domain has no default."""

    def test_the_base_config_alone_is_not_a_runnable_configuration(self):
        with self.assertRaises(SystemExit):
            parse(MAIN)

    def test_a_domain_config_completes_it(self):
        data = parse(MAIN, ROOT / "pddl/hanoi/config.yaml").data.init_args
        self.assertEqual(data.domain_path, "pddl/hanoi/domain.pddl")
        self.assertEqual(data.problem_path, "pddl/hanoi/train.pddl")


class SeedTest(unittest.TestCase):
    """The data's seed is the run's unless set."""

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
    """Later layers win."""

    def test_a_domain_config_beats_the_base_config(self):
        base = parse_base()
        tuned = parse(MAIN, A_DOMAIN)
        self.assertEqual(base.model.init_args.aux_loss_alpha, 0.1)
        self.assertEqual(tuned.model.init_args.aux_loss_alpha, 1.0)

    def test_an_overlay_beats_the_domain_config(self):
        config = parse(MAIN, A_DOMAIN, ROOT / "configs/experiment-mlp.yaml")
        self.assertTrue(config.model.init_args.mlp_effects)

    def test_the_noise_overlay_changes_two_things_beside_the_noise(self):
        clean = parse(MAIN, A_DOMAIN)
        noisy = parse(MAIN, A_DOMAIN, ROOT / "configs/experiment-noise.yaml")
        self.assertEqual(noisy.model.init_args.observation_noise, 0.1)
        self.assertEqual(noisy.data.init_args.batch_size, 50)
        self.assertIsNotNone(noisy.data.init_args.noise)
        for section, key in (
            ("model", "precondition_tau"),
            ("model", "derivation_threshold"),
            ("data", "min_per_action_train"),
        ):
            self.assertEqual(
                noisy[section].init_args[key], clean[section].init_args[key], key
            )

    def test_a_command_line_override_beats_every_file(self):
        config = parse(
            MAIN,
            A_DOMAIN,
            ROOT / "configs/experiment-noise.yaml",
            overrides=["--model.init_args.precondition_tau.every=100"],
        )
        schedule = config.model.init_args.precondition_tau
        self.assertEqual(schedule.every, 100)
        self.assertEqual(schedule.floor, 0.0)
        self.assertEqual(schedule.start, 1.0)


class SavedConfigTest(unittest.TestCase):
    def test_a_saved_config_reproduces_the_schedule_it_was_run_with(self):
        """A run's saved config rebuilds its schedule."""
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
