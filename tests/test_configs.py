"""The config chains a run is actually composed from.

Every training run and every launch composes two or three YAML files -- the
base config, the domain's tuned overrides, an experiment overlay -- and until
this module nothing read any of them. A renamed ``init_args`` key therefore
broke all 13 domains with a fully green suite, and surfaced as a Slurm array
dying on every task.

The chains are checked by *parsing*, not by instantiating: the CLI's parser is
what validates class paths, argument names and types, and skipping construction
keeps the whole grid to a few seconds. The one test that does instantiate is
the round trip, which is about what a saved config means rather than whether it
parses.
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
OVERLAYS = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
MAIN = ROOT / "configs" / "main.yaml"
# A domain to carry the overlays. They are domain-independent, so the whole
# cross product would parse the same files again for no extra cover.
A_DOMAIN = ROOT / "pddl" / "gripper" / "config.yaml"


class ParseOnlyCLI(CLI):
    """The real CLI, stopped after parsing.

    ``instantiate_classes`` is where the datamodule reads its PDDL and the
    argument links fire; neither is needed to know that a chain composes. The
    seed is left alone because these run beside tests that pin a trajectory,
    and seeding globally from a config file would move their random stream.
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


@contextlib.contextmanager
def unseeded():
    """Undo the seeding a fully constructed CLI performs.

    ``seed_everything`` is global, and this module runs beside tests that pin a
    training trajectory; leaving their random stream where a config file put it
    would make them pass or fail depending on test order.
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
        parse(MAIN, A_DOMAIN, ROOT / "scaling" / "configs" / "scaling.yaml")


class DomainSelectionTest(unittest.TestCase):
    """Each domain config names its own folder's files.

    A domain's files follow from its folder name, and ``pddl/<domain>/
    config.yaml`` repeats two of them so that composing that config trains
    that domain. Repetition nothing checks is how the two drift apart, and the
    drift is invisible: the run trains the wrong domain under the right name.
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


class PrecedenceTest(unittest.TestCase):
    """Later layers win. This is what the launcher relies on and nothing states."""

    def test_a_domain_config_beats_the_base_config(self):
        base = parse(MAIN)
        tuned = parse(MAIN, A_DOMAIN)
        self.assertEqual(base.model.init_args.aux_loss_weight, 0.2)
        self.assertEqual(tuned.model.init_args.aux_loss_weight, 1.0)

    def test_a_domain_config_beats_the_base_config_on_the_domain_itself(self):
        # The defect this closes: composing a domain config used to leave the
        # base config's domain in place, so the run trained satellite under
        # gripper's name and nothing said so.
        base = parse(MAIN).data.init_args.domain_path
        composed = parse(MAIN, ROOT / "pddl/hanoi/config.yaml").data.init_args
        self.assertNotEqual(base, composed.domain_path)
        self.assertEqual(composed.domain_path, "pddl/hanoi/domain.pddl")

    def test_an_overlay_beats_the_domain_config(self):
        config = parse(MAIN, A_DOMAIN, ROOT / "configs/experiments/mlp.yaml")
        self.assertTrue(config.model.init_args.state_dependent_effects)

    def test_the_noise_overlay_raises_the_per_action_floors(self):
        # The floors exist because an overlay composed after a domain config
        # must raise the smallest minimum without shrinking the domains that
        # already sample more.
        config = parse(MAIN, A_DOMAIN, ROOT / "configs/experiments/noise.yaml")
        self.assertEqual(config.data.init_args.min_per_label_train_floor, 200)
        self.assertEqual(config.data.init_args.min_per_label_valid_floor, 100)
        self.assertEqual(config.data.init_args.min_per_label_train, 200)
        self.assertEqual(config.model.init_args.gmean_temperature.floor, 0.1)

    def test_a_command_line_override_beats_every_file(self):
        # How a sweep point reaches a run: appended last, one dotted key, and
        # setting one field of the schedule leaves the others alone.
        config = parse(
            MAIN,
            A_DOMAIN,
            ROOT / "configs/experiments/noise.yaml",
            overrides=["--model.init_args.gmean_temperature.every=100"],
        )
        schedule = config.model.init_args.gmean_temperature
        self.assertEqual(schedule.every, 100)
        self.assertEqual(schedule.floor, 0.1)
        self.assertEqual(schedule.start, 1.0)


class SavedConfigTest(unittest.TestCase):
    def test_a_saved_config_reproduces_the_schedule_it_was_run_with(self):
        """The config a run saves is the record of what that run was.

        The tau schedule is a union of three dataclasses, and the parser
        resolves it structurally: the dump carries the schedule's *fields* and
        no ``class_path``. That round-trips today because no two schedules
        share a field set. Adding one that does would silently turn an archived
        run's exponential decay into something else, which is what this pins.
        """
        # The accelerator and the logger are overridden because this is the one
        # test that builds a Trainer: main.yaml asks for a GPU and for W&B, and
        # neither is a property of the schedule being checked.
        plain = ["--trainer.accelerator=cpu", "--trainer.logger=false"]
        with unseeded():
            cli = CLI(args=["--config", str(MAIN), *plain], run=False)
            original = cli.model.gmean_temperature

            with tempfile.TemporaryDirectory() as tmp:
                saved = Path(tmp) / "config.yaml"
                saved.write_text(cli.parser.dump(cli.config, skip_none=False))
                reloaded = CLI(args=["--config", str(saved), *plain], run=False).model

        self.assertIs(type(reloaded.gmean_temperature), type(original))
        self.assertEqual(reloaded.gmean_temperature, original)


if __name__ == "__main__":
    unittest.main()
