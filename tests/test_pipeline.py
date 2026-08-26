"""Tests for what the experiment scripts share.

These are the pieces every launcher had its own copy of, so a divergence used
to be invisible: an sbatch array spec built two ways, a validator named in four
places, ``--log-dir`` in one script and ``--logs-dir`` in the next.
"""

import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pipeline import (
    EXPERIMENT_VALIDATOR,
    PDDL_DIR,
    VALIDATORS,
    add_domain_arguments,
    add_path_arguments,
    available_domains,
    domain_config,
    domain_pddl,
    format_array,
    parse_seeds,
    prepare_slurm_logs,
    resolve_resources,
    resolve_seeds,
    sbatch_resource_opts,
    select_domains,
    train_problem,
    validation_problems,
)


class SeedTest(unittest.TestCase):
    def test_single_values_and_inclusive_ranges(self):
        self.assertEqual(parse_seeds(["3", "7-9"]), [3, 7, 8, 9])
        self.assertEqual(parse_seeds(["3,7-9"]), [3, 7, 8, 9])

    def test_duplicates_collapse_and_the_result_is_sorted(self):
        self.assertEqual(parse_seeds(["5", "1-3", "2"]), [1, 2, 3, 5])

    def test_a_malformed_range_is_refused(self):
        for spec in ("9-3", "x", "-1"):
            with self.subTest(spec=spec), self.assertRaises(SystemExit):
                parse_seeds([spec])

    def test_seeds_and_num_runs_are_mutually_exclusive(self):
        args = SimpleNamespace(seeds=["1"], num_runs=3, validation_config=None)
        with self.assertRaisesRegex(SystemExit, "mutually exclusive"):
            resolve_seeds(args)

    def test_num_runs_counts_from_zero(self):
        args = SimpleNamespace(seeds=None, num_runs=3, validation_config=None)
        self.assertEqual(resolve_seeds(args), [0, 1, 2])


class ArrayTest(unittest.TestCase):
    def test_consecutive_seeds_compress_to_a_range(self):
        self.assertEqual(format_array([0, 1, 2, 3]), "0-3")

    def test_gaps_become_separate_terms(self):
        self.assertEqual(format_array([3, 7, 8, 9]), "3,7-9")
        self.assertEqual(format_array([0, 2, 4]), "0,2,4")

    def test_one_seed(self):
        self.assertEqual(format_array([5]), "5")


class ResourceTest(unittest.TestCase):
    def test_a_domain_overrides_the_defaults_it_names(self):
        cfg = {"defaults": {"time": "12:00:00", "mem": "8G"}, "hanoi": {"mem": "32G"}}
        self.assertEqual(
            resolve_resources(cfg, "hanoi"), {"time": "12:00:00", "mem": "32G"}
        )

    def test_a_domain_with_no_entry_gets_the_defaults(self):
        cfg = {"defaults": {"time": "12:00:00"}, "hanoi": None}
        self.assertEqual(resolve_resources(cfg, "gripper"), {"time": "12:00:00"})
        self.assertEqual(resolve_resources(cfg, "hanoi"), {"time": "12:00:00"})

    def test_every_setting_becomes_its_sbatch_flag(self):
        opts = sbatch_resource_opts(
            {
                "time": "01:00:00",
                "mem": "8G",
                "cpus": 4,
                "gres": "shard:4",
                "sbatch_extra": ["--partition=cpu"],
            }
        )
        self.assertEqual(
            opts,
            [
                "--time=01:00:00",
                "--mem=8G",
                "--cpus-per-task=4",
                "--gres=shard:4",
                "--partition=cpu",
            ],
        )

    def test_an_absent_setting_asks_for_nothing(self):
        self.assertEqual(sbatch_resource_opts({}), [])


class SlurmLogTest(unittest.TestCase):
    def test_the_output_directory_follows_the_log_dir(self):
        """sbatch refuses a job whose --output directory does not exist, and
        this used to be created under the repository's logs/ even when the run
        was directed elsewhere."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            prepare_slurm_logs(Path(tmp) / "elsewhere")
            self.assertTrue((Path(tmp) / "elsewhere" / "slurm").is_dir())


class DomainTest(unittest.TestCase):
    """A domain is a folder under pddl/; nothing lists them.

    Every file of a domain follows from its name, so these check that the
    layout the whole pipeline assumes actually holds for all of them. The
    manifest that used to state it is gone, and with it the drift between the
    two; what is left to get wrong is a folder missing a file.
    """

    def test_no_selection_means_every_domain_folder(self):
        domains = available_domains()
        self.assertIn("gripper", domains)
        self.assertEqual(select_domains(None), domains)

    def test_an_unknown_domain_names_the_ones_that_exist(self):
        with self.assertRaises(SystemExit) as raised:
            select_domains(["grippre"])
        self.assertIn("gripper", str(raised.exception))

    def test_every_domain_has_the_files_the_pipeline_asks_it_for(self):
        for domain in available_domains():
            with self.subTest(domain=domain):
                for path in (
                    domain_pddl(domain),
                    train_problem(domain),
                    domain_config(domain),
                    *validation_problems(domain),
                ):
                    self.assertTrue(path.is_file(), path)

    def test_the_validation_problems_are_the_held_out_ones(self):
        self.assertEqual(
            validation_problems("gripper"),
            [PDDL_DIR / "gripper" / f"test-{i}.pddl" for i in (1, 2, 3)],
        )
        self.assertNotIn(train_problem("gripper"), validation_problems("gripper"))


class ValidatorRegistryTest(unittest.TestCase):
    def test_every_experiment_type_maps_to_a_registered_validator(self):
        for experiment_type, validator in EXPERIMENT_VALIDATOR.items():
            with self.subTest(experiment_type=experiment_type):
                self.assertIn(validator, VALIDATORS)

    def test_only_a_lifted_schema_can_be_binarized(self):
        """The threshold is a property of reading a schema off a model, so the
        two validators that produce no lifted schema must refuse one."""
        self.assertTrue(VALIDATORS["dias"].binarizes_schema)
        self.assertFalse(VALIDATORS["mlp"].binarizes_schema)
        self.assertFalse(VALIDATORS["l1"].binarizes_schema)

    def test_l1_is_the_one_validator_without_a_checkpoint(self):
        self.assertFalse(VALIDATORS["l1"].from_checkpoint)
        self.assertTrue(VALIDATORS["dias"].from_checkpoint)
        self.assertTrue(VALIDATORS["mlp"].from_checkpoint)

    def test_each_validator_module_holds_a_ModelValidation(self):
        from pipeline import validator_class

        for name in VALIDATORS:
            with self.subTest(validator=name):
                self.assertTrue(callable(validator_class(name)))


class SharedArgumentTest(unittest.TestCase):
    def test_the_launchers_spell_the_log_directory_the_same_way(self):
        parser = argparse.ArgumentParser()
        add_domain_arguments(parser)
        add_path_arguments(parser, train_config=True)
        args = parser.parse_args([])
        for name in ("domains", "num_runs", "seeds"):
            self.assertTrue(hasattr(args, name))
        for name in ("validation_config", "log_dir", "train_config"):
            self.assertIsInstance(getattr(args, name), Path)
        self.assertFalse(hasattr(args, "logs_dir"))


if __name__ == "__main__":
    unittest.main()
