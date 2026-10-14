import logging
import os
import random
import warnings
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.io import CheckpointIO

from dias.data.datamodule import PddlGraphDataModule  # noqa: F401  (registered for CLI)
from dias.models.dias import DIAS  # noqa: F401  (registered for CLI)
from dias.schema.export import learned_domain

logger = logging.getLogger("dias.run")

warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")
warnings.filterwarnings("ignore", message=".*Lightning on SLURM.*")


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        for name in (
            "num_objects",
            "relation_arities",
            "num_actions",
            "action_arities",
            "action_arg_positions",
        ):
            parser.link_arguments(
                f"data.{name}",
                f"model.init_args.{name}",
                apply_on="instantiate",
            )

    def before_instantiate_classes(self):
        """Say what this run is, before anything slow starts: a chain of config files plus command-line overrides does not say on its face which domain it landed on."""
        super().before_instantiate_classes()
        config = self.config.get(self.subcommand, self.config)
        self._seed_the_data_from_the_run(config)
        logger.info(
            "Composed from %s.",
            ", ".join(str(path) for path in config.get("config") or []) or "defaults",
        )
        data = config.get("data", {}).get("init_args", {})
        logger.info(
            "Training on %s / %s, seed %s.",
            data.get("domain_path"),
            data.get("problem_path"),
            config.get("seed_everything"),
        )

    @staticmethod
    def _seed_the_data_from_the_run(config) -> None:
        """Let the walk follow the run's seed unless it was given one of its own, so the saved config records the number it sampled under."""
        data = config.get("data", {}).get("init_args", {})
        seed = config.get("seed_everything")
        # A bool passes for an int here: True means "draw one", already replaced by the drawn number unless the run is not seeding at all.
        if isinstance(seed, bool) or not isinstance(seed, int):
            return
        if getattr(data, "seed", False) is None:
            data.seed = seed


LEARNED_DOMAIN = "learned_domain"


class LearnedDomainCheckpoints(Callback):
    """Every checkpoint is saved with its learned domain beside it, as PDDL under the same name, and removed with it."""

    def setup(self, trainer, pl_module, stage):
        if not isinstance(trainer.strategy.checkpoint_io, _LearnedDomainIO):
            trainer.strategy.checkpoint_io = _LearnedDomainIO(
                trainer.strategy.checkpoint_io
            )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint[LEARNED_DOMAIN] = learned_domain(
            pl_module,
            trainer.datamodule.schema,
            trainer.datamodule.hparams,
            pl_module.hparams.schema_threshold,
        )


class _LearnedDomainIO(CheckpointIO):
    def __init__(self, io: CheckpointIO):
        self.io = io

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        domain = checkpoint.pop(LEARNED_DOMAIN, None)
        self.io.save_checkpoint(checkpoint, path, storage_options)
        if domain is not None:
            Path(path).with_suffix(".pddl").write_text(domain)

    def load_checkpoint(self, *args, **kwargs):
        return self.io.load_checkpoint(*args, **kwargs)

    def remove_checkpoint(self, path):
        self.io.remove_checkpoint(path)
        Path(path).with_suffix(".pddl").unlink(missing_ok=True)

    def teardown(self):
        self.io.teardown()


class SaveAndLogConfigCallback(SaveConfigCallback):
    """Save config to the logger run directory and log all hyper-parameters."""

    def setup(self, trainer, pl_module, stage):
        if isinstance(trainer.logger, Logger):
            run_dir = Path(trainer.logger.name or "") / str(trainer.logger.version)
            os.makedirs(Path(trainer.log_dir) / run_dir, exist_ok=True)
            self.config_filename = run_dir / self.config_filename
            super().setup(trainer, pl_module, stage)
            logger.info("Writing this run to %s.", Path(trainer.log_dir) / run_dir)
        else:
            logger.info("Writing this run to %s.", trainer.default_root_dir)

    def save_config(self, trainer, pl_module, stage):
        if isinstance(trainer.logger, Logger):
            import yaml

            yaml_string = self.parser.dump(self.config, skip_unset=False)
            config = yaml.safe_load(yaml_string)
            trainer.logger.log_hyperparams(config)


def main():
    # The library discards INFO until someone configures a handler; an entry point is the only place that may.
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    torch.set_float32_matmul_precision("medium")

    CLI(
        # No logger: the W&B account belongs in configs/main.yaml, not the library.
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(filename="final-{epoch}-{step}"),
                ModelCheckpoint(
                    filename="best-{epoch}-{step}", monitor="valid/loss_adjacency"
                ),
                LearnedDomainCheckpoints(),
                TQDMProgressBar(),
            ],
        },
        save_config_callback=SaveAndLogConfigCallback,
        # Run ids are deterministic, so a re-run reuses its directory and overwrites its own saved config.
        save_config_kwargs={"overwrite": True},
        # An unseeded run still records the seed it drew, in the config it saves.
        seed_everything_default=random.randrange(2**32),
    )


if __name__ == "__main__":
    main()
