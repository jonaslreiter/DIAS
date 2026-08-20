import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import Logger

from dias.data.datamodule import PddlGraphDataModule  # noqa: F401  (registered for CLI)
from dias.models.dias import DIAS  # noqa: F401  (registered for CLI)

warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")
warnings.filterwarnings("ignore", message=".*Lightning on SLURM.*")


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--experiment", type=str, help="Experiment description")

        for name in (
            "num_objects",
            "arity_relations",
            "num_actions",
            "action_arities",
            "action_arg_positions",
            "batch_size",
        ):
            parser.link_arguments(
                f"data.{name}",
                f"model.init_args.{name}",
                apply_on="instantiate",
            )


class SaveAndLogConfigCallback(SaveConfigCallback):
    """Save config to the logger run directory and log all hyper-parameters."""

    def setup(self, trainer, pl_module, stage):
        if isinstance(trainer.logger, Logger):
            run_dir = Path(trainer.logger.name) / trainer.logger.version
            os.makedirs(Path(trainer.log_dir) / run_dir, exist_ok=True)
            self.config_filename = run_dir / self.config_filename
            super().setup(trainer, pl_module, stage)

    def save_config(self, trainer, pl_module, stage):
        if isinstance(trainer.logger, Logger):
            import yaml

            yaml_string = self.parser.dump(self.config, skip_none=False)
            config = yaml.safe_load(yaml_string)
            trainer.logger.log_hyperparams(config)


def main():
    # The library reports through `logging`, which discards INFO until someone
    # configures a handler; the entry point is the one place that may.
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    torch.set_float32_matmul_precision("medium")

    CLI(
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(filename="final-{epoch}-{step}"),
                ModelCheckpoint(
                    filename="best-{epoch}-{step}", monitor="valid/loss_adjacency"
                ),
                TQDMProgressBar(),
            ],
            "logger": {
                "class_path": "WandbLogger",
                "init_args": {
                    "save_dir": "logs",
                    "project": "dias",
                    "entity": "team-jonas-reiter",
                },
            },
        },
        save_config_callback=SaveAndLogConfigCallback,
        # Let a re-launched (experiment, domain, seed) overwrite its own saved
        # config instead of aborting — deterministic run ids mean re-runs reuse
        # the same run directory.
        save_config_kwargs={"overwrite": True},
        seed_everything_default=np.random.randint(2**32),
    )


if __name__ == "__main__":
    main()
