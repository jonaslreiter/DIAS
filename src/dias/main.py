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

logger = logging.getLogger("dias.run")

warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")
warnings.filterwarnings("ignore", message=".*Lightning on SLURM.*")


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        for name in (
            "num_objects",
            "arity_relations",
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
        """Say what this run is, before anything slow starts.

        A run is composed from two or three config files and then overridden on the command line, so which domain it ended up on is not something a reader can see from the command they typed. Reported from the parsed config rather than from the datamodule, so it comes first.
        """
        super().before_instantiate_classes()
        config = self.config.get(self.subcommand, self.config)
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


class SaveAndLogConfigCallback(SaveConfigCallback):
    """Save config to the logger run directory and log all hyper-parameters."""

    def setup(self, trainer, pl_module, stage):
        if isinstance(trainer.logger, Logger):
            run_dir = Path(trainer.logger.name or "") / str(trainer.logger.version)
            os.makedirs(Path(trainer.log_dir) / run_dir, exist_ok=True)
            self.config_filename = run_dir / self.config_filename
            super().setup(trainer, pl_module, stage)
            # The one line a user needs when a run lands somewhere unexpected.
            logger.info("Writing this run to %s.", Path(trainer.log_dir) / run_dir)
        else:
            logger.info("Writing this run to %s.", trainer.default_root_dir)

    def save_config(self, trainer, pl_module, stage):
        if isinstance(trainer.logger, Logger):
            import yaml

            yaml_string = self.parser.dump(self.config, skip_none=False)
            config = yaml.safe_load(yaml_string)
            trainer.logger.log_hyperparams(config)


def main():
    # The library reports through `logging`, which discards INFO until someone configures a handler; the entry point is the one place that may.
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    torch.set_float32_matmul_precision("medium")

    CLI(
        # No logger here: which W&B account a run belongs to is a property of whoever is running it, so it is declared in configs/main.yaml (where it can be edited or overridden) rather than compiled into the library. Without a config, Lightning's own account-free default applies.
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(filename="final-{epoch}-{step}"),
                ModelCheckpoint(
                    filename="best-{epoch}-{step}", monitor="valid/loss_adjacency"
                ),
                TQDMProgressBar(),
            ],
        },
        save_config_callback=SaveAndLogConfigCallback,
        # Let a re-launched (experiment, domain, seed) overwrite its own saved config instead of aborting: deterministic run ids mean re-runs reuse the same run directory.
        save_config_kwargs={"overwrite": True},
        seed_everything_default=np.random.randint(2**32),
    )


if __name__ == "__main__":
    main()
