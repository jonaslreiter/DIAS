"""Train one run, as the `dias` command does, within the GPU shards of the Slurm job. It is the training step of experiments/launch.py."""

import shardguard  # noqa: F401

from dias.main import main

if __name__ == "__main__":
    main()
