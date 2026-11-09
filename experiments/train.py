"""Train one run, as the `dias` command does, within the Slurm job's GPU shards and memory; the training step of experiments/launch.py."""

import safe_slurm  # noqa: F401

from dias.main import main

if __name__ == "__main__":
    main()
