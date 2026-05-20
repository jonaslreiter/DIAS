#!/bin/bash
#SBATCH --job-name=dias
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/%x-%j.out

# Example slurm submission template for a single DIAS training run.
# Adjust --partition / --account / module loads for your cluster, and point
# PY at the python interpreter of the environment where `pip install -e .`
# was run.

PY=${PY:-python}

$PY -m dias.main fit --config configs/pddl_graph_learner.yaml
