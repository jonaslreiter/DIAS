#!/usr/bin/env zsh
# ── site configuration ────────────────────────────────────────────────────────
# These are the only lines in this file specific to one cluster; everything below the end marker is portable. Adjust them for yours (README: "Running on your own cluster"). The partition is a fallback only: resources, the array spec and the output path come from experiments/launch.py, which reads experiments/configs/slurm-resources.yaml.
#SBATCH --job-name=dias-train
#SBATCH --partition=rleap_gpu_24gb,rleap_gpu_48gb
#SBATCH --output=logs/slurm/%x-%A_%a.out
#
# How a compute node gets a `python` with dias importable. Export DIAS_ENV_SETUP before submitting to replace it; note that it is evaluated by the shell in the shebang above, which is part of this block too.
: ${DIAS_ENV_SETUP:="source ~/.zshrc && conda activate dias"}
# ── end site configuration ────────────────────────────────────────────────────
#
# DIAS training array task. Do NOT run directly; submit via
#   python experiments/launch.py ...
# which sets resources (--time/--mem/--gres/--array/--output, overriding the #SBATCH fallbacks above), exports DIAS_ROOT/EXPERIMENT/DOMAIN/LOGS_DIR, and passes the composed Lightning config chain as positional args ("$@").

eval "$DIAS_ENV_SETUP"
set -euo pipefail

# A job past its memory allocation is ended rather than swapped out without bound; see memory_guard.sh.
source "${DIAS_ROOT}/experiments/slurm/memory_guard.sh"

# PyTorch kept within the share of the GPU this job reserved; see gpu_memory_cap.sh.
source "${DIAS_ROOT}/experiments/slurm/gpu_memory_cap.sh"

SEED=${SLURM_ARRAY_TASK_ID:-0}
# Deterministic run id. It MUST match validate_runs.run_id_for(experiment, domain, seed) so the dependent validation task finds this run's checkpoint.
RUN_ID=$(echo "${EXPERIMENT}-${DOMAIN}-s${SEED}" | sed 's/[^0-9A-Za-z_-]/-/g')

echo "train  domain=$DOMAIN  seed=$SEED  run_id=$RUN_ID"

# Where this run's checkpoints go. With W&B that is the logger's own directory; without one Lightning falls back to default_root_dir, which then has to name the same path so validation still finds the checkpoint. Must agree with launch.run_directory_args.
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  RUN_DIR=(--trainer.default_root_dir "${LOGS_DIR:-logs}/dias/$RUN_ID")
else
  RUN_DIR=(--trainer.logger.init_args.id "$RUN_ID"
           --trainer.logger.init_args.name "$RUN_ID")
fi

# The datamodule's own seed follows this one unless a config gave it one.
python -m dias.main fit "$@" \
  --seed_everything "$SEED" \
  "${RUN_DIR[@]}"
