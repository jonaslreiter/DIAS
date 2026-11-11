#!/usr/bin/env zsh
# ── site configuration ────────────────────────────────────────────────────────
#SBATCH --job-name=dias-train
#SBATCH --partition=rleap_gpu_24gb,rleap_gpu_48gb
#SBATCH --output=logs/slurm/%x-%A_%a.out
# This puts a python with dias and safe_slurm on PATH. Export DIAS_ENV_SETUP to replace it.
: ${DIAS_ENV_SETUP:="source ~/.zshrc && conda activate dias"}
# ── end site configuration ────────────────────────────────────────────────────
#
# A training array task, submitted by experiments/launch.py, which passes the Lightning arguments and exports DIAS_ROOT, EXPERIMENT, DOMAIN and LOGS_DIR.

eval "$DIAS_ENV_SETUP"
set -euo pipefail

SEED=${SLURM_ARRAY_TASK_ID:-0}
# The run id is built as in validate_runs.run_id_for.
RUN_ID=$(echo "${EXPERIMENT}-${DOMAIN}-s${SEED}" | sed 's/[^0-9A-Za-z_-]/-/g')
echo "train  domain=$DOMAIN  seed=$SEED  run_id=$RUN_ID"

# The run directory is set as in launch.run_directory_args.
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  RUN_DIR=(--trainer.default_root_dir "${LOGS_DIR:-logs}/dias/$RUN_ID")
else
  RUN_DIR=(--trainer.logger.init_args.id "$RUN_ID"
           --trainer.logger.init_args.name "$RUN_ID")
fi

python "${DIAS_ROOT}/experiments/train.py" fit "$@" \
  --seed_everything "$SEED" \
  "${RUN_DIR[@]}"
