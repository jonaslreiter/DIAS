#!/usr/bin/env zsh
#SBATCH --job-name=dias-train
#SBATCH --partition=rleap_gpu_24gb,rleap_gpu_48gb
#SBATCH --output=logs/slurm/%x-%A_%a.out
#
# DIAS training array task. Do NOT run directly — submit via
#   python scripts/launch.py ...
# which sets resources (--time/--mem/--gres/--array, overriding any #SBATCH
# fallbacks above), exports EXPERIMENT/DOMAIN, and passes the composed Lightning
# config chain as positional args ("$@").

source ~/.zshrc
conda activate ddl
set -euo pipefail

SEED=${SLURM_ARRAY_TASK_ID:-0}
# Deterministic run id — MUST match validate_runs.run_id_for(experiment, domain, seed)
# so the dependent validation task finds this run's checkpoint.
RUN_ID=$(echo "${EXPERIMENT}-${DOMAIN}-s${SEED}" | sed 's/[^0-9A-Za-z_-]/-/g')

echo "train  domain=$DOMAIN  seed=$SEED  run_id=$RUN_ID"

python -m dias.main fit "$@" \
  --seed_everything "$SEED" \
  --data.init_args.seed "$SEED" \
  --trainer.logger.init_args.id "$RUN_ID" \
  --trainer.logger.init_args.name "$RUN_ID"
