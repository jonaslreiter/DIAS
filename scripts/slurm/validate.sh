#!/usr/bin/env zsh
#SBATCH --job-name=dias-val
#SBATCH --partition=rleap_gpu_24gb,rleap_gpu_48gb
#SBATCH --output=logs/slurm/%x-%A_%a.out
#
# DIAS validation array task. Do NOT run directly — submit via
#   python scripts/launch.py ...
# which submits this array with --dependency=aftercorr on the training array
# (task i validates the checkpoint from training task i) and exports the
# EXPERIMENT/DOMAIN/EXP_TYPE/paths below.

source ~/.zshrc
conda activate ddl
set -euo pipefail

SEED=${SLURM_ARRAY_TASK_ID:-0}
echo "validate  domain=$DOMAIN  seed=$SEED  type=$EXP_TYPE"

python scripts/validate_runs.py \
  --experiment "$EXPERIMENT" \
  --domain "$DOMAIN" \
  --seed "$SEED" \
  --experiment-type "$EXP_TYPE" \
  --logs-dir "$LOGS_DIR" \
  --domain-settings "$DOMAIN_SETTINGS" \
  --validation-config "$VALIDATION_CONFIG"
