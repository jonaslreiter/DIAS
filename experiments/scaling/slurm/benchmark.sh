#!/usr/bin/env zsh
# ── site configuration ────────────────────────────────────────────────────────
#SBATCH --job-name=dias-scaling
#SBATCH --partition=rleap_gpu_24gb
#SBATCH --output=logs/slurm/%x-%A_%a.out
# See experiments/slurm/train.sh.
: ${DIAS_ENV_SETUP:="source ~/.zshrc && conda activate dias"}
# ── end site configuration ────────────────────────────────────────────────────
#
# A scaling array task, submitted by experiments/scaling/launch_scaling.py, which exports DIAS_ROOT, DOMAIN, EXPERIMENT, LOGS_DIR and SIZES_CONFIG.

eval "$DIAS_ENV_SETUP"
set -euo pipefail

# The allocator setting of the training jobs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=${SLURM_ARRAY_TASK_ID:-0}
echo "scaling  domain=$DOMAIN  seed=$SEED  experiment=$EXPERIMENT"

EXTRA=()
[[ "${SKIP_EXISTING:-0}" == "1" ]] && EXTRA+=(--skip-existing)

python "${DIAS_ROOT}/experiments/scaling/run_scaling.py" \
  --domain "$DOMAIN" \
  --seed "$SEED" \
  --experiment "$EXPERIMENT" \
  --log-dir "$LOGS_DIR" \
  --sizes-config "$SIZES_CONFIG" \
  "${EXTRA[@]}"
