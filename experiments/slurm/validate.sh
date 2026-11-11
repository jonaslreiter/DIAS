#!/usr/bin/env zsh
# ── site configuration ────────────────────────────────────────────────────────
#SBATCH --job-name=dias-val
#SBATCH --partition=rleap_gpu_24gb,rleap_gpu_48gb,rleap_cpu,rleap_cpu_modern
#SBATCH --output=logs/slurm/%x-%A_%a.out
# The setup is the one of train.sh.
: ${DIAS_ENV_SETUP:="source ~/.zshrc && conda activate dias"}
# ── end site configuration ────────────────────────────────────────────────────
#
# A validation array task, submitted by experiments/launch.py, which exports DIAS_ROOT, EXPERIMENT, DOMAIN, EXP_TYPE, LOGS_DIR and VALIDATION_CONFIG, and the optional variables below.

eval "$DIAS_ENV_SETUP"
set -euo pipefail

SEED=${SLURM_ARRAY_TASK_ID:-0}
echo "validate  domain=$DOMAIN  seed=$SEED  type=$EXP_TYPE"

EXTRA=()
[[ -n "${THRESHOLD:-}" ]] && EXTRA+=(--threshold "$THRESHOLD")
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(--checkpoint "$CHECKPOINT")
[[ -n "${RESULTS_EXPERIMENT:-}" ]] && EXTRA+=(--results-experiment "$RESULTS_EXPERIMENT")
[[ "${NO_WANDB:-0}" == "1" ]] && EXTRA+=(--no-wandb)
[[ -n "${NUM_WORKERS:-}" ]] && EXTRA+=(--num-workers "$NUM_WORKERS")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(--resume)

python "${DIAS_ROOT}/experiments/validate_runs.py" \
  --experiment "$EXPERIMENT" \
  --domain "$DOMAIN" \
  --seed "$SEED" \
  --experiment-type "$EXP_TYPE" \
  --log-dir "$LOGS_DIR" \
  --validation-config "$VALIDATION_CONFIG" \
  "${EXTRA[@]}"
