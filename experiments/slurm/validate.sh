#!/usr/bin/env zsh
# ── site configuration ────────────────────────────────────────────────────────
# See experiments/slurm/train.sh for what this block is and how to adapt it. The partition list here names CPU partitions too: validation runs the model on CPU and spends its time in pymimir successor generation.
#SBATCH --job-name=dias-val
#SBATCH --partition=rleap_gpu_24gb,rleap_gpu_48gb,rleap_cpu,rleap_cpu_modern
#SBATCH --output=logs/slurm/%x-%A_%a.out
#
: ${DIAS_ENV_SETUP:="source ~/.zshrc && conda activate ddl"}
# ── end site configuration ────────────────────────────────────────────────────
#
# DIAS validation array task. Do NOT run directly; submit via
#   python experiments/launch.py ...
# which submits this array with --dependency=aftercorr on the training array (task i validates the checkpoint from training task i) and exports the DIAS_ROOT/EXPERIMENT/DOMAIN/EXP_TYPE/paths below.
#
# experiments/launch_threshold_validation.py reuses this same script to re-validate already-trained checkpoints, exporting the three optional variables below.

eval "$DIAS_ENV_SETUP"
set -euo pipefail

SEED=${SLURM_ARRAY_TASK_ID:-0}
echo "validate  domain=$DOMAIN  seed=$SEED  type=$EXP_TYPE"

# Optional: re-validation knobs (unset => plain validation of a fresh run).
EXTRA=()
if [[ -n "${THRESHOLD:-}" ]]; then
  EXTRA+=(--threshold "$THRESHOLD")
fi
if [[ -n "${RESULTS_EXPERIMENT:-}" ]]; then
  EXTRA+=(--results-experiment "$RESULTS_EXPERIMENT")
fi
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  EXTRA+=(--no-wandb)
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
  EXTRA+=(--num-workers "$NUM_WORKERS")
fi
if [[ -n "${STATES_PER_WORKER:-}" ]]; then
  EXTRA+=(--states-per-worker "$STATES_PER_WORKER")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA+=(--resume)
fi

# DIAS_ROOT rather than a relative path: Slurm runs the batch script from a spool copy, so nothing here may assume the submission directory.
python "${DIAS_ROOT}/experiments/validate_runs.py" \
  --experiment "$EXPERIMENT" \
  --domain "$DOMAIN" \
  --seed "$SEED" \
  --experiment-type "$EXP_TYPE" \
  --log-dir "$LOGS_DIR" \
  --validation-config "$VALIDATION_CONFIG" \
  "${EXTRA[@]}"
