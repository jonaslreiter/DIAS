#!/usr/bin/env zsh
# ── site configuration ────────────────────────────────────────────────────────
# See experiments/slurm/train.sh for what this block is and how to adapt it.
#SBATCH --job-name=dias-scaling
#SBATCH --partition=rleap_gpu_24gb
#SBATCH --output=logs/slurm/%x-%A_%a.out
#
: ${DIAS_ENV_SETUP:="source ~/.zshrc && conda activate ddl"}
# ── end site configuration ────────────────────────────────────────────────────
#
# Scaling-study array task: one task = one (domain, seed) = the whole size ladder of that domain, measured on one GPU. Do NOT run directly; submit via
#   python experiments/scaling/launch_scaling.py ...
# which sets resources (--time/--mem/--gres/--array/--partition, overriding the #SBATCH fallbacks above) and exports the variables read below.
#
# Unlike experiments/slurm/train.sh this asks for an exclusive GPU rather than a shard: co-tenants on a shared GPU would show up in the iterations/second.

eval "$DIAS_ENV_SETUP"
set -euo pipefail

# Same allocator setting as experiments/slurm/train.sh, so the reserved-memory numbers this study reports are the ones real training would see.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=${SLURM_ARRAY_TASK_ID:-0}
echo "scaling  domain=$DOMAIN  seed=$SEED  experiment=$EXPERIMENT"

EXTRA=()
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EXTRA+=(--skip-existing)
fi

# DIAS_ROOT rather than a relative path: Slurm runs the batch script from a spool copy, so nothing here may assume the submission directory.
python "${DIAS_ROOT}/experiments/scaling/run_scaling.py" \
  --domain "$DOMAIN" \
  --seed "$SEED" \
  --experiment "$EXPERIMENT" \
  --results-dir "$RESULTS_DIR" \
  --sizes-config "$SIZES_CONFIG" \
  "${EXTRA[@]}"
