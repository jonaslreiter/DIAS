#!/usr/bin/env zsh
#SBATCH --job-name=dias-scaling
#SBATCH --partition=rleap_gpu_24gb
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --qos=rleap_deadline
#
# Scaling-study array task: one task = one (domain, seed) = the whole size
# ladder of that domain, measured on one GPU. Do NOT run directly — submit via
#   python scaling/launch_scaling.py ...
# which sets resources (--time/--mem/--gres/--array/--partition, overriding the
# #SBATCH fallbacks above) and exports the variables read below.
#
# Unlike scripts/slurm/train.sh this asks for an exclusive GPU rather than a
# shard: co-tenants on a shared GPU would show up in the iterations/second.

source ~/.zshrc
conda activate ddl
set -euo pipefail

# Same allocator setting as scripts/slurm/train.sh, so the reserved-memory
# numbers this study reports are the ones real training would see.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=${SLURM_ARRAY_TASK_ID:-0}
echo "scaling  domain=$DOMAIN  seed=$SEED  experiment=$EXPERIMENT"

EXTRA=()
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EXTRA+=(--skip-existing)
fi

python scaling/run_scaling.py \
  --domain "$DOMAIN" \
  --seed "$SEED" \
  --experiment "$EXPERIMENT" \
  --results-dir "$RESULTS_DIR" \
  --sizes-config "$SIZES_CONFIG" \
  --domain-settings "$DOMAIN_SETTINGS" \
  "${EXTRA[@]}"
