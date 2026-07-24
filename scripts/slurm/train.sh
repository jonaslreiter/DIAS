#!/usr/bin/env zsh
#SBATCH --job-name=dias-train
#SBATCH --partition=rleap_gpu_24gb,rleap_gpu_48gb
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --qos=rleap_deadline
#
# DIAS training array task. Do NOT run directly — submit via
#   python scripts/launch.py ...
# which sets resources (--time/--mem/--gres/--array, overriding any #SBATCH
# fallbacks above), exports EXPERIMENT/DOMAIN, and passes the composed Lightning
# config chain as positional args ("$@").

source ~/.zshrc
conda activate ddl
set -euo pipefail

# Use the expandable-segments allocator to avoid the large reserved-vs-allocated
# fragmentation gap (satellite reserved ~15.7 GB for ~5.7 GB of live tensors,
# visitall ~10 GB for ~4 GB). Under GPU sharing co-tenants see the reserved
# pool, so shrinking it (satellite -> 6.2 GB, visitall -> 4.2 GB) is what lets
# the per-domain shard reservations in configs/slurm-resources.yaml actually hold.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
