#!/usr/bin/env zsh
# ── site configuration ────────────────────────────────────────────────────────
# These are the only lines in this file specific to one cluster; everything
# below the end marker is portable. Adjust them for yours (README: "Running on
# your own cluster"). The partition and QoS are fallbacks only -- resources,
# the array spec and the output path come from scripts/launch.py, which reads
# configs/slurm-resources.yaml.
#SBATCH --job-name=dias-train
#SBATCH --partition=rleap_gpu_24gb,rleap_gpu_48gb
#SBATCH --qos=rleap_deadline
#SBATCH --output=logs/slurm/%x-%A_%a.out
#
# How a compute node gets a `python` with dias importable. Export
# DIAS_ENV_SETUP before submitting to replace it; note that it is evaluated by
# the shell in the shebang above, which is part of this block too.
: ${DIAS_ENV_SETUP:="source ~/.zshrc && conda activate ddl"}
# ── end site configuration ────────────────────────────────────────────────────
#
# DIAS training array task. Do NOT run directly — submit via
#   python scripts/launch.py ...
# which sets resources (--time/--mem/--gres/--array/--output, overriding the
# #SBATCH fallbacks above), exports DIAS_ROOT/EXPERIMENT/DOMAIN/LOGS_DIR, and
# passes the composed Lightning config chain as positional args ("$@").

eval "$DIAS_ENV_SETUP"
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

# Where this run's checkpoints go. With W&B that is the logger's own directory;
# without one Lightning falls back to default_root_dir, which then has to name
# the same path so validation still finds the checkpoint. Must agree with
# launch.run_directory_args.
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  RUN_DIR=(--trainer.default_root_dir "${LOGS_DIR:-logs}/dias/$RUN_ID")
else
  RUN_DIR=(--trainer.logger.init_args.id "$RUN_ID"
           --trainer.logger.init_args.name "$RUN_ID")
fi

python -m dias.main fit "$@" \
  --seed_everything "$SEED" \
  --data.init_args.seed "$SEED" \
  "${RUN_DIR[@]}"
