#!/usr/bin/env zsh
# ── site configuration ────────────────────────────────────────────────────────
# These are the only lines in this file specific to one cluster; everything below the end marker is portable. Adjust them for yours (README: "Running on your own cluster"). The partition is a fallback only: resources, the array spec and the output path come from experiments/launch.py, which reads experiments/configs/slurm-resources.yaml.
#SBATCH --job-name=dias-train
#SBATCH --partition=rleap_gpu_24gb,rleap_gpu_48gb
#SBATCH --output=logs/slurm/%x-%A_%a.out
#
# How a compute node gets a `python` with dias importable. Export DIAS_ENV_SETUP before submitting to replace it; note that it is evaluated by the shell in the shebang above, which is part of this block too.
: ${DIAS_ENV_SETUP:="source ~/.zshrc && conda activate dias"}
# ── end site configuration ────────────────────────────────────────────────────
#
# DIAS training array task. Do NOT run directly; submit via
#   python experiments/launch.py ...
# which sets resources (--time/--mem/--gres/--array/--output, overriding the #SBATCH fallbacks above), exports DIAS_ROOT/EXPERIMENT/DOMAIN/LOGS_DIR, and passes the composed Lightning config chain as positional args ("$@").

eval "$DIAS_ENV_SETUP"
set -euo pipefail

# The expandable-segments allocator, which closes most of the reserved-vs-allocated fragmentation gap. Co-tenants on a shared GPU see the reserved pool, so shrinking it is what lets the per-domain shard reservations in experiments/configs/slurm-resources.yaml hold.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cap the allocator at the share of the card the shards pay for, so a run that outgrows its reservation fails with an out-of-memory error instead of taking memory a co-tenant was promised. A shard is a fraction of one GPU, not a number of bytes: the fraction is this job's shards over the shards Slurm defines per GPU on this node (24 on a 24 GB card), read off the node's Gres unless DIAS_SHARDS_PER_GPU says otherwise. All of a job's shards are on one GPU unless the site sets MULTIPLE_SHARING_GRES_PJ. A job holding a whole GPU (gres gpu:N) gets no cap. The cap covers what PyTorch allocates, not the CUDA context itself (a few hundred MB).
if [[ -n "${SLURM_SHARDS_ON_NODE:-}" ]]; then
  if [[ -z "${DIAS_SHARDS_PER_GPU:-}" ]]; then
    DIAS_SHARDS_PER_GPU=$(scontrol show node "$SLURMD_NODENAME" | awk '
      match($0, /Gres=[^ ]*/) {
        n = split(substr($0, RSTART + 5, RLENGTH - 5), gres, ",")
        for (i = 1; i <= n; i++) {
          sub(/\(.*/, "", gres[i]); k = split(gres[i], f, ":")
          if (f[1] == "gpu") gpus += f[k]; else if (f[1] == "shard") shards += f[k]
        }
      }
      END { if (gpus > 0 && shards > 0) print shards / gpus }')
  fi
  if [[ -z "$DIAS_SHARDS_PER_GPU" ]]; then
    echo "cannot tell how many shards one GPU of $SLURMD_NODENAME holds; set DIAS_SHARDS_PER_GPU" >&2
    exit 1
  fi
  MEMORY_FRACTION=$(awk -v s="$SLURM_SHARDS_ON_NODE" -v g="$DIAS_SHARDS_PER_GPU" 'BEGIN { f = s / g; printf "%.4f", (f > 1 ? 1 : f) }')
  PYTORCH_CUDA_ALLOC_CONF+=",per_process_memory_fraction:$MEMORY_FRACTION"
  echo "gpu memory capped at $SLURM_SHARDS_ON_NODE of $DIAS_SHARDS_PER_GPU shards (fraction $MEMORY_FRACTION)"
fi

SEED=${SLURM_ARRAY_TASK_ID:-0}
# Deterministic run id. It MUST match validate_runs.run_id_for(experiment, domain, seed) so the dependent validation task finds this run's checkpoint.
RUN_ID=$(echo "${EXPERIMENT}-${DOMAIN}-s${SEED}" | sed 's/[^0-9A-Za-z_-]/-/g')

echo "train  domain=$DOMAIN  seed=$SEED  run_id=$RUN_ID"

# Where this run's checkpoints go. With W&B that is the logger's own directory; without one Lightning falls back to default_root_dir, which then has to name the same path so validation still finds the checkpoint. Must agree with launch.run_directory_args.
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  RUN_DIR=(--trainer.default_root_dir "${LOGS_DIR:-logs}/dias/$RUN_ID")
else
  RUN_DIR=(--trainer.logger.init_args.id "$RUN_ID"
           --trainer.logger.init_args.name "$RUN_ID")
fi

# The datamodule's own seed follows this one unless a config gave it one.
python -m dias.main fit "$@" \
  --seed_everything "$SEED" \
  "${RUN_DIR[@]}"
