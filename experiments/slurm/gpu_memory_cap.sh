# Sourced by the Slurm scripts that run PyTorch on a GPU shard: the allocator settings that keep a job within the share of the card it reserved.

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
