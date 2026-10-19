# Sourced by every Slurm script here: ends the job once its RAM and swap together exceed its allocation.
#
# Slurm caps a job's RAM at --mem, but where cgroup.conf leaves swap unconstrained (ConstrainSwapSpace=no) a job past its allocation is swapped out rather than killed, and nothing stops it: validations whose learned domains ground without bound have filled a node's swap until the node went down. A job cannot lower its own swap limit, since its cgroup's control files belong to root, but it can read how much it holds, so a background loop compares that with the allocation every few seconds and, once it is over, kills the batch script, which ends the job as an out-of-memory kill would. Where swap is constrained the job is killed before the comparison can fire, and outside a Slurm memory cgroup, or without swap accounting, there is nothing to read and the guard says so and stays off.

dias_memory_guard() {
  local cgroup limit used file
  local -a usage
  cgroup=$(awk -F: '$2 == "memory" { print $3; exit }' /proc/self/cgroup)
  if [[ -n "$cgroup" ]]; then
    # cgroup v1: memsw is RAM and swap together.
    cgroup=/sys/fs/cgroup/memory${cgroup%%/step_*}
    usage=("$cgroup/memory.memsw.usage_in_bytes")
    limit=$cgroup/memory.limit_in_bytes
  else
    cgroup=$(awk -F: '$1 == "0" { print $3; exit }' /proc/self/cgroup)
    cgroup=/sys/fs/cgroup${cgroup%%/step_*}
    usage=("$cgroup/memory.current" "$cgroup/memory.swap.current")
    limit=$cgroup/memory.max
  fi
  if [[ "${cgroup##*/}" != "job_${SLURM_JOB_ID:-}" ]] || ! cat "$limit" "${usage[@]}" >/dev/null 2>&1; then
    echo "memory guard off: no readable Slurm memory cgroup for this job"
    return 0
  fi
  limit=$(<"$limit")
  if [[ "$limit" == max ]]; then
    echo "memory guard off: the job has no memory limit"
    return 0
  fi
  echo "memory guard on: RAM and swap held to the allocation of $(( limit >> 20 )) MiB"
  {
    while sleep 5; do
      used=0
      for file in "${usage[@]}"; do
        used=$(( used + $(<"$file") ))
      done
      if (( used > limit )); then
        echo "memory guard: the job holds $(( used >> 20 )) MiB in RAM and swap, over its allocation of $(( limit >> 20 )) MiB; killing it" >&2
        # $$ is the batch script even in this subshell; once it is gone, Slurm kills the rest of the job.
        kill -KILL $$
      fi
    done
  } &
}

dias_memory_guard
