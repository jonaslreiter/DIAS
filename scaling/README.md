# Instance-size scaling study

How DIAS training behaves as the *instance* grows: **iterations per second** and
**peak GPU memory** against the number of objects, on `blocks-3`, `logistics`
and `delivery`, three seeds each.

This is a one-off analysis and is deliberately kept out of the main pipeline —
it trains no model, learns no schema, and writes nothing to W&B. Everything it
needs lives in this directory; the only things it borrows from the repository
are the training config chain, the domain files, and two sbatch helpers from
`scripts/launch.py`.

## What is measured

Each point is a real DIAS training run — the ordinary Lightning config chain
(`configs/main.yaml` → `pddl/<domain>/config.yaml` → `scaling/configs/scaling.yaml`)
instantiated through `dias.main.CLI`, with a timing callback attached. Nothing is
stubbed. It is cut short, though: **20 warm-up + 100 measured steps** per point,
not the 10 000 steps of a real run. Throughput and memory reach steady state
within a few steps, so a converged run would cost ~100× more for the same
numbers.

* the first 20 steps are discarded (CUDA context, cuDNN autotuning, allocator
  growth) and the peak-memory counters are reset right after them;
* every step is bracketed by `torch.cuda.synchronize()`, so the times are real
  and not queue-submission times;
* `iterations_per_second` is `1 / median(step time)` over the measured window;
* GPU memory is reported both as `max_memory_allocated` (live tensors) and
  `max_memory_reserved` (allocator pool) — the run sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` exactly as
  `scripts/slurm/train.sh` does, so the reserved figure is the one real training
  would see;
* transition sampling, peak host RSS, and the size of the pre-computed sample
  pool are recorded separately — they are host-side costs, not part of the
  step time.

`scaling/configs/scaling.yaml` holds everything but the instance fixed, which is
the whole point: **100 train + 50 valid transitions per action** for every domain
and every size (delivery would otherwise sample 10× more than blocks-3), batch
size 200 throughout, and no validation, logging or checkpointing inside the
measured window.

## The ladder

Six sizes per domain, aligned on a common target of ≈12 / 20 / 40 / 68 / 100 /
150 objects so the three curves are comparable (`scaling/configs/sizes.yaml`):

| domain | scaled by | object counts |
|---|---|---|
| `blocks-3` | number of blocks | 12, 20, 40, 68, 100, 150 |
| `logistics` | cities × locations × packages × airplanes | 12, 21, 38, 67, 103, 147 |
| `delivery` | grid size (3×3 … 12×12) × packages | 12, 20, 40, 68, 105, 150 |

The three domains scale in different units and cannot all land on the same count,
so each size entry also carries the rung it belongs to (`target`). That is what
aligns the columns of `collect_scaling.py`; the counts actually reached get their
own table there.

Memory grows quadratically in the object count — the model works on dense
`batch × relations × objects × objects` adjacency tensors — so the top of the
ladder is expected to exceed the 24 GB card on `delivery`, which has the most
binary predicates. **Out-of-memory is recorded as a result** (`status: "oom"`),
not treated as a crash, and shows up as `oom` in the aggregated table.

## Instances

`scaling/instances/<domain>/n<objects>.pddl` are generated once from the
[upstream PDDL generators](https://github.com/AI-Planning/pddl-generators) and
committed, together with `index.json` (parameters + object/action/predicate
counts per instance). **Running the study needs neither the generators nor this
section** — they are only needed to change the ladder, which is why they are not
vendored here.

```bash
git clone https://github.com/AI-Planning/pddl-generators.git ../pddl-generators
make -C ../pddl-generators/blocksworld/bwstates.1
make -C ../pddl-generators/blocksworld/3ops/2pddl
make -C ../pddl-generators/logistics

python scaling/generate_instances.py --generators-dir ../pddl-generators
python scaling/generate_instances.py --generators-dir ../pddl-generators \
    --domains delivery --force
```

The generators emit their own dialect of each domain, so the script rewrites
their output to match the repository's `pddl/<domain>/domain.pddl` — the
`blocksworld-3ops` instances get the reflexive `(eq b b)` facts the repo domain's
`(not (eq ?x ?y))` preconditions need, and the `logistics-strips` instances get
`OBJ` → `object`, `in-city` → `loc` and the missing `(vehicle ?v)` facts. The
delivery generator already emits the repo's domain, but needs `tarski`, so it
runs out of a throwaway venv created on demand at `scaling/.venv-delivery`
(gitignored). Every generated instance is loaded with pymimir and its object
count checked against the ladder before it is written.

## Running it

```bash
# Full grid: 3 domains x 6 sizes x 3 seeds = 54 points.
python scaling/launch_scaling.py --experiment scaling-a10

# What would be submitted:
python scaling/launch_scaling.py --dry-run

# Re-run seeds that died, keeping the finished points:
python scaling/launch_scaling.py --experiment scaling-a10 --domains delivery \
    --seeds 1 2 --skip-existing

# Sequentially, here (smoke test):
python scaling/launch_scaling.py --backend local --domains blocks-3 --seeds 0 \
    --experiment smoke
```

One Slurm array per domain, **array task id = seed**, and one task walks that
domain's whole ladder. Sizes share a task on purpose: a curve stitched together
across different physical GPUs would not mean anything. For the same reason the
task asks for an **exclusive** GPU (`--gres=gpu:1`) on a **pinned** partition —
unlike the main training arrays, which take `shard:N` slices of a shared GPU.
Both come from the `slurm:` block of `scaling/configs/sizes.yaml`; it currently
pins `rleap_gpu_24gb` (A10, 24 GB). Changing the card changes where the ladder
hits OOM, so re-runs on a different partition belong in a different
`--experiment`.

## Results

One JSON per point at
`logs/scaling/<experiment>/<domain>/<instance>-s<seed>.json` (under the
gitignored `logs/`, like the rest of the pipeline's outputs).

```bash
python scaling/collect_scaling.py --experiment scaling-a10
python scaling/collect_scaling.py --experiment scaling-a10 --markdown local/scaling.md
python scaling/collect_scaling.py --experiment scaling-a10 --csv local/scaling.csv
```

Prints a domain × object-count matrix per metric (iterations/second, peak GPU
allocated/reserved, step time, sampling time, peak host RSS), mean ± stdev over
the seeds, plus the number of seeds behind each cell. `--csv` writes the raw
per-run rows for plotting.

## Files

| file | role | main-pipeline analogue |
|---|---|---|
| `configs/sizes.yaml` | the ladder, seeds, step counts, Slurm resources | `configs/slurm-resources.yaml` |
| `configs/scaling.yaml` | Lightning overlay, composed last | `configs/experiments/*.yaml` |
| `generate_instances.py` | drives the upstream generators, rewrites their output | — |
| `benchmark.py` | one (domain, size, seed) point → one JSON | `scripts/validation/run_dias.py` |
| `run_scaling.py` | walks one domain's ladder for one seed | `scripts/validate_runs.py` |
| `launch_scaling.py` | submits the arrays / runs locally | `scripts/launch.py` |
| `collect_scaling.py` | aggregates the JSONs into tables | `scripts/collect_results.py` |
| `slurm/benchmark.sh` | the array task | `scripts/slurm/train.sh` |
