# Instance-size scaling

Training iterations per second and peak GPU memory against the number of objects, on `blocks-3`, `logistics` and `delivery`, six instance sizes each, seeds 0 to 2. Each point is a real training run on the usual config chain plus `configs/scaling.yaml`, cut to 20 warm-up and 100 measured steps on an exclusive GPU.

| domain | object counts |
|---|---|
| `blocks-3` | 12, 20, 40, 68, 100, 150 |
| `logistics` | 12, 21, 38, 67, 103, 147 |
| `delivery` | 12, 20, 40, 68, 105, 150 |

## Running it

```bash
python experiments/scaling/launch_scaling.py --experiment scaling
python experiments/scaling/collect_scaling.py --experiment scaling --markdown experiments/results/scaling.md
```

One Slurm array per domain, one task per seed, each walking the domain's sizes in order. The partition and resources are the `slurm:` block of `configs/sizes.yaml`; results on another GPU belong in another `--experiment`. A size that does not fit on the GPU is recorded as `oom`.

`collect_scaling.py` prints a domain × object-count table per metric (iterations per second, peak GPU memory allocated and reserved, step time, sampling time, peak host memory), mean ± standard deviation over the seeds; `--csv` writes the per-run rows instead.

```bash
python experiments/scaling/launch_scaling.py --experiment scaling --dry-run
python experiments/scaling/launch_scaling.py --experiment scaling --domains delivery --seeds 1 2 --skip-existing
python experiments/scaling/launch_scaling.py --experiment smoke --backend local --domains blocks-3 --seeds 0
```

Results are written to `logs/scaling/<experiment>/<domain>/<instance>-s<seed>.json`.

## Instances

`instances/<domain>/n<objects>.pddl` and `instances/index.json` are committed. To change the sizes in `configs/sizes.yaml`, regenerate them from [PDDL generators](https://github.com/AI-Planning/pddl-generators):

```bash
git clone https://github.com/AI-Planning/pddl-generators.git ../pddl-generators
make -C ../pddl-generators/blocksworld/bwstates.1
make -C ../pddl-generators/blocksworld/3ops/2pddl
make -C ../pddl-generators/logistics
python experiments/scaling/generate_instances.py --generators-dir ../pddl-generators
```

The delivery generator needs `tarski`, which the script installs into `experiments/scaling/.venv-delivery`.

## Files

| file | role |
|---|---|
| `configs/sizes.yaml` | instance sizes, seeds, step counts, Slurm resources |
| `configs/scaling.yaml` | the training config overlay |
| `generate_instances.py` | writes the instances from the generators |
| `launch_scaling.py` | submits the arrays, or runs them here |
| `run_scaling.py` | runs one domain's sizes for one seed |
| `benchmark.py` | measures one (domain, size, seed) point |
| `collect_scaling.py` | aggregates the results into tables |
| `slurm/benchmark.sh` | the Slurm array task |
