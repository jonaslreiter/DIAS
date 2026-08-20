# DIAS

**DIAS** — _Differentiable Induction of Action Schemas._
Code accompanying our paper.

> **Status:** initial public release. Single-domain training, DIAS sweeps,
> and per-checkpoint MLP / L1 baseline validation are supported.

## Installation

```bash
git clone <this-repo>
cd DIAS
pip install -e .
```

This installs the `dias` package (training/model/data code) and the
`dias-train` console entry point.

The 13 PDDL domains used in the paper are bundled under `pddl/`.
Additional instances can be regenerated with the upstream generators
(<https://github.com/AI-Planning/pddl-generators>), vendored as a submodule and
needed only for the scaling study below:

```bash
git submodule update --init submodules/pddl-generators
```

## Quick start

Train DIAS on a single domain/problem:

```bash
dias-train fit --config configs/main.yaml
```

The base config points at one `(domain_path, problem_path)` pair under
`pddl/`. Edit those two fields in
`configs/main.yaml` (or override on the CLI) to switch
domain. Training logs and checkpoints land under `logs/`.

### Noise ablation

Input adjacency noise comes from one of two models, set on `data.init_args.noise`:

```yaml
# Bernoulli edge flips, at a rate per cell
data:
  init_args:
    noise:
      class_path: dias.data.noise.Ratio
      init_args:
        remove: 0.1  # paper sweeps 0.0–0.5
        add: 0.1

# a Poisson flip budget per adjacency tensor, statics untouched
data:
  init_args:
    noise:
      class_path: dias.data.noise.ExpectedCount
      init_args:
        removed_atoms: 2.0
        added_atoms: 2.0
```

Leave `noise` unset for the clean setup. The `noise` experiment (below) uses
`ExpectedCount` via `configs/experiments/noise.yaml`.

## Baseline validation

The shipped runners validate a single checkpoint (or learned PDDL domain)
on a single PDDL problem.

```bash
# DIAS model on a problem
python scripts/validation/run_dias.py \
    --model-id <wandb-run-id> \
    --problem  pddl/satellite/train.pddl

# MLP baseline
python scripts/validation/run_mlp_baseline.py \
    --model-id <wandb-run-id> \
    --problem  pddl/satellite/train.pddl

# L1 baseline (compares a *learned* PDDL domain against the ground truth)
python scripts/validation/run_l1.py \
    --original-domain pddl/satellite/domain.pddl \
    --learned-domain  path/to/l1_learned/satellite.pddl \
    --problem         pddl/satellite/train.pddl
```

## Reproducing paper results (all/partial)

`scripts/launch.py` is the single entry point for sweeps. For each domain it
submits a **training array** (one task per seed) and a **dependent validation
array** (`--dependency=aftercorr`, so validation task _i_ runs after training
task _i_). Every run in a launch shares an `--experiment` name (= W&B group =
results directory). Per-domain sampling comes from `pddl/<domain>/config.yaml`,
per-domain Slurm resources from `configs/slurm-resources.yaml` (fill these in for
your cluster), and validation knobs from `configs/validation.yaml`. The Slurm
scripts `source ~/.zshrc` then `conda activate ddl`.

```bash
# Full grid on Slurm (all domains × 10 seeds), DIAS model
python scripts/launch.py --experiment paper-main

# Two domains only
python scripts/launch.py --experiment paper-main --domains satellite hanoi

# MLP baseline (trains with state_dependent_effects, validates via the MLP validator)
python scripts/launch.py --experiment-type mlp --experiment mlp-main

# One noise level (expected 2 added / 2 removed atoms)
python scripts/launch.py --experiment-type noise --experiment noise-a2r2 \
    --set data.init_args.noise.added_atoms=2 \
    --set data.init_args.noise.removed_atoms=2

# Hyperparameter sweep: all domains × the axis grid × 3 seeds. Each value
# becomes its own sub-experiment `<experiment>-<tag><value>`; the override is
# applied last, so it beats the tuned value in pddl/<domain>/config.yaml.
# Axes: aux-loss-weight {0.01, 0.03, 0.1, 0.3, 1.0}  → `<exp>-aux<α>`
#       num-slots         {5, 7, 9}                  → `<exp>-slots<M>`
#       gmean-temperature {0, 100, 500, 1000, 5000}  → `<exp>-gmeanT<T>`
#       dataset-size      {20…2000 per action}       → `<exp>-n<N>`
# (gmean-temperature T=0 means no annealing: constant τ=0 from step 0, i.e. the
#  exact precondition product instead of the geometric-mean relaxation.)
python scripts/launch.py --experiment aux-sweep --sweep aux-loss-weight --num-runs 3

# --reuse-default takes the axis default (num-slots 5, gmean-temperature 500)
# from an existing experiment instead of re-running it, so this only submits
# `dias-slots7` and `dias-slots9`:
python scripts/launch.py --experiment dias --sweep num-slots \
    --reuse-default dias-full --num-runs 3

# Preview the sbatch commands without submitting
python scripts/launch.py --domains gripper --num-runs 3 --dry-run

# Run locally (no Slurm), sequential — for debugging
python scripts/launch.py --backend local --domains gripper --num-runs 2 \
    --experiment smoke --no-wandb
```

Then collect a finished experiment into a `P / R / N_S / N_C / N_SC` table
(N_S = #seeds sound, i.e. precision 1; N_C = #seeds complete, i.e. recall 1):

```bash
python scripts/collect_results.py --experiment paper-main              # from local JSON
python scripts/collect_results.py --experiment paper-main --source wandb
```

For a sweep, collect all sub-experiments into a domain × value matrix per metric
(`--markdown` also writes them as markdown tables). Pass the same `--sweep` /
`--reuse-default` as the launch, so the columns resolve to the same experiments:

```bash
python scripts/collect_results.py --experiment aux-sweep --sweep aux-loss-weight \
    --markdown logs/aux-sweep.md

python scripts/collect_results.py --experiment dias --sweep num-slots \
    --reuse-default dias-full --markdown logs/num-slots-sweep.md
```

### Dataset-size sensitivity

How `P / R / N_S / N_C / N_SC` behave as the training set is scaled up and down,
clean and under noise. `--sweep dataset-size` pins the sampled pool to exactly
`num_actions × 1.5N` transitions per value N — `min_per_label_train =
max_per_label_train = N`, both `*_valid` = N/2 — so each column is a fixed
dataset size rather than a lower bound with a loose cap. The grid brackets every
tuned per-domain minimum, so each domain is scaled both below and above its own
default.

Six domains × 7 sizes × 3 seeds per setting (126 runs each, 252 in total):

```bash
# Noiseless
python scripts/launch.py --experiment datasize-clean --sweep dataset-size \
    --domains blocks-3 delivery gripper hanoi logistics miconic --num-runs 3

# Noisy: 2 atom flips per transition (0.5 removed + 0.5 added per state, so
# 1 flip in s and 1 in s'). --experiment-type noise also brings the paper's
# noise training setup (batch 50, τ floor 0.1) and validation threshold 0.75.
python scripts/launch.py --experiment datasize-noise --sweep dataset-size \
    --experiment-type noise --num-runs 3 \
    --domains blocks-3 delivery gripper hanoi logistics miconic \
    --set data.init_args.noise.removed_atoms=0.5 \
    --set data.init_args.noise.added_atoms=0.5

# Collect (one domain × size matrix per metric)
python scripts/collect_results.py --experiment datasize-clean \
    --sweep dataset-size --markdown logs/datasize-clean.md
python scripts/collect_results.py --experiment datasize-noise \
    --sweep dataset-size --markdown logs/datasize-noise.md
```

The `noise` overlay floors the per-action minima at 200/100; the sweep overrides
those floors so the 20/50/100 points are not silently clamped back up — the two
launches therefore see identical dataset sizes and differ only in the noise.

### Threshold sensitivity

Reading a schema off a checkpoint binarizes the slot activations and the
add/delete/positive/negative tensors at a single threshold (0.5 for the numbers
in the paper). `scripts/launch_threshold_validation.py` re-runs *only* the
validation step of a finished experiment at other thresholds, on the very same
checkpoints — no retraining. Each threshold writes to its own
`logs/validation/<experiment>-thr<T>/`, so the original results are untouched:

```bash
# 0.1 / 0.25 / 0.5 / 0.75 / 0.9 over all domains × 3 seeds (CPU-only arrays)
python scripts/launch_threshold_validation.py --experiment dias-full --num-runs 3

python scripts/collect_results.py --experiment dias-full --threshold-sweep \
    --markdown logs/threshold-sweep.md
```

### Instance-size scaling (runtime & GPU memory)

A separate one-off study under [`scaling/`](scaling/): how **iterations per
second** and **peak GPU memory** behave as the instance grows, on `blocks-3`,
`logistics` and `delivery` × 6 sizes (≈12 → 150 objects) × 3 seeds. It trains no
model — each point runs 20 warm-up + 100 timed steps of the ordinary training
loop — and keeps everything but the instance fixed (100/50 transitions per
action, batch 200). Instances are generated from the `pddl-generators` submodule
and committed under `scaling/instances/`.

```bash
python scaling/generate_instances.py                       # only to change the ladder
python scaling/launch_scaling.py --experiment scaling-a10  # 3 domains × 6 sizes × 3 seeds
python scaling/collect_scaling.py --experiment scaling-a10 --markdown local/scaling.md
```

Unlike the training arrays these tasks take an **exclusive** GPU on a pinned
partition (a shared shard would put co-tenant load into the throughput numbers).
Out-of-memory at the top of a ladder is recorded as a result, not a crash. See
[`scaling/README.md`](scaling/README.md).

## Repo layout

```
src/dias/
├── main.py              # LightningCLI entry (also `dias-train`)
├── models/              # DIAS model + components (state_dependent_effects = MLP baseline)
├── data/                # PDDL datamodule + sample generation (ratio / expected_count noise)
├── utils/               # schedules, noise, adjacency↔PyG helpers
└── validation/          # ModelValidation classes (DIAS / MLP / L1)

configs/
├── main.yaml            # single-domain training base
├── validation.yaml      # validation knobs + seed list
├── slurm-resources.yaml # per-domain Slurm resources (fill in)
└── experiments/         # mlp.yaml, noise.yaml overlays

scripts/
├── launch.py            # sweep launcher (train array + dependent validation array)
├── launch_threshold_validation.py  # re-validate finished runs at other thresholds
├── validate_runs.py     # compute-side per-run validation
├── collect_results.py   # aggregate results → paper table / sweep matrix (+ markdown)
└── validation/          # run_dias.py, run_mlp_baseline.py, run_l1.py (single checkpoint)

pddl/              # 13 paper domains + domain-settings.yaml
│                        #   each domain folder also holds config.yaml (overrides)
slurm/
├── train.sh             # training array task (source ~/.zshrc; conda activate ddl)
└── validate.sh          # validation array task

scaling/            # one-off runtime / GPU-memory vs. instance-size study
├── configs/             # sizes.yaml (ladder + Slurm), scaling.yaml (overlay)
├── instances/           # generated instances (committed) + index.json
├── generate_instances.py, benchmark.py, run_scaling.py
├── launch_scaling.py, collect_scaling.py
└── slurm/benchmark.sh

submodules/
└── pddl-generators/     # upstream instance generators (submodule)
```

## Citation

<!-- TODO: BibTeX once the camera-ready is out -->

## License

See [`LICENSE`](LICENSE).
