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

The 13 PDDL domains used in the paper are bundled under `pddl_files/`.
Additional instances can be regenerated with the upstream generators:
<https://github.com/...> <!-- TODO: link upstream pddl-generators -->

## Quick start

Train DIAS on a single domain/problem:

```bash
dias-train fit --config configs/main.yaml
```

The base config points at one `(domain_path, problem_path)` pair under
`pddl_files/`. Edit those two fields in
`configs/main.yaml` (or override on the CLI) to switch
domain. Training logs and checkpoints land under `logs/`.

### Noise ablation

Input adjacency noise has two models, selected by `data.init_args.noise_model`:

```yaml
# ratio (default): Bernoulli edge flips
data:
  init_args:
    noise_model: ratio
    edge_remove_ratio: 0.1  # paper sweeps 0.0–0.5
    edge_add_ratio: 0.1

# expected_count: Poisson flip budget per adjacency tensor (statics untouched)
data:
  init_args:
    noise_model: expected_count
    expected_removed_atoms: 2.0
    expected_added_atoms: 2.0
```

Set the active model's knobs to `0.0` for the clean setup. The `noise` experiment
(below) uses `expected_count` via `configs/experiments/noise.yaml`.

## Baseline validation

The shipped runners validate a single checkpoint (or learned PDDL domain)
on a single PDDL problem.

```bash
# DIAS model on a problem
python scripts/validation/run_dias.py \
    --model-id <wandb-run-id> \
    --problem  pddl_files/satellite/train.pddl

# MLP baseline
python scripts/validation/run_mlp_baseline.py \
    --model-id <wandb-run-id> \
    --problem  pddl_files/satellite/train.pddl

# L1 baseline (compares a *learned* PDDL domain against the ground truth)
python scripts/validation/run_l1.py \
    --original-domain pddl_files/satellite/domain.pddl \
    --learned-domain  path/to/l1_learned/satellite.pddl \
    --problem         pddl_files/satellite/train.pddl
```

## Reproducing paper results (all/partial)

`scripts/launch.py` is the single entry point for sweeps. For each domain it
submits a **training array** (one task per seed) and a **dependent validation
array** (`--dependency=aftercorr`, so validation task *i* runs after training
task *i*). Every run in a launch shares an `--experiment` name (= W&B group =
results directory). Per-domain sampling comes from `pddl_files/<domain>/config.yaml`,
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
    --set data.init_args.expected_added_atoms=2 \
    --set data.init_args.expected_removed_atoms=2

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
├── validate_runs.py     # compute-side per-run validation
├── collect_results.py   # aggregate results → paper table
└── validation/          # run_dias.py, run_mlp_baseline.py, run_l1.py (single checkpoint)

pddl_files/              # 13 paper domains + domain-settings.yaml
│                        #   each domain folder also holds config.yaml (overrides)
slurm/
├── train.sh             # training array task (source ~/.zshrc; conda activate ddl)
└── validate.sh          # validation array task
```

## Citation

<!-- TODO: BibTeX once the camera-ready is out -->

## License

See [`LICENSE`](LICENSE).
