# Reproducing the experiments

Each experiment below is one or more `launch.py` commands followed by one `collect_results.py` command, which writes its table to `experiments/results/`. Run everything from the repository root.

## Layout

```
experiments/
├── launch.py                       submits training and validation, start here
├── collect_results.py              validation results -> tables
├── launch_threshold_validation.py  re-validates finished runs at other thresholds
├── validate_checkpoint.py          scores one model on one problem
├── validate_runs.py                scores one (domain, seed); the validation jobs call it
├── precondition_categories.py      counts what the learned preconditions consist of
├── pipeline.py                     shared paths and Slurm helpers
├── configs/                        validation settings and Slurm resources
├── l1_pddl/                        the domains the L1 baseline learned
├── slurm/                          the Slurm job scripts
├── scaling/                        the instance-size study (own README)
└── results/                        collected tables (gitignored)
```

Runs write `logs/dias/<run>/` (checkpoints, each with its learned domain as PDDL beside it), `logs/validation/<experiment>/` (one result JSON per domain and seed) and `logs/slurm/` (job output); `--log-dir` puts them under another directory.

## How a launch works

For each domain, `launch.py` submits a Slurm array that trains one run per seed and an array that validates each run once it has trained; `--backend local` runs the same steps here, one after another. A run is named `<experiment>-<domain>-s<seed>`, and `--experiment` is also its W&B group. By default a launch covers all 14 domains with seeds 0 to 9, labelled with action names. The main table runs all ten seeds; every other experiment runs seeds 0 to 4 (`--seeds 0-4`).

A run's config is composed from these files, later ones winning:

1. `configs/main.yaml`
2. `pddl/<domain>/config.yaml`
3. `configs/experiment-mlp.yaml` or `configs/experiment-noise.yaml` (`--experiment-type`)
4. `configs/labels-strips.yaml` or `configs/labels-strips-plus.yaml` (`--labels`)
5. `configs/ablation-*.yaml` (`--ablation`)
6. `--set key=value` and sweep values

Validation reads `experiments/configs/validation.yaml`: 500 states per held-out problem, drawn from random walks, and the default seeds.

## Experiments

### Main table

```bash
python experiments/launch.py --experiment dias-strips --labels strips
python experiments/launch.py --experiment dias-strips-plus --labels strips-plus
python experiments/launch.py --experiment dias-names
python experiments/launch.py --experiment l1 --experiment-type l1
python experiments/collect_results.py --experiment dias-strips dias-strips-plus dias-names l1 --markdown experiments/results/main.md
```

L1 trains nothing: its launch scores the domains in `l1_pddl/`, every domain but `satellite-recal`.

### MLP effects

```bash
python experiments/launch.py --experiment mlp-names --experiment-type mlp
python experiments/collect_results.py --experiment mlp-names --markdown experiments/results/mlp.md
```

### Noise

`--sweep noise` trains on transitions with 1, 2, 4 and 8 flipped atoms each, every level as its own experiment `noise-f<flips>`.

```bash
python experiments/launch.py --experiment noise --experiment-type noise --sweep noise --seeds 0-4
python experiments/collect_results.py --experiment noise --sweep noise --include-partial --markdown experiments/results/noise.md
```

Under heavy noise some learned schemas are so permissive that a held-out problem cannot be scored within the job's memory or time. `--include-partial` counts such a seed over the problems it did score, and the table ends with a list of the cells concerned.

### Ablations

| `--ablation` | removes |
|---|---|
| `no-slot-weights` | the learned slot activations; every slot stays active |
| `no-precondition-loss` | `L_pre` |
| `no-effect-loss` | `L_eff` |
| `no-aux-loss` | both auxiliary terms (α = 0) |
| `no-pcgrad` | the gradient projection, for a plain weighted sum |
| `no-random-embeddings` | the random initial node features |

```bash
for ablation in no-slot-weights no-precondition-loss no-effect-loss no-aux-loss no-pcgrad no-random-embeddings; do python experiments/launch.py --experiment ablation-$ablation --ablation $ablation --seeds 0-4; done
python experiments/collect_results.py --experiment dias-names ablation-no-slot-weights ablation-no-precondition-loss ablation-no-effect-loss ablation-no-aux-loss ablation-no-pcgrad ablation-no-random-embeddings --seeds 0-4 --markdown experiments/results/ablations.md
```

`--seeds 0-4` compares the ablations with the same seeds of `dias-names`.

### α sweep

`--sweep aux-loss-alpha` runs every domain at α = 0.01, 0.03, 0.1, 0.3 and 1.0, each value as its own experiment `alpha-aux<α>`. The table marks each domain's configured α in bold.

```bash
python experiments/launch.py --experiment alpha --sweep aux-loss-alpha --seeds 0-4
python experiments/collect_results.py --experiment alpha --sweep aux-loss-alpha --markdown experiments/results/alpha-sweep.md
```

### Number of slots

`dias-names` runs M = 5 slots; the sweep adds M = 7 and 9 as `dias-names-slots7` and `dias-names-slots9`, and the table reads M = 5 from `dias-names`.

```bash
python experiments/launch.py --experiment dias-names --sweep num-slots --sweep-values 7 9 --seeds 0-4
python experiments/collect_results.py --experiment dias-names --sweep num-slots --seeds 0-4 --markdown experiments/results/slots.md
```

### τ schedule

τ = 0.1^(t/T) after t steps interpolates the precondition product between the geometric mean (τ = 1) and the plain product (τ = 0). `dias-names` runs T = 500; the sweep adds T = 0 (the plain product from the start), 100, 1000 and 5000 as `dias-names-tau<T>`.

```bash
python experiments/launch.py --experiment dias-names --sweep tau --sweep-values 0 100 1000 5000 --seeds 0-4
python experiments/collect_results.py --experiment dias-names --sweep tau --seeds 0-4 --markdown experiments/results/tau.md
```

### Threshold sensitivity

Re-validates seeds 0 to 4 of the finished `dias-names` runs at thresholds 0.1, 0.25, 0.5, 0.75 and 0.9, each into `dias-names-thr<T>`, without retraining:

```bash
python experiments/launch_threshold_validation.py --experiment dias-names --seeds 0-4
python experiments/collect_results.py --experiment dias-names --threshold-sweep --markdown experiments/results/threshold.md
```

### Instance-size scaling

See [scaling/README.md](scaling/README.md).

## Collecting results

| column | meaning |
|---|---|
| `P` | mean precision over seeds |
| `R` | mean recall over seeds |
| `N_S` | seeds that are sound (precision 1) |
| `N_C` | seeds that are complete (recall 1) |
| `N_SC` | seeds that are both |
| `n` | seeds scored |

`P` and `R` are shown to two decimals, and never as 1.00 unless every seed reaches 1. Precision and recall are micro-averaged per action: a successor reached by the wrong action counts against both actions. A seed whose validation did not score every problem, or ran out of memory, is left out, or with `--include-partial` counted over the problems it scored; the tables end with a list of such seeds per cell either way. A sweep value without a sub-experiment of its own is read from the base experiment's runs where they trained at that value. `--seeds` restricts every experiment to those seeds, and `--source wandb` reads W&B instead of `logs/`.

## Narrowing and re-running

```bash
python experiments/launch.py --experiment dias-names --domains satellite hanoi --seeds 0-2
python experiments/launch.py --experiment dias-names --domains gripper --seeds 3 7-8
python experiments/launch.py --experiment dias-names --domains visitall --validate-only
python experiments/launch.py --experiment dias-names --domains blocks-3 --seeds 3 --validate-only --resume
python experiments/launch.py --experiment dias-names --dry-run
python experiments/launch.py --experiment smoke --backend local --domains gripper --num-runs 1 --no-wandb
```

Re-running seeds into the same experiment replaces their results. `--validate-only` validates runs that finished training; `--resume` continues a validation from the problems and states it already scored, so a killed one loses at most the state in progress. `--dry-run` prints the sbatch commands, and `--backend local` runs here instead of on Slurm. `launch_threshold_validation.py` and `scaling/launch_scaling.py` take `--domains`, `--seeds`, `--num-runs`, `--log-dir`, `--backend` and `--dry-run` with the same meaning, and the first also `--resume`.

## Validating one model

```bash
python experiments/validate_checkpoint.py --validator dias --model-id <run-id> --problem pddl/satellite/test-1.pddl
python experiments/validate_checkpoint.py --validator mlp --model-id <run-id> --problem pddl/satellite/test-1.pddl
python experiments/validate_checkpoint.py --validator l1 --original-domain pddl/satellite/domain.pddl --learned-domain experiments/l1_pddl/satellite.pddl --problem pddl/satellite/test-1.pddl
```

It scores as the launches do, with the settings in `experiments/configs/validation.yaml`. `--validator l1` takes any learned PDDL domain.

## Reproducibility

Training repeats exactly from `(domain, seed)`. Validation draws its states with a fresh seed per run and records it as `sample_seed` in the result JSON; setting that seed in `experiments/configs/validation.yaml`, or passing it to `validate_checkpoint.py --sample-seed`, scores the same states again.

## Running on your own cluster

`experiments/slurm/train.sh`, `experiments/slurm/validate.sh` and `experiments/scaling/slurm/benchmark.sh` each begin with a `── site configuration ──` block holding everything specific to one cluster, including `DIAS_ENV_SETUP`, the snippet that puts a Python with `dias` on `PATH`. Edit that block, or export `DIAS_ENV_SETUP` before launching.

Per-domain resources are in `experiments/configs/slurm-resources.yaml` (training), `slurm-resources-validation.yaml` (validation) and `slurm-resources-validation-mlp.yaml` (MLP-effects validation); `--slurm-resources` and `--slurm-resources-validation` replace them. Training requests GPU shards (`gres: shard:N`); use `gpu:1` where your cluster defines no shards. Validation scores states in one worker per allocated CPU, each limited to an equal share of the job's `mem`; a state that does not fit even alone ends the seed as `out_of_memory`, which the tables leave out. Every job also ends itself once its RAM and swap exceed its `mem`.

## Learned precondition categories

```bash
python experiments/precondition_categories.py --experiment dias-names --markdown experiments/results/precondition-categories.md
```

Sorts every precondition atom, learned and ground truth, into equalities, unary and binary static literals, and positive and negative fluent literals, per domain.
