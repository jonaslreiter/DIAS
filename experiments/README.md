# Running the paper's experiments

This folder is the entry point for everything beyond the [README](../README.md)'s quickstart: the full grids, the ablations, the sweeps, the threshold study, the scaling study, and how results are aggregated into the paper's tables. Every command below is run from the repository root.

## What is in here

```
experiments/
├── launch.py                       the sweep launcher, start here
├── validate_runs.py                one (domain, seed); the arrays call it
├── validate_checkpoint.py          one model by hand; the only route to L1
├── collect_results.py              per-run JSON or W&B -> the paper's tables
├── launch_threshold_validation.py  re-validate at other thresholds
├── pipeline.py                     shared paths, sbatch flags, validators
├── configs/                        validation.yaml, Slurm resources
├── slurm/                          train.sh, validate.sh (the array jobs)
├── scaling/                        the instance-size study (own README)
└── results/                        collected tables land here (gitignored)
```

Three things this folder deliberately does *not* hold, because they are not about running a sweep:

- **`configs/main.yaml` and `configs/experiments/*.yaml`**: the Lightning configs `dias fit` reads. `launch.py` composes them into every run's command line, but paired with a `pddl/<domain>/config.yaml` they train a single domain without this folder.
- **`pddl/<domain>/`**: the 13 domains. A domain *is* its folder, so `--domains <name>` works as soon as the folder exists.
- **`logs/`**: what runs produce: `logs/dias/` (checkpoints), `logs/validation/` (per-run result JSON), `logs/slurm/` (job output).

## How a launch works

Everything goes through **`experiments/launch.py`**. Per selected domain it submits a Slurm **training array** (one task per seed) and a **validation array** that depends on it (`--dependency=aftercorr`, so validation task *i* runs after training task *i*). `--backend local` runs the same steps here, sequentially.

Runs are named deterministically, `<experiment>-<domain>-s<seed>`, so the two arrays need no id handoff: training writes `logs/dias/<run_id>/checkpoints/best-*.ckpt` and validation recomputes the same id. Results land in `logs/validation/<experiment>/<domain>/seed<seed>.json`.

Every run of one launch shares its `--experiment` name, which is at once the W&B group and the results directory.

## Configuration chain

Each run composes, in order:

1. `configs/main.yaml`: the base training config, which names no domain
2. `pddl/<domain>/config.yaml`: that domain's files, plus its tuned sampling and `aux_loss_weight`
3. an experiment overlay, if any: `configs/experiments/{mlp,noise}.yaml`
4. an ablation overlay, if any: `configs/experiments/ablation-*.yaml`
5. the launcher's `--set key=value` overrides, and any sweep override, last

Later layers win, which is why a sweep value beats a domain's tuned one, and each layer states only what it changes about the one below it: the bottom layer is the defaults in `dias.models.DIAS` and `dias.data.datamodule.PddlGraphDataModule`, which are the paper's full model with no noise (`dias fit --model.help dias.models.DIAS` and `--data.help dias.data.datamodule.PddlGraphDataModule` list them all, with a line on what each does). Only `domain_path` and `problem_path` have no default at all, so a chain that names no domain is refused rather than run on a guess. The launcher passes one seed, `--seed_everything`, which the sampling walk follows unless a config gave it a seed of its own. Global validation settings (how many problems, how many states sampled from each, the seed list) live in `experiments/configs/validation.yaml`.

## The main table

```bash
python experiments/launch.py --experiment dias-full                      # DIAS
python experiments/launch.py --experiment mlp --experiment-type mlp      # MLP baseline
python experiments/launch.py --experiment noise --experiment-type noise  # noisy input
```

`--experiment-type` picks the training overlay, the validator, and the schema-derivation threshold: 0.5, except 0.75 for `noise` as in the paper (`--validation-threshold T` overrides). Defaults are all 13 domains × the 10 seeds in `experiments/configs/validation.yaml`.

Useful narrowings:

```bash
# Two domains, three seeds
python experiments/launch.py --experiment dias-full --domains satellite hanoi --num-runs 3

# Re-run only the seeds that died, into the same experiment
python experiments/launch.py --experiment dias-full --domains gripper --seeds 3 7-8

# See exactly what would be submitted
python experiments/launch.py --domains gripper --num-runs 3 --dry-run

# Here instead of on Slurm, no W&B, the smoke test
python experiments/launch.py --backend local --domains gripper --num-runs 1 --experiment smoke --no-wandb
```

Re-running a seed under the same `--experiment` reuses its run id and therefore its checkpoint directory. Lightning does not delete the failed attempt's file; it side-steps the clash with a `-v1` suffix, so validation resolves the *most recently written* `best-epoch*.ckpt`.

### Noise levels

`--experiment-type noise` brings the paper's noise setup (Appendix A.4.2): expected-count input noise, per-action minima floored at 200/100, batch size 50, a floor of 0.1 on τ, and validation at threshold 0.75. Override the atom budgets to move along the noise axis:

```bash
python experiments/launch.py --experiment-type noise --experiment noise-a2r2 --set data.init_args.noise.added_atoms=2 --set data.init_args.noise.removed_atoms=2
```

The two noise models are configured on `data.init_args.noise`:

```yaml
# Bernoulli edge flips, at a rate per cell
noise:
  class_path: dias.data.noise.Ratio
  init_args: { remove: 0.1, add: 0.1 }

# a Poisson flip budget per adjacency tensor, statics untouched
noise:
  class_path: dias.data.noise.ExpectedCount
  init_args: { removed_atoms: 2.0, added_atoms: 2.0, fluents_only: true }
```

Leaving `noise` unset is the clean setup.

## Ablations

`--ablation` composes one `configs/experiments/ablation-*.yaml` on top of the experiment type. It is an orthogonal axis and always uses the DIAS validator.

| `--ablation` | drops |
|---|---|
| `no-slot-weights` | the learned slot activation; all M slots stay active |
| `no-precondition-loss` | L_pre, the restrictive-precondition term |
| `no-effect-loss` | L_eff, the minimal-effects term |
| `no-aux-loss` | both auxiliary terms (α = 0) |
| `no-pcgrad` | the gradient projection, for a plain weighted sum |
| `no-random-embeddings` | the random initial node features |

```bash
python experiments/launch.py --ablation no-pcgrad --experiment abl-no-pcgrad --num-runs 3
```

Collect a baseline and its ablations into one document (a cross-experiment summary over the pooled seeds, then one table per experiment):

```bash
python experiments/collect_results.py --experiment dias-full abl-no-pcgrad abl-no-effect-loss --markdown experiments/results/ablations.md
```

## Hyperparameter sweeps

`--sweep AXIS` submits the whole domain × seed grid once per value of that axis, each into its own sub-experiment `<experiment>-<tag><value>`, with the value applied last so it beats the per-domain config.

| axis | parameter | tag | grid |
|---|---|---|---|
| `aux-loss-weight` | `model.init_args.aux_loss_weight` | `aux` | 0.01, 0.03, 0.1, 0.3, 1.0 |
| `num-slots` | `model.init_args.num_slots` | `slots` | 5, 7, 9 |
| `gmean-temperature` | the τ schedule's `every` | `gmeanT` | 0, 100, 500, 1000, 5000 |
| `dataset-size` | transitions per action | `n` | 20 … 2000 |

```bash
python experiments/launch.py --experiment aux-sweep --sweep aux-loss-weight --num-runs 3
python experiments/collect_results.py --experiment aux-sweep --sweep aux-loss-weight --markdown experiments/results/aux-sweep.md
```

Every point of the grid is submitted, including one that matches what another experiment already ran: a column collected from runs launched under a different chain is a column nothing pins to this axis.

`gmean-temperature = 0` means τ sits at its floor from step 0, the `every → 0` limit of the decay. With the default floor of 0 that is the *exact* precondition product rather than a relaxation.

### Dataset size

The one axis that moves several parameters at once. For value N it pins `min_per_action_train = max_per_action_train = N` and both `*_valid` to N/2, so the sampled pool is *exactly* `num_actions × 1.5N` transitions rather than a lower bound with a loose cap. It also pins the `*_floor` knobs to the same values, which is what stops the noise overlay's 200/100 floors from silently clamping the 20/50/100 points back up.

```bash
# Noiseless
python experiments/launch.py --experiment datasize-clean --sweep dataset-size --domains blocks-3 delivery gripper hanoi logistics miconic --num-runs 3

# 2 atom flips per transition (0.5 removed + 0.5 added per state)
python experiments/launch.py --experiment datasize-noise --sweep dataset-size --experiment-type noise --num-runs 3 --domains blocks-3 delivery gripper hanoi logistics miconic --set data.init_args.noise.removed_atoms=0.5 --set data.init_args.noise.added_atoms=0.5

python experiments/collect_results.py --experiment datasize-clean --sweep dataset-size --markdown experiments/results/datasize-clean.md
```

The two launches therefore see identical dataset sizes and differ only in the noise.

## Threshold sensitivity

Reading a schema off a checkpoint binarizes the slot activations and the add/delete/positive/negative tensors at one threshold (0.5 in the paper). `experiments/launch_threshold_validation.py` re-runs *only* the validation step of a finished experiment at other thresholds, on the very same checkpoints, with no retraining. Each threshold writes to its own `logs/validation/<experiment>-thr<T>/`, so the original results are untouched.

```bash
python experiments/launch_threshold_validation.py --experiment dias-full --num-runs 3
python experiments/collect_results.py --experiment dias-full --threshold-sweep --markdown experiments/results/threshold-sweep.md
```

The default grid is 0.1 / 0.25 / 0.5 / 0.75 / 0.9. 0.5 is re-run rather than reused, so every column comes from the same code path and the same state sampling. W&B logging is forced off here: `validate_runs` mirrors metrics by *resuming* the training run, so a second threshold would overwrite the first's summary. The JSONs are the source of truth.

## Collecting results

```bash
python experiments/collect_results.py --experiment dias-full                  # local JSON
python experiments/collect_results.py --experiment dias-full --source wandb
```

`--markdown <path>` also writes the tables out. The convention here is `experiments/results/<name>.md`, beside the launchers that produced them; the folder is gitignored, since every table in it is regenerated from `logs/` by the commands here. It is created on first write, so a fresh clone needs no setup.

Per domain, aggregated over seeds:

| column | meaning |
|---|---|
| `P` | mean precision over seeds |
| `R` | mean recall over seeds |
| `N_S` | seeds that are **sound** (precision 1) |
| `N_C` | seeds that are **complete** (recall 1) |
| `N_SC` | seeds that are both |
| `n` | seeds aggregated |

Precision and recall are **micro-averaged and attributed to actions**: a successor is a true positive only when the learned schema of the same action reaches it. Reaching the right successor under the wrong operator scores as a false positive there and a false negative under the operator that should have reached it: the stricter reading, and the one every number published from this repository was computed with. Each validator returns `tp`/`fp`/`fn` as the sums of its `action_stats`, and those are summed again across a seed's problems.

The weaker state-level question (did the learned domain generate the same successor *set*, whatever produced it, which is what makes the search tree identical) is `accuracy`: the fraction of sampled states whose two sets agreed exactly.

## Validating one model

For a single checkpoint against a single problem (debugging, or a one-off comparison), use `experiments/validate_checkpoint.py`, which takes all three validators:

```bash
python experiments/validate_checkpoint.py --validator dias --model-id <run-id> --problem pddl/satellite/test-1.pddl

python experiments/validate_checkpoint.py --validator mlp --model-id <run-id> --problem pddl/satellite/test-1.pddl

# L1 compares a learned PDDL domain produced outside this repository
python experiments/validate_checkpoint.py --validator l1 --original-domain pddl/satellite/domain.pddl --learned-domain learned/satellite.pddl --problem pddl/satellite/test-1.pddl
```

Its sampling defaults come from `experiments/configs/validation.yaml`, so it draws the same way the sweep does. `--sample-seed` fixes the draw; unset, the states are drawn fresh, and either way the seed used comes back in the result, so passing it again scores the next run on exactly those states.

**Why L1 has no `--experiment-type`.** The other two validators score a model this pipeline trained, found by run id. L1 scores a learned domain *another* system produced, handed to it as a PDDL file: there is no checkpoint and no `(domain, seed)` pair for `launch.py` to iterate over, so it lives here rather than in the sweep. All three return the same result payload (per-action counts, the `tp`/`fp`/`fn` they sum to, and the rates), because all three finish through the same scorer.
