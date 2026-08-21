# Running the paper's experiments

Everything beyond the [README](../README.md)'s quickstart: the full grids, the
ablations, the sweeps, and how results are aggregated into the paper's tables.

All of it goes through **`scripts/launch.py`**. Per selected domain it submits a
Slurm **training array** (one task per seed) and a **validation array** that
depends on it (`--dependency=aftercorr`, so validation task *i* runs after
training task *i*). `--backend local` runs the same steps here, sequentially.

Runs are named deterministically, `<experiment>-<domain>-s<seed>`, so the two
arrays need no id handoff: training writes
`logs/dias/<run_id>/checkpoints/best-*.ckpt` and validation recomputes the same
id. Results land in `logs/validation/<experiment>/<domain>/seed<seed>.json`.

Every run of one launch shares its `--experiment` name, which is at once the W&B
group and the results directory.

## Configuration chain

Each run composes, in order:

1. `configs/main.yaml` — the base training config
2. `pddl/<domain>/config.yaml` — that domain's tuned sampling and `aux_loss_weight`
3. an experiment overlay, if any — `configs/experiments/{mlp,noise}.yaml`
4. an ablation overlay, if any — `configs/experiments/ablation-*.yaml`
5. the launcher's `--set key=value` overrides, and any sweep override, last

Later layers win, which is why a sweep value beats a domain's tuned one. Global
validation settings — how many problems, how many states sampled from each, the
seed list — live in `configs/validation.yaml`.

## The main table

```bash
python scripts/launch.py --experiment dias-full                      # DIAS
python scripts/launch.py --experiment mlp --experiment-type mlp      # MLP baseline
python scripts/launch.py --experiment noise --experiment-type noise  # noisy input
```

`--experiment-type` picks the training overlay, the validator, and the
schema-derivation threshold: 0.5, except 0.75 for `noise` as in the paper
(`--validation-threshold T` overrides). Defaults are all 13 domains × the 10
seeds in `configs/validation.yaml`.

Useful narrowings:

```bash
# Two domains, three seeds
python scripts/launch.py --experiment dias-full --domains satellite hanoi --num-runs 3

# Re-run only the seeds that died, into the same experiment
python scripts/launch.py --experiment dias-full --domains gripper --seeds 3 7-8

# See exactly what would be submitted
python scripts/launch.py --domains gripper --num-runs 3 --dry-run

# Here instead of on Slurm, no W&B — the smoke test
python scripts/launch.py --backend local --domains gripper --num-runs 1 \
    --experiment smoke --no-wandb
```

Re-running a seed under the same `--experiment` reuses its run id and therefore
its checkpoint directory. Lightning does not delete the failed attempt's file —
it side-steps the clash with a `-v1` suffix — so validation resolves the *most
recently written* `best-epoch*.ckpt`.

### Noise levels

`--experiment-type noise` brings the paper's noise setup (Appendix A.4.2):
expected-count input noise, per-action minima floored at 200/100, batch size 50,
a floor of 0.1 on τ, and validation at threshold 0.75. Override the atom budgets
to move along the noise axis:

```bash
python scripts/launch.py --experiment-type noise --experiment noise-a2r2 \
    --set data.init_args.noise.added_atoms=2 \
    --set data.init_args.noise.removed_atoms=2
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

`--ablation` composes one `configs/experiments/ablation-*.yaml` on top of the
experiment type. It is an orthogonal axis and always uses the DIAS validator.

| `--ablation` | drops |
|---|---|
| `no-slot-weights` | the learned slot activation; all M slots stay active |
| `no-precondition-loss` | L_pre, the restrictive-precondition term |
| `no-effect-loss` | L_eff, the minimal-effects term |
| `no-aux-loss` | both auxiliary terms (α = 0) |
| `no-pcgrad` | the gradient projection, for a plain weighted sum |
| `no-random-embeddings` | the random initial node features |

```bash
python scripts/launch.py --ablation no-pcgrad --experiment abl-no-pcgrad --num-runs 3
```

Collect a baseline and its ablations into one document — a cross-experiment
summary over the pooled seeds, then one table per experiment:

```bash
python scripts/collect_results.py --experiment dias-full abl-no-pcgrad \
    abl-no-effect-loss --markdown logs/ablations.md
```

## Hyperparameter sweeps

`--sweep AXIS` submits the whole domain × seed grid once per value of that axis,
each into its own sub-experiment `<experiment>-<tag><value>`, with the value
applied last so it beats the per-domain config.

| axis | parameter | tag | grid | default |
|---|---|---|---|---|
| `aux-loss-weight` | `model.init_args.aux_loss_weight` | `aux` | 0.01, 0.03, 0.1, 0.3, 1.0 | tuned per domain |
| `num-slots` | `model.init_args.num_slots` | `slots` | 5, 7, 9 | 5 |
| `gmean-temperature` | the τ schedule's `every` | `gmeanT` | 0, 100, 500, 1000, 5000 | 500 |
| `dataset-size` | transitions per action | `n` | 20 … 2000 | tuned per domain |

```bash
python scripts/launch.py --experiment aux-sweep --sweep aux-loss-weight --num-runs 3
python scripts/collect_results.py --experiment aux-sweep --sweep aux-loss-weight \
    --markdown logs/aux-sweep.md
```

`--reuse-default EXPERIMENT` skips the point that equals the axis default and
maps that column onto an existing experiment, so this submits only `slots7` and
`slots9`:

```bash
python scripts/launch.py --experiment dias --sweep num-slots \
    --reuse-default dias-full --num-runs 3
python scripts/collect_results.py --experiment dias --sweep num-slots \
    --reuse-default dias-full --markdown logs/num-slots-sweep.md
```

It is refused when the axis has no reusable default, when `configs/main.yaml` no
longer holds that default, or when a selected domain config overrides the swept
parameter — every case where the reused runs did not actually sit at the
default.

`gmean-temperature = 0` means τ sits at its floor from step 0, the `every → 0`
limit of the decay. With the default floor of 0 that is the *exact* precondition
product rather than a relaxation.

### Dataset size

The one axis that moves several parameters at once. For value N it pins
`min_per_label_train = max_per_label_train = N` and both `*_valid` to N/2, so
the sampled pool is *exactly* `num_actions × 1.5N` transitions rather than a
lower bound with a loose cap. It also pins the `*_floor` knobs to the same
values, which is what stops the noise overlay's 200/100 floors from silently
clamping the 20/50/100 points back up.

```bash
# Noiseless
python scripts/launch.py --experiment datasize-clean --sweep dataset-size \
    --domains blocks-3 delivery gripper hanoi logistics miconic --num-runs 3

# 2 atom flips per transition (0.5 removed + 0.5 added per state)
python scripts/launch.py --experiment datasize-noise --sweep dataset-size \
    --experiment-type noise --num-runs 3 \
    --domains blocks-3 delivery gripper hanoi logistics miconic \
    --set data.init_args.noise.removed_atoms=0.5 \
    --set data.init_args.noise.added_atoms=0.5

python scripts/collect_results.py --experiment datasize-clean \
    --sweep dataset-size --markdown logs/datasize-clean.md
```

The two launches therefore see identical dataset sizes and differ only in the
noise.

## Threshold sensitivity

Reading a schema off a checkpoint binarizes the slot activations and the
add/delete/positive/negative tensors at one threshold (0.5 in the paper).
`scripts/launch_threshold_validation.py` re-runs *only* the validation step of a
finished experiment at other thresholds, on the very same checkpoints — no
retraining. Each threshold writes to its own
`logs/validation/<experiment>-thr<T>/`, so the original results are untouched.

```bash
python scripts/launch_threshold_validation.py --experiment dias-full --num-runs 3
python scripts/collect_results.py --experiment dias-full --threshold-sweep \
    --markdown logs/threshold-sweep.md
```

The default grid is 0.1 / 0.25 / 0.5 / 0.75 / 0.9. 0.5 is re-run rather than
reused, so every column comes from the same code path and the same state
sampling. W&B logging is forced off here: `validate_runs` mirrors metrics by
*resuming* the training run, so a second threshold would overwrite the first's
summary. The JSONs are the source of truth.

## Collecting results

```bash
python scripts/collect_results.py --experiment dias-full                  # local JSON
python scripts/collect_results.py --experiment dias-full --source wandb
```

Per domain, aggregated over seeds:

| column | meaning |
|---|---|
| `P` | mean precision over seeds |
| `R` | mean recall over seeds |
| `N_S` | seeds that are **sound** (precision 1) |
| `N_C` | seeds that are **complete** (recall 1) |
| `N_SC` | seeds that are both |
| `n` | seeds aggregated |

Precision and recall are **micro-averaged**: each validator returns pooled
`tp`/`fp`/`fn`, which are summed across a seed's problems rather than averaging
per-problem rates.

## Validating one model

For a single checkpoint against a single problem — debugging, or a one-off
comparison — use `scripts/validate_checkpoint.py`, which takes all three
validators:

```bash
python scripts/validate_checkpoint.py --validator dias \
    --model-id <run-id> --problem pddl/satellite/test-1.pddl

python scripts/validate_checkpoint.py --validator mlp \
    --model-id <run-id> --problem pddl/satellite/test-1.pddl

# L1 compares a learned PDDL domain produced outside this repository
python scripts/validate_checkpoint.py --validator l1 \
    --original-domain pddl/satellite/domain.pddl \
    --learned-domain  learned/satellite.pddl \
    --problem         pddl/satellite/test-1.pddl
```

Its sampling defaults come from `configs/validation.yaml`, so it draws the same
way the sweep does. `--sample-seed` fixes the draw; unset, the states are drawn
fresh and the numbers are one sample, not reproducible.
