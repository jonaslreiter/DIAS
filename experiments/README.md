# Running the paper's experiments

This folder holds everything beyond the [README](../README.md)'s quickstart: the full grids, the ablations, the sweeps, the threshold study, the instance-size scaling study, and the aggregation into the paper's tables. Every command below is run from the repository root.

## What is in here

```
experiments/
├── launch.py                       the sweep launcher, start here
├── validate_runs.py                one (domain, seed); the arrays call it
├── validate_checkpoint.py          one model by hand; the only route to L1
├── collect_results.py              per-run JSON or W&B -> the paper's tables
├── precondition_categories.py      what the learned preconditions are made of
├── launch_threshold_validation.py  re-validate at other thresholds
├── pipeline.py                     shared paths, sbatch flags, validators
├── configs/                        validation.yaml, Slurm resources
├── slurm/                          train.sh, validate.sh (the array jobs)
├── scaling/                        the instance-size study (own README)
└── results/                        collected tables land here (gitignored)
```

Runs produce `logs/dias/` (checkpoints), `logs/validation/` (per-run result JSON) and `logs/slurm/` (job output).

## How a launch works

Per selected domain, `launch.py` submits a Slurm **training array** (one task per seed) and a **validation array** that depends on it (`--dependency=aftercorr`, so validation task *i* runs after training task *i*). `--backend local` runs the same steps here, sequentially.

Runs are named `<experiment>-<domain>-s<seed>`, so the two arrays need no id handoff: training writes `logs/dias/<run_id>/checkpoints/`, validation recomputes the same id, and results land in `logs/validation/<experiment>/<domain>/seed<seed>.json`. The `--experiment` name is at once the W&B group and the results directory.

Each run composes, in order:

1. `configs/main.yaml`: the base training config, which names no domain
2. `pddl/<domain>/config.yaml`: that domain's files, plus its tuned sampling and `aux_loss_alpha`
3. an experiment overlay, if any: `configs/experiment-{mlp,noise}.yaml`
4. an ablation overlay, if any: `configs/ablation-*.yaml`
5. the launcher's `--set key=value` overrides, and any sweep override, last

Later layers win, which is why a sweep value beats a domain's tuned one. Validation settings (how many problems, how many states sampled from each, the seed list) live in `experiments/configs/validation.yaml`.

## The main table

```bash
python experiments/launch.py --experiment dias-full                      # DIAS
python experiments/launch.py --experiment mlp --experiment-type mlp      # MLP effects
python experiments/launch.py --experiment noise --experiment-type noise  # noisy input
```

`--experiment-type` picks the training overlay, the validator, and the schema-derivation threshold: 0.5, except 0.75 for `noise` as in the paper (`--validation-threshold T` overrides). Defaults are all 13 domains × the 10 seeds in `experiments/configs/validation.yaml`.

Two defaults differ from how the paper's numbers were produced: the model gets no `=` relation (`data.init_args.include_equality`, since injective bindings make it constant), and validation requires a learned schema's parameters to be distinct (`injective` in `experiments/configs/validation.yaml`). The second costs a correct schema nothing, since no domain here has an action applicable to one object twice. For the paper's setting, add `--set data.init_args.include_equality=true` and set `injective: false`.

`--validate-only` submits the validation array alone, for when training survived but validation did not. Runs that stopped short of `max_steps` are refused before anything is submitted, unless `--allow-unfinished`.

The MLP-effects validator is far the most expensive one, and its cost follows how permissive the learned preconditions are. It grounds states across the CPUs the job was allocated and predicts on the GPU; memory bounds the worker count, since a grounding worker holds its ground actions until it is replaced after `--states-per-worker` states (over 20 GB on driverlog test-3 for a bad schema). `--resume` continues a seed from the problems and states an earlier run on the same checkpoint scored.

Useful narrowings:

```bash
# Two domains, three seeds
python experiments/launch.py --experiment dias-full --domains satellite hanoi --num-runs 3

# Re-run only the seeds that died, into the same experiment
python experiments/launch.py --experiment dias-full --domains gripper --seeds 3 7-8

# Re-run only the validation step of an already trained domain
python experiments/launch.py --experiment dias-full --domains visitall --validate-only

# See exactly what would be submitted
python experiments/launch.py --domains gripper --num-runs 3 --dry-run

# Here instead of on Slurm, no W&B, the smoke test
python experiments/launch.py --backend local --domains gripper --num-runs 1 --experiment smoke --no-wandb
```

### Noise levels

`--experiment-type noise` brings the paper's noise setup (Appendix A.4.2): expected-count input noise, per-action minima floored at 200/100, batch size 50, a floor of 0.1 on τ, and validation at threshold 0.75. The atom budgets are per state and both states of a transition are corrupted, so a transition sees `2 × (removed_atoms + added_atoms)` flips in expectation: the overlay's 1/1 is the paper's 4 flips, and Table 2's 1, 2, 4 and 8 flips are budgets of 0.25, 0.5, 1 and 2 each. Override the budgets to move along the noise axis (this one is 8 flips):

```bash
python experiments/launch.py --experiment-type noise --experiment noise-a2r2 --set data.init_args.noise.added_atoms=2 --set data.init_args.noise.removed_atoms=2
```

The two noise models are configured on `data.init_args.noise`; leaving it unset is the clean setup.

```yaml
# Bernoulli edge flips, at a rate per cell
noise:
  class_path: dias.data.noise.Ratio
  init_args: { remove: 0.1, add: 0.1 }

# a Poisson flip budget per adjacency tensor, shared by static and fluent relations as in the paper
noise:
  class_path: dias.data.noise.ExpectedCount
  init_args: { removed_atoms: 2.0, added_atoms: 2.0 }
```

`fluents_only: true` keeps the static relations clean and spends the whole budget on the fluents, which at the same budget is several times the noise on what the effects have to explain (statics take 47% to 96% of the flips on the paper's domains), so its results are not comparable to the paper's columns.

## Ablations

`--ablation` composes one `configs/ablation-*.yaml` on top of the experiment type. It is an orthogonal axis and always uses the DIAS validator.

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

Collecting a baseline and its ablations together gives a cross-experiment summary over the pooled seeds, then one table per experiment:

```bash
python experiments/collect_results.py --experiment dias-full abl-no-pcgrad abl-no-effect-loss --markdown experiments/results/ablations.md
```

## Hyperparameter sweeps

`--sweep AXIS` submits the whole domain × seed grid once per value of that axis, each into its own sub-experiment `<experiment>-<tag><value>`, with the value applied last so it beats the per-domain config.

| axis | parameter | tag | grid |
|---|---|---|---|
| `aux-loss-alpha` | `model.init_args.aux_loss_alpha` | `aux` | 0.01, 0.03, 0.1, 0.3, 1.0 |
| `num-slots` | `model.init_args.num_slots` | `slots` | 5, 7, 9 |
| `tau` | the τ schedule's `every` | `tau` | 0, 100, 500, 1000, 5000 |
| `dataset-size` | transitions per action | `n` | 20 … 2000 |

```bash
python experiments/launch.py --experiment aux-sweep --sweep aux-loss-alpha --num-runs 3
python experiments/collect_results.py --experiment aux-sweep --sweep aux-loss-alpha --markdown experiments/results/aux-sweep.md
```

`tau = 0` means τ sits at its floor from step 0, which with the default floor of 0 is the *exact* precondition product rather than a relaxation.

The `dataset-size` axis moves several parameters at once. For value N it pins `min_per_action_train = max_per_action_train = N` and both `*_valid` to N/2, so the sampled pool is *exactly* `num_actions × 1.5N` transitions, and it pins the `*_floor` knobs to the same values, so the noise overlay's 200/100 floors cannot clamp the smallest points back up. A clean and a noisy launch therefore see identical dataset sizes:

```bash
python experiments/launch.py --experiment datasize-clean --sweep dataset-size --domains blocks-3 delivery gripper hanoi logistics miconic --num-runs 3

python experiments/launch.py --experiment datasize-noise --sweep dataset-size --experiment-type noise --num-runs 3 --domains blocks-3 delivery gripper hanoi logistics miconic --set data.init_args.noise.removed_atoms=0.5 --set data.init_args.noise.added_atoms=0.5
```

## Threshold sensitivity

Reading a schema off a checkpoint binarizes the slot activations and the add/delete/positive/negative tensors at one threshold (0.5 in the paper). `launch_threshold_validation.py` re-runs *only* the validation step of a finished experiment at other thresholds, on the same checkpoints, with no retraining. Each threshold writes to its own `logs/validation/<experiment>-thr<T>/`, so the original results are untouched.

```bash
python experiments/launch_threshold_validation.py --experiment dias-full --num-runs 3
python experiments/collect_results.py --experiment dias-full --threshold-sweep --markdown experiments/results/threshold-sweep.md
```

The default grid is 0.1 / 0.25 / 0.5 / 0.75 / 0.9, with 0.5 re-run rather than reused so that every column comes from the same code path. W&B logging is off here, since mirroring metrics resumes the training run and a second threshold would overwrite the first's summary; the JSONs are the source of truth.

## Collecting results

```bash
python experiments/collect_results.py --experiment dias-full                  # local JSON
python experiments/collect_results.py --experiment dias-full --source wandb
```

`--markdown <path>` also writes the tables out, by convention to `experiments/results/<name>.md` (gitignored, created on first write). Per domain, aggregated over seeds:

A seed whose `status` is not `ok` is missing a problem's score, so it is skipped with a warning; re-run it.

| column | meaning |
|---|---|
| `P` | mean precision over seeds |
| `R` | mean recall over seeds |
| `N_S` | seeds that are **sound** (precision 1) |
| `N_C` | seeds that are **complete** (recall 1) |
| `N_SC` | seeds that are both |
| `n` | seeds aggregated |

Precision and recall are micro-averaged and attributed to actions: a successor counts as a true positive only when the learned schema of the *same* action reaches it, so reaching the right successor under the wrong operator is an error on both sides. `accuracy` reports the weaker state-level question, whether the two successor sets agreed exactly.

## What the learned preconditions are made of

`collect_results.py` scores the learned schemas; this reads what is inside them. Per domain it sorts every precondition atom, learned and ground truth, into classes:

```bash
python experiments/precondition_categories.py --experiment dias-full
python experiments/precondition_categories.py --experiment dias-full --markdown experiments/results/precondition-categories-dias-full.md
```

| class | meaning |
|---|---|
| `eq` | equalities between action parameters: the built-in `=` under `:equality`, or the reflexive predicate a domain declares in its place (`eq`) |
| `stat1` | unary literals over static predicates, either polarity |
| `stat2` | binary literals over static predicates, either polarity |
| `pos` | positive fluent literals |
| `neg` | negative fluent literals |

A static predicate is one that occurs in no action's effect, which is also what pymimir reports and so the flag DIAS is given. The paper quotes a four-way split with no class for `stat2`, folding those literals into `pos` and `neg` by polarity; both groupings are printed, since the four-way one is what the text says and the five-way one is what the class names mean.

Learned columns are per-seed means and ground-truth columns count each action's schema once, which is the unit the two are comparable on. Atoms are counted after de-duplication, as distinct lifted atoms, with `(not (p ...))` as one atom, and type-derived unary predicates left out of the ground-truth counts, since a `?t - truck` in `:parameters` is not a precondition. DIAS does get a channel for those (pymimir materialises them, and `include_statics` defaults to true), which is what the `stat1` column measures.

Schemas are read through `dias.validation.dias.read_learned_schemas`, the same function `DiasValidator` uses, so the atoms counted here are the ones a validator would score, less the parameter distinctness a validator writes in (counted, it would read as learned equalities). `--threshold` defaults to 0.5, what `launch.py` uses for `--experiment-type dias`; a noise experiment wants `--threshold 0.75`.

## Instance-size scaling

How iterations per second and peak GPU memory grow as the *instance* grows is a separate one-off study, with its own launcher and its own [README](scaling/README.md). It produces timings rather than schemas, so `launch.py` knows nothing about it, and its instances are committed, so running it needs neither the upstream PDDL generators nor their toolchain.

## Validating one model

For a single checkpoint against a single problem:

```bash
python experiments/validate_checkpoint.py --validator dias --model-id <run-id> --problem pddl/satellite/test-1.pddl

python experiments/validate_checkpoint.py --validator mlp --model-id <run-id> --problem pddl/satellite/test-1.pddl

# L1 compares a learned PDDL domain produced outside this repository, such as the one committed per domain
python experiments/validate_checkpoint.py --validator l1 --original-domain pddl/satellite/domain.pddl --learned-domain pddl/satellite/l1.pddl --problem pddl/satellite/test-1.pddl
```

Its sampling defaults come from `experiments/configs/validation.yaml`, so it draws the same way the sweep does, and all three validators return the same payload. So does `injective`: a checkpoint's schema is scored with its parameters required distinct, as the model binds them (`--no-injective` scores it as learned, which is how the paper's numbers were produced). L1 is reachable only from here: it scores a domain another system produced, so there is no checkpoint and no `(domain, seed)` pair for `launch.py` to iterate over.

Each `pddl/<domain>/l1.pddl` holds the action schemas L1 (Balyo et al., 2024) learned for the paper's comparison (its Table 17), from one random walk per domain. They are L1's output with two things respelled to match `domain.pddl`, which scores identically: equality, which L1 writes as the static `eq` or `equal` predicate its input domain had, is `=`, and the types are the domain's own (L1 types every parameter of an untyped domain `untyped`). The validator reads either spelling: a binary predicate named `eq` or `equal` that no action changes is `=` in both domains, and a learned parameter type the original declares neither as a type nor as a unary predicate is refused, since no problem would assert it. L1 learned from `train.pddl` in 11 domains, but in hanoi from a 7 disc instance and in spanner from a smaller one (6 spanners, 4 nuts, 8 locations), which were the training instances when its walks were sampled. Neither the walks nor those two instances are in the repository.

TODO: check whether DIAS learns hanoi and spanner from L1's instances as well as from `train.pddl`.

## Reproducibility

Validation samples states from each held-out problem, and that draw is not fixed to a constant: `sample_seed` in `experiments/configs/validation.yaml` is unset, so every run scores on its own sample, as it also trains on its own. Each result JSON records the seed its states were drawn with, so passing that seed back (there, or as `validate_checkpoint.py --sample-seed`) reproduces exactly those states.

Training is reproducible from `(domain, seed)` only if you ask for it: float addition is not associative, so a parallel reduction depends on how the work was split, and the encoder's scatter-add is nondeterministic on GPU by default. `--trainer.deterministic=true` (Lightning's own flag: it selects deterministic kernels and sets `CUBLAS_WORKSPACE_CONFIG`) makes a run bit-identical on re-run, at no measurable throughput cost here. It is off by default because it raises on any operation without a deterministic implementation, which is a poor way for a 130-run sweep to fail; `--trainer.deterministic=warn` is the middle setting.

## Running on your own cluster

`experiments/slurm/train.sh`, `experiments/slurm/validate.sh` and `experiments/scaling/slurm/benchmark.sh` each begin with a block marked `── site configuration ──`, holding every line specific to one cluster: the partition fallbacks, and `DIAS_ENV_SETUP`, the shell snippet that puts a Python with `dias` importable on `PATH`. Edit that block, or export `DIAS_ENV_SETUP` before submitting. Nothing below the end marker is site-specific.

Per-domain resources come from `experiments/configs/slurm-resources.yaml` (training) and `experiments/configs/slurm-resources-validation.yaml` (validation, CPU-only). The training defaults request GPU shards (`gres: shard:N`), which is not a standard Slurm GRES; change it to `gpu:1` if your site does not define it. `train.sh` caps PyTorch's GPU allocator at the fraction of the card those shards represent (the job's shards over the shards Slurm defines per GPU, read from `scontrol show node` or from `DIAS_SHARDS_PER_GPU`), so a domain that needs more than it reserved fails with an out-of-memory error and needs a larger `gres`. This relies on the `per_process_memory_fraction` option of `PYTORCH_CUDA_ALLOC_CONF`, which older PyTorch releases reject.
