# Experiments

Each experiment of the paper is one or more `launch.py` commands and one command that writes its table to `experiments/results/`. A launch submits, per domain, a Slurm array training one run per seed (seeds 0 to 9 unless `--seeds` says otherwise) and an array validating each run once it has trained. Runs are written to `logs/dias/<experiment>-<domain>-s<seed>/`, results to `logs/validation/<experiment>/`.

## Setup

The Slurm jobs import `safe_slurm` (`pip install safe_slurm`), which keeps a job within its GPU shards and memory. Adapt the `site configuration` block at the top of `slurm/train.sh`, `slurm/validate.sh` and `scaling/slurm/benchmark.sh` to your cluster, and the resources in `configs/slurm.yaml`, which are sized on ours. `--backend local` runs a launch here instead, one run after another.

## Paper

### Table 1 and Table 8: main results and L1

```bash
python experiments/launch.py --experiment dias-strips --labels strips
python experiments/launch.py --experiment dias-strips-plus --labels strips-plus
python experiments/launch.py --experiment dias-names
python experiments/launch.py --experiment l1 --experiment-type l1
python experiments/collect_results.py --experiment dias-strips dias-strips-plus dias-names l1 --markdown experiments/results/main.md
```

`l1` validates the domains L1 learned, in `l1_pddl/`.

### Table 2: noise

```bash
python experiments/launch.py --experiment noise --experiment-type noise --sweep noise --seeds 0-4
python experiments/collect_results.py --experiment noise --sweep noise --markdown experiments/results/noise.md
```

### Table 3: MLP effects

```bash
python experiments/launch.py --experiment mlp-names --experiment-type mlp
python experiments/collect_results.py --experiment mlp-names --markdown experiments/results/mlp.md
```

### Tables 6 and 7, Section 5.2: learned schemas

```bash
python experiments/schema_sizes.py --experiment dias-names --markdown experiments/results/schema-sizes.md
python experiments/precondition_categories.py --experiment dias-names --markdown experiments/results/precondition-categories.md
python experiments/noop_share.py --experiment dias-names --markdown experiments/results/noop-share.md
```

### Table 9: learning curves

```bash
python experiments/launch.py --experiment curve --sweep dataset-size --domains blocks-3 delivery gripper hanoi logistics --seeds 0-2
python experiments/launch.py --experiment curve-f2 --experiment-type noise --sweep dataset-size --set data.init_args.noise.added_atoms=0.5 --set data.init_args.noise.removed_atoms=0.5 --domains blocks-3 delivery gripper hanoi logistics --seeds 0-2
python experiments/collect_results.py --experiment curve --sweep dataset-size --markdown experiments/results/curve.md
python experiments/collect_results.py --experiment curve-f2 --sweep dataset-size --markdown experiments/results/curve-f2.md
```

### Table 10: ablations

```bash
for ablation in no-slot-weights no-precondition-loss no-effect-loss no-aux-loss no-pcgrad no-random-features; do python experiments/launch.py --experiment ablation-$ablation --ablation $ablation --seeds 0-4; done
python experiments/collect_results.py --experiment dias-names ablation-no-slot-weights ablation-no-precondition-loss ablation-no-effect-loss ablation-no-aux-loss ablation-no-pcgrad ablation-no-random-features --seeds 0-4 --markdown experiments/results/ablations.md
```

### Table 11: sensitivity

α, the derivation threshold (validating the `dias-names` runs again), T and M; the `dias-names` runs stand for T = 500 and M = 5.

```bash
python experiments/launch.py --experiment alpha --sweep aux-loss-alpha --seeds 0-4
python experiments/launch.py --experiment dias-names --sweep threshold --seeds 0-4
python experiments/launch.py --experiment dias-names --sweep tau --sweep-values 0 100 1000 5000 --seeds 0-4
python experiments/launch.py --experiment dias-names --sweep num-slots --sweep-values 7 9 --seeds 0-4
python experiments/collect_results.py --experiment alpha --sweep aux-loss-alpha --markdown experiments/results/alpha.md
python experiments/collect_results.py --experiment dias-names --sweep threshold --seeds 0-4 --markdown experiments/results/threshold.md
python experiments/collect_results.py --experiment dias-names --sweep tau --seeds 0-4 --markdown experiments/results/tau.md
python experiments/collect_results.py --experiment dias-names --sweep num-slots --seeds 0-4 --markdown experiments/results/slots.md
```

### Table 12: scaling

See [scaling/README.md](scaling/README.md).

## Options

`--domains` and `--seeds` (e.g. `--seeds 3 7-9`) narrow a launch, `--dry-run` prints its commands, `--no-wandb` logs nothing to W&B, `--validate-only` validates runs that finished training, and `--resume` continues validations from what they scored. A seed's validation that does not fit in its job's memory on a problem is counted over the problems it scored; `collect_results.py --exclude-partial` leaves it out.

Validation draws its test states with a seed it records in the result JSON. `validate_checkpoint.py` scores one model on one problem, and `--sample-seed` draws the same states again:

```bash
python experiments/validate_checkpoint.py --validator dias --model-id dias-names-satellite-s0 --problem pddl/satellite/test-1.pddl
python experiments/validate_checkpoint.py --validator l1 --original-domain pddl/satellite/domain.pddl --learned-domain experiments/l1_pddl/satellite.pddl --problem pddl/satellite/test-1.pddl
```
