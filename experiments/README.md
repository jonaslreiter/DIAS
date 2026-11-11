# Experiments

Each experiment of the paper consists of one or more `launch.py` commands and one command that writes its table to `experiments/results/`. A launch submits two Slurm arrays per domain. The first trains one run per seed, which is every seed from 0 to 9 unless `--seeds` says otherwise. The second validates each run once it has trained. Runs are written to `logs/dias/<experiment>-<domain>-s<seed>/`, and validation results to `logs/validation/<experiment>/`.

## Setup

The Slurm jobs use [safe-slurm](https://github.com/jonaslreiter/safe-slurm), which keeps a job within its GPU shards and its memory, so install it with `pip install safe-slurm`. Adapt the `site configuration` block at the top of `slurm/train.sh`, `slurm/validate.sh` and `scaling/slurm/benchmark.sh` to your cluster. The resources in `configs/slurm.yaml` are sized on our cluster and may need changes as well. With `--backend local`, a launch runs on the current machine instead, one run after another.

## Table 1: main results

```bash
python experiments/launch.py --experiment dias-strips --labels strips
python experiments/launch.py --experiment dias-strips-plus --labels strips-plus
python experiments/launch.py --experiment dias-names
python experiments/collect_results.py --experiment dias-strips dias-strips-plus dias-names --markdown experiments/results/main.md
```

## Table 2: noise

```bash
python experiments/launch.py --experiment noise --experiment-type noise --sweep noise --seeds 0-4
python experiments/collect_results.py --experiment noise --sweep noise --markdown experiments/results/noise.md
```

## Table 3: MLP effects

```bash
python experiments/launch.py --experiment mlp-names --experiment-type mlp
python experiments/collect_results.py --experiment mlp-names --markdown experiments/results/mlp.md
```

## Tables 6 and 7 and Section 5.2: learned schemas

These commands read the `dias-names` runs of Table 1. The last one counts the sampled transitions that change nothing.

```bash
python experiments/schema_sizes.py --experiment dias-names --markdown experiments/results/schema-sizes.md
python experiments/precondition_categories.py --experiment dias-names --markdown experiments/results/precondition-categories.md
python experiments/noop_share.py --experiment dias-names --markdown experiments/results/noop-share.md
```

## Table 8: L1

The domains that L1 learned are in `l1_pddl/`. Their launch validates them and trains nothing.

```bash
python experiments/launch.py --experiment l1 --experiment-type l1
python experiments/collect_results.py --experiment l1 --markdown experiments/results/l1.md
```

## Table 9: learning curves

```bash
python experiments/launch.py --experiment curve --sweep dataset-size --domains blocks-3 delivery gripper hanoi logistics --seeds 0-2
python experiments/launch.py --experiment curve-f2 --experiment-type noise --sweep dataset-size --set data.init_args.noise.added_atoms=0.5 --set data.init_args.noise.removed_atoms=0.5 --domains blocks-3 delivery gripper hanoi logistics --seeds 0-2
python experiments/collect_results.py --experiment curve --sweep dataset-size --markdown experiments/results/curve.md
python experiments/collect_results.py --experiment curve-f2 --sweep dataset-size --markdown experiments/results/curve-f2.md
```

## Table 10: ablations

```bash
for ablation in no-slot-weights no-precondition-loss no-effect-loss no-aux-loss no-pcgrad no-random-features
do
  python experiments/launch.py --experiment ablation-$ablation --ablation $ablation --seeds 0-4
done
python experiments/collect_results.py --experiment dias-names ablation-no-slot-weights ablation-no-precondition-loss ablation-no-effect-loss ablation-no-aux-loss ablation-no-pcgrad ablation-no-random-features --seeds 0-4 --markdown experiments/results/ablations.md
```

## Table 11: sensitivity

The four sweeps vary α, the derivation threshold, T and M. The threshold sweep validates the `dias-names` runs again at each threshold. The `dias-names` runs also stand for T = 500 and M = 5.

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

## Table 12: scaling

The scaling study measures training steps per second and peak GPU memory on Blocks-3, Logistics and Delivery instances of 12 to 150 objects, which are in `scaling/instances/`. Each measurement is a training run of 20 warm-up steps and 100 measured steps on a whole GPU. The file `scaling/configs/sizes.yaml` lists the instances, the steps and the Slurm resources, and `scaling/configs/scaling.yaml` is the training config overlay. An instance that does not fit on the GPU is recorded as `oom`.

```bash
python experiments/scaling/launch_scaling.py --experiment scaling --seeds 0-4
python experiments/scaling/collect_scaling.py --experiment scaling --markdown experiments/results/scaling.md
```

## Options

The options `--domains` and `--seeds` (for example `--seeds 3 7-9`) narrow a launch, and `--dry-run` prints its commands without running them. With `--no-wandb`, nothing is logged to W&B. The option `--validate-only` validates runs that finished training, and `--resume` continues validations from what they have already scored. The scaling launcher takes `--domains`, `--seeds`, `--backend` and `--dry-run` as well. When a validation does not fit into the memory of its job on some problem, its seed is counted over the problems it scored, and `collect_results.py --exclude-partial` leaves such seeds out.

Validation draws its test states with a seed that it records in the result file. The script `validate_checkpoint.py` scores one model on one problem, and its option `--sample-seed` draws the same states again:

```bash
python experiments/validate_checkpoint.py --validator dias --model-id dias-names-satellite-s0 --problem pddl/satellite/test-1.pddl
python experiments/validate_checkpoint.py --validator l1 --original-domain pddl/satellite/domain.pddl --learned-domain experiments/l1_pddl/satellite.pddl --problem pddl/satellite/test-1.pddl
```
