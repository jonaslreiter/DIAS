# DIAS: Differentiable Induction of Action Schemas

Code for **[Differentiable Learning of Lifted Action Schemas for Classical Planning](https://arxiv.org/abs/2605.13282)**, by Jonas Reiter, Jakob Elias Gebler and Hector Geffner, arXiv:2605.13282.

DIAS learns **lifted STRIPS action schemas** (preconditions, add effects and delete effects over action *parameters*) from nothing but observed state transitions of a single planning problem. It is trained end to end by gradient descent: a relational GNN embeds the objects of a state, a Sinkhorn-style assignment binds them to the schema's parameter slots, and the learned schema is applied to predict the successor state. The learned parameters are then read off as symbolic PDDL, and evaluated on *held-out problems of different sizes* by whether the recovered domain generates the same successor states as the real one.

## What it learns

Ten thousand steps on gripper's training instance, then read the model's parameters back as PDDL. Both effect sets are exact and the operator's arity is recovered, having been given only `(state, action, successor)` triples:

```lisp
;; ground truth                        ;; recovered (seed 0, threshold 0.5)
(:action pick                          (:action learned_pick
  :parameters (?obj ?room ?gripper)      :parameters (?x0 ?x1 ?x2)
  :precondition (and                     :precondition (and
    (ball    ?obj)                         (ball    ?x0)
    (room    ?room)                        (room    ?x2)
    (gripper ?gripper)                     (gripper ?x1)
    (at      ?obj ?room)                   (at      ?x0 ?x2)
    (at-robby ?room)                       (at-robby ?x2)
    (free    ?gripper))                    (free    ?x1)
                                           ... and 35 negative literals that
                                           hold in every reachable state)
  :effect (and                           :effect (and
    (carry ?obj ?gripper)                  (carry ?x0 ?x1)
    (not (at ?obj ?room))                  (not (at ?x0 ?x2))
    (not (free ?gripper))))                (not (free ?x1))))
```

The extra negative preconditions are type-exclusion facts (`(not (room ?x0))` and such) that no reachable state violates, so they cost nothing behaviourally: this seed scores precision = recall = 1.0 on all three held-out gripper problems. Evaluation is behavioural throughout: a schema is *sound* when it generates no successor the real domain does not, and *complete* when it misses none.

## Install

DIAS is installed from a clone, not from a package index:

```bash
git clone https://github.com/jonaslreiter/DIAS.git
cd DIAS
pip install -e .
```

That gives you the library and the `dias` command. The 13 PDDL domains, the config chain and the experiment scripts are the repository itself, so everything below is run from inside the checkout.

Python 3.10+. A CUDA GPU for full-size runs; the quickstart below is CPU-only. [pymimir](https://pypi.org/project/pymimir/) does the PDDL parsing and grounding and is held to `~=0.13.62`.

## Quickstart

Train on the smallest domain, on CPU, with no Weights & Biases account. Two configs: the base training setup, and the domain to apply it to.

```bash
dias fit --config configs/main.yaml --config pddl/gripper/config.yaml --trainer.accelerator=cpu --trainer.max_steps=300 --trainer.val_check_interval=100 --trainer.logger=false --trainer.default_root_dir=logs/dias/quickstart
```

A few minutes later `logs/dias/quickstart/checkpoints/` holds the trained model. Read the schema it learned back as PDDL and score it on a held-out problem:

```bash
python experiments/validate_checkpoint.py --validator dias --model-id quickstart --log-dir logs --problem pddl/gripper/test-1.pddl
```

which prints per-action `tp`/`fp`/`fn` and the precision and recall they sum to. A successor counts only when the learned schema of *that* action reaches it, so recovering the right state under the wrong operator is an error rather than a hit. 300 steps will not recover gripper; the paper's runs take 10 000. Drop `--trainer.max_steps` and point `--trainer.accelerator` at `gpu` for a real one.

### Choosing a domain

Swap the second config. It carries both the domain's files and its tuned hyperparameters, and it is layered after the base config, so it wins:

```bash
dias fit --config configs/main.yaml --config pddl/hanoi/config.yaml
```

`configs/main.yaml` names no domain of its own, and neither does the datamodule: `domain_path` and `problem_path` have no default anywhere, so a run that says nothing is refused rather than trained on whichever domain the base config happened to carry. Every config states only what it changes about the layer below it, down to the model and datamodule signatures, whose defaults are the paper's setup. To see them all, with a line on what each does, ask the CLI: `dias fit --model.help dias.models.DIAS` and `dias fit --data.help dias.data.datamodule.PddlGraphDataModule`. Any single value is overridable on the command line (`--data.init_args.problem_path=...` to train on a different instance of the same domain, for example), because command-line arguments are applied last of all.

### Weights & Biases

W&B is the default logger, declared in `configs/main.yaml` with `entity: null`, which means whichever account you are logged into (or `$WANDB_ENTITY`). Nothing in the package names an account. To train without it, either `--trainer.logger=false`, or `WANDB_MODE=offline` to record locally and upload later. When it is on, the learned schema is logged every 20 validation epochs as interactive heatmaps: effects, preconditions, slot-to-object assignments and slot weights, with a step slider.

### Your own PDDL

Point `domain_path` / `problem_path` at any domain whose predicates are unary or binary and whose actions are STRIPS with injective parameter binding. To make it a domain like the 13 below, give it a `pddl/<name>/` folder holding `domain.pddl`, `train.pddl`, `test-1.pddl` .. `test-3.pddl` and a `config.yaml` (those first two paths, plus whatever it needs tuned). Nothing lists the domains: the folder is the domain, so `--domains <name>` works in the sweep launcher as soon as the folder exists. The test suite checks that every folder holds all six files and that its config selects its own domain.

## The 13 domains

`blocks-3`, `delivery`, `driverlog`, `gripper`, `hanoi`, `logistics`, `miconic`, `n-puzzle`, `satellite`, `sokoban`, `sokoban-pull`, `spanner`, `visitall`.

Each `pddl/<domain>/` holds `domain.pddl`, a training instance `train.pddl`, three held-out `test-*.pddl` of increasing size, the STRIPS+ argument positions in `strips+.yaml`, and the tuned `config.yaml`.

## Reproducing the paper

The main table is one command per experiment type, plus a collection step:

```bash
python experiments/launch.py --experiment dias-full                      # DIAS
python experiments/launch.py --experiment mlp --experiment-type mlp      # MLP effects
python experiments/launch.py --experiment noise --experiment-type noise  # noisy input

python experiments/collect_results.py --experiment dias-full
```

Each launch submits, per domain, a training array over the seeds plus a validation array that depends on it. **This assumes Slurm**, and the job scripts under `experiments/slurm/` are a worked example for one cluster; see "Running on your own cluster" below. `--backend local` runs the same steps sequentially on this machine instead.

The ablations, the hyperparameter and dataset-size sweeps, the threshold sensitivity study and the aggregation options are all in **[experiments/README.md](experiments/README.md)**, the folder that is the entry point for all of it. The separate instance-size scaling study has its own [README](experiments/scaling/README.md).

### What is and is not reproducible

Validation scores a learned schema on states sampled from each held-out problem, and that draw is **not** fixed to a constant: `sample_seed` in `experiments/configs/validation.yaml` is unset, so every run scores on its own sample, the way every run also trains on its own. Each result JSON records the seed its states were drawn with, so any number can be re-derived afterwards by putting that seed back into `experiments/configs/validation.yaml`.

Training itself is reproducible from `(domain, seed)` only if you ask for it. Float addition is not associative, so a parallel reduction's result depends on how the work was split, and on a GPU the encoder's scatter-add is nondeterministic by default: two runs of one seed give slightly different weights. Pass `--trainer.deterministic=true` (Lightning's own flag; it selects deterministic kernels and sets `CUBLAS_WORKSPACE_CONFIG`) and a run becomes bit-identical on re-run, at no measurable throughput cost here: 82–94 ms/step on gripper either way, which is inside the run-to-run spread. It is off by default because it raises on any operation without a deterministic implementation, which is a poor way for a 130-run sweep to fail; `--trainer.deterministic=warn` is the middle setting.

### Running on your own cluster

`experiments/slurm/train.sh`, `experiments/slurm/validate.sh` and `experiments/scaling/slurm/benchmark.sh` each begin with a block marked `── site configuration ──`. It holds every line specific to one cluster: the partition and QoS fallbacks, and `DIAS_ENV_SETUP`, the shell snippet that puts a Python with `dias` importable on `PATH`. Edit that block, or export `DIAS_ENV_SETUP` before submitting. Nothing below the end marker is site-specific.

Per-domain resources come from `experiments/configs/slurm-resources.yaml` (training) and `experiments/configs/slurm-resources-validation.yaml` (validation, CPU-only). The training defaults request GPU shards (`gres: shard:N`), which is **not** a standard Slurm GRES; change it to `gpu:1` if your site does not define it.

## Repository layout

```
src/dias/
├── main.py           LightningCLI entry point (also `dias`)
├── models/           the model: selection, preconditions, effects, encoder,
│                       objectives, PCGrad, checkpoint migration
├── data/             PDDL datamodule, transition sampling, adjacency noise
├── schema/           DomainSchema (metadata) and ActionSchema (learned output)
├── pddl/             pure PDDL text handling, untyping, the transpiler
├── validation/       one validator per model (dias, mlp_effects, l1) and the
│                       successor scoring all three share
└── reporting/        the W&B custom charts the learned schema is logged as

experiments/          everything you run to reproduce the paper, start here
├── README.md         the full experiment recipes
├── launch.py         the sweep launcher; validate_runs.py, collect_results.py,
│                       validate_checkpoint.py, pipeline.py beside it
├── configs/          validation.yaml, Slurm resources
├── slurm/            the array job scripts
└── scaling/          the one-off runtime / GPU-memory vs instance-size study

configs/              main.yaml and the `dias fit` overlays in experiments/
pddl/                 the 13 domains, one folder each
tests/
```

## Development

```bash
pip install -e . --group dev      # PEP 735 groups; needs pip >= 25.1
pre-commit install                # optional: runs the two below on a commit
pytest
black --check . && isort --check-only .
```

Contributor notes about the pipeline's conventions (naming, checkpoint layout, what must move together when something is renamed) are in [AGENTS.md](AGENTS.md).

## Citation

```bibtex
@misc{reiter2026dias,
  title         = {Differentiable Learning of Lifted Action Schemas for
                   Classical Planning},
  author        = {Reiter, Jonas and Gebler, Jakob Elias and Geffner, Hector},
  year          = {2026},
  eprint        = {2605.13282},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.13282}
}
```

## License

MIT; see [LICENSE](LICENSE).
