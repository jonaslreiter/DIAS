# DIAS: Differentiable Induction of Action Schemas

Code for **[Differentiable Learning of Lifted Action Schemas for Classical Planning](https://arxiv.org/abs/2605.13282)**, by Jonas Reiter, Jakob Elias Gebler and Hector Geffner, arXiv:2605.13282.

DIAS learns **lifted STRIPS action schemas** (preconditions, add effects and delete effects over action *parameters*) from observed state transitions of a single planning problem. A relational GNN embeds the objects of a state, a Sinkhorn-style assignment binds them to the schema's parameter slots, and the learned schema is applied to predict the successor state. The trained parameters are then read back as symbolic PDDL and evaluated on held-out problems of different sizes, by whether the recovered domain generates the same successor states as the real one.

## What it learns

Ten thousand steps on Blocksworld's five-block training instance, given only `(state, action, successor)` triples, then the model's parameters read back as PDDL:

```lisp
;; ground truth                ;; recovered (seed 0, threshold 0.5)
(:action move                  (:action learned_move
  :parameters (?bm ?bf ?bt)      :parameters (?x1 ?x2 ?x4)
  :precondition (and             :precondition (and
    (clear ?bm)                    (clear ?x1)
    (clear ?bt)                    (clear ?x2)
    (on    ?bm ?bf)                (on    ?x1 ?x4)
    (not (= ?bm ?bt))              (not (= ?x1 ?x2))
    (not (= ?bm ?bf))              (not (= ?x1 ?x4))
    (not (= ?bf ?bt)))             (not (= ?x4 ?x2))
                                   ... and 16 redundant literals, such
                                   as (= ?x1 ?x1), that hold wherever
                                   those six do)
  :effect (and                   :effect (and
    (on    ?bm ?bt)                (on    ?x1 ?x2)
    (clear ?bf)                    (clear ?x4)
    (not (clear ?bt))              (not (clear ?x2))
    (not (on    ?bm ?bf))))        (not (on    ?x1 ?x4))))
```

Both effect sets are exact and the arity is recovered. Slots are numbered by their position in the model's slot budget, so which of them are used, and in what order, is the model's to choose: here `?x1`, `?x2` and `?x4` stand for `?bm`, `?bt` and `?bf`. The 16 extra literals are either implied by the six real preconditions or invariants of the domain, so they block nothing the real schema allows. Evaluation is behavioural throughout: a schema is *sound* when it generates no successor the real domain does not, and *complete* when it misses none. This run is both, on all three held-out instances, the largest of them eight times the size of the instance it trained on.

## Install

DIAS is installed from a clone:

```bash
git clone https://github.com/jonaslreiter/DIAS.git
cd DIAS
pip install -e .
```

Python 3.10+, and a CUDA GPU for full-size runs. The domains, the configs and the experiment scripts are the repository itself, so every command below is run from inside the checkout.

## Quickstart

Train on the smallest domain, on CPU, with no Weights & Biases account. Two configs: the base training setup, and the domain to apply it to.

```bash
dias fit --config configs/main.yaml --config pddl/blocks-3/config.yaml --trainer.accelerator=cpu --trainer.max_steps=300 --trainer.val_check_interval=100 --trainer.logger=false --trainer.default_root_dir=logs/dias/quickstart
```

A few minutes later `logs/dias/quickstart/checkpoints/` holds the trained model. Read the schema it learned back as PDDL and score it on a held-out problem:

```bash
python experiments/validate_checkpoint.py --validator dias --model-id quickstart --log-dir logs --problem pddl/blocks-3/test-1.pddl
```

This prints per-action `tp`/`fp`/`fn` and the precision and recall they sum to. 300 steps will not recover Blocksworld; the paper's runs take 10 000 on a GPU.

### Choosing a domain

Swap the second config: it carries both the domain's files and its tuned hyperparameters, and being layered last it wins.

```bash
dias fit --config configs/main.yaml --config pddl/hanoi/config.yaml
```

The 14 domains are `blocks-3`, `delivery`, `driverlog`, `gripper`, `hanoi`, `logistics`, `miconic`, `n-puzzle`, `satellite`, `satellite-recal`, `sokoban`, `sokoban-pull`, `spanner`, `visitall`; `satellite-recal` is satellite with every transition changing something, which the paper did not have. Each `pddl/<domain>/` holds `domain.pddl`, a training instance `train.pddl`, three held-out `test-*.pddl` of increasing size, the STRIPS+ argument positions in `strips+.yaml`, and the tuned `config.yaml`.

`configs/main.yaml` names no domain, so a chain that names none is refused. Every model and datamodule argument is listed, with a line on what it does, by `dias fit --model.help dias.models.DIAS` and `--data.help dias.data.datamodule.PddlGraphDataModule`, and any of them can be overridden on the command line.

### Your own PDDL

Point `domain_path` / `problem_path` at any domain whose predicates are unary or binary and whose actions are STRIPS with injective parameter binding. To make it a domain like the 13 above, give it a `pddl/<name>/` folder holding the same files. The folder *is* the domain, so `--domains <name>` works in the sweep launcher as soon as it exists.

### Weights & Biases

W&B is the default logger, with `entity: null`, meaning whichever account you are logged into (or `$WANDB_ENTITY`). Use `--trainer.logger=false` to train without it, or `WANDB_MODE=offline` to record locally. When it is on, the learned schema is logged periodically as interactive heatmaps: effects, preconditions, slot-to-object assignments and slot weights.

## Reproducing the paper

The recipes for the main table, the ablations, the sweeps, the threshold study, the instance-size scaling study, aggregation and cluster setup are in **[experiments/README.md](experiments/README.md)**. In short:

```bash
python experiments/launch.py --experiment dias-full
python experiments/collect_results.py --experiment dias-full
```

Some defaults have moved on since the paper: the model gets no `=` relation, and the domain it learns requires every schema's parameters to be distinct instead, because bindings are injective; and hanoi and spanner train on the instances the L1 comparison learned from. The recipes say how to restore the paper's setting.

## Repository layout

```
src/dias/
├── main.py           LightningCLI entry point (also `dias`)
├── models/           selection, preconditions, effects, encoder, objectives
├── data/             PDDL datamodule, transition sampling, adjacency noise
├── schema/           DomainSchema (metadata), ActionSchema, the learned domain as PDDL
├── pddl/             PDDL text handling, untyping, the transpiler
├── validation/       scoring a learned domain by its successors
└── reporting/        the W&B charts the learned schema is logged as

experiments/          everything you run to reproduce the paper, start here
configs/              main.yaml and the `dias fit` overlays beside it
pddl/                 the 14 domains, one folder each
tests/                one module per package module, named after it
```

## Development

```bash
pip install -e . --group dev      # PEP 735 groups; needs pip >= 25.1
pre-commit install                # optional: runs the two below on a commit
pytest
black --check . && isort --check-only .
```

Contributor notes about the pipeline's conventions are in [AGENTS.md](AGENTS.md).

## Citation

```bibtex
@misc{reiter2026dias,
  title         = {Differentiable Learning of Lifted Action Schemas for Classical Planning},
  author        = {Reiter, Jonas and Gebler, Jakob Elias and Geffner, Hector},
  year          = {2026},
  eprint        = {2605.13282},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.13282}
}
```
