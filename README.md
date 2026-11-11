# DIAS: Differentiable Induction of Action Schemas

This repository holds the code of **[Differentiable Learning of Lifted Action Schemas for Classical Planning](https://arxiv.org/abs/2605.13282)** by Jonas Reiter, Jakob Elias Gebler and Hector Geffner.

DIAS learns lifted STRIPS action schemas (preconditions, add effects and delete effects) from the state transitions of a single planning problem, labelled with action names, STRIPS+ actions or full STRIPS actions. A relational GNN embeds the objects of a state, a Sinkhorn assignment selects them into the parameter slots of the action, and the lifted schema predicts the successor state. After training, the schemas are derived as a PDDL domain, which is evaluated by the successor states it generates on larger held-out problems.

## Example

On the left is the `move` action of Blocks-3, and on the right is the schema DIAS learns from action names alone (seed 0):

```lisp
(:action move                       (:action move
  :parameters (?bm ?bf ?bt)           :parameters (?x1 ?x2 ?x3)
  :precondition (and                  :precondition (and
    (clear ?bm)                         (clear ?x1)
    (clear ?bt)                         (clear ?x2)
    (on ?bm ?bf)                        (on ?x1 ?x3)
    (not (= ?bm ?bt))                   (not (= ?x1 ?x2))
    (not (= ?bm ?bf))                   (not (= ?x1 ?x3))
    (not (= ?bf ?bt)))                  (not (= ?x2 ?x3))
                                        (not (clear ?x3))
                                        (not (on-table ?x1))
                                        (not (on ?x1 ?x1)) ...)
  :effect (and                        :effect (and
    (not (clear ?bt))                   (not (clear ?x2))
    (not (on ?bm ?bf))                  (not (on ?x1 ?x3))
    (on ?bm ?bt)                        (on ?x1 ?x2)
    (clear ?bf)))                       (clear ?x3)))
```

The learned schema generates exactly the successors of the original one on the held-out problems, since its extra preconditions hold wherever the original ones do.

## Install

```bash
git clone https://github.com/jonaslreiter/DIAS.git
cd DIAS
pip install -e .
```

DIAS needs Python 3.10 or later. A CUDA GPU is optional, but it makes full training runs much faster. All commands below run from the root of the repository.

## Quickstart

The following commands train on Blocks-3 for 300 steps on the CPU without Weights & Biases, and then score the learned domain on a held-out problem:

```bash
dias fit --config configs/main.yaml --config pddl/blocks-3/config.yaml --trainer.accelerator=cpu --trainer.max_steps=300 --trainer.val_check_interval=100 --trainer.logger=false --trainer.default_root_dir=logs/dias/quickstart
python experiments/validate_checkpoint.py --validator dias --model-id quickstart --problem pddl/blocks-3/test-1.pddl
```

The checkpoints of the run are written to `logs/dias/quickstart/checkpoints/`, each with its learned domain as a PDDL file beside it. After 300 steps, the learned domain is still far from the original one, while the runs of the paper train for 10,000 steps.

## Domains

The folder `pddl/` holds the 14 domains of the paper: `blocks-3`, `delivery`, `driverlog`, `gripper`, `hanoi`, `logistics`, `miconic`, `n-puzzle`, `satellite`, `satellite-recal`, `sokoban`, `sokoban-pull`, `spanner` and `visitall`. The domains and their instances come from [PDDL generators](https://github.com/AI-Planning/pddl-generators), some domains with minor changes. Each folder holds `domain.pddl`, the training problem `train.pddl`, the held-out problems `test-1.pddl` to `test-3.pddl`, the STRIPS+ argument positions in `strips+.yaml`, and the training config `config.yaml` of the domain:

```bash
dias fit --config configs/main.yaml --config pddl/hanoi/config.yaml
```

A domain of your own needs unary and binary predicates, STRIPS actions and injective parameter bindings. A folder with the same files makes it available to the experiment scripts. The commands `dias fit --model.help dias.models.DIAS` and `dias fit --data.help dias.data.datamodule.PddlGraphDataModule` list every argument.

Weights & Biases is the default logger. It logs to the account you are logged into, or to `$WANDB_ENTITY`, and includes heatmaps of the learned schema. The option `--trainer.logger=false` trains without it, and `WANDB_MODE=offline` logs locally.

## Reproducing the paper

The file [experiments/README.md](experiments/README.md) lists the commands of every experiment of the paper.

## Layout

```
src/dias/       the package: model, data, schema derivation, PDDL handling, validation
configs/        the training config and its overlays
pddl/           the domains
experiments/    the experiment scripts and their configs
tests/          the tests
```

## Development

```bash
pip install -e . --group dev
pre-commit install
pytest
```

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
