# DIAS: Differentiable Induction of Action Schemas

Code for **[Differentiable Learning of Lifted Action Schemas for Classical Planning](https://arxiv.org/abs/2605.13282)** by Jonas Reiter, Jakob Elias Gebler and Hector Geffner.

DIAS learns lifted STRIPS action schemas (preconditions, add effects and delete effects) from state transitions of a single planning problem, labelled with action names, STRIPS+ actions or full STRIPS actions. A relational GNN embeds the objects, a Sinkhorn assignment selects them into the action's parameter slots, and the lifted schema predicts the successor state. After training, the schemas are derived as a PDDL domain and evaluated by the successor states they generate on larger held-out problems.

## Example

Blocks-3's `move` and the schema DIAS learns from action names (seed 0):

```lisp
;; original                         ;; learned
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

The learned schema generates exactly the original's successors on the held-out problems; its extra preconditions hold wherever the original's do.

## Install

```bash
git clone https://github.com/jonaslreiter/DIAS.git
cd DIAS
pip install -e .
```

Python 3.10 or later, and a CUDA GPU for full runs. Every command below runs from the repository root.

## Quickstart

Train on Blocks-3 for 300 steps on the CPU, without Weights & Biases, and score the learned domain on a held-out problem:

```bash
dias fit --config configs/main.yaml --config pddl/blocks-3/config.yaml --trainer.accelerator=cpu --trainer.max_steps=300 --trainer.val_check_interval=100 --trainer.logger=false --trainer.default_root_dir=logs/dias/quickstart
python experiments/validate_checkpoint.py --validator dias --model-id quickstart --problem pddl/blocks-3/test-1.pddl
```

The run's checkpoints and, beside each, its learned domain as PDDL are in `logs/dias/quickstart/checkpoints/`. The paper's runs train for 10,000 steps.

## Domains

`pddl/` holds the 14 domains: `blocks-3`, `delivery`, `driverlog`, `gripper`, `hanoi`, `logistics`, `miconic`, `n-puzzle`, `satellite`, `satellite-recal`, `sokoban`, `sokoban-pull`, `spanner` and `visitall`. Each folder holds `domain.pddl`, the training problem `train.pddl`, the held-out problems `test-1.pddl` to `test-3.pddl`, the STRIPS+ argument positions in `strips+.yaml`, and `config.yaml`, the domain's training config:

```bash
dias fit --config configs/main.yaml --config pddl/hanoi/config.yaml
```

A domain of your own needs unary and binary predicates, STRIPS actions and injective parameter bindings; a folder with the same files makes it available to the experiment scripts. `dias fit --model.help dias.models.DIAS` and `--data.help dias.data.datamodule.PddlGraphDataModule` list every argument.

W&B is the default logger, logging to the account you are logged into or to `$WANDB_ENTITY`, including heatmaps of the learned schema. `--trainer.logger=false` trains without it, and `WANDB_MODE=offline` logs locally.

## Reproducing the paper

[experiments/README.md](experiments/README.md) lists the commands for every experiment.

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
