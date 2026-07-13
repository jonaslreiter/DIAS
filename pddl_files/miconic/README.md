# Miconic — Generated Scaling Instances

Reference problem: `train.pddl`.

All persons start in the lift at floor 1. Goal: each person delivered to a random destination floor.

| File          | Floors | Persons | Total objects |
| ------------- | ------ | ------- | ------------- |
| `test-1.pddl` | 6      | 4       | 10 (2×)       |
| `test-2.pddl` | 12     | 8       | 20 (4×)       |
| `test-3.pddl` | 24     | 16      | 40 (8×)       |

Note: the upstream generator uses a typed format (`origin`/`destin`/`boarded`/`served`). Generated problems match this repository's untyped Miconic domain format (`in_lift`/`in_floor`/`above`).

Generated with `pddl-generators/miconic/miconic-generator.py`, seed 42.
