# Miconic

All persons start in the lift at floor 1. Goal: each person delivered to a random destination floor.

| File          | Floors | Persons | Total objects |
| ------------- | ------ | ------- | ------------- |
| `train.pddl`  | 5      | 5       | 10            |
| `test-1.pddl` | 6      | 4       | 10            |
| `test-2.pddl` | 12     | 8       | 20            |
| `test-3.pddl` | 24     | 16      | 40            |

Note: the upstream generator uses a typed format (`origin`/`destin`/`boarded`/`served`). Generated problems match this repository's untyped Miconic domain format (`in_lift`/`in_floor`/`above`).

Generated with `pddl-generators/miconic/miconic-generator.py`.
