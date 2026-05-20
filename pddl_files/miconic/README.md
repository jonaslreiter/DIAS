# Miconic — Generated Scaling Instances

Reference problem: `miconic_3_2_full_labels.pddl` (3 floors, 2 persons → 5 objects)

All persons start in the lift at floor 1. Goal: each person delivered to a random destination floor.

| File | Floors | Persons | Total objects |
|---|---|---|---|
| `miconic_6_4.pddl` | 6 | 4 | 10 (2×) |
| `miconic_12_8.pddl` | 12 | 8 | 20 (4×) |
| `miconic_24_16.pddl` | 24 | 16 | 40 (8×) |

Note: the generator `domain.pddl` uses a typed format (`origin`/`destin`/`boarded`/`served`). Generated problems match the untyped `miconic.pddl` format (`in_lift`/`in_floor`/`above`).

Generated with `pddl-generators/miconic/miconic-generator.py`, seed 42.
