# Hanoi — Generated Scaling Instances

Reference problem: `hanoi-3-2.pddl` (3 pegs, 2 discs → 5 objects). Existing instances cover up to D=10.

All discs start stacked on `peg1` (largest at bottom), goal is to move all discs to `peg3`.

| File | Pegs | Discs | Total objects |
|---|---|---|---|
| `hanoi-3-12.pddl` | 3 | 12 | 15 |
| `hanoi-3-24.pddl` | 3 | 24 | 27 |
| `hanoi-3-48.pddl` | 3 | 48 | 51 |

Note: Hanoi requires 2^D moves minimum, so these instances are exponentially harder than existing ones.

Generated with `pddl-generators/hanoi/hanoi-generator.py`.
