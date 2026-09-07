# Hanoi

Existing instances cover up to D=10.

All discs start stacked on `peg1` (largest at bottom), goal is to move all discs to `peg3`.

| File          | Pegs | Discs | Total objects |
| ------------- | ---- | ----- | ------------- |
| `train.pddl`  | 3    | 6     | 9             |
| `test-1.pddl` | 3    | 12    | 15            |
| `test-2.pddl` | 3    | 24    | 27            |
| `test-3.pddl` | 3    | 48    | 51            |

Note: Hanoi requires 2^D moves minimum, so these instances are exponentially harder than existing ones.

Generated with `pddl-generators/hanoi/hanoi-generator.py`.
