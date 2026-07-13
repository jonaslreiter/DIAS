# Cell N-Puzzle — Generated Scaling Instances

Reference problem: `train.pddl`.

Objects are grid cells `c_{x}_{y}` and tiles `t_1..t_{N-1}`. Tiles start at random positions; goal is blank at the bottom-right corner `c_{C-1}_{R-1}`.

Total objects for a C×R grid: 2×C×R − 1.

| File          | Grid | Cells | Tiles | Total objects |
| ------------- | ---- | ----- | ----- | ------------- |
| `test-1.pddl` | 4×4  | 16    | 15    | 31            |
| `test-2.pddl` | 5×5  | 25    | 24    | 49            |
| `test-3.pddl` | 6×6  | 36    | 35    | 71            |

Generated with `pddl-generators/npuzzle/cell-npuzzle-generator.py`, seed 42.
