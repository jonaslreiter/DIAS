# Cell N-Puzzle — Generated Scaling Instances

Reference problem: `cell-npuzzle-3-3.pddl` (3×3 grid → 9 cells + 8 tiles = 17 objects)

Objects are grid cells `c_{x}_{y}` and tiles `t_1..t_{N-1}`. Tiles start at random positions; goal is blank at the bottom-right corner `c_{C-1}_{R-1}`.

Total objects for a C×R grid: 2×C×R − 1.

| File | Grid | Cells | Tiles | Total objects |
|---|---|---|---|---|
| `cell-npuzzle-4-4.pddl` | 4×4 | 16 | 15 | 31 |
| `cell-npuzzle-5-5.pddl` | 5×5 | 25 | 24 | 49 |
| `cell-npuzzle-6-6.pddl` | 6×6 | 36 | 35 | 71 |

Generated with `pddl-generators/npuzzle/cell-npuzzle-generator.py`, seed 42.
