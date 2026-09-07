# Sokoban

Wall-free rectangular grids. Boxes and player placed at random non-overlapping positions; goal positions are distinct from initial positions.

Note: the upstream generator uses a typed format (`typed-sokoban`). Generated problems match this repository's `sokoban-domain` format (`has_player`/`has_box`/`adjacent`/`adjacent_2`).

| File          | Grid | Objects | Boxes |
| ------------- | ---- | ------- | ----- |
| `train.pddl`  | 4×4  | 16      | 3     |
| `test-1.pddl` | 6×6  | 36      | 3     |
| `test-2.pddl` | 7×7  | 49      | 4     |
| `test-3.pddl` | 9×9  | 81      | 5     |

Generated with `pddl-generators/sokoban/sokoban-generator.py`.
