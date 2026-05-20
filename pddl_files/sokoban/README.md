# Sokoban — Generated Scaling Instances

Reference problem: `sokoban-4-4_2.pddl` (4×4 grid → 16 objects, 2 boxes)

Wall-free rectangular grids. Boxes and player placed at random non-overlapping positions; goal positions are distinct from initial positions.

Note: the generator `domain.pddl` uses a typed format (`typed-sokoban`). Generated problems match the `sokoban-domain` format (`has_player`/`has_box`/`adjacent`/`adjacent_2`).

| File | Grid | Objects | Boxes |
|---|---|---|---|
| `sokoban-6-6_3.pddl` | 6×6 | 36 | 3 |
| `sokoban-7-7_4.pddl` | 7×7 | 49 | 4 |
| `sokoban-9-9_5.pddl` | 9×9 | 81 | 5 |

Generated with `pddl-generators/sokoban/sokoban-generator.py`, seed 42.

## Sokoban-Pull — Generated Scaling Instances

Reference problem: `sokoban-4-4_2.pddl` (same domain, push+pull variant)

| File | Grid | Objects | Boxes |
|---|---|---|---|
| `pull_sokoban-6-6_3.pddl` | 6×6 | 36 | 3 |
| `pull_sokoban-7-7_4.pddl` | 7×7 | 49 | 4 |
| `pull_sokoban-9-9_5.pddl` | 9×9 | 81 | 5 |

Generated with `pddl-generators/sokoban/sokoban-generator.py`, seed 42.
