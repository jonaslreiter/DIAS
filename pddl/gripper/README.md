# Gripper

All balls start in `rooma`, goal is to move all balls to `roomb`.

| File          | Balls | Grippers | Rooms | Total objects |
| ------------- | ----- | -------- | ----- | ------------- |
| `train.pddl`  | 5     | 3        | 2     | 10            |
| `test-1.pddl` | 6     | 4        | 2     | 12            |
| `test-2.pddl` | 12    | 8        | 2     | 22            |
| `test-3.pddl` | 24    | 16       | 2     | 42            |

Note: `test-1.pddl` comes from the generated 6-ball/4-gripper instance rather than the older partial-goal variant.

Generated with `pddl-generators/gripper/gripper-generator.py`.
