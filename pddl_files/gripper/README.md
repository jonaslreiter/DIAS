# Gripper — Generated Scaling Instances

Reference problem: `gripper_3_2.pddl` (3 balls, 2 grippers, 2 rooms → 7 objects)

All balls start in `rooma`, goal is to move all balls to `roomb`.

| File | Balls | Grippers | Rooms | Total objects |
|---|---|---|---|---|
| `gripper_6_4_gen.pddl` | 6 | 4 | 2 | 12 (≈2×) |
| `gripper_12_8.pddl` | 12 | 8 | 2 | 22 (≈3×) |
| `gripper_24_16.pddl` | 24 | 16 | 2 | 42 (≈6×) |

Note: `gripper_6_4.pddl` already existed with a different setup (partial goal, named grippers), hence the `_gen` suffix.

Generated with `pddl-generators/gripper/gripper-generator.py`.
