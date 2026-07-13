# Satellite — Generated Scaling Instances

Reference problem: `train.pddl`.

Each satellite carries 3 instruments. Each instrument supports random modes and has a ground station calibration target. Goals include pointing and have_image requirements.

| File          | Satellites | Instruments | Modes | Ground stations | Targets | Total objects |
| ------------- | ---------- | ----------- | ----- | --------------- | ------- | ------------- |
| `test-1.pddl` | 10         | 30          | 5     | 5               | 22      | 72            |
| `test-2.pddl` | 12         | 36          | 5     | 6               | 21      | 80            |
| `test-3.pddl` | 13         | 39          | 5     | 7               | 25      | 89            |

Generated with `pddl-generators/satellite/satellite-generator.py`, seed 42.
