# Satellite — Generated Scaling Instances

Reference problem: `satellite_9.pddl` (5 satellites, 11 instruments, 5 modes, 15 directions → 36 objects)

Each satellite carries 3 instruments. Each instrument supports random modes and has a ground station calibration target. Goals include pointing and have_image requirements.

| File | Satellites | Instruments | Modes | Ground stations | Targets | Total objects |
|---|---|---|---|---|---|---|
| `satellite_10_3_5_5_22.pddl` | 10 | 30 | 5 | 5 | 22 | 72 |
| `satellite_12_3_5_6_21.pddl` | 12 | 36 | 5 | 6 | 21 | 80 |
| `satellite_13_3_5_7_25.pddl` | 13 | 39 | 5 | 7 | 25 | 89 |

Generated with `pddl-generators/satellite/satellite-generator.py`, seed 42.
