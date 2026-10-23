# Satellite with recalibration

Satellite with two changes that make every transition change some atom: take_image uses up the calibration (deletes `calibrated ?i`), so an instrument is recalibrated before every image, and calibrate requires the instrument not to be calibrated yet. In the original, calibrating a calibrated instrument and imaging what was already imaged change nothing. Only calibrate becomes well-formed: take_image can still add a `have_image` that already holds, and switch_on still deletes `calibrated ?i` whether or not it holds. The instances are satellite's.

It exists because the precondition gate refutes a literal only where a transition changes something: on satellite, without the applicability term (`model.init_args.use_applicability_loss`), calibrate learns to forbid the `calibrated` it adds and take_image the `have_image`, since no transition that violates them changes anything. L1 never learned it, so the L1 comparison leaves it out.

| File          | Satellites | Instruments | Modes | Ground stations | Targets | Total objects |
| ------------- | ---------- | ----------- | ----- | --------------- | ------- | ------------- |
| `train.pddl`  | 5          | 11          | 5     | 1               | 14      | 36            |
| `test-1.pddl` | 10         | 30          | 5     | 5               | 22      | 72            |
| `test-2.pddl` | 12         | 36          | 5     | 6               | 21      | 80            |
| `test-3.pddl` | 13         | 39          | 5     | 7               | 25      | 89            |

Generated with `satellite/satellite-generator.py` from [PDDL generators](https://github.com/AI-Planning/pddl-generators).
