# Satellite-recal

Satellite-recal is Satellite with two changes: take_image deletes `calibrated ?i`, and calibrate requires `(not (calibrated ?i))`. It uses the instances of Satellite.

| File          | Satellites | Instruments | Modes | Ground stations | Targets | Total objects |
| ------------- | ---------- | ----------- | ----- | --------------- | ------- | ------------- |
| `train.pddl`  | 5          | 11          | 5     | 1               | 14      | 36            |
| `test-1.pddl` | 10         | 30          | 5     | 5               | 22      | 72            |
| `test-2.pddl` | 12         | 36          | 5     | 6               | 21      | 80            |
| `test-3.pddl` | 13         | 39          | 5     | 7               | 25      | 89            |
