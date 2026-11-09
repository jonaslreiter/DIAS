# Logistics

| File          | Cities | Trucks | Airplanes | Offices | Airports | Packages | Total objects |
| ------------- | ------ | ------ | --------- | ------- | -------- | ------- | ------------- |
| `train.pddl`  | 2      | 3      | 2         | 4       | 2        | 2       | 15            |
| `test-1.pddl` | 4      | 4      | 4         | 4       | 4        | 4       | 24            |
| `test-2.pddl` | 6      | 6      | 6         | 6       | 6        | 6       | 36            |
| `test-3.pddl` | 8      | 8      | 8         | 8       | 8        | 8       | 48            |

Generated with `logistics/logistics-generator.py` from [PDDL generators](https://github.com/AI-Planning/pddl-generators).

The unary predicate marking the packages is `package`, where the standard formulation writes `object`.
