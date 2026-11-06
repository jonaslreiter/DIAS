# Logistics (package predicate)

| File          | Cities | Trucks | Airplanes | Offices | Airports | Packets | Total objects |
| ------------- | ------ | ------ | --------- | ------- | -------- | ------- | ------------- |
| `train.pddl`  | 2      | 3      | 2         | 4       | 2        | 2       | 15            |
| `test-1.pddl` | 4      | 4      | 4         | 4       | 4        | 4       | 24            |
| `test-2.pddl` | 6      | 6      | 6         | 6       | 6        | 6       | 36            |
| `test-3.pddl` | 8      | 8      | 8         | 8       | 8        | 8       | 48            |

Generated with `logistics/logistics-generator.py` from [PDDL generators](https://github.com/AI-Planning/pddl-generators).

The same domain and instances as `pddl/logistics`, with the unary predicate `object` that types the packages renamed to `package` throughout. Under its old name the predicate collides with PDDL's root type `object`, which DIAS leaves out of the state, so packages had no channel.
