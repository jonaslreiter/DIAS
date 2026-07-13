# Logistics — Generated Scaling Instances

Reference problem: `train.pddl`.

Each city has 1 office and 1 airport. Trucks are placed in their city, airplanes at random airports, packets at random offices. Goal: each packet delivered to a different office.

| File          | Cities | Trucks | Airplanes | Offices | Airports | Packets | Total objects |
| ------------- | ------ | ------ | --------- | ------- | -------- | ------- | ------------- |
| `test-1.pddl` | 4      | 4      | 4         | 4       | 4        | 4       | 24            |
| `test-2.pddl` | 6      | 6      | 6         | 6       | 6        | 6       | 36            |
| `test-3.pddl` | 8      | 8      | 8         | 8       | 8        | 8       | 48            |

Note: the generator domain (`logistics-strips`) uses different predicate names than the `pddl_files` domain (`logistics`). Generated problems match the `logistics` domain format.

Generated with `pddl-generators/logistics/logistics-generator.py`, seed 42.
