# Logistics — Generated Scaling Instances

Reference problem: `logistics_2_2_2_2_2_2.pddl` (2 cities, 2 trucks, 2 airplanes, 2 offices, 2 airports, 2 packets → 12 objects)

Each city has 1 office and 1 airport. Trucks are placed in their city, airplanes at random airports, packets at random offices. Goal: each packet delivered to a different office.

| File | Cities | Trucks | Airplanes | Offices | Airports | Packets | Total objects |
|---|---|---|---|---|---|---|---|
| `logistics_4_4_4_4_4_4.pddl` | 4 | 4 | 4 | 4 | 4 | 4 | 24 |
| `logistics_6_6_6_6_6_6.pddl` | 6 | 6 | 6 | 6 | 6 | 6 | 36 |
| `logistics_8_8_8_8_8_8.pddl` | 8 | 8 | 8 | 8 | 8 | 8 | 48 |

Note: the generator domain (`logistics-strips`) uses different predicate names than the `pddl_files` domain (`logistics`). Generated problems match the `logistics` domain format.

Generated with `pddl-generators/logistics/logistics-generator.py`, seed 42.
