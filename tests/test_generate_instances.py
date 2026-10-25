"""The committed scaling-study instances."""

import json
import unittest
from pathlib import Path

from dias.schema.domain import DomainSchema

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "experiments" / "scaling" / "instances" / "index.json"


class CommittedInstanceTest(unittest.TestCase):
    def test_every_instance_loads_against_its_domain_with_the_indexed_objects(self):
        for domain, entries in json.loads(INDEX.read_text()).items():
            for entry in entries:
                with self.subTest(domain=domain, instance=entry["id"]):
                    schema = DomainSchema.from_pddl(
                        str(ROOT / "pddl" / domain / "domain.pddl"),
                        str(ROOT / "experiments" / entry["path"]),
                    )
                    self.assertEqual(schema.num_objects, entry["num_objects"])


if __name__ == "__main__":
    unittest.main()
