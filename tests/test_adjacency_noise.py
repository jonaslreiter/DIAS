import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dias.utils import add_adj_noise


class AdjacencyNoiseTest(unittest.TestCase):
    def test_noise_only_flips_valid_relation_entries(self):
        adjacency = torch.tensor(
            [
                [[1, 1], [0, 0]],
                [[1, 0], [0, 0]],
            ],
            dtype=torch.float32,
        )

        noisy = add_adj_noise(
            adjacency,
            arities=[1, 2],
            remove_ratio=1.0,
            add_ratio=[1.0, 1.0],
            generator=torch.Generator().manual_seed(3),
        )

        expected = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[0, 1], [1, 1]],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(noisy, expected))


if __name__ == "__main__":
    unittest.main()
