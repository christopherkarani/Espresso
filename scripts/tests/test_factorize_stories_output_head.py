import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import espresso_llama_weights
from scripts import factorize_stories_output_head as script
from scripts.tests.test_espresso_llama_weights import write_fixture


class FactorizeStoriesOutputHeadTests(unittest.TestCase):
    def test_factorize_output_head_writes_rank_limited_blobfiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights_dir = root / "weights"
            projection_out = root / "cls_proj.bin"
            expansion_out = root / "cls_expand.bin"
            stats_out = root / "stats.json"
            weights_dir.mkdir(parents=True, exist_ok=True)
            write_fixture(weights_dir)

            stats = script.factorize_output_head(
                weights_dir=weights_dir,
                rank=2,
                projection_out=projection_out,
                expansion_out=expansion_out,
                stats_out=stats_out,
            )

            projection = espresso_llama_weights.read_blobfile_array(projection_out, (2, 4))
            expansion = espresso_llama_weights.read_blobfile_array(expansion_out, (8, 2))
            stats_exists = stats_out.exists()

        self.assertEqual(int(stats["rank"]), 2)
        self.assertEqual(projection.shape, (2, 4))
        self.assertEqual(expansion.shape, (8, 2))
        self.assertGreaterEqual(float(stats["relative_fro_error"]), 0.0)
        self.assertLess(float(stats["relative_fro_error"]), 1.0)
        self.assertTrue(stats_exists)


if __name__ == "__main__":
    unittest.main()
