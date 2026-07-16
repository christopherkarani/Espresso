import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.scripts import package_stories_exact_two_token_draft as script


class PackageStoriesExactTwoTokenDraftTests(unittest.TestCase):
    def test_stage_bundle_model_dir_copies_base_and_draft_and_writes_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_model_dir = root / "base"
            draft_model_dir = root / "draft-model"
            tokenizer_dir = root / "tokenizer"
            output_model_dir = root / "staged"

            (base_model_dir / "layers" / "0").mkdir(parents=True)
            (base_model_dir / "metadata.json").write_text(json.dumps({"name": "stories110m"}), encoding="utf-8")
            (base_model_dir / "final_norm.bin").write_text("base", encoding="utf-8")
            (draft_model_dir / "layers" / "0").mkdir(parents=True)
            (draft_model_dir / "metadata.json").write_text(
                json.dumps({"name": "stories110m-student-copy"}),
                encoding="utf-8",
            )
            (draft_model_dir / "final_norm.bin").write_text("draft", encoding="utf-8")
            tokenizer_dir.mkdir()

            descriptor_path, descriptor = script.stage_bundle_model_dir(
                base_model_dir=base_model_dir,
                draft_model_dir=draft_model_dir,
                output_model_dir=output_model_dir,
                tokenizer_dir=tokenizer_dir,
            )

            self.assertTrue((output_model_dir / "final_norm.bin").exists())
            self.assertTrue((output_model_dir / "draft" / "student" / "final_norm.bin").exists())
            self.assertEqual(descriptor.model_dir, "draft/student")
            self.assertEqual(descriptor.model_id, "stories110m-student-copy")
            self.assertEqual(descriptor.tokenizer_dir, "tokenizer")

            payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_dir"], "draft/student")
            self.assertEqual(payload["model_id"], "stories110m-student-copy")
            self.assertEqual(payload["tokenizer_dir"], "tokenizer")


if __name__ == "__main__":
    unittest.main()
