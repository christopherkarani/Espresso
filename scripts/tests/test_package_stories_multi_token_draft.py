import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import package_stories_multi_token_draft as script


class PackageStoriesMultiTokenDraftTests(unittest.TestCase):
    def test_stage_bundle_model_dir_can_write_descriptor_for_learned_draft(self) -> None:
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
                json.dumps({"name": "stories110m-state-predictor"}),
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

            self.assertIsNotNone(descriptor_path)
            self.assertIsNotNone(descriptor)
            self.assertTrue((output_model_dir / "final_norm.bin").exists())
            self.assertTrue((output_model_dir / "draft" / "student" / "final_norm.bin").exists())
            self.assertEqual(descriptor.model_dir, "draft/student")
            self.assertEqual(descriptor.model_id, "stories110m-state-predictor")
            payload = json.loads(Path(descriptor_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["model_dir"], "draft/student")
            self.assertEqual(payload["model_id"], "stories110m-state-predictor")

    def test_stage_bundle_model_dir_supports_oracle_mode_without_draft_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_model_dir = root / "base"
            tokenizer_dir = root / "tokenizer"
            output_model_dir = root / "staged"

            (base_model_dir / "layers" / "0").mkdir(parents=True)
            (base_model_dir / "metadata.json").write_text(json.dumps({"name": "stories110m"}), encoding="utf-8")
            (base_model_dir / "final_norm.bin").write_text("base", encoding="utf-8")
            tokenizer_dir.mkdir()

            descriptor_path, descriptor = script.stage_bundle_model_dir(
                base_model_dir=base_model_dir,
                draft_model_dir=None,
                output_model_dir=output_model_dir,
                tokenizer_dir=tokenizer_dir,
            )

            self.assertIsNone(descriptor_path)
            self.assertIsNone(descriptor)
            self.assertTrue((output_model_dir / "final_norm.bin").exists())
            self.assertFalse((output_model_dir / "draft").exists())

    def test_parse_supported_verify_steps_requires_sorted_unique_values(self) -> None:
        self.assertEqual(script.parse_supported_verify_steps("1,2,4"), [1, 2, 4])
        with self.assertRaises(ValueError):
            script.parse_supported_verify_steps("2,1,2")

    def test_stage_bundle_model_dir_supports_custom_descriptor_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_model_dir = root / "base"
            draft_model_dir = root / "draft-model"
            tokenizer_dir = root / "tokenizer"
            output_model_dir = root / "staged"

            (base_model_dir / "layers" / "0").mkdir(parents=True)
            (base_model_dir / "metadata.json").write_text(json.dumps({"name": "stories110m"}), encoding="utf-8")
            (draft_model_dir / "layers" / "0").mkdir(parents=True)
            (draft_model_dir / "metadata.json").write_text(
                json.dumps({"name": "stories110m-state-predictor"}),
                encoding="utf-8",
            )
            tokenizer_dir.mkdir()

            descriptor_path, descriptor = script.stage_bundle_model_dir(
                base_model_dir=base_model_dir,
                draft_model_dir=draft_model_dir,
                output_model_dir=output_model_dir,
                tokenizer_dir=tokenizer_dir,
                draft_descriptor_relative_path="weights/draft/custom-descriptor.json",
                draft_model_relative_dir="weights/draft/predictor",
            )

            self.assertEqual(descriptor.model_dir, "weights/draft/predictor")
            self.assertEqual(descriptor_path, output_model_dir / "weights" / "draft" / "custom-descriptor.json")


if __name__ == "__main__":
    unittest.main()
