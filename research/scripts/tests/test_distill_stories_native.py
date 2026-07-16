import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.scripts import distill_stories_native as script


class DistillStoriesNativeTests(unittest.TestCase):
    def test_load_config_parses_student_and_export_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            prompts_path = Path(directory) / "prompts.txt"
            prompts_path.write_text("hello:Hello\nlab:Long prompt\n", encoding="utf-8")
            config_path.write_text(
                json.dumps(
                    {
                        "teacher": {
                            "source": "hf",
                            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        },
                        "student": {
                            "name": "stories-gqa4",
                            "n_layer": 4,
                            "n_head": 8,
                            "n_kv_head": 2,
                            "d_model": 512,
                            "head_dim": 64,
                            "hidden_dim": 1536,
                            "vocab": 32000,
                            "max_seq": 256,
                            "norm_eps": 1e-5,
                            "rope_theta": 10000.0,
                        },
                        "train": {
                            "texts": ["Hello world"],
                            "texts_file": str(prompts_path),
                            "text_globs": [str(Path(directory) / "*.story.txt")],
                            "sequence_length": 32,
                            "window_stride": 8,
                            "max_samples": 1,
                            "batch_size": 1,
                            "steps": 0,
                            "learning_rate": 1e-5,
                            "device": "cpu",
                        },
                        "export": {
                            "output_dir": "/tmp/stories-gqa4",
                            "context_target_tokens": 256,
                            "optimization_recipe": "stories-gqa4-proof",
                            "quality_gate": "proof-only",
                        },
                        "initialization": {
                            "mode": "teacher_copy",
                        },
                        "future_head": {
                            "steps": 3,
                            "learning_rate": 2e-5,
                            "output_path": "/tmp/future-sidecar.bin",
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = script.DistillationConfig.load(config_path)

        self.assertEqual(config.student.n_kv_head, 2)
        self.assertEqual(config.student.to_llama_config().num_key_value_heads, 2)
        self.assertEqual(config.export.context_target_tokens, 256)
        self.assertEqual(config.export.optimization_recipe, "stories-gqa4-proof")
        self.assertEqual(config.initialization.mode, "teacher_copy")
        self.assertEqual(config.train.window_stride, 8)
        self.assertEqual(config.train.texts[:3], ["Hello world", "Hello", "Long prompt"])
        self.assertEqual(config.train.text_globs, [str(Path(directory) / "*.story.txt")])
        self.assertIsNotNone(config.future_head)
        self.assertEqual(config.future_head.output_path, "/tmp/future-sidecar.bin")

    def test_generated_corpus_spec_parses_prompt_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts_path = Path(directory) / "prompts.txt"
            prompts_path.write_text("# comment\nhello:Hello\nfox:The quick brown fox\n", encoding="utf-8")

            spec = script.GeneratedCorpusSpec.from_dict(
                {
                    "prompts_file": str(prompts_path),
                    "samples_per_prompt": 3,
                    "max_new_tokens": 40,
                    "temperature": 0.7,
                    "top_k": 12,
                    "seed": 99,
                }
            )

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.prompts, ["Hello", "The quick brown fox"])
        self.assertEqual(spec.samples_per_prompt, 3)
        self.assertEqual(spec.max_new_tokens, 40)
        self.assertEqual(spec.top_k, 12)
        self.assertEqual(spec.seed, 99)

    def test_initialize_student_from_teacher_supports_truncate_prefix(self) -> None:
        class TinyModel:
            def __init__(self, state):
                self._state = state

            def state_dict(self):
                return self._state

        teacher = TinyModel(
            {
                "model.embed_tokens.weight": torch.tensor([[1.0, 2.0]]),
                "model.layers.0.self_attn.q_proj.weight": torch.tensor([[3.0, 4.0]]),
                "model.layers.1.self_attn.q_proj.weight": torch.tensor([[5.0, 6.0]]),
                "model.norm.weight": torch.tensor([7.0, 8.0]),
                "lm_head.weight": torch.tensor([[9.0, 10.0]]),
            }
        )
        student_state = {
            "model.embed_tokens.weight": torch.zeros((1, 2)),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros((1, 2)),
            "model.norm.weight": torch.zeros(2),
            "lm_head.weight": torch.zeros((1, 2)),
        }
        student = TinyModel(student_state)

        mode = script.initialize_student_from_teacher(
            student_model=student,
            teacher_model=teacher,
            initialization=script.InitializationSpec(mode="teacher_truncate_prefix"),
        )

        self.assertEqual(mode, "teacher_truncate_prefix")
        self.assertTrue(torch.equal(student_state["model.embed_tokens.weight"], teacher.state_dict()["model.embed_tokens.weight"]))
        self.assertTrue(
            torch.equal(
                student_state["model.layers.0.self_attn.q_proj.weight"],
                teacher.state_dict()["model.layers.0.self_attn.q_proj.weight"],
            )
        )

    def test_build_training_examples_uses_text_inputs(self) -> None:
        class Tokenizer:
            def encode(self, text: str) -> list[int]:
                return list(range(len(text.split()) + 1))

        examples = script.build_training_examples(
            ["one two three", "alpha beta"],
            Tokenizer(),
            sequence_length=8,
            max_samples=2,
            window_stride=4,
        )

        self.assertEqual(len(examples), 2)
        self.assertTrue(all(example.numel() >= 2 for example in examples))

    def test_build_training_examples_slides_windows_for_long_inputs(self) -> None:
        class Tokenizer:
            def encode(self, text: str) -> list[int]:
                return list(range(10))

        examples = script.build_training_examples(
            ["long input"],
            Tokenizer(),
            sequence_length=4,
            max_samples=3,
            window_stride=2,
        )

        self.assertEqual(len(examples), 3)
        self.assertEqual([example.tolist() for example in examples], [[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8]])

    def test_load_texts_from_globs_reads_unique_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.story.txt").write_text("first file", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "two.story.txt").write_text("second file", encoding="utf-8")

            texts, matched_paths = script.load_texts_from_globs([str(root / "**" / "*.story.txt")])

        self.assertCountEqual(texts, ["first file", "second file"])
        self.assertEqual(len(matched_paths), 2)
        self.assertTrue(all(path.endswith(".story.txt") for path in matched_paths))

    def test_export_future_sidecar_writes_expected_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future-sidecar.bin"
            student = script.StudentSpec(
                name="stories",
                n_layer=12,
                n_head=12,
                n_kv_head=12,
                d_model=4,
                head_dim=2,
                hidden_dim=8,
                vocab=6,
                max_seq=16,
                norm_eps=1e-5,
                rope_theta=10000.0,
                eos_token=None,
            )

            exported_path = script.export_future_sidecar(
                path=path,
                student=student,
                future_rms=torch.arange(4, dtype=torch.float32),
                future_classifier=torch.arange(24, dtype=torch.float32).reshape(6, 4),
                teacher_classifier_was_shared=True,
            )

            payload = exported_path.read_bytes()

        header = struct.unpack("<iiiiiiII", payload[:32])
        self.assertEqual(header[0], 0x32535446)
        self.assertEqual(header[1], 1)
        self.assertEqual(header[2], 4)
        self.assertEqual(header[3], 6)
        self.assertEqual(header[4], 12)
        self.assertEqual(header[5], 2)
        self.assertEqual(header[6], 0b111)
        self.assertEqual(len(payload), 32 + (4 * 4) + (24 * 4))


if __name__ == "__main__":
    unittest.main()
