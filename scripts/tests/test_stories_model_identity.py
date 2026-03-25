import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stories_model_identity as script


def write_sentencepiece_fixture(path: Path) -> None:
    pieces = [
        ("▁", 0.0),
        ("H", 0.0),
        ("e", 0.0),
        ("l", 0.0),
        ("o", 0.0),
        ("▁H", 1.0),
        ("▁He", 2.0),
        ("▁Hel", 3.0),
        ("▁Hell", 4.0),
        ("▁Hello", 5.0),
    ]
    data = bytearray()
    data.extend(struct.pack("<i", 16))
    for piece, score in pieces:
        piece_bytes = piece.encode("utf-8")
        data.extend(struct.pack("<f", score))
        data.extend(struct.pack("<i", len(piece_bytes)))
        data.extend(piece_bytes)
    path.write_bytes(bytes(data))


def write_native_checkpoint(path: Path, *, vocab: int = 4, dim: int = 2, hidden_dim: int = 3, layers: int = 1) -> None:
    payload = bytearray()
    payload.extend(struct.pack("<7i", dim, hidden_dim, layers, 1, 1, vocab, 8))

    float_cursor = 1.0

    def append(count: int) -> None:
        nonlocal float_cursor
        values = [float_cursor + index for index in range(count)]
        payload.extend(struct.pack(f"<{count}f", *values))
        float_cursor += count

    append(vocab * dim)
    per_layer = [
        dim,
        dim * dim,
        dim * dim,
        dim * dim,
        dim * dim,
        dim,
        hidden_dim * dim,
        dim * hidden_dim,
        hidden_dim * dim,
    ]
    for count in per_layer:
        append(count * layers)
    append(dim)
    path.write_bytes(bytes(payload))


class StoriesModelIdentityTests(unittest.TestCase):
    def test_resolve_native_stories_path_prefers_explicit_then_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "explicit.bin"
            explicit.write_bytes(b"model")
            env_path = Path(directory) / "env.bin"
            env_path.write_bytes(b"env")

            resolved = script.resolve_native_stories_path(
                str(explicit),
                env={"STORIES_MODEL_PATH": str(env_path)},
            )

        self.assertEqual(resolved, explicit.resolve())

    def test_resolve_native_stories_path_uses_env_when_explicit_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / "env.bin"
            env_path.write_bytes(b"env")

            resolved = script.resolve_native_stories_path(
                "/tmp/does-not-exist-stories.bin",
                env={"STORIES_MODEL_PATH": str(env_path)},
            )

        self.assertEqual(resolved, env_path.resolve())

    def test_read_native_header_and_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stories.bin"
            write_native_checkpoint(checkpoint, vocab=4, dim=2, hidden_dim=3, layers=1)

            header = script.read_native_header(checkpoint)
            layout = script.native_layout(header)

        self.assertEqual(header.dim, 2)
        self.assertEqual(header.hidden_dim, 3)
        self.assertEqual(header.n_layers, 1)
        self.assertEqual(layout[0], ("embed", 8))
        self.assertEqual(layout[-1], ("rms_final", 2))

    def test_load_native_checkpoint_maps_expected_tensor_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stories.bin"
            write_native_checkpoint(checkpoint, vocab=4, dim=2, hidden_dim=3, layers=1)

            header, weights = script.load_native_checkpoint(checkpoint)

        self.assertEqual(header.vocab_size, 4)
        self.assertEqual(weights["model.embed_tokens.weight"].shape, (4, 2))
        self.assertEqual(weights["model.layers.0.self_attn.q_proj.weight"].shape, (2, 2))
        self.assertEqual(weights["model.layers.0.mlp.gate_proj.weight"].shape, (3, 2))
        self.assertEqual(weights["model.norm.weight"].shape, (2,))
        self.assertEqual(weights["lm_head.weight"].shape, (4, 2))

    def test_espresso_sentencepiece_tokenizer_round_trips_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "tokenizer.model"
            write_sentencepiece_fixture(model_path)
            tokenizer = script.EspressoSentencePieceTokenizer(model_path)

            encoded = tokenizer.encode("Hello")
            decoded = tokenizer.decode(encoded)

        self.assertEqual(decoded, "Hello")
        self.assertEqual(encoded, [9])

    def test_resolve_hf_snapshot_uses_cache_for_repo_id(self) -> None:
        with mock.patch("huggingface_hub.snapshot_download", return_value="/tmp/stories-hf-cache") as mocked:
            resolved = script.resolve_hf_snapshot("Xenova/llama2.c-stories110M")

        self.assertEqual(resolved, Path("/tmp/stories-hf-cache").resolve())
        mocked.assert_called_once_with("Xenova/llama2.c-stories110M", local_files_only=True)

    def test_compare_tokenizers_reports_prompt_level_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tokenizer_dir = Path(directory) / "native"
            tokenizer_dir.mkdir()
            write_sentencepiece_fixture(tokenizer_dir / "tokenizer.model")

            class DummyHFTokenizer:
                def encode(self, text, add_special_tokens=False):
                    self.last_text = text
                    return [1, 2, 3]

                def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
                    return "mismatch"

            with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyHFTokenizer()):
                report = script.compare_tokenizers(
                    tokenizer_dir,
                    Path("/tmp/hf"),
                    ["Hello"],
                )

        self.assertFalse(report["all_token_ids_match"])
        self.assertFalse(report["all_decoded_text_match"])
        self.assertEqual(report["prompts"][0]["prompt"], "Hello")


if __name__ == "__main__":
    unittest.main()
