import importlib.util
import unittest
from pathlib import Path


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ChatVsMLXFourTurnTests(unittest.TestCase):
    def test_four_scripted_turns_are_the_hn_prompts(self) -> None:
        script = load_script("chat_vs_mlx_four_turn.py")
        self.assertEqual(len(script.TURNS), 4)
        self.assertEqual(script.TURNS[0], "what is a good way to learn Swift concurrency?")

    def test_command_is_vs_mlx_greedy_and_does_not_default_to_quant(self) -> None:
        script = load_script("chat_vs_mlx_four_turn.py")

        class Args:
            espresso = Path("./espresso")
            model = Path("/tmp/qwen.esp")
            max_tokens = 64
            mlx_quant = None

        command = script.build_command(Args())
        self.assertIn("--vs", command)
        self.assertIn("mlx", command)
        self.assertIn("--greedy", command)
        self.assertIn("--plain", command)
        self.assertIn("--power", command)
        self.assertNotIn("--mlx-quant", command)
        self.assertNotIn("4bit", command)

        labeled = Args()
        labeled.mlx_quant = "4bit"
        labeled_command = script.build_command(labeled)
        self.assertIn("--mlx-quant", labeled_command)
        self.assertIn("4bit", labeled_command)


class MLXStreamScriptTests(unittest.TestCase):
    def test_completion_tokens_per_second_excludes_compile(self) -> None:
        script = load_script("mlx_qwen_stream.py")
        self.assertEqual(script.completion_tokens_per_second(13, 1000), 13)
        self.assertEqual(script.completion_tokens_per_second(10, 500), 20)
        compile_plus_completion = 8400 + 500
        self.assertNotEqual(
            script.completion_tokens_per_second(10, 500),
            script.completion_tokens_per_second(10, compile_plus_completion),
        )

    def test_inspect_precision_flags_quant_config(self) -> None:
        script = load_script("mlx_qwen_stream.py")

        class Dummy:
            def parameters(self):
                return {}

        precision, quantized = script.inspect_precision(
            Dummy(), {"quantization": {"bits": 4, "group_size": 64}}
        )
        self.assertTrue(quantized)
        self.assertIn("4", precision)


if __name__ == "__main__":
    unittest.main()
