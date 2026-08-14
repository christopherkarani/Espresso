import unittest

from pathlib import Path
import importlib.util


def load_script():
    path = Path(__file__).resolve().parents[1] / "chat_plain_name_recall.py"
    spec = importlib.util.spec_from_file_location("chat_plain_name_recall", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ChatPlainSmokeTests(unittest.TestCase):
    def test_ten_turns_introduce_ada_then_ask_name(self) -> None:
        script = load_script()
        self.assertEqual(len(script.TURNS), 10)
        self.assertEqual(script.TURNS[0], "my name is Ada")
        self.assertEqual(script.TURNS[script.RECALL_TURN_INDEX], "what is my name?")

    def test_extract_assistant_replies_splits_plain_stream(self) -> None:
        script = load_script()
        stdout = """
espresso chat  model=Qwen2.5-1.5B-Instruct  sampling=greedy  fallback=disabled
you> my name is Ada
qwen> Hello Ada.
tok/s 8.0  TTFT 90ms  path=hybrid  ctx 40/1024
ANE 1.60W  CPU 0.85W  pkg 3.25W  0.406 J/tok
you> what is my name?
qwen> Your name is Ada.
tok/s 7.2  TTFT 80ms  path=hybrid  ctx 80/1024
power: unavailable (sudo)
"""
        replies = script.extract_assistant_replies(stdout)
        self.assertEqual(replies, ["Hello Ada.", "Your name is Ada."])


if __name__ == "__main__":
    unittest.main()
