#!/usr/bin/env python3
"""Run Stories parity across Torch-from-Espresso weights, Espresso, and Core ML."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from espresso_llama_weights import load_espresso_llama_for_causal_lm
from stories_model_identity import EspressoSentencePieceTokenizer


DEFAULT_WEIGHTS_DIR = Path.home() / "Library/Application Support/Espresso/demo/stories110m"
DEFAULT_TOKENIZER_DIR = DEFAULT_WEIGHTS_DIR
DEFAULT_PROMPT_SUITE = REPO_ROOT / "scripts" / "stories_prompt_suite.txt"
DEFAULT_ESPRESSO_BIN = REPO_ROOT / "espresso"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    parser.add_argument("--coreml-model", required=True, help="Core ML .mlpackage to compare")
    parser.add_argument("--prompt-suite", default=str(DEFAULT_PROMPT_SUITE))
    parser.add_argument("--espresso-bin", default=str(DEFAULT_ESPRESSO_BIN))
    parser.add_argument("--compute-units", default="cpu_only")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compare-warmup", type=int, default=0)
    parser.add_argument("--compare-iterations", type=int, default=1)
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def load_prompt_suite(path: Path) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid prompt suite line {line_number}: expected id:prompt_text")
        prompt_id, prompt_text = line.split(":", 1)
        prompt_id = prompt_id.strip()
        prompt_text = bytes(prompt_text.strip(), "utf-8").decode("unicode_escape")
        if not prompt_id or not prompt_text:
            raise ValueError(f"Invalid prompt suite line {line_number}: empty id or prompt")
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id in suite: {prompt_id}")
        seen_ids.add(prompt_id)
        prompts.append((prompt_id, prompt_text))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def greedy_generate_torch(
    model,
    prompt_tokens: list[int],
    max_tokens: int,
) -> list[int]:
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    generated: list[int] = []
    with torch.inference_mode():
        for _ in range(max_tokens):
            outputs = model(input_ids=input_ids, use_cache=False)
            next_token = int(outputs.logits[0, -1].argmax().item())
            generated.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            input_ids = torch.cat((input_ids, next_token_tensor), dim=1)
    return generated


def run_espresso_compare(
    espresso_bin: Path,
    weights_dir: Path,
    tokenizer_dir: Path,
    coreml_model: Path,
    seq_len: int,
    compute_units: str,
    max_tokens: int,
    seed: int,
    compare_warmup: int,
    compare_iterations: int,
    prompt: str,
) -> dict[str, object]:
    command = [
        str(espresso_bin),
        "compare",
        "--bench",
        "--json",
        "--model",
        "stories110m",
        "--weights",
        str(weights_dir),
        "--tokenizer",
        str(tokenizer_dir),
        "--coreml-model",
        str(coreml_model),
        "--coreml-seq-len",
        str(seq_len),
        "--coreml-compute-units",
        compute_units,
        "--compare-warmup",
        str(compare_warmup),
        "--compare-iterations",
        str(compare_iterations),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0",
        "--seed",
        str(seed),
        prompt,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def compare_results(
    tokenizer: EspressoSentencePieceTokenizer,
    prompt_tokens: list[int],
    torch_tokens: list[int],
    compare_payload: dict[str, object],
) -> dict[str, object]:
    espresso_tokens = compare_payload["espresso"]["generated_tokens"]
    coreml_tokens = compare_payload["coreml"]["generated_tokens"]
    torch_text = tokenizer.decode(prompt_tokens + torch_tokens)
    espresso_text = compare_payload["espresso"]["text"]
    coreml_text = compare_payload["coreml"]["text"]
    return {
        "torch": {
            "generated_tokens": torch_tokens,
            "text": torch_text,
        },
        "espresso": compare_payload["espresso"],
        "coreml": compare_payload["coreml"],
        "torch_matches_espresso_tokens": torch_tokens == espresso_tokens,
        "torch_matches_coreml_tokens": torch_tokens == coreml_tokens,
        "torch_matches_espresso_text": torch_text == espresso_text,
        "torch_matches_coreml_text": torch_text == coreml_text,
        "espresso_matches_coreml_tokens": compare_payload["token_match"],
        "espresso_matches_coreml_text": compare_payload["text_match"],
    }


def main() -> None:
    args = parse_args()
    weights_dir = Path(args.weights_dir).expanduser().resolve()
    tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    coreml_model = Path(args.coreml_model).expanduser().resolve()
    espresso_bin = Path(args.espresso_bin).expanduser().resolve()
    prompt_suite = load_prompt_suite(Path(args.prompt_suite).expanduser().resolve())
    tokenizer = EspressoSentencePieceTokenizer(tokenizer_dir / "tokenizer.model")
    _, model = load_espresso_llama_for_causal_lm(weights_dir, torch_dtype=torch.float16)

    prompt_reports = []
    for prompt_id, prompt_text in prompt_suite:
        prompt_tokens = tokenizer.encode(prompt_text)
        torch_tokens = greedy_generate_torch(model, prompt_tokens, args.max_tokens)
        compare_payload = run_espresso_compare(
            espresso_bin=espresso_bin,
            weights_dir=weights_dir,
            tokenizer_dir=tokenizer_dir,
            coreml_model=coreml_model,
            seq_len=args.seq_len,
            compute_units=args.compute_units,
            max_tokens=args.max_tokens,
            seed=args.seed,
            compare_warmup=args.compare_warmup,
            compare_iterations=args.compare_iterations,
            prompt=prompt_text,
        )
        prompt_reports.append(
            {
                "id": prompt_id,
                "prompt": prompt_text,
                "prompt_tokens": prompt_tokens,
                **compare_results(tokenizer, prompt_tokens, torch_tokens, compare_payload),
            }
        )

    payload = {
        "weights_dir": str(weights_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "coreml_model": str(coreml_model),
        "compute_units": args.compute_units,
        "seq_len": args.seq_len,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "all_torch_matches_espresso_tokens": all(
            report["torch_matches_espresso_tokens"] for report in prompt_reports
        ),
        "all_torch_matches_coreml_tokens": all(
            report["torch_matches_coreml_tokens"] for report in prompt_reports
        ),
        "all_torch_matches_espresso_text": all(
            report["torch_matches_espresso_text"] for report in prompt_reports
        ),
        "all_torch_matches_coreml_text": all(
            report["torch_matches_coreml_text"] for report in prompt_reports
        ),
        "all_espresso_matches_coreml_tokens": all(
            report["espresso_matches_coreml_tokens"] for report in prompt_reports
        ),
        "all_espresso_matches_coreml_text": all(
            report["espresso_matches_coreml_text"] for report in prompt_reports
        ),
        "prompts": prompt_reports,
    }
    print(json.dumps(payload, indent=args.json_indent, sort_keys=True))


if __name__ == "__main__":
    main()
