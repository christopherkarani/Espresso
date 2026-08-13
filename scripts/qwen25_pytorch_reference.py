#!/usr/bin/env python3
"""PyTorch reference oracle for Qwen2.5-0.5B-Instruct parity work.

Two subcommands:

  layer-parity   Capture per-layer hidden states from the fp32 PyTorch model, run the
                 same layer inputs through Espresso's CPU oracle (and optionally the ANE
                 hybrid kernel) via `swift run EspressoQwenParity`, and write a per-layer
                 max/mean absolute difference report.

  fixtures       Greedy-decode a fixed prompt suite and write the reference token IDs as
                 a JSON fixture for the hardware-gated exact-match test.

Both subcommands read the safetensors checkpoint cached by
`scripts/convert_qwen25_05b_to_esp.py`, so no second download happens.

The reference deliberately uses HuggingFace `transformers` rather than a hand-written
Qwen2 forward pass: the point of an oracle is that it is independently trustworthy, and
`output_hidden_states=True` already exposes exactly the per-layer boundaries we need
(`hidden_states[L]` enters layer `L`, `hidden_states[L + 1]` leaves it).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_TAG = "qwen25-parity"
REQUIREMENTS = ("torch", "transformers", "safetensors", "numpy")


class ReferenceError(RuntimeError):
    """Raised for unrecoverable problems, printed without a traceback."""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def espresso_state_root() -> Path:
    override = os.environ.get("ESPRESSO_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Application Support" / "Espresso"


def espresso_cache_root() -> Path:
    override = os.environ.get("ESPRESSO_CACHE_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Caches" / "Espresso"


def _missing_requirements() -> list[str]:
    import importlib.util

    return [name for name in REQUIREMENTS if importlib.util.find_spec(name) is None]


def ensure_runtime_dependencies(allow_bootstrap: bool) -> None:
    """Re-exec inside a managed venv when torch/transformers are unavailable.

    Mirrors the managed-venv convention used by the converter:
    `<state root>/tools/python/<tag>-<pyver>`.
    """
    missing = _missing_requirements()
    if not missing:
        return

    if os.environ.get("ESPRESSO_QWEN_PARITY_BOOTSTRAPPED") == "1":
        raise ReferenceError(
            f"still missing {', '.join(missing)} after bootstrapping a managed venv. "
            "Install them manually and re-run."
        )
    if not allow_bootstrap:
        raise ReferenceError(
            f"missing Python packages: {', '.join(missing)}. "
            "Re-run without --no-bootstrap to create a managed venv."
        )

    version_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
    venv_dir = espresso_state_root() / "tools" / "python" / f"{VENV_TAG}-{version_tag.replace('.', '_')}"
    venv_python = venv_dir / "bin" / "python3"
    uv = shutil.which("uv")

    if not venv_python.exists():
        print(f"[bootstrap] creating managed venv at {venv_dir}", flush=True)
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        if uv:
            subprocess.run([uv, "venv", "--python", version_tag, str(venv_dir)], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    print(f"[bootstrap] installing {', '.join(REQUIREMENTS)} (this can take a few minutes)", flush=True)
    if uv:
        subprocess.run([uv, "pip", "install", "--python", str(venv_python), *REQUIREMENTS], check=True)
    else:
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install", *REQUIREMENTS], check=True)

    print(f"[bootstrap] re-executing under {venv_python}", flush=True)
    env = dict(os.environ)
    env["ESPRESSO_QWEN_PARITY_BOOTSTRAPPED"] = "1"
    os.execve(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


# ---------------------------------------------------------------------------
# Model + prompts
# ---------------------------------------------------------------------------


def load_reference_model(source_dir: Path):
    """Loads the checkpoint in fp32 on CPU with deterministic settings."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (source_dir / "model.safetensors").exists():
        raise ReferenceError(
            f"{source_dir} has no model.safetensors. "
            "Run scripts/convert_qwen25_05b_to_esp.py first to download the checkpoint."
        )

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(str(source_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(source_dir),
        dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise ReferenceError(f"prompt file {path} not found")
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not prompts:
        raise ReferenceError(f"prompt file {path} contains no prompts")
    return prompts


def encode_prompt(tokenizer, prompt: str, use_chat_template: bool) -> list[int]:
    """Tokenizes a prompt, optionally through Qwen's instruct chat template."""
    if not use_chat_template:
        return tokenizer(prompt, add_special_tokens=False)["input_ids"]
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------------------
# Binary float32 I/O shared with the Swift driver
# ---------------------------------------------------------------------------


def write_float32(path: Path, values) -> None:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes())


def read_float32(path: Path, shape) -> "object":
    import numpy as np

    expected = 1
    for dim in shape:
        expected *= dim
    raw = path.read_bytes()
    if len(raw) != expected * 4:
        raise ReferenceError(
            f"{path} has {len(raw)} bytes, expected {expected * 4} ({expected} float32)"
        )
    return np.frombuffer(raw, dtype="<f4").reshape(shape)


# ---------------------------------------------------------------------------
# layer-parity
# ---------------------------------------------------------------------------


def run_swift_driver(
    native_dir: Path,
    inputs_path: Path,
    output_path: Path,
    positions: int,
    backend: str,
    layers: list[int],
    extra_env: dict[str, str] | None = None,
) -> None:
    command = [
        "swift",
        "run",
        "--package-path",
        str(REPO_ROOT),
        "EspressoQwenParity",
        "--native-dir",
        str(native_dir),
        "--inputs",
        str(inputs_path),
        "--out",
        str(output_path),
        "--positions",
        str(positions),
        "--backend",
        backend,
        "--layers",
        ",".join(str(layer) for layer in layers),
    ]
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    print(f"[parity] {backend}: {' '.join(command[-8:])}", flush=True)
    result = subprocess.run(command, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        raise ReferenceError(
            f"EspressoQwenParity failed for backend {backend} with exit code {result.returncode}"
        )


def capture_layer_boundaries(model, token_ids: list[int], n_layer: int):
    """Records the hidden states entering and leaving every decoder layer.

    Forward hooks are used rather than `output_hidden_states=True` because the final entry
    of that tuple is emitted *after* the model's final norm, which would misreport the last
    layer. Hooks read the real layer boundary regardless of transformers version.
    """
    import numpy as np
    import torch

    layers = model.model.layers
    if len(layers) != n_layer:
        raise ReferenceError(f"model exposes {len(layers)} decoder layers, expected {n_layer}")

    inputs: dict[int, "np.ndarray"] = {}
    outputs: dict[int, "np.ndarray"] = {}

    def make_hook(index: int):
        def hook(_module, args, kwargs, output):
            hidden = kwargs.get("hidden_states") if kwargs else None
            if hidden is None:
                if not args:
                    raise ReferenceError(f"layer {index} hook saw no hidden_states argument")
                hidden = args[0]
            produced = output[0] if isinstance(output, (tuple, list)) else output
            inputs[index] = hidden[0].detach().to(torch.float32).numpy().copy()
            outputs[index] = produced[0].detach().to(torch.float32).numpy().copy()

        return hook

    handles = [
        layer.register_forward_hook(make_hook(index), with_kwargs=True)
        for index, layer in enumerate(layers)
    ]
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([token_ids], dtype=torch.long), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [index for index in range(n_layer) if index not in inputs or index not in outputs]
    if missing:
        raise ReferenceError(f"no hidden states captured for layers {missing}")
    return inputs, outputs


def command_layer_parity(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    source_dir = Path(args.source_dir).expanduser()
    native_dir = Path(args.native_dir).expanduser()
    metadata = json.loads((native_dir / "metadata.json").read_text(encoding="utf-8"))
    n_layer = int(metadata["nLayer"])
    d_model = int(metadata["dModel"])

    tokenizer, model = load_reference_model(source_dir)
    if model.config.num_hidden_layers != n_layer or model.config.hidden_size != d_model:
        raise ReferenceError(
            "native metadata disagrees with the checkpoint: "
            f"metadata nLayer={n_layer} dModel={d_model} vs config "
            f"nLayer={model.config.num_hidden_layers} dModel={model.config.hidden_size}"
        )

    token_ids = encode_prompt(tokenizer, args.prompt, use_chat_template=not args.raw_prompt)
    if args.max_positions and len(token_ids) > args.max_positions:
        token_ids = token_ids[: args.max_positions]
    positions = len(token_ids)
    if positions < 2:
        raise ReferenceError("layer parity needs at least 2 token positions")

    print(f"[parity] prompt tokens: {positions}", flush=True)
    captured_inputs, captured_outputs = capture_layer_boundaries(
        model, token_ids, n_layer=n_layer
    )

    layers = list(range(n_layer))
    work_dir = Path(args.work_dir).expanduser() if args.work_dir else native_dir.parent / "parity-work"
    work_dir.mkdir(parents=True, exist_ok=True)

    reference_inputs = np.stack([captured_inputs[layer] for layer in layers])
    reference_outputs = np.stack([captured_outputs[layer] for layer in layers])
    inputs_path = work_dir / "layer_inputs.f32"
    write_float32(inputs_path, reference_inputs)
    write_float32(work_dir / "layer_reference_outputs.f32", reference_outputs)

    backends = list(args.backends)
    measurements: dict[str, list[dict[str, float]]] = {}
    for backend in backends:
        output_path = work_dir / f"layer_outputs_{backend.replace('-', '_')}.f32"
        run_swift_driver(
            native_dir=native_dir,
            inputs_path=inputs_path,
            output_path=output_path,
            positions=positions,
            backend=backend,
            layers=layers,
            extra_env={"ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK": "1"} if backend == "ane" else None,
        )
        actual = read_float32(output_path, (len(layers), positions, d_model))
        rows = []
        for slot, layer in enumerate(layers):
            diff = np.abs(actual[slot].astype(np.float64) - reference_outputs[slot].astype(np.float64))
            scale = float(np.abs(reference_outputs[slot]).max()) or 1.0
            rows.append(
                {
                    "layer": layer,
                    "max_abs": float(diff.max()),
                    "mean_abs": float(diff.mean()),
                    "max_rel_to_layer_scale": float(diff.max() / scale),
                    "reference_max_abs": scale,
                }
            )
        measurements[backend] = rows

    report = {
        "model": metadata["name"],
        "nLayer": n_layer,
        "dModel": d_model,
        "positions": positions,
        "promptTokens": token_ids,
        "prompt": args.prompt,
        "chatTemplate": not args.raw_prompt,
        "reference": "pytorch fp32 (transformers, output_hidden_states)",
        "weights": "fp16 blobs converted from bf16 (lossless for this checkpoint)",
        "backends": measurements,
    }
    report_json = Path(args.report_json).expanduser() if args.report_json else work_dir / "layer-parity.json"
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[parity] wrote {report_json}", flush=True)

    if args.report_markdown:
        markdown_path = Path(args.report_markdown).expanduser()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"[parity] wrote {markdown_path}", flush=True)

    worst = {backend: max(row["max_abs"] for row in rows) for backend, rows in measurements.items()}
    for backend, value in worst.items():
        print(f"[parity] {backend}: worst layer max abs diff = {value:.6e}", flush=True)

    if args.gate_layer0_max_abs is not None:
        for backend, rows in measurements.items():
            layer0 = rows[0]["max_abs"]
            if layer0 > args.gate_layer0_max_abs:
                print(
                    f"[parity] FAIL {backend}: layer 0 max abs diff {layer0:.6e} "
                    f"exceeds gate {args.gate_layer0_max_abs:.6e}",
                    file=sys.stderr,
                )
                return 1
    return 0


def render_markdown(report: dict) -> str:
    lines = [
        f"# Per-layer parity: {report['model']}",
        "",
        f"- Reference: {report['reference']}",
        f"- Espresso weights: {report['weights']}",
        f"- Prompt positions: {report['positions']} (chat template: {report['chatTemplate']})",
        "",
        "Each layer is fed the **reference** input hidden states for every position, so",
        "errors are measured per layer instead of compounding across the stack.",
        "",
    ]
    for backend, rows in report["backends"].items():
        lines += [
            f"## Backend `{backend}`",
            "",
            "| layer | max abs diff | mean abs diff | max diff / layer scale |",
            "| ----: | -----------: | ------------: | ---------------------: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['layer']} | {row['max_abs']:.3e} | {row['mean_abs']:.3e} "
                f"| {row['max_rel_to_layer_scale']:.3e} |"
            )
        worst = max(rows, key=lambda row: row["max_abs"])
        lines += [
            "",
            f"Worst layer: {worst['layer']} at {worst['max_abs']:.3e} max abs diff.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def command_fixtures(args: argparse.Namespace) -> int:
    import torch

    source_dir = Path(args.source_dir).expanduser()
    tokenizer, model = load_reference_model(source_dir)
    prompts = load_prompts(Path(args.prompts).expanduser())
    if len(prompts) < args.min_prompts:
        raise ReferenceError(
            f"prompt suite has {len(prompts)} prompts, need at least {args.min_prompts}"
        )

    eos_ids = set()
    if model.generation_config.eos_token_id is not None:
        raw_eos = model.generation_config.eos_token_id
        eos_ids = set(raw_eos) if isinstance(raw_eos, (list, tuple)) else {int(raw_eos)}

    cases = []
    for index, prompt in enumerate(prompts):
        prompt_tokens = encode_prompt(tokenizer, prompt, use_chat_template=not args.raw_prompt)
        with torch.no_grad():
            generated = model.generate(
                input_ids=torch.tensor([prompt_tokens], dtype=torch.long),
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens if args.forbid_early_stop else None,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = generated[0][len(prompt_tokens):].tolist()
        stopped_on_eos = bool(completion) and completion[-1] in eos_ids
        cases.append(
            {
                "index": index,
                "prompt": prompt,
                "promptTokens": prompt_tokens,
                "expectedTokens": completion,
                "expectedText": tokenizer.decode(completion, skip_special_tokens=True),
                "stoppedOnEOS": stopped_on_eos,
            }
        )
        print(
            f"[fixtures] {index}: {len(prompt_tokens)} prompt + {len(completion)} generated tokens",
            flush=True,
        )

    fixture = {
        "model": "Qwen2.5-0.5B-Instruct",
        "reference": "pytorch fp32 greedy (transformers generate, do_sample=False, num_beams=1)",
        "chatTemplate": not args.raw_prompt,
        "maxNewTokens": args.max_new_tokens,
        "forbidEarlyStop": args.forbid_early_stop,
        "eosTokenIds": sorted(eos_ids),
        "cases": cases,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"[fixtures] wrote {output} ({len(cases)} cases)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_source_dir() -> Path:
    return espresso_cache_root() / "qwen25-05b-src"


def default_native_dir() -> Path:
    return espresso_cache_root() / "qwen25-05b" / "Qwen2.5-0.5B-Instruct-native"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Fail instead of creating a managed venv when torch/transformers are missing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parity = subparsers.add_parser("layer-parity", help="Write the per-layer parity report")
    parity.add_argument("--source-dir", default=str(default_source_dir()))
    parity.add_argument("--native-dir", default=str(default_native_dir()))
    parity.add_argument("--work-dir", default=None, help="Where to stage .f32 intermediates")
    parity.add_argument(
        "--prompt",
        default="Explain what a neural engine is in two sentences.",
        help="Prompt whose hidden states drive the comparison",
    )
    parity.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Tokenize the prompt verbatim instead of through the instruct chat template",
    )
    parity.add_argument("--max-positions", type=int, default=32, help="Truncate the prompt (0 = keep all)")
    parity.add_argument(
        "--backends",
        nargs="+",
        default=["cpu-fp32"],
        choices=["cpu-fp32", "cpu-fp16", "ane"],
        help="Espresso backends to measure",
    )
    parity.add_argument("--report-json", default=None)
    parity.add_argument("--report-markdown", default=None)
    parity.add_argument(
        "--gate-layer0-max-abs",
        type=float,
        default=None,
        help="Exit non-zero when layer 0 max abs diff exceeds this value",
    )
    parity.set_defaults(func=command_layer_parity)

    fixtures = subparsers.add_parser("fixtures", help="Write greedy reference token fixtures")
    fixtures.add_argument("--source-dir", default=str(default_source_dir()))
    fixtures.add_argument("--prompts", default=str(REPO_ROOT / "scripts" / "qwen25_prompts.txt"))
    fixtures.add_argument("--output", default=None, required=True)
    fixtures.add_argument("--max-new-tokens", type=int, default=32)
    fixtures.add_argument("--min-prompts", type=int, default=8)
    fixtures.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Tokenize prompts verbatim instead of through the instruct chat template",
    )
    fixtures.add_argument(
        "--forbid-early-stop",
        action="store_true",
        help="Force exactly --max-new-tokens tokens so every case exercises the full horizon",
    )
    fixtures.set_defaults(func=command_fixtures)

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ensure_runtime_dependencies(allow_bootstrap=not args.no_bootstrap)
    return int(args.func(args))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ReferenceError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
