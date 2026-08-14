#!/usr/bin/env python3
"""PyTorch reference oracle for Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct.

Two subcommands:

  layer-parity   Capture per-layer hidden states from the fp32 PyTorch model, run the
                 same layer inputs through Espresso's CPU oracle (and optionally the ANE
                 hybrid kernel) via `swift run EspressoQwenParity`, and write a per-layer
                 max/mean absolute difference report.

  fixtures       Greedy-decode a fixed prompt suite and write the reference token IDs as
                 a JSON fixture for the hardware-gated exact-match test.

`--model` selects the checkpoint. Source and native directories default to the same
cache slugs as `scripts/convert_qwen25_05b_to_esp.py` (and prefer a complete Hugging
Face hub snapshot when one is present).

The reference deliberately uses HuggingFace `transformers` rather than a hand-written
Qwen2 forward pass: the point of an oracle is that it is independently trustworthy.
Per-layer states are captured with forward hooks at decoder layer boundaries
(pre-final-norm). HuggingFace `output_hidden_states=True` is not used: its last entry
is emitted after the model's final norm, which is not a layer boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import convert_qwen25_05b_to_esp as convert

VENV_TAG = "qwen25-parity"
REQUIREMENTS = ("torch", "transformers", "safetensors", "numpy")
DEFAULT_MODEL = convert.DEFAULT_MODEL


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


def model_short_name(model: str) -> str:
    return convert.SUPPORTED_MODELS[model].short_name


def parse_layer_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    layers = [int(entry.strip()) for entry in raw.split(",") if entry.strip()]
    if not layers:
        raise ReferenceError("--layers must name at least one layer")
    return layers


def resolved_reference_paths(
    model: str,
    source_dir: str | None,
    native_dir: str | None,
    force_hub_lookup: bool = True,
) -> tuple[Path, Path]:
    """Resolve checkpoint and native-dir paths for `--model`.

    Explicit `--source-dir` / `--native-dir` win. Otherwise native dir follows the
    converter's cache slug, and the source prefers a complete Hugging Face snapshot
    when `force_hub_lookup` is true.
    """
    profile = convert.SUPPORTED_MODELS[model]
    cache_root = espresso_cache_root()
    paths = convert.default_paths(profile, cache_root)
    if source_dir:
        resolved_source = Path(source_dir).expanduser()
    elif force_hub_lookup:
        resolved_source = convert.resolve_source_dir(
            repo=profile.repo,
            explicit=None,
            cache_root=cache_root,
            source_slug=profile.source_slug,
            force_download=False,
        )
    else:
        resolved_source = paths.source
    resolved_native = Path(native_dir).expanduser() if native_dir else paths.native
    return resolved_source, resolved_native


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


def write_float32(path: Path, values: np.ndarray) -> None:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes())


def read_float32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
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
    chain: bool = False,
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
    if chain:
        command.append("--chain")
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

    source_dir, native_dir = resolved_reference_paths(
        args.model, args.source_dir, args.native_dir
    )
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

    requested = parse_layer_list(args.layers)
    layers = requested if requested is not None else list(range(n_layer))
    for layer in layers:
        if layer < 0 or layer >= n_layer:
            raise ReferenceError(f"layer {layer} out of range for nLayer {n_layer}")
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
        "reference": "pytorch fp32 (transformers, forward hooks at decoder layer boundaries, pre-final-norm)",
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

    if args.gate_layer0_max_abs is not None or args.gate_layer0_max_rel is not None:
        for backend, rows in measurements.items():
            layer0_rows = [row for row in rows if row["layer"] == 0]
            if not layer0_rows:
                continue
            layer0 = layer0_rows[0]
            if args.gate_layer0_max_abs is not None and layer0["max_abs"] > args.gate_layer0_max_abs:
                print(
                    f"[parity] FAIL {backend}: layer 0 max abs diff {layer0['max_abs']:.6e} "
                    f"exceeds gate {args.gate_layer0_max_abs:.6e}",
                    file=sys.stderr,
                )
                return 1
            if (
                args.gate_layer0_max_rel is not None
                and layer0["max_rel_to_layer_scale"] > args.gate_layer0_max_rel
            ):
                print(
                    f"[parity] FAIL {backend}: layer 0 relative diff "
                    f"{layer0['max_rel_to_layer_scale']:.6e} exceeds gate "
                    f"{args.gate_layer0_max_rel:.6e}",
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


def greedy_generate(model, prompt_tokens: list[int], max_new_tokens: int, eos_ids: set[int]):
    """Pure greedy decoding: argmax over raw logits, no logits processors.

    `model.generate` is deliberately avoided. Qwen2.5 ships `repetition_penalty: 1.1` in
    `generation_config.json`, which `generate` applies even with `do_sample=False`, so its
    output is not the greedy argmax sequence an inference runtime reproduces. Getting this
    wrong makes a correct runtime look broken.

    HuggingFace's incremental `past_key_values` is used so 1.5B (28 layers) can emit a
    32-token suite in minutes rather than hours. This is still an explicit argmax over
    raw logits — not `model.generate`, which would apply repetition_penalty=1.1.
    """
    import torch

    tokens = list(prompt_tokens)
    produced: list[int] = []
    top_gaps: list[float] = []
    runner_ups: list[int] = []
    past = None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            step_input = tokens if past is None else [tokens[-1]]
            output = model(
                input_ids=torch.tensor([step_input], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
        past = output.past_key_values
        final = output.logits[0, -1]
        top2 = torch.topk(final, 2)
        next_token = int(top2.indices[0])
        top_gaps.append(float(top2.values[0] - top2.values[1]))
        runner_ups.append(int(top2.indices[1]))
        produced.append(next_token)
        tokens.append(next_token)
        if next_token in eos_ids:
            break
    return produced, top_gaps, runner_ups


def rms_norm(values: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    import numpy as np

    scale = np.sqrt((values.astype(np.float64) ** 2).mean() + eps)
    return (values / scale).astype(np.float32) * gamma


def read_blob_fp16(path: Path, expected_count: int) -> np.ndarray:
    """Reads an Espresso BLOBFILE (128-byte header + fp16 payload) as float32."""
    import numpy as np

    raw = path.read_bytes()
    values = np.frombuffer(raw[128:], dtype="<f2").astype(np.float32)
    if values.size != expected_count:
        raise ReferenceError(
            f"{path} holds {values.size} fp16 values, expected {expected_count}"
        )
    return values


def command_logit_parity(args: argparse.Namespace) -> int:
    """Compares chained probe hidden states + a NumPy LM head against PyTorch fp32.

    This is not the served `cpu_fp16_tiled` generate classifier. The Swift driver runs
    in `--chain` mode so layer N+1 consumes layer N's output; logits are then
    `lm_head @ rms_norm(final_hidden)` in NumPy.
    """
    import numpy as np
    import torch

    source_dir, native_dir = resolved_reference_paths(
        args.model, args.source_dir, args.native_dir
    )
    metadata = json.loads((native_dir / "metadata.json").read_text(encoding="utf-8"))
    d_model = int(metadata["dModel"])
    vocab = int(metadata["vocab"])
    norm_eps = float(metadata["normEps"])

    fixture = json.loads(Path(args.fixture).expanduser().read_text(encoding="utf-8"))
    cases = fixture["cases"]
    if args.cases:
        cases = [case for case in cases if case["index"] in set(args.cases)]
    if not cases:
        raise ReferenceError("no fixture cases selected")

    _, model = load_reference_model(source_dir)
    embeddings = model.model.embed_tokens.weight.detach().numpy()
    gamma = read_blob_fp16(native_dir / "final_norm.bin", d_model)
    lm_head = read_blob_fp16(native_dir / "lm_head.bin", vocab * d_model).reshape(vocab, d_model)

    work_dir = Path(args.work_dir).expanduser() if args.work_dir else native_dir.parent / "parity-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = work_dir / "chain_inputs.f32"

    rows = []
    for case in cases:
        token_ids = case["promptTokens"]
        positions = len(token_ids)
        write_float32(inputs_path, embeddings[token_ids])
        with torch.no_grad():
            reference_logits = (
                model(input_ids=torch.tensor([token_ids], dtype=torch.long), use_cache=False)
                .logits[0, -1]
                .numpy()
            )
        order = np.argsort(-reference_logits)
        reference_gap = float(reference_logits[order[0]] - reference_logits[order[1]])

        for backend in args.backends:
            output_path = work_dir / f"chain_outputs_{backend.replace('-', '_')}.f32"
            run_swift_driver(
                native_dir=native_dir,
                inputs_path=inputs_path,
                output_path=output_path,
                positions=positions,
                backend=backend,
                layers=list(range(int(metadata["nLayer"]))),
                chain=True,
                extra_env={"ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK": "1"} if backend == "ane" else None,
            )
            final_hidden = read_float32(output_path, (positions, d_model))[-1]
            logits = lm_head @ rms_norm(final_hidden, gamma, norm_eps)
            rows.append(
                {
                    "case": case["index"],
                    "backend": backend,
                    "max_abs_logit_diff": float(np.abs(logits - reference_logits).max()),
                    "argmax": int(logits.argmax()),
                    "reference_argmax": int(reference_logits.argmax()),
                    "reference_top_gap": reference_gap,
                }
            )
            print(
                f"[logit-parity] case {case['index']} {backend}: "
                f"max|dlogit|={rows[-1]['max_abs_logit_diff']:.4f} "
                f"argmax={rows[-1]['argmax']} ref={rows[-1]['reference_argmax']} "
                f"ref_gap={reference_gap:.4f}",
                flush=True,
            )

    summary = {
        backend: max(row["max_abs_logit_diff"] for row in rows if row["backend"] == backend)
        for backend in args.backends
    }
    for backend, worst in summary.items():
        print(f"[logit-parity] {backend}: worst max|dlogit| = {worst:.4f}", flush=True)
    argmax_agreement = sum(1 for row in rows if row["argmax"] == row["reference_argmax"])
    print(f"[logit-parity] argmax agreement {argmax_agreement}/{len(rows)}", flush=True)

    if args.report_json:
        path = Path(args.report_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"rows": rows, "worstMaxAbsLogitDiff": summary}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[logit-parity] wrote {path}", flush=True)
    return 0


def command_fixtures(args: argparse.Namespace) -> int:
    source_dir, _native_dir = resolved_reference_paths(
        args.model, args.source_dir, args.native_dir
    )
    tokenizer, model = load_reference_model(source_dir)
    prompts = load_prompts(Path(args.prompts).expanduser())
    if len(prompts) < args.min_prompts:
        raise ReferenceError(
            f"prompt suite has {len(prompts)} prompts, need at least {args.min_prompts}"
        )

    # Match the runtime's stop condition exactly. The artifact declares a single EOS in
    # metadata.json, so stopping on Qwen's wider generation-config EOS set would make the
    # reference stop somewhere the runtime does not.
    eos_ids = {int(token) for token in args.eos_token_ids} if args.eos_token_ids else set()

    cases = []
    for index, prompt in enumerate(prompts):
        prompt_tokens = encode_prompt(tokenizer, prompt, use_chat_template=not args.raw_prompt)
        completion, top_gaps, runner_ups = greedy_generate(
            model, prompt_tokens, args.max_new_tokens, eos_ids
        )
        cases.append(
            {
                "index": index,
                "prompt": prompt,
                "promptTokens": prompt_tokens,
                "expectedTokens": completion,
                "expectedText": tokenizer.decode(completion, skip_special_tokens=True),
                "stoppedOnEOS": bool(completion) and completion[-1] in eos_ids,
                # Per-step top-1/top-2 logit gap and runner-up token. A flip at a step whose
                # gap is below the runtime's logit error, and whose replacement is exactly the
                # reference runner-up, is arithmetic precision rather than a wrong result.
                "topLogitGaps": top_gaps,
                "runnerUpTokens": runner_ups,
                "minTopLogitGap": min(top_gaps) if top_gaps else None,
            }
        )
        print(
            f"[fixtures] {index}: {len(prompt_tokens)} prompt + {len(completion)} generated "
            f"tokens, min top-1/top-2 gap {min(top_gaps):.4f}",
            flush=True,
        )

    fixture = {
        "model": model_short_name(args.model),
        "reference": "pytorch fp32 pure greedy argmax over raw logits (no logits processors)",
        "chatTemplate": not args.raw_prompt,
        "maxNewTokens": args.max_new_tokens,
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(convert.SUPPORTED_MODELS),
        help=(
            "Hugging Face repo whose cache slugs resolve source/native paths "
            f"(default: {DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Fail instead of creating a managed venv when torch/transformers are missing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parity = subparsers.add_parser("layer-parity", help="Write the per-layer parity report")
    parity.add_argument(
        "--source-dir",
        default=None,
        help="Override the resolved Hugging Face / cache source directory",
    )
    parity.add_argument(
        "--native-dir",
        default=None,
        help="Override the resolved native weight directory",
    )
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
        "--layers",
        default=None,
        help="Comma-separated layer indices to measure (default: all)",
    )
    parity.add_argument(
        "--gate-layer0-max-abs",
        type=float,
        default=None,
        help="Exit non-zero when layer 0 max abs diff exceeds this value",
    )
    parity.add_argument(
        "--gate-layer0-max-rel",
        type=float,
        default=None,
        help="Exit non-zero when layer 0 max diff / layer scale exceeds this value",
    )
    parity.set_defaults(func=command_layer_parity)

    logits = subparsers.add_parser(
        "logit-parity", help="Compare final logits through the whole stack against PyTorch fp32"
    )
    logits.add_argument(
        "--source-dir",
        default=None,
        help="Override the resolved Hugging Face / cache source directory",
    )
    logits.add_argument(
        "--native-dir",
        default=None,
        help="Override the resolved native weight directory",
    )
    logits.add_argument("--work-dir", default=None)
    logits.add_argument(
        "--fixture",
        default=str(
            REPO_ROOT / "Tests/RealModelInferenceTests/Fixtures/qwen25-05b-greedy-reference.json"
        ),
    )
    logits.add_argument(
        "--cases", type=int, nargs="*", default=None, help="Fixture case indices (default: all)"
    )
    logits.add_argument(
        "--backends", nargs="+", default=["cpu-fp32", "ane"], choices=["cpu-fp32", "cpu-fp16", "ane"]
    )
    logits.add_argument("--report-json", default=None)
    logits.set_defaults(func=command_logit_parity)

    fixtures = subparsers.add_parser("fixtures", help="Write greedy reference token fixtures")
    fixtures.add_argument(
        "--source-dir",
        default=None,
        help="Override the resolved Hugging Face / cache source directory",
    )
    fixtures.add_argument(
        "--native-dir",
        default=None,
        help="Unused for fixtures; accepted so --model path resolution stays uniform",
    )
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
        "--eos-token-ids",
        type=int,
        nargs="*",
        default=[151645],
        help=(
            "Token IDs that stop generation. Must match the artifact's metadata.json eosToken "
            "so the reference stops exactly where the runtime does (default: 151645)"
        ),
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
