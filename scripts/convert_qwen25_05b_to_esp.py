#!/usr/bin/env python3
"""Convert Qwen2.5-0.5B-Instruct safetensors into an Espresso `.esp` bundle.

Qwen2.5-0.5B-Instruct is Apache-2.0 and ungated, so the source weights are
fetched over plain HTTPS without any HuggingFace token.

The model maps onto Espresso's llama-family layer layout with one addition that
plain llama does not have: Qwen2 carries a bias on q/k/v (and only on q/k/v).
Those land as `bq.bin` / `bk.bin` / `bv.bin` next to the projections.

One-command usage (bootstraps its own Python env when numpy is missing):

    ./scripts/convert_qwen25_05b_to_esp.py --output /tmp/qwen25-05b.esp

Outputs:
  * a native model directory (fp16 BLOBFILE tensors + metadata.json + tokenizer)
  * a packed `.esp` bundle produced by `swift run espc pack-native`
  * `conversion-report.json` describing every tensor that was written
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HF_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

# Files copied verbatim into the bundle's tokenizer/ directory. `espc pack-native`
# recognizes these names and splits them out of weights/ automatically.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
SOURCE_FILES = ("config.json", "generation_config.json", "model.safetensors", "LICENSE") + TOKENIZER_FILES

# BLOBFILE payload size is a UInt32, which caps a square causal mask.
MAX_BLOBFILE_DATA_SIZE = 0xFFFF_FFFF

BOOTSTRAP_GUARD_ENV = "ESPRESSO_QWEN_CONVERT_BOOTSTRAPPED"


class ConversionError(RuntimeError):
    """Raised for every failure this script can explain to the caller."""


# ---------------------------------------------------------------------------
# Dependency bootstrap
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


def ensure_runtime_dependencies(allow_bootstrap: bool) -> None:
    """Re-exec inside a managed venv when numpy is unavailable.

    Mirrors the managed-venv convention used by the GPT-2 demo bootstrap:
    `<state root>/tools/python/qwen25-tools-<pyver>`.
    """
    try:
        import numpy  # noqa: F401

        return
    except ImportError:
        pass

    if os.environ.get(BOOTSTRAP_GUARD_ENV) == "1":
        raise ConversionError(
            "numpy is still unavailable after bootstrapping a managed venv. "
            "Install numpy manually and re-run with --no-bootstrap."
        )
    if not allow_bootstrap:
        raise ConversionError(
            "numpy is required. Either install it for this interpreter or drop --no-bootstrap "
            "so a managed venv can be created."
        )

    version_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
    venv_dir = espresso_state_root() / "tools" / "python" / f"qwen25-tools-{version_tag}"
    venv_python = venv_dir / "bin" / "python3"

    if not venv_python.exists():
        print(f"[bootstrap] creating managed venv at {venv_dir}", flush=True)
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        uv = shutil.which("uv")
        if uv:
            subprocess.run([uv, "venv", "--python", version_tag, str(venv_dir)], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    print("[bootstrap] installing numpy", flush=True)
    uv = shutil.which("uv")
    if uv:
        subprocess.run(
            [uv, "pip", "install", "--python", str(venv_python), "numpy"],
            check=True,
        )
    else:
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install", "numpy"], check=True)

    print(f"[bootstrap] re-executing under {venv_python}", flush=True)
    env = dict(os.environ)
    env[BOOTSTRAP_GUARD_ENV] = "1"
    os.execve(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


# ---------------------------------------------------------------------------
# Source download
# ---------------------------------------------------------------------------


def download_source(source_dir: Path, force: bool) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        target = source_dir / name
        if target.exists() and target.stat().st_size > 0 and not force:
            continue
        url = f"{HF_BASE_URL}/{name}"
        print(f"[download] {name}", flush=True)
        tmp = target.with_suffix(target.suffix + ".partial")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1 << 20)
        except OSError as error:
            tmp.unlink(missing_ok=True)
            raise ConversionError(f"failed to download {url}: {error}") from error
        tmp.replace(target)
    return source_dir


# ---------------------------------------------------------------------------
# safetensors reading
# ---------------------------------------------------------------------------


@dataclass
class SafetensorsFile:
    path: Path
    header: dict
    data_offset: int

    @classmethod
    def open(cls, path: Path) -> "SafetensorsFile":
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ConversionError(f"{path} is too short to be a safetensors file")
            header_length = struct.unpack("<Q", raw_length)[0]
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise ConversionError(f"{path} has a truncated safetensors header")
        header = json.loads(header_bytes)
        header.pop("__metadata__", None)
        return cls(path=path, header=header, data_offset=8 + header_length)

    def tensor_names(self) -> list[str]:
        return sorted(self.header.keys())

    def read(self, name: str):
        """Return a float32 numpy array for `name`, widening bf16 losslessly."""
        import numpy as np

        entry = self.header.get(name)
        if entry is None:
            raise ConversionError(f"tensor {name!r} is missing from {self.path.name}")
        dtype = entry["dtype"]
        begin, end = entry["data_offsets"]
        shape = tuple(entry["shape"])
        with self.path.open("rb") as handle:
            handle.seek(self.data_offset + begin)
            payload = handle.read(end - begin)
        if len(payload) != end - begin:
            raise ConversionError(f"tensor {name!r} is truncated in {self.path.name}")

        if dtype == "BF16":
            # bf16 shares fp32's exponent field, so a 16-bit left shift is exact.
            raw = np.frombuffer(payload, dtype="<u2").astype(np.uint32) << 16
            values = raw.view(np.float32)
        elif dtype == "F16":
            values = np.frombuffer(payload, dtype="<f2").astype(np.float32)
        elif dtype == "F32":
            values = np.frombuffer(payload, dtype="<f4").astype(np.float32)
        else:
            raise ConversionError(f"tensor {name!r} has unsupported dtype {dtype}")

        expected = math.prod(shape) if shape else 1
        if values.size != expected:
            raise ConversionError(
                f"tensor {name!r} has {values.size} elements but shape {shape} implies {expected}"
            )
        return values.reshape(shape)


# ---------------------------------------------------------------------------
# BLOBFILE writing
# ---------------------------------------------------------------------------


def make_blob_header(data_size: int) -> bytes:
    if data_size > MAX_BLOBFILE_DATA_SIZE:
        raise ConversionError(f"blob payload {data_size} exceeds BLOBFILE UInt32 size field")
    header = bytearray(128)
    header[0] = 0x01
    header[4] = 0x02
    header[64:68] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
    header[68] = 0x01
    struct.pack_into("<I", header, 72, data_size)
    struct.pack_into("<I", header, 80, 128)
    return bytes(header)


@dataclass
class TensorRecord:
    relative_path: str
    source: str
    shape: list[int]
    count: int
    max_abs: float
    fp16_overflow: int
    fp16_max_abs_error: float


@dataclass
class ConversionStats:
    tensors: list[TensorRecord] = field(default_factory=list)

    def overflowing(self) -> list[TensorRecord]:
        return [record for record in self.tensors if record.fp16_overflow > 0]

    def worst_fp16_error(self) -> TensorRecord | None:
        if not self.tensors:
            return None
        return max(self.tensors, key=lambda record: record.fp16_max_abs_error)


def write_blob(
    values,
    path: Path,
    *,
    source: str,
    stats: ConversionStats,
    write_fp32_sidecar: bool,
) -> None:
    """Write `values` as an fp16 BLOBFILE and record the rounding cost."""
    import numpy as np

    array = np.ascontiguousarray(values, dtype=np.float32)
    half = array.astype(np.float16)
    # fp16 has a narrower exponent than bf16, so a large bf16 value can become inf.
    # That would be a silent accuracy cliff, so count it and report it.
    overflow = int(np.count_nonzero(~np.isfinite(half) & np.isfinite(array)))
    finite = np.isfinite(half)
    error = float(np.max(np.abs(half[finite].astype(np.float32) - array[finite]))) if finite.any() else 0.0

    payload = half.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(make_blob_header(len(payload)))
        handle.write(payload)

    if write_fp32_sidecar:
        sidecar = path.with_name(path.stem + ".float32.bin")
        with sidecar.open("wb") as handle:
            handle.write(array.tobytes())

    stats.tensors.append(
        TensorRecord(
            relative_path=str(path),
            source=source,
            shape=list(array.shape),
            count=int(array.size),
            max_abs=float(np.max(np.abs(array))) if array.size else 0.0,
            fp16_overflow=overflow,
            fp16_max_abs_error=error,
        )
    )


def max_supported_mask_sequence_length() -> int:
    return int(math.isqrt(MAX_BLOBFILE_DATA_SIZE // 2))


def write_causal_masks(output_dir: Path, max_seq: int) -> None:
    import numpy as np

    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    size = 1
    while size <= max_seq:
        mask = np.zeros((size, size), dtype=np.float16)
        mask[np.triu_indices(size, k=1)] = np.float16(-1e4)
        payload = mask.tobytes()
        with (mask_dir / f"causal_{size}.bin").open("wb") as handle:
            handle.write(make_blob_header(len(payload)))
            handle.write(payload)
        size *= 2


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


@dataclass
class QwenShape:
    name: str
    n_layer: int
    n_head: int
    n_kv_head: int
    d_model: int
    head_dim: int
    hidden_dim: int
    vocab: int
    norm_eps: float
    rope_theta: float
    eos_token: int
    tie_word_embeddings: bool
    max_position_embeddings: int

    @property
    def attention_dim(self) -> int:
        return self.n_head * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_head * self.head_dim


def read_shape(source_dir: Path) -> QwenShape:
    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    generation = {}
    generation_path = source_dir / "generation_config.json"
    if generation_path.exists():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))

    architectures = config.get("architectures") or []
    if "Qwen2ForCausalLM" not in architectures:
        raise ConversionError(
            f"expected a Qwen2ForCausalLM checkpoint, found architectures={architectures!r}"
        )

    n_head = int(config["num_attention_heads"])
    d_model = int(config["hidden_size"])
    head_dim = int(config.get("head_dim", d_model // n_head))

    eos = generation.get("eos_token_id", config.get("eos_token_id"))
    if isinstance(eos, list):
        eos = eos[0]

    return QwenShape(
        name=HF_REPO.split("/")[-1],
        n_layer=int(config["num_hidden_layers"]),
        n_head=n_head,
        n_kv_head=int(config.get("num_key_value_heads", n_head)),
        d_model=d_model,
        head_dim=head_dim,
        hidden_dim=int(config["intermediate_size"]),
        vocab=int(config["vocab_size"]),
        norm_eps=float(config["rms_norm_eps"]),
        rope_theta=float(config.get("rope_theta", 10_000.0)),
        eos_token=int(eos),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        max_position_embeddings=int(config["max_position_embeddings"]),
    )


def validate_shape(shape: QwenShape) -> None:
    if shape.d_model != shape.attention_dim:
        raise ConversionError(
            f"Espresso requires dModel == nHead * headDim; got {shape.d_model} != {shape.attention_dim}"
        )
    if shape.n_head % shape.n_kv_head != 0:
        raise ConversionError(
            f"Espresso requires nHead % nKVHead == 0; got {shape.n_head} % {shape.n_kv_head}"
        )
    # The ANE wants channel counts aligned to 64. Every Qwen2.5-0.5B dim already is;
    # fail loudly rather than discover it as a compile error with no diagnostics.
    for label, value in (
        ("dModel", shape.d_model),
        ("kvDim", shape.kv_dim),
        ("hiddenDim", shape.hidden_dim),
    ):
        if value % 64 != 0:
            raise ConversionError(
                f"{label}={value} is not a multiple of 64; ANE kernels require 64-aligned channels"
            )


def write_metadata(output_dir: Path, shape: QwenShape, max_seq: int) -> dict:
    metadata = {
        # `name` must contain "qwen" so `espc pack-native` infers model_family = qwen
        # and the qwen-bpe-v1 tokenizer contract.
        "name": shape.name,
        "nLayer": shape.n_layer,
        "nHead": shape.n_head,
        "nKVHead": shape.n_kv_head,
        "dModel": shape.d_model,
        "headDim": shape.head_dim,
        "hiddenDim": shape.hidden_dim,
        "vocab": shape.vocab,
        "maxSeq": max_seq,
        "normEps": shape.norm_eps,
        # Qwen2.5 uses 1e6, not the llama default of 1e4. Getting this wrong produces
        # plausible-but-wrong output rather than an error.
        "ropeTheta": shape.rope_theta,
        "eosToken": shape.eos_token,
        # MultiModelConfig has no `qwen` case; qwen is served by the llama layer family.
        "architecture": "llama",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def convert(
    source_dir: Path,
    output_dir: Path,
    shape: QwenShape,
    max_seq: int,
    write_fp32_sidecars: bool,
) -> ConversionStats:
    tensors = SafetensorsFile.open(source_dir / "model.safetensors")
    available = set(tensors.tensor_names())
    stats = ConversionStats()

    # Qwen2.5 has no q_norm/k_norm (that is Qwen3). Assert instead of assuming, because
    # the engine treats those files as an optional pair and would silently skip them.
    unexpected_norms = sorted(n for n in available if ".q_norm." in n or ".k_norm." in n)
    if unexpected_norms:
        raise ConversionError(
            "checkpoint carries per-head q/k RMSNorm tensors this converter does not export: "
            + ", ".join(unexpected_norms[:4])
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    def emit(source: str, relative: str) -> None:
        write_blob(
            tensors.read(source),
            output_dir / relative,
            source=source,
            stats=stats,
            write_fp32_sidecar=write_fp32_sidecars,
        )

    print("[convert] embeddings + final norm + lm head", flush=True)
    emit("model.embed_tokens.weight", "embeddings/token.bin")
    emit("model.norm.weight", "final_norm.bin")

    # tie_word_embeddings=true means there is no lm_head.weight tensor. The runtime
    # always loads a separate lm_head blob, so materialize the tied copy.
    if "lm_head.weight" in available:
        emit("lm_head.weight", "lm_head.bin")
    elif shape.tie_word_embeddings:
        emit("model.embed_tokens.weight", "lm_head.bin")
    else:
        raise ConversionError("checkpoint has no lm_head.weight and does not tie word embeddings")

    for layer in range(shape.n_layer):
        print(f"[convert] layer {layer}", flush=True)
        prefix = f"model.layers.{layer}"
        base = f"layers/{layer}"

        emit(f"{prefix}.input_layernorm.weight", f"{base}/rms_att.bin")
        emit(f"{prefix}.post_attention_layernorm.weight", f"{base}/rms_ffn.bin")

        # HF nn.Linear weights are already row-major (out, in), which is exactly what
        # the llama loaders and the ANE WeightBlob packer expect. No transpose.
        emit(f"{prefix}.self_attn.q_proj.weight", f"{base}/wq.bin")
        emit(f"{prefix}.self_attn.k_proj.weight", f"{base}/wk.bin")
        emit(f"{prefix}.self_attn.v_proj.weight", f"{base}/wv.bin")
        emit(f"{prefix}.self_attn.o_proj.weight", f"{base}/wo.bin")

        # Qwen2 biases q/k/v only; o_proj and the MLP are bias-free.
        emit(f"{prefix}.self_attn.q_proj.bias", f"{base}/bq.bin")
        emit(f"{prefix}.self_attn.k_proj.bias", f"{base}/bk.bin")
        emit(f"{prefix}.self_attn.v_proj.bias", f"{base}/bv.bin")
        for banned in ("self_attn.o_proj.bias", "mlp.gate_proj.bias", "mlp.up_proj.bias", "mlp.down_proj.bias"):
            if f"{prefix}.{banned}" in available:
                raise ConversionError(
                    f"{prefix}.{banned} exists but the llama/qwen layer layout has nowhere to put it"
                )

        # Espresso SwiGLU naming: w1 = gate, w3 = up, w2 = down.
        emit(f"{prefix}.mlp.gate_proj.weight", f"{base}/w1.bin")
        emit(f"{prefix}.mlp.up_proj.weight", f"{base}/w3.bin")
        emit(f"{prefix}.mlp.down_proj.weight", f"{base}/w2.bin")

    print("[convert] causal masks", flush=True)
    write_causal_masks(output_dir, max_seq)

    for name in TOKENIZER_FILES:
        candidate = source_dir / name
        if candidate.exists():
            shutil.copy2(candidate, output_dir / name)

    return stats


def expected_counts(shape: QwenShape) -> dict[str, int]:
    counts = {
        "embeddings/token.bin": shape.vocab * shape.d_model,
        "final_norm.bin": shape.d_model,
        "lm_head.bin": shape.vocab * shape.d_model,
    }
    for layer in range(shape.n_layer):
        base = f"layers/{layer}"
        counts.update(
            {
                f"{base}/rms_att.bin": shape.d_model,
                f"{base}/rms_ffn.bin": shape.d_model,
                f"{base}/wq.bin": shape.d_model * shape.attention_dim,
                f"{base}/wk.bin": shape.d_model * shape.kv_dim,
                f"{base}/wv.bin": shape.d_model * shape.kv_dim,
                f"{base}/wo.bin": shape.d_model * shape.attention_dim,
                f"{base}/bq.bin": shape.attention_dim,
                f"{base}/bk.bin": shape.kv_dim,
                f"{base}/bv.bin": shape.kv_dim,
                f"{base}/w1.bin": shape.hidden_dim * shape.d_model,
                f"{base}/w2.bin": shape.d_model * shape.hidden_dim,
                f"{base}/w3.bin": shape.hidden_dim * shape.d_model,
            }
        )
    return counts


def verify_native_directory(output_dir: Path, shape: QwenShape) -> None:
    """Re-read every blob header the way the Swift loader does."""
    for relative, count in expected_counts(shape).items():
        path = output_dir / relative
        if not path.exists():
            raise ConversionError(f"missing converted tensor {relative}")
        with path.open("rb") as handle:
            header = handle.read(128)
        if len(header) != 128:
            raise ConversionError(f"{relative} is shorter than a BLOBFILE header")
        magic = struct.unpack_from("<I", header, 64)[0]
        if magic != 0xDEADBEEF:
            raise ConversionError(f"{relative} has magic {magic:#x}, expected 0xDEADBEEF")
        data_offset = struct.unpack_from("<I", header, 80)[0]
        if data_offset != 128:
            raise ConversionError(f"{relative} has data offset {data_offset}, expected 128")
        declared = struct.unpack_from("<I", header, 72)[0]
        actual = path.stat().st_size - 128
        if declared != actual:
            raise ConversionError(f"{relative} declares {declared} payload bytes but holds {actual}")
        if declared != count * 2:
            raise ConversionError(
                f"{relative} holds {declared // 2} fp16 values, expected {count}"
            )


# ---------------------------------------------------------------------------
# Bundle packing
# ---------------------------------------------------------------------------


def pack_bundle(native_dir: Path, bundle_path: Path, context_target: int | None) -> None:
    if bundle_path.exists():
        shutil.rmtree(bundle_path)
    command = [
        "swift",
        "run",
        "--disable-sandbox",
        "espc",
        "pack-native",
        str(native_dir),
        str(bundle_path),
        "--overwrite",
    ]
    if context_target is not None:
        command += ["--context-target", str(context_target)]
    print(f"[pack] {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise ConversionError(f"`espc pack-native` failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        default=None,
        help="Path of the .esp bundle to write (default: <cache>/qwen25-05b/Qwen2.5-0.5B-Instruct.esp)",
    )
    parser.add_argument(
        "--native-dir",
        default=None,
        help="Where to stage the native model directory (default: alongside the bundle)",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Where to cache downloaded safetensors (default: <cache>/qwen25-05b-src)",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=1024,
        help="Exported context length. Qwen2.5 advertises 32768; masks scale as maxSeq^2 (default: 1024)",
    )
    parser.add_argument(
        "--fp32-sidecars",
        choices=("none", "all"),
        default="none",
        help=(
            "Write exact fp32 sidecars next to each fp16 blob. Use 'all' to separate "
            "conversion error from fp16 rounding error in the parity report (default: none)"
        ),
    )
    parser.add_argument("--force-download", action="store_true", help="Re-download source files")
    parser.add_argument("--skip-pack", action="store_true", help="Stop after writing the native directory")
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Fail instead of creating a managed venv when numpy is missing",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ensure_runtime_dependencies(allow_bootstrap=not args.no_bootstrap)

    cache_root = espresso_cache_root()
    source_dir = Path(args.source_dir).expanduser() if args.source_dir else cache_root / "qwen25-05b-src"
    bundle_path = (
        Path(args.output).expanduser()
        if args.output
        else cache_root / "qwen25-05b" / "Qwen2.5-0.5B-Instruct.esp"
    )
    native_dir = (
        Path(args.native_dir).expanduser()
        if args.native_dir
        else bundle_path.parent / "Qwen2.5-0.5B-Instruct-native"
    )

    if args.max_seq <= 0:
        raise ConversionError("--max-seq must be > 0")
    mask_limit = max_supported_mask_sequence_length()
    if args.max_seq > mask_limit:
        raise ConversionError(f"--max-seq {args.max_seq} exceeds the BLOBFILE mask limit {mask_limit}")

    download_source(source_dir, force=args.force_download)
    shape = read_shape(source_dir)
    validate_shape(shape)
    if args.max_seq > shape.max_position_embeddings:
        raise ConversionError(
            f"--max-seq {args.max_seq} exceeds the model context {shape.max_position_embeddings}"
        )

    print(
        f"[shape] {shape.name}: {shape.n_layer}L d={shape.d_model} heads={shape.n_head}/{shape.n_kv_head} "
        f"headDim={shape.head_dim} hidden={shape.hidden_dim} vocab={shape.vocab} "
        f"ropeTheta={shape.rope_theta:g} eos={shape.eos_token}",
        flush=True,
    )

    if native_dir.exists():
        shutil.rmtree(native_dir)
    stats = convert(
        source_dir=source_dir,
        output_dir=native_dir,
        shape=shape,
        max_seq=args.max_seq,
        write_fp32_sidecars=args.fp32_sidecars == "all",
    )
    metadata = write_metadata(native_dir, shape, args.max_seq)
    verify_native_directory(native_dir, shape)

    overflowing = stats.overflowing()
    if overflowing:
        detail = ", ".join(f"{r.relative_path} ({r.fp16_overflow} values)" for r in overflowing[:4])
        raise ConversionError(
            "fp16 conversion overflowed to infinity for: " + detail +
            ". These weights cannot be served in fp16 without rescaling."
        )

    worst = stats.worst_fp16_error()
    report = {
        "model": HF_REPO,
        "metadata": metadata,
        "nativeDirectory": str(native_dir),
        "bundle": str(bundle_path),
        "fp32Sidecars": args.fp32_sidecars,
        "tensorCount": len(stats.tensors),
        "totalElements": sum(record.count for record in stats.tensors),
        "worstFP16RoundingTensor": worst.relative_path if worst else None,
        "worstFP16RoundingAbsError": worst.fp16_max_abs_error if worst else None,
        "tensors": [record.__dict__ for record in stats.tensors],
    }
    (native_dir / "conversion-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"[convert] wrote {len(stats.tensors)} tensors "
        f"({report['totalElements']:,} elements); worst fp16 rounding "
        f"{report['worstFP16RoundingAbsError']:.3e} on {report['worstFP16RoundingTensor']}",
        flush=True,
    )

    if args.skip_pack:
        print(f"[done] native directory at {native_dir}", flush=True)
        return 0

    pack_bundle(native_dir, bundle_path, context_target=args.max_seq)
    print(f"[done] bundle at {bundle_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ConversionError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
