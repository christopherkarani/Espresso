#!/usr/bin/env python3
"""Stage and pack a Stories `.esp` bundle with an exact two-token draft artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTOR_RELATIVE_PATH = "draft/exact-two-token.json"
DEFAULT_DRAFT_MODEL_RELATIVE_DIR = "draft/student"


@dataclass(frozen=True)
class DraftDescriptor:
    model_dir: str
    tokenizer_dir: str
    model_id: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_dir": self.model_dir,
                "tokenizer_dir": self.tokenizer_dir,
                "model_id": self.model_id,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def load_model_id(model_dir: Path) -> str:
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return str(metadata.get("name", model_dir.name))


def stage_bundle_model_dir(
    *,
    base_model_dir: Path,
    draft_model_dir: Path,
    output_model_dir: Path,
    tokenizer_dir: Path,
    draft_descriptor_relative_path: str = DEFAULT_DESCRIPTOR_RELATIVE_PATH,
    draft_model_relative_dir: str = DEFAULT_DRAFT_MODEL_RELATIVE_DIR,
) -> tuple[Path, DraftDescriptor]:
    copy_tree(base_model_dir, output_model_dir)

    staged_draft_dir = output_model_dir / draft_model_relative_dir
    staged_draft_dir.parent.mkdir(parents=True, exist_ok=True)
    copy_tree(draft_model_dir, staged_draft_dir)

    descriptor = DraftDescriptor(
        model_dir=draft_model_relative_dir,
        tokenizer_dir="tokenizer",
        model_id=load_model_id(draft_model_dir),
    )
    descriptor_path = output_model_dir / draft_descriptor_relative_path
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_text(descriptor.to_json(), encoding="utf-8")
    return descriptor_path, descriptor


def pack_bundle(
    *,
    staged_model_dir: Path,
    bundle_path: Path,
    tokenizer_dir: Path,
    descriptor_path: Path,
    context_target_tokens: int,
    model_tier: str,
    behavior_class: str,
    optimization_recipe: str,
    quality_gate: str,
    teacher_model: str | None,
    draft_model: str | None,
    performance_target: str | None,
    overwrite: bool,
) -> None:
    espc = REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "espc"
    if not espc.exists():
        raise FileNotFoundError(f"espc not found at {espc}")

    command = [
        str(espc),
        "pack-native",
        str(staged_model_dir),
        str(bundle_path),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--context-target",
        str(context_target_tokens),
        "--model-tier",
        model_tier,
        "--behavior-class",
        behavior_class,
        "--optimization-recipe",
        optimization_recipe,
        "--quality-gate",
        quality_gate,
        "--draft-kind",
        "exact_two_token",
        "--draft-behavior-class",
        "exact",
        "--draft-horizon",
        "2",
        "--draft-verifier",
        "exact",
        "--draft-rollback",
        "replay_from_checkpoint",
        "--draft-artifact",
        f"weights/{descriptor_path.relative_to(staged_model_dir).as_posix()}",
        "--draft-acceptance-metric",
        "accepted_future_tokens",
    ]
    if teacher_model:
        command.extend(["--teacher-model", teacher_model])
    if draft_model:
        command.extend(["--draft-model", draft_model])
    if performance_target:
        command.extend(["--performance-target", performance_target])
    if overwrite:
        command.append("--overwrite")

    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--draft-model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--context-target", type=int, required=True)
    parser.add_argument("--model-tier", default="optimized")
    parser.add_argument("--behavior-class", default="exact")
    parser.add_argument("--optimization-recipe", required=True)
    parser.add_argument("--quality-gate", required=True)
    parser.add_argument("--teacher-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--performance-target")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_model_dir = Path(args.output_model_dir).expanduser().resolve()
    descriptor_path, descriptor = stage_bundle_model_dir(
        base_model_dir=Path(args.base_model_dir).expanduser().resolve(),
        draft_model_dir=Path(args.draft_model_dir).expanduser().resolve(),
        output_model_dir=output_model_dir,
        tokenizer_dir=Path(args.tokenizer_dir).expanduser().resolve(),
    )
    pack_bundle(
        staged_model_dir=output_model_dir,
        bundle_path=Path(args.bundle_path).expanduser().resolve(),
        tokenizer_dir=Path(args.tokenizer_dir).expanduser().resolve(),
        descriptor_path=descriptor_path,
        context_target_tokens=args.context_target,
        model_tier=args.model_tier,
        behavior_class=args.behavior_class,
        optimization_recipe=args.optimization_recipe,
        quality_gate=args.quality_gate,
        teacher_model=args.teacher_model,
        draft_model=args.draft_model or descriptor.model_id,
        performance_target=args.performance_target,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_model_dir": str(output_model_dir),
                "draft_descriptor": str(descriptor_path),
                "bundle_path": str(Path(args.bundle_path).expanduser().resolve()),
                "draft_model_id": descriptor.model_id,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
