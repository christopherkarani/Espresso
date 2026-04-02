import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_state_predictor as script


class TrainStatePredictorTests(unittest.TestCase):
    def test_load_config_parses_train_and_predictor_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "base_bundle": {
                            "weights_dir": "/tmp/stories/weights",
                            "tokenizer_dir": "/tmp/stories/tokenizer",
                            "teacher_model": "espresso://stories",
                        },
                        "train": {
                            "texts": ["Hello world"],
                            "sequence_length": 32,
                            "window_stride": 8,
                            "max_samples": 4,
                            "steps": 10,
                            "learning_rate": 1e-4,
                            "hidden_weight": 1.0,
                            "token_weight": 0.25,
                            "confidence_weight": 0.1,
                            "device": "cpu",
                        },
                        "draft": {
                            "proposer_mode": "state_predictor",
                            "max_horizon": 4,
                            "supported_verify_steps": [1, 2, 4],
                            "model_dir": "/tmp/stories/predictor",
                            "bundle_descriptor_path": "draft/custom.json",
                        },
                        "predictor": {
                            "name": "stories-predictor",
                            "architecture": "residual_mlp",
                            "residual_blocks": 2,
                            "internal_width": 32,
                            "conditioning": "token_embedding_plus_committed_hidden",
                            "prediction_target": "projected_hidden",
                            "projection_dim": 16,
                            "future_steps": 3,
                            "confidence_head": "per_step_scalar",
                        },
                        "export": {
                            "output_model_dir": "/tmp/stories/output",
                            "bundle_path": "/tmp/stories/output.esp",
                            "context_target_tokens": 256,
                            "optimization_recipe": "stories-state-predictor-v1",
                            "quality_gate": "oracle-first",
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = script.StatePredictorConfig.load(config_path)

        self.assertEqual(config.draft.supported_verify_steps, [1, 2, 4])
        self.assertEqual(config.predictor.future_steps, 3)
        self.assertEqual(config.train.hidden_weight, 1.0)
        self.assertEqual(config.export.context_target_tokens, 256)

    def test_export_state_predictor_writes_metadata_and_expected_blobs(self) -> None:
        predictor = script.PredictorSpec(
            name="stories-predictor",
            architecture="residual_mlp",
            residual_blocks=2,
            internal_width=16,
            conditioning="token_embedding_plus_committed_hidden",
            prediction_target="projected_hidden",
            projection_dim=8,
            future_steps=3,
            confidence_head="per_step_scalar",
        )
        model = script.TokenConditionedStatePredictor(d_model=8, predictor=predictor)
        draft = script.DraftSpec(
            proposer_mode="state_predictor",
            max_horizon=4,
            supported_verify_steps=[1, 2, 4],
            model_dir="/tmp/unused",
            bundle_descriptor_path="draft/multi-token-draft.json",
        )

        class BaseMetadata:
            d_model = 8
            vocab = 32
            norm_eps = 1e-5

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "predictor"
            exported = script.export_state_predictor(
                predictor_model=model,
                base_metadata=BaseMetadata(),
                draft=draft,
                predictor=predictor,
                output_dir=output_dir,
            )

            metadata = json.loads((exported / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((exported / "input_proj.bin").exists())
            self.assertTrue((exported / "blocks" / "0" / "ff1.bin").exists())
            self.assertTrue((exported / "blocks" / "1" / "ff2_bias.bin").exists())
            self.assertTrue((exported / "hidden_head.bin").exists())
            self.assertTrue((exported / "confidence_head.bin").exists())

        self.assertEqual(metadata["architecture"], "state_predictor")
        self.assertEqual(metadata["futureSteps"], 3)
        self.assertEqual(metadata["supportedVerifySteps"], [1, 2, 4])

    def test_build_predictor_examples_extracts_token_conditioned_targets(self) -> None:
        class TinyModel:
            def eval(self):
                return self

            def __call__(self, *, input_ids, output_hidden_states, use_cache):
                del output_hidden_states, use_cache
                seq = input_ids.shape[1]
                hidden = torch.arange(seq * 4, dtype=torch.float32).reshape(1, seq, 4)
                logits = torch.zeros((1, seq, 6), dtype=torch.float32)
                for index in range(seq):
                    logits[0, index, int(input_ids[0, min(index + 1, seq - 1)].item())] = 5.0
                return type(
                    "Outputs",
                    (),
                    {
                        "hidden_states": [hidden],
                        "logits": logits,
                    },
                )()

        samples = [torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)]
        examples = script.build_predictor_examples(
            exact_model=TinyModel(),
            samples=samples,
            future_steps=2,
            device="cpu",
        )

        self.assertEqual(len(examples), 2)
        self.assertEqual(int(examples[0].committed_token.item()), 1)
        self.assertEqual(examples[0].future_hidden.shape, (2, 4))
        self.assertEqual(examples[0].future_labels.tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
