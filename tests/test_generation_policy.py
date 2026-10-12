"""Optional CPU model integration; no downloaded model or GPU is used."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sop_qwen.assets import verify_assets
from sop_qwen.generation import (
    GENERATION_POLICY,
    locked_generation_config,
    validate_generation_policy,
)


class GenerationPolicyTests(unittest.TestCase):
    def test_profile_requires_exact_generation_policy(self):
        validate_generation_policy(
            {"generation_policy": copy.deepcopy(GENERATION_POLICY)}
        )
        with self.assertRaises(ValueError):
            validate_generation_policy({})
        changed = copy.deepcopy(GENERATION_POLICY)
        changed["num_beams"] = 3
        with self.assertRaises(ValueError):
            validate_generation_policy({"generation_policy": changed})

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
        "Run this CPU integration in the locked model environment",
    )
    def test_real_model_ignores_untracked_generation_config(self):
        import torch
        from sop_qwen.backends import TransformersBackend
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(17)
        tokenizer = SimpleNamespace(bos_token_id=0, eos_token_id=1, pad_token_id=1)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixture = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=32,
                    n_positions=16,
                    n_embd=16,
                    n_layer=1,
                    n_head=2,
                    bos_token_id=0,
                    eos_token_id=1,
                )
            ).eval()
            fixture.save_pretrained(directory)
            files = [
                p for p in directory.iterdir() if p.name != "generation_config.json"
            ]
            profile = {
                "generation_policy": copy.deepcopy(GENERATION_POLICY),
                "sampling": {"temperature": 0.7, "top_p": 0.9, "top_k": 40},
                "files": [
                    {
                        "file": p.name,
                        "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in files
                ],
            }

            def load_and_generate():
                backend = TransformersBackend.__new__(TransformersBackend)
                backend.model = None
                backend.config = fixture.config
                backend.profile, backend.runtime = profile, {"device": "cpu"}
                backend.path, backend.tokenizer, backend.torch = tmp, tokenizer, torch
                backend._load()
                self.assertEqual(backend.model.generation_config.num_beams, 1)
                self.assertEqual(
                    backend.model.generation_config.repetition_penalty, 1.0
                )
                self.assertIsNone(backend.model.generation_config.suppress_tokens)
                config = locked_generation_config(
                    profile, fixture.config, tokenizer, max_tokens=2
                )
                with torch.inference_mode():
                    return backend.model.generate(
                        torch.tensor([[0, 2]]), generation_config=config
                    ).tolist()

            before = verify_assets(profile, {"model_path": tmp})
            expected = load_and_generate()
            (directory / "generation_config.json").write_text(
                json.dumps(
                    {
                        "num_beams": 3,
                        "repetition_penalty": 4.0,
                        "suppress_tokens": list(range(2, 32)),
                        "bos_token_id": 0,
                        "eos_token_id": 1,
                    }
                )
            )
            after = verify_assets(profile, {"model_path": tmp})
            self.assertEqual(before, after)
            self.assertEqual(load_and_generate(), expected)


if __name__ == "__main__":
    unittest.main()
