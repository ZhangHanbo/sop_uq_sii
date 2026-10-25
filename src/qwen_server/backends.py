"""Inference backends: Transformers (M9) and a verified llama.cpp server (M27).

Both backends render the chat template with thinking disabled, enforce the
profile's context limit (prompt plus reply) and decode with its locked settings.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from test_client.client import NoRedirect

from .generation import locked_generation_config, validate_generation_policy


class ContextOverflow(RuntimeError):
    """The prompt plus the requested output would exceed the context limit."""


class TransformersBackend:
    """Local BF16 Transformers model; weights are loaded on first generation."""

    def __init__(self, profile, runtime):
        import torch
        from transformers import AutoConfig, AutoTokenizer

        validate_generation_policy(profile)
        self.profile, self.runtime, self.torch = profile, runtime, torch
        self.path = str(Path(runtime["model_path"]).expanduser().resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.config = AutoConfig.from_pretrained(self.path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None

    def _load(self):
        """Load the model once and replace its bundled generation config."""
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM

        model_class = AutoModelForCausalLM
        if self.config.model_type == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration

            model_class = Qwen3_5ForConditionalGeneration
        self.model = (
            model_class.from_pretrained(
                self.path,
                local_files_only=True,
                dtype=self.torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(self.runtime.get("device", "cuda"))
            .eval()
        )
        self.model.generation_config = locked_generation_config(
            self.profile, self.config, self.tokenizer
        )

    def generate(self, messages, *, sample=False, seed=0, max_tokens=512):
        """Generate one reply; greedy unless ``sample`` is true."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        count = batch.input_ids.shape[1]
        if count + max_tokens > self.profile["context_limit"]:
            raise ContextOverflow("Prompt plus output exceeds public context limit")
        self._load()
        self.torch.manual_seed(seed)
        generation_config = locked_generation_config(
            self.profile,
            self.config,
            self.tokenizer,
            sample=sample,
            max_tokens=max_tokens,
        )
        try:
            with self.torch.inference_mode():
                sequence = self.model.generate(
                    **batch.to(self.runtime.get("device", "cuda")),
                    generation_config=generation_config,
                )[0, count:].tolist()
        finally:
            # Prompts of different lengths fragment PyTorch's cache; without
            # releasing it, reserved memory creeps up across calls until the
            # GPU runs out (measured: 22 GiB peak per call, 27+ GiB reserved).
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        return {
            "text": self.tokenizer.decode(sequence, skip_special_tokens=True),
            "usage": {"prompt_tokens": count, "completion_tokens": len(sequence)},
            "finish_reason": "length" if len(sequence) == max_tokens else "stop",
        }


class LlamaBackend:
    """Adapter for an owned or separately launched, verified llama.cpp server."""

    def __init__(self, profile, runtime):
        from transformers import AutoTokenizer

        validate_generation_policy(profile)
        self.profile, self.runtime = profile, runtime
        self.url = runtime.get("upstream_url", "http://127.0.0.1:18083").rstrip("/")
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        props = self._request("/props")
        if (
            Path(props["model_path"]).resolve()
            != Path(runtime["model_path"]).expanduser().resolve()
        ):
            raise ValueError("llama.cpp server model differs from the locally verified model file")
        if props.get("build_info") != profile["server_build_info"]:
            raise ValueError("llama.cpp build differs from public model profile")
        # A matching executable and weights can still be launched with different
        # context/parallel settings. /props reports the live per-slot context,
        # speculative decoding and input modalities in the pinned server build.
        # Require this evidence rather than assuming that its launch flags match.
        server_spec = profile["server_spec"]
        settings = props.get("default_generation_settings")
        modalities = props.get("modalities")
        if (
            type(props.get("total_slots")) is not int
            or props["total_slots"] != server_spec["total_slots"]
            or not isinstance(settings, dict)
            or type(settings.get("n_ctx")) is not int
            or settings["n_ctx"] != server_spec["n_ctx_per_slot"]
            or settings.get("speculative") is not server_spec["speculative_decoding"]
            or not isinstance(modalities, dict)
            or modalities.get("vision") is not server_spec["vision_input"]
        ):
            raise ValueError(
                "llama.cpp live context, slots, speculative decoding or vision "
                "properties are missing or differ from public model profile"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            runtime.get("tokenizer_path", profile["model_id"]),
            revision=profile["revision"],
            local_files_only=True,
        )

    def _request(self, path, body=None):
        """POST (or GET without ``body``) to the llama.cpp server and decode JSON."""
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with self.http.open(request, timeout=600) as response:
            return json.load(response)

    def generate(self, messages, *, sample=False, seed=0, max_tokens=512):
        """Generate one reply through ``/completion`` with locked sampler settings."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        count = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if count + max_tokens > self.profile["context_limit"]:
            raise ContextOverflow("Prompt plus output exceeds public context limit")
        sampling = self.profile["sampling"]
        result = self._request(
            "/completion",
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "stream": False,
                "temperature": sampling["temperature"] if sample else 0.0,
                "top_p": sampling["top_p"] if sample else 1.0,
                "top_k": sampling["top_k"] if sample else 0,
                "min_p": 0.0,
                "typical_p": 1.0,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "mirostat": 0,
                "samplers": ["temperature", "top_k", "top_p"],
                "seed": seed,
                "cache_prompt": False,
                "return_tokens": True,
            },
        )
        if result.get("truncated"):
            raise ContextOverflow("llama.cpp truncated context")
        return {
            "text": result["content"],
            "usage": {
                "prompt_tokens": result.get("tokens_evaluated", count),
                "completion_tokens": result.get("tokens_predicted", len(result.get("tokens", []))),
            },
            "finish_reason": "length" if result.get("stop_type") == "limit" else "stop",
        }
