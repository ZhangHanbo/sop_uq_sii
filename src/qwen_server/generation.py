"""Explicit decoding policy, independent of an unlisted generation_config.json."""

from __future__ import annotations

import copy

GENERATION_POLICY = {
    "version": "explicit-generation-2.1",
    "num_beams": 1,
    "num_return_sequences": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "min_length": 0,
    "min_new_tokens": 0,
    "length_penalty": 1.0,
    "early_stopping": False,
    "use_cache": True,
    "renormalize_logits": False,
    "remove_invalid_values": False,
    "return_dict_in_generate": False,
    "output_scores": False,
    "output_logits": False,
    "output_attentions": False,
    "output_hidden_states": False,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "bad_words_ids": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "stop_strings": None,
    "sequence_bias": None,
}


def validate_generation_policy(profile):
    """Require the profile's complete, explicit generation policy."""
    if profile.get("generation_policy") != GENERATION_POLICY:
        raise ValueError("Use a 2.1 profile with the complete explicit generation policy")


def locked_generation_config(profile, model_config, tokenizer, *, sample=False, max_tokens=512):
    """Construct from library defaults and locked token IDs, never model.generation_config.

    The backend also replaces model.generation_config with this clean base: recent
    Transformers versions fill unspecified fields from that object at generate().
    """
    from transformers import GenerationConfig

    validate_generation_policy(profile)
    settings = copy.deepcopy(GENERATION_POLICY)
    settings.pop("version")
    text_config = (
        model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
    )
    settings.update(
        bos_token_id=getattr(text_config, "bos_token_id", tokenizer.bos_token_id),
        eos_token_id=getattr(text_config, "eos_token_id", tokenizer.eos_token_id),
        pad_token_id=tokenizer.pad_token_id,
        do_sample=sample,
        max_new_tokens=max_tokens,
        temperature=profile["sampling"]["temperature"] if sample else 1.0,
        top_p=profile["sampling"]["top_p"] if sample else 1.0,
        top_k=profile["sampling"]["top_k"] if sample else 0,
        typical_p=1.0,
        min_p=None,
    )
    return GenerationConfig(**settings)
