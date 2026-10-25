"""Load public model profiles and verify or download the locked model files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from test_client.protocol import digest

from .generation import validate_generation_policy


def load_profile(path):
    """Read a public profile and check backend, context, thinking and decoding policy."""
    value = json.loads(Path(path).read_text())
    if value.get("backend") not in ("transformers", "llama.cpp"):
        raise ValueError("Unsupported model profile backend")
    context = value.get("context_limit")
    if type(context) is not int or context <= value.get("max_output_tokens", 512):
        raise ValueError("The profile needs an integer context_limit above max_output_tokens")
    if value.get("thinking") is not False:
        raise ValueError("Public profiles require disabled thinking")
    validate_generation_policy(value)
    return value


def verify_assets(profile, runtime):
    """Only the deploying process checks local weights; no paths enter public profiles."""
    groups = [(runtime["model_path"], profile["files"])]
    if profile.get("tokenizer_files"):
        if not runtime.get("tokenizer_path"):
            raise ValueError("This profile requires a local tokenizer_path with locked files")
        groups.append((runtime["tokenizer_path"], profile["tokenizer_files"]))
    verified = 0
    for path, entries in groups:
        base = Path(path).expanduser().resolve()
        if not entries:
            raise ValueError("Weight hash manifest is empty")
        for entry in entries:
            rel = Path(entry["file"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe model manifest path")
            target = base / rel if base.is_dir() else base
            if not base.is_dir() and len(entries) != 1:
                raise ValueError("A directory is required for a multi-file model")
            with target.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha != entry["sha256"] or target.stat().st_size != entry["bytes"]:
                raise ValueError("Model file hash/size mismatch: " + entry["file"])
            verified += 1
    return {"profile_hash": digest(profile), "verified_files": verified}


def download_assets(profile, destination):
    """Download the locked revision's files, then verify their sizes and hashes."""
    from huggingface_hub import snapshot_download

    repo = profile.get("quant_repo", profile["model_id"])
    revision = profile.get("quant_revision", profile["revision"])
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=destination,
        allow_patterns=[entry["file"] for entry in profile["files"]],
    )
    runtime = {"model_path": destination}
    if profile.get("tokenizer_files"):
        tokenizer_path = str(Path(destination) / "tokenizer")
        snapshot_download(
            repo_id=profile["model_id"],
            revision=profile["revision"],
            local_dir=tokenizer_path,
            allow_patterns=[entry["file"] for entry in profile["tokenizer_files"]],
        )
        runtime["tokenizer_path"] = tokenizer_path
    verify_assets(profile, runtime)
