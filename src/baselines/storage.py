"""Crash-safe local files for resumable runs."""

import json
import os
import tempfile
from pathlib import Path


def atomic_json(path, value):
    """Write JSON to a temporary file, fsync it, then atomically replace ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_jsonl(path):
    """Read a JSON Lines file, skipping blank lines."""
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]
