"""Command-line entry point ``sop-qwen``: ``download``, ``verify`` and ``serve``."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from .assets import download_assets, load_profile, verify_assets
from .backends import LlamaBackend, TransformersBackend, VllmBackend
from .server import make_server


def start_llama(runtime):
    """Start an owned llama.cpp server with the locked launch settings and wait until ready."""
    executable = str(Path(runtime["executable"]).expanduser().resolve())
    command = [
        executable,
        "--model",
        str(Path(runtime["model_path"]).expanduser().resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(runtime.get("upstream_port", 18083)),
        "--gpu-layers",
        "99",
        "--ctx-size",
        "32768",
        "--parallel",
        "4",
        "--batch-size",
        "1024",
        "--ubatch-size",
        "256",
        "--threads",
        "8",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--no-context-shift",
    ]
    env = os.environ.copy()
    library_path = ":".join(
        runtime.get("library_paths", [])
        + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    if runtime.get("loader"):
        command = [runtime["loader"], "--library-path", library_path, *command]
    elif library_path:
        env["LD_LIBRARY_PATH"] = library_path
    env["GGML_BACKEND_PATH"] = str(Path(executable).parent / "libggml-cuda.so")
    child = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + runtime.get("ready_timeout", 600)
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Owned llama.cpp process exited")
            try:
                with http.open(runtime["upstream_url"] + "/health", timeout=2) as response:
                    if response.status == 200:
                        return child
            except (OSError, ValueError):
                time.sleep(0.5)
        raise TimeoutError("Owned llama.cpp readiness timeout")
    except BaseException:
        stop_llama(child)
        raise


def stop_llama(child):
    """Terminate the owned llama.cpp process group, escalating to SIGKILL."""
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait()
            return
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main():
    """Parse arguments and run one subcommand."""
    parser = argparse.ArgumentParser(
        description="Deploy the same public Qwen service on candidate or examiner hardware"
    )
    parser.add_argument("action", choices=("verify", "download", "serve"))
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--runtime",
        help="Local-only JSON configuration; never send this to a remote client",
    )
    parser.add_argument("--destination", help="Download destination")
    parser.add_argument("--token-env", default="SOP_MODEL_TOKEN")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    if args.action == "download":
        if not args.destination:
            parser.error("download requires --destination")
        download_assets(profile, args.destination)
        return
    if not args.runtime:
        parser.error("verify/serve require --runtime")
    runtime = json.loads(Path(args.runtime).read_text())
    verification = verify_assets(profile, runtime)
    if args.action == "verify":
        print(json.dumps(verification))
        return
    token = os.environ.get(args.token_env, "")
    if len(token) < 24:
        parser.error("Set a model-service token of at least 24 characters in " + args.token_env)
    child = server = None

    def interrupted(signum, frame):
        """Turn SIGTERM/SIGINT into a clean shutdown."""
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        if profile["backend"] == "llama.cpp" and runtime.get("launch_server"):
            runtime["upstream_url"] = "http://127.0.0.1:" + str(runtime.get("upstream_port", 18083))
            child = start_llama(runtime)
        backend = {
            "vllm": VllmBackend,
            "transformers": TransformersBackend,
            "llama.cpp": LlamaBackend,
        }[profile["backend"]](profile, runtime)
        server = make_server(
            backend,
            profile,
            token,
            runtime.get("host", "127.0.0.1"),
            runtime.get("port", 18080),
            audit_path=runtime.get("audit_log_path"),
            max_connections=runtime.get("max_connections", 16),
            socket_timeout=runtime.get("socket_timeout", 15.0),
            queue_timeout=runtime.get("queue_timeout", 30.0),
        )
        print(
            json.dumps({"status": "ready", **verification, "port": server.server_port}),
            flush=True,
        )
        server.serve_forever()
    finally:
        if server:
            server.server_close()
        stop_llama(child)


if __name__ == "__main__":
    main()
