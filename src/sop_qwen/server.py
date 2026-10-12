"""Authenticated HTTP model service (``/health``, ``/v1/profile``, ``/v1/generate``).

Requests are size- and time-bounded, generations run one at a time, and an
optional audit log records request hashes and usage, never prompts or tokens.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sop_protocol import digest

from .backends import ContextOverflow


class InferenceQueueFull(RuntimeError):
    """No generation slot became free within the queue timeout."""


def make_server(
    backend,
    profile,
    token,
    host="127.0.0.1",
    port=18080,
    *,
    audit_path=None,
    max_connections=16,
    socket_timeout=15.0,
    queue_timeout=30.0,
):
    """Create (but do not start) the model service for ``backend`` and ``profile``."""
    if not isinstance(token, str) or len(token) < 24:
        raise ValueError("Use a separate model-service token of at least 24 characters")
    profile_hash = digest(profile)
    if (
        type(max_connections) is not int
        or not 1 <= max_connections <= 1024
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (socket_timeout, queue_timeout)
        )
    ):
        raise ValueError(
            "Model-service connection limits and timeouts must be positive"
        )
    inference_lock = threading.Lock()
    connections = threading.BoundedSemaphore(max_connections)
    generation_count = 0
    audit_file = Path(audit_path) if audit_path else None
    if audit_file:
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit_file.touch(exist_ok=True)
        audit_file.chmod(0o600)

    def audit(record):
        if audit_file:
            with audit_file.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    @contextmanager
    def inference_slot():
        if not inference_lock.acquire(timeout=queue_timeout):
            raise InferenceQueueFull("Inference queue is busy")
        try:
            yield
        finally:
            inference_lock.release()

    class BoundedServer(ThreadingHTTPServer):
        # Bound sockets before parsing headers: authentication comes too late to
        # protect against clients that send only part of an HTTP request.
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(socket_timeout)
            return connection, address

        def process_request(self, request, client_address):
            if not connections.acquire(blocking=False):
                try:
                    request.settimeout(1)
                    # No handler was started, so this request cannot have
                    # reached the backend. Keep that assurance machine-readable.
                    body = json.dumps(
                        {
                            "error": {
                                "code": "model_busy",
                                "message": "Model-service connection capacity reached",
                                "not_executed": True,
                            }
                        }
                    ).encode()
                    request.sendall(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                        + body
                    )
                except OSError:
                    pass
                finally:
                    self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                connections.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                connections.release()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.input_deadline = time.monotonic() + socket_timeout
            self.input_timer = threading.Timer(socket_timeout, self.expire_input)
            self.input_timer.daemon = True
            self.input_timer.start()

        def expire_input(self):
            # An inactivity timeout alone permits one-byte trickle connections.
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        def input_complete(self):
            # Generation can take minutes. The absolute deadline limits receipt
            # of headers/body only, never an admitted model generation.
            self.input_timer.cancel()
            if time.monotonic() > self.input_deadline:
                raise TimeoutError("Model request input deadline exceeded")

        def finish(self):
            self.input_timer.cancel()
            try:
                super().finish()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True

        def log_message(self, *args):
            pass  # Never log another participant's prompts, bearer tokens or answers.

        def respond(self, status, body):
            raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def authenticated(self):
            if hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            ):
                return True
            self.respond(
                401,
                {
                    "error": {
                        "code": "unauthorized",
                        "message": "Model-service authentication required",
                    }
                },
            )
            return False

        def do_GET(self):
            self.input_complete()
            if self.path == "/health":
                return self.respond(
                    200,
                    {
                        "status": "ready",
                        "service": "sop-qwen",
                        "profile_hash": profile_hash,
                    },
                )
            if not self.authenticated():
                return
            if self.path == "/v1/profile":
                return self.respond(
                    200,
                    {
                        "profile": profile,
                        "profile_hash": profile_hash,
                        "identity_evidence": "server_declared_profile; not remote model attestation",
                    },
                )
            self.respond(
                404, {"error": {"code": "not_found", "message": "Unknown route"}}
            )

        def do_POST(self):
            nonlocal generation_count
            if not self.authenticated():
                return
            if self.path != "/v1/generate":
                return self.respond(
                    404, {"error": {"code": "not_found", "message": "Unknown route"}}
                )
            generation_admitted = False
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Invalid request size")
                payload = self.rfile.read(length)
                self.input_complete()
                if len(payload) != length:
                    raise ValueError("Incomplete request body")
                body = json.loads(payload)
                if not isinstance(body, dict) or set(body) - {
                    "messages",
                    "profile_hash",
                    "sample",
                    "seed",
                    "max_tokens",
                }:
                    raise ValueError("Unexpected generation fields")
                if body.get("profile_hash") != profile_hash:
                    raise ValueError("Model profile mismatch")
                messages = body.get("messages")
                if not isinstance(messages, list) or not 1 <= len(messages) <= 150:
                    raise ValueError("Invalid messages")
                if any(
                    not isinstance(m, dict)
                    or set(m) != {"role", "content"}
                    or m["role"] not in ("system", "user", "assistant")
                    or not isinstance(m["content"], str)
                    for m in messages
                ):
                    raise ValueError("Only text chat messages are accepted")
                sample, seed, max_tokens = (
                    body.get("sample", False),
                    body.get("seed", 0),
                    body.get("max_tokens", 512),
                )
                if (
                    type(sample) is not bool
                    or type(seed) is not int
                    or not 0 <= seed < 2**31
                ):
                    raise ValueError("Invalid sampling arguments")
                if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
                    raise ValueError("Output must use at most 512 tokens")
                with inference_slot():
                    generation_admitted = True
                    generation_count += 1
                    started = time.monotonic()
                    record = {
                        "event": "generation",
                        "sequence": generation_count,
                        "started_unix": time.time(),
                        "profile_hash": profile_hash,
                        "request_hash": digest(body),
                        "backend_class": type(backend).__module__
                        + "."
                        + type(backend).__name__,
                        "declared_backend": profile.get("backend"),
                        "sample": sample,
                        "seed": seed,
                        "max_tokens": max_tokens,
                        "status": "started",
                    }
                    try:
                        result = backend.generate(
                            messages, sample=sample, seed=seed, max_tokens=max_tokens
                        )
                        record.update(
                            status="completed",
                            usage=result.get("usage", {}),
                            finish_reason=result.get("finish_reason"),
                            response_hash=digest(result.get("text")),
                        )
                    except BaseException as exc:
                        record.update(
                            status="failed", exception_class=type(exc).__name__
                        )
                        raise
                    finally:
                        record.update(
                            latency_seconds=time.monotonic() - started,
                            model_loaded=bool(
                                getattr(backend, "model", None) is not None
                            ),
                        )
                        audit(record)
                # Deliberate allowlist excludes server paths, raw upstream props and prompts.
                self.respond(
                    200,
                    {
                        "text": result["text"],
                        "usage": result.get("usage", {}),
                        "finish_reason": result.get("finish_reason", "stop"),
                        "profile_hash": profile_hash,
                        "request_id": str(uuid.uuid4()),
                        "request_hash": digest(body),
                    },
                )
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(
                    400, {"error": {"code": "invalid_request", "message": str(exc)}}
                )
            except ContextOverflow:
                self.respond(
                    400,
                    {
                        "error": {
                            "code": "context_overflow",
                            "message": "Prompt plus output exceeds 8192 tokens",
                        }
                    },
                )
            except InferenceQueueFull:
                self.respond(
                    503,
                    {
                        "error": {
                            # The backend could itself raise the same exception
                            # type. Once admitted, never claim it did not execute.
                            "code": (
                                "generation_failed"
                                if generation_admitted
                                else "model_busy"
                            ),
                            "message": (
                                "Model backend failed"
                                if generation_admitted
                                else "Model-service queue capacity reached"
                            ),
                            **({} if generation_admitted else {"not_executed": True}),
                        }
                    },
                )
            except (
                Exception
            ):  # noqa: BLE001 - backend details must not escape the authenticated API.
                self.respond(
                    503,
                    {
                        "error": {
                            "code": "generation_failed",
                            "message": "Model backend failed",
                        }
                    },
                )

    server = BoundedServer((host, port), Handler)
    audit(
        {
            "event": "server_started",
            "started_unix": time.time(),
            "profile_hash": profile_hash,
            "backend_class": type(backend).__module__ + "." + type(backend).__name__,
            "declared_backend": profile.get("backend"),
            "host": host,
            "port": server.server_port,
            "model_loaded": bool(getattr(backend, "model", None) is not None),
        }
    )
    return server
