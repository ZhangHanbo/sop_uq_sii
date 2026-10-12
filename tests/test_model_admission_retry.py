"""Retry only proven pre-execution admission failures; no real model is loaded."""

import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from sop_client import ModelClient, RemoteError
from sop_protocol import digest
from sop_qwen.server import InferenceQueueFull, make_server
from test_public_candidate import FakeModel

TOKEN = "admission-fixture-not-a-secret-123"
MESSAGES = [{"role": "user", "content": "synthetic retry fixture"}]


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def scripted_server(error, *, status=503, reject_count=99):
    """Return controlled HTTP responses and count received/executed requests."""
    state = {"requests": [], "executions": 0}
    profile = {"model_id": "scripted-http-only", "backend": "test-only-fake"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status_code, body):
            data = json.dumps(body).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(200, {"profile": profile, "profile_hash": digest(profile)})

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["requests"].append(body)
            if len(state["requests"]) <= reject_count:
                self.send(status, {"error": error})
            else:
                state["executions"] += 1
                self.send(
                    200,
                    {"text": "synthetic completion", "profile_hash": digest(profile)},
                )

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler), state


def test_explicit_capacity_retry_keeps_identical_request_and_executes_once():
    server, state = scripted_server(
        {"code": "model_busy", "message": "busy", "not_executed": True}, reject_count=2
    )
    with serving(server) as url:
        client = ModelClient(url, TOKEN)
        with patch("sop_client.client.time.sleep") as sleep:
            result = client.generate(MESSAGES, sample=True, seed=137, max_tokens=16)
        assert result["text"] == "synthetic completion"
        assert [call.args[0] for call in sleep.call_args_list] == [0.1, 0.2]
    assert len(state["requests"]) == 3
    assert len(set(state["requests"])) == 1
    assert state["executions"] == 1


def test_three_capacity_rejections_remain_an_explicit_failure():
    server, state = scripted_server(
        {"code": "model_busy", "message": "busy", "not_executed": True}
    )
    with serving(server) as url:
        client = ModelClient(url, TOKEN, retries=9)
        with patch("sop_client.client.time.sleep"), pytest.raises(
            RemoteError
        ) as failed:
            client.generate(MESSAGES)
        assert failed.value.code == "model_busy" and failed.value.not_executed
    assert len(state["requests"]) == 3
    assert state["executions"] == 0


@pytest.mark.parametrize(
    "status,code,marker",
    [
        (503, "model_busy", None),
        (503, "model_busy", False),
        (503, "model_busy", "true"),
        (503, "generation_failed", True),
        (500, "model_busy", True),
    ],
)
def test_unknown_or_executed_errors_are_never_retried(status, code, marker):
    server, state = scripted_server(
        {"code": code, "not_executed": marker}, status=status
    )
    with serving(server) as url:
        client = ModelClient(url, TOKEN)
        with patch("sop_client.client.time.sleep") as sleep, pytest.raises(RemoteError):
            client.generate(MESSAGES)
        sleep.assert_not_called()
    assert len(state["requests"]) == 1


def test_backend_exception_cannot_impersonate_an_admission_rejection():
    class FailedBackend(FakeModel):
        executions = 0

        def generate(self, *args, **kwargs):
            self.executions += 1
            raise InferenceQueueFull("raised inside an admitted backend")

    backend = FailedBackend()
    with serving(make_server(backend, backend.configuration, TOKEN, port=0)) as url:
        client = ModelClient(url, TOKEN)
        with pytest.raises(RemoteError) as failed:
            client.generate(MESSAGES)
        assert failed.value.code == "generation_failed"
        assert not failed.value.not_executed
    assert backend.executions == 1


def test_timeout_after_admission_never_draws_a_second_completion():
    class SlowBackend(FakeModel):
        executions = 0

        def generate(self, *args, **kwargs):
            self.executions += 1
            time.sleep(0.15)
            return super().generate(*args, **kwargs)

    backend = SlowBackend()
    with serving(make_server(backend, backend.configuration, TOKEN, port=0)) as url:
        client = ModelClient(url, TOKEN, timeout=0.03)
        with pytest.raises((TimeoutError, socket.timeout)):
            client.generate(MESSAGES)
    assert backend.executions == 1
