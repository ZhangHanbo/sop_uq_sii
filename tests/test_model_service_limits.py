"""Resource bounds use a local fake backend, never Qwen or a shared service."""

import socket
import threading
import time
import unittest
import urllib.request

from sop_client import ModelClient, RemoteError
from sop_qwen.server import make_server
from test_public_candidate import FakeModel


class ModelServiceLimits(unittest.TestCase):
    def test_timeouts_require_finite_positive_numbers(self):
        model = FakeModel()
        for value in (float("nan"), float("inf"), True, "1", 0):
            for argument in ("socket_timeout", "queue_timeout"):
                with (
                    self.subTest(argument=argument, value=value),
                    self.assertRaises(ValueError),
                ):
                    make_server(
                        model,
                        model.configuration,
                        "fixture-token-" * 3,
                        port=0,
                        **{argument: value},
                    )

    def test_trickled_headers_cannot_extend_the_input_deadline(self):
        model = FakeModel()
        server = make_server(
            model,
            model.configuration,
            "fixture-token-" * 3,
            port=0,
            max_connections=1,
            socket_timeout=0.15,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = socket.create_connection(server.server_address, timeout=1)
        stopped = threading.Event()

        def trickle():
            try:
                connection.sendall(b"GET /health HTTP/1.1\r\nHost: ")
                while not stopped.wait(0.03):
                    connection.sendall(b"a")
            except OSError:
                pass

        sender = threading.Thread(target=trickle)
        sender.start()
        started = time.monotonic()
        try:
            self.assertEqual(connection.recv(1), b"")
            self.assertLess(time.monotonic() - started, 0.7)
        finally:
            stopped.set()
            sender.join(1)
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_slow_headers_release_the_connection_slot(self):
        model = FakeModel()
        server = make_server(
            model,
            model.configuration,
            "fixture-token-" * 3,
            port=0,
            max_connections=1,
            socket_timeout=0.08,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = socket.create_connection(server.server_address, timeout=1)
        try:
            connection.sendall(b"GET /health HTTP/1.1\r\nHost: unfinished")
            connection.settimeout(1)
            # Header timeout closes the accepted socket without requiring a token.
            self.assertEqual(connection.recv(1), b"")
            with urllib.request.urlopen(
                "http://127.0.0.1:" + str(server.server_port) + "/health", timeout=1
            ) as response:
                self.assertEqual(response.status, 200)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_waiting_inference_has_bounded_queue_time(self):
        started, release = threading.Event(), threading.Event()

        class BlockingModel(FakeModel):
            def generate(self, *args, **kwargs):
                started.set()
                if not release.wait(2):
                    raise RuntimeError("fixture request was not released")
                return super().generate(*args, **kwargs)

        model = BlockingModel()
        server = make_server(
            model,
            model.configuration,
            "fixture-token-" * 3,
            port=0,
            queue_timeout=0.1,
            socket_timeout=0.04,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = ModelClient(
            "http://127.0.0.1:" + str(server.server_port), "fixture-token-" * 3
        )
        errors = []

        def first_request():
            try:
                client.generate([{"role": "user", "content": "fixture"}])
            except (
                BaseException
            ) as exc:  # noqa: BLE001 - propagate fixture-thread errors to assertions.
                errors.append(exc)

        first = threading.Thread(target=first_request)
        first.start()
        try:
            self.assertTrue(started.wait(1))
            with self.assertRaises(RemoteError) as result:
                client.generate([{"role": "user", "content": "second fixture"}])
            self.assertEqual(result.exception.code, "model_busy")
            self.assertTrue(result.exception.not_executed)
        finally:
            release.set()
            first.join(2)
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertFalse(errors)
        self.assertFalse(first.is_alive())

    def test_connection_capacity_rejection_proves_no_backend_execution(self):
        started, release = threading.Event(), threading.Event()

        class BlockingModel(FakeModel):
            def generate(self, *args, **kwargs):
                started.set()
                if not release.wait(3):
                    raise RuntimeError("fixture request was not released")
                return super().generate(*args, **kwargs)

        model = BlockingModel()
        server = make_server(
            model, model.configuration, "fixture-token-" * 3, port=0, max_connections=1
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = ModelClient(
            f"http://127.0.0.1:{server.server_port}", "fixture-token-" * 3
        )
        errors = []

        def hold_connection():
            try:
                client.generate([{"role": "user", "content": "first fixture"}])
            except Exception as exc:
                errors.append(exc)

        first = threading.Thread(target=hold_connection)
        first.start()
        try:
            self.assertTrue(started.wait(1))
            with self.assertRaises(RemoteError) as rejected:
                client.generate([{"role": "user", "content": "connection rejected"}])
            self.assertEqual(rejected.exception.status, 503)
            self.assertEqual(rejected.exception.code, "model_busy")
            self.assertTrue(rejected.exception.not_executed)
        finally:
            release.set()
            first.join(2)
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertFalse(errors)
        self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main()
