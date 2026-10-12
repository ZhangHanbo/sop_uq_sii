"""Authenticated JSON-over-HTTP clients with bounded, protocol-safe retries."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from sop_protocol import CAPABILITIES, SERVICE_PROTOCOL_VERSION, digest

from .context import project_messages


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so a bearer token is never sent to another URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RemoteError(RuntimeError):
    """An error returned by a service; ``code`` is its machine-readable code."""

    def __init__(self, code, message, status=None, *, not_executed=False):
        self.code, self.status = code, status
        # This assertion is usable only together with the documented admission
        # error code and HTTP status; generic transport errors prove nothing.
        self.not_executed = not_executed is True
        super().__init__(message)


class HTTPClient:
    """Bearer-token JSON client for one service base URL."""

    def __init__(self, url, token, *, timeout=600, retries=2):
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            raise ValueError("An HTTP(S) service URL is required")
        if not token:
            raise ValueError("A bearer token is required")
        self.url, self.token = url.rstrip("/"), token
        self.timeout, self.retries = timeout, retries
        # Explicit endpoints should not accidentally traverse a host proxy.
        self.http = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )

    def request(self, path, body=None, *, content_type="application/json", retry=False):
        """Send one request and return the decoded JSON response.

        With ``retry=True`` only network failures are retried, with exponential
        back-off. An error answer from the service is raised immediately as
        ``RemoteError`` and is never retried.
        """
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        headers = {
            "Authorization": "Bearer " + self.token,
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = content_type
        attempts = self.retries + 1 if retry else 1
        for attempt in range(attempts):
            req = urllib.request.Request(self.url + path, data=body, headers=headers)
            try:
                with self.http.open(req, timeout=self.timeout) as response:
                    result = json.load(response)
                if isinstance(result, dict) and result.get("error"):
                    err = result["error"]
                    raise RemoteError(
                        err.get("code", "remote_error"),
                        err.get("message", str(err)),
                        not_executed=err.get("not_executed") is True,
                    )
                return result
            except urllib.error.HTTPError as exc:
                not_executed = False
                try:
                    error = json.load(exc)
                    error = error.get("error", error)
                    code, message = (
                        error.get("code", "http_error"),
                        error.get("message", str(exc)),
                    )
                    not_executed = error.get("not_executed") is True
                except (ValueError, AttributeError):
                    code, message = "http_error", str(exc)
                # Protocol errors must never be silently retried.
                raise RemoteError(
                    code, message, exc.code, not_executed=not_executed
                ) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt + 1 == attempts:
                    raise
                time.sleep(0.25 * 2**attempt)


def check_service_capabilities(spec):
    """Reject old single-submission services before sending mutable work."""
    if spec.get("protocol_version") != SERVICE_PROTOCOL_VERSION or not set(
        CAPABILITIES
    ) <= set(spec.get("capabilities", [])):
        raise ValueError(
            "Examiner must support protocol 2.2 checkpoints, shared ledger, five versions and dev practice"
        )
    return spec


class ExaminerClient(HTTPClient):
    """Client of the examiner service (``/v2`` API)."""

    def spec(self):
        """Public settings: data release, model slots, costs, budget and limits."""
        return check_service_capabilities(self.request("/v2/spec", retry=True))

    def call(self, operation, body):
        """Call one RPC operation, e.g. ``tools``, ``oracle``, ``initial`` or ``final``."""
        if not operation.replace("_", "").isalnum():
            raise ValueError("Invalid operation")
        # Tools/oracle have caller UUIDs and draft operations are idempotent.
        # Restart creates a new practice each time; after an ambiguous failure
        # the caller must inspect its runs instead of silently creating another.
        return self.request(
            "/v2/rpc/" + operation, body, retry=operation != "restart_dev"
        )

    def evaluate(self, submission):
        """Upload a ``submission.zip`` (path or bytes) for evaluation."""
        data = (
            Path(submission).read_bytes()
            if not isinstance(submission, bytes)
            else submission
        )
        return self.request(
            "/v2/evaluations", data, content_type="application/zip", retry=True
        )

    def evaluation(self, evaluation_id):
        """Read one of your own evaluation reports."""
        return self.request(
            "/v2/evaluations/" + urllib.parse.quote(evaluation_id, safe=""), retry=True
        )

    def checkpoint(self, episode_id):
        """Prepare all official initial slots before this case's first oracle.

        The first successful fact return locks the checkpoint. Before that, a
        risk update invalidates this preparation and may be checkpointed again.
        No other case, method or continuation model is frozen.
        """
        return self.call("checkpoint_initial", {"episode_id": episode_id})

    def close(self, run_id):
        """Close an exam after at least one accepted answer version."""
        return self.call("finalize", {"run_id": run_id})


class ModelClient(HTTPClient):
    """Client of a Qwen model service started with ``sop-qwen serve``.

    The served profile is fetched once and must hash to the expected public
    profile; every generation request repeats that hash.
    """

    def __init__(self, url, token, *, expected_profile_hash=None, **kwargs):
        super().__init__(url, token, **kwargs)
        record = self.request("/v1/profile", retry=True)
        self.configuration = record["profile"]
        self.profile_hash = digest(self.configuration)
        if record.get("profile_hash") != self.profile_hash:
            raise ValueError("Model server returned an inconsistent profile digest")
        if expected_profile_hash and expected_profile_hash != self.profile_hash:
            raise ValueError("Model server does not match the selected public profile")
        self.model_id = self.configuration["model_id"]

    def generate(self, messages, *, sample=False, seed=0, max_tokens=512):
        """Generate one reply; ``sample=False`` is greedy decoding.

        At most three admission attempts are made, and only after an explicit
        503 model_busy / not_executed=true rejection. Timeouts, unknown errors
        and admitted generations are never retried: their execution state is
        uncertain and retrying could silently draw another completion.
        """
        body = {
            "messages": project_messages(messages),
            "sample": sample,
            "seed": seed,
            "max_tokens": max_tokens,
            "profile_hash": self.profile_hash,
        }
        for attempt in range(3):
            try:
                result = self.request("/v1/generate", body, retry=False)
                break
            except RemoteError as exc:
                safe_to_retry = (
                    exc.status == 503 and exc.code == "model_busy" and exc.not_executed
                )
                if not safe_to_retry or attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))
        if result.get("profile_hash") != self.profile_hash or not isinstance(
            result.get("text"), str
        ):
            raise ValueError("Invalid model response or model profile changed")
        return result
