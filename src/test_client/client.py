"""JSON-over-HTTP clients for the examiner and the Qwen model service."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so a bearer token is never sent to another URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Refuse every redirect."""
        return None


class RemoteError(RuntimeError):
    """An error answer from a service; ``code`` is its machine-readable code."""

    def __init__(self, code, message, status=None):
        self.code, self.status = code, status
        super().__init__(f"{code}: {message}")


class HTTPClient:
    """Bearer-token JSON client for one service base URL."""

    def __init__(self, url, token=None, *, timeout=600):
        if not url or not url.startswith(("http://", "https://")):
            raise ValueError("An http(s):// service URL is required")
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout
        # Explicit endpoints should not accidentally traverse a host proxy.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path, body=None, *, content_type="application/json"):
        """Send one request; service errors are raised as ``RemoteError``."""
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if body is not None:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.url + path, data=body, headers=headers)
        try:
            with self.http.open(req, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                error = json.load(exc)["error"]
                code, message = error["code"], error["message"]
            except (ValueError, KeyError, TypeError):
                code, message = "http_error", str(exc)
            raise RemoteError(code, message, exc.code) from None


class ExaminerClient(HTTPClient):
    """Client of one team's examiner instance."""

    def spec(self):
        """Settings: costs, limits, test and interactive case IDs, tries left."""
        return self.request("/spec")

    def episode(self, case_id):
        """Open an episode: a fresh one for train/dev, the team's ledger for test."""
        return self.request("/episode", {"case_id": case_id})

    def tool(self, episode_id, name, arguments, mode="default"):
        """Call one business tool; returns ``{"receipt": ...}``."""
        return self.request(
            "/tool",
            {"episode_id": episode_id, "name": name, "arguments": arguments, "mode": mode},
        )

    def oracle(self, episode_id, field, phase):
        """Read the true value of one field; returns ``{"receipt": ...}``.

        A test case's oracle needs one submitted results file first
        (``submit_first``). The team's first test-case oracle read closes
        Problem 1: only earlier tries count for p_initial_error. The reply also
        gives the time of the case's first read (``first_oracle_at``).
        """
        return self.request("/oracle", {"episode_id": episode_id, "field": field, "phase": phase})

    def history(self, episode_id):
        """All receipts of an episode, its view, cost and call counts."""
        return self.request("/history/" + episode_id)

    def submit(self, path):
        """Upload a results.csv file; uses one try and returns its score report.

        With a trial token the file is only checked: no score, no try used.
        """
        data = Path(path).read_text(encoding="utf-8-sig").encode("utf-8")
        return self.request("/submit", data, content_type="text/csv")

    def submissions(self):
        """Every submission of this team and the best try of each problem so far."""
        return self.request("/submissions")


class ModelClient(HTTPClient):
    """Client of a Qwen model service started with ``sop-qwen serve``."""

    def __init__(self, url, token, *, busy_wait=900, **kwargs):
        super().__init__(url, token, **kwargs)
        record = self.request("/v1/profile")
        self.profile, self.profile_hash = record["profile"], record["profile_hash"]
        self.model_id = self.profile["model_id"]
        self.busy_wait = busy_wait

    def generate(self, messages, *, sample=False, seed=0, max_tokens=None):
        """Generate one reply; ``sample=False`` is greedy decoding.

        ``max_tokens`` defaults to the served profile's ``max_output_tokens``.
        A shared service answers ``model_busy`` when its queue is full. The
        request was not executed, so it is retried with back-off for up to
        ``busy_wait`` seconds.
        """
        body = {
            "messages": messages,
            "sample": sample,
            "seed": seed,
            "max_tokens": max_tokens or self.profile.get("max_output_tokens", 512),
            "profile_hash": self.profile_hash,
        }
        deadline, delay = time.monotonic() + self.busy_wait, 1.0
        while True:
            try:
                return self.request("/v1/generate", body)
            except RemoteError as exc:
                if exc.code != "model_busy" or time.monotonic() + delay > deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
