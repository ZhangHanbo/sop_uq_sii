"""Canonical form of the messages shown to the model.

Transport identifiers (receipt, episode and request IDs) differ between otherwise
identical runs. They are removed from receipts before inference and kept in the
local history, so the same evidence always yields the same prompt.
"""

from __future__ import annotations

import json

PROJECTION_VERSION = "receipt-semantics-2.1"
TRANSPORT_FIELDS = frozenset({"receipt_id", "episode_id", "request_id"})


def project_evidence(value):
    """Remove identities only from recognized receipts, never arbitrary inputs.

    Observation IDs, source, time, phase, event sequence, costs and business values
    intentionally remain: they can change the interpretation of evidence.
    """
    if isinstance(value, list):
        return [project_evidence(item) for item in value]
    if isinstance(value, dict):
        is_receipt = (
            "receipt_id" in value
            and "phase" in value
            and ("tool" in value or "authority" in value)
        )
        return {
            key: project_evidence(item)
            for key, item in value.items()
            if not (is_receipt and key in TRANSPORT_FIELDS)
        }
    return value


def project_messages(messages):
    """Canonicalize all JSON message envelopes before every model invocation."""
    result = []
    for message in messages:
        message = dict(message)
        try:
            parsed = json.loads(message["content"])
        except (ValueError, TypeError):
            pass
        else:
            message["content"] = json.dumps(
                project_evidence(parsed),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        result.append(message)
    return result
