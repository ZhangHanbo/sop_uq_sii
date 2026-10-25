"""Public constants shared by students and the examiner.

Domains, scored answer fields, legal answer values and the terminal decision
costs from the test description.
"""

from __future__ import annotations

import hashlib
import json
import math

DOMAINS = ("dangerous_goods", "aircraft_inspection", "customer_service")
AIRCRAFT_FIELDS = (
    "aircraft_ready",
    "mechanical_inspection_result",
    "electrical_inspection_result",
    "component_incident_response",
    "component_mismatch_response",
    "cross_check_response",
    "cross_check_reporting_response",
)
TARGET_FIELDS = {
    "dangerous_goods": ("hazard_class",),
    "aircraft_inspection": AIRCRAFT_FIELDS,
    "customer_service": ("final_resolution_status",),
}
HAZARD_CLASSES = (
    "Unable to Decide",
    "Hazard Class A",
    "Hazard Class B",
    "Hazard Class C",
    "Hazard Class D",
)
RESOLUTION_STATUSES = ("FAILED", "PENDING_ACTION", "RESOLVED", "ESCALATED")

# Terminal decision costs; the examiner adds each case's tool/oracle cost.
DEFER_COST = 1.0
ORDINARY_ERROR_COST = 1.5
SEVERE_ERROR_COST = 5.0
INVALID_TERMINAL_COST = 5.0


def digest(obj):
    """SHA-256 of the canonical JSON form of ``obj``."""
    raw = json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def valid_probability(p):
    """True for a finite real number in [0, 1]; booleans are rejected."""
    return (
        isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1
    )


def valid_answer(answer, domain):
    """True if ``answer`` has exactly the scored fields of ``domain`` with legal values."""
    if (
        domain not in TARGET_FIELDS
        or not isinstance(answer, dict)
        or set(answer) != set(TARGET_FIELDS[domain])
    ):
        return False
    if domain == "dangerous_goods":
        return answer["hazard_class"] in HAZARD_CLASSES
    if domain == "customer_service":
        return answer["final_resolution_status"] in RESOLUTION_STATUSES
    if type(answer["aircraft_ready"]) is not bool:
        return False
    for key in AIRCRAFT_FIELDS[1:]:
        allowed = (
            ("success", "fail", "retest")
            if key in ("mechanical_inspection_result", "electrical_inspection_result")
            else ("success", "failed", None)
        )
        if answer[key] not in allowed:
            return False
    return True
