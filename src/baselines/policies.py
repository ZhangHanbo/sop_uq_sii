"""Baseline query and decision policies.

These are transparent heuristics: they choose which facts to verify with the
oracle and whether to defer, using only public data and the visible history.
They do not estimate the value of information.
"""

from __future__ import annotations

from collections import defaultdict

POLICIES = ("no_oracle", "all_defer", "threshold", "fixed", "all_facts", "active")

# Fields the oracle can verify, in the fixed order used by the ``fixed`` policy.
ORACLE_FIELDS = {
    "dangerous_goods": ("sds", "handling", "transportation", "disposal"),
    "aircraft_inspection": (
        "aircraft_ready",
        "mechanical_inspection_result",
        "electrical_inspection_result",
        "cross_check_response",
    ),
    "customer_service": ("latency", "jitter", "bandwidth"),
}

# Tool payload keys of the dangerous-goods scores, mapped to their oracle names.
SCORE_ALIASES = {
    "sds_label_score": "sds",
    "handling_score": "handling",
    "transportation_score": "transportation",
    "disposal_score": "disposal",
}


def legal_queries(domain, history):
    """List the ``(field, phase)`` pairs the oracle accepts in this episode.

    Customer-service KPIs exist per phase: ``pre_repair`` after a diagnostic
    call and ``post_repair`` after troubleshooting.
    """
    if domain != "customer_service":
        return [(field, "assessment") for field in ORACLE_FIELDS[domain]]
    tools = {r.get("tool") for r in history["receipts"]}
    phases = []
    if "performTechnicalDiagnostics" in tools:
        phases.append("pre_repair")
    if "executeTroubleshooting" in tools:
        phases.append("post_repair")
    return [(field, phase) for phase in phases for field in ORACLE_FIELDS[domain]]


def _observed_values(history):
    """Group observed field values by ``(field, phase)`` across all receipts."""
    values = defaultdict(list)
    for receipt in history["receipts"]:
        payload = receipt.get("payload", {})
        metrics = payload.get("service metrics", payload.get("updated service metrics", payload))
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[(SCORE_ALIASES.get(name, name), receipt.get("phase", "assessment"))].append(
                    value
                )
    return values


def query_plan(policy, snapshot, history, *, p_initial, threshold=0.2, fixed_count=1):
    """Return the ordered oracle queries ``[(field, phase), ...]`` for one episode.

    * ``no_oracle``, ``all_defer``, ``threshold``: no query;
    * ``fixed``: the first ``fixed_count`` legal fields in ``ORACLE_FIELDS`` order;
    * ``all_facts``: every legal field the remaining budget can pay for;
    * ``active``: only if ``p_initial >= threshold``; rank fields by observed
      conflict, then missing values, then numeric spread, and take ``fixed_count``.
    """
    legal = legal_queries(snapshot["domain"], history)
    if policy in ("no_oracle", "all_defer", "threshold") or (
        policy == "active" and p_initial < threshold
    ):
        return []
    budget = history["view"]["remaining_budget"]
    price = history["view"]["costs"]["oracle"]
    cap = min(len(legal), int((budget + 1e-9) / price)) if price else len(legal)
    if policy == "fixed":
        return legal[: min(cap, fixed_count)]
    if policy == "all_facts":
        return legal[:cap]
    values = _observed_values(history)

    def score(query):
        """Sort key of one legal query: conflict, then missing value, then spread."""
        observed = values[query]
        # Compare type and value: JSON false is a Boolean reading, not the
        # dangerous-goods missing-score sentinel 0.
        conflict = len({(type(value).__name__, str(value)) for value in observed}) > 1
        missing = not observed or any(
            value is None
            or (
                snapshot["domain"] == "dangerous_goods"
                and type(value) in (int, float)
                and value == 0
            )
            for value in observed
        )
        numbers = [float(x) for x in observed if type(x) in (int, float)]
        spread = (
            (max(numbers) - min(numbers)) / (1 + abs(sum(numbers) / len(numbers)))
            if len(numbers) > 1
            else 0.0
        )
        return (int(conflict), int(missing), spread)

    return sorted(legal, key=score, reverse=True)[: min(cap, fixed_count)]


def final_mode(policy, p_final, threshold):
    """Decide AUTO or DEFER from the final-answer risk ``p_final``."""
    if policy == "all_defer":
        return "DEFER"
    if policy in ("threshold", "active", "fixed", "all_facts") and p_final >= threshold:
        return "DEFER"
    return "AUTO"
