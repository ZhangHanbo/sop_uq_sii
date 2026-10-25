"""Scoring rules shared by local train/dev evaluation and the examiner.

Three risks are scored against answer errors: ``p_initial_error`` against the
initial answer on every case, ``p_initial_error_after_queries`` against the
initial answer and ``p_final_error`` against the final answer on interactive
cases. Interactive cases also get a decision loss: the case's tool/oracle cost
plus the terminal cost of AUTO/DEFER. Every metric is computed per domain and
then averaged with equal domain weights (``domain_macro``).
"""

from __future__ import annotations

import math
from collections import defaultdict

from .protocol import (
    DEFER_COST,
    INVALID_TERMINAL_COST,
    ORDINARY_ERROR_COST,
    SEVERE_ERROR_COST,
    valid_answer,
    valid_probability,
)

PRIMARY_METRICS = {
    "initial_brier": ("initial", "brier"),
    "decision_loss": ("decision", "loss"),
}
HAZARD_LEVELS = {"Hazard Class A": 0, "Hazard Class B": 1, "Hazard Class C": 2, "Hazard Class D": 3}


def answer_error(answer, label, domain):
    """1 if the answer is missing, illegal or differs from the label in any field."""
    return int(not valid_answer(answer, domain) or answer != label)


def severe_error(answer, label, domain):
    """Wrong in the dangerous direction: under-classified hazard, unsafe release
    of an aircraft, or a customer problem marked RESOLVED when it is not.

    Only a legal answer can be severely wrong; a missing or illegal answer is
    charged through ``terminal_cost`` instead.
    """
    if not valid_answer(answer, domain) or answer == label:
        return False
    if domain == "dangerous_goods":
        proposed, actual = answer["hazard_class"], label["hazard_class"]
        return proposed != "Unable to Decide" and (
            actual == "Unable to Decide" or HAZARD_LEVELS[proposed] < HAZARD_LEVELS[actual]
        )
    if domain == "aircraft_inspection":
        return answer["aircraft_ready"] is True and label["aircraft_ready"] is False
    return (
        answer["final_resolution_status"] == "RESOLVED"
        and label["final_resolution_status"] != "RESOLVED"
    )


def terminal_cost(mode, answer, label, domain, defer_cost=DEFER_COST):
    """DEFER 1; AUTO correct 0, ordinary error 1.5, severe/illegal/missing 5; no decision 5."""
    if mode == "DEFER":
        return float(defer_cost)
    if mode != "AUTO" or not valid_answer(answer, domain):
        return INVALID_TERMINAL_COST
    if severe_error(answer, label, domain):
        return SEVERE_ERROR_COST
    return ORDINARY_ERROR_COST if answer_error(answer, label, domain) else 0.0


def _bins_error(pairs, adaptive, count=10):
    """Expected calibration error with 10 equal-width or equal-mass (tie-preserving) bins."""
    if not pairs:
        return None
    pairs = sorted(pairs)
    if adaptive:
        tied = defaultdict(list)
        for p, y in pairs:
            tied[p].append((p, y))
        target, groups, group = max(1, math.ceil(len(pairs) / count)), [], []
        for members in tied.values():
            group.extend(members)
            if len(group) >= target and len(groups) < count - 1:
                groups.append(group)
                group = []
        if group:
            groups.append(group)
    else:
        buckets = defaultdict(list)
        for p, y in pairs:
            buckets[min(count - 1, int(p * count))].append((p, y))
        groups = list(buckets.values())
    return sum(abs(sum(p for p, _ in g) - sum(y for _, y in g)) for g in groups) / len(pairs)


def probability_metrics(probabilities, errors):
    """Brier (invalid/missing risk counts 1), NLL, ECE, aECE, AP and AURC."""
    n = len(errors)
    if not n:
        return {"n": 0}
    legal = [(float(p), int(y)) for p, y in zip(probabilities, errors) if valid_probability(p)]
    m = len(legal)
    clip = lambda p: max(1e-6, min(1 - 1e-6, p))  # noqa: E731
    nll = sum(-(y * math.log(clip(p)) + (1 - y) * math.log(clip(1 - p))) for p, y in legal)
    result = {
        "n": n,
        "valid": m,
        "error_rate": sum(errors) / n,
        "brier": (sum((p - y) ** 2 for p, y in legal) + n - m) / n,
        "nll": (nll - (n - m) * math.log(1e-6)) / n,
        "ece": _bins_error(legal, adaptive=False),
        "aece": _bins_error(legal, adaptive=True),
    }
    tied = defaultdict(list)
    for p, y in legal:
        tied[p].append(y)
    positives = sum(y for _, y in legal)
    ap = tp = seen = 0
    for p in sorted(tied, reverse=True):
        seen += len(tied[p])
        tp += sum(tied[p])
        ap += (sum(tied[p]) / positives) * (tp / seen) if positives else 0
    result["ap"] = ap if positives else None
    # Area under the risk-coverage curve, averaging over orders within ties.
    seen = failures = 0
    risk_sum = 0.0
    for p in sorted(tied):
        size, rate = len(tied[p]), sum(tied[p]) / len(tied[p])
        for k in range(1, size + 1):
            risk_sum += (failures + k * rate) / (seen + k)
        seen += size
        failures += sum(tied[p])
    result["aurc"] = risk_sum / m if m else None
    return result


def decision_metrics(records):
    """Mean decision loss and AUTO/DEFER behaviour of interactive cases."""
    n = len(records)
    if not n:
        return {"n": 0}
    auto = [r for r in records if r["mode"] == "AUTO"]
    return {
        "n": n,
        "loss": sum(r["cost"] + r["terminal"] for r in records) / n,
        "mean_cost": sum(r["cost"] for r in records) / n,
        "auto_rate": len(auto) / n,
        "defer_rate": sum(r["mode"] == "DEFER" for r in records) / n,
        "no_decision_rate": sum(r["mode"] not in ("AUTO", "DEFER") for r in records) / n,
        "auto_error_rate": sum(r["final_error"] for r in auto) / len(auto) if auto else None,
        "severe_rate": sum(r["severe"] for r in auto) / n,
    }


def _section(records, metric):
    """Apply ``metric`` per domain and average its numeric outputs over domains."""
    groups = defaultdict(list)
    for record in records:
        groups[record["domain"]].append(record)
    by_domain = {domain: metric(group) for domain, group in sorted(groups.items())}
    keys = sorted({k for values in by_domain.values() for k in values if k not in ("n", "valid")})
    macro = {}
    for key in keys:
        values = [values.get(key) for values in by_domain.values()]
        macro[key] = (
            sum(values) / len(values) if values and all(v is not None for v in values) else None
        )
    return {"domain_macro": macro, "by_domain": by_domain}


def score(
    rows, cases, interactive_ids, *, costs=None, defer_cost=DEFER_COST, primary="initial_brier"
):
    """Score result rows against labels.

    ``rows`` maps case_id to a parsed results row (see ``results.read_results``);
    ``cases`` maps case_id to ``{"domain", "label"}``. Cases without a row count
    as blank. ``costs`` maps case_id to its tool/oracle cost; without it the
    row's ``cost`` column (or 0) is used.
    """
    if primary not in PRIMARY_METRICS:
        raise ValueError("primary must be one of " + ", ".join(PRIMARY_METRICS))
    records = []
    for case_id, case in sorted(cases.items()):
        row = rows.get(case_id, {})
        domain, label = case["domain"], case["label"]
        cost = costs.get(case_id, 0.0) if costs is not None else row.get("cost") or 0.0
        record = {
            "domain": domain,
            "interactive": case_id in interactive_ids,
            "p": row.get("p_initial_error"),
            "q": row.get("p_initial_error_after_queries"),
            "r": row.get("p_final_error"),
            "mode": row.get("mode"),
            "cost": float(cost),
            "initial_error": answer_error(row.get("initial_answer_json"), label, domain),
            "final_error": answer_error(row.get("final_answer_json"), label, domain),
            "severe": row.get("mode") == "AUTO"
            and severe_error(row.get("final_answer_json"), label, domain),
        }
        record["terminal"] = terminal_cost(
            record["mode"], row.get("final_answer_json"), label, domain, defer_cost
        )
        records.append(record)
    interactive = [r for r in records if r["interactive"]]
    report = {
        "cases": len(records),
        "interactive_cases": len(interactive),
        "initial": _section(
            records,
            lambda rs: probability_metrics([r["p"] for r in rs], [r["initial_error"] for r in rs]),
        ),
        "updated_initial": _section(
            interactive,
            lambda rs: probability_metrics([r["q"] for r in rs], [r["initial_error"] for r in rs]),
        ),
        "final": _section(
            interactive,
            lambda rs: probability_metrics([r["r"] for r in rs], [r["final_error"] for r in rs]),
        ),
        "decision": _section(interactive, decision_metrics),
        "total_cost": round(sum(r["cost"] for r in records), 8),
    }
    section, key = PRIMARY_METRICS[primary]
    report["primary"] = {
        "metric": primary,
        "value": report[section]["domain_macro"].get(key),
        "lower_is_better": True,
    }
    return report
