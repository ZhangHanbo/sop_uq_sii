"""The reference agent: one model call at a time, at most one tool call per reply.

An episode lives in a JSON-serialisable *snapshot*: chat messages, model
replies, tool receipts, cost, the proposed answer and its self-reported risk.
``save`` is called after every step so an interrupted case can be inspected.
"""

from __future__ import annotations

import copy
import json
import re

from test_client import RemoteError
from test_client.protocol import TARGET_FIELDS, valid_answer, valid_probability

from .prompts import build_messages, sampling_instruction, system_message

INITIAL_CALL_LIMIT = 20  # main model calls before the initial proposal
TOTAL_CALL_LIMIT = 32  # main model calls including the post-query continuation
SHARED_SAMPLES = 4  # tool-free samples drawn from the proposal's evidence

# Tool errors that end the episode: retrying cannot succeed.
TERMINAL_TOOL_ERRORS = ("budget_exhausted", "tool_limit", "noninteractive_case")


def canonical_hazard_class(value):
    """Map short spellings such as ``"C"`` or ``"class c"`` to ``"Hazard Class C"``.

    The model often abbreviates the dangerous-goods class; the meaning is
    unambiguous, so the agent accepts it instead of asking again. Any other
    value is returned unchanged (and is then rejected if illegal).
    """
    if not isinstance(value, str):
        return value
    text = re.sub(r"^(hazard )?class ", "", re.sub(r"\s+", " ", value.strip()).lower())
    if text in ("a", "b", "c", "d"):
        return "Hazard Class " + text.upper()
    if text == "unable to decide":
        return "Unable to Decide"
    return value


def parse_proposal(text, domain):
    """Parse one model reply into a tool call, a final proposal, or ``None``.

    A final proposal must contain every scored field of ``domain`` with a legal
    value; extra fields are dropped and short hazard classes are expanded.
    Probabilities outside [0, 1] become ``None``.
    """
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "tool_call":
        valid_call = isinstance(obj.get("tool"), str) and isinstance(obj.get("arguments"), dict)
        return obj if valid_call else None
    if obj.get("type") != "final_proposal" or not isinstance(obj.get("result"), dict):
        return None
    result = {key: obj["result"].get(key) for key in TARGET_FIELDS[domain]}
    if domain == "dangerous_goods":
        result["hazard_class"] = canonical_hazard_class(result["hazard_class"])
    if any(key not in obj["result"] for key in TARGET_FIELDS[domain]) or not valid_answer(
        result, domain
    ):
        return None
    obj["result"] = result
    for key in (
        "self_reported_error_probability",
        "self_reported_initial_error_probability",
    ):
        if not valid_probability(obj.get(key)):
            obj[key] = None
    return obj


def new_snapshot(episode, model):
    """Create the empty snapshot of a freshly opened episode."""
    view = episode["view"]
    return {
        "episode_id": episode["episode_id"],
        "case_id": view["case_id"],
        "domain": view["domain"],
        "model_id": model.model_id,
        "messages": build_messages(view),
        "history": [],
        "usage": {"main_calls": 0, "sample_calls": 0},
        "answer": None,
        "p_error": None,
        "samples": [],
        "failure": None,
        "cost": 0.0,
        "phase": view["phase"],
    }


def frozen_snapshot(record, episode):
    """Rebuild a full snapshot from one line of ``data/test/initial_M9.jsonl``.

    The public file omits the system message, which depends only on the domain.
    """
    snapshot = copy.deepcopy(record)
    snapshot["episode_id"] = episode["episode_id"]
    snapshot["messages"] = [system_message(record["domain"])] + snapshot["messages"]
    return snapshot


def _add_usage(snapshot, response):
    """Accumulate the token counts reported by the model service."""
    for key in ("prompt_tokens", "completion_tokens"):
        snapshot["usage"][key] = snapshot["usage"].get(key, 0) + response.get("usage", {}).get(
            key, 0
        )


def main_loop(snapshot, model, examiner, *, limit, save=lambda: None, seed=0):
    """Run the agent until it proposes an answer, fails, or uses ``limit`` calls.

    ``limit`` counts main model calls cumulatively, so the post-query
    continuation passes the total limit, not an increment.
    """
    while snapshot["usage"]["main_calls"] < limit:
        index = snapshot["usage"]["main_calls"]
        snapshot["usage"]["main_calls"] += 1
        before = copy.deepcopy(snapshot["messages"])
        try:
            response = model.generate(before, seed=seed + index)
        except (RemoteError, OSError, ValueError) as exc:
            snapshot["failure"] = getattr(exc, "code", "model_transport_failure")
            snapshot["history"].append({"kind": "model_failure", "code": snapshot["failure"]})
            break
        _add_usage(snapshot, response)
        snapshot["history"].append({"kind": "model_response", "text": response["text"]})
        snapshot["messages"].append({"role": "assistant", "content": response["text"]})
        obj = parse_proposal(response["text"], snapshot["domain"])
        if obj is None:
            snapshot["history"].append({"kind": "invalid_response"})
            snapshot["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "Invalid format. Emit one JSON tool_call or final_proposal "
                        "using the public schema."
                    ),
                }
            )
            save()
            continue
        if obj["type"] == "final_proposal":
            snapshot.update(
                answer=obj["result"],
                p_error=obj["self_reported_error_probability"],
                p_initial_after=obj.get("self_reported_initial_error_probability"),
                failure=None,
                # The shared samples are drawn from exactly this evidence.
                proposal_messages=before,
            )
            save()
            return snapshot
        try:
            receipt = examiner.tool(
                snapshot["episode_id"], obj["tool"], obj["arguments"], obj.get("mode", "default")
            )["receipt"]
            snapshot["cost"] = round(snapshot["cost"] + receipt.get("cost", 0.0), 8)
            snapshot["phase"] = receipt.get("phase", snapshot["phase"])
            snapshot["history"].append({"kind": "tool_receipt", "receipt": receipt})
            content = {"tool_result": receipt}
        except RemoteError as exc:
            # Rejected requests are shown to the model so it can correct itself.
            snapshot["history"].append({"kind": "tool_error", "code": exc.code})
            content = {"tool_error": {"code": exc.code, "message": str(exc)}}
            if exc.code in TERMINAL_TOOL_ERRORS:
                snapshot["failure"] = exc.code
        snapshot["messages"].append(
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)}
        )
        save()
        if snapshot.get("failure"):
            break
    if snapshot.get("answer") is None and not snapshot.get("failure"):
        snapshot["failure"] = "model_call_limit"
    save()
    return snapshot


def draw_sample(snapshot, model, index, *, seed=0):
    """Draw shared sample number ``index`` (seed ``seed + 10000 + index``).

    The sample sees exactly the evidence behind the proposal, followed by a
    request for a tool-free final answer. Returns ``{"answer", "p_error",
    "failure"}``; an unparseable reply has answer ``None``.
    """
    messages = copy.deepcopy(snapshot["proposal_messages"])
    messages.append({"role": "user", "content": sampling_instruction(snapshot["domain"])})
    try:
        response = model.generate(messages, sample=True, seed=seed + 10000 + index)
    except (RemoteError, OSError, ValueError) as exc:
        return {"answer": None, "failure": getattr(exc, "code", "sample_transport_failure")}
    _add_usage(snapshot, response)
    obj = parse_proposal(response["text"], snapshot["domain"])
    answer = obj["result"] if obj and obj["type"] == "final_proposal" else None
    return {
        "answer": answer,
        "p_error": obj.get("self_reported_error_probability") if answer else None,
        "failure": None if answer is not None else "invalid_sample",
    }


def shared_samples(snapshot, model, *, save=lambda: None, seed=0):
    """Draw the four shared samples that feed the ``disagreement`` risk.

    Nothing is drawn without a proposal. Already drawn samples are kept, so an
    interrupted call continues with the next sample.
    """
    if snapshot.get("answer") is None:
        return
    while len(snapshot["samples"]) < SHARED_SAMPLES:
        index = len(snapshot["samples"])
        snapshot["usage"]["sample_calls"] += 1
        snapshot["samples"].append(draw_sample(snapshot, model, index, seed=seed))
        save()


# Snapshot keys exported to ``initial.jsonl``/``posterior.jsonl`` for fitting.
VISIBLE_KEYS = (
    "case_id",
    "domain",
    "model_id",
    "answer",
    "p_error",
    "samples",
    "history",
    "usage",
    "cost",
    "phase",
    "failure",
    "initial_answer",
    "p_initial_after",
    "oracle_failure",
)


def visible_snapshot(snapshot):
    """Copy the part of a snapshot used for fitting (no chat messages)."""
    return {key: copy.deepcopy(snapshot[key]) for key in VISIBLE_KEYS if key in snapshot}
