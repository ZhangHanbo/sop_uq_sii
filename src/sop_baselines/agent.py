"""The fixed reference agent: one model call at a time, one tool call per reply.

An episode is kept in a JSON-serialisable *snapshot* that is saved after every
step, so an interrupted run resumes exactly where it stopped. Only data the
participant may observe enters the snapshot: messages, model replies, tool
receipts, costs and failures. This module is part of the agent contract.
"""

from __future__ import annotations

import copy
import json
import uuid

from sop_client import RemoteError
from sop_client.context import project_messages
from sop_protocol import TARGET_FIELDS, digest, valid_answer, valid_probability

from .contracts import SHARED_SAMPLES, agent_contract
from .prompts import build_messages, sampling_instruction

# Tool errors that end the episode: retrying cannot succeed.
TERMINAL_TOOL_ERRORS = ("protocol_failure_limit", "budget_exhausted", "tool_limit")


def parse_proposal(text, domain):
    """Parse one model reply into a tool call, a final proposal, or ``None``.

    A final proposal is accepted only if it contains every main scored field of
    ``domain`` with a legal value; extra report fields are dropped. Probabilities
    that are not finite numbers in [0, 1] are replaced by ``None``.
    """
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "tool_call":
        valid_call = isinstance(obj.get("tool"), str) and isinstance(
            obj.get("arguments"), dict
        )
        return obj if valid_call else None
    if obj.get("type") != "final_proposal" or not isinstance(obj.get("result"), dict):
        return None
    result = {key: obj["result"].get(key) for key in TARGET_FIELDS[domain]}
    if any(
        key not in obj["result"] for key in TARGET_FIELDS[domain]
    ) or not valid_answer(result, domain):
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
    contract = agent_contract(model.profile_hash)
    return {
        "episode_id": episode["episode_id"],
        "case_id": view["case_id"],
        "domain": view["domain"],
        "profile_hash": model.profile_hash,
        "model_id": model.model_id,
        "agent_contract": contract,
        "agent_contract_hash": digest(contract),
        "messages": build_messages(view),
        "history": [],
        "usage": {"main_calls": 0, "sample_calls": 0},
        "event_seq": episode["event_seq"],
        "answer": None,
        "p_error": None,
        "samples": [],
        "failure": None,
        "pending": None,
        "cost": 0.0,
        "phase": view["phase"],
    }


def _add_usage(snapshot, response):
    """Accumulate the token counts reported by the model service."""
    for key in ("prompt_tokens", "completion_tokens"):
        snapshot["usage"][key] = snapshot["usage"].get(key, 0) + response.get(
            "usage", {}
        ).get(key, 0)


def main_loop(snapshot, model, client, *, limit, save, seed=0):
    """Run the agent until it proposes an answer, fails, or uses ``limit`` calls.

    ``limit`` counts main model calls cumulatively in ``snapshot["usage"]``, so
    the post-query continuation passes the total limit, not an increment.
    A tool request is stored as ``pending`` with its UUID before it is sent; after
    a crash the same request is re-sent and the examiner returns the original
    receipt instead of charging twice.
    """
    while snapshot["usage"]["main_calls"] < limit or snapshot.get("pending"):
        pending = snapshot.get("pending")
        if pending is None:
            before = copy.deepcopy(snapshot["messages"])
            index = snapshot["usage"]["main_calls"]
            # Count the call before making it, so an interrupted call still counts.
            snapshot["usage"]["main_calls"] += 1
            save()
            try:
                response = model.generate(project_messages(before), seed=seed + index)
            except (RemoteError, OSError, ValueError) as exc:
                snapshot["failure"] = getattr(exc, "code", "model_transport_failure")
                snapshot["history"].append(
                    {"kind": "model_failure", "code": snapshot["failure"]}
                )
                save()
                break
            _add_usage(snapshot, response)
            snapshot["history"].append({"kind": "model_response", "response": response})
            snapshot["messages"].append(
                {"role": "assistant", "content": response["text"]}
            )
            obj = parse_proposal(response["text"], snapshot["domain"])
            if obj is None:
                snapshot["history"].append({"kind": "invalid_response"})
                snapshot["messages"].append(
                    {
                        "role": "user",
                        "content": "Invalid format. Emit one JSON tool_call or final_proposal using the public schema.",
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
            pending = {
                "episode_id": snapshot["episode_id"],
                "name": obj["tool"],
                "arguments": obj["arguments"],
                "mode": obj.get("mode", "default"),
                "request_id": str(uuid.uuid4()),
            }
            snapshot["pending"] = pending
            save()
        try:
            result = client.call("tools", pending)
            receipt = result["receipt"]
            snapshot["event_seq"] = result["event_seq"]
            snapshot["cost"] += receipt.get("cost", 0.0)
            snapshot["phase"] = receipt.get("phase", snapshot["phase"])
            snapshot["history"].append({"kind": "tool_receipt", "receipt": receipt})
            snapshot["messages"].append(
                {
                    "role": "user",
                    "content": json.dumps({"tool_result": receipt}, ensure_ascii=False),
                }
            )
        except RemoteError as exc:
            # Rejected requests are shown to the model so it can correct itself.
            snapshot["history"].append({"kind": "tool_error", "code": exc.code})
            snapshot["messages"].append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {"tool_error": {"code": exc.code, "message": str(exc)}}
                    ),
                }
            )
            if exc.code in TERMINAL_TOOL_ERRORS:
                snapshot["failure"] = exc.code
            if exc.code == "protocol_failure_limit":
                snapshot["controller_valid"] = False
        snapshot["pending"] = None
        save()
        if snapshot.get("failure"):
            break
    if snapshot.get("answer") is None and not snapshot.get("failure"):
        snapshot["failure"] = "model_call_limit"
    save()
    return snapshot


def shared_samples(snapshot, model, *, save, seed=0):
    """Draw the shared tool-free samples used by every initial risk baseline.

    Samples are generated only for a legal initial proposal and condition on the
    same messages that produced it, followed by a forced-final instruction.
    """
    if snapshot.get("answer") is None:
        return
    messages = copy.deepcopy(snapshot["proposal_messages"])
    messages.append(
        {"role": "user", "content": sampling_instruction(snapshot["domain"])}
    )
    while len(snapshot["samples"]) < SHARED_SAMPLES:
        index = len(snapshot["samples"])
        # Reserve the slot first: an interrupted sample stays invalid instead of
        # being silently redrawn with a different outcome.
        snapshot["samples"].append({"answer": None, "failure": "interrupted"})
        snapshot["usage"]["sample_calls"] += 1
        save()
        try:
            response = model.generate(
                project_messages(messages), sample=True, seed=seed + 10000 + index
            )
            _add_usage(snapshot, response)
            obj = parse_proposal(response["text"], snapshot["domain"])
            answer = obj["result"] if obj and obj["type"] == "final_proposal" else None
            snapshot["samples"][index] = {
                "answer": answer,
                "response": response,
                "failure": None if answer is not None else "invalid_sample",
            }
        except (RemoteError, OSError, ValueError) as exc:
            snapshot["samples"][index] = {
                "answer": None,
                "failure": getattr(exc, "code", "sample_transport_failure"),
            }
        save()


# Snapshot keys exported to ``*-initial.jsonl``/``*-posterior.jsonl`` for fitting.
VISIBLE_KEYS = (
    "case_id",
    "domain",
    "profile_hash",
    "model_id",
    "answer",
    "p_error",
    "samples",
    "history",
    "usage",
    "cost",
    "phase",
    "failure",
    "agent_contract",
    "agent_contract_hash",
    "evaluation_contract",
    "evaluation_contract_hash",
    "controller_valid",
    "solver_profile_hash",
    "initial_answer",
    "initial_status",
    "p_initial_after",
    "posterior_contract",
    "oracle_failure",
)


def visible_snapshot(snapshot):
    """Copy the exportable part of a snapshot (no transport or resume state)."""
    return {
        key: copy.deepcopy(snapshot[key]) for key in VISIBLE_KEYS if key in snapshot
    }
