"""Prompt construction for the reference agent.

Editing this text changes the agent: trajectories and calibrators collected with
the old text no longer describe the new agent, so collect and fit again.
"""

import copy
import json
from importlib.resources import files

from test_client.protocol import TARGET_FIELDS

# System instructions shared by all three domains. The domain SOP and the
# published amendment are appended to it in ``build_messages``.
SYSTEM = """Follow the full business SOP and published benchmark amendments below.
Tool records are observations and may be imperfect. Never invent missing measurements.
Return exactly one JSON object, without Markdown:
{"type":"tool_call","tool":"NAME","arguments":{...},"mode":"default"}
or
{"type":"final_proposal","result":{...},"self_reported_error_probability":0.25}
Only one ordinary tool call is allowed per response. The probability refers to any
error in the main scored fields of your own proposal. You cannot call the oracle.
You have 20 initial main model calls and 32 main calls total; ordinary actions are
subject to the server's 24-call limit and budget. Optional observation modes are
default, refresh, remeasure and independent. Repeated observations
may be dependent. This JSON envelope overrides the SOP's XML presentation examples.
The result must contain exactly the main target fields listed in the amendment.
For continuation also provide self_reported_initial_error_probability: probability
that the explicitly frozen initial answer is wrong given all evidence now available.
Do not replace that frozen answer when estimating its error probability.
"""


def _resource(domain, name):
    """Read one packaged public specification file of ``domain``."""
    return files("baselines").joinpath("resources", domain, name).read_text()


def _compact_schema(value):
    """Drop prose ``description``/``examples`` keys from a JSON schema.

    Tool descriptions are kept once at the top level of each tool; nested prose
    only consumes context without changing the accepted arguments.
    """
    if isinstance(value, dict):
        return {
            key: _compact_schema(item)
            for key, item in value.items()
            if key not in ("description", "examples")
        }
    if isinstance(value, list):
        return [_compact_schema(item) for item in value]
    return value


def system_message(domain):
    """Instructions, the full domain SOP and its published amendment."""
    return {
        "role": "system",
        "content": SYSTEM
        + "\nFULL SOP:\n"
        + _resource(domain, "sop.txt")
        + "\nAMENDMENTS:\n"
        + _resource(domain, "amendment.txt"),
    }


def build_messages(view):
    """Return the initial chat for one episode.

    ``view`` is the episode view returned by the examiner: case inputs, the
    tools of this case, observation modes, costs and budget. The user message
    carries the view with compacted tool schemas.
    """
    model_view = copy.deepcopy(view)
    model_view["tools"] = [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": _compact_schema(tool["inputSchema"]["json"]),
        }
        for tool in view["tools"]
    ]
    return [
        system_message(view["domain"]),
        {
            "role": "user",
            "content": json.dumps(model_view, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def sampling_instruction(domain):
    """Instruction appended for the four shared samples (no tools allowed)."""
    return (
        "Using exactly the evidence above, emit one complete JSON object, without "
        'Markdown or tools: {"type":"final_proposal","result":{...},'
        '"self_reported_error_probability":0.25}. Replace the example probability '
        "with your own estimate. Keep the top-level type and result keys; do not "
        "return a bare answer. Only the nested result object is restricted to "
        "these target fields, with the legal values specified in the amendment: "
        + ", ".join(TARGET_FIELDS[domain])
    )
