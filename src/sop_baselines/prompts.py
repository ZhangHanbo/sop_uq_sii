"""Prompt construction for the fixed reference agent.

The text in this module is part of the agent contract (see ``contracts.py``):
editing it changes the reference agent, so trajectories and calibrators collected
with the old text can no longer be reused.
"""

import copy
import json
from importlib.resources import files

from sop_protocol import TARGET_FIELDS

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
    return files("sop_baselines").joinpath("resources", domain, name).read_text()


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


def build_messages(view):
    """Return the initial chat for one episode.

    ``view`` is the public episode view returned by the examiner: case inputs,
    the tools available in this case, observation modes, costs and budget.
    The system message carries the instructions, the full SOP and the amendment;
    the user message carries the view with compacted tool schemas.
    """
    domain = view["domain"]
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
        {
            "role": "system",
            "content": SYSTEM
            + "\nFULL SOP:\n"
            + _resource(domain, "sop.txt")
            + "\nAMENDMENTS:\n"
            + _resource(domain, "amendment.txt"),
        },
        {
            "role": "user",
            "content": json.dumps(
                model_view, ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]


def sampling_instruction(domain):
    """Instruction appended for the four shared samples (no tools allowed)."""
    return (
        "Using exactly the evidence above, emit one final_proposal now, without tools. Return only these result fields: "
        + ", ".join(TARGET_FIELDS[domain])
    )
