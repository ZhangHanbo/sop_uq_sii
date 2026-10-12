"""Bind trajectories and fitted artifacts to the exact code that produced them.

Three independent bindings exist:

* the agent contract covers the fixed reference agent (prompt text, agent loop,
  evidence projection, call limits and sampling). The examiner freezes formal
  initial trajectories under this contract, so participants must not change it;
* the risk-estimator contract covers ``uq.py``. Changing an estimator requires
  refitting its calibrator, but never invalidates the shared initial trajectories;
* the query-implementation hash covers ``policies.py`` and ``runner.py``. Post-query
  trajectories depend on the query policy, so their calibrators bind to it.
"""

import copy
from pathlib import Path

import sop_protocol
from sop_client import context
from sop_protocol import digest

AGENT_VERSION = "2.2"
INITIAL_CALL_LIMIT = 20  # main model calls before the initial proposal
TOTAL_CALL_LIMIT = 32  # main model calls including the post-query continuation
SHARED_SAMPLES = 4  # tool-free samples drawn from the proposal's evidence

_ROOT = Path(__file__).parent


def protocol_fingerprint():
    """Bind executable public validators, not only their declared version."""
    return digest(Path(sop_protocol.__file__).read_text())


def evaluation_contract(spec, *, split="train"):
    """Effective public conditions used by a static baseline artifact.

    Human-readable profile IDs, deadlines and run quotas are deliberately not
    distribution fingerprints. Costs, per-case limits and public interface
    semantics are. Dev practice has a separate explicit unlimited quota policy.
    This check constrains reuse of a fitted artifact, not online algorithms.
    """
    required = (
        "costs",
        "budget",
        "tool_limit",
        "oracle_limit",
        "oracle_schema",
        "tool_schema_hash",
    )
    if any(key not in spec for key in required):
        raise ValueError("Service lacks the effective calibration configuration")
    effective = {key: copy.deepcopy(spec[key]) for key in required}
    if split == "dev":
        policy = spec.get("dev_policy")
        if (
            not isinstance(policy, dict)
            or "budget" not in policy
            or "oracle_limit" not in policy
        ):
            raise ValueError("Service lacks the effective dev practice configuration")
        for key in ("budget", "oracle_limit", "tool_limit"):
            if key in policy:
                effective[key] = copy.deepcopy(policy[key])
    return {
        "version": "1",
        "service_protocol": spec.get("protocol_version"),
        **effective,
        "quota_policy": "dev_unlimited_oracle" if split == "dev" else "bounded",
        "public_tool_schema_hash": digest(
            {
                str(path.relative_to(_ROOT)): path.read_text()
                for path in sorted(_ROOT.glob("resources/*/toolspecs.json"))
            }
        ),
    }


def check_evaluation_contract(snapshot, expected=None):
    contract = snapshot.get("evaluation_contract")
    if not isinstance(contract, dict) or snapshot.get(
        "evaluation_contract_hash"
    ) != digest(contract):
        raise ValueError("Trajectory lacks a valid evaluation contract; recollect")
    if expected is not None and contract != expected:
        raise ValueError(
            "Calibration evaluation configuration differs; recollect and refit"
        )
    return contract


def agent_contract(profile_hash):
    """Describe the fixed reference agent for one model profile."""
    paths = sorted(
        [
            *(_ROOT / name for name in ("agent.py", "prompts.py", "contracts.py")),
            *_ROOT.glob("resources/*/*.txt"),
            *_ROOT.glob("resources/*/*.json"),
        ]
    )
    source_hash = digest(
        {str(path.relative_to(_ROOT)): path.read_text() for path in paths}
    )
    return {
        "version": AGENT_VERSION,
        "source_hash": source_hash,
        "protocol_source_hash": protocol_fingerprint(),
        "projection_version": context.PROJECTION_VERSION,
        "projection_source_hash": digest(Path(context.__file__).read_text()),
        "model_profile_hash": profile_hash,
        "initial_call_limit": INITIAL_CALL_LIMIT,
        "total_call_limit": TOTAL_CALL_LIMIT,
        "shared_samples": SHARED_SAMPLES,
        "sampling_context": "pre-proposal-evidence-plus-forced-final-instruction",
    }


def risk_estimator_contract():
    """Describe the risk-estimator implementation in ``uq.py``."""
    return {
        "version": AGENT_VERSION,
        "source_hash": digest((_ROOT / "uq.py").read_text()),
        "protocol_source_hash": protocol_fingerprint(),
    }


def query_implementation_hash():
    """Hash the query-policy and run-orchestration implementation."""
    return digest(
        {name: (_ROOT / name).read_text() for name in ("policies.py", "runner.py")}
    )


def check_snapshot_contract(snapshot, expected=None):
    """Reject a trajectory produced by a different reference agent.

    ``expected`` defaults to the contract of the code currently installed.
    Returns the trajectory's contract hash when it matches.
    """
    contract = snapshot.get("agent_contract")
    contract_hash = snapshot.get("agent_contract_hash")
    if not isinstance(contract, dict) or contract_hash != digest(contract):
        raise ValueError(
            "Trajectory lacks a valid agent contract; recollect legacy trajectories"
        )
    current = expected or agent_contract(snapshot["profile_hash"])
    if contract != current:
        raise ValueError(
            "Trajectory agent/prompt/projection/sampling contract differs; recollect and refit"
        )
    return contract_hash
