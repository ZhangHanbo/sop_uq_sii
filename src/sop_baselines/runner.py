"""Collect one reproducible baseline attempt and package authoritative drafts.

The supplied baseline computes its initial estimates first; this is an algorithm
choice, not a service-wide barrier. Formal oracle checkpoints are per case and
do not freeze the method. Other algorithms may use the public client directly to
update drafts and submit up to five versions while sharing the same tool ledger.
"""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

from sop_client import RemoteError
from sop_client.client import check_service_capabilities
from sop_protocol import (
    SERVICE_PROTOCOL_VERSION,
    SUBMISSION_SCHEMA_VERSION,
    digest,
    valid_probability,
    write_submission,
)

from .agent import (
    TERMINAL_TOOL_ERRORS,
    main_loop,
    new_snapshot,
    shared_samples,
    visible_snapshot,
)
from .contracts import (
    AGENT_VERSION,
    INITIAL_CALL_LIMIT,
    SHARED_SAMPLES,
    TOTAL_CALL_LIMIT,
    agent_contract,
    check_snapshot_contract,
    check_evaluation_contract,
    evaluation_contract,
    query_implementation_hash,
    risk_estimator_contract,
)
from .policies import POLICIES, final_mode, query_plan
from .storage import atomic_json
from .uq import raw_risk

QUERY_POLICIES = {"fixed", "all_facts", "active"}


def posterior_policy_contract(
    policy,
    uq,
    threshold,
    fixed_count,
    *,
    trigger_estimator_hash=None,
    solver_profile_hash=None,
):
    """Describe everything that decides which post-query histories exist.

    Post-query calibrators are valid only for trajectories collected under the
    same contract: query implementation, policy, query count and, for ``active``,
    the trigger estimator and its threshold.
    """
    return {
        "implementation_hash": query_implementation_hash(),
        "policy": policy,
        "solver_profile_hash": solver_profile_hash,
        "fixed_count": fixed_count if policy in ("fixed", "active") else None,
        "trigger_threshold": threshold if policy == "active" else None,
        "trigger_uq": uq if policy == "active" else None,
        "trigger_estimator_hash": (
            trigger_estimator_hash if policy == "active" else None
        ),
    }


def trusted_snapshot(episode, history, model):
    """Return the examiner's frozen initial trajectory, or ``None`` outside a formal test.

    In the formal test the initial answer, its evidence and the shared samples are
    fixed by the examiner. The trajectory is checked against the agent contract,
    the model profile and the case before it is reused.
    """
    proposal = history.get("initial_proposal") or episode.get("initial_proposal")
    if not proposal or proposal.get("frozen") is not True:
        return None
    snapshot = copy.deepcopy(proposal["snapshot"])
    check_snapshot_contract(snapshot)
    if (
        snapshot["profile_hash"] != model.profile_hash
        or snapshot["case_id"] != episode["view"]["case_id"]
    ):
        raise ValueError(
            "Examiner's frozen initial trajectory belongs to another case/model"
        )
    if snapshot["answer"] != proposal["answer"]:
        raise ValueError("Frozen proposal and its public trajectory disagree")
    snapshot.update(
        episode_id=episode["episode_id"], event_seq=history["event_seq"], pending=None
    )
    snapshot["trusted_initial_status"] = proposal["status"]
    return snapshot


def run_diagnostics(state):
    """Summarise a run for degenerate behaviour (e.g. all DEFER, few risk values)."""
    entries = list(state["episodes"].values())
    initial_risks = [
        entry.get("initial_commit", {}).get("p_initial_error") for entry in entries
    ]
    valid = [risk for risk in initial_risks if valid_probability(risk)]
    modes = [
        entry["final_commit"]["mode"] for entry in entries if "final_commit" in entry
    ]
    return {
        "cases": len(entries),
        "initial_valid_probabilities": len(valid),
        "initial_risk_min": min(valid) if valid else None,
        "initial_risk_max": max(valid) if valid else None,
        "initial_risk_distinct": len(set(valid)),
        "nonempty_query_plans": sum(bool(entry.get("query_plan")) for entry in entries),
        "active_risk_gate_skipped": sum(
            entry.get("risk_gate_skipped", False) for entry in entries
        ),
        "final_auto": modes.count("AUTO"),
        "final_defer": modes.count("DEFER"),
        "degenerate_all_defer": bool(modes) and all(mode == "DEFER" for mode in modes),
        "degenerate_all_auto": bool(modes) and all(mode == "AUTO" for mode in modes),
        "interpretation": "Descriptive diagnostics; online updates may use only information lawfully available at their recorded evidence watermark",
    }


def package_run(client, run_id, output):
    """Package authoritative drafts (formal) or finalized public practice rows."""
    run = client.call("get_run", {"run_id": run_id})
    if run["state"] != "FINALIZED" and not (
        run["state"] == "OPEN" and run.get("evaluation_mode") == "formal"
    ):
        raise ValueError("Finalize the run before packaging its authoritative rows")
    rows = copy.deepcopy(run["rows"])
    for row in rows:
        for key in ("episode_id", "initial_commit_id", "final_commit_id", "mode"):
            if row[key] is None:
                row[key] = ""
    manifest = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "protocol_version": SERVICE_PROTOCOL_VERSION,
        "dataset_release_id": run["release_id"],
        "evaluation_profile_id": run["evaluation_profile_id"],
        "run_id": run_id,
        "method_id": run["method_id"],
        "model_profiles": run["model_profiles"],
        "file_sha256": {},
    }
    write_submission(output, manifest, rows)
    return manifest


def _check_calibrators(models, calibrators, spec, uq, *, split="train"):
    """Every calibrator must match its model profile, data release and agent."""
    for slot, model in calibrators.items():
        model.check_estimator()
        if model.evaluation_contract != evaluation_contract(spec, split=split):
            raise ValueError(
                "Calibration evaluation configuration differs; recollect and refit"
            )
        if slot not in models or model.profile_hash != models[slot].profile_hash:
            raise ValueError("Calibration artifact belongs to another model profile")
        if model.release_id != spec["dataset_release_id"]:
            raise ValueError(
                "Training calibration artifact belongs to another data release"
            )
        if model.agent_contract != agent_contract(models[slot].profile_hash):
            raise ValueError(
                "Calibration artifact belongs to another agent/prompt/projection contract"
            )
    if uq not in ("self_report", "disagreement") and set(calibrators) != set(models):
        raise ValueError(
            "This UQ method requires fitted calibrators for every model slot"
        )


def _resolve_thresholds(
    models,
    calibrators,
    *,
    policy,
    uq,
    threshold,
    fixed_count,
    posterior_uq,
    initial_only,
    continuation_models=None,
):
    """Return per-slot initial thresholds, final thresholds and posterior contracts.

    Without an explicit ``threshold`` the values selected on the public holdout
    are used. A query run with calibrated post-query risk requires a posterior
    calibrator fitted under exactly the same query contract.
    """
    thresholds, final_thresholds, posterior_contracts = {}, {}, {}
    for slot in models:
        selected = calibrators[slot].selections if slot in calibrators else {}
        if (
            threshold is None
            and policy not in ("no_oracle", "all_defer")
            and uq not in selected
        ):
            raise ValueError(
                "Provide an explicit heuristic --threshold or select it on public cal_select/dev"
            )
        thresholds[slot] = (
            threshold
            if threshold is not None
            else selected.get(uq, {}).get("threshold", 0.5)
        )
        if slot in calibrators:
            trigger_hash = calibrators[slot].initial_estimator_hash(uq)
        else:
            # Same fields as Calibrator.initial_estimator_hash for unfitted methods.
            trigger_hash = digest(
                {
                    "method": uq,
                    "agent_contract_hash": digest(
                        agent_contract(models[slot].profile_hash)
                    ),
                    "estimator_contract": risk_estimator_contract(),
                    "fitted": None,
                    "fit_cases": None,
                    "training_trajectory_hash": None,
                }
            )
        posterior_contracts[slot] = posterior_policy_contract(
            policy,
            uq,
            thresholds[slot],
            fixed_count,
            trigger_estimator_hash=trigger_hash,
            solver_profile_hash=(
                continuation_models[slot].profile_hash
                if continuation_models
                and continuation_models[slot].profile_hash != models[slot].profile_hash
                else None
            ),
        )
        final_thresholds[slot] = thresholds[slot]
        if threshold is None and "posterior_final" in selected:
            final_thresholds[slot] = selected["posterior_final"]["threshold"]
    return thresholds, final_thresholds, posterior_contracts


def _require_posterior(calibrators, slot, contract, *, require_selection=False):
    """Check before the first real query, never on a zero-query branch.

    A legitimate all-AUTO selection can yield zero posterior training rows. Such
    a baseline remains executable without inventing data or swapping estimators.
    """
    fitted = calibrators.get(slot)
    if fitted is None or set(fitted.posterior) != {"initial", "final"}:
        raise ValueError(
            "Actual queries require public posterior calibration; collect public train trajectories with --collect-posterior, then fit"
        )
    if fitted.posterior_contract != contract:
        raise ValueError(
            "Query policy differs from the posterior calibration collection policy"
        )
    if require_selection and "posterior_final" not in fitted.selections:
        raise ValueError(
            "Select a posterior abstention threshold on held-out public query trajectories"
        )


def _initial_stage(
    client, models, calibrators, run, state, save, *, uq, seed, progress
):
    """Prepare baseline initial drafts; only reference collection seals globally.

    Outside the formal test the agent produces the initial answer here; in the
    formal test the examiner's frozen trajectory is reused unchanged.
    """
    for slot, model in models.items():
        for index, case_id in enumerate(run["case_ids"]):
            key = slot + ":" + case_id
            episode = client.call(
                "start_episode",
                {"run_id": run["run_id"], "case_id": case_id, "model_slot": slot},
            )
            history = client.call("history", {"episode_id": episode["episode_id"]})
            if key not in state["episodes"]:
                frozen = trusted_snapshot(episode, history, model)
                # Server-side activity without a local record cannot be reconstructed.
                if (
                    history["initial"] and history["initial"].get("status") != "missing"
                ) or (frozen is None and (history["event_seq"] or history["receipts"])):
                    raise ValueError(
                        "Missing local checkpoint for an existing episode; restore the original run directory"
                    )
                state["episodes"][key] = {
                    "snapshot": frozen or new_snapshot(episode, model),
                    "trusted_initial": frozen is not None,
                }
                save()
            entry = state["episodes"][key]
            snapshot = entry["snapshot"]
            if snapshot.get("evaluation_contract"):
                check_evaluation_contract(
                    snapshot, state["config"]["evaluation_contract"]
                )
            snapshot["evaluation_contract"] = copy.deepcopy(
                state["config"]["evaluation_contract"]
            )
            snapshot["evaluation_contract_hash"] = digest(
                snapshot["evaluation_contract"]
            )
            if history["initial"] and history["initial"].get("status") != "missing":
                entry["initial_commit"] = history["initial"]
                save()
                continue
            if not entry.get("initial_generated"):
                if not entry.get("trusted_initial"):
                    if snapshot["answer"] is None:
                        main_loop(
                            snapshot,
                            model,
                            client,
                            limit=INITIAL_CALL_LIMIT,
                            save=save,
                            seed=seed,
                        )
                    shared_samples(snapshot, model, save=save, seed=seed)
                entry["initial_generated"] = True
                entry["initial_snapshot"] = visible_snapshot(snapshot)
                save()
            risk = (
                calibrators[slot].predict(snapshot, uq)
                if slot in calibrators
                else raw_risk(snapshot, uq)
            )
            if snapshot["answer"] is None:
                risk = None
            entry["initial_commit"] = client.call(
                "initial",
                {
                    "episode_id": snapshot["episode_id"],
                    "answer": snapshot["answer"],
                    "p_initial_error": risk,
                    "event_seq": snapshot["event_seq"],
                    "status": (
                        "completed"
                        if snapshot["answer"] is not None
                        and valid_probability(risk)
                        and not snapshot["failure"]
                        and snapshot.get("trusted_initial_status", "completed")
                        == "completed"
                        else "failed"
                    ),
                },
            )
            save()
            progress(f"{slot} initial {index + 1}/{len(run['case_ids'])}")
    if run["state"] == "OPEN":
        return client.call("get_run", {"run_id": run["run_id"]})
    return client.call("seal_initial", {"run_id": run["run_id"]})


def _query_oracle(client, entry, snapshot, save):
    """Send the remaining planned oracle queries; stop at the first rejection."""
    for body in entry["query_plan"][
        entry.get("query_index", len(entry["oracle_receipts"])) :
    ]:
        try:
            result = client.call("oracle", body)
        except RemoteError as exc:
            snapshot["failure"] = exc.code
            if exc.code == "protocol_failure_limit":
                snapshot["controller_valid"] = False
            entry["oracle_failure"] = exc.code
            snapshot["history"].append({"kind": "oracle_error", "code": exc.code})
            entry["query_index"] = len(entry["query_plan"])
            save()
            break
        receipt = result["receipt"]
        snapshot["event_seq"] = result["event_seq"]
        if not result.get("cached"):
            snapshot["cost"] += receipt["cost"]
        entry["oracle_receipts"].append(receipt)
        entry["query_index"] = entry.get("query_index", 0) + 1
        snapshot["history"].append({"kind": "oracle_receipt", "receipt": receipt})
        save()


def _continue_after_queries(
    client, model, entry, snapshot, initial, contract, save, seed
):
    """Show the oracle evidence to the agent and let it propose a final answer."""
    if not entry.get("continuation_started"):
        # This is a solver declaration, not a change to the examiner's A0 slot.
        # The optional static posterior artifact separately binds this solver.
        snapshot["solver_profile_hash"] = model.profile_hash
        snapshot["messages"].append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "frozen_initial_answer": initial["answer"],
                        "oracle_evidence": entry["oracle_receipts"],
                        "instruction": "Continue using the new evidence. Propose your own final answer and both self_reported_error_probability and self_reported_initial_error_probability.",
                    },
                    ensure_ascii=False,
                ),
            }
        )
        snapshot.update(answer=None, p_error=None, p_initial_after=None, failure=None)
        snapshot["initial_answer"] = copy.deepcopy(initial["answer"])
        snapshot["initial_status"] = initial["status"]
        snapshot["posterior_contract"] = contract
        entry["continuation_started"] = True
        save()
    if not entry.get("continuation_finished"):
        if snapshot["answer"] is None:
            main_loop(
                snapshot, model, client, limit=TOTAL_CALL_LIMIT, save=save, seed=seed
            )
        entry["continuation_finished"] = True
        save()
    entry["posterior_snapshot"] = visible_snapshot(snapshot)
    if entry.get("oracle_failure"):
        snapshot["oracle_failure"] = entry["oracle_failure"]
        entry["posterior_snapshot"] = visible_snapshot(snapshot)


def _interactive_stage(
    client,
    models,
    calibrators,
    run,
    state,
    save,
    *,
    policy,
    fixed_count,
    seed,
    posterior_uq,
    thresholds,
    final_thresholds,
    posterior_contracts,
    continuation_models=None,
):
    """Query, continue and commit the final decision of each interactive case."""
    for slot, model in models.items():
        for case_id in run["interactive_ids"]:
            key = slot + ":" + case_id
            if key not in state["episodes"]:
                # Missing local evidence cannot be silently reconstructed as a completed method.
                raise ValueError(
                    "Missing local checkpoint for an interactive episode; preserve the run directory"
                )
            entry = state["episodes"][key]
            snapshot = entry["snapshot"]
            history = client.call("history", {"episode_id": snapshot["episode_id"]})
            snapshot["controller_valid"] = history.get(
                "controller_valid", snapshot.get("controller_valid", True)
            )
            snapshot["event_seq"] = history["event_seq"]
            snapshot["cost"] = sum(
                float(receipt.get("cost", 0)) for receipt in history["receipts"]
            )
            if history["final"] and history["final"].get("status") != "missing":
                entry["final_commit"] = history["final"]
                save()
                continue
            initial = history["initial"]
            risk = initial.get("p_initial_error")
            if "query_plan" not in entry:
                # The plan is fixed once, with request IDs, so a resumed run
                # re-sends identical requests instead of new, separately billed ones.
                plan = query_plan(
                    policy,
                    snapshot,
                    history,
                    p_initial=risk if valid_probability(risk) else 1.0,
                    threshold=thresholds[slot],
                    fixed_count=fixed_count,
                )
                entry["risk_gate_skipped"] = (
                    policy == "active"
                    and valid_probability(risk)
                    and risk < thresholds[slot]
                )
                if snapshot.get("failure") in TERMINAL_TOOL_ERRORS:
                    plan = []
                entry["query_plan"] = [
                    {
                        "episode_id": snapshot["episode_id"],
                        "field": field,
                        "phase": phase,
                        "request_id": str(uuid.uuid4()),
                    }
                    for field, phase in plan
                ]
                entry["oracle_receipts"] = []
                entry["query_index"] = 0
                save()
            if entry["query_plan"] and posterior_uq == "calibrated_self_report":
                _require_posterior(
                    calibrators,
                    slot,
                    posterior_contracts[slot],
                    require_selection=state["config"]["threshold_source"]
                    != "explicit_heuristic",
                )
            if (
                entry["query_plan"]
                and run.get("evaluation_mode") == "formal"
                and run["state"] == "OPEN"
            ):
                checkpoint = client.call(
                    "checkpoint_initial", {"episode_id": snapshot["episode_id"]}
                )
                initial = checkpoint["initials"][slot]
                entry["initial_commit"] = initial
                entry["initial_checkpoint_id"] = checkpoint["checkpoint_id"]
                snapshot["event_seq"] = checkpoint["event_seq"]
                save()
            _query_oracle(client, entry, snapshot, save)
            if entry["oracle_receipts"]:
                _continue_after_queries(
                    client,
                    (continuation_models or models)[slot],
                    entry,
                    snapshot,
                    initial,
                    posterior_contracts[slot],
                    save,
                    seed,
                )
                if posterior_uq == "calibrated_self_report":
                    p_initial_after = calibrators[slot].predict_posterior(
                        snapshot, "initial"
                    )
                    p_final = calibrators[slot].predict_posterior(snapshot, "final")
                else:
                    p_initial_after = snapshot.get("p_initial_after")
                    p_final = (
                        snapshot["p_error"]
                        if valid_probability(snapshot.get("p_error"))
                        else None
                    )
            else:
                # No new evidence: the final answer is the initial one.
                p_initial_after = p_final = risk
            answer = snapshot["answer"]
            decision_threshold = (
                final_thresholds[slot] if entry["oracle_receipts"] else thresholds[slot]
            )
            mode = final_mode(
                policy,
                p_final if valid_probability(p_final) else 1.0,
                decision_threshold,
            )
            # A failed shadow answer may still be legitimately deferred; the server grades it explicitly.
            entry["final_commit"] = client.call(
                "final",
                {
                    "episode_id": snapshot["episode_id"],
                    "answer": answer,
                    "mode": mode,
                    "p_initial_error_after_queries": p_initial_after,
                    "p_final_error": p_final if answer is not None else None,
                    "event_seq": snapshot["event_seq"],
                    "status": (
                        "completed"
                        if answer is not None
                        and valid_probability(p_initial_after)
                        and valid_probability(p_final)
                        and not snapshot["failure"]
                        and not entry.get("oracle_failure")
                        else "failed"
                    ),
                },
            )
            save()
    if run.get("evaluation_mode") == "formal" and run["state"] == "OPEN":
        return client.call("get_run", {"run_id": run["run_id"]})
    return client.call("finalize", {"run_id": run["run_id"]})


def _write_trajectories(output_dir, models, state, name, field):
    """Export one JSON line per exportable trajectory of each model slot."""
    for slot in models:
        snapshots = [
            entry[field]
            for key, entry in state["episodes"].items()
            if key.startswith(slot + ":") and field in entry
        ]
        (output_dir / (slot + "-" + name + ".jsonl")).write_text(
            "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in snapshots)
        )


def run_panel(
    client,
    models,
    output_dir,
    *,
    split="test",
    method_id="baseline",
    policy="no_oracle",
    uq="disagreement",
    calibrators=None,
    threshold=None,
    fixed_count=1,
    seed=0,
    posterior_uq=None,
    collect_posterior=False,
    initial_only=False,
    continuation_models=None,
    progress=print,
):
    """Run one method end to end and return the run summary.

    ``split`` is ``train`` (public pool), ``dev`` (unlimited practice) or ``test``.
    ``collect_posterior`` records raw post-query reports on training data so that
    post-query calibrators can be fitted; ``initial_only`` stops after sealing.
    A configuration hash identifies this resumable baseline attempt. This local
    resume check is not a ban on changing methods through the public client.
    """
    if (
        policy not in POLICIES
        or (threshold is not None and not valid_probability(threshold))
        or type(fixed_count) is not int
        or not 0 <= fixed_count <= 16
    ):
        raise ValueError("Invalid public baseline policy configuration")
    if collect_posterior and split not in ("train", "dev"):
        raise ValueError(
            "Uncalibrated posterior collection is allowed only on public training data"
        )
    calibrators = calibrators or {}
    spec = client.spec()
    check_service_capabilities(spec)
    if set(models) != set(spec["model_slots"]):
        raise ValueError("Provide exactly the model slots requested by /v2/spec")
    continuation_models = continuation_models or models
    if set(continuation_models) != set(models):
        raise ValueError("Provide one continuation method per official initial slot")
    _check_calibrators(models, calibrators, spec, uq, split=split)
    posterior_uq = posterior_uq or (
        "self_report"
        if uq == "self_report" or collect_posterior
        else "calibrated_self_report"
    )
    if posterior_uq not in ("self_report", "calibrated_self_report"):
        raise ValueError(
            "Posterior UQ must be self_report or separately calibrated_self_report"
        )
    if (
        policy in QUERY_POLICIES
        and not initial_only
        and posterior_uq == "self_report"
        and uq != "self_report"
        and not collect_posterior
    ):
        raise ValueError(
            "Use self_report at both stages, or fit separate posterior calibration; silent estimator switching is forbidden"
        )
    thresholds, final_thresholds, posterior_contracts = _resolve_thresholds(
        models,
        calibrators,
        policy=policy,
        uq=uq,
        threshold=threshold,
        fixed_count=fixed_count,
        posterior_uq=posterior_uq,
        initial_only=initial_only,
        continuation_models=continuation_models,
    )
    config = {
        "split": split,
        "policy": policy,
        "uq": uq,
        "thresholds": thresholds,
        "final_thresholds": final_thresholds,
        "threshold_source": (
            "explicit_heuristic"
            if threshold is not None
            else "public_holdout_selection"
        ),
        "posterior_uq": posterior_uq,
        "posterior_collection_only": collect_posterior,
        "posterior_contracts": posterior_contracts,
        "initial_only": initial_only,
        "fixed_count": fixed_count,
        "seed": seed,
        "model_profiles": {slot: model.profile_hash for slot, model in models.items()},
        "solver_profiles": {
            slot: model.profile_hash for slot, model in continuation_models.items()
        },
        "evaluation_contract": evaluation_contract(spec, split=split),
        "calibration": {
            slot: digest(model.to_dict()) for slot, model in calibrators.items()
        },
        "agent_version": AGENT_VERSION,
        "agent_contracts": {
            slot: agent_contract(model.profile_hash) for slot, model in models.items()
        },
        "initial_call_limit": INITIAL_CALL_LIMIT,
        "total_call_limit": TOTAL_CALL_LIMIT,
        "shared_samples": SHARED_SAMPLES,
    }
    run = client.call(
        "create_run",
        {
            "protocol_version": SERVICE_PROTOCOL_VERSION,
            "split": split,
            "method_id": method_id + ":" + digest(config)[:16],
            "model_profiles": config["model_profiles"],
        },
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["config"] != config or state["run"]["run_id"] != run["run_id"]:
            raise ValueError("Output directory belongs to another run/configuration")
    else:
        state = {"config": config, "run": run, "episodes": {}}

    def save():
        atomic_json(state_path, state)

    save()
    if run["state"] in ("INITIAL_OPEN", "OPEN"):
        run = _initial_stage(
            client,
            models,
            calibrators,
            run,
            state,
            save,
            uq=uq,
            seed=seed,
            progress=progress,
        )
        state["run"] = run
        save()
    if initial_only:
        _write_trajectories(output_dir, models, state, "initial", "initial_snapshot")
        return {
            "run_id": run["run_id"],
            "state": run["state"],
            "initial_only": True,
            "state_file": str(state_path),
        }
    if run["state"] in ("INITIAL_SEALED", "OPEN"):
        run = _interactive_stage(
            client,
            models,
            calibrators,
            run,
            state,
            save,
            policy=policy,
            fixed_count=fixed_count,
            seed=seed,
            posterior_uq=posterior_uq,
            thresholds=thresholds,
            final_thresholds=final_thresholds,
            posterior_contracts=posterior_contracts,
            continuation_models=continuation_models,
        )
        state["run"] = run
        save()
    _write_trajectories(output_dir, models, state, "initial", "initial_snapshot")
    _write_trajectories(output_dir, models, state, "posterior", "posterior_snapshot")
    atomic_json(output_dir / "diagnostics.json", run_diagnostics(state))
    package_run(client, run["run_id"], output_dir / "submission.zip")
    return {
        "run_id": run["run_id"],
        "state": run["state"],
        "submission": str(output_dir / "submission.zip"),
    }
