"""Run the baseline over a list of cases and write trajectories and results.csv.

For each case: get an initial answer (run the agent, or reuse the examiner's
frozen test answer), estimate its risk, and on interactive cases optionally
query the oracle, let the agent continue, estimate both post-query risks and
choose AUTO/DEFER.

Progress is saved in ``<output>/cases/<case_id>.json`` after every agent step.
A finished case is never redone. An unfinished case resumes only if the
examiner returns the same episode again (test cases); before resuming, receipts
the examiner recorded after the last save are appended to the trajectory. An
unfinished labeled case simply starts again in a fresh episode.

``<output>/config.json`` records the method, models, data release and frozen
panel of the run; resuming with anything else is refused. Outputs for the
requested cases: ``initial.jsonl`` and ``posterior.jsonl`` (trajectories for
fitting) and ``results.csv`` (the submission format).

A trajectory ("snapshot") holds ``case_id, domain, model_id, messages,
history, usage, answer, p_error, samples, failure, cost, phase`` and, after
queries, ``initial_answer, p_initial_after, oracle_failure``. ``history``
lists ``model_response``, ``tool_receipt``, ``tool_error``, ``oracle_receipt``
and ``invalid_response`` events in order.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from test_client import RemoteError
from test_client.protocol import digest, valid_probability
from test_client.results import write_results

from .agent import (
    INITIAL_CALL_LIMIT,
    TERMINAL_TOOL_ERRORS,
    TOTAL_CALL_LIMIT,
    frozen_snapshot,
    main_loop,
    new_snapshot,
    shared_samples,
    visible_snapshot,
)
from .policies import POLICIES, final_mode, query_plan
from .storage import atomic_json, read_jsonl
from .uq import UQ_METHODS, raw_risk

QUERY_POLICIES = {"fixed", "all_facts", "active"}
LABELED_SPLITS = ("train", "cal_fit", "cal_select", "dev")
FROZEN_FILE = Path("test") / "initial_M9.jsonl"
CONTINUE_INSTRUCTION = (
    "Continue using the new evidence. Propose your own final answer and both "
    "self_reported_error_probability and self_reported_initial_error_probability."
)


def load_cases(data_dir, splits=LABELED_SPLITS):
    """Read public cases: ``train/<split>.jsonl`` for labeled splits, ``test/cases.jsonl``."""
    cases = []
    for split in splits:
        path = Path(data_dir) / ("test/cases.jsonl" if split == "test" else f"train/{split}.jsonl")
        cases.extend(read_jsonl(path))
    return cases


def balanced(cases, limit=None):
    """The first ``limit`` cases, taken in turn from each domain (file order kept)."""
    if not limit:
        return list(cases)
    queues = defaultdict(list)
    for case in cases:
        queues[case["domain"]].append(case)
    chosen, index = [], 0
    while len(chosen) < limit and any(index < len(q) for q in queues.values()):
        for domain in sorted(queues):
            if index < len(queues[domain]) and len(chosen) < limit:
                chosen.append(queues[domain][index])
        index += 1
    return chosen


def labeled_cases(data_dir, splits=LABELED_SPLITS, limit_per_split=None):
    """Labeled cases of ``splits``; with a limit, a domain-balanced prefix of each split."""
    return [
        case
        for split in splits
        for case in balanced(load_cases(data_dir, [split]), limit_per_split)
    ]


def load_frozen(data_dir):
    """The examiner's frozen initial test trajectories, keyed by case_id."""
    path = Path(data_dir) / FROZEN_FILE
    return {row["case_id"]: row for row in read_jsonl(path)} if path.exists() else {}


def frozen_digest(frozen):
    """Digest of a frozen panel; the examiner publishes the same value in ``spec``."""
    return digest([frozen[case_id] for case_id in sorted(frozen)])


def show(message):
    """Default progress output; flushed so redirected logs stay current."""
    print(message, flush=True)


def initial_risk(snapshot, uq, calibrator):
    """Risk of the initial answer; a missing answer is certainly wrong (1.0)."""
    if snapshot.get("answer") is None:
        return 1.0
    risk = calibrator.predict(snapshot, uq) if calibrator else raw_risk(snapshot, uq)
    return risk if valid_probability(risk) else None


def _model_id(model):
    """``model_id@profile_hash`` of a model client, recorded in config.json."""
    return None if model is None else f"{model.model_id}@{model.profile_hash}"


def run_cases(
    examiner,
    cases,
    output,
    *,
    model=None,
    continuation_model=None,
    frozen=None,
    uq="self_report",
    calibrator=None,
    policy="no_oracle",
    threshold=None,
    fixed_count=1,
    posterior_uq="auto",
    seed=0,
    initial_only=False,
    keep_messages=False,
    progress=show,
):
    """Run one method over ``cases`` and export its files; returns a summary.

    ``frozen`` maps test case IDs to frozen initial trajectories (they must
    match the examiner's panel). ``threshold`` overrides the DEFER thresholds
    selected in ``calibrator``. ``posterior_uq`` is ``self_report``,
    ``calibrated`` or ``auto`` (calibrated when fitted). ``keep_messages``
    keeps the chat in the exported initial trajectories (used for freezing).
    """
    if uq not in UQ_METHODS or policy not in POLICIES:
        raise ValueError(f"uq must be one of {UQ_METHODS}; policy one of {POLICIES}")
    if uq not in ("self_report", "disagreement") and calibrator is None:
        raise ValueError(f"uq={uq} needs a fitted calibrator")
    if type(fixed_count) is not int or fixed_count < 0:
        raise ValueError("fixed_count must be a non-negative integer")
    if threshold is not None and not valid_probability(threshold):
        raise ValueError("threshold must be a number in [0, 1]")
    if posterior_uq == "auto":
        posterior_uq = "calibrated" if calibrator and calibrator.posterior else "self_report"
    selections = calibrator.selections if calibrator else {}
    t_initial = threshold if threshold is not None else selections.get(uq, {}).get("threshold", 0.5)
    t_final = (
        threshold
        if threshold is not None
        else selections.get("posterior_final", {}).get("threshold", t_initial)
    )
    spec = examiner.spec()
    frozen = frozen or {}
    if frozen and spec.get("frozen_digest") and frozen_digest(frozen) != spec["frozen_digest"]:
        raise ValueError(
            "Your data/test/initial_M9.jsonl differs from this examiner's frozen panel; "
            "use the file from the current student package"
        )
    continuation_model = continuation_model or model
    output = Path(output)
    config = {
        "uq": uq,
        "policy": policy,
        "fixed_count": fixed_count,
        "posterior_uq": posterior_uq,
        "thresholds": [t_initial, t_final],
        "seed": seed,
        "initial_only": initial_only,
        "calibrator": calibrator.to_dict() if calibrator else None,
        "model": _model_id(model),
        "continuation_model": _model_id(continuation_model),
        "dataset_release_id": spec["dataset_release_id"],
        "frozen_digest": spec.get("frozen_digest"),
    }
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(
            f"{output} holds a run with another method, model or data release; use a new output"
        )
    atomic_json(config_path, config)
    test_ids, interactive_test = set(spec["test_ids"]), set(spec["interactive_ids"])
    method = {
        "uq": uq,
        "calibrator": calibrator,
        "policy": policy,
        "thresholds": (t_initial, t_final),
        "fixed_count": fixed_count,
        "posterior_uq": posterior_uq,
        "seed": seed,
    }
    for number, case in enumerate(cases, start=1):
        path = output / "cases" / f"{case['case_id']}.json"
        if path.exists() and json.loads(path.read_text()).get("stage", "done") == "done":
            continue
        is_test = case["case_id"] in test_ids
        record = _run_case(
            examiner,
            case,
            path,
            model=model,
            continuation_model=continuation_model,
            frozen=frozen.get(case["case_id"]),
            interactive=(case["case_id"] in interactive_test) if is_test else True,
            method=method,
            initial_only=initial_only,
            keep_messages=keep_messages,
        )
        row = record["row"]
        progress(
            f"[{number}/{len(cases)}] {case['case_id'][:8]} {case['domain']}: "
            f"p={row['p_initial_error']} mode={row.get('mode')} cost={row['cost']}"
        )
    return export(output, [case["case_id"] for case in cases])


def _run_case(
    examiner,
    case,
    path,
    *,
    model,
    continuation_model,
    frozen,
    interactive,
    method,
    initial_only,
    keep_messages,
):
    """Run (or resume) one case through its stages; returns the finished record.

    Stages: ``initial`` (answer and risk), ``query`` (oracle reads),
    ``continue`` (agent continues with the evidence) and ``done``. The record
    is saved after every agent step and stage change.
    """
    episode = examiner.episode(case["case_id"])
    episode_id = episode["episode_id"]
    record = json.loads(path.read_text()) if path.exists() else None
    if record is not None and record["episode_id"] == episode_id:
        snapshot = record["snapshot"]
        _recover(snapshot, examiner.history(episode_id))
    else:
        snapshot = frozen_snapshot(frozen, episode) if frozen else None
        if snapshot is None:
            if model is None:
                raise ValueError("A model client is needed to produce initial answers")
            snapshot = new_snapshot(episode, model)
        record = {
            "case_id": case["case_id"],
            "episode_id": episode_id,
            "stage": "initial",
            "snapshot": snapshot,
        }

    def save():
        """Write the case record (called after every agent step)."""
        atomic_json(path, record)

    save()
    if record["stage"] == "initial":
        if not frozen and snapshot["answer"] is None and not snapshot.get("failure"):
            main_loop(
                snapshot, model, examiner, limit=INITIAL_CALL_LIMIT, save=save, seed=method["seed"]
            )
        if not frozen:
            shared_samples(snapshot, model, save=save, seed=method["seed"])
        snapshot["initial_answer"] = snapshot["answer"]
        record["p_initial"] = initial_risk(snapshot, method["uq"], method["calibrator"])
        initial = visible_snapshot(snapshot)
        initial["status"] = (
            "completed"
            if snapshot["answer"] is not None and not snapshot.get("failure")
            else "failed"
        )
        if keep_messages:
            initial["messages"] = snapshot["messages"][1:]  # the system message is rebuilt
        record["initial"] = initial
        record["stage"] = "query" if interactive and not initial_only else "finish"
        save()
    if record["stage"] == "query":
        _query(examiner, record, snapshot, method, save)
    if record["stage"] == "continue":
        if continuation_model is None:
            raise ValueError("A model client is needed to continue after oracle queries")
        if snapshot["answer"] is None and not snapshot.get("failure"):
            main_loop(
                snapshot,
                continuation_model,
                examiner,
                limit=TOTAL_CALL_LIMIT,
                save=save,
                seed=method["seed"],
            )
        record["stage"] = "finish"
        save()
    record.update(_finish(examiner, record, snapshot, interactive and not initial_only, method))
    record.pop("snapshot")
    save()
    return record


def _query(examiner, record, snapshot, method, save):
    """Plan the oracle reads once, read them, and prepare the continuation."""
    if "plan" not in record:
        plan = []
        if (
            method["policy"] in QUERY_POLICIES
            and snapshot.get("failure") not in TERMINAL_TOOL_ERRORS
        ):
            p_initial = record["p_initial"]
            plan = query_plan(
                method["policy"],
                snapshot,
                examiner.history(record["episode_id"]),
                p_initial=p_initial if valid_probability(p_initial) else 1.0,
                threshold=method["thresholds"][0],
                fixed_count=method["fixed_count"],
            )
        record["plan"], record["receipts"] = [list(query) for query in plan], []
        save()
    for field, phase in record["plan"][len(record["receipts"]) :]:
        if snapshot.get("oracle_failure"):
            break
        receipt = _oracle_already_read(snapshot, field, phase)
        if receipt is None:
            try:
                receipt = examiner.oracle(record["episode_id"], field, phase)["receipt"]
            except RemoteError as exc:
                snapshot["oracle_failure"] = exc.code
                save()
                break
            snapshot["cost"] = round(snapshot["cost"] + receipt.get("cost", 0.0), 8)
            snapshot["history"].append({"kind": "oracle_receipt", "receipt": receipt})
        record["receipts"].append(receipt)
        save()
    if record["receipts"]:
        snapshot["messages"].append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "frozen_initial_answer": snapshot["initial_answer"],
                        "oracle_evidence": record["receipts"],
                        "instruction": CONTINUE_INSTRUCTION,
                    },
                    ensure_ascii=False,
                ),
            }
        )
        snapshot.update(answer=None, p_error=None, p_initial_after=None, failure=None)
        record["stage"] = "continue"
    else:
        record["stage"] = "finish"
    save()


def _oracle_already_read(snapshot, field, phase):
    """An oracle receipt for ``(field, phase)`` already in the trajectory, if any."""
    for event in snapshot["history"]:
        receipt = event.get("receipt", {})
        if event["kind"] == "oracle_receipt" and (receipt.get("field"), receipt.get("phase")) == (
            field,
            phase,
        ):
            return receipt
    return None


def _recover(snapshot, history):
    """Append receipts the examiner recorded after the last local save.

    The examiner's ledger is append-only and the trajectory's receipts are a
    prefix of it, so anything beyond that prefix was billed but never saved
    locally (for example after a crash right after a tool call). Tool results
    are shown to the agent again; oracle results reach it with the evidence.
    """
    known = sum(
        event["kind"] in ("tool_receipt", "oracle_receipt") for event in snapshot["history"]
    )
    for receipt in history["receipts"][known:]:
        oracle = receipt.get("authority") == "exact_fact"
        kind = "oracle_receipt" if oracle else "tool_receipt"
        snapshot["history"].append({"kind": kind, "receipt": receipt, "recovered": True})
        snapshot["cost"] = round(snapshot["cost"] + receipt.get("cost", 0.0), 8)
        if not oracle:
            snapshot["messages"].append(
                {
                    "role": "user",
                    "content": json.dumps({"tool_result": receipt}, ensure_ascii=False),
                }
            )


def _finish(examiner, record, snapshot, decide, method):
    """The case's results row and posterior trajectory."""
    p_initial = record["p_initial"]
    row = {
        "case_id": record["case_id"],
        "initial_answer_json": snapshot["initial_answer"],
        "p_initial_error": p_initial,
    }
    posterior = None
    if decide:
        receipts = record.get("receipts", [])
        final_answer, q, r = snapshot["initial_answer"], p_initial, p_initial
        if receipts:
            final_answer = snapshot["answer"]
            if method["posterior_uq"] == "calibrated":
                q = method["calibrator"].predict_posterior(snapshot, "initial")
                r = method["calibrator"].predict_posterior(snapshot, "final")
            else:
                q, r = snapshot.get("p_initial_after"), snapshot.get("p_error")
            if final_answer is None:
                r = 1.0
            posterior = visible_snapshot(snapshot)
        can_auto = final_answer is not None and valid_probability(r) and not snapshot.get("failure")
        threshold = method["thresholds"][1] if receipts else method["thresholds"][0]
        row.update(
            mode=final_mode(method["policy"], r, threshold) if can_auto else "DEFER",
            final_answer_json=final_answer,
            p_initial_error_after_queries=q,
            p_final_error=r,
        )
    row["cost"] = examiner.history(record["episode_id"])["cost"]
    return {"stage": "done", "row": row, "posterior": posterior}


def export(output, case_ids=None):
    """Write ``initial.jsonl``, ``posterior.jsonl`` and ``results.csv`` of finished cases.

    With ``case_ids`` only those cases are exported (in file-name order).
    """
    output = Path(output)
    records = [json.loads(p.read_text()) for p in sorted((output / "cases").glob("*.json"))]
    wanted = None if case_ids is None else set(case_ids)
    records = [
        r
        for r in records
        if r.get("stage", "done") == "done" and (wanted is None or r["case_id"] in wanted)
    ]
    for name in ("initial", "posterior"):
        lines = [json.dumps(r[name], ensure_ascii=False) + "\n" for r in records if r.get(name)]
        (output / f"{name}.jsonl").write_text("".join(lines))
    rows = [r["row"] for r in records]
    write_results(output / "results.csv", rows)
    modes = [row.get("mode") for row in rows if row.get("mode")]
    return {
        "output": str(output),
        "cases": len(rows),
        "posterior_trajectories": sum(bool(r.get("posterior")) for r in records),
        "auto": modes.count("AUTO"),
        "defer": modes.count("DEFER"),
        "total_cost": round(sum(row["cost"] for row in rows), 8),
        "results": str(output / "results.csv"),
    }
