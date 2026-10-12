"""Public-only regressions for risk locking, calibration-file checks and report checks.

The fake model below is a protocol fixture, never evidence of real-model quality.
Real service transaction and five-version tests live on the examiner side.
"""

import copy
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from sop_baselines.agent import main_loop, new_snapshot, shared_samples
from sop_baselines.contracts import (
    agent_contract,
    evaluation_contract,
    risk_estimator_contract,
)
from sop_baselines.policies import query_plan
from sop_baselines.runner import _check_calibrators, run_panel
from sop_baselines.uq import Calibrator, select_cost_threshold
from sop_client.client import ExaminerClient, check_service_capabilities
from sop_client.report_check import check_public_report
from sop_protocol import (
    DATA_SCHEMA_VERSION,
    SERVICE_PROTOCOL_VERSION,
    SUBMISSION_SCHEMA_VERSION,
    decision_terminal_loss,
    digest,
    read_submission,
)
from test_public_candidate import FakeExaminer, FakeModel
from test_revision_contracts import calibration_fixture
from test_model_admission_retry import serving


def test_effective_cost_budget_and_schema_binding(tmp_path):
    cases, snapshots = calibration_fixture()
    fitted = Calibrator().fit(cases, snapshots)
    fitted.select_thresholds(cases, snapshots[12:])
    path = tmp_path / "fit.json"
    fitted.save(path)
    loaded = Calibrator.load(path)
    spec = FakeExaminer().spec()
    models = {"M9": FakeModel()}
    _check_calibrators(models, {"M9": loaded}, spec, "supervised")
    for changed in (
        {**spec, "budget": 0.5},
        {**spec, "costs": {**spec["costs"], "defer": 0.1}},
        {**spec, "oracle_schema": {"changed": True}},
    ):
        with pytest.raises(ValueError, match="evaluation configuration"):
            _check_calibrators(models, {"M9": loaded}, changed, "supervised")
    missing = copy.deepcopy(snapshots)
    missing[0].pop("evaluation_contract")
    with pytest.raises(ValueError, match="evaluation contract"):
        Calibrator().fit(cases, missing)


def test_shared_protocol_implementation_invalidates_both_contracts(tmp_path):
    cases, snapshots = calibration_fixture()
    fitted = Calibrator().fit(cases, snapshots)
    saved = tmp_path / "fit.json"
    fitted.save(saved)
    before_agent = agent_contract(FakeModel.profile_hash)
    before_risk = risk_estimator_contract()
    read = Path.read_text

    def revised(path, *args, **kwargs):
        value = read(path, *args, **kwargs)
        return (
            value + "\n# changed public validator\n"
            if path.parent.name == "sop_protocol"
            else value
        )

    with patch.object(Path, "read_text", revised):
        assert agent_contract(FakeModel.profile_hash) != before_agent
        assert risk_estimator_contract() != before_risk
        with pytest.raises(ValueError, match="estimator implementation"):
            Calibrator.load(saved)


def test_all_auto_active_needs_no_posterior(tmp_path):
    cases, snapshots = calibration_fixture()
    for case, snapshot in zip(cases, snapshots):
        case["label"] = copy.deepcopy(snapshot["answer"])
        snapshot["p_error"] = 0.2
    fitted = Calibrator().fit(cases[:12], snapshots[:12])
    fitted.select_thresholds(cases[12:], snapshots[12:])
    assert fitted.selections["calibrated_disagreement"]["degenerate_all_auto"]
    for collect in (True, False):
        client = FakeExaminer()
        path = tmp_path / str(collect)
        result = run_panel(
            client,
            {"M9": FakeModel()},
            path,
            split="train",
            policy="active",
            uq="calibrated_disagreement",
            calibrators={"M9": fitted},
            collect_posterior=collect,
            progress=lambda _: None,
        )
        assert result["state"] == "FINALIZED"
        assert not any(receipt.get("field") for receipt in client.receipts)
        assert (path / "M9-posterior.jsonl").read_text() == ""
        trajectory = json.loads((path / "M9-initial.jsonl").read_text())
        assert trajectory["evaluation_contract"] == fitted.evaluation_contract


def test_fit_cli_accepts_actual_empty_posterior_files(tmp_path):
    cases, snapshots = calibration_fixture()
    pool = tmp_path / "data"
    pool.mkdir()
    (pool / "pool.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    initial = tmp_path / "initial.jsonl"
    initial.write_text("".join(json.dumps(s) + "\n" for s in snapshots))
    empty = tmp_path / "empty-posterior.jsonl"
    empty.write_text("")
    saved = tmp_path / "calibration.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sop_baselines.cli",
            "fit",
            "--data",
            str(pool),
            "--snapshots",
            str(initial),
            "--selection-snapshots",
            str(initial),
            "--posterior-snapshots",
            str(empty),
            "--posterior-selection-snapshots",
            str(empty),
            "--output",
            str(saved),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["posterior_calibration_count"] == 0
    loaded = Calibrator.load(saved)
    assert loaded.posterior == {}
    assert loaded.selections


def test_actual_query_refuses_missing_posterior_before_charging(tmp_path):
    client = FakeExaminer()
    with pytest.raises(ValueError, match="posterior calibration"):
        run_panel(
            client,
            {"M9": FakeModel()},
            tmp_path,
            policy="fixed",
            threshold=0.2,
            progress=lambda _: None,
        )
    assert not any(receipt.get("field") for receipt in client.receipts)


def test_controller_penalty_and_configured_defer_cost():
    cases, snapshots = calibration_fixture()
    broken = {**snapshots[0], "failure": "protocol_failure_limit", "cost": 0.06}
    actual = select_cost_threshold([(cases[0], broken)], [0.2], defer_cost=0.7)
    assert actual["domain_macro_loss"] == pytest.approx(5.06)
    model_failure = {**broken, "failure": "model_transport_failure"}
    ordinary = select_cost_threshold([(cases[0], model_failure)], [0.2], defer_cost=0.7)
    assert ordinary["domain_macro_loss"] == pytest.approx(0.76)
    assert decision_terminal_loss("DEFER", controller_valid=False, defer_cost=0.7) == 5
    assert decision_terminal_loss("DEFER", answer_present=False, defer_cost=0.7) == 0.7
    mixed = [(cases[0], snapshots[0])] * 4 + [
        (cases[1], {**snapshots[1], "answer": {"hazard_class": "Hazard Class C"}})
    ]
    assert select_cost_threshold(mixed, [0.4] * 5, defer_cost=0.5)[
        "degenerate_all_auto"
    ]
    assert select_cost_threshold(mixed, [0.4] * 5, defer_cost=0.1)[
        "degenerate_all_defer"
    ]


def test_local_report_scanner_has_no_truth_or_credential_echo(tmp_path):
    source = tmp_path / "report.md"
    secret = "synthetic-secret-for-test-only-123456789"
    source.write_text(
        f"Authorization: Bearer {secret}\ncase_id,oracle_value\nfictional-case,999\n"
    )
    result = check_public_report([source])
    assert result["status"] == "review_required"
    assert not result["uses_private_labels"]
    assert secret not in json.dumps(result)
    assert "999" not in json.dumps(result)
    source.write_text(source.read_text().replace("999", "123"))
    assert (
        check_public_report([source]) == result
    )  # value-independent, no answer oracle
    source.write_text("Aggregate Brier: 0.20. All examples are synthetic.\n")
    assert check_public_report([source])["status"] == "no_pattern_detected"
    picture = tmp_path / "report.pdf"
    picture.write_bytes(b"%PDF fake")
    assert check_public_report([picture])["unscanned"]


def test_old_capabilities_rejected_and_data_version_unchanged():
    assert DATA_SCHEMA_VERSION == "2.1"
    assert SERVICE_PROTOCOL_VERSION == SUBMISSION_SCHEMA_VERSION == "2.2"
    with pytest.raises(ValueError, match="protocol 2.2"):
        check_service_capabilities({"protocol_version": "2.1"})
    check_service_capabilities(FakeExaminer().spec())


def test_lost_dev_restart_response_does_not_create_more_practices():
    created = []

    class DropResponse(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            assert self.path == "/v2/rpc/restart_dev"
            created.append("practice created before response loss")
            self.close_connection = True

    with serving(ThreadingHTTPServer(("127.0.0.1", 0), DropResponse)) as url:
        client = ExaminerClient(url, "synthetic-only-token", retries=3)
        with pytest.raises(OSError):
            client.call("restart_dev", {"run_id": "synthetic-practice"})
    assert len(created) == 1


def test_dev_unlimited_oracle_plan_accepts_null_budget():
    history = {
        "view": {"budget": None, "remaining_budget": None, "costs": {"oracle": 0.1}},
        "receipts": [],
    }
    assert (
        len(
            query_plan(
                "all_facts", {"domain": "dangerous_goods"}, history, p_initial=0.9
            )
        )
        == 4
    )
    spec = FakeExaminer().spec()
    effective = evaluation_contract(spec, split="dev")
    assert effective["budget"] is None
    assert effective["oracle_limit"] is None
    assert effective["tool_limit"] == 24
    cases, snapshots = calibration_fixture()
    fitted = Calibrator().fit(cases, snapshots)
    with pytest.raises(ValueError, match="evaluation configuration"):
        _check_calibrators(
            {"M9": FakeModel()}, {"M9": fitted}, spec, "supervised", split="dev"
        )


def test_solver_profile_does_not_change_official_initial_slot(tmp_path):
    class OtherSolver(FakeModel):
        profile_hash = digest({"test_only_solver": "different"})

    initial, solver = FakeModel(), OtherSolver()
    result = run_panel(
        FakeExaminer(),
        {"M9": initial},
        tmp_path,
        policy="fixed",
        uq="self_report",
        threshold=0.1,
        continuation_models={"M9": solver},
        progress=lambda _: None,
    )
    manifest, _ = read_submission(result["submission"])
    assert manifest["model_profiles"] == {"M9": initial.profile_hash}
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["config"]["solver_profiles"] == {"M9": solver.profile_hash}
    assert solver.calls
    assert len(initial.calls) == 6
    posterior = json.loads((tmp_path / "M9-posterior.jsonl").read_text())
    assert posterior["solver_profile_hash"] == solver.profile_hash
    assert posterior["posterior_contract"]["solver_profile_hash"] == solver.profile_hash


def test_formal_runner_checkpoints_per_case_and_packages_before_close(tmp_path):
    source, reference_model = FakeExaminer(), FakeModel()
    snapshot = new_snapshot(source.call("start_episode", {}), reference_model)
    main_loop(snapshot, reference_model, source, limit=20, save=lambda: None)
    shared_samples(snapshot, reference_model, save=lambda: None)
    snapshot["evaluation_contract"] = evaluation_contract(source.spec())
    snapshot["evaluation_contract_hash"] = digest(snapshot["evaluation_contract"])

    class FormalExaminer(FakeExaminer):
        def __init__(self):
            super().__init__()
            self.state = "OPEN"
            self.seq, self.cost = source.seq, source.cost
            self.receipts = copy.deepcopy(source.receipts)
            self.checkpointed = False

        def run(self):
            return {**super().run(), "evaluation_mode": "formal"}

        def call(self, operation, body):
            assert operation not in ("seal_initial", "finalize")
            if operation == "checkpoint_initial":
                self.checkpointed = True
                return {
                    "case_id": "fixture-case",
                    "checkpoint_id": "checkpoint",
                    "initials": {"M9": self.initial},
                    "event_seq": self.seq,
                }
            if operation == "oracle":
                assert self.checkpointed
                self.state = "INITIAL_SEALED"  # adapt the legacy public double only
                try:
                    return super().call(operation, body)
                finally:
                    self.state = "OPEN"
            if operation == "get_run" and self.final is None:
                return {**self.run(), "rows": []}
            result = super().call(operation, body)
            if operation in ("start_episode", "history"):
                result["initial_proposal"] = {
                    "frozen": True,
                    "answer": snapshot["answer"],
                    "status": "completed",
                    "snapshot": copy.deepcopy(snapshot),
                }
            return result

    class NoReferenceGeneration(FakeModel):
        def generate(self, *args, **kwargs):
            raise AssertionError("fixed A0 must not be regenerated")

    client = FormalExaminer()
    result = run_panel(
        client,
        {"M9": NoReferenceGeneration()},
        tmp_path,
        policy="fixed",
        uq="self_report",
        threshold=0.1,
        continuation_models={"M9": FakeModel()},
        progress=lambda _: None,
    )
    assert result["state"] == "OPEN"
    assert client.checkpointed
    manifest, rows = read_submission(result["submission"])
    assert manifest["protocol_version"] == "2.2"
    assert rows[0]["initial_answer_json"] == snapshot["answer"]
