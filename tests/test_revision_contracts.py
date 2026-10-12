"""Negative regressions for evidence, calibration, policy and fixed-agent repair."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sop_baselines.agent import main_loop, new_snapshot, shared_samples
from sop_baselines.contracts import (
    agent_contract,
    check_snapshot_contract,
    evaluation_contract,
)
from sop_baselines.policies import query_plan
from sop_baselines.runner import posterior_policy_contract, run_panel
from sop_baselines.uq import Calibrator, labeled_error_cost, select_cost_threshold
from sop_client.context import project_messages
from sop_protocol import INPUT_FIELDS, VERSION, digest
from test_public_candidate import FakeExaminer, FakeModel


def calibration_fixture():
    cases, snapshots = [], []
    contract = agent_contract(FakeModel.profile_hash)
    evaluation = evaluation_contract(FakeExaminer().spec())
    for index in range(18):
        split = ("train", "cal_fit", "cal_select")[index // 6]
        label = {"hazard_class": "Hazard Class B"}
        answer = {"hazard_class": "Hazard Class A" if index % 2 else "Hazard Class B"}
        cases.append(
            {
                "schema_version": VERSION,
                "dataset_release_id": "fixture-release",
                "case_id": str(index),
                "domain": "dangerous_goods",
                "inputs": {key: "fixture" for key in INPUT_FIELDS["dangerous_goods"]},
                "training_split": split,
                "label": label,
            }
        )
        snapshots.append(
            {
                "case_id": str(index),
                "domain": "dangerous_goods",
                "profile_hash": FakeModel.profile_hash,
                "answer": answer,
                "p_error": 0.8 if index % 2 else 0.1,
                "samples": [],
                "history": [],
                "cost": 0.01,
                "agent_contract": contract,
                "agent_contract_hash": digest(contract),
                "evaluation_contract": evaluation,
                "evaluation_contract_hash": digest(evaluation),
            }
        )
    return cases, snapshots


class RevisionContracts(unittest.TestCase):
    def test_uq_and_policy_changes_preserve_fixed_initial_but_invalidate_calibration(
        self,
    ):
        cases, snapshots = calibration_fixture()
        fitted = Calibrator().fit(cases[:12], snapshots[:12])
        initial_contract = agent_contract(FakeModel.profile_hash)
        query_contract = posterior_policy_contract("fixed", "self_report", 0.5, 1)
        original_read = Path.read_text

        def revised(path, *args, **kwargs):
            text = original_read(path, *args, **kwargs)
            if path.name in ("uq.py", "policies.py"):
                return text + "\n# simulated participant implementation revision\n"
            return text

        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / "calibration.json"
            fitted.save(saved)
            with patch.object(Path, "read_text", revised):
                self.assertEqual(
                    agent_contract(FakeModel.profile_hash), initial_contract
                )
                check_snapshot_contract(snapshots[0])
                self.assertNotEqual(
                    posterior_policy_contract("fixed", "self_report", 0.5, 1),
                    query_contract,
                )
                with self.assertRaisesRegex(ValueError, "estimator implementation"):
                    Calibrator.load(saved)

    def test_projection_preserves_business_identity_and_is_idempotent(self):
        receipt = {
            "receipt_id": "transport",
            "episode_id": "transport",
            "request_id": "transport",
            "phase": "pre_repair",
            "tool": "diagnostics",
            "observation_id": "business-observation",
            "source": "independent",
            "timestamp": "2026-10-11T10:00:00Z",
            "event_seq": 4,
            "cost": 0.03,
            "payload": {
                "latency": 0,
                "aircraft_ready": False,
                "request_id": "business-value",
            },
        }
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {"tool_result": receipt, "request_id": "input-id"}, indent=2
                ),
            }
        ]
        projected = project_messages(messages)
        result = json.loads(projected[0]["content"])
        self.assertEqual(result["request_id"], "input-id")
        self.assertEqual(result["tool_result"]["payload"], receipt["payload"])
        self.assertEqual(
            result["tool_result"]["observation_id"], "business-observation"
        )
        self.assertEqual(result["tool_result"]["timestamp"], receipt["timestamp"])
        self.assertEqual(result["tool_result"]["source"], receipt["source"])
        self.assertEqual(result["tool_result"]["cost"], 0.03)
        self.assertEqual(result["tool_result"]["event_seq"], 4)
        self.assertNotIn("receipt_id", result["tool_result"])
        self.assertEqual(project_messages(projected), projected)
        self.assertLess(len(projected[0]["content"]), len(messages[0]["content"]))
        self.assertEqual(json.loads(messages[0]["content"])["tool_result"], receipt)

    def test_every_agent_call_projects_evidence_including_oracle_and_samples(self):
        class InspectModel(FakeModel):
            def generate(self, messages, **kwargs):
                for message in messages:
                    self_outer.assertNotIn('"receipt_id"', message["content"])
                return super().generate(messages, **kwargs)

        self_outer = self
        with tempfile.TemporaryDirectory() as tmp:
            run_panel(
                FakeExaminer(),
                {"M9": InspectModel()},
                tmp,
                policy="fixed",
                uq="self_report",
                threshold=0.4,
                progress=lambda _: None,
            )
            state = json.loads((Path(tmp) / "state.json").read_text())
            snapshot = next(iter(state["episodes"].values()))["snapshot"]
            self.assertTrue(
                any("receipt_id" in e.get("receipt", {}) for e in snapshot["history"])
            )

    def test_false_and_zero_are_valid_observations_outside_dangerous_goods(self):
        history = {
            "view": {"remaining_budget": 2.0, "budget": 2.0, "costs": {"oracle": 0.1}},
            "receipts": [
                {
                    "phase": "assessment",
                    "payload": {
                        "aircraft_ready": False,
                        "electrical_inspection_result": "success",
                        "cross_check_response": "success",
                    },
                }
            ],
        }
        self.assertEqual(
            query_plan(
                "active",
                {"domain": "aircraft_inspection"},
                history,
                p_initial=0.9,
                fixed_count=1,
            ),
            [("mechanical_inspection_result", "assessment")],
        )
        history["receipts"] = [
            {
                "tool": "performTechnicalDiagnostics",
                "phase": "pre_repair",
                "payload": {"service metrics": {"latency": 0, "bandwidth": 100}},
            }
        ]
        self.assertEqual(
            query_plan(
                "active",
                {"domain": "customer_service"},
                history,
                p_initial=0.9,
                fixed_count=1,
            ),
            [("jitter", "pre_repair")],
        )
        history["receipts"] = [
            {
                "phase": "assessment",
                "payload": {
                    "sds_label_score": 2,
                    "handling_score": 0,
                    "transportation_score": 2,
                    "disposal_score": 2,
                },
            }
        ]
        self.assertEqual(
            query_plan(
                "active",
                {"domain": "dangerous_goods"},
                history,
                p_initial=0.9,
                fixed_count=1,
            ),
            [("handling", "assessment")],
        )

    def test_calibrator_rejects_legacy_and_stale_contracts(self):
        cases, snapshots = calibration_fixture()
        missing = copy.deepcopy(snapshots)
        missing[0].pop("agent_contract_hash")
        with self.assertRaisesRegex(ValueError, "contract"):
            Calibrator().fit(cases, missing)
        calibrator = Calibrator().fit(cases, snapshots)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibrator.json"
            calibrator.save(path)
            loaded = Calibrator.load(path)
            changed = copy.deepcopy(snapshots[0])
            changed["agent_contract"]["projection_version"] = "different"
            changed["agent_contract_hash"] = digest(changed["agent_contract"])
            with self.assertRaisesRegex(ValueError, "differs"):
                loaded.predict(changed, "supervised")
            value = json.loads(path.read_text())
            value["version"] = "2.0"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "legacy"):
                Calibrator.load(path)

    def test_threshold_selection_rejects_train_and_accounts_for_severity(self):
        cases, snapshots = calibration_fixture()
        calibrator = Calibrator().fit(cases, snapshots)
        with self.assertRaisesRegex(ValueError, "holdout"):
            calibrator.select_thresholds(cases, snapshots[:6])
        calibrator.select_thresholds(cases, snapshots[12:])
        selected = calibrator.selections["self_report"]
        self.assertAlmostEqual(selected["coverage"], 0.5)
        self.assertFalse(selected["degenerate_all_defer"])
        self.assertEqual(
            labeled_error_cost(
                {"hazard_class": "Hazard Class A"},
                {"hazard_class": "Hazard Class B"},
                "dangerous_goods",
            ),
            5,
        )
        self.assertEqual(
            labeled_error_cost(
                {"hazard_class": "Hazard Class C"},
                {"hazard_class": "Hazard Class B"},
                "dangerous_goods",
            ),
            1,
        )
        # Identical binary risks can have different optimal decisions because severity differs.
        row = (cases[0], {**snapshots[0], "answer": {"hazard_class": "Hazard Class A"}})
        self.assertTrue(select_cost_threshold([row], [0.2])["degenerate_all_defer"])

    def test_posterior_targets_remain_distinct_and_policy_is_bound(self):
        cases, snapshots = calibration_fixture()
        posterior = copy.deepcopy(snapshots[:12])
        contract = posterior_policy_contract("fixed", "calibrated_disagreement", 0.2, 1)
        for index, s in enumerate(posterior):
            s.update(
                initial_answer=s["answer"],
                initial_status="completed",
                answer=cases[index]["label"],
                p_initial_after=0.8,
                p_error=0.1,
                posterior_contract=contract,
            )
        # A correct A0 with missing initial risk was committed failed. It cannot
        # supervise p0|T, while its independently completed final answer remains.
        posterior[6]["initial_status"] = "failed"
        calibrated = Calibrator().fit(cases, snapshots, posterior_snapshots=posterior)
        self.assertNotIn("6", calibrated.posterior_fit_cases_by_target["initial"])
        self.assertIn("6", calibrated.posterior_fit_cases_by_target["final"])
        self.assertGreater(
            calibrated.predict_posterior(posterior[6], "initial"),
            calibrated.predict_posterior(posterior[6], "final"),
        )
        changed = copy.deepcopy(posterior[6])
        changed["posterior_contract"]["fixed_count"] = 2
        with self.assertRaisesRegex(ValueError, "query policy"):
            calibrated.predict_posterior(changed, "final")
        with tempfile.TemporaryDirectory() as tmp:
            calibrated.save(Path(tmp) / "fitted.json")
            loaded = Calibrator.load(Path(tmp) / "fitted.json")
            self.assertAlmostEqual(
                loaded.predict_posterior(posterior[6], "final"),
                calibrated.predict_posterior(posterior[6], "final"),
            )

    def test_queries_reject_silent_posterior_estimator_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel()
            with self.assertRaisesRegex(ValueError, "posterior calibration"):
                run_panel(
                    FakeExaminer(), {"M9": model}, tmp, policy="fixed", threshold=0.2
                )
            # Initial evidence can be prepared, but no paid oracle may happen
            # before a needed post-query calibration file has been checked.
            self.assertTrue(model.calls)
            with self.assertRaisesRegex(ValueError, "Training|training"):
                run_panel(FakeExaminer(), {"M9": model}, tmp, collect_posterior=True)

    def test_active_posterior_binds_initial_estimator_parameters(self):
        cases, snapshots = calibration_fixture()
        calibrator = Calibrator().fit(cases, snapshots)
        contract = posterior_policy_contract(
            "active",
            "calibrated_disagreement",
            0.2,
            1,
            trigger_estimator_hash=calibrator.initial_estimator_hash(
                "calibrated_disagreement"
            ),
        )
        posterior = copy.deepcopy(snapshots[:12])
        for snapshot in posterior:
            snapshot.update(
                initial_answer=snapshot["answer"],
                initial_status="completed",
                p_initial_after=0.5,
                posterior_contract=contract,
            )
        calibrator.fit_posterior(cases, posterior)
        original_hash = calibrator.initial_estimator_hash("calibrated_disagreement")
        calibrator.platt["disagreement"].weights[0] += 1.0
        self.assertNotEqual(
            calibrator.initial_estimator_hash("calibrated_disagreement"), original_hash
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "Query policy differs"),
        ):
            run_panel(
                FakeExaminer(),
                {"M9": FakeModel()},
                tmp,
                policy="active",
                uq="calibrated_disagreement",
                threshold=0.2,
                calibrators={"M9": calibrator},
            )

    def test_initial_only_seals_without_oracle_or_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeExaminer()
            result = run_panel(
                client,
                {"M9": FakeModel()},
                tmp,
                initial_only=True,
                progress=lambda _: None,
            )
            self.assertEqual(result["state"], "INITIAL_SEALED")
            self.assertIsNone(client.final)
            self.assertFalse((Path(tmp) / "submission.zip").exists())

    def test_trusted_initial_is_reused_without_candidate_generation(self):
        source, source_model = FakeExaminer(), FakeModel()
        episode = source.call("start_episode", {})
        snapshot = new_snapshot(episode, source_model)
        main_loop(snapshot, source_model, source, limit=20, save=lambda: None)
        shared_samples(snapshot, source_model, save=lambda: None)

        class FrozenExaminer(FakeExaminer):
            def call(self, operation, body):
                result = super().call(operation, body)
                if operation in ("history", "start_episode"):
                    result["initial_proposal"] = {
                        "frozen": True,
                        "answer": snapshot["answer"],
                        "status": "completed",
                        "snapshot": copy.deepcopy(snapshot),
                    }
                if operation == "tools":
                    raise AssertionError(
                        "Candidate must not repeat trusted initial tools"
                    )
                return result

        class NoGenerate(FakeModel):
            def generate(self, *args, **kwargs):
                raise AssertionError(
                    "Trusted initial model output must not be regenerated"
                )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_panel(
                FrozenExaminer(), {"M9": NoGenerate()}, tmp, progress=lambda _: None
            )
            self.assertEqual(result["state"], "FINALIZED")
            state = json.loads((Path(tmp) / "state.json").read_text())
            entry = next(iter(state["episodes"].values()))
            self.assertTrue(entry["trusted_initial"])
            self.assertEqual(entry["initial_commit"]["answer"], snapshot["answer"])


if __name__ == "__main__":
    unittest.main()
