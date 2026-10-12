"""CPU contract tests. Fake outputs below are NOT Qwen predictions or scores."""

import copy
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from sop_baselines.contracts import agent_contract, evaluation_contract
from sop_baselines.runner import run_panel
from sop_baselines.uq import Calibrator, disagreement, features, raw_risk
from sop_client import ModelClient, RemoteError
from sop_client.client import HTTPClient
from sop_protocol import (
    CAPABILITIES,
    SERVICE_PROTOCOL_VERSION,
    INPUT_FIELDS,
    VERSION,
    digest,
    read_submission,
)
from sop_qwen.assets import verify_assets
from sop_qwen.server import make_server


class FakeModel:
    model_id = "test-only-fake-not-qwen"
    configuration: ClassVar[dict] = {"model_id": model_id, "backend": "test-only-fake"}
    profile_hash = digest(configuration)

    def __init__(self, missing_probability=False):
        self.calls = []
        self.missing_probability = missing_probability

    def generate(self, messages, *, sample=False, seed=0, max_tokens=512):
        self.calls.append({"sample": sample, "seed": seed})
        if not sample and not any('"tool_result"' in m["content"] for m in messages):
            obj = {
                "type": "tool_call",
                "tool": "calculate_sds_label_score",
                "arguments": {"product_id": "P_10001"},
            }
        else:
            obj = {
                "type": "final_proposal",
                "result": {"hazard_class": "Hazard Class B"},
                "self_reported_error_probability": 0.2,
                "self_reported_initial_error_probability": 0.3,
            }
            if self.missing_probability:
                obj.pop("self_reported_error_probability")
                obj.pop("self_reported_initial_error_probability")
        return {
            "text": json.dumps(obj),
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            "profile_hash": self.profile_hash,
            "finish_reason": "stop",
        }


class FakeExaminer:
    """Small public protocol double; no private state or target labels are present."""

    def __init__(self):
        self.state = "INITIAL_OPEN"
        self.seq, self.receipts = 0, []
        self.initial = self.final = None
        self.cost = 0.0
        self.requests = {}
        self.view = {
            "case_id": "fixture-case",
            "domain": "dangerous_goods",
            "inputs": {"product_id": "P_10001"},
            "tools": [],
            "phase": "assessment",
            "costs": {
                "ordinary": 0.01,
                "independent": 0.03,
                "oracle": 0.1,
                "defer": 0.5,
            },
            "budget": 2.0,
        }

    def spec(self):
        return {
            "protocol_version": SERVICE_PROTOCOL_VERSION,
            "capabilities": list(CAPABILITIES),
            "model_slots": ["M9"],
            "dataset_release_id": "fixture-release",
            "budget": self.view["budget"],
            "costs": self.view["costs"],
            "tool_limit": 24,
            "oracle_limit": 16,
            "oracle_schema": {},
            "tool_schema_hash": digest({"test_only_tools": []}),
            "dev_policy": {"budget": None, "oracle_limit": None},
        }

    def run(self):
        return {
            "run_id": "fixture-run",
            "state": self.state,
            "case_ids": ["fixture-case"],
            "interactive_ids": ["fixture-case"],
            "model_slots": ["M9"],
            "release_id": "fixture-release",
            "evaluation_profile_id": "fixture-profile",
            "method_id": self.method,
            "model_profiles": self.profiles,
        }

    def call(self, operation, body):
        if operation == "create_run":
            self.method, self.profiles = body["method_id"], body["model_profiles"]
            return self.run()
        if operation == "start_episode":
            return {
                "episode_id": "fixture-episode",
                "view": self.view,
                "event_seq": self.seq,
                "interactive": True,
            }
        if operation == "history":
            return {
                "event_seq": self.seq,
                "receipts": self.receipts,
                "view": {**self.view, "remaining_budget": 2 - self.cost},
                "initial": self.initial,
                "final": self.final,
            }
        if operation in ("tools", "oracle"):
            if operation == "oracle" and self.state != "INITIAL_SEALED":
                raise AssertionError("Oracle before initial-panel barrier")
            if body["request_id"] in self.requests:
                return self.requests[body["request_id"]]
            self.seq += 1
            cost = 0.01 if operation == "tools" else 0.1
            receipt = {
                "tool": body.get("name"),
                "field": body.get("field"),
                "phase": "assessment",
                "payload": {"sds_label_score": 2},
                "cost": cost,
                "receipt_id": str(self.seq),
            }
            self.cost += cost
            self.receipts.append(receipt)
            result = {"receipt": receipt, "event_seq": self.seq}
            self.requests[body["request_id"]] = result
            return result
        if operation in ("initial", "final"):
            assert body["event_seq"] == self.seq
            result = {k: v for k, v in body.items() if k != "episode_id"}
            result["commit_id"] = operation + "-commit"
            setattr(self, operation, result)
            return result
        if operation == "seal_initial":
            assert self.initial
            self.state = "INITIAL_SEALED"
            return self.run()
        if operation == "finalize":
            assert self.final
            self.state = "FINALIZED"
            return self.run()
        if operation == "get_run":
            initial, final = self.initial, self.final
            row = {
                "case_id": "fixture-case",
                "model_slot": "M9",
                "episode_id": "fixture-episode",
                "initial_commit_id": initial["commit_id"],
                "initial_answer_json": initial["answer"],
                "p_initial_error": initial["p_initial_error"],
                "mode": final["mode"],
                "final_commit_id": final["commit_id"],
                "final_answer_json": final["answer"],
                "p_initial_error_after_queries": final["p_initial_error_after_queries"],
                "p_final_error": final["p_final_error"],
                "status": (
                    "failed"
                    if "failed" in (initial["status"], final["status"])
                    else "completed"
                ),
            }
            return {**self.run(), "rows": [row]}
        raise AssertionError(operation)


class PublicCandidateTests(unittest.TestCase):
    def test_bearer_client_does_not_follow_redirect(self):
        paths = []

        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                paths.append(self.path)
                self.send_response(302)
                self.send_header("Location", "/other-recipient")
                self.send_header("Content-Length", "0")
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = HTTPClient(
                f"http://127.0.0.1:{server.server_port}", "private-token"
            )
            with self.assertRaises(RemoteError):
                client.request("/original")
            self.assertEqual(paths, ["/original"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_model_http_authentication_and_remote_profile(self):
        fake = FakeModel()
        temporary = tempfile.TemporaryDirectory()
        audit = Path(temporary.name) / "model-calls.jsonl"
        server = make_server(
            fake, fake.configuration, "m" * 24, port=0, audit_path=audit
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            client = ModelClient(url, "m" * 24, expected_profile_hash=fake.profile_hash)
            result = client.generate([{"role": "user", "content": "fixture"}])
            self.assertEqual(result["profile_hash"], fake.profile_hash)
            self.assertNotIn("model_path", json.dumps(client.configuration))
            with self.assertRaises(RemoteError):
                ModelClient(url, "wrong")
            with self.assertRaises(ValueError):
                ModelClient(url, "m" * 24, expected_profile_hash="not-this-model")
            events = [json.loads(line) for line in audit.read_text().splitlines()]
            self.assertEqual(
                [e["event"] for e in events], ["server_started", "generation"]
            )
            self.assertEqual(events[-1]["sequence"], 1)
            self.assertEqual(events[-1]["status"], "completed")
            self.assertGreaterEqual(events[-1]["latency_seconds"], 0.0)
            self.assertNotIn("messages", events[-1])
            self.assertFalse(events[-1]["model_loaded"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            temporary.cleanup()

    def test_agent_tools_four_samples_and_active_submission(self):
        for policy in (
            "no_oracle",
            "all_defer",
            "threshold",
            "fixed",
            "all_facts",
            "active",
        ):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                model, client = FakeModel(), FakeExaminer()
                result = run_panel(
                    client,
                    {"M9": model},
                    tmp,
                    policy=policy,
                    uq="self_report",
                    threshold=0.1,
                    progress=lambda _: None,
                )
                self.assertEqual(result["state"], "FINALIZED")
                self.assertEqual(sum(c["sample"] for c in model.calls), 4)
                self.assertLessEqual(sum(not c["sample"] for c in model.calls), 32)
                manifest, rows = read_submission(result["submission"])
                self.assertEqual(rows[0]["status"], "completed")
                self.assertEqual(manifest["model_profiles"]["M9"], model.profile_hash)
                before = len(model.calls)
                run_panel(
                    client,
                    {"M9": model},
                    tmp,
                    policy=policy,
                    uq="self_report",
                    threshold=0.1,
                    progress=lambda _: None,
                )
                self.assertEqual(len(model.calls), before)

    def test_missing_probability_stays_failed_not_imputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_panel(
                FakeExaminer(),
                {"M9": FakeModel(missing_probability=True)},
                tmp,
                uq="self_report",
                policy="no_oracle",
                progress=lambda _: None,
            )
            _, rows = read_submission(result["submission"])
            self.assertIsNone(rows[0]["p_initial_error"])
            self.assertIsNone(rows[0]["p_final_error"])
            self.assertEqual(rows[0]["status"], "failed")

    def test_initial_model_limit_keeps_failure_in_complete_run(self):
        class InvalidModel(FakeModel):
            def generate(self, messages, **kwargs):
                self.calls.append(kwargs)
                return {"text": "not-json", "usage": {}}

        with tempfile.TemporaryDirectory() as tmp:
            model = InvalidModel()
            result = run_panel(
                FakeExaminer(), {"M9": model}, tmp, progress=lambda _: None
            )
            _, rows = read_submission(result["submission"])
            self.assertEqual(len(model.calls), 20)
            self.assertEqual(result["state"], "FINALIZED")
            self.assertEqual(rows[0]["status"], "failed")
            self.assertIsNone(rows[0]["initial_answer_json"])

    def test_lost_checkpoint_rejects_existing_server_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeExaminer()
            client.seq = 1
            client.receipts = [{"cost": 0.01}]
            with self.assertRaisesRegex(ValueError, "Missing local checkpoint"):
                run_panel(client, {"M9": FakeModel()}, tmp, progress=lambda _: None)
            self.assertEqual(client.state, "INITIAL_OPEN")

    def test_shared_disagreement_counts_invalid_samples(self):
        self.assertEqual(
            disagreement(
                {
                    "answer": {"x": 1},
                    "samples": [{"answer": {"x": 1}}, {"answer": None}],
                }
            ),
            0.5,
        )
        self.assertIsNone(raw_risk({"p_error": None}, "self_report"))

    def test_customer_features_use_observed_kpis_only(self):
        snapshot = {
            "answer": None,
            "history": [
                {
                    "kind": "tool_receipt",
                    "receipt": {
                        "phase": "pre_repair",
                        "payload": {
                            "service metrics": {"latency": 100.0, "bandwidth": 50.0}
                        },
                    },
                },
                {
                    "kind": "tool_receipt",
                    "receipt": {
                        "phase": "pre_repair",
                        "payload": {
                            "service metrics": {"latency": 150.0, "bandwidth": 50.0}
                        },
                    },
                },
            ],
        }
        self.assertEqual(features(snapshot)[7], 0.5)

    def test_public_label_calibration_and_test_rejection(self):
        cases, snapshots = [], []
        for i in range(8):
            case = {
                "schema_version": VERSION,
                "dataset_release_id": "fixture-release",
                "case_id": str(i),
                "domain": "dangerous_goods",
                "inputs": {k: "fixture" for k in INPUT_FIELDS["dangerous_goods"]},
                "training_split": "train" if i < 4 else "cal_fit",
                "label": {"hazard_class": "Hazard Class A"},
            }
            snapshot = {
                "case_id": str(i),
                "domain": "dangerous_goods",
                "profile_hash": FakeModel.profile_hash,
                "answer": {
                    "hazard_class": "Hazard Class A" if i % 2 else "Hazard Class B"
                },
                "p_error": None if i % 3 else 0.8,
                "samples": [],
                "history": [],
                "cost": 0.0,
            }
            snapshot["agent_contract"] = agent_contract(FakeModel.profile_hash)
            snapshot["agent_contract_hash"] = digest(snapshot["agent_contract"])
            snapshot["evaluation_contract"] = evaluation_contract(FakeExaminer().spec())
            snapshot["evaluation_contract_hash"] = digest(
                snapshot["evaluation_contract"]
            )
            cases.append(case)
            snapshots.append(snapshot)
        with tempfile.TemporaryDirectory() as tmp:
            calibrator = Calibrator().fit(cases, snapshots)
            path = Path(tmp) / "calibrator.json"
            calibrator.save(path)
            loaded = Calibrator.load(path)
            for method in (
                "historical_failure",
                "supervised",
                "calibrated_self_report",
                "calibrated_disagreement",
            ):
                self.assertTrue(0 <= loaded.predict(snapshots[0], method) <= 1)
            bad = copy.deepcopy(cases)
            bad[0].pop("label")
            with self.assertRaises(ValueError):
                Calibrator().fit(bad, snapshots)

    def test_asset_hash_check_happens_only_deployment_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "test.bin"
            file.write_bytes(b"fake-weight-file-for-test")
            profile = {
                "files": [
                    {
                        "file": file.name,
                        "bytes": file.stat().st_size,
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    }
                ]
            }
            self.assertEqual(
                verify_assets(profile, {"model_path": tmp})["verified_files"], 1
            )
            tokenizer = Path(tmp) / "tokenizer"
            tokenizer.mkdir()
            template = tokenizer / "chat_template.jinja"
            template.write_text("test-only-template")
            profile["tokenizer_files"] = [
                {
                    "file": template.name,
                    "bytes": template.stat().st_size,
                    "sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
                }
            ]
            with self.assertRaisesRegex(ValueError, "tokenizer_path"):
                verify_assets(profile, {"model_path": tmp})
            self.assertEqual(
                verify_assets(
                    profile, {"model_path": tmp, "tokenizer_path": str(tokenizer)}
                )["verified_files"],
                2,
            )
            template.write_text("replacement-template")
            with self.assertRaises(ValueError):
                verify_assets(
                    profile, {"model_path": tmp, "tokenizer_path": str(tokenizer)}
                )
            profile.pop("tokenizer_files")
            file.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                verify_assets(profile, {"model_path": tmp})


if __name__ == "__main__":
    unittest.main()
