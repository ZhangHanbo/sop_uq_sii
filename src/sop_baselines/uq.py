"""Public risk-estimation baselines and their calibration.

Every estimator returns a probability that an answer is wrong. Supervision comes
only from public training labels: a trajectory's answer is compared with the
label of its case. Fitting uses the ``train`` split, probability calibration the
``cal_fit`` split, and threshold selection ``cal_select`` (or ``dev``), so each
public split has one role. Test cases are never accepted.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sop_protocol import (
    AIRCRAFT_FIELDS,
    decision_terminal_loss,
    digest,
    valid_answer,
    valid_probability,
    validate_public_case,
)

from .contracts import (
    agent_contract,
    check_snapshot_contract,
    check_evaluation_contract,
    risk_estimator_contract,
)
from .storage import atomic_json

# Inputs of the ``supervised`` logistic model, in column order.
FEATURES = (
    "self_report",
    "self_report_missing",
    "disagreement",
    "invalid_samples",
    "answer_missing",
    "tool_count",
    "cost",
    "source_conflict_fraction",
    "invalid_responses",
)
# Initial-risk methods selectable with ``--uq``.
UQ_METHODS = (
    "self_report",
    "disagreement",
    "historical_failure",
    "calibrated_self_report",
    "calibrated_disagreement",
    "supervised",
)


def disagreement(snapshot):
    """Smoothed share of shared samples that differ from the proposal.

    An invalid sample counts as a disagreement; ``(d + 1) / (n + 2)`` keeps the
    estimate strictly inside (0, 1).
    """
    samples = snapshot.get("samples", [])
    different = sum(
        s.get("answer") is None
        or snapshot.get("answer") is None
        or s["answer"] != snapshot["answer"]
        for s in samples
    )
    return (different + 1) / (len(samples) + 2)


def features(snapshot):
    """Build the ``FEATURES`` vector of one initial trajectory.

    ``source_conflict_fraction`` is the share of repeatedly observed fields whose
    readings differ within one phase. Counts are scaled to roughly [0, 1] by the
    published limits (24 tool calls, budget 2.00).
    """
    p = snapshot.get("p_error")
    samples = snapshot.get("samples", [])
    receipts = [
        e["receipt"]
        for e in snapshot.get("history", [])
        if e.get("kind") == "tool_receipt"
    ]
    values = defaultdict(list)
    measurement_fields = set(AIRCRAFT_FIELDS) | {
        "sds_label_score",
        "handling_score",
        "transportation_score",
        "disposal_score",
        "latency",
        "jitter",
        "bandwidth",
    }
    for receipt in receipts:
        payload = receipt.get("payload", {})
        measurements = payload.get(
            "service metrics", payload.get("updated service metrics", payload)
        )
        for key, value in measurements.items():
            if key in measurement_fields and (
                isinstance(value, (str, int, float, bool)) or value is None
            ):
                values[(receipt.get("phase"), key)].append(digest(value))
    repeated = [v for v in values.values() if len(v) > 1]
    conflict = sum(len(set(v)) > 1 for v in repeated) / max(1, len(repeated))
    return np.asarray(
        [
            float(p) if valid_probability(p) else 0.5,
            float(not valid_probability(p)),
            disagreement(snapshot),
            sum(s.get("answer") is None for s in samples) / max(1, len(samples)),
            float(snapshot.get("answer") is None),
            min(24, len(receipts)) / 24,
            snapshot.get("cost", 0.0) / 2.0,
            conflict,
            min(
                5,
                sum(
                    e.get("kind") in ("invalid_response", "tool_error")
                    for e in snapshot.get("history", [])
                ),
            )
            / 5.0,
        ]
    )


class Logistic:
    """L2-regularised logistic regression fitted by damped Newton steps.

    Inputs are standardised; the intercept is not penalised and starts at the
    smoothed log-odds of the labels, which keeps tiny calibration sets stable.
    """

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(y) == 0:
            raise ValueError("Empty supervised training split")
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.scale])
        weights = np.zeros(z.shape[1])
        weights[0] = math.log((sum(y) + 1) / (len(y) - sum(y) + 1))
        regularizer = np.diag([0.0] + [1.0] * (z.shape[1] - 1))
        for _ in range(60):
            p = 1 / (1 + np.exp(-np.clip(z @ weights, -30, 30)))
            h = (
                z.T @ ((p * (1 - p))[:, None] * z)
                + regularizer
                + np.eye(z.shape[1]) * 1e-6
            )
            step = np.linalg.solve(h, z.T @ (p - y) + regularizer @ weights)
            step *= min(1.0, 3 / max(3.0, float(np.linalg.norm(step))))
            weights -= step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.weights = weights
        return self

    def predict(self, row):
        """Return the fitted probability for one feature row."""
        x = np.asarray(row, dtype=float)
        value = np.r_[1.0, (x - self.mean) / self.scale] @ self.weights
        return float(1 / (1 + np.exp(-np.clip(value, -30, 30))))

    def to_dict(self):
        return {
            key: getattr(self, key).tolist() for key in ("mean", "scale", "weights")
        }

    @classmethod
    def from_dict(cls, value):
        obj = cls()
        for key in ("mean", "scale", "weights"):
            setattr(obj, key, np.asarray(value[key], dtype=float))
        return obj


class Calibrator:
    """All fitted baseline estimators of one model profile and data release.

    The artifact records the agent contract, the estimator implementation and the
    exact training trajectories, and refuses trajectories produced otherwise.
    """

    def fit(self, cases, snapshots, *, posterior_snapshots=None):
        """Fit the initial-risk estimators from public labeled trajectories.

        ``snapshots`` may contain the whole public pool; ``train`` rows fit the
        historical rates and the supervised model, ``cal_fit`` rows fit the
        probability calibration of self-report and disagreement.
        """
        for case in cases:
            validate_public_case(case, training=True)
        index = {case["case_id"]: case for case in cases}
        if len(index) != len(cases):
            raise ValueError("Duplicate public training cases")
        if len({s["case_id"] for s in snapshots}) != len(snapshots):
            raise ValueError("Duplicate training snapshots")
        profiles = {s["profile_hash"] for s in snapshots}
        releases = {c["dataset_release_id"] for c in cases}
        if len(profiles) != 1 or len(releases) != 1:
            raise ValueError("Fit one model profile and data release at a time")
        self.profile_hash, self.release_id = next(iter(profiles)), next(iter(releases))
        self.agent_contract = agent_contract(self.profile_hash)
        self.agent_contract_hash = digest(self.agent_contract)
        self.estimator_contract = risk_estimator_contract()
        self.evaluation_contract = copy.deepcopy(
            check_evaluation_contract(snapshots[0])
        )
        for snapshot in snapshots:
            check_snapshot_contract(snapshot, self.agent_contract)
            check_evaluation_contract(snapshot, self.evaluation_contract)
        rows = [(index[s["case_id"]], s) for s in snapshots]
        if any(c["domain"] != s["domain"] for c, s in rows):
            raise ValueError("Trajectory domain differs from the public labeled case")
        train = [(c, s) for c, s in rows if c["training_split"] == "train"]
        cal = [(c, s) for c, s in rows if c["training_split"] == "cal_fit"]
        if not train or not cal:
            raise ValueError(
                "Need visible train and cal_fit trajectories; test inputs are never accepted"
            )

        # A failed trajectory counts as an error, as in the official scoring.
        def error(c, s):
            return int(bool(s.get("failure")) or s.get("answer") != c["label"])

        self.rates = {}
        for domain in {c["domain"] for c, _ in train}:
            subset = [(c, s) for c, s in train if c["domain"] == domain]
            self.rates[domain] = (1 + sum(error(c, s) for c, s in subset)) / (
                len(subset) + 2
            )
        self.supervised = Logistic().fit(
            [features(s) for _, s in train], [error(c, s) for c, s in train]
        )
        self.platt = {}
        for method in ("self_report", "disagreement"):
            self.platt[method] = Logistic().fit(
                [calibration_features(s, method) for _, s in cal],
                [error(c, s) for c, s in cal],
            )
        self.fit_cases = {
            name: sorted(c["case_id"] for c, _ in part)
            for name, part in (("train", train), ("cal_fit", cal))
        }
        self.training_trajectory_hash = digest(snapshots)
        self.posterior = {}
        self.posterior_contract = None
        self.posterior_fit_cases = []
        self.posterior_fit_cases_by_target = {}
        self.selections = {}
        if posterior_snapshots:
            self.fit_posterior(cases, posterior_snapshots)
        return self

    def fit_posterior(self, cases, snapshots):
        """Fit the two post-query risks on public ``cal_fit`` query trajectories.

        ``initial`` predicts whether the frozen initial answer is wrong given the
        full history; ``final`` predicts whether the final answer is wrong. All
        trajectories must come from one query policy, because the policy changes
        which histories exist after querying.
        """
        index = {}
        for case in cases:
            validate_public_case(case, training=True)
            if case["dataset_release_id"] != self.release_id:
                raise ValueError("Posterior calibration data release differs")
            index[case["case_id"]] = case
        if len({s["case_id"] for s in snapshots}) != len(snapshots):
            raise ValueError("Duplicate posterior calibration trajectories")
        contracts = []
        fitted = []
        for snapshot in snapshots:
            self.check_snapshot(snapshot)
            if "initial_answer" not in snapshot or not isinstance(
                snapshot.get("posterior_contract"), dict
            ):
                raise ValueError(
                    "Posterior snapshots must preserve frozen initial_answer and query contract"
                )
            if snapshot.get("initial_status") not in ("completed", "failed"):
                raise ValueError(
                    "Posterior snapshots must preserve the initial commit status"
                )
            case = index[snapshot["case_id"]]
            if case["domain"] != snapshot["domain"]:
                raise ValueError("Posterior trajectory domain differs")
            if case["training_split"] not in ("train", "cal_fit"):
                raise ValueError(
                    "Posterior fitting accepts train/cal_fit only; selection must remain held out"
                )
            contracts.append(snapshot["posterior_contract"])
            if case["training_split"] == "cal_fit":
                fitted.append((case, snapshot))
        if not fitted or any(c != contracts[0] for c in contracts):
            raise ValueError(
                "Need cal_fit posterior trajectories from one frozen query policy"
            )
        self.posterior_contract = contracts[0]
        if self.posterior_contract.get("policy") == "active":
            trigger_uq = self.posterior_contract.get("trigger_uq")
            if self.posterior_contract.get(
                "trigger_estimator_hash"
            ) != self.initial_estimator_hash(trigger_uq):
                raise ValueError(
                    "Posterior collection used a different initial risk estimator"
                )
        for target, answer_field in (
            ("initial", "initial_answer"),
            ("final", "answer"),
        ):
            # The scorer never evaluates p0|T when the initial commitment failed.
            # Exclude those rows from this calibrator, without dropping them from
            # final-answer calibration or from any score/report denominator.
            target_rows = [
                (c, s)
                for c, s in fitted
                if target == "final" or s["initial_status"] == "completed"
            ]
            if not target_rows:
                raise ValueError(
                    "Need completed initial cal_fit commitments to calibrate their posterior risk"
                )
            self.posterior[target] = Logistic().fit(
                [posterior_features(s, target) for _, s in target_rows],
                [
                    int(
                        bool(
                            s[answer_field] != c["label"]
                            or (
                                target == "final"
                                and (s.get("failure") or s.get("oracle_failure"))
                            )
                        )
                    )
                    for c, s in target_rows
                ],
            )
            self.posterior_fit_cases_by_target[target] = sorted(
                c["case_id"] for c, _ in target_rows
            )
        self.posterior_fit_cases = sorted(c["case_id"] for c, _ in fitted)
        self.posterior_trajectory_hash = digest(snapshots)
        return self

    def check_snapshot(self, snapshot):
        """Reject trajectories from another model profile or reference agent."""
        self.check_estimator()
        if snapshot["profile_hash"] != self.profile_hash:
            raise ValueError("Calibration profile mismatch")
        check_snapshot_contract(snapshot, self.agent_contract)
        check_evaluation_contract(snapshot, self.evaluation_contract)

    def check_estimator(self):
        """Reject use of this artifact after ``uq.py`` has been modified."""
        if self.estimator_contract != risk_estimator_contract():
            raise ValueError("Risk estimator implementation changed; refit calibration")

    def initial_estimator_hash(self, method):
        """Identify one fitted initial-risk estimator.

        The ``active`` policy triggers queries with this estimator, so its hash is
        stored in the posterior contract. Posterior and selection state are
        excluded to avoid a circular binding.
        """
        if method in ("self_report", "disagreement"):
            fitted = None
        elif method == "historical_failure":
            fitted = self.rates
        elif method == "supervised":
            fitted = self.supervised.to_dict()
        elif method in ("calibrated_self_report", "calibrated_disagreement"):
            fitted = self.platt[method.removeprefix("calibrated_")].to_dict()
        else:
            raise ValueError("Unknown initial UQ method")
        return digest(
            {
                "method": method,
                "agent_contract_hash": self.agent_contract_hash,
                "estimator_contract": self.estimator_contract,
                "fitted": fitted,
                "fit_cases": self.fit_cases if fitted is not None else None,
                "training_trajectory_hash": (
                    self.training_trajectory_hash if fitted is not None else None
                ),
            }
        )

    def predict_posterior(self, snapshot, target):
        """Post-query risk; ``target`` is ``"initial"`` or ``"final"``."""
        self.check_snapshot(snapshot)
        if target not in ("initial", "final") or target not in self.posterior:
            raise ValueError(
                "Fit separate public posterior calibration before a calibrated query run"
            )
        if snapshot.get("posterior_contract") != self.posterior_contract:
            raise ValueError(
                "Posterior query policy differs from the fitted calibration trajectories"
            )
        return self.posterior[target].predict(posterior_features(snapshot, target))

    def predict(self, snapshot, method):
        """Initial risk of ``snapshot`` under one of ``UQ_METHODS``."""
        self.check_snapshot(snapshot)
        if method == "historical_failure":
            return self.rates[snapshot["domain"]]
        if method == "supervised":
            return self.supervised.predict(features(snapshot))
        if method.startswith("calibrated_"):
            base = method.removeprefix("calibrated_")
            return self.platt[base].predict(calibration_features(snapshot, base))
        return raw_risk(snapshot, method)

    def to_dict(self):
        """Serialise the artifact; ``load`` restores and re-checks it."""
        return {
            "version": "2.2",
            "feature_names": list(FEATURES),
            "profile_hash": self.profile_hash,
            "dataset_release_id": self.release_id,
            "agent_contract": self.agent_contract,
            "agent_contract_hash": self.agent_contract_hash,
            "estimator_contract": self.estimator_contract,
            "evaluation_contract": self.evaluation_contract,
            "training_trajectory_hash": self.training_trajectory_hash,
            "rates": self.rates,
            "supervised": self.supervised.to_dict(),
            "platt": {key: value.to_dict() for key, value in self.platt.items()},
            "fit_cases": self.fit_cases,
            "posterior": {
                key: value.to_dict() for key, value in self.posterior.items()
            },
            "posterior_contract": self.posterior_contract,
            "posterior_fit_cases": self.posterior_fit_cases,
            "posterior_fit_cases_by_target": self.posterior_fit_cases_by_target,
            "posterior_trajectory_hash": getattr(
                self, "posterior_trajectory_hash", None
            ),
            "selections": self.selections,
        }

    def save(self, path):
        atomic_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        """Load a saved artifact, rejecting other versions and changed code."""
        value = json.loads(Path(path).read_text())
        if value.get("version") != "2.2" or value.get("feature_names") != list(
            FEATURES
        ):
            raise ValueError(
                "Calibration version mismatch; recollect and refit legacy trajectories"
            )
        obj = cls()
        obj.profile_hash, obj.release_id = (
            value["profile_hash"],
            value["dataset_release_id"],
        )
        obj.agent_contract = value["agent_contract"]
        obj.agent_contract_hash = value["agent_contract_hash"]
        obj.estimator_contract = value.get("estimator_contract")
        obj.evaluation_contract = value["evaluation_contract"]
        obj.check_estimator()
        check_snapshot_contract(
            {
                "profile_hash": obj.profile_hash,
                "agent_contract": obj.agent_contract,
                "agent_contract_hash": obj.agent_contract_hash,
            }
        )
        obj.training_trajectory_hash = value["training_trajectory_hash"]
        obj.rates, obj.fit_cases = value["rates"], value["fit_cases"]
        obj.supervised = Logistic.from_dict(value["supervised"])
        obj.platt = {
            key: Logistic.from_dict(row) for key, row in value["platt"].items()
        }
        obj.posterior = {
            key: Logistic.from_dict(row) for key, row in value["posterior"].items()
        }
        obj.posterior_contract = value["posterior_contract"]
        obj.posterior_fit_cases = value["posterior_fit_cases"]
        obj.posterior_fit_cases_by_target = value["posterior_fit_cases_by_target"]
        obj.posterior_trajectory_hash = value["posterior_trajectory_hash"]
        obj.selections = value["selections"]
        return obj

    def select_thresholds(
        self, cases, snapshots, *, split="cal_select", posterior=False
    ):
        """Select actual 1/5-error-cost abstention thresholds on one public holdout.

        A binary error probability alone does not identify severe-error risk. We
        evaluate each threshold against observed labeled costs instead of assuming
        every error has one severity. This selects abstention, not query value.
        """
        if split not in ("cal_select", "dev"):
            raise ValueError("Threshold selection requires public cal_select or dev")
        index = {}
        for case in cases:
            validate_public_case(case, training=True)
            if case["dataset_release_id"] != self.release_id:
                raise ValueError("Threshold selection data release differs")
            index[case["case_id"]] = case
        if not snapshots or len({s["case_id"] for s in snapshots}) != len(snapshots):
            raise ValueError("Need distinct held-out selection trajectories")
        rows = []
        for snapshot in snapshots:
            self.check_snapshot(snapshot)
            case = index[snapshot["case_id"]]
            if case["training_split"] != split or case["domain"] != snapshot["domain"]:
                raise ValueError(
                    "Selection snapshots must all belong to the selected public holdout split"
                )
            rows.append((case, snapshot))
        methods = ("posterior_final",) if posterior else UQ_METHODS
        for method in methods:
            probabilities = [
                (
                    self.predict_posterior(s, "final")
                    if posterior
                    else self.predict(s, method)
                )
                for _, s in rows
            ]
            record = select_cost_threshold(
                rows,
                probabilities,
                defer_cost=self.evaluation_contract["costs"]["defer"],
            )
            record.update(
                split=split,
                case_count=len(rows),
                trajectory_hash=digest(snapshots),
                posterior=posterior,
                query_policy=self.posterior_contract if posterior else None,
            )
            self.selections[method] = record
        return self


def raw_risk(snapshot, method="disagreement"):
    """Unfitted risk: raw disagreement or the model's self-reported probability."""
    if method == "disagreement":
        return disagreement(snapshot)
    if method == "self_report":
        return (
            snapshot["p_error"] if valid_probability(snapshot.get("p_error")) else None
        )
    raise ValueError(
        "This UQ method needs an artifact fitted on public training labels"
    )


def calibration_features(snapshot, method):
    """Inputs of a one-score calibrator: the score and a missing-score flag."""
    score = raw_risk(snapshot, method)
    # Imputation is an input feature with an explicit missingness bit. The output
    # is a fitted probability; raw self-report itself remains missing.
    return [
        score if valid_probability(score) else 0.5,
        float(not valid_probability(score)),
    ]


def posterior_features(snapshot, target):
    """Inputs of a post-query calibrator: the matching self-report and its flag."""
    probability = snapshot.get("p_initial_after" if target == "initial" else "p_error")
    return [
        probability if valid_probability(probability) else 0.5,
        float(not valid_probability(probability)),
    ]


def labeled_error_cost(answer, label, domain):
    """Terminal loss of an AUTO answer under the published 0/1/5 costs.

    Severe errors (cost 5): an underestimated hazard class, including a class
    given where the label is Unable to Decide; ``aircraft_ready`` true where the
    label is false; RESOLVED where the label is not RESOLVED. An illegal answer
    also costs 5. Any other error costs 1.
    """
    if not valid_answer(answer, domain):
        return 5.0
    if answer == label:
        return 0.0
    if domain == "dangerous_goods":
        levels = {
            "Hazard Class A": 0,
            "Hazard Class B": 1,
            "Hazard Class C": 2,
            "Hazard Class D": 3,
        }
        proposed, actual = answer["hazard_class"], label["hazard_class"]
        if proposed != "Unable to Decide" and (
            actual == "Unable to Decide" or levels[proposed] < levels[actual]
        ):
            return 5.0
    elif domain == "aircraft_inspection":
        if answer["aircraft_ready"] and not label["aircraft_ready"]:
            return 5.0
    elif (
        domain == "customer_service" and answer["final_resolution_status"] == "RESOLVED"
    ):
        return 5.0
    return 1.0


def select_cost_threshold(rows, probabilities, *, defer_cost=0.5):
    """Choose the DEFER threshold with the lowest domain-macro average loss.

    A case is answered AUTO when its probability is valid and strictly below the
    threshold, and deferred otherwise at the configured cost. Tool costs are
    added to every case. Candidate thresholds sit just above each observed
    probability, plus 0 (defer all) and 1. Ties prefer the lower threshold.
    """
    if not rows or len(rows) != len(probabilities):
        raise ValueError("Need one probability per nonempty selection row")
    if (
        not isinstance(defer_cost, (int, float))
        or not math.isfinite(defer_cost)
        or defer_cost < 0
    ):
        raise ValueError("DEFER cost must be finite and nonnegative")

    def controller_valid(snapshot):
        # A model/transport failure is not a controller violation. Preserve the
        # service's explicit status; also recognise its terminal violation code
        # in older visible error records instead of treating it as cheap DEFER.
        return snapshot.get("controller_valid", True) and not any(
            snapshot.get(key) == "protocol_failure_limit"
            for key in ("failure", "oracle_failure")
        )

    valid = [
        valid_probability(p)
        and controller_valid(s)
        and s.get("answer") is not None
        and not s.get("failure")
        and not s.get("oracle_failure")
        for (_, s), p in zip(rows, probabilities)
    ]
    observed = sorted({p for p, ok in zip(probabilities, valid) if ok})
    thresholds = sorted(
        {0.0, 1.0, *(math.nextafter(p, 1.0) for p in observed if p < 1.0)}
    )
    curve = []
    for threshold in thresholds:
        losses = defaultdict(list)
        automatic = 0
        for (case, snapshot), probability, ok in zip(rows, probabilities, valid):
            use_auto = ok and probability < threshold
            automatic += use_auto
            auto_cost = labeled_error_cost(
                snapshot.get("answer"), case["label"], case["domain"]
            )
            terminal = decision_terminal_loss(
                "AUTO" if use_auto else "DEFER",
                controller_valid=controller_valid(snapshot),
                answer_present=snapshot.get("answer") is not None,
                error=auto_cost > 0,
                severe=auto_cost == 5,
                defer_cost=defer_cost,
            )
            losses[case["domain"]].append(float(snapshot.get("cost", 0.0)) + terminal)
        curve.append(
            {
                "threshold": threshold,
                "coverage": automatic / len(rows),
                "domain_macro_loss": sum(sum(v) / len(v) for v in losses.values())
                / len(losses),
            }
        )
    best = min(curve, key=lambda row: (row["domain_macro_loss"], row["threshold"]))
    return {
        **best,
        "curve": curve,
        "domains": sorted({c["domain"] for c, _ in rows}),
        "valid_probability_count": sum(valid),
        "unique_valid_probabilities": len(observed),
        "degenerate_all_defer": best["coverage"] == 0.0,
        "degenerate_all_auto": best["coverage"] == 1.0,
        "defer_cost": float(defer_cost),
        "objective": "public held-out domain-macro tool cost + terminal cost under the bound public configuration",
        "limitation": "Abstention threshold only; query trigger remains a declared heuristic, not estimated information value",
    }
