"""Risk-estimation baselines and their calibration.

Every estimator returns a probability that an answer is wrong. Supervision comes
from public labels: a trajectory's answer is compared with its case's label.
``train`` fits the historical rates and the supervised model, ``cal_fit`` the
probability calibration, and ``cal_select`` (or ``dev``) the DEFER thresholds.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from test_client.protocol import AIRCRAFT_FIELDS, DEFER_COST, digest, valid_probability
from test_client.scoring import answer_error, terminal_cost

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
        e["receipt"] for e in snapshot.get("history", []) if e.get("kind") == "tool_receipt"
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
        """Fit on feature rows ``x`` and 0/1 labels ``y``; returns self."""
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
            h = z.T @ ((p * (1 - p))[:, None] * z) + regularizer + np.eye(z.shape[1]) * 1e-6
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
        """JSON form of the fitted standardisation and weights."""
        return {key: getattr(self, key).tolist() for key in ("mean", "scale", "weights")}

    @classmethod
    def from_dict(cls, value):
        """Rebuild a model saved with ``to_dict``."""
        obj = cls()
        for key in ("mean", "scale", "weights"):
            setattr(obj, key, np.asarray(value[key], dtype=float))
        return obj


class Calibrator:
    """All fitted baseline estimators of one agent/model configuration."""

    def __init__(self):
        """An unfitted calibrator; call ``fit`` (and optionally ``fit_posterior``)."""
        self.rates, self.supervised, self.platt = {}, None, {}
        self.posterior, self.selections, self.fit_counts = {}, {}, {}
        self.defer_cost = DEFER_COST

    @staticmethod
    def _rows(cases, snapshots, splits):
        """Pairs (case, trajectory) whose case belongs to one of ``splits``."""
        index = {case["case_id"]: case for case in cases}
        rows = [(index[s["case_id"]], s) for s in snapshots if s["case_id"] in index]
        return [(c, s) for c, s in rows if c["training_split"] in splits]

    def fit(self, cases, snapshots, *, posterior_snapshots=None):
        """Fit the initial-risk estimators from labeled initial trajectories.

        ``cases`` are public labeled cases; ``snapshots`` may cover all public
        splits and each estimator selects its own split.
        """
        train = self._rows(cases, snapshots, {"train"})
        cal = self._rows(cases, snapshots, {"cal_fit"})
        if not train or not cal:
            raise ValueError("Need initial trajectories of both train and cal_fit cases")

        # The official rule: a missing, illegal or wrong answer is an error.
        def error(c, s):
            """1 if the trajectory's answer is wrong for its case."""
            return answer_error(s.get("answer"), c["label"], c["domain"])

        for domain in sorted({c["domain"] for c, _ in train}):
            subset = [(c, s) for c, s in train if c["domain"] == domain]
            self.rates[domain] = (1 + sum(error(c, s) for c, s in subset)) / (len(subset) + 2)
        self.supervised = Logistic().fit(
            [features(s) for _, s in train], [error(c, s) for c, s in train]
        )
        for method in ("self_report", "disagreement"):
            self.platt[method] = Logistic().fit(
                [calibration_features(s, method) for _, s in cal],
                [error(c, s) for c, s in cal],
            )
        self.fit_counts = {"train": len(train), "cal_fit": len(cal)}
        if posterior_snapshots:
            self.fit_posterior(cases, posterior_snapshots)
        return self

    def fit_posterior(self, cases, snapshots):
        """Fit the two post-query risks on queried train/cal_fit trajectories.

        ``initial`` predicts whether the initial answer is wrong given all
        evidence; ``final`` whether the continued final answer is wrong. Use
        trajectories collected with the query policy you will run on test.
        """
        rows = self._rows(cases, snapshots, {"train", "cal_fit"})
        if not rows:
            raise ValueError("Need queried train/cal_fit trajectories for posterior fitting")
        for target in ("initial", "final"):
            self.posterior[target] = Logistic().fit(
                [posterior_features(s, target) for _, s in rows],
                [posterior_targets(c, s)[target] for c, s in rows],
            )
        self.fit_counts["posterior"] = len(rows)
        return self

    def predict(self, snapshot, method):
        """Initial risk of ``snapshot`` under one of ``UQ_METHODS``."""
        if method == "historical_failure":
            return self.rates[snapshot["domain"]]
        if method == "supervised":
            return self.supervised.predict(features(snapshot))
        if method.startswith("calibrated_"):
            base = method.removeprefix("calibrated_")
            return self.platt[base].predict(calibration_features(snapshot, base))
        return raw_risk(snapshot, method)

    def predict_posterior(self, snapshot, target):
        """Post-query risk; ``target`` is ``"initial"`` or ``"final"``."""
        if target not in self.posterior:
            raise ValueError("Fit posterior calibration first (fit --posterior ...)")
        return self.posterior[target].predict(posterior_features(snapshot, target))

    def select_thresholds(self, cases, snapshots, *, split="cal_select", posterior=False):
        """Choose cost-minimising DEFER thresholds on one held-out labeled split.

        Without ``posterior`` a threshold is chosen for every initial method;
        with it, one threshold for the calibrated final-answer risk.
        """
        rows = self._rows(cases, snapshots, {split})
        if not rows:
            raise ValueError(f"No trajectories of the {split} split")
        methods = ("posterior_final",) if posterior else UQ_METHODS
        for method in methods:
            probabilities = [
                self.predict_posterior(s, "final") if posterior else self.predict(s, method)
                for _, s in rows
            ]
            record = select_cost_threshold(rows, probabilities, defer_cost=self.defer_cost)
            record.update(split=split, case_count=len(rows))
            self.selections[method] = record
        return self

    def to_dict(self):
        """JSON form of every fitted estimator, threshold and fit count."""
        return {
            "version": "3",
            "feature_names": list(FEATURES),
            "rates": self.rates,
            "supervised": self.supervised.to_dict(),
            "platt": {key: value.to_dict() for key, value in self.platt.items()},
            "posterior": {key: value.to_dict() for key, value in self.posterior.items()},
            "selections": self.selections,
            "fit_counts": self.fit_counts,
        }

    def save(self, path):
        """Write ``to_dict`` to ``path``."""
        atomic_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        """Read a calibrator saved by ``save``."""
        value = json.loads(Path(path).read_text())
        if value.get("version") != "3" or value.get("feature_names") != list(FEATURES):
            raise ValueError("Calibration file from another version; fit again")
        obj = cls()
        obj.rates = value["rates"]
        obj.supervised = Logistic.from_dict(value["supervised"])
        obj.platt = {key: Logistic.from_dict(row) for key, row in value["platt"].items()}
        obj.posterior = {key: Logistic.from_dict(row) for key, row in value["posterior"].items()}
        obj.selections, obj.fit_counts = value["selections"], value["fit_counts"]
        return obj


def raw_risk(snapshot, method="disagreement"):
    """Unfitted risk: raw disagreement or the model's self-reported probability."""
    if method == "disagreement":
        return disagreement(snapshot)
    if method == "self_report":
        return snapshot["p_error"] if valid_probability(snapshot.get("p_error")) else None
    raise ValueError("This UQ method needs a calibration file fitted on public training labels")


def calibration_features(snapshot, method):
    """Inputs of a one-score calibrator: the score and a missing-score flag."""
    score = raw_risk(snapshot, method)
    # Imputation is an input feature with an explicit missingness bit. The output
    # is a fitted probability; raw self-report itself remains missing.
    return [
        score if valid_probability(score) else 0.5,
        float(not valid_probability(score)),
    ]


def posterior_targets(case, snapshot):
    """Official errors of a queried trajectory's initial and final answers.

    Only the answer itself counts, as in the official scoring; a failed later
    oracle read or tool call does not make a correct answer wrong.
    """
    return {
        "initial": answer_error(snapshot.get("initial_answer"), case["label"], case["domain"]),
        "final": answer_error(snapshot.get("answer"), case["label"], case["domain"]),
    }


def posterior_features(snapshot, target):
    """Inputs of a post-query calibrator: the matching self-report and its flag."""
    probability = snapshot.get("p_initial_after" if target == "initial" else "p_error")
    return [
        probability if valid_probability(probability) else 0.5,
        float(not valid_probability(probability)),
    ]


def select_cost_threshold(rows, probabilities, *, defer_cost=DEFER_COST):
    """Choose the DEFER threshold with the lowest domain-macro average loss.

    A case is answered AUTO when it has an answer and a valid probability
    strictly below the threshold, and deferred otherwise. Tool costs are added
    to every case. Candidates sit just above each observed probability, plus 0
    (defer all) and 1. Ties prefer the lower threshold.
    """
    valid = [
        valid_probability(p) and s.get("answer") is not None and not s.get("failure")
        for (_, s), p in zip(rows, probabilities)
    ]
    observed = sorted({p for p, ok in zip(probabilities, valid) if ok})
    thresholds = sorted({0.0, 1.0, *(math.nextafter(p, 1.0) for p in observed if p < 1.0)})
    curve = []
    for threshold in thresholds:
        losses = defaultdict(list)
        automatic = 0
        for (case, snapshot), probability, ok in zip(rows, probabilities, valid):
            mode = "AUTO" if ok and probability < threshold else "DEFER"
            automatic += mode == "AUTO"
            terminal = terminal_cost(
                mode, snapshot.get("answer"), case["label"], case["domain"], defer_cost
            )
            losses[case["domain"]].append(float(snapshot.get("cost", 0.0)) + terminal)
        curve.append(
            {
                "threshold": threshold,
                "coverage": automatic / len(rows),
                "domain_macro_loss": sum(sum(v) / len(v) for v in losses.values()) / len(losses),
            }
        )
    best = min(curve, key=lambda row: (row["domain_macro_loss"], row["threshold"]))
    return {**best, "valid_probability_count": sum(valid)}
