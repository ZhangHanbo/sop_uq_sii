"""Command-line entry point ``sop-candidate``.

Subcommands: ``spec`` (service settings), ``train`` (collect public training
trajectories), ``fit`` (fit baseline calibrators), ``run`` (formal test run),
``pack`` (rebuild ``submission.zip``), ``submit`` (upload it) and ``report``
(read an evaluation report).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sop_client import ExaminerClient, ModelClient
from sop_client.report_check import check_public_report
from sop_protocol import digest, read_submission

from .policies import POLICIES
from .runner import package_run, run_panel
from .storage import read_jsonl
from .uq import UQ_METHODS, Calibrator


def public_split_snapshots(cases, path, allowed_splits, *, allow_empty=False):
    """Keep only trajectories whose public case belongs to ``allowed_splits``.

    A training run exports the whole public pool, so each fitting step selects
    its own split here; this keeps fitting, calibration and selection disjoint.
    """
    index = {case["case_id"]: case for case in cases}
    snapshots = read_jsonl(path)
    if any(snapshot["case_id"] not in index for snapshot in snapshots):
        raise ValueError("Trajectory is absent from the provided public training pool")
    selected = [
        snapshot
        for snapshot in snapshots
        if index[snapshot["case_id"]].get("training_split") in allowed_splits
    ]
    if not selected and not (allow_empty and not snapshots):
        raise ValueError(
            "No trajectories from the required public split: "
            + ", ".join(sorted(allowed_splits))
        )
    return selected


def main():
    """Parse arguments and run one subcommand; results are printed as JSON."""
    parser = argparse.ArgumentParser(
        description="SOP-UQ candidate: public training, remote runs and result bundles"
    )
    commands = parser.add_subparsers(dest="action", required=True)
    report_check = commands.add_parser(
        "check-public-report",
        help="Local text-format screening; never queries hidden labels",
    )
    report_check.add_argument("paths", nargs="+")
    for action in (
        "run",
        "train",
        "practice",
        "pack",
        "submit",
        "report",
        "spec",
        "close",
        "rpc",
    ):
        command = commands.add_parser(action)
        command.add_argument("--examiner-url", required=True)
        command.add_argument("--examiner-token-env", default="SOP_EXAMINER_TOKEN")
        if action in ("run", "train", "practice"):
            command.add_argument("--model-url", required=True)
            command.add_argument("--model-token-env", default="SOP_MODEL_TOKEN")
            command.add_argument("--model-slot", default="M9")
            command.add_argument(
                "--continuation-model-url",
                help="Optional independent solver service; never changes the official initial slot",
            )
            command.add_argument("--continuation-token-env", default="SOP_MODEL_TOKEN")
            command.add_argument(
                "--profile",
                help="Optional public model profile JSON expected from remote server",
            )
            command.add_argument("--method-id", default="baseline")
            command.add_argument("--policy", choices=POLICIES, default="no_oracle")
            command.add_argument("--uq", choices=UQ_METHODS, default="disagreement")
            command.add_argument("--calibrator")
            command.add_argument(
                "--threshold",
                type=float,
                help="Explicit heuristic threshold; otherwise use public holdout selection from the calibrator",
            )
            command.add_argument("--fixed-count", type=int, default=1)
            command.add_argument(
                "--posterior-uq",
                choices=("self_report", "calibrated_self_report"),
                help="Queries require consistent self_report or independently fitted posterior calibration",
            )
            command.add_argument(
                "--collect-posterior",
                action="store_true",
                help="Training only: collect raw posterior reports before fitting their calibration",
            )
            command.add_argument(
                "--initial-only",
                action="store_true",
                help="Seal initial trajectories and stop; examiner can freeze this local reference state",
            )
            command.add_argument("--seed", type=int, default=0)
            command.add_argument(
                "--output", required=True, help="Durable, resumable local run directory"
            )
        elif action == "pack":
            command.add_argument("--run-id", required=True)
            command.add_argument("--output", required=True)
        elif action == "submit":
            command.add_argument("submission")
            command.add_argument(
                "--close", action="store_true", help="Close after this accepted version"
            )
        elif action == "report":
            command.add_argument("evaluation_id")
        elif action == "close":
            command.add_argument("--run-id", required=True)
        elif action == "rpc":
            command.add_argument(
                "operation",
                help="Public RPC, e.g. initial/final/checkpoint_initial/restart_dev",
            )
            command.add_argument(
                "--body-file",
                required=True,
                help="UTF-8 JSON request, without credentials",
            )
    fit = commands.add_parser("fit")
    fit.add_argument("--data", required=True, help="Public data/train directory")
    fit.add_argument(
        "--snapshots",
        required=True,
        help="Visible M9-initial.jsonl from sop-candidate train",
    )
    fit.add_argument(
        "--posterior-snapshots",
        help="Public train/cal_fit M9-posterior.jsonl collected with the selected query policy",
    )
    fit.add_argument(
        "--selection-snapshots",
        help="Held-out public initial trajectories for cost-sensitive threshold selection",
    )
    fit.add_argument(
        "--posterior-selection-snapshots",
        help="Held-out public posterior trajectories for final abstention threshold selection",
    )
    fit.add_argument(
        "--selection-split", choices=("cal_select", "dev"), default="cal_select"
    )
    fit.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "check-public-report":
        result = check_public_report(args.paths)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "no_pattern_detected":
            raise SystemExit(2)
        return
    if args.action == "fit":
        cases = []
        for path in sorted(Path(args.data).glob("*.jsonl")):
            cases.extend(read_jsonl(path))
        # Fitting and threshold selection have distinct public split allowlists.
        initial = read_jsonl(args.snapshots)
        posterior = (
            public_split_snapshots(
                cases, args.posterior_snapshots, {"train", "cal_fit"}, allow_empty=True
            )
            if args.posterior_snapshots
            else None
        )
        calibrator = Calibrator().fit(cases, initial, posterior_snapshots=posterior)
        if args.selection_snapshots:
            selected = public_split_snapshots(
                cases, args.selection_snapshots, {args.selection_split}
            )
            calibrator.select_thresholds(cases, selected, split=args.selection_split)
        if args.posterior_selection_snapshots:
            selected = public_split_snapshots(
                cases,
                args.posterior_selection_snapshots,
                {args.selection_split},
                allow_empty=True,
            )
            if selected:
                calibrator.select_thresholds(
                    cases, selected, split=args.selection_split, posterior=True
                )
        calibrator.save(args.output)
        print(
            json.dumps(
                {
                    "calibrator": args.output,
                    "dataset_release_id": calibrator.release_id,
                    "training_counts": {
                        k: len(v) for k, v in calibrator.fit_cases.items()
                    },
                    "posterior_calibration_count": len(calibrator.posterior_fit_cases),
                    "posterior_fit_status": (
                        "fitted"
                        if calibrator.posterior
                        else "no_posterior_fit; zero-query runs remain available"
                    ),
                    "selections": calibrator.selections,
                }
            )
        )
        return
    client = ExaminerClient(
        args.examiner_url, os.environ.get(args.examiner_token_env, "")
    )
    if args.action == "spec":
        result = client.spec()
    elif args.action == "pack":
        result = package_run(client, args.run_id, args.output)
    elif args.action == "submit":
        result = client.evaluate(args.submission)
        if args.close:
            manifest, _ = read_submission(Path(args.submission).read_bytes())
            result = {"receipt": result, "closed_run": client.close(manifest["run_id"])}
    elif args.action == "report":
        result = client.evaluation(args.evaluation_id)
    elif args.action == "close":
        result = client.close(args.run_id)
    elif args.action == "rpc":
        # The service owns all permissions and checkpoints. This small entrypoint
        # also supports custom online algorithms without prescribing a model.
        client.spec()
        result = client.call(
            args.operation, json.loads(Path(args.body_file).read_text())
        )
    else:
        expected = (
            digest(json.loads(Path(args.profile).read_text())) if args.profile else None
        )
        model = ModelClient(
            args.model_url,
            os.environ.get(args.model_token_env, ""),
            expected_profile_hash=expected,
        )
        continuation = (
            ModelClient(
                args.continuation_model_url,
                os.environ.get(args.continuation_token_env, ""),
            )
            if args.continuation_model_url
            else model
        )
        calibrators = (
            {args.model_slot: Calibrator.load(args.calibrator)}
            if args.calibrator
            else {}
        )
        if args.uq not in ("self_report", "disagreement") and not calibrators:
            parser.error(
                "This UQ method requires --calibrator fitted on public train labels"
            )
        if args.seed < 0 or args.seed + 10004 >= 2**31:
            parser.error(
                "--seed must leave room for shared-sample seeds in the 31-bit range"
            )
        result = run_panel(
            client,
            {args.model_slot: model},
            args.output,
            split={"train": "train", "practice": "dev", "run": "test"}[args.action],
            method_id=args.method_id,
            policy=args.policy,
            uq=args.uq,
            calibrators=calibrators,
            threshold=args.threshold,
            fixed_count=args.fixed_count,
            seed=args.seed,
            posterior_uq=args.posterior_uq,
            collect_posterior=args.collect_posterior,
            initial_only=args.initial_only,
            continuation_models={args.model_slot: continuation},
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
