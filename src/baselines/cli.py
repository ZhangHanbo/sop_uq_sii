"""Command line ``sop-candidate``.

    collect  run the agent on labeled train/dev cases and save trajectories
    fit      fit risk estimators and DEFER thresholds from saved trajectories
    run      run a method on the 360 test cases and write results.csv
    score    score a train/dev results.csv locally with the public labels
    submit   upload a test results.csv to your examiner (uses one try)

One ``collect --policy fixed`` gives both initial and post-query trajectories;
``fit`` then fits the initial and the post-query risks from the same folder.
Connections come from SOP_EXAMINER_URL / SOP_EXAMINER_TOKEN and
SOP_MODEL_URL / SOP_MODEL_TOKEN, or the matching options.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from test_client import ExaminerClient, ModelClient, RemoteError
from test_client.results import check_results, read_results
from test_client.scoring import score

from .policies import POLICIES
from .runner import LABELED_SPLITS, balanced, labeled_cases, load_cases, load_frozen, run_cases
from .storage import read_jsonl
from .uq import UQ_METHODS, Calibrator


def _connections(command, *, model):
    """Add examiner (and model) URL/token options, defaulting to the environment."""
    command.add_argument("--examiner-url", default=os.environ.get("SOP_EXAMINER_URL"))
    command.add_argument("--examiner-token", default=os.environ.get("SOP_EXAMINER_TOKEN"))
    if model:
        command.add_argument("--model-url", default=os.environ.get("SOP_MODEL_URL"))
        command.add_argument("--model-token", default=os.environ.get("SOP_MODEL_TOKEN"))


def _method(command):
    """Add the options that define a method and its output folder."""
    command.add_argument("--uq", choices=UQ_METHODS, default="self_report")
    command.add_argument("--calibrator", help="JSON file written by 'fit'")
    command.add_argument("--policy", choices=POLICIES, default="no_oracle")
    command.add_argument("--threshold", type=float, help="DEFER/query threshold override")
    command.add_argument("--fixed-count", type=int, default=1, help="oracle queries per case")
    command.add_argument(
        "--posterior-uq", choices=("auto", "self_report", "calibrated"), default="auto"
    )
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--output", required=True, help="run directory (resumable)")
    command.add_argument("--data", default="data", help="the public data directory")


def _method_kwargs(args):
    """``run_cases`` keyword arguments from the parsed method options."""
    return {
        "uq": args.uq,
        "calibrator": Calibrator.load(args.calibrator) if args.calibrator else None,
        "policy": args.policy,
        "threshold": args.threshold,
        "fixed_count": args.fixed_count,
        "posterior_uq": args.posterior_uq,
        "seed": args.seed,
    }


def main(argv=None):
    """Parse arguments, run one command and print its JSON result."""
    parser = argparse.ArgumentParser(
        prog="sop-candidate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="action", required=True)

    collect = commands.add_parser("collect", help="trajectories on labeled cases")
    _connections(collect, model=True)
    _method(collect)
    collect.add_argument(
        "--splits", default=",".join(LABELED_SPLITS), help="comma-separated labeled splits"
    )
    collect.add_argument("--limit", type=int, help="cases per split, domain-balanced (trial runs)")
    collect.add_argument("--initial-only", action="store_true", help="skip queries/decisions")

    fit = commands.add_parser("fit", help="fit risk estimators and thresholds")
    fit.add_argument("--trajectories", required=True, help="a 'collect' output directory")
    fit.add_argument(
        "--posterior", help="query trajectories (default: the --trajectories folder, if it has any)"
    )
    fit.add_argument("--select-split", choices=("cal_select", "dev"), default="cal_select")
    fit.add_argument("--output", required=True)
    fit.add_argument("--data", default="data")

    run = commands.add_parser("run", help="run a method on the test cases")
    _connections(run, model=True)
    _method(run)
    run.add_argument("--limit", type=int, help="domain-balanced number of test cases (trial)")

    score_cmd = commands.add_parser("score", help="score a train/dev results.csv locally")
    score_cmd.add_argument("results")
    score_cmd.add_argument("--data", default="data")

    submit = commands.add_parser("submit", help="upload a test results.csv (one try)")
    _connections(submit, model=False)
    submit.add_argument("results")

    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except (ValueError, OSError, RemoteError) as error:
        parser.exit(1, f"sop-candidate: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _dispatch(args):
    """Run the selected command; user errors surface as ValueError/RemoteError."""
    if args.action == "fit":
        cases = load_cases(args.data)
        initial = read_jsonl(Path(args.trajectories) / "initial.jsonl")
        calibrator = Calibrator().fit(cases, initial)
        calibrator.select_thresholds(cases, initial, split=args.select_split)
        posterior_file = Path(args.posterior or args.trajectories) / "posterior.jsonl"
        posterior = read_jsonl(posterior_file) if posterior_file.exists() else []
        if args.posterior and not posterior:
            raise ValueError(f"{posterior_file} has no query trajectories")
        if posterior:
            calibrator.fit_posterior(cases, posterior)
            try:
                calibrator.select_thresholds(
                    cases, posterior, split=args.select_split, posterior=True
                )
            except ValueError:
                pass  # no queried cases in the selection split: initial thresholds apply
        calibrator.save(args.output)
        result = {
            "calibrator": args.output,
            "fit_counts": calibrator.fit_counts,
            "thresholds": {k: v["threshold"] for k, v in calibrator.selections.items()},
        }
    elif args.action == "score":
        cases = {c["case_id"]: c for c in load_cases(args.data)}
        rows = check_results(read_results(args.results), cases, set(cases), require_all=False)
        result = score(rows, {cid: cases[cid] for cid in rows}, set(cases))
    else:
        examiner = ExaminerClient(args.examiner_url, args.examiner_token)
        if args.action == "submit":
            result = examiner.submit(args.results)
        else:
            model = ModelClient(args.model_url, args.model_token) if args.model_url else None
            if args.action == "collect":
                cases = labeled_cases(args.data, args.splits.split(","), args.limit)
                result = run_cases(
                    examiner,
                    cases,
                    args.output,
                    model=model,
                    initial_only=args.initial_only,
                    **_method_kwargs(args),
                )
            else:
                cases = balanced(load_cases(args.data, ["test"]), args.limit)
                result = run_cases(
                    examiner,
                    cases,
                    args.output,
                    model=model,
                    frozen=load_frozen(args.data),
                    **_method_kwargs(args),
                )
    return result


if __name__ == "__main__":
    main()
