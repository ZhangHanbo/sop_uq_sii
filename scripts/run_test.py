#!/usr/bin/env python3
"""Check a results.csv, then submit it to your examiner or score it locally.

    python scripts/run_test.py results.csv             # submit test results (uses one try)
    python scripts/run_test.py results.csv --local     # score train/dev results with public labels
    python scripts/run_test.py --submissions           # list your scored tries and the best one

The examiner URL and token come from SOP_EXAMINER_URL and SOP_EXAMINER_TOKEN.
Installing the package is not required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_client import ExaminerClient, RemoteError, read_results, score  # noqa: E402
from test_client.results import check_results  # noqa: E402


def read_cases(data, names):
    """Cases of the given data files (relative to ``data``), keyed by case_id."""
    cases = {}
    for name in names:
        for line in (data / name).read_text().splitlines():
            if line.strip():
                case = json.loads(line)
                cases[case["case_id"]] = case
    return cases


def summary(report):
    """One line per risk setting and the decision loss (domain-macro values)."""
    lines = [f"primary {report['primary']['metric']} = {report['primary']['value']}"]
    for section in ("initial", "updated_initial", "final"):
        m = report[section]["domain_macro"]
        lines.append(f"{section:16s} brier={m.get('brier')} nll={m.get('nll')} ece={m.get('ece')}")
    d = report["decision"]["domain_macro"]
    lines.append(
        f"{'decision':16s} loss={d.get('loss')} auto={d.get('auto_rate')} cost={d.get('mean_cost')}"
    )
    return "\n".join(lines)


def best_text(best):
    """The best try of each problem: Problem 1 among tries before the first test oracle read."""
    shown = [f"try {b['submission']} = {b['value']:.4f}" if b else "-" for b in best.values()]
    return f"problem 1 (p_i Brier) {shown[0]}; problem 2 (decision loss) {shown[1]}"


def main(argv=None):
    """Parse arguments, then submit, score locally or list tries; returns the exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="?", type=Path)
    parser.add_argument(
        "--local", action="store_true", help="score train/dev rows with public labels"
    )
    parser.add_argument("--submissions", action="store_true", help="list scored tries")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--examiner-url", default=os.environ.get("SOP_EXAMINER_URL"))
    parser.add_argument("--examiner-token", default=os.environ.get("SOP_EXAMINER_TOKEN"))
    parser.add_argument("--output", type=Path, help="also save the full JSON report here")
    args = parser.parse_args(argv)
    if (args.results is None) != args.submissions:
        parser.error("give a results.csv, or --submissions")
    try:
        if args.local:
            cases = read_cases(
                args.data,
                [
                    "train/train.jsonl",
                    "train/cal_fit.jsonl",
                    "train/cal_select.jsonl",
                    "train/dev.jsonl",
                ],
            )
            rows = check_results(read_results(args.results), cases, set(cases), require_all=False)
            report = score(rows, {cid: cases[cid] for cid in rows}, set(cases))
            print(summary(report))
        else:
            examiner = ExaminerClient(args.examiner_url, args.examiner_token)
            if args.submissions:
                report = examiner.submissions()
                for item in report["submissions"]:
                    print(f"try {item['submission']}: {item['primary']}")
                print("best:", best_text(report["best"]))
                print(
                    "tries left:",
                    report["submissions_left"],
                    "| Problem 1 closed at:",
                    report["problem1_closed_at"] or "- (still open)",
                )
            else:
                spec = examiner.spec()
                cases = read_cases(args.data, ["test/cases.jsonl"])
                check_results(read_results(args.results), cases, set(spec["interactive_ids"]))
                report = examiner.submit(args.results)
                if report.get("trial"):
                    print("trial token: the file is valid; not scored, no try used")
                else:
                    if report.get("duplicate"):
                        print(f"identical to try {report['submission']}; no try used")
                    else:
                        print(
                            f"try {report['submission']} scored; "
                            f"tries left: {report['submissions_left']}"
                        )
                    print(summary(report["report"]))
                    print("best so far:", best_text(report["best"]))
    except (ValueError, OSError, RemoteError) as error:
        parser.exit(1, f"run_test: {error}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
