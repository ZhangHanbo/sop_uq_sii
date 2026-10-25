"""The ``results.csv`` format: write, read and check one row per case.

Columns, in order::

    case_id, initial_answer_json, p_initial_error, mode, final_answer_json,
    p_initial_error_after_queries, p_final_error[, cost]

Answers are JSON objects (blank for no answer); probabilities are numbers in
[0, 1] (blank when unavailable); ``mode`` is AUTO, DEFER or blank. The last four
required columns are filled only for interactive cases. The optional ``cost``
column is used by local scoring of train/dev runs; the examiner ignores it and
uses its own ledger.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from .protocol import valid_answer, valid_probability

RESULT_COLUMNS = (
    "case_id",
    "initial_answer_json",
    "p_initial_error",
    "mode",
    "final_answer_json",
    "p_initial_error_after_queries",
    "p_final_error",
)
OPTIONAL_COLUMNS = ("cost",)
FINAL_COLUMNS = (
    "mode",
    "final_answer_json",
    "p_initial_error_after_queries",
    "p_final_error",
)
ANSWER_COLUMNS = ("initial_answer_json", "final_answer_json")
NUMBER_COLUMNS = ("p_initial_error", "p_initial_error_after_queries", "p_final_error", "cost")


def write_results(path, rows):
    """Write rows (dicts keyed by column name) as a sorted results.csv."""
    columns = RESULT_COLUMNS + (
        OPTIONAL_COLUMNS if any(row.get("cost") is not None for row in rows) else ()
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda value: value["case_id"]):
        encoded = {}
        for column in columns:
            value = row.get(column)
            if value is None:
                encoded[column] = ""
            elif column in ANSWER_COLUMNS:
                encoded[column] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                encoded[column] = value if isinstance(value, str) else repr(float(value))
        writer.writerow(encoded)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return path


def parse_results(text):
    """Parse results.csv text into ``{case_id: row}``; raise ValueError on bad cells."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿"), newline=""))
    header = tuple(reader.fieldnames or ())
    if header not in (RESULT_COLUMNS, RESULT_COLUMNS + OPTIONAL_COLUMNS):
        raise ValueError("Header must be: " + ",".join(RESULT_COLUMNS) + "[,cost]")
    rows = {}
    for line, raw in enumerate(reader, start=2):
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError(f"line {line}: wrong number of cells")
        row = {"case_id": raw["case_id"].strip()}
        for column in header[1:]:
            cell = raw[column].strip()
            try:
                if column in ANSWER_COLUMNS:
                    row[column] = json.loads(cell) if cell else None
                elif column in NUMBER_COLUMNS:
                    row[column] = float(cell) if cell else None
                    if row[column] is not None and not math.isfinite(row[column]):
                        raise ValueError("non-finite number")
                else:
                    row[column] = cell or None
            except ValueError as exc:
                raise ValueError(f"line {line}, {column}: {exc}") from None
        if not row["case_id"]:
            raise ValueError(f"line {line}: empty case_id")
        if row["case_id"] in rows:
            raise ValueError(f"line {line}: duplicate case_id {row['case_id']}")
        rows[row["case_id"]] = row
    return rows


def read_results(path):
    """Read a results.csv file into ``{case_id: row}``."""
    return parse_results(Path(path).read_text(encoding="utf-8-sig"))


def check_results(rows, cases, interactive_ids, *, require_all=True):
    """Check values against the cases; return the rows or raise ValueError.

    ``cases`` maps case_id to a dict with a ``domain``. With ``require_all``
    every case must appear exactly once; otherwise rows may cover a subset.
    """
    problems = []
    unknown = sorted(set(rows) - set(cases))
    if unknown:
        problems.append(f"{len(unknown)} unknown case_id(s), e.g. {unknown[0]}")
    missing = sorted(set(cases) - set(rows))
    if require_all and missing:
        problems.append(f"{len(missing)} case(s) missing, e.g. {missing[0]}")
    for case_id, row in sorted(rows.items()):
        if case_id not in cases:
            continue
        domain = cases[case_id]["domain"]
        for column in ANSWER_COLUMNS:
            if row.get(column) is not None and not valid_answer(row[column], domain):
                problems.append(f"{case_id}: {column} is not a legal {domain} answer")
        for column in ("p_initial_error", "p_initial_error_after_queries", "p_final_error"):
            if row.get(column) is not None and not valid_probability(row[column]):
                problems.append(f"{case_id}: {column} must be in [0, 1] or blank")
        if row.get("mode") not in (None, "AUTO", "DEFER"):
            problems.append(f"{case_id}: mode must be AUTO, DEFER or blank")
        if case_id not in interactive_ids and any(
            row.get(column) is not None for column in FINAL_COLUMNS
        ):
            problems.append(f"{case_id}: final columns must be blank for a non-interactive case")
    if problems:
        shown = "; ".join(problems[:10])
        more = f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""
        raise ValueError(shown + more)
    return rows
