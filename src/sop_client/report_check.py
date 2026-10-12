"""Local public-report checks that never consult hidden answers or a service.

Findings identify suspicious formats, not whether a reported value is true.
Only location and category are returned, so a credential is not echoed into a
second log. Human review remains necessary, including images and attachments.
"""

from __future__ import annotations

import re
from pathlib import Path

_TEXT_EXTENSIONS = {".md", ".txt", ".json", ".jsonl", ".csv", ".tex", ".html"}
_CREDENTIAL = re.compile(
    r"(?i)(?:\bbearer\s+[a-z0-9._~+/-]{12,}|"
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|"
    r"SOP_EXAMINER_TOKEN|SOP_MODEL_TOKEN)\s*[=:]\s*[\"']?[a-z0-9._~+/-]{12,})"
)
_CASE = re.compile(r"(?i)(?:case[_ -]?id|案例编号|测试题号)")
_FACT = re.compile(
    r"(?i)(?:oracle[_ -]?(?:value|evidence|receipt)|verified[_ -]?value|"
    r"\bgold\b|核验真值|核验值|真实值|真值表)"
)
_PRIVATE = re.compile(
    r"(?i)(?:\bprivate_state\b|\bnoise_condition\b|\bdeclared_parent\b|"
    r"\bother_participant\b|其他参加者.*(?:凭据|回执|核验))"
)


def check_public_report(paths):
    """Scan explicit UTF-8 text paths without transmitting or modifying them.

    The caller must review unscanned formats separately. A clear result is not a
    secrecy certificate: paraphrases, screenshots and unrecognised formats can
    evade these intentionally public, label-independent checks.
    """
    findings, unscanned, checked = [], [], []
    for raw in paths:
        path = Path(raw)
        name = str(path)
        if not path.is_file() or path.suffix.lower() not in _TEXT_EXTENSIONS:
            unscanned.append({"path": name, "reason": "not a supported text file"})
            continue
        if path.stat().st_size > 8 * 1024 * 1024:
            unscanned.append({"path": name, "reason": "text exceeds 8 MiB limit"})
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            unscanned.append({"path": name, "reason": "cannot read UTF-8 text"})
            continue
        checked.append(name)
        for index, line in enumerate(lines):
            kinds = []
            if _CREDENTIAL.search(line):
                kinds.append("possible_credential")
            if _PRIVATE.search(line):
                kinds.append("possible_private_record")
            # Cover adjacent CSV/Markdown header fields and multi-line JSON.
            if _FACT.search(line) and _CASE.search(
                "\n".join(lines[max(0, index - 8) : index + 9])
            ):
                kinds.append("possible_case_to_oracle_value_mapping")
            for kind in kinds:
                findings.append({"path": name, "line": index + 1, "category": kind})
    return {
        "status": "review_required" if findings or unscanned else "no_pattern_detected",
        "checked_files": checked,
        "findings": findings,
        "unscanned": unscanned,
        "uses_private_labels": False,
        "limitation": "Format screening only; human review is required before public release. No claim of complete detection or execution authenticity.",
    }
