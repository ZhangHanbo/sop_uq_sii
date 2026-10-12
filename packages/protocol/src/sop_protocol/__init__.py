"""Public data and submission contracts shared by participant and examiner.

Contains only public schemas: domains, input and target fields, legal answers,
canonical hashing, and the ``submission.zip`` format. No generator, private
state or hidden label is defined here.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import tempfile
import zipfile
import zlib

# Data files keep their published schema independently of the mutable service
# and submission contracts. VERSION remains a data-only compatibility alias.
DATA_SCHEMA_VERSION = "2.1"
VERSION = DATA_SCHEMA_VERSION
SERVICE_PROTOCOL_VERSION = "2.2"
SUBMISSION_SCHEMA_VERSION = "2.2"
CAPABILITIES = (
    "per_case_risk_checkpoint",
    "shared_test_ledger",
    "submission_versions",
    "dev_unlimited_oracle",
)
# Public field schemas of the three business domains.
DOMAINS = ("dangerous_goods", "aircraft_inspection", "customer_service")
AIRCRAFT_FIELDS = (
    "aircraft_ready",
    "mechanical_inspection_result",
    "electrical_inspection_result",
    "component_incident_response",
    "component_mismatch_response",
    "cross_check_response",
    "cross_check_reporting_response",
)
INPUT_FIELDS = {
    "dangerous_goods": (
        "product_id",
        "sds_label_text",
        "handling_and_storage_guidelines",
        "transportation_requirements",
        "disposal_guidelines",
    ),
    "aircraft_inspection": (
        "aircraft_id",
        "tail_number",
        "maintenance_record_id",
        "expected_departure_time",
        "component_serial_number",
        "inspection_location_id",
        "component_weight",
        "physical_condition_observation",
        "installation_time",
        "battery_status",
        "circuit_continuity_check",
        "avionics_diagnostics_response",
        "installed_component_serial_number",
        "expected_component_weight",
        "actual_inspection_time",
    ),
    "customer_service": (
        "account_id",
        "service_area_code",
        "service_type",
        "subscribed_bandwidth",
    ),
}
TARGET_FIELDS = {
    "dangerous_goods": ("hazard_class",),
    "aircraft_inspection": AIRCRAFT_FIELDS,
    "customer_service": ("final_resolution_status",),
}
# Columns of results.csv inside submission.zip.
RESULT_COLUMNS = (
    "case_id",
    "model_slot",
    "episode_id",
    "initial_commit_id",
    "initial_answer_json",
    "p_initial_error",
    "mode",
    "final_commit_id",
    "final_answer_json",
    "p_initial_error_after_queries",
    "p_final_error",
    "status",
)
PROBABILITY_COLUMNS = (
    "p_initial_error",
    "p_initial_error_after_queries",
    "p_final_error",
)
ANSWER_COLUMNS = ("initial_answer_json", "final_answer_json")
# Size limits that protect the evaluator from oversized or zip-bomb uploads.
MAX_PACKAGE_BYTES = 16 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


def json_bytes(obj):
    """Canonical UTF-8 JSON: sorted keys, no spaces, no NaN or infinity."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(obj):
    """SHA-256 hex digest of the canonical JSON form of ``obj``."""
    return hashlib.sha256(json_bytes(obj)).hexdigest()


def decision_terminal_loss(
    mode,
    *,
    controller_valid=True,
    answer_present=True,
    error=False,
    severe=False,
    defer_cost=0.5,
):
    """Public terminal cost, excluding the examiner's accumulated tool bill.

    A model failure can be deferred; an invalid controller cannot erase its
    penalty by declaring DEFER. Both selection and official scoring use this
    function, with correctness/severity computed from their own allowed labels.
    """
    if not controller_valid or mode not in ("AUTO", "DEFER"):
        return 5.0
    if mode == "DEFER":
        return float(defer_cost)
    if not answer_present or severe:
        return 5.0
    return float(bool(error))


def valid_probability(p):
    """True for a finite real number in [0, 1]; booleans are rejected."""
    return (
        isinstance(p, (int, float))
        and not isinstance(p, bool)
        and math.isfinite(p)
        and 0 <= p <= 1
    )


def valid_answer(answer, domain):
    """True if ``answer`` has exactly the main fields of ``domain`` with legal values."""
    if (
        domain not in TARGET_FIELDS
        or not isinstance(answer, dict)
        or set(answer) != set(TARGET_FIELDS[domain])
    ):
        return False
    if domain == "dangerous_goods":
        return answer["hazard_class"] in (
            "Unable to Decide",
            "Hazard Class A",
            "Hazard Class B",
            "Hazard Class C",
            "Hazard Class D",
        )
    if domain == "customer_service":
        return answer["final_resolution_status"] in (
            "FAILED",
            "PENDING_ACTION",
            "RESOLVED",
            "ESCALATED",
        )
    if type(answer["aircraft_ready"]) is not bool:
        return False
    for key in AIRCRAFT_FIELDS[1:]:
        allowed = (
            ("success", "fail", "retest")
            if key in ("mechanical_inspection_result", "electrical_inspection_result")
            else ("success", "failed", None)
        )
        if answer[key] not in allowed:
            return False
    return True


def validate_public_case(row, training=False):
    """Check one public case row against the field allowlist and return it.

    Training rows must also carry a legal ``label`` and a ``training_split``.
    """
    fields = {"schema_version", "dataset_release_id", "case_id", "domain", "inputs"}
    if training:
        fields.update(("label", "training_split"))
    if not isinstance(row, dict) or set(row) != fields:
        raise ValueError("Public case must match the exact field allowlist")
    if row["schema_version"] != VERSION or row["domain"] not in DOMAINS:
        raise ValueError("Unsupported public case version/domain")
    if not all(
        isinstance(row[k], str) and row[k] for k in ("case_id", "dataset_release_id")
    ):
        raise ValueError("Public identifiers must be nonempty strings")
    inputs = row["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != set(INPUT_FIELDS[row["domain"]]):
        raise ValueError("Unexpected public input field")
    if any(
        not isinstance(v, (str, int, float, bool))
        or isinstance(v, float)
        and not math.isfinite(v)
        for v in inputs.values()
    ):
        raise ValueError("Inputs must contain finite scalar business values")
    if training:
        if row["training_split"] not in (
            "train",
            "cal_fit",
            "cal_select",
            "dev",
        ) or not valid_answer(row["label"], row["domain"]):
            raise ValueError("Invalid training target/split")
    return row


def _strict_json(raw):
    """Parse JSON, rejecting duplicate keys and non-finite numbers."""

    def pairs(values):
        result = {}
        for k, v in values:
            if k in result:
                raise ValueError("Duplicate JSON key")
            result[k] = v
        return result

    def invalid(_):
        raise ValueError("Nonfinite JSON value")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _manifest_valid(manifest):
    """Check the exact field set and value types of ``manifest.json``."""
    required = {
        "schema_version",
        "protocol_version",
        "dataset_release_id",
        "evaluation_profile_id",
        "run_id",
        "method_id",
        "model_profiles",
        "file_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("Manifest fields do not match the public schema")
    if (
        manifest["schema_version"] != SUBMISSION_SCHEMA_VERSION
        or manifest["protocol_version"] != SERVICE_PROTOCOL_VERSION
    ):
        raise ValueError("Unsupported submission version")
    if not all(
        isinstance(manifest[k], str) and 0 < len(manifest[k]) <= 256
        for k in required - {"model_profiles", "file_sha256"}
    ):
        raise ValueError("Invalid manifest identifier")
    if (
        not isinstance(manifest["model_profiles"], dict)
        or not manifest["model_profiles"]
        or not all(
            isinstance(k, str) and isinstance(v, str) and k and v
            for k, v in manifest["model_profiles"].items()
        )
    ):
        raise ValueError("model_profiles must map slots to declared profile hashes")
    if not isinstance(manifest["file_sha256"], dict):
        raise ValueError("Invalid file hashes")


def _validate_row(row):
    """Check the column set and value types of one results.csv row."""
    if set(row) != set(RESULT_COLUMNS):
        raise ValueError("Unexpected result columns")
    if any(
        not isinstance(row[k], str)
        for k in set(RESULT_COLUMNS) - set(PROBABILITY_COLUMNS) - set(ANSWER_COLUMNS)
    ):
        raise ValueError("Identifiers, mode and status must be strings")
    if any(not row[k] for k in ("case_id", "model_slot")):
        raise ValueError("case_id and model_slot are required")
    if row["mode"] not in ("", "AUTO", "DEFER") or row["status"] not in (
        "completed",
        "failed",
        "missing",
    ):
        raise ValueError("Invalid mode/status")
    for key in PROBABILITY_COLUMNS:
        if row[key] is not None and not valid_probability(row[key]):
            raise ValueError("Probability must be null or a finite number in [0,1]")
    for key in ANSWER_COLUMNS:
        if row[key] is not None and not isinstance(row[key], dict):
            raise ValueError("Answer must be a JSON object or null")
    json_bytes(row)


def write_submission(output, manifest, rows):
    """Write a deterministic ``submission.zip`` with ``manifest.json`` and ``results.csv``.

    Rows are sorted and encoded canonically, the CSV hash is stored in the
    manifest, and the bundle is re-read before it atomically replaces ``output``.
    Costs are not part of the bundle: the examiner takes them from its ledger.
    """
    data = io.StringIO(newline="")
    writer = csv.DictWriter(data, fieldnames=RESULT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    seen = set()
    logical_rows = [dict(row) for row in rows]
    for row in logical_rows:
        _validate_row(row)
    for row in sorted(
        logical_rows, key=lambda value: (value["model_slot"], value["case_id"])
    ):
        key = (row["model_slot"], row["case_id"])
        if key in seen:
            raise ValueError("Duplicate case/model row")
        seen.add(key)
        encoded = dict(row)
        for name in ANSWER_COLUMNS:
            encoded[name] = json_bytes(row[name]).decode()
        for name in PROBABILITY_COLUMNS:
            encoded[name] = "" if row[name] is None else repr(row[name])
        writer.writerow(encoded)
    csv_bytes = data.getvalue().encode("utf-8")
    meta = copy.deepcopy(manifest)
    meta["file_sha256"] = {"results.csv": hashlib.sha256(csv_bytes).hexdigest()}
    _manifest_valid(meta)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
        # Repacking the same sealed records must remain an exact retry even
        # after a process restart or a different wall-clock time.
        for name, payload in (
            ("manifest.json", json_bytes(meta) + b"\n"),
            ("results.csv", csv_bytes),
        ):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
    raw = buffer.getvalue()
    read_submission(raw)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".submission-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
        os.replace(tmp, output)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return meta


def _zip_directory_offset(raw):
    """Bound the directory before ZipFile allocates one object per entry."""
    offset = raw.rfind(b"PK\x05\x06", max(0, len(raw) - 65557))
    if offset < 0 or offset + 22 > len(raw):
        raise ValueError("Missing ZIP directory")
    signature, disk, directory_disk, entries_disk, entries, size, start, comment = (
        struct.unpack_from("<4s4H2IH", raw, offset)
    )
    if (
        offset + 22 + comment != len(raw)
        or disk
        or directory_disk
        or entries != entries_disk
    ):
        raise ValueError("Split or ambiguous ZIP archives are forbidden")
    if entries not in (2, 3) or start + size != offset or start > len(raw):
        raise ValueError("Unexpected ZIP directory size/count")
    # The EOCD count is untrusted too: ZipFile parses by directory byte size.
    cursor, actual_entries = start, 0
    while cursor < offset:
        if (
            actual_entries >= 3
            or cursor + 46 > offset
            or raw[cursor : cursor + 4] != b"PK\x01\x02"
        ):
            raise ValueError("Malformed or excessive ZIP directory entries")
        name_size, extra_size, comment_size = struct.unpack_from(
            "<HHH", raw, cursor + 28
        )
        cursor += 46 + name_size + extra_size + comment_size
        actual_entries += 1
    if cursor != offset or actual_entries != entries:
        raise ValueError("ZIP directory entry count/size mismatch")
    return start


def _bounded_zip_payloads(raw, infos, directory_start):
    """Decode actual bytes with a hard budget, then verify declared size/CRC.

    ZipExtFile.read() trusts file_size when truncating decoded bytes, and its
    unbounded form can allocate far beyond that forged size before truncation.
    This decoder controls zlib's output size and requires the true stream EOF.
    """
    decoded, spans, actual_total = {}, [], 0
    chunk_size = 65536
    for info in infos:
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ValueError("Only stored and deflate ZIP entries are supported")
        start = info.header_offset
        if start < 0 or start + 30 > directory_start:
            raise ValueError("Invalid local ZIP header")
        header = struct.unpack_from("<4s5H3I2H", raw, start)
        (
            signature,
            version,
            flags,
            method,
            mod_time,
            mod_date,
            crc,
            compressed_size,
            file_size,
            name_size,
            extra_size,
        ) = header
        data_start = start + 30 + name_size + extra_size
        data_end = data_start + info.compress_size
        if (
            signature != b"PK\x03\x04"
            or flags != info.flag_bits
            or method != info.compress_type
            or data_end > directory_start
        ):
            raise ValueError("Local and central ZIP headers disagree")
        if flags & ~0x80E:
            raise ValueError("Unsupported ZIP flags")
        name_bytes = raw[start + 30 : start + 30 + name_size]
        try:
            name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        except UnicodeError:
            raise ValueError("Invalid ZIP entry name") from None
        if name != info.filename or "\0" in name:
            raise ValueError("Local and central ZIP names disagree")
        expected = (info.CRC, info.compress_size, info.file_size)
        if flags & 8:
            if any(
                actual not in (0, required)
                for actual, required in zip((crc, compressed_size, file_size), expected)
            ):
                raise ValueError("Invalid ZIP streaming header")
            descriptor_start = data_end + (
                4 if raw[data_end : data_end + 4] == b"PK\x07\x08" else 0
            )
            span_end = descriptor_start + 12
            if (
                span_end > directory_start
                or struct.unpack_from("<III", raw, descriptor_start) != expected
            ):
                raise ValueError("Invalid ZIP data descriptor")
        else:
            if (crc, compressed_size, file_size) != expected:
                raise ValueError("Local and central ZIP sizes/checksums disagree")
            span_end = data_end
        if any(
            start < previous_end and previous_start < span_end
            for previous_start, previous_end in spans
        ):
            raise ValueError("Overlapping ZIP entries are forbidden")
        spans.append((start, span_end))
        chunks, actual_size, actual_crc = [], 0, 0
        if method == zipfile.ZIP_STORED:
            if info.compress_size != info.file_size:
                raise ValueError("Stored ZIP sizes disagree")
            if info.compress_size > MAX_UNCOMPRESSED_BYTES - actual_total:
                raise ValueError("Expanded submission too large")
            value = raw[data_start:data_end]
            chunks.append(value)
            actual_size = len(value)
            actual_total += actual_size
            actual_crc = zlib.crc32(value)
        else:
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            cursor = data_start
            while cursor < data_end:
                block = raw[cursor : min(cursor + chunk_size, data_end)]
                cursor += len(block)
                while block:
                    remaining = min(
                        MAX_UNCOMPRESSED_BYTES - actual_total,
                        info.file_size - actual_size,
                    )
                    try:
                        value = decompressor.decompress(
                            block, min(chunk_size, remaining + 1)
                        )
                    except zlib.error:
                        raise ValueError("Invalid deflate stream") from None
                    actual_total += len(value)
                    actual_size += len(value)
                    if actual_total > MAX_UNCOMPRESSED_BYTES:
                        raise ValueError("Expanded submission too large")
                    if actual_size > info.file_size:
                        raise ValueError("Actual ZIP size exceeds its declared size")
                    if value:
                        chunks.append(value)
                        actual_crc = zlib.crc32(value, actual_crc)
                    if decompressor.eof:
                        if decompressor.unused_data or cursor != data_end:
                            raise ValueError("Trailing data after deflate stream")
                        break
                    block = decompressor.unconsumed_tail
            if not decompressor.eof:
                raise ValueError("Truncated deflate stream")
        if actual_size != info.file_size or actual_crc != info.CRC:
            raise ValueError("Actual ZIP size/checksum does not match its directory")
        decoded[info.filename] = b"".join(chunks)
    return decoded


def read_submission(path_or_bytes):
    """Validate a ``submission.zip`` and return ``(manifest, rows)``.

    The archive is parsed defensively: entry count, sizes, checksums, names and
    decompressed volume are verified before any content is trusted.
    """
    if isinstance(path_or_bytes, (str, Path)):
        if Path(path_or_bytes).stat().st_size > MAX_PACKAGE_BYTES:
            raise ValueError("Submission too large")
        raw = Path(path_or_bytes).read_bytes()
    else:
        raw = bytes(path_or_bytes)
    if len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("Submission too large")
    directory_start = _zip_directory_offset(raw)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            infos = zf.infolist()
            names = [i.filename for i in infos]
            if any(i.orig_filename != i.filename for i in infos):
                raise ValueError("Ambiguous ZIP entry names are forbidden")
            if (
                len(set(names)) != len(names)
                or not {"manifest.json", "results.csv"} <= set(names)
                or set(names) - {"manifest.json", "results.csv", "trace.jsonl"}
            ):
                raise ValueError("ZIP must contain unique, allowlisted file names")
            if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Expanded submission too large")
            if any(
                i.flag_bits & 1 or stat.S_ISLNK(i.external_attr >> 16) or i.is_dir()
                for i in infos
            ):
                raise ValueError(
                    "Encrypted files, symlinks and directories are forbidden"
                )
            decoded = _bounded_zip_payloads(raw, infos, directory_start)
            manifest = _strict_json(decoded.pop("manifest.json"))
            _manifest_valid(manifest)
            payloads = decoded
    except (zipfile.BadZipFile, UnicodeError, KeyError, TypeError) as exc:
        raise ValueError("Malformed submission bundle") from exc
    if set(manifest["file_sha256"]) != set(payloads) or any(
        manifest["file_sha256"][n] != hashlib.sha256(b).hexdigest()
        for n, b in payloads.items()
    ):
        raise ValueError("Submission file hash mismatch")
    reader = csv.DictReader(
        io.StringIO(payloads["results.csv"].decode("utf-8"), newline="")
    )
    if reader.fieldnames != list(RESULT_COLUMNS):
        raise ValueError("CSV header does not match versioned result columns")
    rows, seen = [], set()
    for row in reader:
        if len(rows) >= 32000 or None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed CSV row or excessive row count")
        for name in ANSWER_COLUMNS:
            row[name] = _strict_json(row[name])
        for name in PROBABILITY_COLUMNS:
            row[name] = None if row[name] == "" else float(row[name])
        _validate_row(row)
        key = (row["model_slot"], row["case_id"])
        if key in seen:
            raise ValueError("Duplicate case/model row")
        seen.add(key)
        rows.append(row)
    return manifest, rows
