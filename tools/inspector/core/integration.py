"""Read-only Platform Contract v5 readiness checks.

The large series payload is streamed in passes. Split masks are read before
numeric/provenance arrays, and payload values at test indices are never retained
or validated as research evidence.
"""
from __future__ import annotations

import json
import math
import re
from array import array
from datetime import date
from pathlib import Path
from typing import Any

from scripts.platform_channels import CHANNEL_REGISTRY, SCORE_CHECKPOINTS, SCORE_REGISTRY

from .config import InspectorConfig
from .security import assert_within

try:
    from ijson.common import JSONError as _IjsonError
except ImportError:  # ijson is optional; the guards below report its absence
    _IjsonError = None

# ijson raises JSONError (not ValueError) on malformed JSON, including bare NaN
# literals. A single unreadable artifact must surface as a finding, never abort
# the whole scan (see tools/inspector/app.py initial-scan flow).
STREAM_ERRORS: tuple[type[BaseException], ...] = (
    OSError, PermissionError, UnicodeError, ValueError, TypeError, OverflowError, json.JSONDecodeError,
)
if _IjsonError is not None:
    STREAM_ERRORS += (_IjsonError,)

FARM_ID = re.compile(r"^F-\d{3,}$")
ALLOWED_SPLITS = {"train", "dev", "test", "monitoring"}
ALLOWED_CHANNEL_PROVENANCE = {"observed", "modelled", "proxy", "absent"}
ALLOWED_SCORE_PROVENANCE = {"model", "proxy", "absent"}
ALLOWED_FLAGS = {"normal", "suspected_stress", "no_data", None}
MAX_AXIS = 20_000
MAX_FARMS = 5_000
MAX_FARM_OBSERVATIONS = 5_000_000


def _finding(name: str, status: str, reason: str, critical: bool = False, **details: Any) -> dict[str, Any]:
    return {"check": name, "status": status, "critical": critical, "reason": reason, **details}


def _read_farms(config: InspectorConfig, checks: list[dict[str, Any]]) -> tuple[set[str], dict[str, dict[str, Any]]]:
    root = next((item for item in config.roots if item.name == "platform_data"), None)
    if root is None:
        checks.append(_finding("farm_registry", "WARNING", "No platform_data root is configured"))
        return set(), {}
    path = root.path / "farms_khuzestan.json"
    try:
        assert_within(path, root.path)
        if not path.is_file():
            checks.append(_finding("farm_registry", "WARNING", "farms_khuzestan.json is absent"))
            return set(), {}
        if path.stat().st_size > config.max_json_bytes:
            checks.append(_finding("farm_registry", "WARNING", "Farm registry exceeds configured JSON read limit"))
            return set(), {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except STREAM_ERRORS as exc:
        checks.append(_finding("farm_registry", "FAIL", f"Farm registry unreadable: {exc.__class__.__name__}", True))
        return set(), {}
    if not isinstance(value, list) or not value or len(value) > MAX_FARMS:
        checks.append(_finding("farm_registry", "FAIL", "Farm registry must be a non-empty bounded array", True))
        return set(), {}
    records: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    duplicate: list[str] = []
    bad_geometry: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            invalid.append("<non-object>")
            continue
        farm_id = item.get("id")
        if not isinstance(farm_id, str) or not FARM_ID.fullmatch(farm_id):
            invalid.append(str(farm_id)[:40])
            continue
        if farm_id in records:
            duplicate.append(farm_id)
            continue
        centroid, polygon = item.get("centroid"), item.get("polygon")
        valid = isinstance(centroid, list) and len(centroid) == 2
        if valid:
            lat, lon = centroid
            valid = (isinstance(lat, (int, float)) and not isinstance(lat, bool)
                     and isinstance(lon, (int, float)) and not isinstance(lon, bool)
                     and math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180)
        if not isinstance(polygon, list) or len(polygon) < 4:
            valid = False
        else:
            for point in polygon:
                if not isinstance(point, list) or len(point) != 2:
                    valid = False
                    break
                plat, plon = point
                if (not isinstance(plat, (int, float)) or isinstance(plat, bool)
                        or not isinstance(plon, (int, float)) or isinstance(plon, bool)
                        or not math.isfinite(plat) or not math.isfinite(plon)
                        or not -90 <= plat <= 90 or not -180 <= plon <= 180):
                    valid = False
                    break
        if not valid:
            bad_geometry.append(farm_id)
        records[farm_id] = item
    checks.append(_finding("farm_ids", "FAIL" if invalid or duplicate else "PASS",
        f"Checked {len(records)} unique registry IDs; invalid={len(invalid)}, duplicate={len(duplicate)}",
        bool(invalid or duplicate), examples=(invalid + duplicate)[:5]))
    checks.append(_finding("farm_geometry", "FAIL" if bad_geometry else "PASS",
        f"Centroids and polygons checked in registry lat/lon order; invalid={len(bad_geometry)}",
        bool(bad_geometry), examples=bad_geometry[:5]))
    checks.append(_finding("farm_crs", "WARNING", "Registry coordinates are interpreted as WGS84 latitude/longitude; CRS is not embedded in the source JSON"))
    return set(records), records


def _axis_and_version(path: Path) -> tuple[int | None, list[str] | None, str | None]:
    try:
        import ijson
    except ImportError:
        return None, None, "Install optional ijson for bounded series validation"
    version = None
    axis: list[str] = []
    try:
        with path.open("rb") as stream:
            for prefix, event, value in ijson.parse(stream):
                if prefix == "contractVersion" and event == "number":
                    version = int(value)
                elif prefix == "axis.item" and event == "string":
                    if len(axis) >= MAX_AXIS:
                        return version, None, "Date axis exceeds the configured safety bound"
                    axis.append(value)
    except STREAM_ERRORS as exc:
        return version, None, f"Series structure unreadable: {exc.__class__.__name__}"
    if not axis:
        return version, None, "Series date axis is absent"
    try:
        if any(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None for value in axis):
            raise ValueError("date format")
        parsed_dates = [date.fromisoformat(value) for value in axis]
        if any(left >= right for left, right in zip(parsed_dates, parsed_dates[1:])):
            raise ValueError("dates must be unique and increasing")
    except ValueError:
        return version, None, "Date axis contains invalid ISO calendar dates"
    return version, axis, None


def _checkpoint_alignment(path: Path, config: InspectorConfig) -> dict[str, Any]:
    try:
        import ijson
    except ImportError:
        return _finding("checkpoint_sidecars", "WARNING", "Cannot verify checkpoint references without optional ijson")
    try:
        with path.open("rb") as stream:
            payload_refs = dict(ijson.kvitems(stream, "checkpoints"))
    except STREAM_ERRORS as exc:
        return _finding("checkpoint_sidecars", "FAIL", f"Checkpoint mapping unreadable: {exc.__class__.__name__}", True)
    expected = {str(key): str(value) for key, value in SCORE_CHECKPOINTS.items()}
    if payload_refs != expected:
        return _finding("checkpoint_sidecars", "FAIL", "Payload checkpoint mapping does not match scripts/platform_channels.py", True,
                        expected_count=len(expected), actual_count=len(payload_refs))
    checkpoint_root = next((item.path for item in config.roots if item.name == "checkpoints"), None)
    if checkpoint_root is None:
        return _finding("checkpoint_sidecars", "WARNING", "Checkpoint root is not configured; weight/sidecar presence was not checked")
    missing: list[str] = []
    mismatched: list[str] = []
    try:
        for score, stem in expected.items():
            sidecar = checkpoint_root / f"{stem}.json"
            assert_within(sidecar, checkpoint_root)
            if not sidecar.is_file():
                missing.append(stem)
                continue
            if sidecar.stat().st_size > config.max_json_bytes:
                missing.append(f"{stem} (sidecar exceeds metadata bound)")
                continue
            metadata: dict[str, Any] = {}
            # Parse only identity and ladder metadata; dev metrics and any other
            # sidecar values are deliberately not materialized or used in gates.
            with sidecar.open("rb") as stream:
                for prefix, event, value in ijson.parse(stream):
                    if prefix in {"model_id", "ladder_rung"} and event in {"string", "null"}:
                        metadata[prefix] = value
            if metadata.get("model_id") != stem:
                mismatched.append(stem)
    except STREAM_ERRORS as exc:
        return _finding("checkpoint_sidecars", "WARNING", f"Checkpoint sidecar metadata unavailable: {exc.__class__.__name__}")
    if mismatched:
        return _finding("checkpoint_sidecars", "FAIL", "Checkpoint sidecar model IDs do not align with registered stems", True, examples=mismatched[:5])
    if missing:
        return _finding("checkpoint_sidecars", "WARNING", f"Sidecar metadata missing/unavailable for {len(missing)} registered checkpoint(s); weights were not loaded", examples=missing[:5])
    return _finding("checkpoint_sidecars", "PASS", "Registered checkpoint references align with metadata-only sidecar IDs; weights were not loaded")


def _read_test_masks(path: Path, farm_ids: set[str], axis_length: int) -> tuple[dict[str, bytearray], dict[str, array], list[str]]:
    import ijson
    masks = {farm_id: bytearray(axis_length) for farm_id in farm_ids}
    positions = {farm_id: 0 for farm_id in farm_ids}
    issues: list[str] = []
    active: str | None = None
    pair: list[Any] = []
    try:
        with path.open("rb") as stream:
            for prefix, event, value in ijson.parse(stream):
                if prefix.startswith("fields.") and prefix.endswith(".split") and event == "start_array":
                    farm_id = prefix[len("fields."):-len(".split")]
                    active = farm_id if farm_id in masks else None
                    pair = []
                elif active is not None and prefix == f"fields.{active}.split.item.item" and event in {"string", "number", "null", "boolean"}:
                    pair.append(value)
                elif active is not None and prefix == f"fields.{active}.split.item" and event == "end_array":
                    if len(pair) != 2:
                        issues.append(f"{active}: malformed split RLE pair")
                    else:
                        split, count = pair
                        start = positions[active]
                        if not isinstance(count, int) or isinstance(count, bool) or count < 0 or start + count > axis_length:
                            issues.append(f"{active}: invalid split RLE count")
                        elif split not in ALLOWED_SPLITS:
                            issues.append(f"{active}: unknown split tag")
                        else:
                            if split == "test":
                                masks[active][start:start + count] = b"\x01" * count
                            positions[active] = start + count
                    pair = []
                elif active is not None and prefix == f"fields.{active}.split" and event == "end_array":
                    active = None
                    pair = []
    except STREAM_ERRORS as exc:
        issues.append(f"split tags unreadable: {exc.__class__.__name__}")
    incomplete = [farm_id for farm_id, pos in positions.items() if pos != axis_length]
    issues.extend(f"{farm_id}: split tags do not align with axis" for farm_id in incomplete[:20])
    prefixes: dict[str, array] = {}
    for farm_id, mask in masks.items():
        prefix = array("I", [0])
        count = 0
        for is_test in mask:
            count += is_test
            prefix.append(count)
        prefixes[farm_id] = prefix
    return masks, prefixes, issues


def _validate_series_values(path: Path, axis_length: int, masks: dict[str, bytearray], test_prefixes: dict[str, array], farm_ids: set[str]) -> tuple[list[dict[str, Any]], set[str]]:
    import ijson
    channels = {spec.name: spec for spec in CHANNEL_REGISTRY}
    scores = {spec.name for spec in SCORE_REGISTRY}
    found_fields: set[str] = set()
    channel_arrays: set[tuple[str, str]] = set()
    score_arrays: set[tuple[str, str]] = set()
    channel_prov: set[tuple[str, str]] = set()
    score_prov: set[tuple[str, str]] = set()
    score_presence: dict[tuple[str, str], bytearray] = {}
    flags: set[str] = set()
    models: set[str] = set()
    suspected_masks: dict[str, bytearray] = {farm_id: bytearray(axis_length) for farm_id in farm_ids}
    rle_positions: dict[tuple[str, str, str], int] = {}
    active_data: tuple[str, str, str] | None = None
    data_position = 0
    active_rle: tuple[str, str, str] | None = None
    pair: list[Any] = []
    checks: list[dict[str, Any]] = []
    range_violations: dict[str, list[str]] = {}
    malformed: list[str] = []
    arrays_bad: list[str] = []
    provenance_bad: list[str] = []
    field_ids_seen: set[str] = set()
    ladder_missing: list[str] = []
    def add_issue(target: list[str], message: str) -> None:
        if len(target) < 100:
            target.append(message)
        elif len(target) == 100:
            target.append("Additional findings omitted by safety limit")
    try:
        with path.open("rb") as stream:
            for prefix, event, value in ijson.parse(stream):
                parts = prefix.split(".")
                if len(parts) == 2 and parts[0] == "fields" and event == "start_map":
                    farm_id = parts[1]
                    if farm_id in farm_ids:
                        field_ids_seen.add(farm_id)
                    else:
                        add_issue(malformed, f"unknown field ID {farm_id[:40]}")
                if len(parts) == 4 and parts[0] == "fields" and parts[1] in farm_ids and parts[2] in {"channels", "scores"} and event == "start_array":
                    farm_id, kind, name = parts[1], parts[2], parts[3]
                    active_data = (farm_id, kind, name)
                    data_position = 0
                    if kind == "channels":
                        if name in channels:
                            channel_arrays.add((farm_id, name))
                        else:
                            add_issue(malformed, f"{farm_id}: unregistered channel {name}")
                    else:
                        if name in scores:
                            score_arrays.add((farm_id, name))
                            score_presence[(farm_id, name)] = bytearray(axis_length)
                        else:
                            add_issue(malformed, f"{farm_id}: unregistered score {name}")
                elif active_data is not None and prefix == f"fields.{active_data[0]}.{active_data[1]}.{active_data[2]}.item" and event in {"number", "null", "string", "boolean"}:
                    farm_id, kind, name = active_data
                    is_test = bool(masks.get(farm_id, bytearray())[data_position]) if data_position < axis_length else True
                    if data_position >= axis_length:
                        add_issue(arrays_bad, f"{farm_id}.{kind}.{name}: exceeds date axis")
                    elif not is_test:
                        if kind == "scores" and (farm_id, name) in score_presence and event == "number":
                            score_presence[(farm_id, name)][data_position] = 1
                        if event == "number":
                            try:
                                number = float(value)
                            except (TypeError, ValueError, OverflowError):
                                number = math.nan
                            if not math.isfinite(number):
                                add_issue(malformed, f"{farm_id}.{kind}.{name}: non-finite numeric value")
                            elif kind == "channels" and name in channels:
                                lo, hi = channels[name].physical_range
                                if not lo <= number <= hi:
                                    examples = range_violations.setdefault(name, [])
                                    if len(examples) < 5:
                                        examples.append(f"{farm_id}@{data_position}")
                        elif event != "null":
                            add_issue(malformed, f"{farm_id}.{kind}.{name}: expected number or null")
                    data_position += 1
                elif active_data is not None and prefix == f"fields.{active_data[0]}.{active_data[1]}.{active_data[2]}" and event == "end_array":
                    if data_position != axis_length:
                        add_issue(arrays_bad, f"{active_data[0]}.{active_data[1]}.{active_data[2]}: {data_position}/{axis_length} values")
                    active_data = None
                if prefix.startswith("fields.") and event == "start_array":
                    rle_key = None
                    if len(parts) == 3 and parts[1] in farm_ids and parts[2] in {"stressFlag", "stressFlagModel"}:
                        rle_key = (parts[1], parts[2], "")
                    elif len(parts) == 4 and parts[1] in farm_ids and parts[2] in {"channelProvenance", "scoreProvenance"}:
                        rle_key = (parts[1], parts[2], parts[3])
                    if rle_key is not None:
                        if ((rle_key[1] == "channelProvenance" and rle_key[2] not in channels)
                                or (rle_key[1] == "scoreProvenance" and rle_key[2] not in scores)):
                            add_issue(malformed, f"{rle_key[0]}: unregistered provenance key {rle_key[2]}")
                        else:
                            active_rle = rle_key
                            pair = []
                            rle_positions.setdefault(rle_key, 0)
                            if rle_key[1] == "stressFlag": flags.add(rle_key[0])
                            if rle_key[1] == "stressFlagModel": models.add(rle_key[0])
                            if rle_key[1] == "channelProvenance": channel_prov.add((rle_key[0], rle_key[2]))
                            if rle_key[1] == "scoreProvenance": score_prov.add((rle_key[0], rle_key[2]))
                elif active_rle is not None and prefix == f"fields.{active_rle[0]}.{active_rle[1]}.{active_rle[2] + '.' if active_rle[2] else ''}item.item" and event in {"string", "number", "null", "boolean"}:
                    pair.append(value)
                elif active_rle is not None and prefix.endswith(".item") and event == "end_array":
                    if len(pair) != 2:
                        add_issue(provenance_bad, f"{active_rle[0]}.{active_rle[1]}: malformed RLE pair")
                    else:
                        run_value, count = pair
                        farm_id, kind, name = active_rle
                        start = rle_positions[active_rle]
                        end = start + count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else axis_length + 1
                        if end > axis_length:
                            add_issue(provenance_bad, f"{farm_id}.{kind}.{name}: invalid RLE count")
                        else:
                            non_test = count - (test_prefixes[farm_id][end] - test_prefixes[farm_id][start])
                            if non_test:
                                if kind == "stressFlag":
                                    if run_value not in ALLOWED_FLAGS:
                                        add_issue(provenance_bad, f"{farm_id}: invalid stress flag")
                                    elif run_value == "suspected_stress":
                                        for index in range(start, end):
                                            if not masks[farm_id][index]:
                                                suspected_masks[farm_id][index] = 1
                                elif kind == "stressFlagModel":
                                    if run_value is not None and run_value not in set(SCORE_CHECKPOINTS.values()):
                                        add_issue(provenance_bad, f"{farm_id}: unregistered stress flag model")
                                    if run_value is None and sum(suspected_masks[farm_id][start:end]):
                                        add_issue(provenance_bad, f"{farm_id}: suspected-stress flag has no model provenance")
                                elif kind == "channelProvenance":
                                    if name not in channels or run_value not in ALLOWED_CHANNEL_PROVENANCE:
                                        add_issue(provenance_bad, f"{farm_id}: invalid channel provenance ({name})")
                                elif kind == "scoreProvenance":
                                    if name not in scores or run_value not in ALLOWED_SCORE_PROVENANCE:
                                        add_issue(provenance_bad, f"{farm_id}: invalid score provenance ({name})")
                                    elif (farm_id, name) in score_presence:
                                        present_count = sum(score_presence[(farm_id, name)][start:end])
                                        if (run_value == "absent" and present_count != 0) or (run_value != "absent" and present_count != non_test):
                                            add_issue(provenance_bad, f"{farm_id}: score/provenance disagreement ({name})")
                            rle_positions[active_rle] = end
                    pair = []
                elif active_rle is not None and prefix.startswith(f"fields.{active_rle[0]}.{active_rle[1]}") and event == "end_array" and prefix.count(".") == (3 if active_rle[2] else 2):
                    if rle_positions[active_rle] != axis_length:
                        add_issue(provenance_bad, f"{active_rle[0]}.{active_rle[1]}.{active_rle[2]}: RLE does not align with axis")
                    active_rle = None
                    pair = []
                if len(parts) == 3 and parts[0] == "fields" and parts[1] in farm_ids and parts[2] == "ladderRung" and event == "string":
                    if value not in {"L0", "L1", "L2", "L3", "L4"}:
                        add_issue(ladder_missing, parts[1])
                elif len(parts) == 3 and parts[0] == "fields" and parts[1] in farm_ids and parts[2] == "ladderRung" and event == "null":
                    add_issue(ladder_missing, parts[1])
    except STREAM_ERRORS as exc:
        add_issue(malformed, f"series values unreadable: {exc.__class__.__name__}")
    missing_fields = sorted(farm_ids - field_ids_seen)
    if missing_fields:
        add_issue(malformed, f"missing {len(missing_fields)} registered farm fields")
    for farm_id in field_ids_seen:
        missing_channels = [spec.name for spec in CHANNEL_REGISTRY if spec.required and (farm_id, spec.name) not in channel_arrays]
        missing_scores = [name for name in scores if (farm_id, name) not in score_arrays]
        if missing_channels or missing_scores:
            add_issue(malformed, f"{farm_id}: missing required arrays channels={missing_channels[:3]} scores={missing_scores[:3]}")
        if farm_id not in flags or farm_id not in models:
            add_issue(provenance_bad, f"{farm_id}: missing v5 stress flag/model arrays")
        if any((farm_id, spec.name) not in channel_prov for spec in CHANNEL_REGISTRY):
            add_issue(provenance_bad, f"{farm_id}: missing channel provenance")
        if any((farm_id, name) not in score_prov for name in scores):
            add_issue(provenance_bad, f"{farm_id}: missing score provenance")
    checks.append(_finding("series_structure", "FAIL" if malformed or arrays_bad else "PASS",
        f"Validated structure for {len(field_ids_seen)} farm fields; issues={len(malformed) + len(arrays_bad)}",
        bool(malformed or arrays_bad), examples=(malformed + arrays_bad)[:5]))
    checks.append(_finding("channel_ranges", "FAIL" if range_violations else "PASS",
        "Physical ranges checked against scripts/platform_channels.py; only non-test observations were evaluated",
        bool(range_violations), violations={name: examples for name, examples in range_violations.items()}))
    checks.append(_finding("provenance_and_flags", "FAIL" if provenance_bad else "PASS",
        f"Validated v5 split-safe provenance and stress flag fields; issues={len(provenance_bad)}",
        bool(provenance_bad), examples=provenance_bad[:5]))
    checks.append(_finding("ladder_rung", "WARNING" if ladder_missing else "PASS",
        "Every field has a recognized ladder rung" if not ladder_missing else f"Missing/unknown ladder rung for {len(set(ladder_missing))} field(s); sidecars are never deserialized",
        False, examples=ladder_missing[:5]))
    return checks, field_ids_seen


def validate_platform(config: InspectorConfig, records: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    farm_ids, _ = _read_farms(config, checks)
    root = next((item for item in config.roots if item.name == "platform_data"), None)
    series_path = root.path / "series_khuzestan.json" if root else None
    if series_path is None or not series_path.is_file():
        checks.append(_finding("series_payload", "WARNING", "series_khuzestan.json is absent; readiness is not established"))
    else:
        try:
            assert_within(series_path, root.path)
            if series_path.stat().st_size > config.max_series_bytes:
                checks.append(_finding("series_payload", "WARNING", "Series exceeds configured streaming byte limit"))
            elif not farm_ids:
                checks.append(_finding("series_payload", "WARNING", "Cannot validate series without a valid farm registry"))
            else:
                try:
                    import ijson  # noqa: F401
                except ImportError:
                    checks.append(_finding("series_payload", "WARNING", "Install optional ijson for full streaming readiness checks"))
                else:
                    version, axis, error = _axis_and_version(series_path)
                    checks.append(_finding("contract_version", "FAIL" if version != 5 else "PASS",
                        f"Expected Contract v5; found {version!r}", version != 5))
                    if error or axis is None:
                        checks.append(_finding("observation_dates", "FAIL", error or "Date axis unavailable", True))
                    else:
                        checks.append(_finding("observation_dates", "PASS", f"Validated {len(axis)} actual ISO calendar dates"))
                        if len(farm_ids) * len(axis) > MAX_FARM_OBSERVATIONS:
                            checks.append(_finding("series_payload", "WARNING", "Farm-by-observation validation exceeds configured memory safety bound"))
                        else:
                            masks, prefixes, split_issues = _read_test_masks(series_path, farm_ids, len(axis))
                            checks.append(_finding("split_tags", "FAIL" if split_issues else "PASS",
                                f"Split RLE checked before values; issues={len(split_issues)}", bool(split_issues), examples=split_issues[:5]))
                            if split_issues:
                                checks.append(_finding("series_values", "WARNING", "Value-level checks skipped because split membership could not be established"))
                            else:
                                value_checks, _ = _validate_series_values(series_path, len(axis), masks, prefixes, farm_ids)
                                checks.extend(value_checks)
                                checks.append(_checkpoint_alignment(series_path, config))
        except (OSError, PermissionError) as exc:
            checks.append(_finding("series_payload", "FAIL", f"Series path unavailable: {exc.__class__.__name__}", True))
    for record in records:
        if record.get("role") == "platform_export" and record.get("format") in {"json", "geojson"} and record.get("status") in {"ERROR", "INCOMPLETE"}:
            checks.append(_finding("artifact_readability", "WARNING", record.get("message") or "A platform artifact could not be inspected", artifact_id=record.get("id")))
    failures = [check for check in checks if check["status"] == "FAIL" and check.get("critical")]
    warnings = [check for check in checks if check["status"] == "WARNING" or (check["status"] == "FAIL" and not check.get("critical"))]
    evaluated = any(check["check"] == "contract_version" for check in checks)
    state = "BLOCKED" if failures else "WARNING" if evaluated and warnings else "PASS" if evaluated else "NOT_EVALUATED"
    return {"contract_version": 5, "status": state, "critical_failures": failures, "warnings": warnings,
            "candidate_payload_count": sum(1 for record in records if record.get("role") == "platform_export" and record.get("format") in {"json", "geojson"}),
            "checks": checks, "values_examined": "non-test observations only; locked-test values were skipped"}
