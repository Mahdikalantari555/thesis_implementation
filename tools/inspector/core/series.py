"""On-demand, split-safe streaming access to a single Platform farm series."""
from __future__ import annotations

import re
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Any

from .config import InspectorConfig
from .security import assert_within

FARM_ID = re.compile(r"^F-\d{3,}$")
SCALAR_EVENTS = {"string", "number", "null", "boolean"}
ALLOWED_SPLITS = {"train", "dev", "test", "monitoring"}


def _load_registry(repo_root: Path) -> tuple[dict[str, str], list[str]]:
    registry_path = repo_root / "scripts" / "platform_channels.py"
    if not registry_path.is_file():
        return {}, []
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("inspector_series_channels", registry_path)
        if spec is None or spec.loader is None:
            return {}, []
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        channels = {entry.name: entry.unit for entry in module.CHANNEL_REGISTRY}
        scores = [entry.name for entry in module.SCORE_REGISTRY]
        return channels, scores
    except Exception:
        return {}, []


def _expand_runs(runs: list[Any], expected: int) -> list[Any] | None:
    expanded: list[Any] = []
    for run in runs:
        if not isinstance(run, (tuple, list)) or len(run) != 2:
            return None
        value, count = run
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or len(expanded) + count > expected:
            return None
        expanded.extend([value] * count)
    return expanded if len(expanded) == expected else None


def _series_path(config: InspectorConfig) -> tuple[Path | None, str | None]:
    root = next((item for item in config.roots if item.name == "platform_data"), None)
    if root is None:
        return None, "No platform_data root is configured"
    path = root.path / "series_khuzestan.json"
    try:
        assert_within(path, root.path)
        if not path.is_file():
            return None, "series_khuzestan.json is absent"
        if path.stat().st_size > config.max_series_bytes:
            return None, "Series file exceeds configured streaming limit"
        return path, None
    except (OSError, PermissionError) as exc:
        return None, f"Series file is unavailable: {exc.__class__.__name__}"


def farm_series(config: InspectorConfig, farm_id: str) -> dict[str, Any]:
    """Read one farm's non-test timeline without materializing the archive.

    The split RLE is read first. The second streaming pass discards each test
    observation value as it arrives and never stores it in the response.
    """
    if not FARM_ID.fullmatch(farm_id):
        return {"available": False, "status": "INVALID_ID", "reason": "Invalid farm ID", "observations": []}
    path, error = _series_path(config)
    if path is None:
        return {"available": False, "status": "UNAVAILABLE", "reason": error, "observations": []}
    try:
        import ijson
    except ImportError:
        return {"available": False, "status": "UNSUPPORTED", "reason": "Install optional ijson for streaming access to the large series payload", "observations": []}

    before = path.stat()
    try:
        with path.open("rb") as stream:
            split_runs = list(ijson.items(stream, f"fields.{farm_id}.split.item"))
        # The v4 payload has no per-observation split tags. Reading its values
        # would make locked-test separation impossible, so fail closed.
        if not split_runs:
            return {"available": False, "status": "RESTRICTED/UNKNOWN_SPLIT", "reason": "Series has no split tags; value-level inspection is disabled", "observations": []}
        # Read the shared date axis in one bounded streaming pass. Max axis size
        # is a safety bound, not an assumed archive length.
        with path.open("rb") as stream:
            axis = list(ijson.items(stream, "axis.item"))
        valid_axis = bool(axis) and len(axis) <= 20_000 and all(
            isinstance(day, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) is not None
            for day in axis
        )
        if valid_axis:
            try:
                for day in axis:
                    date_type.fromisoformat(day)
            except ValueError:
                valid_axis = False
        if not valid_axis:
            return {"available": False, "status": "INCOMPLETE", "reason": "Series date axis is absent or malformed", "observations": []}
        splits = _expand_runs(split_runs, len(axis))
        if splits is None or any(split not in ALLOWED_SPLITS for split in splits):
            return {"available": False, "status": "RESTRICTED/UNKNOWN_SPLIT", "reason": "Series split tags do not align with the date axis", "observations": []}

        channel_units, score_names = _load_registry(config.repo_root)
        index_channels = [name for name, unit in channel_units.items() if unit == "index"]
        if not index_channels:
            return {"available": False, "status": "WARNING", "reason": "Project channel registry is unavailable", "observations": []}

        # Prepare event paths only for registry-backed index channels, registered
        # scores and provenance/flag metadata. Array events tagged test are
        # counted but not retained.
        data_paths: dict[str, tuple[str, str]] = {}
        for channel in index_channels:
            data_paths[f"fields.{farm_id}.channels.{channel}.item"] = ("channels", channel)
        for score in score_names:
            data_paths[f"fields.{farm_id}.scores.{score}.item"] = ("scores", score)
        rle_targets: dict[str, tuple[str, str]] = {}
        for channel in channel_units:
            rle_targets[f"fields.{farm_id}.channelProvenance.{channel}"] = ("channel_provenance", channel)
        for score in score_names:
            rle_targets[f"fields.{farm_id}.scoreProvenance.{score}"] = ("score_provenance", score)
        for prop in ("stressFlag", "stressFlagModel"):
            rle_targets[f"fields.{farm_id}.{prop}"] = (prop, "")

        per_index: dict[int, dict[str, dict[str, Any]]] = {}
        array_positions: dict[str, int] = {prefix: 0 for prefix in data_paths}
        # Keep RLE metadata only at non-test indices. A run value covering only
        # test indices is discarded immediately after its pair is decoded.
        rle_values: dict[tuple[str, str], dict[int, Any]] = {target: {} for target in rle_targets.values()}
        rle_positions: dict[str, int] = {prefix: 0 for prefix in rle_targets}
        rle_seen: set[str] = set()
        active_rle: str | None = None
        active_pair: list[Any] = []
        contract_version: int | None = None
        ladder_rung: str | None = None
        with path.open("rb") as stream:
            for prefix, event, value in ijson.parse(stream):
                if prefix == "contractVersion" and event == "number":
                    contract_version = int(value)
                elif prefix == f"fields.{farm_id}.ladderRung" and event in {"string", "null"}:
                    ladder_rung = value
                elif prefix in data_paths and event in {"number", "null"}:
                    index = array_positions[prefix]
                    array_positions[prefix] += 1
                    if index < len(splits) and splits[index] != "test":
                        group, name = data_paths[prefix]
                        per_index.setdefault(index, {}).setdefault(group, {})[name] = value
                elif prefix in rle_targets and event == "start_array":
                    active_rle = prefix
                    active_pair = []
                elif active_rle is not None and prefix == active_rle + ".item.item" and event in SCALAR_EVENTS:
                    active_pair.append(value)
                elif active_rle is not None and prefix == active_rle + ".item" and event == "end_array":
                    if len(active_pair) != 2:
                        return {"available": False, "status": "INCOMPLETE", "reason": "Malformed provenance/flag RLE pair", "observations": []}
                    run_value, count = active_pair
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        return {"available": False, "status": "INCOMPLETE", "reason": "Malformed provenance/flag RLE count", "observations": []}
                    position = rle_positions[active_rle]
                    if position + count > len(axis):
                        return {"available": False, "status": "INCOMPLETE", "reason": "Provenance/flag RLE exceeds date axis", "observations": []}
                    target = rle_targets[active_rle]
                    for index in range(position, position + count):
                        if splits[index] != "test":
                            rle_values[target][index] = run_value
                    rle_positions[active_rle] = position + count
                    rle_seen.add(active_rle)
                    active_pair = []
                elif active_rle is not None and prefix == active_rle and event == "end_array":
                    active_rle = None
                    active_pair = []

        if any(position != len(axis) for position in array_positions.values()):
            return {"available": False, "status": "INCOMPLETE", "reason": "One or more channel/score arrays do not align with the date axis", "observations": []}
        if any(rle_positions[prefix] != len(axis) for prefix in rle_seen):
            return {"available": False, "status": "INCOMPLETE", "reason": "Provenance/flag RLE does not align with the date axis", "observations": []}
        flags = rle_values.get(("stressFlag", ""), {})
        flag_models = rle_values.get(("stressFlagModel", ""), {})
        observations = []
        excluded_test = 0
        for index, (date, split) in enumerate(zip(axis, splits, strict=True)):
            if split == "test":
                excluded_test += 1
                continue
            item = per_index.get(index, {})
            channels = item.get("channels", {})
            provenance = {name: values[index] for (kind, name), values in rle_values.items() if kind == "channel_provenance" and index in values}
            scores = item.get("scores", {})
            score_provenance = {name: values[index] for (kind, name), values in rle_values.items() if kind == "score_provenance" and index in values}
            observations.append({
                "date": date,
                "split": split,
                "channels": channels,
                "channel_provenance": provenance,
                "scores": scores,
                "score_provenance": score_provenance,
                "stress_flag": flags.get(index) if contract_version == 5 else None,
                "stress_flag_model": flag_models.get(index) if contract_version == 5 else None,
                "ladder_rung": ladder_rung,
            })
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return {"available": False, "status": "INCOMPLETE", "reason": "Series changed during streaming read", "observations": []}
        flags_available = contract_version == 5 and bool(flags) and any(value == "suspected_stress" for value in flags.values())
        marker_status = ("unavailable_v4" if contract_version is not None and contract_version < 5
                         else "available" if flags_available
                         else "no_suspected_flags" if flags
                         else "unavailable_missing_flags")
        return {"available": True, "status": "OK", "farm_id": farm_id, "contract_version": contract_version,
                "index_channels": index_channels, "observations": observations,
                "axis_count": len(axis), "excluded_test_observation_count": excluded_test,
                "suspected_stress_available": flags_available, "stress_marker_status": marker_status,
                "flag_note": "Markers reflect recorded v5 model flags only; they are not verified stress labels."}
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        return {"available": False, "status": "INCOMPLETE", "reason": f"Series could not be read safely: {exc.__class__.__name__}", "observations": []}
