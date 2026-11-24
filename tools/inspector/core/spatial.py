"""Read bounded farm geometry already present in configured platform data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import InspectorConfig
from .security import assert_within


def farm_map(config: InspectorConfig) -> dict[str, Any]:
    root = next((item for item in config.roots if item.name == "platform_data"), None)
    if root is None:
        return {"available": False, "reason": "No platform_data root is configured", "farms": []}
    path = root.path / "farms_khuzestan.json"
    try:
        assert_within(path, root.path)
        if not path.is_file():
            return {"available": False, "reason": "Farm registry artifact is absent", "farms": []}
        if path.stat().st_size > config.max_json_bytes:
            return {"available": False, "reason": "Farm registry exceeds configured JSON read limit", "farms": []}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return {"available": False, "reason": "Farm registry has an unsupported structure", "farms": []}
        farms = []
        for item in data[:config.max_sample_size]:
            if not isinstance(item, dict):
                continue
            centroid = item.get("centroid")
            polygon = item.get("polygon")
            if not isinstance(centroid, list) or len(centroid) < 2:
                continue
            try:
                lat, lon = float(centroid[0]), float(centroid[1])
            except (TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            points = []
            if isinstance(polygon, list):
                for point in polygon:
                    if isinstance(point, list) and len(point) >= 2:
                        try:
                            points.append([float(point[0]), float(point[1])])
                        except (TypeError, ValueError):
                            continue
            farms.append({"id": str(item.get("id", "")), "code": str(item.get("code", "")),
                          "block": str(item.get("block", "")), "centroid": [lat, lon], "polygon": points})
        return {"available": True, "source": "farms_khuzestan.json", "contract_version": 5,
                "farm_count": len(farms), "is_sampled": len(data) > len(farms), "farms": farms}
    except (OSError, UnicodeError, json.JSONDecodeError, PermissionError) as exc:
        return {"available": False, "reason": f"Farm registry unavailable: {exc.__class__.__name__}", "farms": []}
