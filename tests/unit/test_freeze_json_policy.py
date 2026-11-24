"""The freeze path must emit strict JSON under the single NaN policy (audit 14)."""
from __future__ import annotations

import json
import math

from scripts.freeze_main_model import _jsonable


def test_jsonable_maps_non_finite_floats_to_null_at_every_depth():
    value = {
        "median_delay_days": float("nan"),
        "nested": {"upper": float("inf"), "lower": float("-inf"), "ok": 1.5},
        "list": [float("nan"), 2, None],
    }
    cleaned = _jsonable(value)
    assert cleaned == {
        "median_delay_days": None,
        "nested": {"upper": None, "lower": None, "ok": 1.5},
        "list": [None, 2, None],
    }
    # Default json.dumps would emit bare NaN/Infinity; this must not.
    text = json.dumps(cleaned, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)))


def test_jsonable_preserves_finite_values_and_keys():
    value = {"a": 0.0, "b": -3, "c": "NaN is text here", "d": True, "e": None}
    assert _jsonable(value) == value
    assert all(math.isfinite(v) for v in (value["a"], value["b"]))
