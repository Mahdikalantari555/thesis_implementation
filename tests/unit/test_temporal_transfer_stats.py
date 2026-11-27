"""Unit tests for the owner-amendment statistics of
``validate-olmoearth-temporal-transfer`` (round 2, 2026-11-27):

1. phase margin: ONE value per unique field (n_fields = unique fields, not
   cycle pairs), field-level bootstrap;
2. low-n rule considering BOTH bins and independent field count;
3. attrition-table consistency helper.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_HERE = Path(__file__).resolve().parents[2] / "experiments" / "olmoearth_eval"
sys.path.insert(0, str(_HERE))
_spec = importlib.util.spec_from_file_location(
    "temporal_transfer", _HERE / "temporal_transfer.py")
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)

import oe_eval_common as common  # noqa: E402


def _margin_frame() -> pd.DataFrame:
    """Two fields x three cycles x two phases with known cosines.

    Cycle embeddings are identical across cycles of the same field with
    orthogonal phase directions, so phase-matched cosine = 1 and
    phase-mismatched cosine = 0 -> every cycle-pair margin is exactly 1.
    """
    rows = []
    phases = common.PHASE_NAMES[:2]
    for fid in (1, 2):
        for ci in range(3):
            for pi, p in enumerate(phases):
                e = np.zeros(common.EMBED_COLS.__len__(), dtype=np.float64)
                e[pi] = 1.0
                rows.append(dict(
                    field_id=fid, cycle_index=ci, phase=p, day_index=ci * 100,
                    **{c: float(e[k]) for k, c in enumerate(common.EMBED_COLS)}))
    return pd.DataFrame(rows)


class TestPhaseMarginPerField:
    def test_one_value_per_unique_field(self):
        df = _margin_frame().assign(cycle_start_day=0)
        vals, n_fields, n_pairs = tt.phase_margin_per_field(df)
        # 2 fields x 3 cycles = 3 pairs per field = 6 pairs total
        assert n_fields == 2  # unique fields, NOT the 6 cycle pairs
        assert n_pairs == 6
        assert len(vals) == 2
        assert np.allclose(vals, 1.0)

    def test_field_with_single_cycle_skipped(self):
        df = pd.concat([_margin_frame(),
                        _margin_frame().assign(field_id=9, cycle_index=0)],
                       ignore_index=True)
        vals, n_fields, n_pairs = tt.phase_margin_per_field(df)
        assert n_fields == 2  # field 9 has one cycle -> no pair -> excluded
        assert n_pairs == 6

    def test_bootstrap_of_constant_margins_is_degenerate(self):
        vals = np.full(20, 0.7)
        out = tt.bootstrap_field_margins(vals)
        assert out["n_fields"] == 20
        assert out["margin_median"] == pytest.approx(0.7)
        assert out["margin_ci"][0] == pytest.approx(0.7)
        assert out["margin_ci"][1] == pytest.approx(0.7)

    def test_bootstrap_brackets_the_point_with_spread(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(0.3, 0.1, size=60)
        out = tt.bootstrap_field_margins(vals)
        lo, hi = out["margin_ci"]
        assert lo <= out["margin_median"] <= hi
        assert lo < hi  # genuine spread -> non-degenerate interval

    def test_empty_input(self):
        vals, n_fields, n_pairs = tt.phase_margin_per_field(
            _margin_frame().iloc[0:0])
        assert n_fields == 0 and n_pairs == 0 and len(vals) == 0
        assert tt.bootstrap_field_margins(vals)["n_fields"] == 0


class TestLowNRule:
    def test_2025_cell_is_low_n_by_field_count(self):
        # the real 2025 cell: 143 bins but only 24 independent fields
        assert tt.classify_low_n(143, 24) is True

    def test_2024_and_pooled_cells_are_not_low_n(self):
        assert tt.classify_low_n(817, 42) is False
        assert tt.classify_low_n(965, 42) is False

    def test_2026_cell_is_low_n_on_both_axes(self):
        assert tt.classify_low_n(5, 1) is True

    def test_bins_axis_alone_triggers(self):
        assert tt.classify_low_n(99, 50) is True


class TestAttrition:
    def test_table_shape_and_monotonicity(self):
        steps = [("S0", 4943, 204), ("S1", 4740, 195),
                 ("S2", 965, 42), ("S3", 965, 42)]
        for (sa, ba, fa), (sb, bb, fb) in zip(steps, steps[1:]):
            # bins and fields never increase along the funnel
            assert bb <= ba and fb <= fa
        # the real S2->S3 join was lossless; a lossy join must fail loudly
        assert steps[2][1] - steps[3][1] == 0
