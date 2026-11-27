"""Retention-flag handling in cycle-table consumers (owner-approved flags).

The owner approved keeping over-long / short bounded intervals as
``overlong_unresolved`` / ``short_unresolved`` evidence (design decisions 4
and 7 of cycle-detection-vi-repair). These tests pin the contract: unresolved
rows are excluded from training/stage assignment but are **surfaced** —
counted, logged and retained in the summary — never silently deleted, and a
pre-flag schema fails closed.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssl_anomaly.mini_lab.config import RunConfig, RepresentationConfig
from ssl_anomaly.mini_lab.data import _temporal_units, load_normal_cycles
from ssl_anomaly.phenology.mini import split_normal_cycles

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flagged_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"farm": "F", "cycle_id": 1, "start_date": "2014-01-01",
             "end_date": "2014-12-01", "length_days": 335, "status": "normal"},
            {"farm": "F", "cycle_id": 2, "start_date": "2018-08-20",
             "end_date": "2020-03-09", "length_days": 567,
             "status": "overlong_unresolved"},
            {"farm": "F", "cycle_id": 3, "start_date": "2021-01-01",
             "end_date": "2021-12-01", "length_days": 334, "status": "normal"},
            {"farm": "F", "cycle_id": 4, "start_date": "2019-02-12",
             "end_date": "2019-06-27", "length_days": 135,
             "status": "short_unresolved"},
        ]
    )


def test_split_normal_cycles_excludes_but_retains_unresolved_evidence():
    normal, summary = split_normal_cycles(_flagged_table(), source="test")
    assert list(normal["cycle_id"]) == [1, 3]
    assert summary["n_total"] == 4
    assert summary["n_normal"] == 2
    assert summary["n_unresolved"] == 2
    assert summary["unresolved_by_status"] == {
        "overlong_unresolved": 1,
        "short_unresolved": 1,
    }
    by_status = {r["status"]: r for r in summary["unresolved_rows"]}
    assert by_status["overlong_unresolved"]["start_date"] == "2018-08-20"
    assert by_status["overlong_unresolved"]["length_days"] == 567
    assert by_status["short_unresolved"]["end_date"] == "2019-06-27"
    # evidence is retained, not deleted: every unresolved row is in the summary
    assert len(summary["unresolved_rows"]) == 2


def test_split_normal_cycles_fails_closed_without_status_column():
    df = _flagged_table().drop(columns=["status"])
    with pytest.raises(ValueError, match="no 'status' column"):
        split_normal_cycles(df, source="test")


def test_split_normal_cycles_fails_closed_on_unknown_status():
    df = _flagged_table()
    df.loc[0, "status"] = "weird_status"
    with pytest.raises(ValueError, match="unknown cycle status"):
        split_normal_cycles(df, source="test")


def test_load_normal_cycles_logs_exclusion_once_per_table(caplog, tmp_path):
    table_path = tmp_path / "cycles_mini.parquet"
    _flagged_table().to_parquet(table_path, index=False)
    with caplog.at_level(logging.WARNING, logger="ssl_anomaly.mini_lab.data"):
        _, s1 = load_normal_cycles(table_path)
        _, s2 = load_normal_cycles(table_path)  # second call: no re-log
    assert s1["n_unresolved"] == 2 and s2["n_unresolved"] == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "the exclusion must be surfaced exactly once per table"
    assert "excluded as unresolved" in warnings[0].getMessage()
    # the surfaced message retains the evidence rows verbatim
    assert "overlong_unresolved" in warnings[0].getMessage()
    assert "2018-08-20" in warnings[0].getMessage()


def _seasonal_cfg(tmp_root: Path) -> RunConfig:
    return RunConfig(
        store_root=tmp_root,
        representation=RepresentationConfig(temporal="seasonal", spatial="full_field"),
    )


def test_temporal_units_seasonal_excludes_unresolved_intervals(tmp_path):
    cyc_dir = tmp_path / "phenology"
    cyc_dir.mkdir()
    _flagged_table().to_parquet(cyc_dir / "cycles_mini.parquet", index=False)
    cfg = _seasonal_cfg(tmp_path)
    # One continuous series spanning all four cycle intervals.
    dates = [
        "2014-03-01", "2014-06-01", "2014-09-01", "2014-11-15",      # cycle 1
        "2018-09-01", "2019-01-01", "2019-12-01",                    # overlong
        "2021-03-01", "2021-06-01", "2021-11-15",                    # cycle 3
        "2019-03-01", "2019-05-01",                                  # short
    ]
    values = np.zeros((len(dates), 2), dtype=float)
    observed = np.ones_like(values, dtype=bool)
    delta_t = np.ones(len(dates), dtype=float)
    units = _temporal_units(values, observed, dates, delta_t, cfg, store_root=tmp_path)
    # Only the two normal cycles survive; the overlong and short intervals are
    # excluded from training selection but remain in the logged evidence.
    assert len(units) == 2
    covered = {d for u in units for d in u[0]}
    assert "2018-09-01" not in covered and "2019-12-01" not in covered
    assert "2019-03-01" not in covered and "2019-05-01" not in covered
    assert "2014-03-01" in covered and "2021-11-15" in covered


def test_temporal_units_seasonal_fails_closed_on_unflagged_schema(tmp_path):
    cyc_dir = tmp_path / "phenology"
    cyc_dir.mkdir()
    _flagged_table().drop(columns=["status"]).to_parquet(
        cyc_dir / "cycles_mini.parquet", index=False
    )
    cfg = _seasonal_cfg(tmp_path)
    dates = ["2014-03-01", "2014-06-01", "2014-09-01", "2014-11-15"]
    with pytest.raises(ValueError, match="no 'status' column"):
        _temporal_units(
            np.zeros((len(dates), 2)), np.ones((len(dates), 2), dtype=bool),
            dates, np.ones(len(dates)), cfg, store_root=tmp_path,
        )


def test_stage_of_skips_unresolved_and_fails_closed():
    mod = _load_script("label_mini_suspected_stress")
    cycles = _flagged_table()
    # inside a normal cycle → staged; inside unresolved → None
    assert mod.stage_of("2014-06-01", cycles) == "S2"
    assert mod.stage_of("2019-03-01", cycles) is None  # short_unresolved span
    assert mod.stage_of("2019-01-01", cycles) is None  # overlong_unresolved span
    with pytest.raises(ValueError, match="no 'status' column"):
        mod.stage_of("2014-06-01", cycles.drop(columns=["status"]))
