"""Persistence gate + cold-LST gate (change task 2.3) and event framing (2.4).

- **Persistence**: a flagged (farm, date) count as event evidence only when the
  farm has ≥ ``min_obs`` flagged observations ≥ ``min_gap_days`` apart within
  the event window (tasks.md 2.3: "≥2 obs, ≥5 d").
- **Cold-LST gate**: ΔLST = farm LST − cross-sectional median LST (over
  observed farms) at that date; ΔLST < ``cold_lst_thresh`` (−2 °C) ⇒ the flag
  is **rejected as probable cloud/shadow** — kept in the output as evidence,
  excluded from the flagged set (P0 precedent: "178 rejected (ΔLST < −2 °C,
  cloud/shadow)").
- **Threshold rule** (§7: threshold-dependent metrics need an explicit rule):
  the flag threshold is the per-method score quantile ``flag_q`` over TRAIN
  dates only — nothing about eval dates enters calibration.

Framing: outputs use **anomaly/flag/event wording only** — never "stress"
(no L4 ground truth; scores are L2 screening evidence).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def flag_threshold(scores: np.ndarray, train_mask: np.ndarray,
                   flag_q: float = 0.99) -> float:
    """Per-method threshold = flag_q-quantile of TRAIN-date scores."""
    tr = scores[:, train_mask]
    tr = tr[np.isfinite(tr)]
    if tr.size == 0:
        raise RuntimeError("No finite train-date scores for threshold calibration")
    return float(np.quantile(tr, flag_q))


def delta_lst(lst_c: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """(F, D) ΔLST vs cross-sectional median over observed farms."""
    x = np.where(obs_mask, lst_c, np.nan)
    med = np.nanmedian(x, axis=0)                    # (D,)
    return x - med[None, :]


def apply_gates(
    df: pd.DataFrame,
    score_cols: list[str],
    train_mask: np.ndarray,
    dates: list[str],
    cold_lst_thresh: float = -2.0,
    flag_q: float = 0.99,
    min_obs: int = 2,
    min_gap_days: int = 5,
    event_gap_days: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Adds per-method flag columns, gate_status, event_id; returns (df, info).

    gate_status ∈ {absent, below_threshold, rejected_cold_lst, non_persistent,
    flagged}. ``flagged`` requires: above threshold AND not cold-LST-rejected
    AND persistent (≥min_obs obs ≥min_gap_days apart within the event).
    """
    dates_ts = pd.to_datetime(pd.Series(dates))
    train_dates = set(np.asarray(dates)[train_mask])
    thresholds = {}
    for c in score_cols:
        tr_vals = df.loc[df["date"].isin(train_dates), c].to_numpy(dtype=np.float64)
        tr_vals = tr_vals[np.isfinite(tr_vals)]
        if tr_vals.size == 0:
            raise RuntimeError(f"No finite train-date scores for {c}")
        thresholds[c] = float(np.quantile(tr_vals, flag_q))

    flagged_any = np.zeros(len(df), dtype=bool)
    for c, thr in thresholds.items():
        flagged_any |= (df[c].to_numpy(dtype=np.float64) > thr)

    dlst = df["delta_lst_c"].to_numpy(dtype=np.float64)
    cold = np.isfinite(dlst) & (dlst < cold_lst_thresh)
    df["gate_flag_threshold_max"] = max(thresholds.values())
    df["above_threshold"] = flagged_any
    df["cold_lst_rejected"] = cold

    # persistence + events per farm over flagged (non-cold) dates — positional
    # indexing throughout (df must have a RangeIndex; asserted below)
    if not df.index.equals(pd.RangeIndex(len(df))):
        df = df.reset_index(drop=True)
    date_idx_of_row = df["date"].map({d: i for i, d in enumerate(dates)})
    if date_idx_of_row.isna().any():
        raise RuntimeError("gates: rows with dates outside the date list")
    date_idx_of_row = date_idx_of_row.astype(int)
    gate_status = np.full(len(df), "below_threshold", dtype=object)
    gate_status[df["absent"].to_numpy()] = "absent"
    gate_status[cold] = "rejected_cold_lst"
    event_id = np.full(len(df), -1, dtype=np.int64)
    next_event = 0
    for farm, grp in df.groupby("farm_id", sort=False):
        pos = grp.index.to_numpy()
        flags = [int(i) for i in pos if flagged_any[i] and not cold[i]]
        if not flags:
            continue
        # split into events by event_gap_days (gap measured on real dates)
        events = [[flags[0]]]
        for j in flags[1:]:
            gap = (dates_ts.iloc[date_idx_of_row[j]] -
                   dates_ts.iloc[date_idx_of_row[events[-1][-1]]]).days
            if gap > event_gap_days:
                events.append([j])
            else:
                events[-1].append(j)
        for e_pos in events:
            span_days = (dates_ts.iloc[date_idx_of_row[e_pos[-1]]] -
                         dates_ts.iloc[date_idx_of_row[e_pos[0]]]).days
            persistent = (len(e_pos) >= min_obs) and (span_days >= min_gap_days)
            eid = -1
            if persistent:
                eid = next_event
                next_event += 1
            for i in e_pos:
                gate_status[i] = "flagged" if persistent else "non_persistent"
                if persistent:
                    event_id[i] = eid
    df["gate_status"] = gate_status
    df["event_id"] = event_id
    info = {"thresholds": thresholds, "flag_q": flag_q,
            "cold_lst_thresh": cold_lst_thresh, "min_obs": min_obs,
            "min_gap_days": min_gap_days, "event_gap_days": event_gap_days,
            "n_events": next_event}
    return df, info


def build_events(df: pd.DataFrame, method_cols: list[str]) -> pd.DataFrame:
    """Event-level table over persistent flagged rows (tasks.md 2.4).

    Columns carry anomaly/flag wording only — no "stress" anywhere (enforced
    by tests/unit/test_probe.py::test_events_table_and_no_stress_wording).
    """
    ev = df[df["event_id"] >= 0].copy()
    if ev.empty:
        return pd.DataFrame(columns=[
            "event_id", "farm_id", "date_start", "date_end", "n_obs",
            "span_days", "max_combined_score", "methods_contributing"])
    rows = []
    for eid, grp in ev.groupby("event_id"):
        methods = [c for c in method_cols
                   if np.isfinite(grp[c].to_numpy(dtype=np.float64)).any()]
        rows.append({
            "event_id": int(eid),
            "farm_id": grp["farm_id"].iloc[0],
            "date_start": grp["date"].min(),
            "date_end": grp["date"].max(),
            "n_obs": int(len(grp)),
            "span_days": int((pd.to_datetime(grp["date"].max()) -
                              pd.to_datetime(grp["date"].min())).days),
            "max_combined_score": float(grp["combined_score"].max()),
            "methods_contributing": ",".join(m for m in methods),
        })
    return pd.DataFrame(rows)
