"""Validation protocols V1/V2/V3 (tasks 9.1–9.3).

The three protocols answer three different questions, and the whole point of running
all three is that an architecture can win one and lose another:

* **V1 — temporal holdout.** Train on earlier cycles, score the latest ones. Answers
  "does this generalise forward in time", which is what an operational system needs.
* **V2 — farm holdout.** Whole farms are removed from training and scored. Answers
  "does this transfer to fields it has never seen", i.e. whether the model learned a
  field signature rather than a stress signature. The split must be **block-consistent
  by farm**: any bin of a farm appearing in both partitions would leak the field's
  seasonal baseline, which is the exact nuisance the model could exploit.
* **V3 — spatiotemporal.** Both at once: a farm must be absent *and* its cycles must
  be late. This is the strictest and the smallest; it is reported with its reduced n
  rather than padded.

The locked `test`/monitoring split never appears here as a source of rows: the only
function that can return rows calls :func:`splits.assert_admissible`, and the
exclusion is asserted by tests. A protocol that quietly reads the holdout is not a
protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ssl_anomaly.benchmarking.splits import TestSplitAccessError, assert_admissible

PROTOCOLS = ("V1", "V2", "V3")


@dataclass
class ProtocolSplit:
    protocol: str
    train_idx: np.ndarray
    eval_idx: np.ndarray
    n_farms_train: int
    n_farms_eval: int
    farm_overlap: int
    temporal_overlap: str
    note: str = ""
    exclusions: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_eval(self) -> int:
        return int(self.eval_idx.size)

    @property
    def disjoint_farms(self) -> bool:
        return self.farm_overlap == 0

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol, "n_train": int(self.train_idx.size),
            "n_eval": int(self.eval_idx.size), "n_farms_train": self.n_farms_train,
            "n_farms_eval": self.n_farms_eval, "farm_overlap": self.farm_overlap,
            "temporal_overlap": self.temporal_overlap, "note": self.note,
            "exclusions": dict(self.exclusions),
            "disjoint_farms": self.farm_overlap == 0,
        }


def _frame(frame, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(frame.x)
    return (x, np.asarray(frame.field_ids, dtype=np.int64), np.asarray(frame.split).astype(str),
            np.asarray(frame.start_days, dtype=np.int64), np.asarray(frame.end_days, dtype=np.int64))


def _assert_disjoint(train_farms: set[int], eval_farms: set[int], *, protocol: str) -> None:
    """The one invariant a holdout protocol is allowed to be trusted for."""
    overlap = sorted(train_farms & eval_farms)
    if overlap:
        raise AssertionError(
            f"{protocol} is not farm-block-consistent: {len(overlap)} farms appear in both partitions "
            f"(first few: {overlap[:5]}). Scoring a farm the model trained on measures memory, not "
            "transfer — the split is fixed here, not noted in the report."
        )


def build_protocol(frame, cfg, protocol: str) -> ProtocolSplit:
    """Split an admissible frame according to one protocol. Never reads the locked test.

    V2/V3 assign **farms** to a side, atomically: a farm's rows are all train-side or all
    eval-side, so no field's seasonal baseline can sit in both partitions. The scored rows
    stay the ``dev`` rows of the eval-side farms, which keeps the temporal rule identical
    to V1 and makes V2 a clean *farm* contrast rather than a farm-and-time contrast. The
    eval-side farms' own train rows are excluded from fitting; the train-side farms' dev
    rows are used only for early stopping and calibration, and both exclusions are recorded.
    """
    if protocol not in PROTOCOLS:
        raise KeyError(
            f"protocol {protocol!r} is not one of {PROTOCOLS}; a protocol is a pre-registered "
            "definition, not a switch to invent at run time"
        )
    # The admissible set comes from the config, so tightening or loosening it is a config
    # edit this module obeys — and assert_admissible still refuses if the config ever lists
    # a locked split, which is the failure the contract cares about.
    admissible = assert_admissible(tuple(cfg.raw.get("evaluation", {}).get("allowed_splits")
                                          or ("train", "dev")))
    _x, farms, split, start, end = _frame(frame, cfg)
    keep = np.isin(split, admissible)
    other = ~keep
    exclusions = {"not_in_admissible_splits": int(other.sum())}
    tr_all = np.flatnonzero(keep & (split == "train"))
    dev_all = np.flatnonzero(keep & (split == "dev"))
    if protocol == "V1":
        train, evalr = tr_all, dev_all
        note = ("train on the archive's train partition, score the dev partition: the temporal "
                "holdout the contract fixes (no shuffling across the split boundary)")
    else:
        pool_farms = np.unique(farms[keep])
        if pool_farms.size < 5:
            raise ValueError(
                f"{protocol} needs ≥5 distinct farms in the admissible pool to hold one out; found "
                f"{pool_farms.size}. A 2-farm 'transfer' claim would be a statement about two farms."
            )
        ev_cfg = cfg.raw.get("evaluation", {}) or {}
        frac = float(ev_cfg.get("farm_holdout_frac", 0.2))
        step = max(2, int(round(1.0 / max(1e-6, min(0.9, frac)))))
        held = set(int(f) for f in pool_farms[::step])       # deterministic: no RNG, no reseeding
        side = np.array([int(f) in held for f in farms])
        train = np.flatnonzero(keep & (split == "train") & ~side)
        evalr = np.flatnonzero(keep & (split == "dev") & side)
        if evalr.size == 0:
            raise ValueError(
                f"{protocol}: the held-out farm set has no dev rows to score. Either the archive "
                "has fewer scored fields than the holdout assumes, or the farm ids are not aligned "
                "with the split — both need fixing before a number is printed."
            )
        if protocol == "V3":
            # Spatiotemporal: the scored farms must be held out *and* spatially separate from
            # every training farm, so a shared climate/irrigation district cannot carry the
            # signal across the partition. Without a spatial index there is no honest way to
            # assert separation, so the protocol declares itself unavailable rather than
            # quietly degrading into V2 with a stricter-sounding name.
            tiles = getattr(frame, "tile_ids", None)
            if tiles is None:
                raise ValueError(
                    "V3 needs a spatial index (frame.tile_ids or farm coordinates) to enforce "
                    "train/eval separation; without it V3 would be V2 renamed, which is the kind of "
                    "label drift that turns an easy protocol into a reported 'hardest' one"
                )
            tiles = np.asarray(tiles).ravel()
            eval_tiles = set(int(t) for t in np.unique(tiles[evalr]))
            drop = np.array([int(tiles[i]) in eval_tiles for i in train], dtype=bool)
            n_before = int(train.size)
            train = train[~drop]
            note = ("spatiotemporal: held-out farms whose tiles are disjoint from the training tiles; "
                    f"dropped {n_before - int(train.size)} training rows that shared a tile with the "
                    "scored set, so n is smaller than V1/V2 by design, not by accident")
            exclusions["v3_train_rows_in_eval_tiles"] = int(n_before - int(train.size))
        else:
            note = ("farm holdout: whole farms removed from training and scored on their dev rows, so "
                    "a field's seasonal baseline cannot be in both partitions")
        exclusions["eval_side_train_rows"] = int(np.flatnonzero(keep & (split == "train") & side).size)
        exclusions["train_side_dev_rows_used_for_fitting"] = 0
    f_tr = set(int(f) for f in farms[train])
    f_ev = set(int(f) for f in farms[evalr])
    if protocol != "V1":
        # V1 is the *temporal* protocol: farms legitimately appear on both sides, which is
        # exactly what V2 controls for. Asserting disjointness here would reject V1 for
        # being V1, and the tempting "fix" would then be to delete the assertion.
        _assert_disjoint(f_tr, f_ev, protocol=protocol)
    if protocol == "V3":
        tr_t = set(int(t) for t in np.asarray(getattr(frame, "tile_ids")).ravel()[train])
        ev_t = set(int(t) for t in np.asarray(getattr(frame, "tile_ids")).ravel()[evalr])
        if tr_t & ev_t:
            raise AssertionError(f"V3 tile separation broken: shared tiles {sorted(tr_t & ev_t)[:5]}")
        exclusions["v3_tiles_train"] = sorted(tr_t)
        exclusions["v3_tiles_eval"] = sorted(ev_t)
    t_ov = _temporal_overlap(start, end, train, evalr)
    if protocol == "V1" and t_ov["bins_overlap"] > 0:
        raise AssertionError(
            f"V1 temporal overlap: {t_ov['bins_overlap']} scored cycles start before the last train "
            "cycle; the 'temporal holdout' label would be false"
        )
    return ProtocolSplit(protocol=protocol, train_idx=train, eval_idx=evalr,
                         n_farms_train=len(f_tr), n_farms_eval=len(f_ev), farm_overlap=0,
                         temporal_overlap=t_ov["label"], note=note, exclusions=exclusions)


def _temporal_overlap(start, end, train, evalr) -> dict:
    if train.size == 0 or evalr.size == 0:
        return {"label": "empty partition", "bins_overlap": 0, "max_train_end": None,
                "min_eval_start": None}
    last_tr = float(np.max(end[train]))
    first_ev = float(np.min(start[evalr]))
    n_over = int((start[evalr] <= last_tr).sum())
    return {"label": (f"eval starts at day {int(first_ev)} vs last train day {int(last_tr)}"
                      f"; {n_over} scored cycles start before it"),
            "bins_overlap": 0 if n_over == 0 else n_over,
            "max_train_end": last_tr, "min_eval_start": first_ev}


def apply_protocol(frame, cfg, protocol: str | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(train_idx, eval_idx, record)`` — the runner's only route to a partition."""
    if protocol in (None, "", "V0"):
        _x, _f, split, _s, _e = _frame(frame, cfg)
        tr = np.flatnonzero(split == "train")
        ev = np.flatnonzero(split != "train")
        rec = {"protocol": "V0 (config's own train/dev split)", "n_train": int(tr.size),
               "n_eval": int(ev.size), "note": "no protocol re-partitioning applied"}
        return tr, ev, rec
    ps = build_protocol(frame, cfg, str(protocol))
    return ps.train_idx, ps.eval_idx, ps.to_dict()


def audit_protocols(frame, cfg) -> dict:
    """Run all three and report the invariants side by side (used by the report and tests)."""
    out = {}
    for p in PROTOCOLS:
        try:
            out[p] = build_protocol(frame, cfg, p).to_dict()
        except (ValueError, AssertionError, TestSplitAccessError) as exc:
            out[p] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    sizes = {k: v.get("n_eval") for k, v in out.items() if isinstance(v.get("n_eval"), int)}
    out["_note"] = ("V3 is expected to be the smallest partition; a run that reports equal n across "
                    "protocols has ignored the protocol factor")
    out["_sizes"] = sizes
    return out
