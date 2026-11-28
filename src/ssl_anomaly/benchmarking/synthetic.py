"""Synthetic anomaly injection: evaluation panel + S3 train contamination (tasks 11.x).

Five families × three severities at one pre-registered prevalence, per
``benchmark_config.yaml: synthetic``. Two properties decide whether this module is
usable as a benchmark, and both are implemented rather than asserted:

**Severity is defined in robust SDs of the channel's own clean distribution**
(``severity_scale_source: per-channel_MAD_of_clean_dev_cycles``). A "strong" event
must mean something on LST as well as on NDVI; a fixed absolute delta would silently
turn the severity axis into a per-channel noise comparison.

**Trivial separability is audited and the offending cells are excluded, with their
IDs listed.** The contract's rule is ``flag_if: ">= 0.98 of injected |delta| exceeds
4 × the channel MAD"``: an event that a one-line threshold already solves measures
the generator, not the architecture. Excluding it is the pre-registered action
(``action: flag_and_exclude``), and the exclusion list travels into the manifest so
a reader can ask "which cells were dropped, and would keeping them change the
ranking?" — the honest answer being that it can be recomputed from the recorded IDs.

The train contamination stream (S3's fuel) uses a **different seed and a disjoint
cycle pool** from the evaluation panel; same pool + different order would still share
the generator's structure and let the eval signature leak into training.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

#: The contract's five families. Each names a *mechanism*, not a shape: the shape and
#: the channel response together define what is injected.
FAMILY_MECHANISMS: tuple[str, ...] = (
    "spectral", "temporal", "phenological", "multivariate", "persistent",
)

#: channel responses. Signs follow the design's qualitative rules (a vegetation
#: decline lowers NDVI/NIR, raises red/SWIR/LST); the magnitudes are relative
#: weights that the severity multiplier scales.
FAMILY_CHANNELS: dict[str, dict[str, float]] = {
    "spectral": {"ndvi": -1.0, "nir": -0.7, "red": 0.7, "swir1": 0.5},
    "temporal": {"ndvi": -0.8, "lst": 0.9, "green": -0.4},
    "phenological": {"ndvi": -0.6, "green": -0.8, "nir": -0.4, "red": 0.4},
    "multivariate": {"ndvi": -0.9, "swir2": 0.8, "lst": 0.6, "blue": 0.3},
    "persistent": {"ndvi": -1.0, "nir": -0.9, "red": 0.8, "swir1": 0.7, "lst": 0.5},
}

#: relative event length per family, in bins: a phenological shift needs several
#: bins to be visible at all, a spectral step can be a single bin.
FAMILY_LENGTH: dict[str, tuple[int, int]] = {
    "spectral": (1, 3), "temporal": (1, 2), "phenological": (4, 8),
    "multivariate": (2, 5), "persistent": (6, 12),
}


def _shape_step(T, onset, length, rng):
    out = np.zeros(T)
    out[onset : onset + length] = 1.0
    return out


def _shape_spike(T, onset, length, rng):
    out = np.zeros(T)
    for k in range(length):
        i = min(T - 1, onset + k)
        out[i] = 1.0 if k == 0 else float(rng.uniform(0.4, 0.9))
    return out


def _shape_ramp(T, onset, length, rng):
    out = np.zeros(T)
    end = min(T, onset + length)
    n = max(1, end - onset)
    out[onset:end] = np.linspace(0.0, 1.0, n)
    return out


def _shape_shift(T, onset, length, rng):
    """A trajectory displaced in phase: the phenological signature (rise delayed)."""
    out = np.zeros(T)
    end = min(T, onset + length)
    n = max(1, end - onset)
    t = np.arange(n) / max(1, T)
    out[onset:end] = np.sin(2 * np.pi * t * 2.0) * np.linspace(1.0, 0.3, n)
    return out


def _shape_plateau(T, onset, length, rng):
    """Step down, stay down, partial recovery — the persistent-stress shape."""
    out = np.zeros(T)
    end = min(T, onset + length)
    n = max(1, end - onset)
    prof = np.ones(n)
    if n >= 4:
        prof[int(0.7 * n) :] = 0.4
    out[onset:end] = prof
    return out


#: family → shape function (the mechanism made concrete).
FAMILY_SHAPES: dict[str, Callable[..., np.ndarray]] = {
    "spectral": _shape_step,
    "temporal": _shape_spike,
    "phenological": _shape_shift,
    "multivariate": _shape_ramp,
    "persistent": _shape_plateau,
}


@dataclass
class InjectionCell:
    """One (family, severity) cell: where events were placed and how they audited.

    ``family`` starting with ``combined:`` marks a *designed combined scenario* (task
    4.9): the cell carries several families at once, ``scenario_families`` names them in
    the contract's order, and ``components`` records each family's own (onset, length,
    channels) per event. The headline macro stays over the single-family 5 × 3 grid —
    the contract pre-registers ``headline_grid: "5 families × 3 severities"`` — while
    combined cells are scored per-cell and reported separately.
    """

    family: str
    severity: str
    magnitude: float                       # severity multiplier from the contract
    cycle_index: np.ndarray
    onset_index: np.ndarray
    labels: np.ndarray                     # (N, T) bool
    delta: np.ndarray                      # (N, T) signed NDVI-scale delta applied
    channels: tuple[str, ...] = ()
    n_events: int = 0
    n_affected_bins: int = 0
    mean_length_bins: float = float("nan")
    audit: dict = field(default_factory=dict)
    scenario: str | None = None
    scenario_families: tuple[str, ...] = ()
    components: list[dict] = field(default_factory=list)

    @property
    def cell(self) -> str:
        return f"{self.family}__{self.severity}"

    @property
    def is_combined(self) -> bool:
        return self.family.startswith("combined:")

    def component_events(self) -> list[dict]:
        """Ground truth per family for every combined event (task 4.10 metadata)."""
        return [dict(c) for c in self.components]


@dataclass
class SyntheticPanel:
    """Injected tensors + per-cell labels + the exclusion record."""

    x: np.ndarray
    m: np.ndarray
    cells: list[InjectionCell]
    labels_bin: np.ndarray
    labels_cycle: np.ndarray
    prevalence_target: float
    seed: int
    pool: str
    channel_names: tuple[str, ...]
    generator_sha256: str
    clean_sha256: str
    audit_summary: dict = field(default_factory=dict)
    excluded_cells: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def usable_cells(self) -> list[str]:
        drop = {e["cell"] for e in self.excluded_cells}
        return [c.cell for c in self.cells if c.cell not in drop]

    def cell(self, name: str) -> InjectionCell:
        for c in self.cells:
            if c.cell == name:
                return c
        raise KeyError(f"no panel cell {name!r}; cells are {self.usable_cells}")

    def to_dict(self) -> dict:
        return {
            "n_cycles": int(self.x.shape[0]),
            "n_bins": int(self.x.shape[1]),
            "n_channels": int(self.x.shape[2]),
            "channel_names": list(self.channel_names),
            "prevalence_target": self.prevalence_target,
            "n_prevalence_bin": round(float(self.labels_bin.mean()), 5),
            "n_prevalence_cycle": round(float(self.labels_cycle.mean()), 5),
            "seed": self.seed,
            "pool": self.pool,
            "generator_sha256": self.generator_sha256,
            "clean_sha256": self.clean_sha256,
            "n_events_total": int(sum(c.n_events for c in self.cells)),
            "cells": [
                {
                    "cell": c.cell,
                    "magnitude": c.magnitude,
                    "channels": list(c.channels),
                    "n_events": c.n_events,
                    "n_affected_bins": c.n_affected_bins,
                    "mean_length_bins": round(float(c.mean_length_bins), 2),
                    "audit": c.audit,
                    **({"scenario": c.scenario, "scenario_families": list(c.scenario_families),
                        "n_component_events": len(c.components)}
                       if c.is_combined else {}),
                }
                for c in self.cells
            ],
            "excluded_cells": list(self.excluded_cells),
            "audit_summary": dict(self.audit_summary),
            "notes": list(self.notes),
        }


def _tensor_sha(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.dtype.str.encode())
    h.update(a.view(np.uint8).reshape(-1)[: 1 << 22].tobytes())   # bounded: a fingerprint, not a copy
    return h.hexdigest()


def event_records(panel: SyntheticPanel) -> list[dict]:
    """The ``sha256_of_(field_id,cycle_id,start_bin,family,severity)`` isolation check.

    The contract names this check for proving the train and eval generators did not
    draw the same events; it is computed on the *event identity* rather than on the
    tensors, so two panels over the same cycles with different families are
    correctly seen as different.
    """
    out: list[dict] = []
    for c in panel.cells:
        for k, cy in enumerate(c.cycle_index):
            out.append(
                {
                    "cell": c.cell,
                    "family": c.family,
                    "severity": c.severity,
                    "cycle_index": int(cy),
                    "start_bin": int(c.onset_index[k]) if k < len(c.onset_index) else None,
                    "identity_sha256": hashlib.sha256(
                        f"{panel.pool}|{int(cy)}|{int(c.onset_index[k]) if k < len(c.onset_index) else -1}"
                        f"|{c.family}|{c.severity}".encode()
                    ).hexdigest(),
                }
            )
    return out


def _draw_onsets(cycles: np.ndarray, n_bins: int, *, family: str, rng: np.random.Generator,
                 observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Onsets + labels for a FIXED set of cycles.

    ``labels`` is enumeration-ordered — row ``i`` belongs to ``cycles[i]`` — and callers
    must never index it by cycle id. Onsets are restricted to windows that contain at
    least one observed bin for the family's channels. A cloud-covered event is invisible
    in both the input and the score, so including it would add a guaranteed miss for
    every architecture and compress the differences the benchmark exists to measure.
    """
    lo, hi = FAMILY_LENGTH[family]
    onsets = np.zeros(len(cycles), dtype=np.int64)
    labels = np.zeros((len(cycles), n_bins), dtype=bool)
    for i, ci in enumerate(cycles):
        length = int(rng.integers(lo, hi + 1))
        cand = np.arange(0, max(1, n_bins - length + 1))
        ch_ok = observed[int(ci)]
        if ch_ok.any():
            good = np.array(
                [bool(ch_ok[j : min(n_bins, j + length)].any()) for j in cand], dtype=bool
            )
            if good.any():
                cand = cand[good]
        onset = int(rng.choice(cand))
        onsets[i] = onset
        labels[i, onset : onset + length] = True
    return onsets, labels


def plan_events(
    n_cycles: int, n_bins: int, *, family: str, events: int, prevalence: float,
    rng: np.random.Generator, observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick event cycles and onsets; returns ``(cycles, onsets, labels)``.

    ``labels`` row ``i`` belongs to ``cycles[i]`` (enumeration order, NOT cycle id).
    """
    n_ev = max(1, min(int(events), int(round(prevalence * n_cycles * 1.0)), n_cycles))
    cyc = rng.choice(n_cycles, size=n_ev, replace=False)
    onsets, labels = _draw_onsets(cyc, n_bins, family=family, rng=rng, observed=observed)
    return cyc.astype(np.int64), onsets, labels


def apply_panel(
    clean, cfg, *, seed: int, pool: str, role: str = "evaluation",
    cycle_subset: np.ndarray | None = None,
) -> SyntheticPanel:
    """Inject the contract's full family × severity grid into a clean tensor set.

    ``clean`` is any object with ``x`` (N,T,C float32, 0 at unobserved), ``m``
    (N,T,C bool) and ``channel_names``; :class:`CycleTensors` satisfies it, and the
    unit tests feed a synthetic stand-in so the generator is testable without data.
    """
    syn = cfg.raw["synthetic"]
    families = tuple(syn.get("families") or FAMILY_MECHANISMS)
    unknown = [f for f in families if f not in FAMILY_SHAPES]
    if unknown:
        raise KeyError(
            f"synthetic families {unknown} are not implemented; registered: {sorted(FAMILY_SHAPES)}"
        )
    sevs = list(syn.get("severities") or ["weak", "moderate", "strong"])
    mults = {k: float(v) for k, v in (syn.get("severity_multipliers") or {}).items()}
    missing = [s for s in sevs if s not in mults]
    if missing:
        raise KeyError(f"severity {missing} has no multiplier in synthetic.severity_multipliers")
    events = int(syn.get("events_per_cell", 160))
    prevalence = float(syn["test_prevalence"])
    names = tuple(getattr(clean, "channel_names", ()) or ())
    if not names:
        names = tuple(f"ch{i}" for i in range(clean.x.shape[2]))
    ch = {n: i for i, n in enumerate(names)}
    x = np.array(clean.x, dtype=np.float32, copy=True)
    m = np.asarray(clean.m, dtype=bool)
    n, T, C = x.shape
    # Robust per-channel scale of the CLEAN data (the contract's MAD source). The
    # mask is honoured: an unobserved bin is 0 in `x`, and including those zeros in
    # the scale would shrink every magnitude toward "no change".
    scale = np.empty(C)
    for c in range(C):
        v = x[..., c][m[..., c]]
        scale[c] = float(np.median(np.abs(v - np.median(v))) * 1.4826) if v.size else np.nan
    bad = ~np.isfinite(scale) | (scale <= 0)
    if bad.all():
        raise ValueError("no channel has usable clean data to define severity against")
    scale[bad] = np.nanmax(scale[~bad])           # a channel with no data borrows the largest scale
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n) if cycle_subset is None else np.asarray(cycle_subset, dtype=np.int64)
    cells: list[InjectionCell] = []
    labels_bin = np.zeros((n, T), dtype=bool)
    combined = {c["name"]: tuple(c["families"]) for c in (syn.get("combined_scenarios") or [])}
    unknown_combo = sorted({f for fams in combined.values() for f in fams if f not in FAMILY_SHAPES})
    if unknown_combo:
        raise KeyError(
            f"combined_scenarios {unknown_combo} name families that are not implemented; "
            f"registered: {sorted(FAMILY_SHAPES)}"
        )
    grid = [(f, s) for f in families for s in sevs]
    for fam, sev in grid:
        cyc, ons, lab = plan_events(
            len(idx), T, family=fam, events=events, prevalence=prevalence, rng=rng,
            observed=m[idx].any(-1),
        )
        gidx = idx[cyc]
        delta = np.zeros((n, T), dtype=np.float64)
        shape = FAMILY_SHAPES[fam]
        resp = FAMILY_CHANNELS[fam]
        for j, gi in enumerate(gidx):
            length = max(1, int(lab[j].sum()))
            prof = shape(T, int(ons[j]), length, rng) * mults[sev]
            for cname, sign in resp.items():
                c = ch.get(cname)
                if c is None:
                    continue
                x[gi, :, c] = (x[gi, :, c] + prof * sign * scale[c]).astype(np.float32)
            delta[gi] += prof * resp.get("ndvi", 0.0) * scale[ch.get("ndvi", 0)]
        lab_full = np.zeros((n, T), dtype=bool)
        # `cyc` indexes the *subset* and `gidx` are the global rows in the same order;
        # `lab` is enumeration-ordered (row j ↔ gidx[j]), so this assignment maps each
        # event to its own cycle even when a pool (train-only / dev-only) is narrower
        # than the archive.
        lab_full[gidx] = lab
        labels_bin |= lab_full
        cells.append(
            InjectionCell(
                family=fam, severity=sev, magnitude=mults[sev], cycle_index=gidx,
                onset_index=ons, labels=lab_full, delta=delta, channels=tuple(resp),
                n_events=int(len(gidx)), n_affected_bins=int(lab_full.sum()),
                mean_length_bins=float(lab_full.sum(1)[lab_full.sum(1) > 0].mean())
                if lab_full.any() else float("nan"),
            )
        )
    # ---- task 4.9: the designed combined scenarios ------------------------------------
    # Each scenario draws ONE set of cycles per severity and injects every constituent
    # family into the SAME cycle at that family's own (observed-window) onset, so one
    # event genuinely carries several mechanisms at once. The labels are the union of the
    # component windows, and `components` records each family's onset/length/channels —
    # the ground-truth metadata 4.10 requires for a multi-family event. These cells are
    # scored per-cell but stay outside the pre-registered 5 × 3 headline macro.
    for name, fams in combined.items():
        for sev in sevs:
            n_ev = max(1, min(int(events), int(round(prevalence * len(idx) * 1.0)), len(idx)))
            cyc = rng.choice(len(idx), size=n_ev, replace=False)
            gidx = idx[cyc]
            observed_rows = m[idx].any(-1)
            lab_union = np.zeros((n_ev, T), dtype=bool)
            delta = np.zeros((n, T), dtype=np.float64)
            components: list[dict] = []
            channels_all: list[str] = []
            for fam in fams:
                ons, lab_f = _draw_onsets(cyc, T, family=fam, rng=rng, observed=observed_rows)
                shape = FAMILY_SHAPES[fam]
                resp = FAMILY_CHANNELS[fam]
                for j, gi in enumerate(gidx):
                    length = max(1, int(lab_f[j].sum()))
                    prof = shape(T, int(ons[j]), length, rng) * mults[sev]
                    for cname, sign in resp.items():
                        c = ch.get(cname)
                        if c is None:
                            continue
                        x[gi, :, c] = (x[gi, :, c] + prof * sign * scale[c]).astype(np.float32)
                    delta[gi] += prof * resp.get("ndvi", 0.0) * scale[ch.get("ndvi", 0)]
                    components.append({"family": fam, "cycle_position": int(j),
                                       "onset_bin": int(ons[j]), "length_bins": length,
                                       "magnitude": float(mults[sev]),
                                       "channels": sorted(resp)})
                lab_union |= lab_f
                channels_all += [c for c in resp if c not in channels_all]
            lab_full = np.zeros((n, T), dtype=bool)
            lab_full[gidx] = lab_union
            labels_bin |= lab_full
            cells.append(
                InjectionCell(
                    family=f"combined:{name}", severity=sev, magnitude=mults[sev],
                    cycle_index=gidx,
                    onset_index=np.array([c["onset_bin"] for c in components
                                          if c["family"] == fams[0]][:n_ev], dtype=np.int64),
                    labels=lab_full, delta=delta, channels=tuple(channels_all),
                    n_events=int(len(gidx)), n_affected_bins=int(lab_full.sum()),
                    mean_length_bins=float(lab_full.sum(1)[lab_full.sum(1) > 0].mean())
                    if lab_full.any() else float("nan"),
                    scenario=name, scenario_families=tuple(fams), components=components,
                )
            )
    panel = SyntheticPanel(
        x=x.astype(np.float32), m=m, cells=cells, labels_bin=labels_bin,
        labels_cycle=labels_bin.any(axis=1), prevalence_target=prevalence, seed=int(seed),
        pool=pool, channel_names=names,
        generator_sha256=_tensor_sha(x) , clean_sha256=_tensor_sha(np.asarray(clean.x)),
        notes=[f"role={role}", "severity multiplier × per-channel MAD of clean data",
               f"events_per_cell={events}",
               (f"combined_scenarios={sorted(combined)}: cells named combined:<scenario>__<severity>, "
                "reported per-cell, excluded from the pre-registered 5×3 headline macro"
                if combined else "no combined_scenarios declared")],
    )
    audit_realism(panel, clean_x=np.asarray(clean.x), scale=scale, cfg=cfg)
    for name, fams in combined.items():
        panel.audit_summary.setdefault("combined_scenarios", {})[name] = {
            "families": list(fams),
            "cells": [c.cell for c in panel.cells if c.scenario == name],
        }
    return panel


def audit_realism(panel: SyntheticPanel, *, clean_x: np.ndarray, scale: np.ndarray, cfg) -> dict:
    """Apply the contract's trivial-separability rule and record the excluded IDs."""
    syn = cfg.raw["synthetic"]
    ts = syn.get("trivial_separability") or {}
    rule = str(ts.get("flag_if", ""))
    # Parse the contract's own wording so the code cannot drift from the document.
    frac_thr, mult_thr = 0.98, 4.0
    import re

    mm = re.search(r"([0-9.]+)\s*of", rule)
    if mm:
        frac_thr = float(mm.group(1))
    mm = re.search(r"exceeds\s*([0-9.]+)", rule)
    if mm:
        mult_thr = float(mm.group(1))
    if str(ts.get("action", "flag_and_exclude")) != "flag_and_exclude":
        raise ValueError(
            "synthetic.trivial_separability.action must stay 'flag_and_exclude': a different action "
            "would mean the panel's saturated cells were kept, which the pre-registration forbids"
        )
    ch = {n: i for i, n in enumerate(panel.channel_names)}
    for c in panel.cells:
        d = np.abs(c.delta[c.labels]) if c.labels.any() else np.zeros(1)
        base = float(np.nanmedian(scale[[ch[n] for n in c.channels if n in ch]])) if c.channels else 1.0
        over = float((d > mult_thr * max(base, 1e-6)).mean()) if d.size else 0.0
        c.audit = {
            "median_abs_delta": round(float(np.median(d)), 6) if d.size else None,
            "clean_mad_scale": round(base, 6),
            "delta_over_mad": round(float(np.median(d) / max(base, 1e-9)), 3) if d.size else None,
            "frac_delta_above_threshold": round(over, 4),
            "threshold": f"{frac_thr} of |delta| > {mult_thr} × MAD",
            "n_channels_moved": len(c.channels),
            "trivial_rule": "ndvi_threshold",
            "trivial_rule_auprc": _trivial_rule_auprc(panel, c, clean_x, ch),
        }
        if over >= frac_thr:
            panel.excluded_cells.append(
                {"cell": c.cell, "reason": "trivial_separability",
                 "frac_delta_above_threshold": over, "cycle_ids": [int(v) for v in c.cycle_index[:40]]}
            )
    panel.audit_summary = {
        "n_cells": len(panel.cells),
        "n_flagged": len(panel.excluded_cells),
        "flag_action": "flag_and_exclude",
        "usable_cells": panel.usable_cells,
        "trivial_separability_rule": rule or "unparsed_default",
    }
    return panel.audit_summary


def _trivial_rule_auprc(panel: SyntheticPanel, c: InjectionCell, clean_x: np.ndarray, ch: dict) -> float | None:
    """How well a bare ``-NDVI`` rule already separates this cell (on the CLEAN input)."""
    if "ndvi" not in ch or not c.labels.any():
        return None
    try:
        from sklearn.metrics import average_precision_score
    except Exception:  # pragma: no cover - sklearn is a hard dependency here
        return None
    s = -clean_x[:, :, ch["ndvi"]]
    y = c.labels.astype(int).ravel()
    if np.unique(y).size < 2:
        return None
    return float(average_precision_score(y, s.ravel()))


def build_evaluation_panel(clean, cfg, *, seed: int | None = None) -> SyntheticPanel:
    syn = cfg.raw["synthetic"]["eval_generator"]
    return apply_panel(
        clean, cfg, seed=int(syn["seed"] if seed is None else seed),
        pool=str(syn.get("pool", "dev_cycles_only")), role="evaluation",
    )


def build_train_contamination(clean, cfg, *, seed: int | None = None,
                             cycle_subset: np.ndarray | None = None) -> SyntheticPanel:
    g = cfg.raw["synthetic"]["train_generator"]
    if not g.get("independent_from_eval") or not g.get("disjoint_sample_pool"):
        raise ValueError(
            "synthetic.train_generator must keep independent_from_eval and disjoint_sample_pool true; "
            "contaminating training with the evaluation stream would invalidate every cell"
        )
    return apply_panel(
        clean, cfg, seed=int(g["seed"] if seed is None else seed),
        pool=str(g.get("pool", "train_cycles_only")), role="train_contamination",
        cycle_subset=cycle_subset,
    )


def pool_disjointness(a: SyntheticPanel, b: SyntheticPanel) -> dict:
    """The recorded proof that the two generators drew different events."""
    ia = {r["identity_sha256"] for r in event_records(a)}
    ib = {r["identity_sha256"] for r in event_records(b)}
    shared = sorted(ia & ib)
    return {
        "n_eval_events": len(ia), "n_train_events": len(ib), "shared": len(shared),
        "shared_examples": shared[:5], "disjoint": not shared,
        "isolation_check": "sha256_of_(pool,cycle_index,start_bin,family,severity)",
    }


def manifest_block(panel: SyntheticPanel) -> dict:
    return panel.to_dict()
