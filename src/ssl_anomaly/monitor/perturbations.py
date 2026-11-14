"""MAD-scaled synthetic perturbations for the development metric (design D6).

Severity is expressed in multiples of the TRAIN MAD of the affected band at
the affected phase bin (0.5 / 1 / 1.5 / 2 × MAD), so a "1×MAD" perturbation
means the same thing for every band and every phase. The onset bin is kept as
ground truth; detection delay is measured from it.

Interim perturbation set (until the advisor answer, todo OQ6):
  1. ndvi_drop      — gradual NDVI drop, partial recovery (P1)
  2. lst_rise       — fast LST rise, full recovery (P2)
  3. phenology_delay — growth phase shifted later (P4)
  4. growth_stall   — canopy plateau at the onset level (P5)
  5. early_senescence — progressive NDVI decline, no recovery (P6)
  6. transient_drop — one-bin drop with recovery (P7-style)

All injections operate on the raw (bins, channels) tensor + observation mask,
modify only observed entries, and never touch CWSI/ETa (none exist in these
tensors — the forbidden-target test enforces this stays true).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INTERIM_TYPES: tuple[str, ...] = (
    "ndvi_drop",
    "lst_rise",
    "phenology_delay",
    "growth_stall",
    "early_senescence",
    "transient_drop",
)

SEVERITIES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
PHASES: tuple[str, ...] = ("early", "mid", "late")

CH_NDVI = 0
CH_BLUE, CH_GREEN, CH_RED, CH_NIR, CH_SWIR1, CH_SWIR2 = 1, 2, 3, 4, 5, 6
CH_LST = 7

# Channels touched by each perturbation type.
TYPE_CHANNELS: dict[str, tuple[int, ...]] = {
    "ndvi_drop": (CH_NDVI,),
    "lst_rise": (CH_LST,),
    "phenology_delay": (CH_NDVI, CH_GREEN, CH_RED, CH_NIR, CH_SWIR1, CH_SWIR2),
    "growth_stall": (CH_NDVI, CH_GREEN, CH_NIR),
    "early_senescence": (CH_NDVI, CH_NIR),
    "transient_drop": (CH_NDVI,),
}


@dataclass
class InjectionResult:
    x: np.ndarray  # perturbed (bins, channels)
    m: np.ndarray  # observation mask (unchanged)
    onset_bin: int
    pert_type: str
    severity: float
    phase: str
    source_row: int = -1  # index of the source cycle in the injected array


@dataclass
class PerturbationConfig:
    types: tuple[str, ...] = INTERIM_TYPES
    severities: tuple[float, ...] = SEVERITIES
    phases: tuple[str, ...] = PHASES
    durations: tuple[int, ...] = (1, 2, 3, 5)
    bin_days: int = 15
    n_per_stratum: int = 8


def phase_start_bin(phase: str, n_bins: int) -> int:
    if phase == "early":
        return max(1, n_bins // 6)
    if phase == "mid":
        return n_bins // 2
    if phase == "late":
        return (5 * n_bins) // 6
    raise ValueError(f"Unknown phase {phase}")


def mad_at(bin_idx: int, channel: int, mad: np.ndarray) -> float:
    """Train MAD (raw units) of one band at one phase bin.

    ``mad`` is the per-(bin, channel) scale array from the robust stats
    (scale = MAD × 1.4826); the MAD itself is scale / 1.4826.
    """
    return float(mad[bin_idx, channel]) / 1.4826


def inject_perturbation(
    x: np.ndarray,
    m: np.ndarray,
    *,
    pert_type: str,
    onset_bin: int,
    duration: int,
    severity: float,
    mad: np.ndarray,  # (bins, channels) scale = MAD×1.4826 from train stats
    rng: np.random.Generator | None = None,
) -> InjectionResult:
    """Inject one perturbation into a raw cycle tensor.

    Only observed entries are modified. Returns the perturbed copy with the
    onset bin as ground truth.
    """
    del rng  # injections are deterministic given (type, onset, duration, severity)
    out = x.copy()
    T, C = out.shape
    onset = int(onset_bin)
    end = min(T, onset + int(duration))
    if onset >= T or duration <= 0:
        return InjectionResult(out, m, onset, pert_type, severity, "")

    def obs(bin_idx: int, ch: int) -> bool:
        return bool(m[bin_idx, ch])

    def mad_of(ch: int, b: int) -> float:
        return mad_at(min(b, mad.shape[0] - 1), ch, mad)

    if pert_type == "ndvi_drop":
        # Gradual ramp to full depth, hold, then partial (50 %) recovery.
        for b in range(onset, end):
            frac = (b - onset + 1) / max(1, end - onset)
            depth = severity * mad_of(CH_NDVI, b) * frac
            if obs(b, CH_NDVI):
                out[b, CH_NDVI] -= depth
        for b in range(end, min(T, end + 2)):
            depth = severity * mad_of(CH_NDVI, b) * 0.5 * (1.0 - (b - end) / 2.0)
            if obs(b, CH_NDVI):
                out[b, CH_NDVI] -= depth

    elif pert_type == "lst_rise":
        # Fast rise over the duration bins, full recovery afterwards.
        for b in range(onset, end):
            if obs(b, CH_LST):
                out[b, CH_LST] += severity * mad_of(CH_LST, b)

    elif pert_type == "phenology_delay":
        # Growth trajectory continues as if the crop were `k` bins younger.
        k = max(1, int(round(severity)))  # 1×MAD ↔ 1 bin (15 days)
        channels = TYPE_CHANNELS[pert_type]
        shifted = out.copy()
        for b in range(onset, T):
            src = b - k
            if src < 0:
                continue
            for ch in channels:
                # Never erase observations: shift only where the source bin
                # has a value too; otherwise leave the target untouched.
                if obs(b, ch) and obs(src, ch) and np.isfinite(out[src, ch]):
                    shifted[b, ch] = out[src, ch]
        out = shifted

    elif pert_type == "growth_stall":
        # Canopy frozen at the onset level for the duration (level taken from
        # the first observed bin at or after the onset).
        channels = TYPE_CHANNELS[pert_type]
        for ch in channels:
            level_bin = next((b for b in range(onset, end) if obs(b, ch)), None)
            if level_bin is None:
                continue
            level = out[level_bin, ch]
            for b in range(level_bin, end):
                if obs(b, ch):
                    out[b, ch] = level

    elif pert_type == "early_senescence":
        # Progressive deepening decline, no recovery inside the horizon.
        for b in range(onset, end):
            frac = (b - onset + 1) / max(1, end - onset)
            depth = severity * mad_of(CH_NDVI, b) * frac
            if obs(b, CH_NDVI):
                out[b, CH_NDVI] -= depth
            if obs(b, CH_NIR):
                out[b, CH_NIR] -= 0.5 * severity * mad_of(CH_NIR, b) * frac

    elif pert_type == "transient_drop":
        # One-bin drop then linear recovery over the next two bins.
        if obs(onset, CH_NDVI):
            out[onset, CH_NDVI] -= severity * mad_of(CH_NDVI, onset)
        for b in range(onset + 1, min(T, onset + 3)):
            depth = severity * mad_of(CH_NDVI, b) * (1.0 - (b - onset) / 2.0)
            if depth > 0 and obs(b, CH_NDVI):
                out[b, CH_NDVI] -= depth

    else:
        raise ValueError(f"Unknown perturbation type {pert_type}")

    return InjectionResult(out, m, onset, pert_type, severity, "")


def sample_strata(
    cfg: PerturbationConfig,
    rng: np.random.Generator,
    n_cycles: int,
) -> list[tuple[str, float, str, int]]:
    """Enumerate the (type, severity, phase, duration) strata for one cycle set.

    Each of the n_per_stratum sampled cycles draws one stratum uniformly, so
    the injected set covers types × phases with severity 1×MAD getting the
    primary weight (severity 1 is sampled twice as often as the others).
    """
    strata: list[tuple[str, float, str, int]] = []
    for _ in range(min(cfg.n_per_stratum, n_cycles)):
        ptype = str(cfg.types[int(rng.integers(len(cfg.types)))])
        phase = str(cfg.phases[int(rng.integers(len(cfg.phases)))])
        # Primary severity (1×MAD) oversampled 2:1 against the others.
        if rng.random() < 0.5:
            sev = 1.0
        else:
            sev = float(cfg.severities[int(rng.integers(len(cfg.severities)))])
        duration = int(cfg.durations[int(rng.integers(len(cfg.durations)))])
        strata.append((ptype, sev, phase, duration))
    return strata


def inject_dataset(
    X: np.ndarray,          # (N, bins, channels) raw values
    M: np.ndarray,          # (N, bins, channels) observation mask
    mad: np.ndarray,        # (bins, channels) scale = MAD×1.4826
    cfg: PerturbationConfig,
    rng: np.random.Generator,
) -> list[InjectionResult]:
    """Inject one perturbation into a random subset of the N cycles.

    The subset (size ``min(cfg.n_per_stratum, N)``) is drawn uniformly
    without replacement each round, so repeated rounds cover different
    cycles as well as different strata. Result j corresponds to source cycle
    ``cycle_idx[j]`` (also stored on the result as ``source_row``).
    """
    n = X.shape[0]
    n_inj = min(cfg.n_per_stratum, n)
    cycle_idx = rng.choice(n, size=n_inj, replace=False)
    strata = sample_strata(cfg, rng, n_inj)
    results: list[InjectionResult] = []
    for j, (ptype, sev, phase, duration) in enumerate(strata):
        i = int(cycle_idx[j])
        onset = phase_start_bin(phase, X.shape[1])
        res = inject_perturbation(
            X[i], M[i], pert_type=ptype, onset_bin=onset, duration=duration,
            severity=sev, mad=mad, rng=rng,
        )
        res.phase = phase
        res.source_row = i
        results.append(res)
    return results
