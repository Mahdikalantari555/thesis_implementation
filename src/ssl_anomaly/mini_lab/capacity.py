"""Capacity matching for mini-lab arms, ±20 % band (task 2.4, design §4.5/§4.6).

Finding 28: mamba ≈ 293 k vs GRU ≈ 160 k params — a cross-architecture
comparison at unequal capacity is not fair (AGENTS §8 checklist). This module
picks, deterministically, a width for the TARGET arm that lands its parameter
count within ±20 % of a REFERENCE arm, and records the full probe trace so a
report can say "closest achievable was X (ratio r)" when no width fits —
never silently dropping or fudging an arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ModelConfig
from .models import MODEL_REGISTRY, build_model

#: pre-registered capacity band (design §4.6: ±20 %)
CAPACITY_BAND = 0.20

#: default width grid searched in order (deterministic first-fit)
DEFAULT_GRID: tuple[int, ...] = (16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128)


@dataclass
class CapacityProbe:
    d_model: int
    params: int
    ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {"d_model": self.d_model, "params": self.params, "ratio": round(self.ratio, 4)}


@dataclass
class CapacityMatch:
    reference: str
    target: str
    reference_params: int
    chosen: CapacityProbe | None
    probes: list[CapacityProbe] = field(default_factory=list)
    band: float = CAPACITY_BAND

    @property
    def in_band(self) -> bool:
        return self.chosen is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "target": self.target,
            "reference_params": self.reference_params,
            "band": self.band,
            "in_band": self.in_band,
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "trace": [p.to_dict() for p in self.probes],
            "note": (
                f"{self.target} matched to {self.chosen.d_model} wide "
                f"({self.chosen.params} params, ratio {self.chosen.ratio:.3f})"
                if self.chosen
                else "no width in the grid lands inside the ±20 % band; "
                     "report the closest achievable ratio instead of comparing"
            ),
        }


def _param_count(model_name: str, d_in: int, cfg: ModelConfig, d_model: int) -> int:
    probe_cfg = ModelConfig(
        name=model_name, d_model=d_model, latent_dim=cfg.latent_dim,
        n_layers=cfg.n_layers, dropout=cfg.dropout,
    )
    if not MODEL_REGISTRY[model_name].available:
        raise RuntimeError(f"model {model_name} unavailable")
    model = build_model(model_name, d_in, probe_cfg)
    return int(sum(p.numel() for p in model.parameters()))


def match_capacity(
    reference: str,
    target: str,
    d_in: int,
    cfg: ModelConfig,
    *,
    band: float = CAPACITY_BAND,
    grid: tuple[int, ...] = DEFAULT_GRID,
) -> CapacityMatch:
    """Pick the target arm's ``d_model`` so its params land within ±band.

    Deterministic: the grid is searched in order and the FIRST in-band width
    wins; ties impossible (strict inequalities). No RNG anywhere.
    """
    ref_params = _param_count(reference, d_in, cfg, cfg.d_model)
    lo, hi = ref_params * (1 - band), ref_params * (1 + band)
    probes: list[CapacityProbe] = []
    chosen: CapacityProbe | None = None
    for w in grid:
        p = _param_count(target, d_in, cfg, w)
        ratio = p / ref_params if ref_params else float("inf")
        probe = CapacityProbe(d_model=w, params=p, ratio=ratio)
        probes.append(probe)
        if chosen is None and lo <= p <= hi:
            chosen = probe
    return CapacityMatch(
        reference=reference, target=target, reference_params=ref_params,
        chosen=chosen, probes=probes, band=float(band),
    )


def capacity_report_field(
    arms: list[str],
    d_in: int,
    cfg: ModelConfig,
    *,
    band: float = CAPACITY_BAND,
) -> dict[str, Any]:
    """Report field: every arm's ratio to the SMALLEST arm + in-band flags.

    This is the field the sweep/report writes so a reader can see, per arm,
    whether a comparison is capacity-fair (ratio inside ±band of the
    smallest arm) or must carry the caveat.
    """
    counts = {a: _param_count(a, d_in, cfg, cfg.d_model) for a in arms}
    smallest = min(counts.values())
    return {
        "band": float(band),
        "smallest_params": smallest,
        "arms": {
            a: {
                "params": p,
                "ratio_to_smallest": round(p / smallest, 4) if smallest else None,
                "in_band": bool(abs(p / smallest - 1.0) <= band) if smallest else False,
            }
            for a, p in counts.items()
        },
        "note": "in_band = parameter count within ±20 % of the smallest arm "
                "(design §4.6); comparisons outside the band need the matched "
                "width from match_capacity()",
    }
