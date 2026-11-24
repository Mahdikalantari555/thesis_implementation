"""Equal-capacity tuning across paradigms and backbones (task 3.4 / 9.13).

The contract gives a target and a band (``capacity.target_params``, ``min_ratio``,
``max_ratio``) plus a search grid (``capacity.d_model_grid``). This module is the
only place those numbers become a ``d_model`` per arm, so the report's capacity
column can be pointed at a deterministic search rather than at a sequence of manual
edits to per-model YAMLs.

Two properties make it usable as *evidence*:

* **Deterministic.** Candidates are tried in grid order and the first in-band
  configuration wins; equal distances break the tie by the smaller model. No RNG,
  no early stopping, no seeds — a capacity choice that changes between runs would
  confound the paradigm comparison it is supposed to protect.
* **The full trace is returned.** Every probe's parameter count and whether it was
  in band, above or below. When no candidate fits — which is normal for the wider
  trunk at a small ``d_channels`` — the trace is what lets the report say "the
  closest achievable was 142 k (1.18×), so this arm is reported outside the band"
  instead of silently dropping the arm or fudging the number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ssl_anomaly.models.bench.arms import ARM_BY_MODEL, build_arm, count_trainable


@dataclass
class CapacityProbe:
    d_model: int
    params: int
    ratio: float
    in_band: bool
    side: str = "in"

    def to_dict(self) -> dict:
        return {"d_model": self.d_model, "params": self.params, "ratio": round(self.ratio, 4),
                "in_band": self.in_band, "side": self.side}


@dataclass
class CapacityChoice:
    model_id: str
    backbone: str
    chosen: CapacityProbe | None
    probes: list[CapacityProbe] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id, "backbone": self.backbone,
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "in_band": bool(self.chosen and self.chosen.in_band),
            "trace": [p.to_dict() for p in self.probes],
            "note": self.note,
        }


def _probe(arm_cls_kwargs: dict, d_model: int) -> int:
    """Count parameters without touching the GPU: build on the meta-like small path."""
    from ssl_anomaly.models.bench.arms import ARM_CLASSES

    kw = dict(arm_cls_kwargs)
    kw["d_model"] = int(d_model)
    cls = ARM_CLASSES[kw.pop("paradigm")]
    arm = cls(**kw)
    n = count_trainable(arm)
    del arm
    return n


def tune_capacity(
    cfg, *, model_id: str, d_channels: int, d_cov: int = 0, variant: str = "base",
    grid: list[int] | None = None,
) -> CapacityChoice:
    """Pick the in-band ``d_model`` for one arm, or explain why none exists."""
    cap = cfg.raw["capacity"]
    target = int(cap["target_params"])
    lo, hi = float(cap["min_ratio"]), float(cap["max_ratio"])
    if not (0 < lo <= 1 <= hi):
        raise ValueError(
            f"capacity band {lo}–{hi}× must bracket 1.0; the validator would otherwise accept any model"
        )
    grid = list(grid if grid is not None else cap.get("d_model_grid") or [64, 96, 128])
    from ssl_anomaly.benchmarking.config import ARCHITECTURES

    row = ARCHITECTURES.get(model_id) or {}
    backbone = row.get("backbone") or ARM_BY_MODEL.get(model_id, ("", "gru"))[1]
    paradigm = row.get("paradigm") or ARM_BY_MODEL.get(model_id, ("input_reconstruction", ""))[0]
    t = cfg.raw["training"]
    rates = (t.get("masking") or {}).get("rate_by_paradigm") or {}
    lam = dict(t)
    variant_map = t.get("lambda_variants") or {}
    if variant in variant_map:
        lam = {**t, **{k: v for k, v in variant_map[variant].items() if k.startswith("lambda_")}}
    kw = {
        "paradigm": paradigm,
        "d_channels": int(d_channels), "d_cov": int(d_cov), "backbone": backbone,
        "n_bins": int(t.get("n_bins", 36)), "n_layers": int(t.get("n_layers", 1)),
        "dropout": float(t.get("dropout", 0.1)), "d_state": int(t.get("d_state", 16)),
        "latent_dim": int(t.get("latent_dim", 32)), "huber_beta": float(lam.get("huber_beta", 1.0)),
        "horizon_k": int(lam.get("horizon_k", 4)), "ema_momentum": float(lam.get("ema_momentum", 0.996)),
        "lambda_rec": float(lam.get("lambda_rec", 1.0)), "lambda_pred": float(lam.get("lambda_pred", 1.0)),
        "variance_weight": float(cfg.raw["collapse"].get("variance_weight", 0.05)),
        "mask_rate": float(rates.get(paradigm, 0.0)), "causal": True,
        "bottleneck_frac": float(cap.get("bottleneck_frac", 0.25)),
    }
    probes: list[CapacityProbe] = []
    for d in grid:
        n = _probe(kw, int(d))
        r = n / float(target)
        probes.append(CapacityProbe(d_model=int(d), params=int(n), ratio=r,
                                    in_band=bool(lo <= r <= hi),
                                    side="in" if lo <= r <= hi else ("above" if r > hi else "below")))
    chosen = next((p for p in probes if p.in_band), None)
    note = ""
    if chosen is None:
        # Closest by ratio, then smaller model — the tie-break is stated so a reader
        # knows the fallback is a rule and not a preference.
        best = min(probes, key=lambda p: (abs(p.ratio - 1.0), p.d_model))
        chosen = best
        note = (
            f"no grid point lands inside {lo:.2f}–{hi:.2f}× of {target:,} params for {model_id} at "
            f"d_channels={d_channels}; the closest is {best.params:,} ({best.ratio:.2f}×) and the arm "
            "is reported outside the band rather than dropped"
        )
    return CapacityChoice(model_id=model_id, backbone=backbone, chosen=chosen, probes=probes, note=note)


def resolve_capacities(cfg, *, d_channels: int, d_cov: int = 0, model_ids=None,
                       variants: dict | None = None) -> dict:
    """Tune every arm once, and return the table the manifest embeds."""
    ids = list(model_ids if model_ids is not None else cfg.architectures[:10])
    variants = dict(variants or {})
    arms = {}
    for mid in ids:
        c = tune_capacity(cfg, model_id=mid, d_channels=d_channels, d_cov=d_cov,
                          variant=variants.get(mid, "base"))
        arms[mid] = c.to_dict()
    in_band = [a for a in arms.values() if a["in_band"]]
    return {
        "target_params": int(cfg.raw["capacity"]["target_params"]),
        "band": [float(cfg.raw["capacity"]["min_ratio"]), float(cfg.raw["capacity"]["max_ratio"])],
        "d_channels": int(d_channels), "d_cov": int(d_cov),
        "arms": arms,
        "n_in_band": len(in_band), "n_arms": len(arms),
        "all_in_band": len(in_band) == len(arms),
        "spread": _spread(arms),
        "note": (
            "capacity is tuned per arm over the pre-registered d_model grid; a fair comparison is "
            "in-band, not equal-d_model"
        ),
    }


def _spread(arms: dict) -> dict:
    vals = [a["chosen"]["params"] for a in arms.values() if a.get("chosen")]
    if not vals:
        return {"available": False, "reason": "no arm resolved"}
    lo, hi = min(vals), max(vals)
    return {"available": True, "min_params": int(lo), "max_params": int(hi),
            "max_over_min": round(float(hi / max(1, lo)), 4),
            "ratios_to_target": {k: a["chosen"]["ratio"] for k, a in sorted(arms.items())}}


def capacity_table(res: dict) -> str:
    """Markdown table for the report (model, backbone, d_model, params, ratio, band)."""
    rows = ["| model | backbone | d_model | params | ratio | in band |", "|---|---|---:|---:|---:|:--:|"]
    for mid, a in sorted(res["arms"].items()):
        ch = a.get("chosen") or {}
        rows.append(
            f"| {mid} | {a.get('backbone', '')} | {ch.get('d_model', '—')} | {ch.get('params', '—'):,} | "
            f"{ch.get('ratio', float('nan')):.3f} | {'✅' if a['in_band'] else '⚠️'} |"
            if isinstance(ch.get("params"), int) else
            f"| {mid} | {a.get('backbone', '')} | — | — | — | ⚠️ |"
        )
    if not res["all_in_band"]:
        rows.append("")
        rows.append("_Arms marked ⚠️ could not be placed inside the pre-registered band by the grid; "
                    "the closest achievable configuration is reported and the deviation is a finding, "
                    "not a rounding error._")
    return "\n".join(rows)
