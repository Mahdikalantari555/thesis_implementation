"""Per-architecture hyper-parameter search under an equal budget (tasks 11.1–11.6).

The fairness rule the contract fixes is that every arm gets the **same search
budget** — same number of trials, same wall-clock cap — and that the selection
signal is validation-only. Both are easy to write down and easy to break:

* Unequal budgets turn "A1 beats A6" into "A1 with a better-tuned A6 wins". So the
  budget table is *compared* here, and :func:`assert_equal_budgets` refuses an
  asymmetric table rather than reporting the comparison anyway.
* A search that selects on the scored partition manufactures the winner. So the split
  is a required argument, checked against the pre-registration's admissible set, and
  :func:`audit_search` re-reads the finished trace and reports any trial that touched
  a forbidden split. The audit runs on the *record*, not on the caller's promise.

Sampling is a seeded random search over the declared grid (``log[a, b]`` for
continuous ones), with the grid order fixed: deterministic given the seed, so a
re-run reproduces the same configurations, and the trace lists every trial — including
the ones that lost, which is what makes "the winner was not cherry-picked" checkable.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ssl_anomaly.benchmarking.splits import assert_admissible

LOG_RE = re.compile(r"^log\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]$")
FORBIDDEN = ("test", "monitoring", "holdout")


class SearchError(RuntimeError):
    """The search cannot proceed fairly (bad space, forbidden split, unequal budget)."""


@dataclass
class Trial:
    index: int
    config: dict
    score: float
    split: str
    seconds: float
    params: int | None = None
    note: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {"index": self.index, "config": dict(self.config), "score": self.score,
                "split": self.split, "seconds": round(float(self.seconds), 3),
                "params": self.params, "note": self.note, "error": self.error}


@dataclass
class SearchRecord:
    model_id: str
    trials: list[Trial]
    best: Trial | None
    budget: dict
    split: str
    stopped_by: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def validation_only(self) -> bool:
        return str(self.split).lower() in ("dev", "validation", "val")

    def to_dict(self) -> dict:
        return {"model_id": self.model_id, "split": self.split, "budget": dict(self.budget),
                "n_trials": len(self.trials), "best": self.best.to_dict() if self.best else None,
                "best_over": [t.to_dict() for t in self.trials],
                "stopped_by": self.stopped_by, "notes": list(self.notes),
                "selection_metric": self.budget.get("selection_metric"),
                "validation_only": self.split in ("dev", "validation", "val")}


# ---------------------------------------------------------------------------
# spaces
# ---------------------------------------------------------------------------


def parse_space(raw: dict) -> dict[str, dict]:
    """Turn the YAML space into typed samplers, refusing anything unparseable.

    ``lr: log[3e-4, 3e-3]`` is a *string* in the config; if the pattern does not
    match, guessing would silently turn a log-uniform draw into a categorical one.
    """
    out: dict[str, dict] = {}
    for k, v in (raw or {}).items():
        if isinstance(v, (list, tuple)):
            vals = list(v)
            if not vals:
                raise SearchError(f"search space {k!r} is empty")
            out[str(k)] = {"kind": "choice", "values": vals}
            continue
        m = LOG_RE.match(str(v).strip())
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if not (0 < lo < hi):
                raise SearchError(f"{k}: log range {v!r} must be 0 < low < high")
            out[str(k)] = {"kind": "loguniform", "low": lo, "high": hi}
            continue
        m = re.match(r"^uniform\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]$", str(v).strip())
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if not hi > lo:
                raise SearchError(f"{k}: uniform range {v!r} must have high > low")
            out[str(k)] = {"kind": "uniform", "low": lo, "high": hi}
            continue
        raise SearchError(
            f"search space {k!r} = {v!r} is neither a candidate list nor log[a,b]/uniform[a,b]; "
            "an unparseable space would be sampled as nothing, so the arm would be reported as "
            "'tuned' with zero tuning done"
        )
    return out


def sample(space: dict[str, dict], rng: np.random.Generator) -> dict:
    cfg: dict[str, object] = {}
    for k, spec in space.items():
        if spec["kind"] == "choice":
            cfg[k] = spec["values"][int(rng.integers(len(spec["values"])))]
        elif spec["kind"] == "loguniform":
            lo, hi = math.log(spec["low"]), math.log(spec["high"])
            cfg[k] = float(math.exp(rng.uniform(lo, hi)))
        else:
            cfg[k] = float(rng.uniform(spec["low"], spec["high"]))
    return cfg


def budget_for(cfg, model_id: str) -> dict:
    s = cfg.raw["search"]
    b = dict(s.get("budgets") or {}).get(model_id)
    if b is None:
        un = s.get("unsearchable") or {}
        if model_id in un:
            raise SearchError(f"{model_id} is declared unsearchable: {un[model_id]}")
        raise SearchError(
            f"{model_id} has no entry in search.budgets; an arm tuned with an undeclared budget is "
            "not a fairly tuned arm"
        )
    return {"trials": int(b.get("trials", s.get("n_trials", 8))),
            "wall_minutes_cap": float(s.get("wall_minutes_cap", 25)),
            "selection_metric": s.get("selection_metric", "macro_auprc"),
            "selection_panel": s.get("selection_panel"), "validation_only": bool(s.get("validation_only")),
            "space": parse_space(b.get("space") or {})}


def assert_equal_budgets(cfg) -> dict:
    """Every searchable arm gets the same trial count and the same time cap."""
    s = cfg.raw["search"]
    rows = {k: {"trials": int((v or {}).get("trials", s.get("n_trials", 8)))}
            for k, v in (s.get("budgets") or {}).items()}
    cap = float(s.get("wall_minutes_cap", 25))
    bad = [k for k, v in rows.items() if v["trials"] != max(r["trials"] for r in rows.values())]
    if bad:
        raise SearchError(
            f"search budgets are unequal — {bad} get fewer trials than the maximum; the comparison "
            "would then measure tuning effort as well as architecture"
        )
    for v in rows.values():
        v["wall_minutes_cap"] = cap
    return {"arms": sorted(rows), "trials_per_arm": max(r["trials"] for r in rows.values()),
            "wall_minutes_cap": cap, "equal": True,
            "unsearchable": sorted((s.get("unsearchable") or {})),
            "note": "equal trials and equal wall cap per arm; no arm may search a wider space in secret"}


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


def run_search(cfg, model_id: str, evaluate: Callable[[dict], tuple[float, dict]], *,
               split: str = "dev", seed: int = 13, trials: int | None = None,
               wall_minutes: float | None = None, max_failures: int = 3) -> SearchRecord:
    """Seeded random search on the validation partition only.

    ``evaluate(config)`` must return ``(score, meta)`` with higher score better; the
    caller owns the training loop, so this module never needs to know what a model is,
    and cannot quietly start evaluating on a different split.
    """
    if str(split).lower() in FORBIDDEN:
        raise SearchError(
            f"selection on split {split!r} is refused: the search may only select on the validation "
            "partition (evaluation.threshold_rule and search.validation_only), and this refusal is the "
            "leakage guard, not a warning"
        )
    assert_admissible((split,))
    b = budget_for(cfg, model_id)
    if not b["validation_only"]:
        raise SearchError("search.validation_only is false in the contract — restore it before tuning")
    n = int(trials if trials is not None else b["trials"])
    cap = float(wall_minutes if wall_minutes is not None else b["wall_minutes_cap"])
    # A stable per-arm stream. Python's built-in hash() of a string is salted per
    # process, so using it here would make a re-run in a new process draw different
    # configurations — the classic "I could not reproduce my own search" bug.
    import hashlib

    stream = int(hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    rng = np.random.default_rng(int(seed) * 7919 + stream)
    rec = SearchRecord(model_id=model_id, trials=[], best=None, budget={k: v for k, v in b.items()
                       if k != "space"}, split=str(split))
    t0 = time.time()
    fails = 0
    for i in range(n):
        if (time.time() - t0) / 60.0 > cap:
            rec.stopped_by = "wall_minutes_cap"
            rec.notes.append(
                f"stopped after {i}/{n} trials at the {cap:.0f} min cap; the winner is the best of the "
                f"{i} trials actually run, and the truncation is part of the record")
            break
        c = sample(b["space"], rng)
        tb = time.time()
        try:
            score, meta = evaluate(dict(c))
            s = float(score)
        except Exception as exc:  # noqa: BLE001 - a failed trial is data, not a reason to abort the search
            fails += 1
            rec.trials.append(Trial(i, c, float("nan"), str(split), time.time() - tb,
                                    error=f"{type(exc).__name__}: {exc}"))
            if fails >= max_failures:
                rec.stopped_by = f"{fails}_consecutive_failures"
                rec.notes.append("aborted after repeated evaluation failures; the errors are in the trace")
                break
            continue
        if not math.isfinite(s):
            rec.notes.append(f"trial {i} returned a non-finite score and is excluded from selection")
            rec.trials.append(Trial(i, c, float("nan"), str(split), time.time() - tb,
                                    note="non-finite score", error=str(meta.get("error") or "") or None))
            continue
        rec.trials.append(Trial(i, c, round(s, 6), str(split), time.time() - tb,
                                params=(meta or {}).get("params"),
                                note=str((meta or {}).get("note") or "")))
        if rec.best is None or s > rec.best.score:
            rec.best = rec.trials[-1]
    if rec.best is None:
        rec.stopped_by = rec.stopped_by or "no_successful_trial"
        rec.notes.append(
            "no trial produced a finite selection score: the arm keeps its pre-registered default "
            "configuration, and the report says so instead of using a tuned winner that does not exist")
    return rec


def audit_search(records: list[SearchRecord]) -> dict:
    """Re-read the finished traces and check the two things that must be true."""
    rows = []
    for r in records:
        bad_split = sorted({t.split for t in r.trials if str(t.split).lower() in FORBIDDEN})
        rows.append({"model_id": r.model_id, "n_trials": len(r.trials),
                     "n_scored": int(sum(1 for t in r.trials if math.isfinite(t.score))),
                     "n_error": int(sum(1 for t in r.trials if t.error)),
                     "split": r.split, "violations": bad_split,
                     "validation_only": r.split in ("dev", "validation", "val"),
                     "best": r.best.to_dict() if r.best else None,
                     "trials_equal_to_budget": len(r.trials) >= int(r.budget.get("trials", 0))
                     or r.stopped_by is not None,
                     "stopped_by": r.stopped_by})
    viol = [r for r in rows if r["violations"]]
    return {"ok": not viol, "per_arm": rows,
            "budget_equal": None, "n_arms": len(rows),
            "note": ("no trial may have been selected on a forbidden split; this reads the recorded "
                     "trials rather than trusting the caller"),
            "violations": viol}


def search_table(records: list[SearchRecord]) -> str:
    lines = ["| arm | trials | scored | errors | best score | best config | wall cap | stopped by |",
             "|---|---:|---:|---:|---:|---|---:|---|"]
    for r in records:
        best = r.best
        cfg_s = ", ".join(f"{k}={v if not isinstance(v, float) else f'{v:.4g}'}"
                          for k, v in (best.config.items() if best else {})) or "—"
        lines.append(f"| {r.model_id} | {len(r.trials)} | "
                     f"{sum(1 for t in r.trials if math.isfinite(t.score))} | "
                     f"{sum(1 for t in r.trials if t.error)} | "
                     f"{'—' if best is None else f'{best.score:.4f}'} | {cfg_s} | "
                     f"{r.budget.get('wall_minutes_cap')} | {r.stopped_by or '—'} |")
    return "\n".join(lines)
