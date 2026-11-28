"""Report generation: three winner tables, honest coverage, and the question ledger.

The report is deliberately a *consumer* of the run directory — it reads manifests,
metrics and the registry and never recomputes a metric. If a number cannot be
produced from what is on disk it is printed as missing, with the reason. That
asymmetry is the point: a report that could synthesise a value would also be able to
synthesise a conclusion.

Three separate winner reports, because the design has three independent questions and
collapsing them into one leaderboard is the classic way to over-claim:

* **Paradigm** (what the representation is trained to predict) — the thesis question.
* **Backbone** (GRU vs Mamba at matched capacity) — an architecture question that must
  not be allowed to smuggle itself into the paradigm claim.
* **Suppression** (S0–S3 as training-time regularisers) — an intervention question
  whose answer is only valid per paradigm, so the table is per-paradigm.

Each winner is stated with its uncertainty (paired cluster bootstrap over *farms*),
a non-parametric test, an effect size, and the coverage caveat that applies to it.
Where the evidence cannot support a winner, the section says so in those words.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ssl_anomaly.benchmarking import stats as _st
from ssl_anomaly.benchmarking.config import ARCHITECTURES, PREREGISTERED_SEEDS
from ssl_anomaly.benchmarking.matrix import coverage_line, status_counts
from ssl_anomaly.benchmarking.registry import REGISTRY_STATES, Registry

#: per-cell key in every manifest's ``metrics.per_cell[*]``
PRIMARY = "auprc"
#: the headline key inside ``metrics.macro``. The macro summary is produced by
#: ``metrics.macro_summary``, which names its central value ``macro``; reading
#: ``auprc`` there would return nothing and every table would come out empty — an
#: easy failure to miss, because an empty table looks like an honest "no data".
PRIMARY_MACRO = "macro"
MIN_BOOT = 1000                      # the pre-registered floor; below it, no winner
#: The 14 questions the change asks. ``status`` is computed from what is on disk, so
#: the file cannot claim an answer it has no data for.
QUESTIONS = [
    ("Q1", "Does training to predict future observations produce a better anomaly "
           "score than training to reconstruct the present, at matched capacity?"),
    ("Q2", "Does a latent (decoder-free) reconstruction objective beat input-space "
           "reconstruction, or does it only look better because it is easier to optimise?"),
    ("Q3", "Is the JEPA-style objective viable at all at this sequence length and "
           "capacity, or does it need the collapse machinery to produce any signal?"),
    ("Q4", "Does a hybrid reconstruction+prediction objective sit above both parents, "
           "or is it an interpolation of noise?"),
    ("Q5", "At equal parameters and equal FLOPs convention, does the selective-SSM "
           "trunk beat the GRU on any paradigm?"),
    ("Q6", "Does the answer to Q5 survive the feasibility gate (wall time on the "
           "target accelerator), or is the Mamba arm only interesting if it gets cheaper?"),
    ("Q7", "Does masking context (S2) transfer across paradigms, or is it a "
           "reconstruction-only regulariser?"),
    ("Q8", "Does score-space suppression (S1) reduce false alarms at the same alarm "
           "budget, and does it cost detection delay?"),
    ("Q9", "Does training-time contamination (S3) reduce the score's dependence on "
           "transient artefacts, or does it just teach the arm to ignore weak events?"),
    ("Q10", "How much of the total variance in macro-AUPRC is architecture, how much "
            "is seed, farm and anomaly cell — i.e. is a single-seed ranking meaningful?"),
    ("Q11", "Is the winner robust to the choice of aggregation (max over bins vs mean, "
            "patch vs farm) and to the calibration method (C1/C2/C3)?"),
    ("Q12", "Are the per-channel error patterns interpretable, i.e. does the winner "
            "win for a reason a domain reader would accept?"),
    ("Q13", "Does adding weather covariates (W1 vs W0) change the ranking, and does "
            "the learned fusion gate (W2) move from its initialisation?"),
    ("Q14", "Is the whole benchmark's conclusion stable under the pre-registered seed "
            "set, or is it one unlucky seed with a large effect?"),
]


@dataclass
class ReportArtifacts:
    path: Path
    sections: list[str]
    warnings: list[str]

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_cells(out_dir: Path) -> list[dict]:
    """Every cell with a readable metrics payload, from the registry + run dirs."""
    out_dir = Path(out_dir)
    reg = Registry(out_dir / "registry.json")
    rows: list[dict] = []
    for e in reg.entries():
        # The registry stores what a *transition* needs; the seed lives in the
        # experiment id (and the manifest). Reading it from the id keeps the row
        # complete even for a cell that never reached a run directory.
        row = {"experiment_id": e.experiment_id, "cell_hash": e.cell_hash, "status": e.status,
               "model_id": e.model_id, "tier": e.tier, "priority": e.priority,
               "seed": _seed_of(e.experiment_id), "block": _block_of(e.experiment_id),
               "reason": e.reason, "error": e.error, "wall_seconds": e.wall_seconds or 0.0,
               "metrics": {}, "manifest": {}}
        md = _find_manifest(out_dir, e)
        if md is not None:
            row["run_dir"] = str(md.parent)
            try:
                man = json.loads(md.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                row["reason"] = f"{row['reason'] or ''} | unreadable manifest: {exc}".strip(" |")
            else:
                row["manifest"] = man
                row["metrics"] = man.get("metrics") or _load_json(md.parent / "metrics.json") or {}
                row["factors"] = man.get("factors") or {}
                row["cost"] = man.get("cost") or {}
                row["collapse"] = (man.get("collapse") or {}).get("verdict")
                row["per_cell"] = {c["cell"]: c for c in (row["metrics"].get("per_cell") or [])}
                # A fixture run is *usable for the pipeline* and unusable as evidence, and
                # the report keeps those two facts separate: the tables are computed (so a
                # broken statistic fails loudly at the local gate instead of hiding behind
                # an empty report), while every caption and the banner say what the numbers
                # are made of. Hiding the tables would only hide the bug.
                row["usable"] = bool(row["status"] == "Completed" and row["metrics"])
                row["fixture"] = bool(man.get("stamp"))
        else:
            row["run_dir"] = None
            row["usable"] = False
        row.setdefault("factors", _factors_from_id(row["experiment_id"]))
        row.setdefault("cost", {})
        row.setdefault("per_cell", {})
        row.setdefault("collapse", None)
        rows.append(row)
    return rows


def _seed_of(experiment_id: str) -> int:
    tail = str(experiment_id).rsplit(".s", 1)
    try:
        return int(tail[1])
    except (IndexError, ValueError):
        return -1


def _block_of(experiment_id: str) -> str:
    parts = str(experiment_id).split(".")
    return parts[-2] if len(parts) >= 2 else "?"


def _find_manifest(out_dir: Path, entry) -> Path | None:
    if entry.manifest:
        p = Path(entry.manifest)
        if p.is_file():
            return p
    for cand in (out_dir / "runs").glob(f"*/{entry.experiment_id.split('.')[-1]}/run_manifest.json"):
        return cand
    hits = sorted((out_dir / "runs").glob(f"*/run_manifest.json"))
    for p in hits:
        if p.parent.name.startswith(entry.experiment_id):
            return p
    return None


#: experiment ids are ``MODEL.SUP.INPUT.FUSION.CAL.PROTOCOL.LEVEL[.variant].BLOCK.sSEED``
_ID_ORDER = ("model_id", "suppression", "input", "fusion", "calibration", "protocol", "level")


def _factors_from_id(experiment_id: str) -> dict:
    """Recover the factor levels from the id alone, for cells without a run directory.

    Only used to *describe* an unrun cell (so the coverage table can group it); no
    metric is ever attributed to a row built this way, because ``usable`` stays False.
    """
    parts = str(experiment_id).split(".")
    if len(parts) < len(_ID_ORDER):
        return {"model_id": parts[0] if parts else "?"}
    f = dict(zip(_ID_ORDER, parts[: len(_ID_ORDER)]))
    rest = parts[len(_ID_ORDER):]
    if rest:
        f["variant"], f["block"] = rest[0], (rest[1] if len(rest) > 1 else "full")
    f["paradigm"] = (ARCH_BY_PARADIGM.get(f["model_id"]) or "?")
    return f


ARCH_BY_PARADIGM: dict[str, str] = {k: str(v.get("paradigm", "?")) for k, v in ARCHITECTURES.items()}


def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _cell_metric(row: dict, key: str = PRIMARY_MACRO) -> float | None:
    m = (row.get("metrics") or {}).get("macro") or {}
    v = m.get(key, m.get("macro") if key == PRIMARY_MACRO else None)
    return None if v is None else float(v)


def aggregate(rows: list[dict], by: str) -> list[dict]:
    """Mean/spread of the primary metric across the levels of one factor."""
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        if not r.get("usable"):
            continue
        key = str((r.get("factors") or {}).get(by) or r.get(by) or "?")
        v = _cell_metric(r)
        if v is not None:
            groups[key].append((r["experiment_id"], v))
    out = []
    for k, vals in sorted(groups.items()):
        arr = np.array([v for _e, v in vals], dtype=np.float64)
        out.append({by: k, "n_cells": len(arr), "mean": float(arr.mean()) if arr.size else None,
                    "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
                    "best_cell": max(vals, key=lambda t: t[1])[0] if arr.size else None,
                    "ids": [e for e, _ in vals]})
    out.sort(key=lambda d: -(d["mean"] or -1))
    return out


def farm_matrix(rows: list[dict], by: str) -> dict[str, dict[str, dict[str, float]]]:
    """level → farm → cell → macro-AUPRC, straight from the recorded per-farm scores.

    The report's only legitimate unit of replication is the farm, and the per-farm
    values exist only because the runner recorded them; nothing here re-derives a
    metric from the raw arrays, so the report and the runner cannot disagree about
    what a number means.
    """
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if not r.get("usable"):
            continue
        lvl = str((r.get("factors") or {}).get(by) or r.get(by) or "?")
        for cid, c in (r.get("per_cell") or {}).items():
            for farm, v in ((c.get("aux") or {}).get("per_farm") or {}).items():
                if v is not None and math.isfinite(float(v)):
                    out[lvl][farm][cid] = float(v)
    return out


def paired_comparisons(rows: list[dict], by: str, *, n_boot: int = 4000) -> list[dict]:
    """Farm-clustered paired bootstrap + Wilcoxon + Cliff's δ for each pair of levels.

    Pairing is on the farm *and* the anomaly cell: two arms evaluated on different
    farms are not comparable, and treating them as if they were is how a benchmark
    manufactures a winner out of sampling noise. Each farm contributes the macro over
    the cells *it* shares between the two levels — a global cell intersection would
    be empty by construction, because a farm carries a cell only where its cycles
    actually received that family/severity's event.

    The interval comes from resampling whole farms (one draw = a set of farms, each
    contributing all of its cells), so the correlation within a farm is carried into
    the interval instead of being assumed away. The p-values are printed next to it,
    not instead of it.
    """
    mat = farm_matrix(rows, by)
    keys = sorted(mat)
    boot_n = max(int(n_boot), MIN_BOOT)
    out: list[dict] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = sorted(set(mat[a]) & set(mat[b]))
            per_farm_cells = {f: sorted(set(mat[a][f]) & set(mat[b][f])) for f in shared}
            per_farm_cells = {f: c for f, c in per_farm_cells.items() if c}
            if not per_farm_cells:
                out.append({"a": a, "b": b, "available": False, "n_farms": len(shared),
                            "reason": "no farm has an anomaly cell in common between the two levels "
                                      "(a Deferred or failed cell removed the pairing)"})
                continue
            farms = sorted(per_farm_cells)
            grid = np.array([[float(np.mean([mat[a][f][c] for c in per_farm_cells[f]])),
                              float(np.mean([mat[b][f][c] for c in per_farm_cells[f]]))]
                             for f in farms], dtype=np.float64)
            n_cells_mean = float(np.mean([len(per_farm_cells[f]) for f in farms]))
            grid = grid[np.isfinite(grid).all(axis=1)]
            if grid.shape[0] < 4:
                out.append({"a": a, "b": b, "available": False, "n_farms": int(grid.shape[0]),
                            "reason": (f"only {int(grid.shape[0])} farms are scoreable for both arms; a "
                                       "two-point paired test would be arithmetic, not evidence")})
                continue
            data = {"grid": grid, "n_cells": n_cells_mean, "n_farms": int(grid.shape[0])}

            def metric_fn(idx, d, _a=a, _b=b):
                g = d["grid"][idx]
                return {_a: float(np.nanmean(g[:, 0])), _b: float(np.nanmean(g[:, 1]))}

            res = _st.paired_cluster_bootstrap(metric_fn, data,
                                               clusters=np.arange(grid.shape[0]), n_boot=boot_n,
                                               seed=int(_BOOT_SEED))
            pair = res.get(f"{a}-minus-{b}") or {}
            w = _st.wilcoxon_paired(grid[:, 0] - grid[:, 1])
            dl = _st.cliffs_delta(grid[:, 0], grid[:, 1])
            out.append({
                "a": a, "b": b, "available": True, "n_farms": int(grid.shape[0]),
                "n_cells_mean_per_farm": round(n_cells_mean, 2),
                "pairing_unit": "(farm, cell), averaged per farm",
                "mean_a": round(float(np.nanmean(grid[:, 0])), 5),
                "mean_b": round(float(np.nanmean(grid[:, 1])), 5),
                "mean_diff": pair.get("mean_diff"), "ci95": pair.get("ci"),
                "excludes_zero": pair.get("excludes_zero"),
                "p_bootstrap": pair.get("p_two_sided"), "bootstrap_draws": pair.get("n_boot"),
                "resampled_clustered_by": "farm", "paired": pair.get("paired"),
                "wilcoxon_p": w.get("p"), "wilcoxon_note": w.get("reason"),
                "wilcoxon_ties": w.get("ties"),
                "cliffs_delta": dl.get("delta"), "cliffs_magnitude": dl.get("magnitude"),
                "prob_a_better": dl.get("prob_greater"),
            })
    ps = {f"{c['a']}|{c['b']}": float(c["wilcoxon_p"]) for c in out
          if c.get("available") and c.get("wilcoxon_p") is not None}
    if ps:
        # Holm–step-down over the declared family: a 5-level sweep makes 10 pairwise
        # claims, and 10 uncorrected 5 % tests find something whether or not it is there.
        adj = _st.holm(ps)
        for c in out:
            got = adj.get(f"{c['a']}|{c['b']}") or {}
            c["p_adjusted_holm"] = got.get("p_adjusted", got.get("p_holm"))
            c["reject_after_holm"] = got.get("reject_at_0.05", got.get("reject"))
    return out


def interaction_table(rows: list[dict], by: str) -> dict:
    """Task 10.7: does the ranking transfer, or is it an interaction talking?

    With ``by`` a model factor (paradigm / backbone / model_id), the (factor, family)
    matrix of per-cell AUPRC asks whether the same factor wins on every anomaly family.
    Kendall's W over the per-family rankings is the summary: W near 1 = the ranking is
    family-robust; W near 0 = the winner is family-specific and an overall rank would
    hide it. The per-family winner is printed beside the table, so an interaction is
    visible without recomputation. A (factor, family) cell with no data is a hole in the
    table, reported as such — never imputed — and W runs over complete rows only.
    """
    from ssl_anomaly.benchmarking.config import FAMILIES

    vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if not r.get("usable"):
            continue
        key = str((r.get("factors") or {}).get(by) or r.get(by) or "?")
        for c in (r.get("per_cell") or {}).values():
            fam = str(c.get("family") or "?")
            a = c.get("auprc")
            if fam in FAMILIES and a is not None and math.isfinite(float(a)):
                vals[(key, fam)].append(float(a))
    fams = [f for f in FAMILIES if any(k[1] == f for k in vals)]
    items = sorted({k[0] for k in vals})
    if len(fams) < 3 or len(items) < 3:
        return {"available": False, "factor": by, "n_families": len(fams), "n_items": len(items),
                "reason": (f"the family interaction needs ≥3 {by} levels × ≥3 families with usable "
                           "cells; an interaction statistic over fewer would be arithmetic, not evidence")}
    mat = np.array([[float(np.mean(vals[(i, f)])) if vals.get((i, f)) else np.nan
                     for f in fams] for i in items], dtype=np.float64)
    counts = np.array([[len(vals.get((i, f), ())) for f in fams] for i in items])
    # W's blocks are the FAMILIES and its items are the factor levels: the question is
    # whether the levels' ranking agrees across families, so the matrix is transposed.
    res = _st.kendall_w(mat.T, names=items)
    res.update({"factor": by, "blocks": fams, "items": items, "matrix": mat.round(5).tolist(),
                "n_per_cell": counts.tolist(), "holes": int((counts == 0).sum()),
                "kind": "factor_x_family",
                "note": (f"rows = {by} levels, blocks = anomaly families; a W below ~0.5 means the "
                         f"{by} ranking changes with the anomaly family — the per-family winner "
                         "table is then the honest leaderboard, not the mean")})
    res["winner_per_block"] = {
        f: (items[int(np.nanargmax(mat[:, j]))] if np.isfinite(mat[:, j]).any() else None)
        for j, f in enumerate(fams)}
    return res


def suppression_deltas(rows: list[dict]) -> dict:
    """Task 10.7: Δ(S_y − S0) per (model, suppression), with sign consistency across models.

    A suppression treatment that helps one paradigm and hurts another is exactly the
    interaction a mean over models hides; the sign consistency column is the honest
    one-line answer to "does S1/S2/S3 help everywhere?".
    """
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if not r.get("usable"):
            continue
        mid = str(r.get("model_id") or "?")
        sup = str((r.get("factors") or {}).get("suppression") or "?")
        v = _cell_metric(r)
        if v is not None:
            per[(mid, sup)].append(float(v))
    models = sorted({k[0] for k in per})
    sups = sorted({k[1] for k in per})
    if not models or "S0" not in sups:
        return {"available": False, "reason": "no usable S0 baseline cell; deltas are undefined"}
    out: dict[str, dict] = {}
    for mid in models:
        base = per.get((mid, "S0"))
        if not base:
            continue
        row = {}
        for s in sups:
            if s == "S0" or not per.get((mid, s)):
                continue
            row[s] = round(float(np.mean(per[(mid, s)])) - float(np.mean(base)), 5)
        if row:
            out[mid] = row
    if not out:
        return {"available": False, "reason": "no model has both an S0 cell and a treated cell"}
    sup_cols = sorted({k for v in out.values() for k in v})
    consistent = {s: (len({d[s] > 0 for d in out.values() if s in d}) == 1) for s in sup_cols}
    return {"available": True, "deltas": out, "n_models": len(out),
            "sign_consistent": consistent,
            "note": ("sign_consistent[suppression]=True means every model moves the same way "
                     "under that treatment; a treatment that helps one paradigm and hurts "
                     "another is exactly the interaction a mean hides")}


def variance_split(rows: list[dict]) -> dict:
    """Seed / farm / anomaly-cell / model, from the recorded per-farm numbers (task 8.7)."""
    vals, sd, fm, cl, md = [], [], [], [], []
    for r in rows:
        if not r.get("usable"):
            continue
        for cid, c in (r.get("per_cell") or {}).items():
            for farm, v in ((c.get("aux") or {}).get("per_farm") or {}).items():
                if v is None or not math.isfinite(float(v)):
                    continue
                vals.append(float(v)); sd.append(int(r["seed"])); fm.append(str(farm))
                cl.append(str(cid)); md.append(str(r["model_id"]))
    if len(vals) < 8:
        return {"available": False, "n": len(vals),
                "reason": ("not enough per-farm values to attribute variance" if vals else
                           "no per-farm records: the run predates farm breakdown, so the "
                           "decomposition cannot be reconstructed from cell means alone")}
    res = _st.variance_decomposition(np.array(vals), seeds=np.array(sd), farms=np.array(fm),
                                     cells=np.array(cl), groups=np.array(md),
                                     labels={"caveat": ("one-way between-group shares over a "
                                                        "non-orthogonal design: the four shares plus the "
                                                        "residual sum to 1, but each share is computed "
                                                        "ignoring the others, so they rank the factors "
                                                        "rather than apportion the effect")})
    if res.get("available"):
        res["n_values"] = int(len(vals))
        res["unit"] = "farm × anomaly cell × seed × arm"
    return res


_BOOT_SEED = 20261123        # fixed so a re-render of the report reproduces every interval


def friedman_table(rows: list[dict], by: str) -> dict:
    """Rank-based omnibus test over a factor's levels, blocked by (farm, cell)."""
    mat = farm_matrix(rows, by)
    levels = sorted(mat)
    if len(levels) < 3:
        return {"available": False, "n_levels": len(levels),
                "reason": f"{len(levels)} levels for {by}: Friedman needs ≥3 to rank anything"}
    blocks: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for lvl, farms in mat.items():
        for farm, cells in farms.items():
            for cid, v in cells.items():
                blocks[(farm, cid)][lvl] = v
    # A block is one (farm, cell) row that *every* level scored; blocks with a hole are
    # dropped rather than imputed, because a rank test on unequal membership is a
    # ranking of whoever happened to be present.
    full = [b for b, d in blocks.items() if all(l in d for l in levels)]
    if len(full) < 10:
        return {"available": False, "n_levels": len(levels), "n_blocks": len(full), "levels": levels,
                "reason": (f"{len(full)} complete (farm, cell) blocks; the χ² approximation needs ≥10, "
                           "and reporting one from fewer would be theatre")}
    table = np.array([[blocks[b][l] for l in levels] for b in sorted(full)], dtype=np.float64)
    res = _st.friedman_nemenyi(table, levels)
    res["levels"] = levels
    res["n_farms"] = len({f for f, _ in full})
    return res


def _fmt(v, nd: int = 4) -> str:
    """One place to decide how a missing number looks: an em dash, never a zero.

    A zero in a results table is a claim; a dash is an admission. The report needs to
    be able to tell them apart at a glance, which is why the formatting is centralised.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f:+.4f}" if nd == 4 else f"{f:.{nd}f}"


def _round(v, nd=5):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def build_report(out_dir: Path, cfg, cells_matrix: list, *, n_boot: int = 4000,
                 title: str = "Architecture benchmark — run report") -> ReportArtifacts:
    out_dir = Path(out_dir)
    rows = load_cells(out_dir)
    usable = [r for r in rows if r.get("usable")]
    warns: list[str] = []
    L: list[str] = [f"# {title}", ""]

    # -- 0. provenance ----------------------------------------------------
    stamp = next((r["manifest"].get("stamp") for r in rows if r.get("manifest")), None)
    L += ["## 0. What this document is", "",
          f"- run directory: `{out_dir}`",
          f"- benchmark config SHA-256: `{cfg.document_sha256}`",
          f"- cells in the committed matrix: **{len(cells_matrix)}**; with a registry record: "
          f"**{len(rows)}**; with a readable metrics payload: **{len([r for r in rows if r.get('metrics')])}**; "
          f"entering the tables below: **{len(usable)}**",
          ""]
    if stamp:
        L += [f"> **{stamp}**", ">",
              "> Every number below was produced on fixture tensors. Nothing here may be quoted "
              "as a result; the file exists to prove the pipeline runs and to fix the wording of the "
              "real report before the compute is spent.", ""]
        warns.append("fixture/smoke run: no scientific claim in this document")
    rung = cfg.raw["evaluation"].get("rung_ceiling", "L3")
    L += ["The validation ladder rung claimed by this document is the one recorded in each manifest "
          f"(ceiling in the contract: `{rung}`). A winner claim needs L4 (multi-seed, pre-registered "
          "stats, complete coverage); a generalisation claim needs L5 (independent study).", ""]

    # -- 1. coverage ------------------------------------------------------
    fixture_only = any(r.get("fixture") for r in usable)
    counts = status_counts(cells_matrix)
    L += ["## 1. Coverage, stated before any result", "",
          f"- {coverage_line(counts, total=len(cells_matrix))}", ""]
    for st in REGISTRY_STATES:
        n = int(counts.get(st, 0))
        if n:
            reasons = sorted({str((c.reason or "—")) for c in cells_matrix if c.status == st})[:4]
            L.append(f"- **{st}**: {n}" + (f" — reasons: {'; '.join(reasons)}" if st != "Completed" else ""))
    L += ["", "A cell that was never run is *absent*, not *unremarkable*. No table below imputes a "
          "value into a Deferred, Not Applicable or Failed cell, and no ranking in this document may "
          "be described as complete while any Planned cell is missing.", ""]

    if fixture_only:
        L += [f"**Every number in §§2–7 below is a fixture number** (rung L0). They are printed "
              "so the statistical path is visibly exercised — the same code that will render the "
              "real result — and so that a difference of the size the design expects can be seen to "
              "be *un*detectable here. Nothing in §§2–7 is a finding.", ""]
    if not usable:
        L += ["## 2–8. Results", "",
              "**Not available.** No cell in this run directory reached `Completed` with a readable "
              "metrics payload, so there is nothing to rank. See §1 for what ran and why the rest did "
              "not; the failure reasons are reproduced verbatim below.", ""]
        for r in rows:
            if r["status"] in ("Failed", "Interrupted"):
                L.append(f"- `{r['experiment_id']}` — {r['reason'] or r['error'] or 'no reason recorded'}")
        p = out_dir / "REPORT.md"
        p.write_text("\n".join(L) + "\n", encoding="utf-8")
        return ReportArtifacts(p, L, warns + ["no completed cell: the report contains no results"])

    # -- 2. primary metric per factor ------------------------------------
    fx = " *(fixture numbers)*" if fixture_only else ""
    L += [f"## 2. Primary metric (macro-AUPRC, equal weight over the 15 synthetic cells){fx}", "",
          "Every mean below is over the cells that actually scored; `prevalence` is the injected "
          "event rate the metric is conditioned on, because macro-AUPRC at 0.4 % prevalence and at "
          "4 % are different claims.", "",
          "### 2.1 Paradigm", "", _table(aggregate(usable, "paradigm"), ("paradigm", "n_cells", "mean", "sd")),
          "", "### 2.2 Backbone", "", _table(aggregate(usable, "backbone"), ("backbone", "n_cells", "mean", "sd")),
          "", "### 2.3 Suppression", "", _table(aggregate(usable, "suppression"),
                                                 ("suppression", "n_cells", "mean", "sd")),
          "", "### 2.4 Input / fusion / calibration / protocol / level", ""]
    for f in ("input", "fusion", "calibration", "protocol", "level", "variant"):
        agg = aggregate(usable, f)
        if agg:
            L += ["", f"**{f}**", "", _table(agg, (f, "n_cells", "mean", "sd"))]
    L.append("")

    # -- 3. three winner reports -----------------------------------------
    for name, factor, extra in (("3.1 Representation paradigm", "paradigm",
                                 "the thesis question: what the objective is trained to predict"),
                                ("3.2 Backbone at matched capacity", "backbone",
                                 "capacity-matched by `capacity.target_params`; a backbone winner that "
                                 "is also a capacity difference is not a backbone winner"),
                                ("3.3 Suppression, per paradigm", "suppression",
                                 "reported inside each paradigm, because S1/S2/S3 act on different "
                                 "quantities per objective")):
        L += [f"## {name}", "", f"*{extra}.*", ""]
        if factor == "suppression":
            for para in sorted({str((r.get("factors") or {}).get("paradigm")) for r in usable}):
                sub = [r for r in usable if str((r.get("factors") or {}).get("paradigm")) == para]
                L += [f"**{para}**", "", _table(aggregate(sub, "suppression"),
                                                 ("suppression", "n_cells", "mean", "sd")), ""]
            L.append("_No cross-paradigm winner is claimed for suppression: the comparisons are not "
                     "on the same objective, so ranking them together would compare apples to "
                     "armchairs with a bootstrap attached._")
        else:
            L += [_table(aggregate(usable, factor), (factor, "n_cells", "mean", "sd")), ""]
            comps = paired_comparisons(usable, factor, n_boot=n_boot)
            L += ["**Paired comparisons** (farm-clustered bootstrap, B≥"
                  f"{MIN_BOOT}; Wilcoxon; Cliff's δ; Holm-adjusted within the family)", "",
                  _comp_table(comps)]
            fr = friedman_table(usable, factor)
            L += ["", "**Omnibus** (Friedman, blocked by cell × seed, with Nemenyi where computable):",
                  "", _friedman_line(fr), ""]
        L.append("")

    # -- 3.4 interaction analyses (task 10.7) ------------------------------
    L += ["## 3.4 Interaction — does the ranking transfer across anomaly families, and does "
          "suppression help everywhere?", "",
          "_Two questions a single leaderboard cannot answer. The matrices come from the recorded "
          "per-cell values; Kendall's W summarises rank agreement across blocks (W≈1: the same "
          "winner everywhere; W≈0: the ranking is block-specific and the per-block winner is the "
          "honest leaderboard). A cell with no data is a hole, printed as such._", ""]
    interactions: dict[str, dict] = {}
    for label, factor in (("paradigm × family", "paradigm"), ("backbone × family", "backbone"),
                          ("model × family", "model_id")):
        res = interaction_table(usable, factor)
        interactions[label] = res
        L += [f"**{label}**", ""]
        if not res.get("available"):
            L += [f"_Not available_: {res.get('reason')}", ""]
            continue
        head = "| " + " | ".join([res["factor"]] + res["blocks"]) + " |"
        sep = "|" + "|".join(["---"] + ["---:"] * len(res["blocks"])) + "|"
        L += [head, sep]
        display_order = sorted(res["items"], key=lambda k: -float(res["mean_rank"][k]))
        for item in display_order:
            i = res["items"].index(item)
            row = res["matrix"][i]
            L.append("| " + " | ".join([item] + [
                _fmt(row[j], 3) if np.isfinite(row[j]) else "—" for j in range(len(res["blocks"]))]) + " |")
        L += ["", f"W = **{res.get('w')}** (χ²={res.get('chi2')}, df={res.get('df')}, "
                  f"p={res.get('p')}, blocks={res.get('n_blocks')}, holes={res.get('holes')})",
              "Winners per family: " + ", ".join(
                  f"{f} → {w}" for f, w in (res.get('winner_per_block') or {}).items()), ""]
    sd = suppression_deltas(usable)
    interactions["suppression_deltas"] = sd
    L += ["**model × suppression — Δ(S − S0) of macro-AUPRC**", ""]
    if not sd.get("available"):
        L += [f"_Not available_: {sd.get('reason')}", ""]
    else:
        cols = sorted({k for v in sd["deltas"].values() for k in v})
        L += ["| model | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        for mid, d in sorted(sd["deltas"].items()):
            L.append(f"| {mid} | " + " | ".join(_fmt(d.get(c), 4) if c in d else "—"
                                                for c in cols) + " |")
        L += ["", "Sign consistency: " + ", ".join(
            f"{s}: {'consistent' if ok else 'MIXED'}" for s, ok in sd["sign_consistent"].items()), ""]
    fixture_int = fixture_only
    if fixture_int:
        L += ["_These interaction numbers are fixture numbers (see §0): they exercise the code path "
              "and carry no claim._", ""]

    # -- 4. secondary metrics --------------------------------------------
    L += ["## 4. Secondary metrics (never used to pick a winner)", "",
          _secondaries(usable), ""]

    # -- 5. variance decomposition ---------------------------------------
    vs = variance_split(usable)
    L += ["## 5. Variance decomposition (is a single-seed ranking meaningful?)", ""]
    if not vs.get("available"):
        L += [f"_Not available_: {vs.get('reason')}", ""]
    else:
        L += ["| factor | share of variance | note |", "|---|---:|---|"]
        for k, v in (vs.get("shares") or {}).items():
            L.append(f"| {k} | {float(v):.3f} | {'between-group one-way share' if k != 'residual' else 'everything else, stated rather than hidden'} |")
        L += ["", f"_method_: {vs.get('method')}", "", f"_caveat_: {vs.get('caveat')}", ""]
        if float((vs.get("shares") or {}).get("seed", 0)) > float((vs.get("shares") or {}).get("model", 0)):
            warns.append("seed variance exceeds model variance: per-seed tables are required before any ranking claim")
            L += ["", "> ⚠️ Seed-to-seed variance is larger than model-to-model variance in this run. Any "
                  "ranking here is at least as much a statement about seeds as about architectures.", ""]

    # -- 6. cost ----------------------------------------------------------
    L += ["## 6. Computational cost (part of the answer, not a footnote)", "",
          _cost_table(usable), ""]

    # -- 7. questions ledger ---------------------------------------------
    L += ["## 7. The 14 questions, answered or declared unanswerable", ""]
    for qid, q in QUESTIONS:
        status, basis = _answer(qid, rows, usable, vs, cfg)
        L += [f"**{qid}.** {q}", "", f"- status: **{status}**", f"- basis: {basis}", ""]

    # -- 8. what would change the conclusion -----------------------------
    L += ["## 8. What would change these conclusions", "",
          _next_steps(cfg, rows, counts), ""]
    p = out_dir / "REPORT.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(
        {"coverage": counts, "usable": len(usable), "aggregate": {f: aggregate(usable, f) for f in
                                                                   ("paradigm", "backbone", "suppression",
                                                                    "input", "fusion", "calibration")},
         "interactions": interactions, "variance": vs, "warnings": warns},
        indent=1, sort_keys=True, default=str), encoding="utf-8")
    return ReportArtifacts(p, L, warns)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _table(rows: list[dict], cols: tuple[str, ...]) -> str:
    if not rows:
        return "_no usable cells for this factor_"
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _comp_table(comps: list[dict]) -> str:
    """One row per level pair; every statistic labelled, every absence explained.

    The column order is deliberate: the farm count comes before any effect size, because
    a comparison over four farms is not the same kind of statement as one over forty,
    and a reader must see that before they see a p-value.
    """
    head = ("| a | b | farms | cells/farm | mean a | mean b | Δ mean | 95 % CI (farms) | "
            "excl. 0 | p (boot) | P(a>b) | Wilcoxon p | Holm p | Cliff's δ |\n"
            "|---|---|---:|---:|---:|---:|---:|---|:--:|---:|---:|---:|---:|---:|")
    lines = [head]
    for c in comps:
        if not c.get("available"):
            lines.append(f"| {c['a']} | {c['b']} | {c.get('n_farms', 0)} | — | — | — | — | — | — | — | — |"
                         f" — | — | _{c.get('reason')}_ |")
            continue
        ci = c.get("ci95") or [None, None]
        lines.append(
            f"| {c['a']} | {c['b']} | {c.get('n_farms')} | {c.get('n_cells_mean_per_farm')} | "
            f"{_fmt(c.get('mean_a'), 3)} | {_fmt(c.get('mean_b'), 3)} | {_fmt(c.get('mean_diff'))} | "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {'✅' if c.get('excludes_zero') else '—'} | "
            f"{_fmt(c.get('p_bootstrap'), 4)} | {_fmt(c.get('prob_a_better'), 3)} | "
            f"{_fmt(c.get('wilcoxon_p'), 4)} | {_fmt(c.get('p_adjusted_holm'), 4)} | "
            f"{_fmt(c.get('cliffs_delta'), 3)} ({c.get('cliffs_magnitude')}) |")
    if len(lines) == 2:
        return "_no pairs share enough farms to compare_"
    note = ("\n\n_Δ mean is a mean of per-farm differences, so the interval is a farm-level interval; "
            "`p (boot)` is the two-sided sign fraction of the same resample (floor 1/B), and P(a>b) is "
            "the probability that a resampled farm-set favours a — the two answer different questions and "
            "are printed side by side rather than merged. Wilcoxon is over the same per-farm differences; "
            "Holm corrects within the factor's own family._")
    return "\n".join(lines) + note


def _friedman_line(fr: dict) -> str:
    if not fr.get("available"):
        return f"_Not run_: {fr.get('reason')}"
    cd = fr.get("nemenyi_critical_difference")
    ranks = fr.get("mean_rank") or {}
    # scipy ranks the *smallest* value as 1, so a higher mean rank is a better score.
    # Stated because the opposite convention is the common off-by-one in rank tables,
    # and because `order` from the helper is best-first only under this reading.
    order = fr.get("order") or sorted(ranks, key=lambda k: -ranks[k])
    sig = fr.get("pairs_significant") or []
    return (
        f"Friedman χ²={fr.get('friedman_chi2')} (df={fr.get('df')}), p={fr.get('p')}; "
        f"Nemenyi critical difference={cd if cd is not None else 'not computed'} over "
        f"{fr.get('n_blocks')} complete (farm, cell) blocks across {fr.get('n_farms')} farms.\n\n"
        "| level | mean rank |\n|---|---:|\n"
        + "\n".join(f"| {k} | {ranks.get(k, float('nan')):.2f} |" for k in order)
        + f"\n\n_Higher mean rank = better AUPRC; ranks are within-block, so {fr.get('note')}._\n\n"
        + ("Pairs separated at the Nemenyi critical difference: "
           + ", ".join(f"{q['a']} vs {q['b']} (Δrank {q['rank_diff']:.2f})" for q in sig)
           if sig else "No pair separates at the Nemenyi critical difference.")
    )


_SECONDARY = ("auroc", "f1", "recall_at_budget", "precision_at_budget", "median_delay_days",
              "spearman_kendall", "brier", "ece", "cost_at_budget")


def _secondaries(rows: list[dict]) -> str:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        lvl = str((r.get("factors") or {}).get("paradigm") or "?")
        for k in _SECONDARY:
            v = (r.get("metrics") or {}).get("macro", {}).get(k)
            if v is None:
                for pc in (r.get("per_cell") or {}).values():
                    vv = pc.get(k)
                    if vv is not None and math.isfinite(float(vv)):
                        acc[k][lvl].append(float(vv))
            elif math.isfinite(float(v)):
                acc[k][lvl].append(float(v))
    lines = ["| metric | " + " | ".join(sorted(acc["auprc"]) if acc.get("auprc") else ["—"]) + " |",
             "|---" * (1 + (len(acc["auprc"]) if acc.get("auprc") else 1)) + "|"]
    if len(lines[0].split("|")) < 3:
        return "_no secondary metrics recorded on a common set of levels_"
    for k in _SECONDARY:
        if k not in acc or not acc[k]:
            lines.append(f"| {k} | " + " | ".join("—" for _ in lines[0].split("|")[1:-1]) + " |")
            continue
        lv = sorted(acc[k])
        vals = [f"{np.mean(acc[k][x]):.4f}" if acc[k][x] else "—" for x in lv]
        lines.append(f"| {k} ({'/'.join(lv)}) | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _cost_table(rows: list[dict]) -> str:
    lines = ["| model | params | MFLOP/batch | train wall s | peak VRAM MB | latency ms/cycle | device |",
             "|---|---:|---:|---:|---:|---:|---|"]
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[str(r["model_id"])].append(r)
    for mid, rs in sorted(by.items()):
        r = rs[0]
        c = r.get("cost") or {}
        f = c.get("flops") or {}
        lines.append(
            f"| {mid} | {(c.get('params') or {}).get('trainable_params', '—')} | "
            f"{_round((f.get('forward_flops_per_batch') or 0) / 1e6, 1)} | "
            f"{_round(np.mean([x.get('wall_seconds', 0.0) for x in rs]), 2)} | "
            f"{_round(((c.get('vram') or {}).get('peak_allocated_mb')), 0)} | "
            f"{_round(c.get('inference_latency_ms_per_cycle'), 2)} | {c.get('device') or '—'} |")
    lines.append("")
    lines.append("_FLOPs use the contract's convention (`capacity.flops_convention`); wall time is the "
                 "run's own measurement on the device named in the last column, so the Mamba numbers are "
                 "not comparable to a T4 measurement and are labelled as what they are._")
    return "\n".join(lines)


def _answer(qid: str, rows: list[dict], usable: list[dict], vs: dict, cfg=None) -> tuple[str, str]:
    """Answer a question only if the data to answer it is in this directory."""
    n = len(usable)
    paras = {str((r.get("factors") or {}).get("paradigm")) for r in usable}
    supps = {str((r.get("factors") or {}).get("suppression")) for r in usable}
    seeds = {int(r["seed"]) for r in usable}
    def has(*xs):
        return all(x in supps for x in xs)
    if qid in ("Q1",):
        ok = {"input_reconstruction", "future_prediction"} <= paras
        return (("answered within this run" if ok and n >= 8 else "insufficient evidence"),
                f"{len(paras)} paradigms present, {n} usable cells, seeds {sorted(seeds)}")
    if qid in ("Q2", "Q3", "Q4"):
        need = {"latent_reconstruction"} if qid == "Q2" else (
            {"jepa"} if qid == "Q3" else {"hybrid"})
        ok = need <= paras and n >= 8
        extra = ""
        if qid == "Q3":
            cl = [r for r in rows if r["model_id"] in ("A7", "A8") and r["status"] == "Failed"]
            extra = (f"; {len(cl)} JEPA cell(s) were disqualified by the collapse gate, which is itself "
                     "part of the answer") if cl else ""
        return (("answered within this run" if ok else "insufficient evidence"),
                f"paradigms present: {sorted(paras)}; cells: {n}{extra}")
    if qid in ("Q5", "Q6"):
        back = sorted({str((r.get("factors") or {}).get("backbone")) for r in usable})
        ok = {"gru", "mamba"} <= set(back) and n >= 8
        if qid == "Q5":
            cap = ("every arm in band" if all((r.get("capacity") or {}).get("in_band") for r in usable)
                   else "some arms sit outside the capacity band, so §2.2 is a joint statement")
            return (("answered within this run" if ok else "insufficient evidence"),
                    f"backbones present: {back}; {cap}; matched-token budget per `search.budgets`")
        recs = [(r["model_id"], (r.get("cost") or {}).get("feasibility") or {}) for r in usable]
        with_ratio = [x for x in recs if x[1].get("ratio") is not None]
        if with_ratio:
            worst = max(with_ratio, key=lambda x: float(x[1]["ratio"]))
            return ("answered within this run",
                    f"worst measured ratio is {worst[1]['ratio']:.2f}× for {worst[0]} on "
                    f"{worst[1].get('device')} against the {worst[1].get('budget_factor')}× budget")
        if any(x[1].get("not_applicable") for x in recs):
            why = sorted({str(x[1].get("not_applicable")) for x in recs if x[1]})
            return ("not measurable in this run",
                    f"the gate abstained ({', '.join(why)}): the budget is a statement about the declared "
                    "accelerator, and a ratio measured elsewhere is recorded as an abstention instead of "
                    "a verdict")
        return ("insufficient evidence", "no feasibility record in this run directory")

    if qid in ("Q7", "Q8", "Q9"):
        need = {"S0", "S2"} if qid == "Q7" else ({"S0", "S1"} if qid == "Q8" else {"S0", "S3"})
        ok = has(*need) and n >= 8
        return (("answered within this run" if ok else "insufficient evidence"),
                f"suppressions present: {sorted(supps)}; per-paradigm table in §3.3")
    if qid == "Q10":
        return (("answered within this run" if vs.get("available") else "insufficient evidence"),
                (f"shares: " + ", ".join(f"{k}={v:.2f}" for k, v in (vs.get("shares") or {}).items())
                 if vs.get("available") else str(vs.get("reason"))))
    if qid == "Q11":
        cals = sorted({str((r.get("factors") or {}).get("calibration")) for r in usable})
        ok = len(cals) >= 2
        basis = (f"calibration levels compared: {cals}; aggregation is not a runner factor — the "
                 "alternative aggregations are re-computations of `bin_scores.npz` (see bundle.py), so "
                 "the sensitivity analysis is only as complete as those bundles")
        return (("partially answered" if ok else "insufficient evidence"), basis)
    if qid in ("Q12",):
        ch = [r for r in usable if (r.get("run_dir")) and
              (Path(r["run_dir"]) / "bin_scores.npz").is_file()]
        return ((f"instrumented, not yet answered: {len(ch)} cells store per-channel error; "
                 "the interpretation section is written by hand from those arrays"
                 if ch else "insufficient evidence"),
                "per-channel error tensors are the raw material; a machine cannot supply the agronomy")
    if qid == "Q13":
        fusions = {str((r.get("factors") or {}).get("fusion")) for r in usable}
        return (("answered within this run" if {"W0", "W1"} <= fusions else "insufficient evidence"),
                f"fusion levels present: {sorted(fusions)}; W2 gate drift is recorded in each manifest's "
                "input block when the covariate sidecar exists")
    if qid == "Q14":
        prereg = {int(x) for x in (cfg.seeds if cfg is not None else ())}
        ok = len(seeds) >= 3 and (not prereg or seeds >= prereg)
        return (("answered within this run" if ok else "insufficient evidence"),
                f"seeds represented: {sorted(seeds)}; pre-registered set: {sorted(prereg) or 'unknown'}; "
                "the per-seed spread is in §2 and the seed share of variance in §5")
    return ("unclassified", "no rule for this question id")


def _next_steps(cfg, rows: list[dict], counts: dict) -> str:
    pend_tiers = {r.get("tier") for r in rows if r.get("status") == "Pending"}
    # the suggested command must resume where the *pending* cells are, not advertise the
    # most expensive tier the matrix happens to contain
    tier_word = "screening" if not pend_tiers or "screening" in pend_tiers else sorted(pend_tiers)[0]
    block_word = "screening" if tier_word == "screening" else "full_benchmark"
    seeds = ",".join(str(x) for x in (cfg.raw.get("seeds") or PREREGISTERED_SEEDS))
    lines: list[str] = []
    defer = [r for r in rows if r["status"] == "Deferred"]
    na = [r for r in rows if r["status"] == "Not Applicable"]
    fail = [r for r in rows if r["status"] == "Failed"]
    pend = [r for r in rows if r["status"] == "Pending"]
    lines.append(
        f"- **{len(pend)} planned cells** are unrun. The command is "
        f"`python scripts/benchmarks/run_benchmark.py --execute --yes --tier {tier_word} "
        f"--block {block_word} --seeds {seeds}` and the budget "
        f"(`{cfg.raw['budget']['t4_hours']} h` of T4) decides how much of the plan fits per launch — "
        "the run defers the rest with a recorded reason instead of pretending it finished.")
    reasons = sorted({str(r.get("reason") or "") for r in defer})
    lines.append(f"- **{len(defer)} deferred** ({', '.join(x for x in reasons if x) or 'no reason recorded'}): "
                 "these need the artifact, not a code change — bind the store's hash in "
                 "`configs/benchmarks/benchmark_config.yaml` under an `amendments:` entry, re-derive the "
                 "matrix, and the cells become Pending on their own.")
    lines.append(f"- **{len(na)} not applicable** are *design* statements (see the applicability matrix). "
                 "They must stay out of the winner tables; they are not failures and must not be reported "
                 "as such.")
    lines.append(f"- **{len(fail)} failed**: reasons are reproduced verbatim in §1; a requeue requires a "
                 "recorded config change, and the registry refuses a silent retry.")
    if any("collapse" in str(r.get("reason")).lower() for r in fail):
        lines.append("- collapse disqualifications are results, not bugs: report which arms collapsed and "
                     "at which epoch, and only then consider whether the mechanism "
                     "(`collapse.mechanism`) needs an amendment.")
    lines.append("- before any finding is written, check §0's rung claim: L3 supports 'within this study', "
                 "nothing stronger.")
    return "\n".join(lines)
