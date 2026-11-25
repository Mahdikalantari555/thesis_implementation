"""Representation-first pilot on the mini dataset.

Produces three things from real artifacts only:
1. CYSP-RG — Cross-Year Same-Phase Retrieval Gain (frozen SSL vs raw vs random init)
   with same-farm/different-stage and different-farm/same-stage hard negatives.
2. A simple model (logistic regression) trained on automatically suspected
   stress patch-windows vs matched raw-feature and B1-style baselines.
3. Collapse diagnostics (effective rank, per-dim std, mean pairwise cosine).

Split (accepted 2026-11-25): train <= 2022, dev 2023, test >= 2024. The dev
year is excluded from retrieval and probes. Fixes to the 2026-11-25 audit:
chance level and a label-permutation null are reported (the old
"shuffled-year" control used train years as queries and reported R@1=0.999);
NaN is never turned into an observed zero (an explicit observation mask is a
model input); patches are sampled on the 5x5 label grid and spread over the
field; windows that cross a split boundary are dropped; the logistic probe is
fit with a converged solver and a farm x year cluster-bootstrap CI. Results
are L1/L2 screening evidence only, never stress validation.

Usage::

    python scripts/probe_mini_suspected_stress.py
    python scripts/probe_mini_suspected_stress.py --epochs 8 --max-patches 80
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.mini_lab.models import build_model  # noqa: E402
from ssl_anomaly.utils.seeds import seed_everything  # noqa: E402
from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402
from ssl_anomaly.utils.paths import localize  # noqa: E402

DATA = localize("D:/RS/Thesis/Data/Processed/mini_research_dataset")
STORE = DATA / "satellite.zarr"
PATCH_LABELS = DATA / "suspected_stress" / "suspected_patches.parquet"
WEATHER = DATA / "weather" / "weather_aligned_mini.parquet"
FEATURES = DATA / "weather" / "weather_features.json"
DEFAULT_OUT = ROOT / "reports" / "mini_dataset" / "representation_first_pilot.json"


class _WindowEncoder(nn.Module):
    """Small shared trunk: per-timestep MLP + GRU -> embedding + recon head."""

    def __init__(self, d_in: int, d_model: int = 32, latent: int = 16):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.to_latent = nn.Linear(d_model, latent)
        self.from_latent = nn.Linear(latent, d_model)
        self.out_proj = nn.Linear(d_model, d_in)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.in_proj(x))
        out, _ = self.gru(h)
        z = self.to_latent(out[:, -1])
        dec = self.from_latent(z).unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.out_proj(dec), z


def _auprc(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision by descending-score ranking (no BLAS dependency)."""
    order = np.argsort(-scores, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float(precision[ys == 1].mean()) if ys.sum() > 0 else float("nan")


def _logistic_probe(
    features: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    groups: np.ndarray | None = None,
    test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Converged logistic probe + farm x year cluster-bootstrap CI.

    The previous 200-step hand-rolled gradient descent did not converge and
    reversed the arm ordering under a converged solver (audit 2026-11-25 §3).
    """
    from sklearn.linear_model import LogisticRegression

    evaluation = ~train if test is None else test
    xtr, ytr = features[train], y[train]
    xte, yte, te_idx = features[evaluation], y[evaluation], np.flatnonzero(evaluation)
    clf = LogisticRegression(max_iter=3000, solver="lbfgs", C=1.0)
    clf.fit(xtr, ytr)
    prob = clf.predict_proba(xte)[:, 1]
    result: dict[str, Any] = {
        "auprc": round(_auprc(yte, prob), 4),
        "prevalence": round(float(yte.mean()), 4),
        "n_test": int(len(yte)),
        "n_positive_test": int(yte.sum()),
        "solver": "sklearn.LogisticRegression(lbfgs, max_iter=3000, C=1.0)",
        "converged": bool(clf.n_iter_[0] < 3000),
    }
    if groups is not None and len(yte) > 0:
        gte = groups[te_idx]
        uniq = np.unique(gte)
        rng = np.random.default_rng(13)
        scores: list[float] = []
        for _ in range(500):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(gte == g) for g in pick])
            if len(np.unique(yte[idx])) < 2:
                continue
            s = _auprc(yte[idx], prob[idx])
            if np.isfinite(s):
                scores.append(s)
        if scores:
            lo, hi = np.percentile(scores, [2.5, 97.5])
            result["ci95"] = [round(float(lo), 4), round(float(hi), 4)]
            result["n_bootstrap_clusters"] = int(len(uniq))
    return result


def _collapse(z: np.ndarray) -> dict[str, float]:
    zc = z - z.mean(0, keepdims=True)
    cov = np.einsum("ni,nj->ij", zc, zc, optimize=False) / max(1, len(z) - 1)
    # Participation-ratio effective rank: (tr C)^2 / tr(C^2). Avoids LAPACK,
    # which crashes natively on this Windows build for repeated calls.
    tot = float(np.trace(cov))
    tr_sq = float(np.trace(np.einsum("ij,jk->ik", cov, cov, optimize=False)))
    effective_rank = float(tot**2 / tr_sq) if tr_sq > 0 else 0.0
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    # Mean off-diagonal cosine of Z Z^T without materialising the n×n matrix.
    n = len(z)
    mean_cos = (float(np.sum(zn) ** 2) - n) / (n * (n - 1)) if n > 1 else 0.0
    return {
        "effective_rank": round(effective_rank, 3),
        "mean_pairwise_cosine": round(mean_cos, 4),
        "per_dim_std_min": round(float(z.std(0).min()), 5),
        "per_dim_std_max": round(float(z.std(0).max()), 5),
    }


TRAIN_YEAR_END = 2022
DEV_YEAR = 2023
TEST_YEAR_START = 2024


def _cysp_rg_raw(
    e: np.ndarray, farms: np.ndarray, stages: np.ndarray, years: np.ndarray, n_perm: int = 200
) -> dict[str, Any]:
    """Corrected negative control for cross-year same-phase retrieval.

    The old version set ``train, test = perm_years < 2023, years < 2023``:
    queries came from TRAIN years and could be their own candidates, yielding
    R@1 = 0.999. Here queries are test years (>= 2024), candidates are train
    years (<= 2022), and the null is a permutation of candidate (farm, stage)
    labels. Both observed R@1 and the null are reported with chance level.
    """
    cand = np.flatnonzero(years <= TRAIN_YEAR_END)
    queries = np.flatnonzero(years >= TEST_YEAR_START)
    if cand.size < 2 or queries.size == 0:
        return {"r_at_1": None, "chance_level": None, "null_mean": None,
                "null_p97_5": None, "n_queries": 0, "n_permutations": n_perm}
    tops: list[int] = []
    chance = 0.0
    for i in queries:
        sims = np.sum(e[cand] * e[i], axis=1)
        tops.append(int(np.argmax(sims)))
        chance += float(np.mean((farms[cand] == farms[i]) & (stages[cand] == stages[i])))
    cand_farms, cand_stages = farms[cand], stages[cand]
    observed = float(np.mean([
        (cand_farms[t] == farms[i] and cand_stages[t] == stages[i])
        for i, t in zip(queries, tops)
    ]))
    rng = np.random.default_rng(13)
    null: list[float] = []
    for _ in range(n_perm):
        pf = rng.permutation(cand_farms)
        ps = rng.permutation(cand_stages)
        null.append(float(np.mean([
            (pf[t] == farms[i] and ps[t] == stages[i]) for i, t in zip(queries, tops)
        ])))
    return {
        "r_at_1": round(observed, 4),
        "chance_level": round(chance / len(queries), 4),
        "null_mean": round(float(np.mean(null)), 4),
        "null_p97_5": round(float(np.percentile(null, 97.5)), 4),
        "n_queries": int(len(queries)),
        "n_permutations": n_perm,
    }


def _cysp_rg(z: np.ndarray, raw: np.ndarray, farms: np.ndarray, stages: np.ndarray, years: np.ndarray) -> dict[str, Any]:
    """R@1 with hard negatives; query = test year (>=2024), candidates = train (<=2022)."""
    out: dict[str, Any] = {}
    for name, emb in (("ssl", z), ("raw", raw)):
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        correct, total, chance = 0, 0, 0.0
        for i in np.flatnonzero(years >= TEST_YEAR_START):
            cand = np.flatnonzero(years <= TRAIN_YEAR_END)
            if cand.size < 2:
                continue
            sims = np.sum(e[cand] * e[i], axis=1)
            top = cand[int(np.argmax(sims))]
            correct += int(farms[top] == farms[i] and stages[top] == stages[i])
            chance += float(np.mean((farms[cand] == farms[i]) & (stages[cand] == stages[i])))
            total += 1
        out[name] = {
            "r_at_1": round(correct / total, 4) if total else None,
            "n_queries": int(total),
            "chance_level": round(chance / total, 4) if total else None,
        }
    out["gain"] = (
        round((out["ssl"]["r_at_1"] or 0) - (out["raw"]["r_at_1"] or 0), 4)
        if out["ssl"]["r_at_1"] is not None and out["raw"]["r_at_1"] is not None
        else None
    )
    # Legacy key kept for compatibility; the value is now the corrected
    # label-permutation null mean, not the broken shuffled-year R@1.
    control = _cysp_rg_raw(
        z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8), farms, stages, years
    )
    out["shuffled_year_control"] = control["null_mean"]
    out["null_control"] = control
    out["control_definition"] = (
        "candidate (farm, stage) label permutation; queries = years >= 2024, "
        "candidates = years <= 2022; dev year 2023 excluded"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-patches", type=int, default=80)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from numpy.lib.stride_tricks import sliding_window_view
    import zarr

    weather = pd.read_parquet(WEATHER)
    weather["date_key"] = weather["acquisition_date"].astype(str).str.slice(0, 10)
    groups = json.loads(FEATURES.read_text())["groups"]["weather_core"]
    weather = weather.set_index("date_key")
    weather_map = {
        str(d): weather.loc[d, groups].to_numpy(dtype=np.float32) for d in weather.index
    }

    patches = pd.read_parquet(PATCH_LABELS)
    stage_map = {
        (r.farm, r.date): r.stage
        for r in patches.itertuples(index=False)
    }
    label_map = {
        (r.farm, r.date, r.patch_row * 5, r.patch_col * 5): r.label
        for r in patches.itertuples(index=False)
    }

    X: list[np.ndarray] = []
    OBS: list[np.ndarray] = []
    raw_sum: list[np.ndarray] = []
    farm_ids: list[str] = []
    stage_ids: list[str] = []
    years: list[int] = []
    y_susp: list[int] = []
    n_crossing = 0
    for d in sorted(p for p in STORE.iterdir() if p.is_dir() and (p / ".zgroup").exists()):
        g = zarr.open_group(str(d), mode="r")
        farm = str(g.attrs["farm_id"])
        t = g.attrs["grid_transform"]
        row0 = int(round((3443010.0 - float(t[5])) / 30.0))
        col0 = int(round((float(t[2]) - 233010.0) / 30.0))
        dates = [str(x) for x in g["dates"][:]]
        cube = np.asarray(g["data"][:], dtype=np.float32)
        valid = np.asarray(g["pixel_valid"][:], dtype=bool)
        in_field = np.asarray(g["in_field"][:], dtype=bool)
        # Patches on the 5x5 label grid, spread over the whole field. The old
        # stride-1 `coords[:80]` took only top rows and was misaligned with the
        # label tiles (audit 2026-11-25 §6.2/6.3).
        h, w = in_field.shape
        # Offset by the global tile phase so each patch covers exactly one
        # 5x5 label tile (row0/col0 are not multiples of 5 for these farms).
        r0tile, c0tile = (-row0) % 5, (-col0) % 5
        grid = [
            (r, c)
            for r in range(r0tile, max(1, h - 4), 5)
            for c in range(c0tile, max(1, w - 4), 5)
        ]
        if len(grid) > args.max_patches:
            pick = np.linspace(0, len(grid) - 1, args.max_patches).astype(int)
            coords = [grid[int(i)] for i in pick]
        else:
            coords = grid
        for r, c in coords:
            series: list[list[float]] = []
            oseries: list[list[float]] = []
            for di, date in enumerate(dates):
                m = valid[di, r : r + 5, c : c + 5] & in_field[r : r + 5, c : c + 5]
                vals: list[float] = []
                obs: list[float] = []
                for bi in range(7):
                    v = cube[di, bi, r : r + 5, c : c + 5][m]
                    v = v[np.isfinite(v)]
                    vals.append(float(np.median(v)) if v.size else float("nan"))
                    obs.append(1.0 if v.size else 0.0)
                nir, red = vals[3], vals[2]
                if np.isfinite(nir) and np.isfinite(red) and abs(nir + red) > 1e-8:
                    vals.append(float((nir - red) / (nir + red + 1e-8)))
                    obs.append(1.0)
                else:
                    vals.append(float("nan"))
                    obs.append(0.0)
                wrow = weather_map.get(date, np.full(len(groups), np.nan, dtype=np.float32))
                vals.extend(wrow.tolist())
                obs.extend(np.isfinite(wrow).astype(float).tolist())
                series.append(vals)
                oseries.append(obs)
            arr = np.asarray(series, dtype=np.float32)
            obsarr = np.asarray(oseries, dtype=np.float32)
            for s in range(0, max(1, len(dates) - args.window + 1), args.window):
                win = arr[s : s + args.window]
                obs_win = obsarr[s : s + args.window]
                if win.shape[0] < 2:
                    continue
                wdates = dates[s : s + args.window]
                wyears = np.array([int(dt[:4]) for dt in wdates])
                # A window must lie wholly on one side of each split boundary.
                if ((wyears.min() <= TRAIN_YEAR_END and wyears.max() >= DEV_YEAR)
                        or (wyears.min() <= DEV_YEAR and wyears.max() >= TEST_YEAR_START)):
                    n_crossing += 1
                    continue
                stage_list = [stage_map.get((farm, dt), "unknown") for dt in wdates]
                stage = Counter(stage_list).most_common(1)[0][0]
                # Global 5x5 tile origin; patch origin is already on the grid.
                pr = ((row0 + r) // 5) * 5
                pc = ((col0 + c) // 5) * 5
                labels = [label_map.get((farm, dt, pr, pc), "control") for dt in wdates]
                with np.errstate(invalid="ignore"):
                    mu_w = np.nanmean(win, axis=0)
                    sd_w = np.nanstd(win, axis=0)
                X.append(win)
                OBS.append(obs_win)
                raw_sum.append(np.concatenate([
                    np.where(np.isfinite(mu_w), mu_w, 0.0),
                    np.where(np.isfinite(sd_w), sd_w, 0.0),
                    obs_win.mean(0),
                ]))
                farm_ids.append(farm)
                stage_ids.append(stage)
                years.append(int(wyears[0]))
                y_susp.append(int(any(lab == "suspected" for lab in labels)))

    x = np.stack(X).astype(np.float32)          # NaN = unobserved
    obs_mask = np.stack(OBS).astype(np.float32)  # 1 = observed, 0 = missing
    raw = np.stack(raw_sum).astype(np.float32)
    farms = np.asarray(farm_ids)
    stages = np.asarray(stage_ids)
    years = np.asarray(years)
    y = np.asarray(y_susp, dtype=int)
    groups = np.array([f"{f}:{yr}" for f, yr in zip(farms, years)])
    print(
        f"Windows: {len(x)} features={x.shape[-1]} suspected={y.sum()} "
        f"prevalence={y.mean():.4f} boundary-crossing windows dropped={n_crossing}",
        flush=True,
    )
    if y.sum() < 5 or (1 - y).sum() < 5:
        raise RuntimeError("Insufficient suspected/control windows for a probe")

    train = years <= TRAIN_YEAR_END
    if not np.any(years >= TEST_YEAR_START) or not np.any(train):
        raise RuntimeError("Missing train (<=2022) or test (>=2024) windows after split")
    flat_train = x[train].reshape(-1, x.shape[-1])
    mu = np.nanmean(flat_train, axis=0)
    sd = np.nanstd(flat_train, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    # Missing values stay missing: the observation mask is a separate model
    # input, never an observed zero (audit 2026-11-25 §6.1).
    x_norm = np.where(obs_mask > 0, (x - mu) / sd, 0.0).astype(np.float32)
    model_input = np.concatenate([x_norm, obs_mask], axis=-1).astype(np.float32)
    rmu, rsd = raw[train].mean(0), raw[train].std(0)
    rsd = np.where(np.isfinite(rsd) & (rsd > 1e-6), rsd, 1.0)
    raw = ((raw - rmu) / rsd).astype(np.float32)
    xt = torch.from_numpy(model_input[train]).to(device)
    model = _WindowEncoder(model_input.shape[-1], d_model=32, latent=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in tqdm(range(args.epochs), desc="train encoder", unit="epoch"):
        model.train()
        perm = torch.randperm(len(xt))
        losses = []
        for i in range(0, len(xt), 64):
            batch = xt[perm[i : i + 64]]
            recon, _ = model(batch)
            loss = ((recon - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        print(f"  epoch {epoch+1}/{args.epochs} loss={np.mean(losses):.5f}", flush=True)

    def embed(model_: nn.Module, arr: np.ndarray) -> np.ndarray:
        model_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), 256):
                _, z = model_(torch.from_numpy(arr[i : i + 256]).to(device))
                out.append(z.cpu().numpy())
        return np.concatenate(out)

    print("  extracting embeddings", flush=True)
    z = embed(model, model_input)
    seed_everything(args.seed)
    random_model = _WindowEncoder(model_input.shape[-1], d_model=32, latent=16).to(device)
    z_rand = embed(random_model, model_input)
    print("  embeddings done", flush=True)

    emb_path = args.out.parent / (args.out.stem + "_embeddings.npz")
    np.savez_compressed(
        emb_path, z=z, z_rand=z_rand, raw=raw, farms=farms, stages=stages,
        years=years, y=y, groups=groups, obs_mask=obs_mask,
    )
    cysp = _cysp_rg(z, raw, farms, stages, years)
    print("  cysp done", flush=True)
    cysp_random = _cysp_rg(z_rand, raw, farms, stages, years)

    print("  fitting probes", flush=True)

    test_mask = years >= TEST_YEAR_START

    def probe(features: np.ndarray) -> dict[str, Any]:
        return _logistic_probe(features, y, train, groups, test_mask)

    result = {
        "schema": "mini-representation-first-pilot/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_commit_sha(),
        "data_source": str(DATA),
        "claim_rung": "L2",
        "claim_note": "Suspected weak labels; no stress or generalization claim.",
        "n_windows": int(len(x)),
        "n_features": int(x.shape[-1]),
        "n_suspected_windows": int(y.sum()),
        "prevalence": float(y.mean()),
        "train_rule": "year <= 2022",
        "dev_rule": "year == 2023 (excluded from retrieval and probes)",
        "test_rule": "year >= 2024",
        "split_counts": {
            "train": int(train.sum()),
            "dev": int(np.sum(years == DEV_YEAR)),
            "test": int(np.sum(years >= TEST_YEAR_START)),
            "boundary_crossing_dropped": n_crossing,
        },
        "audit_fixes": [
            "retrieval control now uses test-year queries + candidate label permutation",
            "NaN kept unobserved; observation mask fed to the encoder",
            "patches sampled on 5x5 label grid and spread across field",
            "windows crossing train/dev/test boundaries dropped",
            "converged sklearn logistic probe + farm x year cluster bootstrap CI",
        ],
        "cysp_rg": cysp,
        "cysp_rg_random_init": cysp_random,
        "probe_ssl_embedding": probe(z),
        "probe_raw_features": probe(raw),
        "probe_random_embedding": probe(z_rand),
        "collapse": _collapse(z),
        "label_source": str(PATCH_LABELS),
        "embeddings": str(emb_path),
        "caveats": [
            "CWSI_ET shared-input with LST; ETaDaily is a METRIC model product.",
            "Suspicion is not stress; no water/salinity typing.",
            "3 separated farms (R15-18/R12-08/L02-02) - L1/L2 screening only, "
            "no geographic generalization claim.",
            "n is small: 3 farms x 13 years, farm x year clusters; report CIs, "
            "do not read point estimates as effects.",
        ],
        "wall_s": round(time.monotonic() - t0, 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PROBE COMPLETE " + json.dumps({
        "cysp_gain": cysp["gain"],
        "cysp_random_gain": cysp_random["gain"],
        "ssl_auprc": result["probe_ssl_embedding"]["auprc"],
        "raw_auprc": result["probe_raw_features"]["auprc"],
        "collapse": result["collapse"],
        "out": str(args.out),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
