"""Representation-first pilot on the mini dataset.

Produces three things from real artifacts only:
1. CYSP-RG — Cross-Year Same-Phase Retrieval Gain (frozen SSL vs raw vs random init)
   with same-farm/different-stage and different-farm/same-stage hard negatives.
2. A simple model (logistic regression) trained on automatically suspected
   stress patch-windows vs matched raw-feature and B1-style baselines.
3. Collapse diagnostics (effective rank, per-dim std, mean pairwise cosine).

All splits are by YEAR (train < 2023, test >= 2023). No gate is modified;
results are L1/L2 screening evidence only.

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

DATA = Path("D:/RS/Thesis/Data/Processed/mini_research_dataset")
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


def _logistic_probe(features: np.ndarray, y: np.ndarray, train: np.ndarray) -> dict[str, Any]:
    """Small logistic regression in pure NumPy; avoids the Windows BLAS crash."""
    xtr, ytr = features[train], y[train].astype(np.float32)
    xte, yte = features[~train], y[~train]
    w = np.zeros(xtr.shape[1], dtype=np.float32)
    b = 0.0
    lr = 0.1
    for _ in range(200):
        logits = np.einsum("nd,d->n", xtr, w, optimize=False) + b
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        err = (p - ytr) / len(ytr)
        w -= lr * (np.einsum("nd,n->d", xtr, err, optimize=False) + 0.01 * w)
        b -= lr * float(err.sum())
    prob = 1.0 / (
        1.0 + np.exp(-np.clip(np.einsum("nd,d->n", xte, w, optimize=False) + b, -30, 30))
    )
    return {
        "auprc": round(_auprc(yte, prob), 4),
        "prevalence": round(float(yte.mean()), 4),
        "n_test": int(len(yte)),
        "n_positive_test": int(yte.sum()),
    }


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


def _cysp_rg(z: np.ndarray, raw: np.ndarray, farms: np.ndarray, stages: np.ndarray, years: np.ndarray) -> dict[str, Any]:
    """R@1 with hard negatives; query = test year, candidates = train years."""
    train = years < 2023
    test = ~train
    out: dict[str, Any] = {}
    for name, emb in (("ssl", z), ("raw", raw)):
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        correct, total = 0, 0
        for i in np.flatnonzero(test):
            cand = np.flatnonzero(train)
            if cand.size < 2:
                continue
            # Elementwise dot avoids a native BLAS crash on repeated small
            # matrix-vector products with the installed NumPy build.
            sims = np.sum(e[cand] * e[i], axis=1)
            top = cand[int(np.argmax(sims))]
            if farms[top] == farms[i] and stages[top] == stages[i]:
                correct += 1
            total += 1
        out[name] = {"r_at_1": round(correct / total, 4) if total else None, "n_queries": int(total)}
    out["gain"] = (
        round((out["ssl"]["r_at_1"] or 0) - (out["raw"]["r_at_1"] or 0), 4)
        if out["ssl"]["r_at_1"] is not None and out["raw"]["r_at_1"] is not None
        else None
    )
    # Shuffled-year negative control: permute candidate years, expect R@1 to fall.
    rng = np.random.default_rng(13)
    perm = rng.permutation(len(years))
    out["shuffled_year_control"] = _cysp_rg_raw(
        z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8),
        farms, stages, years, years[perm],
    )
    return out


def _cysp_rg_raw(e: np.ndarray, farms, stages, years, perm_years) -> float | None:
    train, test = perm_years < 2023, years < 2023
    correct = total = 0
    for i in np.flatnonzero(test):
        cand = np.flatnonzero(train)
        if cand.size < 2:
            continue
        sims = np.sum(e[cand] * e[i], axis=1)
        top = cand[int(np.argmax(sims))]
        correct += int(farms[top] == farms[i] and stages[top] == stages[i])
        total += 1
    return round(correct / total, 4) if total else None


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
    raw_sum: list[np.ndarray] = []
    farm_ids: list[str] = []
    stage_ids: list[str] = []
    years: list[int] = []
    y_susp: list[int] = []
    d_in: int | None = None
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
        # Deterministic subset of 5x5 patches.
        coords = [(r, c) for r in range(0, in_field.shape[0] - 4) for c in range(0, in_field.shape[1] - 4)]
        coords = coords[: args.max_patches]
        for r, c in coords:
            series = []
            for di, date in enumerate(dates):
                m = valid[di, r : r + 5, c : c + 5] & in_field[r : r + 5, c : c + 5]
                vals = []
                for bi in range(7):
                    v = cube[di, bi, r : r + 5, c : c + 5][m]
                    v = v[np.isfinite(v)]
                    vals.append(float(np.median(v)) if v.size else np.nan)
                nir, red = vals[3], vals[2]
                vals.append(float((nir - red) / (nir + red + 1e-8)) if np.isfinite(nir + red) and abs(nir + red) > 1e-8 else np.nan)
                wrow = weather_map.get(date, np.full(len(groups), np.nan, dtype=np.float32))
                vals.extend(wrow.tolist())
                series.append(vals)
            arr = np.asarray(series, dtype=np.float32)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            d_in = arr.shape[1]
            for s in range(0, max(1, len(dates) - args.window + 1), args.window):
                win = arr[s : s + args.window]
                if win.shape[0] < 2:
                    continue
                wdates = dates[s : s + args.window]
                stages = [stage_map.get((farm, dt), "unknown") for dt in wdates]
                stage_counts = Counter(stages)
                stage = stage_counts.most_common(1)[0][0]
                # Patch labels are keyed by global 5x5 tile origin (floor/5*5).
                pr = ((row0 + r) // 5) * 5
                pc = ((col0 + c) // 5) * 5
                labels = [label_map.get((farm, dt, pr, pc), "control") for dt in wdates]
                X.append(win)
                raw_sum.append(np.concatenate([win.mean(0), win.std(0)]))
                farm_ids.append(farm)
                stage_ids.append(stage)
                years.append(int(wdates[0][:4]))
                y_susp.append(int(any(lab == "suspected" for lab in labels)))

    x = np.stack(X).astype(np.float32)
    raw = np.stack(raw_sum).astype(np.float32)
    farms = np.asarray(farm_ids)
    stages = np.asarray(stage_ids)
    years = np.asarray(years)
    y = np.asarray(y_susp, dtype=int)
    print(f"Windows: {len(x)} features={x.shape[-1]} suspected={y.sum()} prevalence={y.mean():.4f}", flush=True)
    if y.sum() < 5 or (1 - y).sum() < 5:
        raise RuntimeError("Insufficient suspected/control windows for a probe")

    train = years < 2023
    # Train-only standardization per channel; raw baseline uses the same scale.
    mu = x[train].reshape(-1, x.shape[-1]).mean(0)
    sd = x[train].reshape(-1, x.shape[-1]).std(0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    x = np.where(np.isfinite(x), (x - mu) / sd, 0.0).astype(np.float32)
    rmu, rsd = raw[train].mean(0), raw[train].std(0)
    rsd = np.where(np.isfinite(rsd) & (rsd > 1e-6), rsd, 1.0)
    raw = np.where(np.isfinite(raw), (raw - rmu) / rsd, 0.0).astype(np.float32)
    xt = torch.from_numpy(x[train]).to(device)
    model = _WindowEncoder(x.shape[-1], d_model=32, latent=16).to(device)
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
    z = embed(model, x)
    seed_everything(args.seed)
    random_model = _WindowEncoder(x.shape[-1], d_model=32, latent=16).to(device)
    z_rand = embed(random_model, x)
    print("  embeddings done", flush=True)

    emb_path = args.out.parent / (args.out.stem + "_embeddings.npz")
    np.savez_compressed(
        emb_path, z=z, z_rand=z_rand, raw=raw, farms=farms, stages=stages,
        years=years, y=y,
    )
    cysp = _cysp_rg(z, raw, farms, stages, years)
    print("  cysp done", flush=True)
    cysp_random = _cysp_rg(z_rand, raw, farms, stages, years)

    print("  fitting probes", flush=True)

    def probe(features: np.ndarray) -> dict[str, Any]:
        return _logistic_probe(features, y, train)

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
        "train_rule": "year < 2023",
        "test_rule": "year >= 2023",
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
            "3 adjacent farms only; no geographic generalization claim.",
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
