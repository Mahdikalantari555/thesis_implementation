"""Probe pipeline: 4 methods → Stage-1 scores → gates → events → Stage-2.

Orchestrates change `farm-subset-anomaly-probe` tasks 1.x/2.x/3.x/4.x:

- fits B2 / pixel CNN-AE / Barlow Twins / GRU-AE on **train dates only**
  (first ``train_years`` calendar years of the archive),
- scores all 32 farms × all dates,
- Stage-1: per (farm, date) score parquet with per-method scores, ΔLST,
  gate_status, event_id (anomaly wording only — never "stress", L2 only),
- Stage-2: pixel-level (48×48) score maps for the top-K suspicious
  (farm, date) pairs via ONNXRuntime CPU,
- provenance JSON with git SHA, seeds, n, date range, data_source, wall times,
  parameter counts, norm stats and thresholds (§6/§7 report checklist).

Progress: tqdm bars + machine-readable ``progress.jsonl``; ``--resume``
supported per NN method (checkpoint last.pt + rng state).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..utils.tracking import get_git_commit_sha
from .b2 import B2Baseline
from .barlow import BarlowProbe, build_windows
from .cnn_ae import (PixelCNNAE, count_params, field_date_scores, fit_norm_stats,
                     make_input, masked_loss)
from .data import BAND_ORDER, attach_cycles_weather, load_subset_features
from .gates import apply_gates, build_events, delta_lst
from .gru import TrajectoryGRUAE, build_tokens
from .stage2 import export_onnx, ort_score_canvas, ort_session

BANDS = BAND_ORDER


def _append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _save_ckpt(path: Path, model, opt, epoch: int, seed: int) -> None:
    torch.save({"model_state": model.state_dict(),
                "opt_state": opt.state_dict() if opt is not None else None,
                "epoch": epoch, "seed": seed}, path)


def _load_ckpt(path: Path, model, opt=None) -> int:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state"])
    if opt is not None and blob.get("opt_state") is not None:
        opt.load_state_dict(blob["opt_state"])
    return int(blob["epoch"])


def run_probe(
    subset: Path,
    out_dir: Path,
    *,
    seed: int = 2026,
    train_years: int = 3,
    methods: tuple[str, ...] = ("b2", "cnn_ae", "barlow", "gru_ae"),
    cnn_epochs: int = 30,
    cnn_batch: int = 32,
    barlow_epochs: int = 5,
    gru_epochs: int = 5,
    top_k: int = 50,
    flag_q: float = 0.99,
    resume: bool = False,
    smoke: bool = False,
    torch_threads: int = 4,
) -> dict:
    """Runs the full probe; returns the summary dict (also printed as JSON)."""
    t0 = time.monotonic()
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    if smoke:
        cnn_epochs = barlow_epochs = gru_epochs = 1
        top_k = 2

    def log(rec: dict) -> None:
        _append_jsonl(progress_path, {"wall_s": round(time.monotonic() - t0, 1), **rec})

    log({"kind": "start", "subset": str(subset), "seed": seed,
         "methods": list(methods), "smoke": smoke,
         "git_commit_sha": get_git_commit_sha()})

    d, _ = load_subset_features(subset)
    d = attach_cycles_weather(d, train_years=train_years)
    F, D, K = d.band_means.shape
    S = int(d.provenance["canvas"])
    log({"kind": "data", "farms": F, "dates": D, "canvas": S,
         "train_dates": int(d.train_mask.sum())})
    band_matrix = d.band_means

    scores: dict[str, np.ndarray] = {}
    fit_info: dict[str, dict] = {}

    # ---------------- 1.1 B2 no-NN baseline ----------------
    if "b2" in methods:
        t = time.monotonic()
        b2 = B2Baseline().fit(d)
        sc = b2.score(d)
        scores["b2_score"] = sc["b2_score"]
        fit_info["b2"] = {"wall_s": round(time.monotonic() - t, 1),
                          "n_bins": b2.n_bins,
                          "sigma_floor": b2.sigma_floor}
        log({"kind": "method_done", "method": "b2",
             "wall_s": fit_info["b2"]["wall_s"]})

    # ---------------- 1.2 pixel CNN-AE ----------------
    cnn_sigma = None
    if "cnn_ae" in methods:
        t = time.monotonic()
        norm = fit_norm_stats(d, d.train_mask)
        model = PixelCNNAE()
        n_params = count_params(model)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        start_ep = 0
        ckpt = ckpt_dir / f"cnn_ae_seed{seed}.pt"
        if resume and ckpt.exists():
            start_ep = _load_ckpt(ckpt, model, opt) + 1
            log({"kind": "resume", "method": "cnn_ae", "start_epoch": start_ep})
        farm_ids = list(d.canvases.keys())
        train_dates = np.where(d.train_mask)[0]
        samples = [(fi, di) for fi in range(F) for di in train_dates
                   if d.obs_mask[fi, di]]
        model.train()
        for ep in range(start_ep, cnn_epochs):
            rng = np.random.default_rng(seed + ep)
            order = rng.permutation(len(samples))
            losses = []
            for bs in tqdm(range(0, len(order), cnn_batch), desc=f"cnn_ae ep{ep+1}/{cnn_epochs}",
                           unit="batch", leave=False):
                batch = [samples[i] for i in order[bs:bs + cnn_batch]]
                xs, ys, ms = [], [], []
                for fi, di in batch:
                    in_field, data, pixel_valid = d.canvases[farm_ids[fi]]
                    x, m = make_input(data[di], pixel_valid[di], in_field, norm)
                    xs.append(x)
                    ms.append(m)
                xb = torch.from_numpy(np.stack(xs))
                mask_t = torch.from_numpy(np.stack(ms))
                tgt = xb[:, :7].clone()  # standardized target
                pred = model(xb)
                loss = masked_loss(pred, tgt, mask_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            log({"kind": "epoch", "method": "cnn_ae", "epoch": ep + 1,
                 "loss": float(np.mean(losses))})
            _save_ckpt(ckpt, model, opt, ep, seed)
        # per-band σ of standardized train residuals → scores in σ units
        model.eval()
        resid_acc = [[] for _ in range(K)]
        with torch.no_grad():
            for bs in tqdm(range(0, len(samples), cnn_batch), desc="cnn_ae train-sigma",
                           unit="batch", leave=False):
                batch = samples[bs:bs + cnn_batch]
                xs, ms = [], []
                for fi, di in batch:
                    in_field, data, pixel_valid = d.canvases[farm_ids[fi]]
                    x, m = make_input(data[di], pixel_valid[di], in_field, norm)
                    xs.append(x)
                    ms.append(m)
                xb = torch.from_numpy(np.stack(xs))
                mt = torch.from_numpy(np.stack(ms))
                pred = model(xb)
                diff = ((pred - xb[:, :7]).abs() * mt.unsqueeze(1)).sum(dim=0).numpy()
                cnt = mt.sum(dim=0).numpy()
                for k in range(K):
                    sel = cnt > 0
                    resid_acc[k].append(diff[k][sel] / np.maximum(cnt[sel], 1))
        cnn_sigma = np.array([np.maximum(np.std(np.concatenate(a)) if a and np.concatenate(a).size > 1 else 1.0,
                                         1e-6) for a in resid_acc])
        cnn_scores = np.full((F, D), np.nan, dtype=np.float32)
        cnn_pixmaps: dict[tuple[int, int], np.ndarray] = {}
        with torch.no_grad():
            all_pairs = [(fi, di) for fi in range(F) for di in range(D)
                         if d.obs_mask[fi, di]]
            for bs in tqdm(range(0, len(all_pairs), cnn_batch), desc="cnn_ae score",
                           unit="batch", leave=False):
                batch = all_pairs[bs:bs + cnn_batch]
                xs, ms = [], []
                for fi, di in batch:
                    in_field, data, pixel_valid = d.canvases[farm_ids[fi]]
                    x, m = make_input(data[di], pixel_valid[di], in_field, norm)
                    xs.append(x)
                    ms.append(m)
                xb = torch.from_numpy(np.stack(xs))
                mt = torch.from_numpy(np.stack(ms))
                tgt = xb[:, :7].numpy()
                pred = model(xb).numpy()
                for j, (fi, di) in enumerate(batch):
                    diff = np.abs(pred[j] - tgt[j]) / cnn_sigma[:, None, None]
                    per_px = diff.mean(axis=0)
                    per_px[~mt[j]] = np.nan
                    cnn_scores[fi, di] = np.nanmean(per_px)
                    cnn_pixmaps[(fi, di)] = per_px.astype(np.float32)
        scores["cnn_ae_score"] = cnn_scores
        fit_info["cnn_ae"] = {"wall_s": round(time.monotonic() - t, 1),
                              "n_params": n_params, "epochs": cnn_epochs,
                              "band_sigma": cnn_sigma.tolist(),
                              "norm_stats": norm}
        log({"kind": "method_done", "method": "cnn_ae",
             "wall_s": fit_info["cnn_ae"]["wall_s"], "n_params": n_params})

    # ---------------- 1.3 Barlow Twins on trajectories ----------------
    if "barlow" in methods:
        t = time.monotonic()
        window = 16
        windows, _ = build_windows(d, band_matrix, window=window)
        bt = BarlowProbe(window=window, seed=seed)
        n_params = count_params(bt.model)
        ckpt = ckpt_dir / f"barlow_seed{seed}.pt"
        start_ep = 0
        opt = None
        if resume and ckpt.exists():
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            bt.model.load_state_dict(blob["model_state"])
            start_ep = int(blob["epoch"]) + 1
            log({"kind": "resume", "method": "barlow", "start_epoch": start_ep})
        torch.manual_seed(seed)

        def cb(ep, loss):
            log({"kind": "epoch", "method": "barlow", "epoch": ep + 1, "loss": loss})

        if len(windows) == 0:
            # e.g. smoke store with fewer observed dates than the window —
            # the method is recorded as ABSENT (never silently zero-filled)
            fit_info["barlow"] = {"skipped": (
                f"no {window}-date windows in the store ({int(d.obs_mask.sum())} "
                "observed farm-dates); barlow_score absent from outputs")}
            log({"kind": "method_skipped", "method": "barlow",
                 "reason": fit_info["barlow"]["skipped"]})
        else:
            loss = bt.train(windows, epochs=barlow_epochs, progress_cb=cb,
                            start_ep=start_ep)
            torch.save({"model_state": bt.model.state_dict(),
                        "epoch": barlow_epochs - 1, "seed": seed}, ckpt)
            bt.fit_scoring(windows)
            scores["barlow_score"] = bt.score(windows, n_farms=F, n_dates=D)
            fit_info["barlow"] = {"wall_s": round(time.monotonic() - t, 1),
                                  "n_params": n_params, "epochs": barlow_epochs,
                                  "n_windows": len(windows), "window": window,
                                  "final_loss": loss}
            log({"kind": "method_done", "method": "barlow",
                 "wall_s": fit_info["barlow"]["wall_s"], "n_params": n_params})

    # ---------------- 1.4 GRU-AE baseline ----------------
    if "gru_ae" in methods:
        t = time.monotonic()
        tokens, target, finite = build_tokens(d, band_matrix)
        gru = TrajectoryGRUAE(seed=seed)
        n_params = gru.count_params()
        ckpt = ckpt_dir / f"gru_ae_seed{seed}.pt"
        start_ep = 0
        if resume and ckpt.exists():
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            gru.model.load_state_dict(blob["model_state"])
            start_ep = int(blob["epoch"]) + 1
            gru.trained = True  # band_sigma recomputed below
            log({"kind": "resume", "method": "gru_ae", "start_epoch": start_ep})

        def cb(ep, loss):
            log({"kind": "epoch", "method": "gru_ae", "epoch": ep + 1, "loss": loss})

        loss = gru.train(tokens, target, finite, d.train_mask, epochs=gru_epochs,
                         progress_cb=cb, start_ep=start_ep)
        torch.save({"model_state": gru.model.state_dict(),
                    "epoch": gru_epochs - 1, "seed": seed}, ckpt)
        scores["gru_ae_score"] = gru.score(tokens, target, finite)
        fit_info["gru_ae"] = {"wall_s": round(time.monotonic() - t, 1),
                              "n_params": n_params, "epochs": gru_epochs,
                              "d_model": gru.d_model, "final_loss": loss}
        log({"kind": "method_done", "method": "gru_ae",
             "wall_s": fit_info["gru_ae"]["wall_s"], "n_params": n_params})

    # ---------------- 2.1 Stage-1 parquet ----------------
    t_stage1 = time.monotonic()
    rows = []
    rank_cols = {}
    for name, arr in scores.items():
        finite = np.isfinite(arr)
        r = np.full_like(arr, np.nan, dtype=np.float64)
        r[finite] = pd.Series(arr[finite]).rank(pct=True).to_numpy()
        rank_cols[name] = r
    dlst = delta_lst(d.lst_c, d.obs_mask)
    for fi, farm in enumerate(d.farm_ids):
        for di, date in enumerate(d.dates):
            row = {"farm_id": farm, "archive_fid": d.archive_fids[fi],
                   "date": date, "n_px": int(d.n_px[fi, di]),
                   "absent": not bool(d.obs_mask[fi, di]),
                   "cycle_id": int(d.cycle_id[fi, di]),
                   "tau": None if not np.isfinite(d.tau[fi, di]) else float(d.tau[fi, di]),
                   "gdd_cum": None if not np.isfinite(d.gdd_cum[fi, di]) else float(d.gdd_cum[fi, di]),
                   "lst_c": None if not np.isfinite(d.lst_c[fi, di]) else float(d.lst_c[fi, di]),
                   "delta_lst_c": None if not np.isfinite(dlst[fi, di]) else float(dlst[fi, di])}
            ranks = []
            for name, arr in scores.items():
                v = arr[fi, di]
                row[name] = None if not np.isfinite(v) else float(v)
                if np.isfinite(rank_cols[name][fi, di]):
                    ranks.append(rank_cols[name][fi, di])
            row["combined_score"] = float(np.mean(ranks)) if ranks else None
            rows.append(row)
    df = pd.DataFrame(rows)
    score_cols = list(scores.keys())
    df, gate_info = apply_gates(df, score_cols, d.train_mask, d.dates,
                                flag_q=flag_q)
    events = build_events(df, score_cols)
    stage1_path = out_dir / "scores_stage1.parquet"
    events_path = out_dir / "events.parquet"
    df.to_parquet(stage1_path, index=False)
    events.to_parquet(events_path, index=False)
    log({"kind": "stage1", "rows": len(df), "wall_s": round(time.monotonic() - t_stage1, 1),
         "n_events": gate_info["n_events"]})

    # ---------------- 2.2 Stage-2 top-K (ONNXRuntime CPU) ----------------
    stage2_info = {"done": False}
    if "cnn_ae" in methods and top_k > 0:
        t_stage2 = time.monotonic()
        cand = df[(~df["absent"]) & (df["gate_status"] != "rejected_cold_lst")
                  & df["combined_score"].notna()].copy()
        cand = cand.sort_values("combined_score", ascending=False).head(top_k)
        onnx_path = out_dir / "cnn_ae_s48.onnx"
        export_onnx(model, onnx_path, canvas=S)
        sess = ort_session(onnx_path)
        norm = fit_info["cnn_ae"]["norm_stats"]
        stage2_rows = []
        pixmaps = {}
        for _, r in cand.iterrows():
            fi = d.farm_ids.index(r["farm_id"])
            di = d.dates.index(r["date"])
            in_field, data, pixel_valid = d.canvases[r["farm_id"]]
            fd, pmap = ort_score_canvas(sess, data[di], pixel_valid[di],
                                        in_field, norm, np.array(cnn_sigma))
            pixmaps[f"{r['farm_id']}|{r['date']}"] = pmap
            stage2_rows.append({"farm_id": r["farm_id"], "date": r["date"],
                                "stage1_combined_score": float(r["combined_score"]),
                                "stage2_field_date_score_ort": fd,
                                "stage1_cnn_ae_score": r["cnn_ae_score"]})
        s2 = pd.DataFrame(stage2_rows)
        s2.to_parquet(out_dir / "stage2_topk.parquet", index=False)
        np.savez_compressed(out_dir / "stage2_pixel_maps.npz", **pixmaps)
        stage2_info = {"done": True, "k": len(s2),
                       "wall_s": round(time.monotonic() - t_stage2, 1),
                       "onnx_path": str(onnx_path)}
        log({"kind": "stage2", **stage2_info})

    # ---------------- 4.2 provenance ----------------
    wall = time.monotonic() - t0
    provenance = {
        "change": "farm-subset-anomaly-probe",
        "purpose": "32-farm CPU anomaly-scoring probe; L2 screening evidence only (no agronomic diagnosis)",
        "git_commit_sha": get_git_commit_sha(),
        "seed": seed,
        "data_source": {
            "subset_store": str(subset),
            "store_git_sha": d.provenance["store_git_sha"],
            "cycles_path": d.provenance["cycles_path"],
            "weather_csv": d.provenance["weather_csv"],
        },
        "n_farms": F, "n_dates": D,
        "date_range": [d.dates[0], d.dates[-1]],
        "train_years": train_years,
        "train_year_list": d.provenance["train_year_list"],
        "n_train_dates": int(d.train_mask.sum()),
        "n_eval_dates": int((~d.train_mask).sum()),
        "methods": list(methods),
        "fit_info": fit_info,
        "gate_info": gate_info,
        "stage2": stage2_info,
        "torch_threads": torch_threads,
        "device": "cpu",
        "wall_s": round(wall, 1),
        "outputs": {"stage1": str(stage1_path), "events": str(events_path)},
        "claim_rung": "L2 (anomaly screening on the 32-farm dev subset; no agronomic diagnosis — L4 evidence absent)",
    }
    prov_path = out_dir / "provenance.json"
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)

    summary = {
        "event": "PROBE COMPLETE",
        "wall_s": round(wall, 1),
        "out_dir": str(out_dir),
        "seed": seed,
        "methods": list(methods),
        "n_stage1_rows": len(df),
        "n_events": gate_info["n_events"],
        "stage2_k": stage2_info.get("k", 0),
        "outputs": {"stage1": str(stage1_path), "events": str(events_path),
                    "provenance": str(prov_path)},
    }
    print(json.dumps(summary))
    log({"kind": "summary", **summary})
    return summary
