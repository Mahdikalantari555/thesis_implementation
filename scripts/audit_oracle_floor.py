#!/usr/bin/env python
"""Oracle-floor audit (openspec change ``oracle-floor-audit``, task 2.1).

Measures, on the clean unassigned pool V (cycles outside train/dev/test),
with the FROZEN patch scorer:

* ``sigma_h(g, c)`` — robust σ (1.4826·MAD) of clean signed in-field-mean
  residuals, per phase group, per channel, per conditioning gap h (forecast
  of bin t from bins ≤ t−h);
* the measured SHIFT PROFILE by distance-from-onset, per gap h: a one-step
  baseline absorbs a sustained drop within ~1–2 bins, so the decision-
  relevant object is the median per-bin z-shift at distance d = 0, 1, 2, …
  from the injection onset (gap-h extends the context-clean run to h bins);
* ``delta_win(h)``  — the best trailing-window mean of that profile (exactly
  what a w=3 window statistic keys on);
* ``retention(h)``  — delta_win(h) / delta_ideal, where delta_ideal is the
  absorption-free onset-bin response; the consultant's pre-registered gate
  in retention form (max_h ≥ 0.72 on BOTH splits = recoverable; < 0.62
  everywhere = branch A);
* a Monte-Carlo Gaussian oracle driven by the MEASURED profile (not a
  constant-δ idealization), at cycle-FAR 5%, and the gap vs the committed
  finding-25 detection rates → the pre-registered (A)/(C)/(B) decision.

All quantities are MEASURED; the oracle's remaining idealization is
iid-Gaussian evidence, recorded as such. One forward pass per (chunk,
clean/injected) serves ALL gap values via ``forward_anchor`` + realignment.

Fail-closed: V disjoint from train/dev/test, pinned parquet hash, h=1 parity
vs ``predict_pixels`` asserted on the first chunk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CHANNEL_ORDER = ("ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst")
TOUCHED_CHANNELS = {"ndvi": (0,), "savi": (0, 3, 4)}  # ndvi / red / nir
Q_RECOVERABLE = 0.72   # retention ≥ this on both splits → branch B/C live
Q_DEAD = 0.62          # retention < this everywhere → branch A


# ---------------------------------------------------------------- pure math
def robust_sigma(x) -> float:
    """1.4826 · MAD, NaN-aware; NaN when nothing is finite."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def trailing_window_means(z: np.ndarray, w: int) -> np.ndarray:
    """(N, T-w+1) means of trailing w bins; entry j ends at bin j+w-1."""
    return np.lib.stride_tricks.sliding_window_view(z, w, axis=1).mean(axis=2)


def gaussian_oracle(profile, *, far: float = 0.05, t_bins: int = 36,
                    window: int = 3, onsets=(6, 18, 30), n_sims: int = 50_000,
                    seed: int = 2026) -> dict:
    """Monte-Carlo Gaussian oracle driven by a MEASURED shift profile.

    ``profile[d]`` is the per-bin shift (in per-bin σ units) at distance d
    from the onset. Alarm = trailing w-bin mean of iid N(0,1) evidence
    exceeding a threshold calibrated to a cycle-FAR of ``far`` over T bins.
    Reports power per onset and the phase-rotation average.
    """
    scalar = np.isscalar(profile) or (hasattr(profile, "ndim") and profile.ndim == 0)
    profile = np.atleast_1d(np.asarray(profile, dtype=float))
    if scalar:
        # a scalar δ means a SUSTAINED shift on every post-onset bin
        profile = np.full(t_bins, float(profile[0]))
    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal((n_sims, t_bins))
    thr = float(np.quantile(trailing_window_means(z0, window).max(axis=1), 1.0 - far))
    ends = np.arange(window - 1, t_bins)
    powers = {}
    for onset in onsets:
        add = np.zeros(t_bins)
        d_max = min(len(profile), t_bins - onset)
        add[onset:onset + d_max] = profile[:d_max]
        z1 = z0 + add[None, :]
        m1 = trailing_window_means(z1, window)
        valid = ends >= onset
        powers[int(onset)] = float((m1[:, valid] > thr).any(axis=1).mean())
    return {"profile": [round(float(v), 6) for v in profile],
            "far_target": float(far), "threshold": thr,
            "window": int(window), "t_bins": int(t_bins),
            "n_sims": int(n_sims), "seed": int(seed),
            "power_per_onset": powers,
            "power": float(np.mean(list(powers.values()))),
            "method": "monte_carlo iid-gaussian trailing-window-mean, measured profile"}


def decision_for_gap(gap: float) -> str:
    """Pre-registered bands: ≤0.10 → A (floor); ≤0.30 → C (statistic); else B."""
    if gap <= 0.10:
        return "A"
    if gap <= 0.30:
        return "C"
    return "B"


def field_blocked_split(fields, seed: int = 13) -> tuple[set, set, str]:
    """Deterministic half/half split of sorted field ids; returns (cal, test, hash)."""
    fields = sorted(int(f) for f in fields)
    perm = np.random.default_rng(seed).permutation(len(fields))
    ordered = [fields[i] for i in perm]
    half = len(ordered) // 2
    cal, test = set(ordered[:half]), set(ordered[half:])
    digest = hashlib.sha256(
        ("cal:" + ",".join(map(str, sorted(cal))) +
         "|test:" + ",".join(map(str, sorted(test)))).encode()
    ).hexdigest()[:16]
    return cal, test, digest


def realign(anchor: np.ndarray, gap: int) -> np.ndarray:
    """pred[:, t] = anchor[:, t-gap] for t ≥ gap; head bins NaN (gap=1 ≡ production)."""
    pred = np.full_like(anchor, np.nan)
    pred[:, gap:] = anchor[:, : anchor.shape[1] - gap]
    return pred


def best_window_median(zmax: np.ndarray, post: np.ndarray, w: int,
                       onsets: np.ndarray, t_bins: int) -> float:
    """Median over cycles of the best post-onset w-window mean of zmax.

    Mirrors the alarm statistic's max-over-windows semantics: per cycle, the
    statistic fires on its BEST window, so the measured per-cycle max is the
    detection-relevant read.
    """
    wm = trailing_window_means(zmax, w)              # (N, T-w+1)
    win_end = np.arange(w - 1, t_bins)               # end bin of window j
    vals = []
    for n in range(zmax.shape[0]):
        valid = (win_end >= onsets[n]) & (win_end >= w - 1)
        if valid.any():
            vals.append(np.nanmax(wm[n, valid]))
    return float(np.nanmedian(vals)) if vals else float("nan")


# ---------------------------------------------------------------- V pool
def build_v_pool(manifest: dict, root):
    """Clean cycles outside train/dev/test, joined to the patch store."""
    import pandas as pd
    from ssl_anomaly.io import patches as P

    rel = str(manifest["cycle_files"]["parquet"]).replace("\\", "/")
    parquet = Path(rel) if Path(rel).is_absolute() else REPO_ROOT / rel
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    if digest != manifest["cycle_file_hashes"]["cycles_parquet_sha256"]:
        raise ValueError(f"cycles parquet {parquet} does not match the manifest-pinned hash")
    df = pd.read_parquet(parquet)
    df["cid"] = df["field"].astype(str) + ":" + df["cycle_id"].astype(str)
    clean_cids = set(df.loc[~df["censored"].astype(bool), "cid"])
    windows = {(int(r.field), int(r.cycle_id)): (int(r.t_start), int(r.t_end) - 1)
               for r in df.itertuples()}

    assigned = set(manifest["cycles"]["train"]) | set(manifest["cycles"]["dev"]) \
        | set(manifest["cycles"]["test"])
    v_keys = [k for k in manifest["cycles"]["monitoring_only"] if k in clean_cids]
    overlap = sorted(set(v_keys) & assigned)
    if overlap:
        raise ValueError(f"V pool overlaps assigned cycles: {overlap[:5]}")

    index = P.patch_cycle_index(root)
    epoch0 = np.datetime64(manifest["base_date"], "D")
    keys, fids, cids, starts, ends = [], [], [], [], []
    for key in v_keys:
        if key not in index:
            continue  # monitoring cycles may fall outside the store's window set
        cycle = index[key]
        pair = (cycle.field_id, cycle.cycle_id)
        if pair not in windows:
            continue
        t_start, t_end_incl = windows[pair]
        if cycle.start_day != t_start or cycle.end_day != t_end_incl:
            raise ValueError(f"patch bounds for {key} disagree with the pinned cycle table")
        keys.append(key)
        fids.append(int(cycle.field_id))
        cids.append(int(cycle.cycle_id))
        starts.append(epoch0 + np.timedelta64(cycle.start_day, "D"))
        ends.append(epoch0 + np.timedelta64(cycle.end_day + 1, "D"))
    if not keys:
        raise ValueError("V pool is empty after the store join")
    return {"cycle_keys": keys, "field_ids": np.array(fids, dtype=np.int64),
            "cycle_ids": np.array(cids, dtype=np.int64),
            "start_dates": np.array(starts, dtype="datetime64[D]"),
            "end_dates": np.array(ends, dtype="datetime64[D]")}


# ---------------------------------------------------------------- scoring core
def _field_mean(resid: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """(N, T, C) in-field observed-pixel mean of signed residuals."""
    num = (resid * observed).sum(axis=(3, 4))
    den = np.maximum(observed.sum(axis=(3, 4)), 1)
    return num / den


def audit_arm(arm: str, args, V: dict, is_cal: np.ndarray, out_dir: Path) -> dict:
    import torch
    from ssl_anomaly.io import patches as P
    from ssl_anomaly.monitor import patch_eval as pE
    from ssl_anomaly.monitor import patch_perturbations as ppert
    from ssl_anomaly.monitor import patch_scores as PS
    from ssl_anomaly.monitor import perturbations as mpert
    from ssl_anomaly.monitor import severity_table as sev
    import eval_dev as ED

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = REPO_ROOT / "checkpoints" / "patch" / args.run_id \
        / f"{arm}_seed{args.checkpoint_seed}" / "last.pt"
    model, sigma, norm, cfg = ED.load_patch_artifacts(args.run_id, checkpoint_path=ckpt)
    if int(cfg.get("seed", -1)) != int(args.checkpoint_seed):
        raise ValueError(f"checkpoint seed {cfg.get('seed')} != requested {args.checkpoint_seed}")
    stats = P.patch_norm_stats(norm)
    iqr = {name: float(norm["stats"]["unknown"][name]["iqr"])
           for name in CHANNEL_ORDER}
    delta = float(next(lv.delta_ndvi for lv in sev.SEVERITY_TABLE
                       if lv.level_id == args.level))
    duration = (int(sev.PERSISTENT_DURATION_BINS) if args.shape == "persistent"
                else int(next(lv.duration_bins for lv in sev.SEVERITY_TABLE
                              if lv.level_id == args.level)))

    h_list = args.h_list
    t_bins = None
    touched = sorted({c for idx in args.index for c in TOUCHED_CHANNELS[idx]})

    acc_clean = {h: [] for h in h_list}
    acc_inj = {(idx, h): [] for idx in args.index for h in h_list}
    onsets = []
    parity_ok = False

    store_path = Path(args.store) if Path(args.store).is_absolute() else REPO_ROOT / args.store
    root = P.open_store(store_path)
    n_v = len(V["cycle_keys"])
    for start in range(0, n_v, args.chunk):
        sl = slice(start, min(start + args.chunk, n_v))
        batch = pE.patch_batch_for_cycles(
            root, V["field_ids"][sl], V["cycle_ids"][sl], norm=norm)
        raw, X = batch["raw"], batch["X"]
        obs, in_field = batch["obs"], batch["in_field"]
        pv = batch["pixel_valid"] if "pixel_valid" in batch else batch["M"]
        sid = batch["sensor_id"]
        cov = batch.get("covariates")
        n, T = X.shape[:2]
        if t_bins is None:
            t_bins = T
        elif T != t_bins:
            raise ValueError(f"store window T changed mid-audit: {T} != {t_bins}")

        arm_obj = ED.build_patch_arm(arm, model, cfg, device, sigma=sigma,
                                     in_field=in_field)
        m_pix = np.asarray(pv, dtype=bool)
        if m_pix.ndim == 5:
            m_pix = m_pix[:, :, 0]
        inf = np.asarray(in_field, dtype=bool)
        if inf.ndim == 3:
            m_pix = m_pix & inf[:, None]
        elif inf.ndim == 2:
            m_pix = m_pix & inf[None, None]
        else:
            raise ValueError(f"in_field has unsupported ndim {inf.ndim}")
        observed = m_pix[:, :, None] & np.asarray(obs, dtype=bool)
        model_mask = observed.any(axis=2)
        xfed = np.where(observed, X, 0.0).astype(np.float32)

        anchor_c, x_in = arm_obj.forward_anchor(xfed, model_mask, cov)
        if not parity_ok:
            ref = arm_obj.predict_pixels(xfed, pv, sensor_ids=sid,
                                         covariates=cov, gap=1)
            if not np.array_equal(ref["resid"], realign(anchor_c, 1) - x_in,
                                  equal_nan=True):
                raise AssertionError("gap=1 realignment is not byte-identical "
                                     "to predict_pixels — audit aborted")
            parity_ok = True

        for h in h_list:
            acc_clean[h].append(_field_mean(realign(anchor_c, h) - x_in, observed))

        ons = [int(mpert.phase_start_bin(
            mpert.PHASES[(start + i) % len(mpert.PHASES)], T)) for i in range(n)]
        onsets.extend(ons)

        for idx in args.index:
            Xinj = np.empty_like(X)
            for i in range(n):
                inj = ppert.inject_patch_drop(
                    raw[i].astype(np.float32), obs[i], index=idx,
                    delta=delta, onset_bin=ons[i], shape=args.shape,
                    duration=duration, in_field=in_field[i],
                    injection_basis=ppert.BASIS_BANDS_CONSISTENT)
                Xinj[i], _ = P.normalize_patch(inj.x, obs[i], sid[i], stats)
            anchor_i, x_in_i = arm_obj.forward_anchor(
                np.where(observed, Xinj, 0.0).astype(np.float32), model_mask, cov)
            for h in h_list:
                acc_inj[(idx, h)].append(
                    _field_mean(realign(anchor_i, h) - x_in_i, observed))
        print(f"  [{arm}] chunk {start}: {n} cycles (clean + {len(args.index)} "
              f"injected; {len(h_list)} gaps post-hoc)", flush=True)

    onsets = np.array(onsets, dtype=int)
    groups = np.minimum(np.arange(t_bins) // PS.PHASE_BIN,
                        int(np.ceil(t_bins / PS.PHASE_BIN)) - 1)
    n_groups = int(groups.max()) + 1
    clean_h = {h: np.concatenate(a, axis=0) for h, a in acc_clean.items()}
    inj_h = {k: np.concatenate(a, axis=0) for k, a in acc_inj.items()}
    n_all = onsets.shape[0]

    # z-shift relative to per-(gap, group, channel) clean sigma, then channel-max
    zmax_by_gap = {}
    for h in h_list:
        per_split_sigma = {}
        for name, mask in (("v_cal", is_cal), ("v_test", ~is_cal)):
            sig = {c: np.array([robust_sigma(clean_h[h][mask][:, groups == g, c])
                                for g in range(n_groups)])
                   for c in touched}
            per_split_sigma[name] = sig
        zmax_by_gap[h] = {}
        for name, mask in (("v_cal", is_cal), ("v_test", ~is_cal)):
            sig = per_split_sigma[name]
            # channel-max per INDEX family: per touched channel take ITS OWN
            # column (N, T), then max across the touched channels
            zmax_by_gap[h][name] = {
                idx: np.nanmax(np.stack(
                    [((inj_h[(idx, h)][mask] - clean_h[h][mask])
                      / sig[c][groups][None, :, None])[:, :, c]
                     for c in TOUCHED_CHANNELS[idx]], axis=0), axis=0)
                for idx in args.index}

    splits = {}
    for name, mask in (("v_cal", is_cal), ("v_test", ~is_cal)):
        per_index = {}
        for idx in args.index:
            curves = {}
            for h in h_list:
                zmax = zmax_by_gap[h][name][idx]              # (n_split, T)
                d_range = t_bins - int(onsets[mask].min())
                prof, prof_n = [], []
                for d in range(d_range):
                    vals = np.full(mask.sum(), np.nan)
                    ok = (onsets[mask] + d) < t_bins
                    vals[ok] = zmax[np.arange(mask.sum())[ok],
                                    onsets[mask][ok] + d]
                    prof.append(float(np.nanmedian(vals)))
                    prof_n.append(int(ok.sum()))
                prof = np.array(prof)
                d_win = best_window_median(zmax, None, args.window,
                                           onsets[mask], t_bins)
                # absorption-free ideal: the onset-bin response of the FIRST
                # gap (identical for every h by construction — sanity anchor)
                curves[h] = {"profile": [round(float(v), 4) for v in prof],
                             "profile_n": prof_n, "delta_win": d_win}
            delta_ideal = curves[h_list[0]]["profile"][0]
            for h in h_list:
                curves[h]["retention"] = (float(curves[h]["delta_win"] / delta_ideal)
                                          if delta_ideal and np.isfinite(delta_ideal)
                                          else float("nan"))
            h_star = max(h_list, key=lambda h: curves[h]["retention"]
                         if np.isfinite(curves[h]["retention"]) else -np.inf)
            per_index[idx] = {"curves": {str(h): curves[h] for h in h_list},
                              "delta_ideal": float(delta_ideal),
                              "h_star": int(h_star)}
        splits[name] = {"n_cycles": int(mask.sum()), "per_index": per_index}

    # measured detection from the committed fixed-protocol metrics
    metrics_path = REPO_ROOT / args.metrics_dir / \
        f"metrics_{arm}_sev_{arm}_s{args.checkpoint_seed}.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cells = metrics["per_seed"][0]["cells"]
    measured = {idx: float(cells[f"{args.level}/{args.shape}/{idx}"]["detection_rate"])
                for idx in args.index}

    verdicts = {}
    for idx in args.index:
        h_star = splits["v_cal"]["per_index"][idx]["h_star"]
        ret_cal = splits["v_cal"]["per_index"][idx]["curves"][str(h_star)]["retention"]
        ret_test = splits["v_test"]["per_index"][idx]["curves"][str(h_star)]["retention"]
        if not np.isfinite(ret_test):
            raise ValueError(f"[{arm}] {idx}: retention on V_test is not finite")
        prof = splits["v_test"]["per_index"][idx]["curves"][str(h_star)]["profile"]
        oracle = gaussian_oracle(prof, far=args.far, t_bins=t_bins,
                                 window=args.window, n_sims=args.n_sims)
        gap = oracle["power"] - measured[idx]
        verdicts[idx] = {"h_star": h_star,
                         "retention_v_cal": ret_cal, "retention_v_test": ret_test,
                         "delta_win_v_test": splits["v_test"]["per_index"][idx]
                             ["curves"][str(h_star)]["delta_win"],
                         "delta_ideal": splits["v_test"]["per_index"][idx]["delta_ideal"],
                         "measured_detection": measured[idx],
                         "oracle_power": oracle["power"],
                         "gap": gap, "decision": decision_for_gap(gap),
                         "q_gate": {
                             "recoverable": bool(ret_cal >= Q_RECOVERABLE
                                                 and ret_test >= Q_RECOVERABLE),
                             "branch_a": bool(max(ret_cal, ret_test) < Q_DEAD)}}

    decisions = {idx: v["decision"] for idx, v in verdicts.items()}
    overall = max(verdicts.values(), key=lambda v: v["gap"])["decision"]
    report = {
        "run_id": args.run_id, "arm": arm,
        "checkpoint_seed": args.checkpoint_seed,
        "git_sha": subprocess_git_sha(),
        "level": args.level, "shape": args.shape, "indices": args.index,
        "delta_ndvi": delta, "h_list": h_list, "far_target": args.far,
        "oracle_window": args.window, "device": device,
        "v_pool": {"n_cycles": n_all,
                   "n_fields": int(len(set(V["field_ids"].tolist()))),
                   "n_cal": int(is_cal.sum()), "n_test": int((~is_cal).sum()),
                   "split_sha16": field_blocked_split(
                       set(V["field_ids"].tolist()), seed=args.seed)[2],
                   "disjoint_from_train_dev_test": True},
        "parity_gap1_vs_predict_pixels": parity_ok,
        "splits": splits,
        "verdicts": verdicts,
        "decision_per_index": decisions,
        "decision_overall": overall,
        "caveats": [
            "iid-Gaussian evidence oracle is optimistic; heavy tails lower true power",
            "the channel-max multiplicity penalty is not in the oracle",
            "V shares fields with train (243/251) — acceptable for a frozen-scorer "
            "ceiling audit, not for confirmatory gates",
            "retention gates replace the consultant's Q_h form 1:1 (same "
            "ideal-fraction semantics, measured on the window statistic)",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"oracle_floor_{arm}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[{arm}] report → {out}", flush=True)
    for idx in args.index:
        v = verdicts[idx]
        print(f"  {idx}: h*={v['h_star']} retention={v['retention_v_test']:.3f} "
              f"δ_win={v['delta_win_v_test']:.3f}σ ideal={v['delta_ideal']:.3f}σ "
              f"oracle={v['oracle_power']:.3f} measured={v['measured_detection']:.3f} "
              f"gap={v['gap']:+.3f} → {v['decision']} "
              f"(recoverable={v['q_gate']['recoverable']}, "
              f"branchA={v['q_gate']['branch_a']})", flush=True)
    print(f"  OVERALL → {overall}", flush=True)
    return report


def subprocess_git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default="patch_full_v2_ep10")
    ap.add_argument("--arm", action="append", required=True,
                    choices=("patch_pixelwise", "patch_cnn_gru"))
    ap.add_argument("--store", default="data/patch_runs/v3/patches.zarr")
    ap.add_argument("--splits", default="data/processed/splits_locked_v2.json")
    ap.add_argument("--metrics-dir", default="reports/dev/patch_full_v2_ep10")
    ap.add_argument("--level", default="S6")
    ap.add_argument("--shape", default="persistent", choices=("persistent", "acute"))
    ap.add_argument("--index", action="append", default=None,
                    choices=("ndvi", "savi"), help="default: both")
    ap.add_argument("--h-list", default="1,2,3,4,6,8")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N V cycles")
    ap.add_argument("--checkpoint-seed", type=int, default=13)
    ap.add_argument("--seed", type=int, default=13, help="field-blocked split seed")
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--n-sims", type=int, default=50_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    args.h_list = [int(h) for h in args.h_list.split(",") if h.strip()]
    if not args.h_list or min(args.h_list) != 1:
        raise SystemExit("--h-list must start at 1 (the parity baseline)")
    args.index = args.index or ["ndvi", "savi"]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / args.metrics_dir

    manifest = json.loads((REPO_ROOT / args.splits).read_text(encoding="utf-8"))
    store_path = Path(args.store) if Path(args.store).is_absolute() \
        else REPO_ROOT / args.store
    from ssl_anomaly.io import patches as P
    root = P.open_store(store_path)

    print("=== V pool ===", flush=True)
    V = build_v_pool(manifest, root)
    if args.limit:
        for k in ("cycle_keys", "field_ids", "cycle_ids", "start_dates", "end_dates"):
            V[k] = V[k][:args.limit]
    cal_fields, test_fields, split_hash = field_blocked_split(
        set(V["field_ids"].tolist()), seed=args.seed)
    is_cal = np.array([f in cal_fields for f in V["field_ids"]])
    print(f"V cycles={len(V['cycle_keys'])} "
          f"fields={len(set(V['field_ids'].tolist()))} "
          f"(cal {int(is_cal.sum())} / test {int((~is_cal).sum())}; split {split_hash})",
          flush=True)

    for arm in args.arm:
        audit_arm(arm, args, V, is_cal, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
