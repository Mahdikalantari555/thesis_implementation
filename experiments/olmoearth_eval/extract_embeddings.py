#!/usr/bin/env python
"""Extract frozen OlmoEarth v1.2 Nano embeddings from the thesis patch store.

Isolated experimental pipeline (openspec change ``evaluate-olmoearth-nano-sugarcane``).
Reads ``data/processed/patches.zarr`` (370 fields x 8 cycles x 36 15-day bins x
8 channels x 48x48) and writes observation-level embeddings (Zarr + Parquet)
and farm-cycle-level embeddings (Parquet) plus provenance, under
``experiments/olmoearth_eval/outputs/``.

Arms:
  default   missing bands filled 0.0 post-norm, native 48x48 grid
  fill05    missing bands filled 0.5 (band mean) instead of 0.0
  b11dup    B11 filled with the B10 (thermal) value instead of 0.0
  up144     bilinear upsample 48x48 -> 144x144 (10 m-equivalent GSD)
Robustness modifiers:
  --drop-bins K   drop K random valid bins per cycle before packing
  --rebin-days D  aggregate bins to D-day medians (30 or 60) before packing

The adapter chain is unit-tested in tests/unit/test_olmoearth_eval_adapter.py.

Usage:
  python extract_embeddings.py --smoke
  python extract_embeddings.py
  python extract_embeddings.py --arm up144 --suffix up144
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THIRD_PARTY = HERE / "third_party" / "olmoearth_pretrain"
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(THIRD_PARTY))  # vendored olmoearth_pretrain (pinned commit)

import oe_adapter as oe  # noqa: E402
from oe_adapter import ARCHIVE_EPOCH, MAX_PATCH_SIZE  # noqa: E402

PATCH_CHANNEL_ORDER = ["ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst"]
PATCH_STORE = REPO_ROOT / "data" / "processed" / "patches.zarr"
WEIGHTS_SHA256 = "2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378"
CONFIG_SHA256 = "4cd2888e79dc543f262cc3d86fcd30d667068fd53a728ca5bd306d5ddb509d1d"
MODEL_CODE_COMMIT = "6583a536c428ad004bb3c223add0576087d67e92"
DEFAULT_OUT = HERE / "outputs"
EMBED_DIM = 128


def load_model(model_store: Path):
    """Load the SSL checkpoint (hash-verified); only .encoder is used."""
    import torch

    from olmoearth_pretrain.model_loader import load_model_from_path

    oe.verify_sha256(model_store / "weights.pth", WEIGHTS_SHA256, "model weights")
    oe.verify_sha256(model_store / "config.json", CONFIG_SHA256, "model config")
    model = load_model_from_path(str(model_store))
    model.eval()
    return model


def rebin_store_arrays(values, pixel_valid, n_obs, sensor_id, rebin_days):
    """Aggregate 15-day bins into rebin_days-day bins (median of valid bins)."""
    step = rebin_days // 15
    n_out = len(range(0, 36, step))
    v_out = np.full((n_out,) + values.shape[1:], np.nan, dtype=np.float64)
    pv_out = np.zeros((n_out,) + pixel_valid.shape[1:], dtype=np.uint8)
    n_out_arr = np.zeros(n_out, dtype=np.int64)
    s_out = np.zeros(n_out, dtype=np.uint8)
    for g, start in enumerate(range(0, 36, step)):
        group = slice(start, min(start + step, 36))
        obs_mask = n_obs[group] > 0
        if not obs_mask.any():
            continue
        stack = values[group][obs_mask].astype(np.float64)
        with np.errstate(all="ignore"):
            v_out[g] = np.nanmedian(stack, axis=0)
        pv_out[g] = np.logical_and.reduce(
            pixel_valid[group][obs_mask][:, 0], axis=0).astype(np.uint8)[None]
        n_out_arr[g] = int(n_obs[group].sum())
        counts = np.bincount(sensor_id[group][obs_mask].astype(int), minlength=3)
        s_out[g] = int(np.argmax(counts))
    return v_out, pv_out, n_out_arr, s_out


def build_field_windows(
    values, pixel_valid, n_obs, sensor_id, in_field, cycle_start_day,
    cycle_index, stats, *, arm="default", fill=0.0, drop_bins=0, rng=None,
    min_valid_frac=0.30,
):
    """Assemble model inputs for every window of one field-cycle.

    Empty/invalid bins never become timesteps; invalid pixels get the model's
    MISSING mask; missing bands are value-filled post-normalization.
    """
    v = values.astype(np.float64)
    pv, n, sid = pixel_valid, n_obs, sensor_id
    if drop_bins and rng is not None:
        cand = np.flatnonzero(n_obs > 0)
        if len(cand) > drop_bins:
            keep = np.ones(36, dtype=bool)
            keep[rng.choice(cand, size=drop_bins, replace=False)] = False
            v = v.copy()
            v[~keep] = np.nan
            pv = pv.copy()
            pv[~keep] = 0
            n = n.copy()
            n[~keep] = 0
            sid = sid.copy()
            sid[~keep] = 0

    valid = oe.select_valid_bins(n, pv, in_field, min_valid_frac=min_valid_frac)
    windows = []
    for w_idx, bins in enumerate(oe.pack_windows(valid)):
        per_bin = []
        for b in bins:
            ch = {name: v[b, i] for i, name in enumerate(PATCH_CHANNEL_ORDER)}
            norm = oe.assemble_and_normalize(
                ch, stats,
                band_fill=fill,
                b11_from_b10=(arm == "b11dup"))
            per_bin.append(norm)
        x = np.stack(per_bin, axis=0)                    # (T,H,W,11)
        mask_px = pv[bins, 0].astype(bool)               # (T,H,W)
        ts = []
        for b in bins:
            d = oe.day_to_date(oe.bin_center_day(int(cycle_start_day), int(b)))
            ts.append([d.day, d.month - 1, d.year])
        windows.append(dict(
            x=x, mask_px=mask_px, timestamps=np.asarray(ts, dtype=np.int64),
            bins=[int(b) for b in bins], window_index=w_idx,
            cycle_index=cycle_index))
    return windows


def upsample_window(w: dict, size: int = 144) -> dict:
    """Bilinear-upsample the normalized input; nearest-upsample the mask."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(w["x"])).permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(
        x, size=(size, size), mode="bilinear", align_corners=False)
    m = torch.from_numpy(w["mask_px"][None].astype(np.float32))
    m = torch.nn.functional.interpolate(m, size=(size, size), mode="nearest")[0]
    out = dict(w)
    out["x"] = x.permute(0, 2, 3, 1).numpy()
    out["mask_px"] = m.numpy() > 0.5
    return out


def run_windows(model, windows, patch_size=MAX_PATCH_SIZE):
    """Encode a batch of windows -> per-window ONLINE-pooled (t,128) arrays."""
    import torch

    from olmoearth_pretrain.datatypes import MaskValue, MaskedOlmoEarthSample

    B = len(windows)
    T = max(w["x"].shape[0] for w in windows)
    H, W = windows[0]["x"].shape[1:3]
    data = np.zeros((B, H, W, T, 11), dtype=np.float32)
    mask = np.full((B, H, W, T, 1), MaskValue.MISSING.value, dtype=np.float32)
    ts = np.zeros((B, T, 3), dtype=np.int64)
    for i, w in enumerate(windows):
        t = w["x"].shape[0]
        data[i, :, :, :t] = w["x"].transpose(1, 2, 0, 3)   # (T,H,W,11)->(H,W,T,11)
        mask[i, :, :, :t, 0] = np.where(
            w["mask_px"].transpose(1, 2, 0),
            MaskValue.ONLINE_ENCODER.value, MaskValue.MISSING.value)
        ts[i, :t] = w["timestamps"]
    sample = MaskedOlmoEarthSample(
        landsat=torch.from_numpy(data),
        landsat_mask=torch.from_numpy(mask),
        timestamps=torch.from_numpy(ts),
    )
    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=patch_size)
    tokens = out["tokens_and_masks"].landsat           # (B,H',W',T,S,128)
    tok_mask = out["tokens_and_masks"].landsat_mask    # (B,H',W',T,S)
    online = tok_mask == MaskValue.ONLINE_ENCODER.value
    pooled = []
    for i, w in enumerate(windows):
        t = w["x"].shape[0]
        embs = []
        for ti in range(t):
            s = online[i, :, :, ti, :]
            if not bool(s.any()):
                raise RuntimeError(
                    f"window {i} timestep {ti} has zero ONLINE tokens — "
                    "mask construction defect")
            embs.append(tokens[i, :, :, ti, :][s].mean(dim=0).cpu().numpy())
        pooled.append(np.stack(embs, axis=0))          # (t,128)
    return pooled


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()[:12]
    except Exception:
        return "unknown"


def _read_store_provenance(store_path: Path) -> dict:
    try:
        with open(store_path / "provenance.json", encoding="utf-8") as f:
            prov = json.load(f)
        keys = ("channel_order", "band_scale", "patch_size", "n_fields",
                "n_cycles", "epoch", "qa_bits", "git_sha", "cycles_sha256")
        return {k: prov.get(k) for k in keys}
    except Exception:
        return {}


def extract(*, store_path, model_store, out_dir, arm, suffix, fields,
            max_cycles, drop_bins, rebin_days, seed, min_valid_frac,
            smoke, batch_windows) -> dict:
    import torch
    import zarr
    from tqdm import tqdm

    from olmoearth_pretrain.datatypes import MaskValue  # noqa: F401 (semantics doc)

    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / f"progress{suffix}.jsonl"
    manifest_path = out_dir / f"manifest{suffix}.jsonl"
    done_fields: set[int] = set()
    if manifest_path.exists() and not smoke:
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                done_fields.add(int(json.loads(line)["field_id"]))

    stats = oe.load_computed_norm(
        THIRD_PARTY / "olmoearth_pretrain" / "data"
        / "norm_configs" / "computed.json")

    store = zarr.open_group(str(store_path), mode="r")
    field_ids = store["field_ids"][:]
    if fields is None:
        sel_fields = [int(f) for f in field_ids]
    else:
        known = {int(f) for f in field_ids}
        missing = sorted(set(fields) - known)
        if missing:
            raise SystemExit(f"fields not in patch store: {missing}")
        sel_fields = list(fields)
    if smoke:
        sel_fields = sel_fields[:3]

    model = load_model(model_store)
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    torch.manual_seed(seed)

    obs_n_windows_total = 0
    forward_s = 0.0
    shard_dir = out_dir / f"shards{suffix}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    progress_f = open(progress_path, "a", encoding="utf-8")
    pbar = tqdm(sel_fields, desc="fields", unit="field", disable=smoke)
    for field_id in pbar:
        if field_id in done_fields:
            continue
        fi = int(np.flatnonzero(field_ids == field_id)[0])
        cycle_valid = store["cycle_valid"][fi].astype(bool)
        in_field = store["in_field"][fi, 0]
        in_field_bool = in_field.astype(bool)
        field_obs: list[dict] = []
        field_emb: list[np.ndarray] = []
        field_cycles: list[dict] = []
        for ci in range(min(max_cycles, 8)):
            if not cycle_valid[ci]:
                continue
            values = store["values"][fi, ci]
            pv = store["pixel_valid"][fi, ci]
            n_obs = store["n_obs"][fi, ci]
            sid = store["sensor_id"][fi, ci]
            if rebin_days:
                values, pv, n_obs, sid = rebin_store_arrays(
                    values, pv, n_obs, sid, rebin_days)
            start_day = int(store["cycle_start_day"][fi, ci])
            windows = build_field_windows(
                values, pv, n_obs, sid, in_field, start_day, ci, stats,
                arm=arm, fill=0.5 if arm == "fill05" else 0.0,
                drop_bins=drop_bins,
                rng=np.random.default_rng(seed * 100000 + field_id * 100 + ci),
                min_valid_frac=min_valid_frac)
            if arm == "up144":
                windows = [upsample_window(w) for w in windows]
            if not windows:
                field_cycles.append(dict(
                    field_id=int(field_id), cycle_index=ci,
                    cycle_start_day=start_day,
                    cycle_end_day=int(store["cycle_end_day"][fi, ci]),
                    n_windows=0, n_bins=0, arm=arm))
                continue
            for i in range(0, len(windows), batch_windows):
                chunk = windows[i : i + batch_windows]
                t0 = time.time()
                pooled = run_windows(model, chunk)
                forward_s += time.time() - t0
                for w, emb in zip(chunk, pooled):
                    obs_n_windows_total += 1
                    field_emb.append(emb.astype(np.float32))
                    for ti, b in enumerate(w["bins"]):
                        day = oe.bin_center_day(start_day, b)
                        d = oe.day_to_date(day)
                        field_obs.append(dict(
                            field_id=int(field_id), cycle_index=ci,
                            window_index=w["window_index"], bin_index=b,
                            t_in_window=ti, day_index=day,
                            date=d.isoformat(), month=d.month, year=d.year,
                            sensor=int(sid[b]), n_obs=int(n_obs[b]),
                            valid_frac=float(pv[b, 0][in_field_bool].mean()),
                            arm=arm))
            field_cycles.append(dict(
                field_id=int(field_id), cycle_index=ci,
                cycle_start_day=start_day,
                cycle_end_day=int(store["cycle_end_day"][fi, ci]),
                n_windows=len(windows),
                n_bins=int(sum(len(w["bins"]) for w in windows)),
                arm=arm))
        # persist this field's shard BEFORE the manifest row (crash-safe order)
        shard_path = shard_dir / f"field_{field_id}.npz"
        np.savez_compressed(
            shard_path,
            emb=np.concatenate(field_emb, axis=0) if field_emb
            else np.zeros((0, EMBED_DIM), dtype=np.float32),
            obs_meta=json.dumps(field_obs).encode(),
            cycle_meta=json.dumps(field_cycles).encode(),
        )
        progress_f.write(json.dumps(dict(
            field_id=int(field_id), done=True,
            elapsed_s=round(time.time() - t_start, 1))) + "\n")
        progress_f.flush()
        with open(manifest_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(dict(field_id=int(field_id))) + "\n")
    progress_f.close()

    # ---- merge shards for every manifest field ----
    obs_rows: list[dict] = []
    emb_blocks: list[np.ndarray] = []
    cycle_rows: list[dict] = []
    merged_fields: list[int] = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        fid = int(json.loads(line)["field_id"])
        shard_path = shard_dir / f"field_{fid}.npz"
        if not shard_path.exists():
            raise RuntimeError(
                f"manifest lists field {fid} but shard {shard_path} is missing "
                "(torn resume state — rerun with the manifest cleared)")
        sh = np.load(shard_path, allow_pickle=False)
        obs_rows.extend(json.loads(sh["obs_meta"].tobytes().decode()))
        emb_blocks.append(sh["emb"])
        cycle_rows.extend(json.loads(sh["cycle_meta"].tobytes().decode()))
        merged_fields.append(fid)

    if len(obs_rows) != sum(b.shape[0] for b in emb_blocks):
        raise RuntimeError("obs rows and embedding blocks desynchronized")

    all_emb = (np.concatenate(emb_blocks, axis=0) if emb_blocks
               else np.zeros((0, EMBED_DIM), dtype=np.float32))
    df = pd.DataFrame(obs_rows)
    emb_cols = [f"e{i:03d}" for i in range(EMBED_DIM)]
    emb_df = pd.DataFrame(all_emb, columns=emb_cols)

    # Farm-cycle level: mean of observation embeddings per (field, cycle).
    group_src = pd.concat([df[["field_id", "cycle_index"]], emb_df], axis=1)
    cyc_mean = group_src.groupby(["field_id", "cycle_index"], sort=True)[emb_cols] \
        .mean().reset_index()
    cyc_meta = pd.DataFrame(cycle_rows)
    cyc_out = cyc_mean.merge(cyc_meta, on=["field_id", "cycle_index"], how="left")

    # ---- write artifacts ----
    zarr_path = out_dir / f"embeddings_v1_2_nano{suffix}.zarr"
    zg = zarr.open_group(str(zarr_path), mode="w")
    zg.create_array("emb_obs", data=all_emb,
                    chunks=(min(4096, max(1, len(all_emb))), EMBED_DIM))
    prov = dict(
        model="OlmoEarth-v1_2-Nano",
        weights_sha256=WEIGHTS_SHA256,
        config_sha256=CONFIG_SHA256,
        model_code_commit=MODEL_CODE_COMMIT,
        adapter_git_sha=_git_sha(),
        data_source=str(store_path),
        patch_store_provenance=_read_store_provenance(store_path),
        normalization="computed.json (x-(mean-2std))/(4std)",
        band_map={b: (oe.LANDSAT_SLOT_SOURCES[b] or f"FILL_{fill_tag}")
                  for b, fill_tag in
                  ((bb, "0.0" if arm != "fill05" else "0.5")
                   for bb in oe.OLMOEARTH_LANDSAT_BANDS)},
        missing_slots=list(oe.MISSING_LANDSAT_SLOTS),
        arm=arm, suffix=suffix, drop_bins=drop_bins, rebin_days=rebin_days,
        seed=seed, min_valid_frac=min_valid_frac,
        patch_size=MAX_PATCH_SIZE, embed_dim=EMBED_DIM,
        encoder_params=int(enc_params), epoch=str(ARCHIVE_EPOCH),
        created=date.today().isoformat(), smoke=bool(smoke),
        validation_ladder="L1-L2 only",
        note="Derived labels and injected deviations are synthetic/derived, "
             "never ground truth; no test-period (>=2024-01-01) claims.",
    )
    with open(zarr_path / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, default=str)

    parquet_path = out_dir / f"embeddings_v1_2_nano{suffix}.parquet"
    df_out = pd.concat([df.reset_index(drop=True),
                        emb_df.reset_index(drop=True)], axis=1)
    df_out.to_parquet(parquet_path, index=False)
    cyc_path = out_dir / f"farmcycle_v1_2_nano{suffix}.parquet"
    cyc_out.to_parquet(cyc_path, index=False)

    elapsed = time.time() - t_start
    summary = dict(
        n_obs_rows=int(len(obs_rows)), n_cycle_rows=int(len(cyc_out)),
        n_windows=int(obs_n_windows_total), n_fields=len(merged_fields),
        encoder_params=int(enc_params),
        finite=bool(np.isfinite(all_emb).all()) if len(all_emb) else False,
        all_zero=bool(len(all_emb) and np.abs(all_emb).max() == 0.0),
        emb_abs_mean=float(np.abs(all_emb).mean()) if len(all_emb) else None,
        forward_seconds=round(forward_s, 2),
        sec_per_window=round(forward_s / obs_n_windows_total, 4) if obs_n_windows_total else None,
        wall_time_s=round(elapsed, 1),
        outputs=[str(zarr_path), str(parquet_path), str(cyc_path)],
        smoke=bool(smoke), arm=arm, suffix=suffix,
    )
    print("SUMMARY " + json.dumps(summary))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=PATCH_STORE)
    ap.add_argument("--model-store", type=Path,
                    default=HERE / "model_store" / "OlmoEarth-v1_2-Nano")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arm", choices=["default", "fill05", "b11dup", "up144"],
                    default="default")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--fields", type=int, nargs="*", default=None)
    ap.add_argument("--max-cycles", type=int, default=8)
    ap.add_argument("--drop-bins", type=int, default=0)
    ap.add_argument("--rebin-days", type=int, default=0, choices=[0, 30, 60])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-valid-frac", type=float, default=0.30)
    ap.add_argument("--batch-windows", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    suffix = args.suffix
    if args.smoke and not suffix:
        suffix = "_smoke"
    summary = extract(
        store_path=args.store, model_store=args.model_store, out_dir=args.out,
        arm=args.arm, suffix=suffix, fields=args.fields,
        max_cycles=args.max_cycles, drop_bins=args.drop_bins,
        rebin_days=args.rebin_days, seed=args.seed,
        min_valid_frac=args.min_valid_frac, smoke=args.smoke,
        batch_windows=args.batch_windows)
    if summary["n_obs_rows"] == 0:
        print("DEFECT: no embeddings produced", file=sys.stderr)
        return 1
    if not summary["finite"]:
        print("DEFECT: non-finite embeddings", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
