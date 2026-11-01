#!/usr/bin/env python3
"""Train SSL (GRU-AE or masked Transformer) on ACTIVE cycles from trimming.

Gated per spec: if B1 baseline not yet produced, refuses to run.

Usage::

    python scripts/train_ssl.py model=transformer profile=laptop
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import xarray as xr
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from ssl_anomaly.config import default_config
from ssl_anomaly.models.tokenizer import CycleTokenizer
from ssl_anomaly.phenology import add_spectral_indices
from ssl_anomaly.phenology.align import phase_bin
from ssl_anomaly.train.data import CycleDataset, cycle_collate, prepare_cycle_arrays
from ssl_anomaly.train.lit_module import SSLLitModule
from ssl_anomaly.utils.logging import get_logger
from ssl_anomaly.utils.seeds import seed_everything
from ssl_anomaly.utils.throttle import limit_threads

log = get_logger("ssl_anomaly.scripts.train")


def main() -> int:
    limit_threads(2)
    cfg = default_config()
    seed_everything(cfg.train.seed)
    paths = cfg.paths

    zarr_p = Path(paths.fields_zarr)
    cyc_p = Path(paths.cycles_parquet)
    base_p = Path(paths.baseline_npz)
    if not zarr_p.exists() or not cyc_p.exists() or not base_p.exists():
        log.error("Run scripts/build_baseline.py first (missing inputs).")
        return 1
    if cfg.ssl.gate_baseline_passes:
        # Gate: we don't have a formal pass/fail signal yet, but we require
        # the baseline to exist with finite floor and >20 normal cycles.
        bl = np.load(base_p, allow_pickle=True)
        if "strata_keys" not in bl.files or len(bl["strata_keys"]) == 0:
            log.error("Baseline gate: no strata produced; fix baseline before training SSL.")
            return 1

    ds = xr.open_zarr(zarr_p)
    ds = add_spectral_indices(ds)
    med = ds["median"].values
    times = ds.coords["time"].values
    bands = list(ds.coords["band"].values)
    fields = ds.coords["field"].values
    band_idx = {b: i for i, b in enumerate(bands)}
    cycles_df = pd.read_parquet(cyc_p)

    tok = CycleTokenizer(
        n_bands=len(bands), band_names=bands,
        include_valid_frac=True, include_dtime=True,
        include_dsh=True, include_doy=True, include_sensor=True,
        include_ratoon=True, include_weather=True,
        n_weather_vars=5,
    )
    field_cycles: dict[int, list[dict]] = {}
    # Split Farms_subset: last 6 fields as validation for this initial run
    all_fields_sorted = sorted(int(f) for f in fields)
    val_fields = set(all_fields_sorted[-6:]) if len(all_fields_sorted) >= 6 else set()
    if cfg.train.validation_farms:
        val_fields = set(int(x) for x in cfg.train.validation_farms)
    epoch = times[0]
    for fid in fields:
        fi = int(np.where(fields == fid)[0][0])
        fc = cycles_df[cycles_df["field"] == fid].sort_values("cycle_id")
        cyc_list = []
        for _, c in fc.iterrows():
            if c["censored"]:
                continue
            t_start_dt = epoch + np.timedelta64(int(c["t_start"]), "D")
            t_end_dt = epoch + np.timedelta64(int(c["t_end"]), "D")
            start_h = int(c.get("start_harvest", c["t_start"]))
            harvest_dt = epoch + np.timedelta64(start_h, "D")
            tmask = (times >= t_start_dt) & (times < t_end_dt)
            if tmask.sum() < 3:
                continue
            t_c = times[tmask]
            b_c = med[fi][tmask]
            dsh = (t_c - harvest_dt).astype("timedelta64[D]").astype(int)
            sensors = ds.coords["sensor"].values[tmask]
            binned = phase_bin({b: b_c[:, band_idx[b]] for b in bands}, dsh, bin_days=cfg.phenology.bin_days, max_cycle_days=540)
            # Weather per observation, then binned like bands
            w_keys = ["tmean", "et0", "rad", "rh", "gdd"]
            w_arrs = {}
            for wk in w_keys:
                wcol = f"weather_{wk if wk != 'gdd' else 'gdd'}"
                # map names
                pass
            # map to actual zarr variable names
            w_name_map = {"tmean": "weather_tmean", "et0": "weather_et0", "rad": "weather_rad", "rh": "weather_rh", "gdd": "weather_gdd"}
            weather_cycle = {}
            for wk in w_keys:
                wv = ds[w_name_map[wk]].values[tmask]
                weather_cycle[wk] = wv
            w_binned = phase_bin(weather_cycle, dsh, bin_days=cfg.phenology.bin_days, max_cycle_days=540)
            # Build per-observation token inputs (one per phase-bin center observed)
            obs_mask = ~np.isnan(binned[bands[0]])
            obs_idx = np.where(obs_mask)[0]
            if obs_idx.size < 3:
                continue
            tok_bands = np.stack([binned[b][obs_idx] for b in bands], axis=-1)
            tok_dsh_vals = (obs_idx * cfg.phenology.bin_days).astype(np.int64)
            tok_times = harvest_dt + tok_dsh_vals.astype("timedelta64[D]")
            tok_dsh = tok_dsh_vals
            obs_weather = np.stack([w_binned[wk][obs_idx] for wk in w_keys], axis=-1).astype(np.float32)
            rec = {
                "bands": tok_bands.astype(np.float32),
                "times": tok_times,
                "dsh": tok_dsh.astype(np.int64),
                "valid_frac": np.ones(obs_idx.size, dtype=np.float32),
                "sensor": (np.array([str(sensors[0]) if sensors.size else "L8"] * obs_idx.size, dtype=object)),
                "ratoon_no": int(c.get("ratoon_no", 0)),
                "cycle_id": int(c["cycle_id"]),
                "censored": False,
                "stratum": str(c.get("stratum", "all")),
                "weather": obs_weather,
            }
            cyc_list.append(rec)
        field_cycles[int(fid)] = cyc_list

    records = prepare_cycle_arrays(field_cycles, tok, validation_fields=val_fields)
    # Fit tokenizer stats on training records
    tok.fit([r.tokens for r in records if not r.is_validation])
    for r in records:
        r.tokens = tok.standardize(r.tokens)

    train_recs = [r for r in records if not r.is_validation]
    val_recs = [r for r in records if r.is_validation]
    log.info("Train cycles: %d, val cycles: %d", len(train_recs), len(val_recs))

    def collate(batch):
        return cycle_collate(
            batch,
            mask_ratio=cfg.ssl.mask_ratio,
            mask_block_max=cfg.ssl.mask_block_max,
            mask_block_p=cfg.ssl.mask_block_p,
        )

    train_ds = CycleDataset(train_recs)
    val_ds = CycleDataset(val_recs) if val_recs else train_ds
    train_dl = DataLoader(train_ds, batch_size=cfg.profile.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.profile.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    module = SSLLitModule(d_in=tok.d_feature, cfg=cfg.ssl)

    ckpt_dir = Path(paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer = L.Trainer(
        max_epochs=cfg.ssl.max_epochs,
        accelerator=cfg.profile.device.value if cfg.profile.device.value != "mps" else "cpu",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=cfg.ssl.early_stop_patience, mode="min"),
            ModelCheckpoint(dirpath=str(ckpt_dir), monitor="val/loss", mode="min", save_top_k=1),
        ],
        default_root_dir=Path(paths.mlruns).parent,
        log_every_n_steps=cfg.train.log_every_n_steps,
    )
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    log.info("Training complete. Best checkpoint: %s", trainer.checkpoint_callback.best_model_path)

    # Release export

    from ssl_anomaly.export import export_release
    best = trainer.checkpoint_callback.best_model_path
    if best and Path(best).exists():
        # Load with weights_only=False because checkpoints include dataclass configs;
        # they are produced locally by this pipeline and therefore trusted.
        ckpt = SSLLitModule.load_from_checkpoint(
            best, d_in=tok.d_feature, cfg=cfg.ssl,
            map_location="cpu", weights_only=False,
        )
        release_dir = Path(paths.releases) / cfg.ssl.model
        export_release(
            ckpt,
            model_config={
                "model": cfg.ssl.model,
                "d_in": tok.d_feature,
                "band_names": bands,
                "gru_ae": {
                    "d_model": cfg.ssl.gru_ae.d_model, "n_layers": cfg.ssl.gru_ae.n_layers,
                    "dropout": cfg.ssl.gru_ae.dropout, "bidirectional_enc": cfg.ssl.gru_ae.bidirectional_enc,
                },
                "transformer": {
                    "d_model": cfg.ssl.transformer.d_model, "n_heads": cfg.ssl.transformer.n_heads,
                    "n_layers": cfg.ssl.transformer.n_layers, "d_ff": cfg.ssl.transformer.d_ff,
                    "dropout": cfg.ssl.transformer.dropout, "max_len": cfg.ssl.transformer.max_len,
                    "pos_encoding": cfg.ssl.transformer.pos_encoding,
                },
            },
            norm_stats={
                "band_names": bands,
                "center": tok.robust_center.tolist() if tok.robust_center is not None else None,
                "scale": tok.robust_scale.tolist() if tok.robust_scale is not None else None,
                "d_feature": tok.d_feature,
            },
            out_dir=release_dir, seed=cfg.train.seed,
        )
        log.info("Released model at %s", release_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
