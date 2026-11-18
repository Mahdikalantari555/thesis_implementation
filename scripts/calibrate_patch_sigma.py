"""Fit and freeze the patch arm's per-(phenophase, channel, sensor) sigma table.

The missing producer of ``patch_sigma.npz`` (patch-level-controlled-trial
task 2.4 artifact handoff): ``eval_dev.py`` refuses to score a patch arm
without the frozen sigma table bound to its checkpoint, and nothing wrote it.

Pipeline: dev-clean cycles of the locked manifest → store batch (normalized)
→ pixel-level residuals through the trained model (unit sigma — the table is
fit on RAW residuals) → ``fit_patch_sigma`` (1.4826 x median |res| per
phenophase x channel x sensor, min_n floor + fallback) → npz carrying the
binding hashes eval_dev verifies (model_sha256, norm_sha256,
input_protocol). Dev-clean-only is the caller's guarantee, honored here by
construction: the batch is built exclusively from the manifest's ``dev`` keys.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ssl_anomaly.io import patches as P  # noqa: E402
from ssl_anomaly.models.patch_cnn_gru import build_patch_model  # noqa: E402
from ssl_anomaly.monitor import patch_eval as PE  # noqa: E402
from ssl_anomaly.monitor import patch_scores as PS  # noqa: E402
from ssl_anomaly.monitor import protocol as mproto  # noqa: E402
from ssl_anomaly.monitor import splits as msplits  # noqa: E402
from ssl_anomaly.train.patch_data import PATCH_INPUT_PROTOCOL  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="per-seed patch run dir holding last.pt + norm_stats.json")
    ap.add_argument("--store", default=str(REPO_ROOT / "data" / "processed" / "patches_poc.zarr"))
    ap.add_argument("--splits", default=str(mproto.SPLITS_LOCKED_PATH) if hasattr(mproto, "SPLITS_LOCKED_PATH")
                    else str(REPO_ROOT / "data" / "processed" / "splits_locked.json"))
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: <run-dir>/patch_sigma.npz")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    ckpt = run_dir / "last.pt"
    norm_path = run_dir / P.NORM_STATS_NAME
    for label, p in (("checkpoint", ckpt), ("norm", norm_path)):
        if not p.is_file():
            raise SystemExit(f"patch {label} not found: {p}")

    import torch

    from ssl_anomaly.train.patch_data import COVARIATES, N_COVARIATES

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    if cfg.get("input_protocol") != PATCH_INPUT_PROTOCOL:
        raise SystemExit("patch checkpoint predates the corrected input protocol; do not calibrate legacy weights")
    norm = P.load_patch_norm(norm_path)
    if norm["sha256"] != cfg.get("norm_sha256"):
        raise SystemExit("patch checkpoint and frozen normalization hashes differ")

    model = build_patch_model(
        use_cnn=bool(cfg["use_cnn"]),
        n_channels=P.N_CHANNELS,
        d_model=int(cfg["d_model"]),
        n_covariates=N_COVARIATES,
        n_horizons=int(cfg["n_horizons"]),
        stem_widths=tuple(cfg.get("stem_widths", (44, 60))),
    )
    model.load_state_dict(payload["model"] if "model" in payload else payload["state_dict"])
    model.eval()

    root = P.open_store(args.store, mode="r")
    index = P.patch_cycle_index(root)
    manifest = msplits.load_splits_locked(path=Path(args.splits))
    dev_keys = sorted(set(manifest["cycles"].get("dev", ())) & set(index.keys()))
    if not dev_keys:
        raise SystemExit("no dev cycles present in the patch store — nothing to calibrate on")
    print(f"[sigma] dev-clean calibration population: {len(dev_keys)} cycles")

    field_ids = np.array([int(k.split(":")[0]) for k in dev_keys], dtype=np.int64)
    cycle_ids = np.array([int(k.split(":")[1]) for k in dev_keys], dtype=np.int64)
    batch = PE.patch_batch_for_cycles(root, field_ids, cycle_ids, norm=norm)

    n_bins = batch["X"].shape[1]
    unit_sigma = np.full((int(np.ceil(n_bins / PS.PHASE_BIN)), P.N_CHANNELS, 3), 1.0)
    unit = PS.PatchSigmaTable(
        sigma=unit_sigma,
        fallback=np.ones(P.N_CHANNELS),
        n=np.zeros((int(np.ceil(n_bins / PS.PHASE_BIN)), P.N_CHANNELS, 3), dtype=np.int64),
        min_n=1,
        sha256=PS.sigma_table_sha256(unit_sigma, np.ones(P.N_CHANNELS),
                                     np.zeros_like(unit_sigma, dtype=np.int64)),
    )
    scorer = PS.PatchDeepScorer(
        "sigma_calibration", model, unit,
        in_field=batch["in_field"], n_channels=P.N_CHANNELS, device="cpu",
    )

    n = batch["X"].shape[0]
    res_blocks, msk_blocks = [], []
    for n0 in range(0, n, max(1, args.chunk)):
        sl = slice(n0, min(n, n0 + args.chunk))
        info = scorer.predict_pixels(
            batch["X"][sl], batch["pixel_valid"][sl],
            sensor_ids=batch["sensor_id"][sl],
            covariates=batch["covariates"][sl],
        )
        pred, mask = info["pred"], info["mask"]
        # Signed residual on observed in-field pixels; NaN elsewhere.
        res = np.where(mask[:, :, None], pred - batch["X"][sl], np.nan)
        res_blocks.append(res.astype(np.float64))
        msk_blocks.append(np.broadcast_to(mask[:, :, None], res.shape).copy())
    residuals = np.concatenate(res_blocks)   # (N, T, C, S, S)
    obs_mask = np.concatenate(msk_blocks)

    table = PS.fit_patch_sigma(
        residuals, obs_mask, batch["sensor_id"], min_n=args.min_n, n_bins=n_bins,
    )
    finite = float(np.isfinite(table.sigma).mean())
    print(f"[sigma] cells with a fitted sigma: {finite:.1%} "
          f"(min_n={args.min_n}; fallback covers the rest)")

    out_path = Path(args.out) if args.out else run_dir / "patch_sigma.npz"
    np.savez(
        out_path,
        sigma=table.sigma, fallback=table.fallback, n=table.n,
        min_n=np.int64(table.min_n), sha256=np.str_(table.sha256),
        model_sha256=np.str_(hashlib.sha256(ckpt.read_bytes()).hexdigest()),
        norm_sha256=np.str_(norm["sha256"]),
        input_protocol=np.str_(PATCH_INPUT_PROTOCOL),
        n_dev_cycles=np.int64(len(dev_keys)),
    )
    print(f"[sigma] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
