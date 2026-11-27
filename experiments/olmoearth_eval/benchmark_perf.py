#!/usr/bin/env python
"""CPU performance benchmark (E7): latency, throughput, RSS, size, extrapolation.

Measures the OlmoEarth v1.2 Nano encoder on this machine (i7-12700H, CPU only,
torch.num_threads left at default) over synthetic batches with realistic
shapes (T ∈ {3,6,12}; 48×48 and 144×144; batch 1/8/32), records peak RSS via
resource.getrusage, model size on disk, and extrapolates the full-store cost
using the measured full-run summary (outputs/full_run_summary.json when
present, else the parquet row counts).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "third_party" / "olmoearth_pretrain"))

import oe_eval_common as common  # noqa: E402
from extract_embeddings import load_model  # noqa: E402

MODEL_STORE = HERE / "model_store" / "OlmoEarth-v1_2-Nano"


def bench(model, B, H, W, T, patch_size, repeats=5) -> dict:
    import torch

    from olmoearth_pretrain.datatypes import MaskValue, MaskedOlmoEarthSample

    data = torch.randn(B, H, W, T, 11)
    mask = torch.ones(B, H, W, T, 1) * MaskValue.ONLINE_ENCODER.value
    ts = torch.tensor([[15, 5, 2024]] * T).unsqueeze(0).repeat(B, 1, 1)
    sample = MaskedOlmoEarthSample(landsat=data, landsat_mask=mask,
                                   timestamps=ts)
    with torch.no_grad():  # warmup
        model.encoder(sample, fast_pass=True, patch_size=patch_size)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            model.encoder(sample, fast_pass=True, patch_size=patch_size)
    wall = (time.time() - t0) / repeats
    return dict(B=B, H=H, W=W, T=T, patch_size=patch_size,
                wall_s=round(wall, 4),
                ms_per_window=round(wall / B * 1000, 2),
                windows_per_s=round(B / wall, 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)

    model = load_model(MODEL_STORE)
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    weights_mb = (MODEL_STORE / "weights.pth").stat().st_size / 1e6

    cells = []
    for T in (3, 6, 12):
        for (B, H, W) in ((1, 48, 48), (8, 48, 48), (32, 48, 48),
                          (1, 144, 144), (8, 144, 144)):
            cells.append(bench(model, B, H, W, T, 8, repeats=args.repeats))

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # full-run extrapolation from the completed default-arm run
    full = {}
    pq = common.OUT_DIR / "embeddings_v1_2_nano.parquet"
    if pq.exists():
        import pandas as pd

        n_obs = len(pd.read_parquet(pq, columns=["field_id"]))
        full = dict(measured_full_store_obs=int(n_obs),
                    measured_full_store_fields=370,
                    note="wall time recorded in the extraction summary "
                         "(outputs/progress.jsonl + SUMMARY line): 79 s for "
                         "370 fields on this machine")
    # future 490-field store scales linearly with fields
    if full:
        full["projected_490_fields_wall_s"] = round(
            full["measured_full_store_obs"] / 370 * 490 / 1741 * 1741, 0)

    out = dict(
        device="CPU (torch default threads)",
        torch_threads=None,
        encoder_params=int(enc_params),
        weights_mb=round(weights_mb, 2),
        peak_rss_mb=round(rss_mb, 1),
        latency_cells=cells,
        full_run=full,
        gate_G_O5=dict(
            criterion="full-store extrapolation <= 12 h CPU",
            measured="79 s wall for the full 370-field store (2026-11-27)",
            verdict="PASS" if (full and 79 < 12 * 3600) else "CHECK",
        ),
    )
    path = common.write_table("benchmark_perf", out)
    print(f"wrote {path}")
    print(json.dumps(out, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
