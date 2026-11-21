# Patch severity synthetic A/B profile

Measured 2026-11-21 on the sandbox CPU, one thread, Python 3.11, NumPy 1.26.4,
PyTorch 2.15.0, Zarr 2.18.7. Independent baseline/optimized processes, one
measurement each: results are illustrative, not confidence intervals or T4
predictions. Baseline is commit `36a5fcb236258c1122fb3ba86a47d2129a7b4809`.

The fixture has 20 synthetic cycles (10 calibration/10 evaluation), shape
`36 × 8 × 48 × 48`, 25% in-field pixels, synthetic sigma, a small untrained
`d_model=8` model, and all 24 severity cells × two rounds. It is **not** the
real 340-cycle panel or its checkpoints. Chunk size 2. Each phase's peak is
process RSS sampled every 2 ms, not an independent allocation delta. Python,
PyTorch and allocator retention are included. CUDA/GPU memory is not measured.

| Architecture | Phase | Baseline peak MiB | Chunked peak MiB | Baseline s | Chunked s |
|---|---|---:|---:|---:|---:|
| pixelwise | batch build | 1041.1 | 956.0 | 0.245 | 0.232 |
| pixelwise | calibration | 1403.4 | 1340.8 | 0.811 | 0.830 |
| pixelwise | cell loop | 1662.8 | 1063.5 | 35.755 | 33.924 |
| CNN-GRU | batch build | 1040.3 | 956.0 | 0.267 | 0.219 |
| CNN-GRU | calibration | 1358.6 | 1384.3 | 0.859 | 0.861 |
| CNN-GRU | cell loop | 1631.0 | 1192.3 | 36.334 | 32.027 |

Highest phase RSS fell 19.4% (pixelwise) and 15.1% (CNN-GRU); loop
wall time fell 5.1% and 11.9%, respectively, in these single measurements.
Calibration peak varied in either direction. Both profile runs compared hashes
of all cell arrays and injection records plus calibration paths: PASS.

Reproduce from the repository root with the project's dependencies installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. python scripts/profile_patch_severity_synthetic.py --cycles 20 --chunk 2 --arch patch_pixelwise --sweep
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. python scripts/profile_patch_severity_synthetic.py --cycles 20 --chunk 2 --arch patch_cnn_gru --sweep
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. python scripts/gate_patch_severity.py --size 48 --chunk 2 --device cpu
```

The gate additionally builds actual synthetic smoke checkpoints, uses the
manifest/parquet panel authority, runs both original and current CLIs for both
architectures, and compares calibration arrays, each cell's arrays and injection
records, and metrics (excluding inherently variable `wall_time_s`). It uses
`git show` on the baseline, so that commit must exist locally. T4 must run the
same gate with `--device cuda` before using chunking; passing a fixture on T4
still does not establish bitwise equality for every real checkpoint or batch
size. No real six-run evaluation was performed in this environment.
