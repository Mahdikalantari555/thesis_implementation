# Tasks — patch-data-pipeline

## 1. Geometry + reader (TDD-first)

- [ ] 1.1 Unit tests: bbox crop/pad to 32×32 (no resampling) for the
      measured size extremes (2.27 ha and 34.6 ha fields); polygon mask
      excludes neighbouring fields; inner-buffer flag default off.
- [ ] 1.2 `src/ssl_anomaly/io/patches.py`: geometry module (bbox origins,
      pad, polygon rasterization on the patch grid) + zarr writer/reader
      with schema validation (io/stats_zarr.py pattern: dims, dtypes,
      NaN discipline).
- [ ] 1.3 Unit tests: per-pixel per-bin median aggregation on a synthetic
      tiny cube (known medians, NaN propagation, n_obs and sensor_id
      recording, valid_frac semantics identical to stats.zarr).

## 2. Exporter

- [ ] 2.1 `scripts/extract_patches.py`: scene loop reusing the
      extract_pixels.py source resolution + io/landsat.py QA/scale handling
      verbatim; writes patches.zarr + `.manifest.jsonl`; flags
      --fields/--dates/--smoke/--estimate-runtime/--resume/--patch-size
      (default locked 32).
- [ ] 2.2 Normalization artifact builder: per-channel × per-sensor
      median/IQR on train fields × train periods ONLY; frozen sidecar with
      sha256 + splits sha; unit test proving dev/test fields contribute
      nothing.
- [ ] 2.3 Provenance manifest (products, QA bits, scale factors, git sha,
      cycles sha, splits sha, output sha256) + schema validation test.
- [ ] 2.4 Agent executes `--smoke` and `--estimate-runtime`, reviews the
      output (shapes, NaN discipline, plausible QA fractions), then hands
      the FULL run to the user as an exact command (conda env geo, cwd,
      expected wall time from the estimate, output paths) — the agent
      never launches it.

## 3. Verification gates

- [ ] 3.1 New-module pytest green under the geo env
      (`D:\Dev\Miniforge3\envs\geo\python.exe -X utf8 -m pytest`).
- [ ] 3.2 After the user-run extraction: verify store schema, counts and
      hashes; record estimated vs actual wall time; confirm the existing
      suite (contract + bilingual 50) stays green and untouched.
