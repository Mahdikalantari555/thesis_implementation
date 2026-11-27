# Tasks — evaluate-olmoearth-nano-sugarcane

TDD-first for pure-numpy adapter logic; model-load and end-to-end paths are
covered by `--smoke` modes (no GPU on this machine). The agent never launches
long jobs — full-store extraction is handed to the owner as an exact command
after unit tests + smoke pass (AGENTS.md §7).

## 0. Registration and provenance

- [x] 0.1 Create this change (proposal / spec / design / tasks) and add the
      row to `openspec/changes/INDEX.md`; register the scope item in
      `todo.md`.
- [x] 0.2 Vendor model code (`third_party/olmoearth_pretrain` @ `6583a536`)
      and model store (`config.json` sha `4cd2888e…`, weights symlink sha
      `2773fca4…`); record license summary in the design.
- [x] 0.3 Install inference deps into `geo` only (`einops 0.8.2`,
      `universal-pathlib 0.3.11`, `huggingface_hub 2.5.0`); probe-verify
      model load (1.7 M encoder params) + CPU forward (0.099 s @ 48×48×T3).

## 1. Adapter (unit-tested, no torch required)

- [x] 1.1 `oe_adapter.py`: channel assembly table, SR→DN inversion, Kelvin→
      B10-DN inverse-Planck conversion, post-norm 0.0 fill, computed.json
      normalization. Unit tests: hand-computed values for one SR band, one
      thermal conversion (±1 DN), fill slots, normalization bounds.
      *2026-11-27: 25 unit tests pass; fills split into whole-band (0.0,
      band-dropout semantics) vs per-pixel (0.5, neutral mean) after the
      store-reality check (outside-field pixels are NaN).*
- [x] 1.2 Window packing: valid-bin selection (`n_obs>0`, ≥30 % in-field
      valid, ≥1 online patch-top-left grid point), ≤12-bin windows, bin-center
      timestamps, day/month/year fields. Unit tests on synthetic store arrays
      (empty bins skipped, ordering preserved, overflow cycles produce
      multiple windows, grid-coverage guard).
- [x] 1.3 Provenance writer: weights/config shas, commit, git sha, band map,
      seeds; hash-mismatch hard error (unit-tested).

## 2. Extraction

- [x] 2.1 `extract_embeddings.py`: per-field streaming over `patches.zarr`
      (lazy chunks), per-pixel MISSING masks, ONLINE-only spatial pooling,
      observation + farm-cycle levels, zarr+parquet output, `--smoke`,
      `--fields`, `--arm` (fill/scale variants), tqdm + `progress.jsonl`,
      crash-safe per-field shard resume. Smoke: 3 fields end-to-end PASS
      (310 obs / 30 windows / finite / 17.9 ms per window).
- [x] 2.2 Default-arm extraction over all 370 fields — **completed
      in-session (79 s wall, 22.8 ms/window)**: 17,307 obs embeddings /
      1,741 windows / 799 farm-cycles, all finite; artifacts +
      `provenance.json` under `experiments/olmoearth_eval/outputs/`.
- [x] 2.3 Robustness-arm extractions completed via `robustness.py` on a
      seeded 120-field subset (seed 2026): fill05, b11dup, up144,
      drop1/2/4/6, rebin30/60 — all 9 arms green (counts/timings in
      `outputs/tables/robustness.json`).

## 3. Evaluation suite

- [x] 3.1 `analyze_representations.py`: PCA + trajectory figures (E1 —
      annual phenological ring in PC1–PC2); phase-matched cross-year
      consistency with bootstrap CI (E2: margin +0.197 [0.192, 0.202],
      219/219 fields positive); kNN purity (E3). Tables + figures in
      `outputs/`.
- [x] 3.2 `probe_stage.py`: derived 4-phase labels (store-native
      percent-of-cycle, labelled derived), logistic probe OE vs raw-channel
      control, field-level splits (train 280 / dev 53 fields), 3 seeds:
      OE 0.854 ± 0.002 vs raw 0.627 ± 0.000 macro-F1 — CI separated (E4).
- [x] 3.3 `detect_anomalies.py`: controlled-deviation protocol on
      dev∩patches fields (105 injected / 2,278 bins, 3 injection seeds),
      OE max-abs-z / mean-abs-z / kNN-dist vs B0 spectral robust-z vs B1
      persistence, train-only calibration + dev FAR-matched secondary point,
      AUC-PR per severity; real-bin concordance pre-2024 (OE∩B0 Jaccard
      0.115, OE∩B1 0.0). No overall detection advantage over B0 (E5).
- [x] 3.4 `robustness.py` + `benchmark_perf.py`: drift tables (E6 — robust
      to missing bins; rebin30 −0.135 F1; up144 best probe cell 0.889);
      latency/RSS/size + full-store extrapolation (E7 — 79 s wall).

## 4. Report and verdict

- [x] 4.1 Gates evaluated; **verdict = ADAPT** (G-O1 ✓, G-O2 ✓, G-O3 ✗ by
      the pre-registered AUC-PR criterion, G-O5 ✓). Full report with
      provenance, per-experiment numbers, limitations (field-only chips,
      L1-vs-L2 gap, 4/11 missing bands, GSD, license) at
      `experiments/olmoearth_eval/REPORT.md`.
- [x] 4.2 Owner-facing summary updated in `todo.md`; durable insight saved
      to project memory.

## 5. Hygiene

- [x] 5.1 Zero writes outside `experiments/olmoearth_eval/` +
      `tests/unit/test_olmoearth_eval_adapter.py` + openspec/todo/INDEX +
      `.gitignore`; scratch under `.tmp/` removed; embedding binaries
      gitignored as regenerable (tables/figures/report committed).
