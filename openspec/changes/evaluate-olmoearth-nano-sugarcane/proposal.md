# Change: evaluate-olmoearth-nano-sugarcane

**Change ID:** `evaluate-olmoearth-nano-sugarcane`
**Project:** Thesis Implementation
**Status:** Proposed
**Priority:** Medium (external evidence, does not touch capstone gates)
**Research context:** Label-free SSL anomaly detection for sugarcane water-and-salinity
stress (Khuzestan archive 2013–2026, Landsat 8/9 C2 L2 → METRIC).
**Base:** `main` @ `4fdad4c`
**Evidence date:** 2026-11-27

---

## Why

The thesis's frozen SSL lineage does not beat the phase-aware statistical
baseline B1 (findings 20–25). Before investing further in in-house SSL, the
owner asked for an evidence-based evaluation of an **external geospatial
foundation model**: **OlmoEarth v1.2 Nano** (Allen Institute for AI; ViT-Nano,
1,698,096 encoder params). This model is uniquely interesting for our archive
because (verified against the official `olmoearth_pretrain` repo @ `6583a536`
and the shipped checkpoint):

1. it has a **native `landsat` modality** (single 11-band token group
   `B8,B1..B7,B9,B10,B11`, ingested from Landsat Collection 2 Level-1 raw DN),
   i.e. Landsat was a *first-class pre-training input*, not a band-mapped hack;
2. it is spatio-temporal (T ≤ 12 timesteps with real calendar timestamps,
   3D RoPE + month embeddings) — matching our phenology-phase-aware framing;
3. it was trained with **random band dropout (rate 0.2) on the landsat
   branch**, i.e. the model was explicitly exposed to missing Landsat bands —
   directly relevant because our L2 archive lacks pan/cirrus/TIRS-2 bands;
4. at 1.7 M params and ~0.1 s per 48×48×T window on this machine's CPU it is
   feasible to run over the full 370-field patch store locally.

The evaluation must answer, with quantitative evidence and honest baselines:
**do frozen OlmoEarth Nano embeddings carry usable phenology-phase and
anomaly-relevant information for our sugarcane fields — beyond what the raw
spectral channels already provide — at CPU-feasible cost?** The verdict is one
of **adopt / adapt / reject**; nothing is integrated into the production
pipeline by this change.

## What Changes

- **New isolated experiment directory** `experiments/olmoearth_eval/`:
  adapter + extraction + evaluation scripts, vendored model code
  (`third_party/olmoearth_pretrain`, pinned commit), model store
  (config.json + symlink to the owner-downloaded `olmoearth/weights.pth`,
  sha `2773fca4…`), and outputs (embeddings zarr/parquet, tables, figures,
  report). No file under `src/ssl_anomaly/` or `scripts/` is modified.
- **New dependency installation inside the existing `geo` micromamba env only**:
  `einops`, `universal-pathlib`, `huggingface_hub` (the repo's declared
  inference-only dependency set). No new environment.
- **OpenSpec artifacts** (this change): proposal, one delta spec
  (`external-foundation-model-evaluation`), design, tasks; INDEX.md updated.
- **Unit tests** for the adapter (band mapping, window packing, fill
  semantics) under `tests/unit/test_olmoearth_eval_adapter.py`; model-load and
  end-to-end paths are covered by the script `--smoke` modes (GPU-free).

## Required experiments (owner-specified scope)

1. Embedding extraction + temporal representation analysis.
2. Phenological consistency across years.
3. Cross-year and cross-farm representation similarity.
4. Linear probing for phenological-stage information (derived reference
   labels, explicitly labelled as derived — never "ground truth").
5. Anomaly detection with frozen embeddings vs classical spectral/statistical
   baselines (controlled-deviation protocol, AUC-PR at matched calibration).
6. Robustness to missing observations and temporal sampling densities.
7. CPU inference performance, RAM, model size, processing time.

## Out of scope

- Full or partial fine-tuning of OlmoEarth (frozen-embedding evaluation only).
- Any integration into `src/ssl_anomaly/`, the platform, or the production
  data chain; no writes into `Data/Processed/`.
- Any claim above validation-ladder rung **L2** (this evaluation operates on
  controlled deviations + derived labels; no L4 agronomic claim).
- Sentinel-1 / ERA5 modalities of the model (no S1 archive; weather enters
  only via existing channel-derived context if needed).
- Re-running the thesis's frozen findings 20–25 or re-opening the capstone
  gates; this change produces **independent external evidence** only.

## Impact

- **Affected specs:** new capability `external-foundation-model-evaluation`.
- **Affected code:** additions only, isolated under `experiments/olmoearth_eval/`
  plus one new unit-test file. `openspec/changes/INDEX.md` gains one row.
- **Data:** read-only access to `data/processed/patches.zarr` (370-field patch
  store), `data/processed/cycles_490_v2.parquet`, `splits_locked_v2.json`,
  and `Data/Processed/satellite/stats.zarr` (field-level series for baselines).
- **Compute:** extraction over the full store is a bounded CPU job (extrapolated
  from the measured per-window cost; handed to the owner as an exact command
  per AGENTS.md §7 if it exceeds ~2 minutes; smoke runs stay short).
