# Tasks — compare-spatial-representation-strategies

## 1. Geometry memo + crop machinery (TDD-first)

- [ ] 1.1 Geometry memo: 64×64 padding/neighbor fractions from the measured
      field-size distribution committed under reports/; exclusion + flag
      recorded.
- [ ] 1.2 Crop sampler: in-polygon-guaranteed 16×16 windows derived from the
      32×32 store (no re-extraction); fixed seed-hashed sets in the
      manifest; reduced-K and padded-fallback recording; unit tests
      (in-polygon invariant, reproducibility, fallback listing).
- [ ] 1.3 R2 Dataset + aggregation: per-crop scoring through the unchanged
      PatchDeepScorer interface (top-q mean primary, majority-vote flag);
      aggregation unit tests against hand-computed values.

## 2. Benchmark module (TDD-first)

- [ ] 2.1 `scripts/bench_spatial_representations.py`: Zarr random-read,
      preload time/RAM, per-representation epoch throughput, CPU util,
      peak RAM, disk I/O → one JSON artifact; runs on CPU.
- [ ] 2.2 Run the benchmark ONCE; record the lazy-vs-materialized decision
      + per-epoch estimates (gates the user-run commands).

## 3. Representations

- [ ] 3.1 R2 config end-to-end (smoke on CPU: shapes, NaN discipline,
      aggregation, estimates).
- [ ] 3.2 R3a multi-dilation stem (flag) + param count check (≈+11k);
      smoke.
- [ ] 3.3 R3b two-scale fusion — built ONLY if the pre-registered trigger
      fires (R2 or R3a context benefit with CI excluding zero).

## 4. Runs (user-executed, estimates first)

- [ ] 4.1 Single-seed screening: R2 K ∈ {4, 6, 8} + R3a; review produced
      reports; pick finalists.
- [ ] 4.2 Finalists × seeds {13, 42, 2026}: full severity sweep (3 shapes +
      secondary) + localization probe + benchmark fields per run.

## 5. Verdict

- [ ] 5.1 The ONE transparent table + figure (four dimensions) committed
      under reports/; pre-registered adoption rule applied verbatim.
- [ ] 5.2 Finding 24 (Persian, persian-writing skill): scale-accuracy
      differences stated plainly (the owner's core ask), capacity and
      compute columns beside every accuracy column, honest negatives
      explicit, L-rung labelling; decision handed to
      patch-full-run-and-ablations.
