# Tasks — validate-olmoearth-temporal-transfer

Isolation rules as the parent change: code under `experiments/olmoearth_eval/`,
no `ssl_anomaly` imports, no writes outside the experiment directory.

## 1. T1 probe-arm battery (runnable now)

- [x] 1.1 `temporal_transfer.py probe-arms`: five arms (timestamp-only,
      cycle-progress-only, raw-only, raw+temporal, olmoearth) on identical
      splits/labels; **field-level bootstrap (1,000 resamples over dev
      fields, seed 2026)** for per-arm macro-F1/accuracy intervals and
      paired OE−arm delta distributions.
- [x] 1.2 Run + interpret (2026-11-27): timestamp_only 0.613, raw_only
      0.629, raw+temporal 0.665, olmoearth 0.854 [0.827, 0.876]; paired
      deltas OE−timestamp **+0.239 [0.200, 0.279]**, OE−raw+temporal
      **+0.188 [0.154, 0.225]**, both P(δ>0)=1.000 → temporal-confound
      hypothesis REJECTED; the parent's "beyond raw channels" claim survives
      and strengthens (beyond calendar time and beyond naive concatenation).
      cycle_progress_only 0.999 = oracle control (the label IS cycle
      progress). Bootstrap-unit defect (resampling all 333 table fields)
      found and fixed before accepting numbers; final CIs resample the 53
      evaluation fields.

## 2. T2 unseen-year generalization (runnable now)

- [x] 2.1 `temporal_transfer.py holdout-years`: train pre-2024 train-field
      bins only; evaluate 2024/2025/2026 + pooled ≥2024 on dev+test fields;
      arms olmoearth / raw_only / timestamp_only / cycle_progress_only;
      runtime leakage assertions (train has zero ≥2024 bins; eval has zero
      train-field bins); field-level bootstrap CIs; per-year n; ≥2024-only
      phase-consistency margin as secondary.
- [x] 2.2 Run + record (2026-11-27): evaluation on k-means spatial block
      fields (73; zero train overlap — the flat `test_fields` overlap train
      by design, caught by the leakage assertion and corrected), train =
      268 fields / 10,226 bins pre-2024. Pooled ≥2024: OE 0.825
      [0.776, 0.867] vs raw 0.537 → **Δ+0.289 [0.242, 0.336]**; 2024
      Δ+0.314; 2025 Δ+0.291 (low-n, wide CIs); 2026 n=5 → not interpretable
      (marked). In-sample 0.854 → holdout 0.825 (−0.03). timestamp-only
      degrades sharply in holdout years (0.613→0.367 in 2025) while OE
      degrades least — harvest-calendar drift is tracked by the
      representation, not by calendar time. ≥2024-only phase-consistency
      margin +0.215 [0.029, 0.252] (n=14, supportive but low-n).

## 3. T3 L1-vs-L2 measurement (data-gated)

- [ ] 3.1 `download_l1.py` (specified in design.md §4; NOT executed): public
      GCS C2 L1 fetch for ≥3 same-date scene pairs (summer growth, late
      season/bare, winter) from 165-039/166-038, band subset + MTL only;
      owner approves storage then runs it.
- [ ] 3.2 Measurement script: per-band TOA-vs-SR and BT-vs-LST distribution
      deltas, normalized-space distance of the adapter conversion, embedding
      drift through true-L1 vs L2-converted paths for 5 sample fields.
      Blocked on 3.1.

## 4. T4 anomaly re-run with spatial context (Deferred)

- [ ] 4.1 Deferred — blocked by `date-centric-raw-store` (0/28, HELD, owner
      decision). No work until the owner unblocks that change.

## 5. Reporting and registration

- [x] 5.1 REPORT.md §11 amendment (parent change): owner-review issues
      registered, temporal-confound + generalization results referenced,
      interpretation verdict recorded, T3/T4 status named.
- [x] 5.2 INDEX.md + todo.md registration; memory updated.
