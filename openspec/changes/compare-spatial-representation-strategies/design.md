## Design — compare-spatial-representation-strategies

### Sub-crop mechanics (R2)

Crop origins are sampled where the 16×16 window lies fully inside the
in-field mask (fraction = 1.0); the crop set is drawn ONCE per (field,
cycle-shape) with a seed and hashed into the run manifest — fixed sets give
deterministic evaluation; per-epoch resampling stays a flag (augmentation
mode) default-off. If a field's in-field area admits fewer than K complete
16×16 windows, K is reduced for that field and the reduction is recorded;
fields with NO valid 16×16 window use a padded full-field 16×16 crop and
are flagged. Field-level scoring: per-crop (bin, channel) scores aggregated
by top-q mean over crops (primary) with majority-vote as the comparison
variant — both feed the UNCHANGED alarm machinery.

### Multi-scale mechanics (R3)

R3a replaces the stem's second conv with parallel dilation branches
(3×3, dilations {1, 2, 4}) summed before GroupNorm — receptive-field
ladder at ~+11k params, no new data path. R3b (true two-scale fusion) runs
the 16×16 crop branch and the 32×32 branch as two stems whose features are
concatenated at the GRU input; it is built only behind the pre-registered
trigger (R2 or R3a showing a context benefit with CI excluding zero).
Capacity matching: every representation's param count lands in the same
table; ±20% is the flagging threshold (AGENTS.md §8 unequal-capacity rule).

### Compute honesty

R2 cost ≈ K × 256 px vs R1's 1024 px per field (K=6 ⇒ ~1.5×) — measured,
not estimated, by the benchmark module. All representations train on the
same device class (CPU-first) and report epoch time + peak RAM.

### Benchmark module

`scripts/bench_spatial_representations.py` measures ONCE, before any
training: Zarr lazy random-read throughput (per-patch ms at 16/32),
materialized preload time + resident RAM, epoch forward/backward throughput
per representation, CPU utilization sampling, peak RAM, disk I/O during a
training epoch. Output: one JSON artifact committed under reports/ + the
recorded lazy-vs-materialized decision + per-epoch estimates that gate the
user-run training commands.

### Relationship to the program

- The `training-holdout-invariant` applies to every representation here:
  crops, folds and subsets draw only from the non-holdout pool, with the
  fail-closed assertion inherited through the shared Dataset base.
- R1 checkpoints and machinery come from `patch-level-controlled-trial`;
  this study adds representations, not a new trial protocol.
- The severity sweep (3 shapes + secondary), localization probe, σ tables
  and alarm calibration are UNCHANGED — only the representation varies.
- The adoption verdict feeds `patch-full-run-and-ablations` (whose E5
  scale sweep is delegated here) and the final architecture decision.

### Risks

- Sub-crop aggregation can wash out localized signals — top-q aggregation
  and the localization AUPRC on partial-footprint injections protect
  against declaring R2 better while it actually blurs.
- K choice bias — swept at single seed before finalists.
- R3b complexity — trigger-gated; if never triggered, recorded as unused.
- Fixed crop sets could under-sample a field's informative region — the
  K-sweep and the resampling flag bound this; noted in the finding.
