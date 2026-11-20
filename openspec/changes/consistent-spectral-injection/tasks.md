# Tasks — consistent-spectral-injection

## 1. Injector (TDD-first)

- [x] 1.1 Tests first (`tests/unit/test_patch_perturbations.py`):
      consistency invariant (stored NDVI ≡ `ndvi_from(red, nir)` at every
      injected pixel-bin, both arms, both footprints); legacy byte-identity
      regression (same params ⇒ identical tensor to the pre-change
      `index_only` behavior); realized-Δ recording for the requested index;
      side-effect realized-Δ recorded; floored-value counts; gradual ramp and
      white-null under `bands_consistent`.
- [x] 1.2 Sum-preserving NDVI solve on (red, nir) in
      `patch_perturbations.py` (clipped at `REFLECTANCE_PHYSICAL_RANGE`,
      floored counted; `None` when unrealizable, matching the SAVI solver
      contract).
- [x] 1.3 `injection_basis` parameter on `inject_patch_drop` /
      `inject_patch_recovery`; `bands_consistent` recomputes the NDVI channel
      from post-injection bands; meta carries `injection_basis` +
      `realized_<other-index>_drop_per_active_bin`; basis joins
      `hash_payload` so injection-set identity separates the bases.
- [x] 1.4 Default-basis policy: `index_only` remains the DEFAULT on the
      function signature (explicit opt-in at call sites), so any historical
      call path is unchanged until the eval plumbing flips the primary gate —
      no silent behavior change in unrelated scripts.

## 2. Eval plumbing

- [x] 2.1 `patch_eval.py`: `injection_basis` through `patch_arm_scores` +
      `patch_localization_scores`; per-cell report carries the basis.
- [x] 2.2 `scripts/eval_dev.py`: `--injection-basis {index_only,
      bands_consistent}` (default `bands_consistent`); protocol id
      `severity_sweep_patch_v2` ⇔ `bands_consistent`, v1 ⇔ `index_only`;
      provenance records both.
- [x] 2.3 Platform byte-identity check: platform tests
      (`test_platform_data_contract.py`, `test_platform_bilingual.py`) stay
      green and the platform payload is untouched.

## 3. Gates and handover

- [x] 3.1 Full unit battery for touched modules green; the two standing
      verification gates (contract + bilingual) green.
- [x] 3.2 `eval_dev.py --smoke` end-to-end with `bands_consistent` on one
      existing checkpoint arm; smoke output shows the basis + realized-Δ
      fields, no NaN, protocol id v2.
- [ ] 3.3 *(superseded: no standalone rerun — the v3 severity sweep runs inside the score-semantics-realignment handover)* No long job launched by the agent; the v3-protocol severity sweep
      runs as part of the score-semantics-realignment handover after the v2
      retrain (this change alone does not schedule a rerun of finding-24).
