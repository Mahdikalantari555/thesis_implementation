# Tasks

## 1. Phase 1 — Core Inspector

- [x] 1.1 Create the `tools/inspector/` package, versioned manifest schema, TOML config, and default root/output resolution; verify unit tests cover repository-root resolution from a different working directory, external roots, and unsupported manifest versions.
- [x] 1.2 Implement safe artifact discovery, stable opaque IDs, research-role classification, metadata-first fingerprints, hidden-path exclusions, and compound-directory handling for Zarr; verify fixture tests detect additions/changes without traversing array payloads.
- [x] 1.3 Implement SQLite cache and immutable scan history with a serialized refresh lock and file-stability checks; verify an unchanged second refresh reuses cached results and concurrent refresh is rejected without modifying source fixtures.
- [x] 1.4 Implement bounded JSON/CSV/NPZ/Zarr/Parquet metadata/sample adapters with lazy optional imports; verify standard-library readers work without scientific packages and optional readers report unsupported/incomplete files safely.
- [x] 1.5 Implement the loopback-only stdlib HTTP server, required read-only API routes, explicit refresh endpoint, static UI shell, and safe text rendering; verify route tests reject path traversal, non-loopback binding, and script-like artifact names.
- [x] 1.6 Implement `app.py` startup, browser-open, `--refresh`, and `--report-only` CLI modes plus user documentation and a deterministic smoke test; verify each mode performs only its specified action and a clean checkout works without data archives.

## 2. Phase 2 — Data QA

- [x] 2.1 Add bounded schema, readability, missing/NaN/Inf, duplicate-ID, temporal-order, and consistency validators; verify fixture tests distinguish missing values from zero and report counts with denominators.
- [x] 2.2 Integrate `scripts/platform_channels.py` as the physical-range and channel-name source, with graceful registry-load failure; verify tests compare checks to current registry values and never embed duplicate channel limits.
- [x] 2.3 Add GeoJSON and GeoTIFF metadata/geometry checks with optional geospatial imports and configured sampling limits; verify valid/invalid CRS, geometry, and unavailable-reader cases using small fixtures.
- [x] 2.4 Add QA report generation and Data Explorer details with tested sampling/memory bounds; verify a large synthetic input reports sampling and does not materialize the whole artifact.
- [x] 2.5 Document data-quality meanings, optional packages, and known unsupported cases; run Inspector QA tests independently of the main pipeline and confirm all source fixture hashes remain unchanged.

## 3. Phase 3 — Scientific Evaluation and Visualization

- [x] 3.1 Implement readers for existing `reports/dev/<run_id>/` evaluation JSON and `reports/dev_log.csv`, retaining protocol, split, run, seed, and provenance; verify representative committed artifacts parse without invoking `scripts/eval_dev.py`.
- [x] 3.2 Implement evidence grouping and comparison eligibility checks, displaying missing/incomparable values as `N/A`; verify synthetic benchmark, proxy agreement, and generalization evidence remain distinct and incompatible protocols cannot be ranked together.
- [x] 3.3 Implement safe checkpoint-sidecar metadata inspection without `.pt` deserialization and add adversarial checkpoint fixtures; verify inspection never imports or calls `torch.load`.
- [x] 3.4 Implement map/farm/time-series APIs and local chart/map asset integration with explicit unavailable fallbacks; overlay only recorded v5 `suspected_stress` dates on registry-defined index timelines with checkpoint/split provenance, exclude `test` observations, and verify actual irregular dates plus missing axes/flags/libraries using fixtures.
- [x] 3.5 Build Overview, Data Explorer, Model Lab, Spatial Viewer, and QA/Reports navigation with responsive static HTML/CSS/JS and offline-safe assets; verify browser smoke tests make no CDN requests and all view routes render without uncaught errors.
- [x] 3.6 Document interpretation limits, supported evaluation schema, and local visualization behavior; run UI/API tests with both available and missing optional assets.
- [x] 3.7 Add independent readers for `reports/architecture_benchmark/` experiment registries/manifests/results and `reports/regional_representation/` pretraining, transfer, retrieval, generalization, and decision-gate artifacts; verify fixtures preserve each study's protocol/schema/status fields, including `Deferred`, `Not Applicable`, and `near_matches` exploratory labels.
- [x] 3.8 Add comparison-boundary and locked-test safeguards for all report readers; verify cross-study ranking is refused, unknown schema versions remain discoverable but unnormalized, test-bearing datasets are metadata-only, and test-only metrics are labelled restricted and excluded from rankings/gates.

## 4. Phase 4 — Integration Readiness Gate and Release

- [x] 4.1 Implement Contract v5 validation for farm registry IDs, geometry/CRS, observation dates, channels/ranges, score provenance, stress flags, splits, checkpoint-sidecar alignment, and payload structure; verify tests cover each check's PASS/WARNING/FAIL outcome and registry-backed limits.
- [x] 4.2 Implement criticality mapping and overall readiness aggregation; verify any critical FAIL yields `BLOCKED`, while warnings alone do not, and reasons are attached to each finding.
- [x] 4.3 Implement immutable scan/report history plus JSON and HTML export confined to Inspector output; verify a later scan cannot change the bytes of an earlier history snapshot or overwrite existing research reports.
- [x] 4.4 Add end-to-end tests for CLI launch, manual refresh, incremental reuse, corrupt/incomplete files, changed Contract v5 payload, and absence of writes to source artifacts; verify source-tree input hashes match before and after a complete run.
- [x] 4.5 Finalize `tools/inspector/README.md` with setup, config/path rules, commands, API, data-security limits, optional dependency installation, and known scope; execute all documented commands and the Inspector test suite on Python 3.11+.
- [x] 4.6 Run an end-to-end integration fixture containing current dev metrics, partial architecture-benchmark registry cells, and regional-capstone gate/retrieval outputs; verify all appear under separate study identities with their own statuses, claim limits, and no cross-study ranking.
