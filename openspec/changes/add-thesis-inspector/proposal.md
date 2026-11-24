# Proposal

## Why

The repository contains research artifacts and evaluation reports across multiple generations and storage formats, but inspecting them currently requires knowing their paths and running project-specific scripts. A standalone, local-first Inspector will provide a reproducible read-only inventory, data-quality view, evaluation summary, and Platform v5 readiness gate without duplicating the pipeline or changing its outputs.

## What Changes

- Add a modular local application under `tools/inspector/` with a static HTML/CSS/Vanilla JS UI and a Python standard-library HTTP server; it runs from the repository root with one command and performs only explicit manual scans.
- Define a versioned Artifact Manifest and configurable repository-relative/external input roots. Discover artifacts by format and research role, track metadata fingerprints and scan status, cache metadata/results locally, and preserve immutable scan history.
- Add isolated format adapters for JSON, CSV, NPZ, Zarr, Parquet, GeoJSON, GeoTIFF, and checkpoint metadata. Optional format-specific packages must not be required to start the app or inspect basic JSON/CSV artifacts. Do not deserialize untrusted `.pt` files.
- Report structural and data-quality issues (including incomplete/unreadable files, missingness, NaN/Inf, physical-range violations, temporal/spatial consistency) and read existing evaluation outputs rather than reimplementing project metrics.
- Integrate independently with all three evaluation families: current `reports/dev/` outputs, the architecture benchmark's `reports/architecture_benchmark/` experiment registry/manifests, and the regional representation capstone's `reports/regional_representation/` reports. Preserve their separate protocols and statuses; do not assume the capstone depends on the broad benchmark or rank across incompatible studies.
- Respect the `model-development-protocol` locked-test boundary: inspect manifests and split metadata, but do not open locked test observations or use test metrics for rankings, threshold selection, or gate recalculation.
- Provide overview, artifact/data explorer, model/evaluation, spatial, and QA/report views with a small read-only API; the only state-changing API action is an explicitly requested refresh of Inspector-owned cache/report state.
- In the Spatial Viewer, overlay the existing Contract v5 `suspected_stress` flags on each selected farm's actual channel-index timeline, showing the source checkpoint and split context. This is a recorded model flag, not a new stress classifier or verified stress label; unavailable flags remain unavailable.
- Audit Platform Contract v5 using the existing Python channel/score registry and payload conventions, including provenance and split tags. A critical failure blocks readiness; warnings remain visible without being misrepresented as failures.
- Keep the application independent of the Platform runtime and prohibit writes to source data, checkpoints, and Platform exports. Do not introduce a cloud service, database server, watcher, or frontend build system.

## Capabilities

### New Capabilities
- `thesis-inspector`: Local artifact discovery, manifest and cache lifecycle, format-aware inspection, research QA/evaluation presentation, safe local UI/API, and Platform v5 readiness reporting.

### Modified Capabilities
- None. The Inspector consumes existing research and Platform artifacts; it does not alter their contracts or production behavior.

## Impact

- New code and documentation: `tools/inspector/` and Inspector-specific tests.
- Read-only integration points: `scripts/platform_channels.py`, `platform/src/data/channels.ts` (parity reference only), existing `reports/dev/<run_id>/` and `reports/dev_log.csv`, plus the benchmark (`reports/architecture_benchmark/`) and capstone (`reports/regional_representation/`) report contracts and configured data/checkpoint/Platform paths.
- The Inspector remains an observer of recorded outputs: it will not launch any benchmark/capstone run, read locked-test observations, update their registries, or feed one study's results into another study's gates.
- Inspector-owned outputs: configurable cache and immutable reports under `reports/inspector/` by default; these must never overwrite existing research reports.
- Runtime: Python 3.11+ standard library for startup/API; optional scientific/geospatial readers and locally vendored UI assets. The app binds only to loopback by default and uses no CDN/network dependency.
