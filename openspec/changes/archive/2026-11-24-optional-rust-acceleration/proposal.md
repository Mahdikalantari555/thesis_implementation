# Proposal: Optional opt-in Rust acceleration for repeated research workflows

## Why

Patch extraction and field-level scoring are rerun many times during
development. A measured prototype (2026-11-23, `reports/fast_anything_benchmark.md`)
showed that three pure-compute functions dominate those runs:

- `aggregate_bin` = 59.5% of a synthetic extraction workload,
- `ewma_detector` = 39.0% and `cusum_detector` = 26.4% of a scoring workload.

Verified Rust ports gave 2.1× and 2.2× end-to-end on those workloads
(hot-function speedups 5.39×, 7.91×, 5.86×). Eval (1.0×) and training (1.0×)
were torch/injection-bound and gained nothing.

Acceleration of this kind must never become a hidden dependency. The canonical
Python implementations must stay in the repository, unchanged, and the Rust
path must be a documented choice that the owner turns on only after parity is
proven. This change formalizes that contract; it does not claim any scientific
result (validation-ladder rung L0, infrastructure only).

## What Changes

- Add a **new capability** `optional-rust-acceleration`: a verified, opt-in Rust
  path for a small set of hot pure functions.
- Keep every Python function, signature, and call site intact. The Rust port is
  a swap appended to the same module; no code is deleted or rewritten.
- Define the opt-in mechanism: a clean clone runs pure Python; enabling requires
  an explicit build + enable step; per-run opt-out stays available.
- Define the promotion gate: full recorded-call parity, targeted unit tests,
  and a measured end-to-end A/B result before any port may be enabled.
- Commit the benchmark driver and the evidence report; keep binaries and
  recordings out of Git.
- Leave training and eval injection out of v1 scope (see design.md).

## Capabilities

### New Capabilities

- `optional-rust-acceleration`: Opt-in, parity-gated Rust acceleration of
  named pure-compute functions, with the Python implementation always retained
  and always available.

### Modified Capabilities

None.

## Impact

- Source: two reversible swap blocks appended to
  `src/ssl_anomaly/baseline/score.py` and `src/ssl_anomaly/io/patches.py`.
- Tool artifacts: `src/.fast/` (Rust crates, recordings, built `.abi3.so`,
  tool report). Binaries and recordings stay untracked; Git stays text-only.
- Evidence: `reports/fast_anything_benchmark.md`, benchmark driver
  `src/_fa_bench_driver.py`.
- No change to data contracts, registry text, platform payloads, frozen splits,
  thresholds, labels, model weights, or any reported research number.
- No new runtime dependency for the default path. The Rust path is optional and
  fails open to Python when its binary is absent.

## Prototype status

The three ports named above already exist and were verified locally in the
prototype that motivated this change. This proposal makes their use explicit,
optional, and repeatable; it does not automatically enable them for any other
checkout.
