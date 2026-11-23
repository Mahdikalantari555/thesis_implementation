# Spec Delta

## Purpose

Define an optional, opt-in Rust acceleration path for named pure-compute
functions while keeping the Python implementation canonical and unchanged.

## ADDED Requirements

### Requirement: Canonical implementation retained

Every accelerated function SHALL keep its original Python definition,
signature, defaults, call sites, and semantics in the repository. Enabling a
port SHALL NOT delete, replace, or rewrite that Python body. Removing the
reversible swap block SHALL restore the original behavior without any other
edit.

#### Scenario: Disable a port

- **WHEN** the owner runs `fast-anything disable <id>` for an enabled port
- **THEN** the module no longer imports the compiled binary and all calls use
  the unchanged Python function
- **AND** no Python source beyond the appended swap block was modified

### Requirement: Acceleration is opt-in

A clean checkout SHALL execute the Python path by default. A compiled binary
SHALL NOT by itself switch behavior without an explicit enable action. The
documented workflow SHALL require `fast-anything build` followed by
`fast-anything enable <id>` from the project's `src/` directory to activate a
port.

#### Scenario: Fresh clone

- **WHEN** the repository is cloned without the untracked `.fast/` artifacts
  and a workflow is run
- **THEN** the original Python functions execute and results are identical to
  the pre-accelerator baseline

#### Scenario: Explicit activation

- **WHEN** the owner runs the documented build and enable commands for a
  verified port
- **THEN** subsequent calls to that function use the Rust implementation
- **AND** the Python implementation remains present and callable via the
  documented opt-out

### Requirement: Per-run opt-out and fail-open fallback

The system SHALL support `FAST_ANYTHING_DISABLE=1 <command>` to run an enabled
port's Python path for a single invocation. If a compiled binary is missing or
cannot be imported, the module SHALL fall back to the Python function without
raising and SHALL produce the same result as the Python path.

#### Scenario: Per-run opt-out

- **WHEN** a command is run with `FAST_ANYTHING_DISABLE=1` and a port is enabled
- **THEN** that command uses the Python implementations of all accelerated
  functions

#### Scenario: Missing binary

- **WHEN** an enabled port's `.so` is absent or fails to import
- **THEN** the call falls back to Python and completes with unchanged values
- **AND** no placeholder or partial result is produced

### Requirement: Verification gate before enablement

A port SHALL NOT be enabled unless `fast-anything verify` has passed every
recorded, edge, and derived call at relative tolerance 1e-9, the targeted unit
test selection has passed with acceleration on and off, and an end-to-end A/B
benchmark has shown a measured gain. Any edit to a Python function or its Rust
crate SHALL invalidate that port's verification until the gate is repeated.

#### Scenario: Unverified port rejected

- **WHEN** a port has any failing parity case or a failing targeted test
- **THEN** it is not enabled and the Python implementation stays in use

#### Scenario: No measured gain

- **WHEN** the benchmark shows no end-to-end improvement for a workflow
- **THEN** the report states 1.0× and the port is not claimed as an acceleration
  of that workflow

### Requirement: Evidence artifact

The change SHALL keep a committed benchmark driver and a Markdown evidence
report that record the profiled time shares, verified replay counts, and
per-workflow A/B timings with environment and loop counts. Compiled binaries,
recordings, and profiles SHALL remain untracked.

#### Scenario: Reviewing the claim

- **WHEN** a reviewer reads the evidence report
- **THEN** each speedup is traceable to a named function, a workload, and a
  measured Python-vs-Rust timing
- **AND** no binary artifact is required to read or audit the report

### Requirement: Scientific neutrality

Acceleration SHALL change wall time only. It SHALL NOT alter scores, hashes,
counts, thresholds, labels, data contracts, model outputs, or any reported
research number. The capability SHALL make no detection or agronomic claim and
SHALL be reported at validation-ladder rung L0 only.

#### Scenario: Same results both ways

- **WHEN** the same workload runs once with acceleration enabled and once with
  `FAST_ANYTHING_DISABLE=1`
- **THEN** all scientific outputs are numerically equivalent and no reported
  metric changes
