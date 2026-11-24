# eval-artifact-cache Specification

## Purpose

Allow repeated evaluation runs to reuse deterministic, fingerprinted
intermediates from disk without changing any scientific result.

## Requirements

### Requirement: Opt-in cache

Evaluation SHALL use the disk cache only when the owner explicitly enables it.
With no flag, the cold path SHALL compute every intermediate and SHALL NOT read
or write cache files. Read and write SHALL be separately controllable so a
research run can reuse an existing cache without silently extending it.

#### Scenario: Default evaluation

- **WHEN** an eval command runs with no cache flag
- **THEN** no file under the cache directory is read or written
- **AND** results match the pre-cache baseline

#### Scenario: Read-only cached evaluation

- **WHEN** cache reads are enabled and writes are disabled
- **THEN** existing valid entries are reused and no new entry is created

### Requirement: Content fingerprint

Each cache entry SHALL be keyed by a SHA-256 fingerprint over a canonical
serialization of all inputs that determine its value, plus a cache format
version. The fingerprint for a severity cell SHALL include the store hash,
split and holdout hashes, norm hash, model checkpoint hash, sigma table hash,
severity table hash, level, family, index, round, and injection basis.

#### Scenario: Input changes

- **WHEN** any fingerprint input changes, including the cache format version
- **THEN** the old entry is not read and the intermediate is recomputed under a
  new key

### Requirement: Fail-open recovery

A missing, unreadable, incomplete, or hash-mismatched entry SHALL cause the cold
path to compute the value and write it atomically. The cached run SHALL NOT
return a partial artifact, silently use a stale entry, or fail because of a
cache problem.

#### Scenario: Corrupt entry

- **WHEN** a cache entry fails a read, shape, dtype, or hash check
- **THEN** the value is recomputed, rewritten atomically, and used
- **AND** the run reports the cache invalidation

### Requirement: Exact parity

A warm run SHALL reproduce the cold run's `z2`, `paths`, `p_value`, `observed`,
and injection hash records exactly for every severity cell. Array comparison
SHALL be NaN-aware (`equal_nan=True`), so NaN entries in the same positions
count as equal. Any other difference SHALL invalidate the affected cache entry
and SHALL be treated as a defect.

#### Scenario: Full severity parity

- **WHEN** all severity cells are evaluated cold and then warm
- **THEN** every compared output is byte-identical
- **AND** no cell is admitted to research use while a mismatch is open

### Requirement: Bounded, untracked artifacts

Cache artifacts SHALL live in an ignored directory outside Git. The writer SHALL
enforce a per-cell size cap; cells above the cap SHALL fall back to the cold
path with a diagnostic. Cache size, build time, and warm-run speedup SHALL be
recorded in the evidence report.

#### Scenario: Oversized cell

- **WHEN** a cell's cached tensor exceeds the configured size cap
- **THEN** it is not written and that cell is recomputed on every run with a
  diagnostic explaining why
