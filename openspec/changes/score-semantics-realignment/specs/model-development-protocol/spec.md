## ADDED Requirements

### Requirement: True one-step-ahead scoring alignment

The patch scorer SHALL compute the alarm residual at bin `t` from a prediction
produced without consuming bin `t`'s values (the GRU state after bin `t-1`),
bin 0 SHALL be unscored, and the model's training objective SHALL align every
horizon with the same rule. A regression test SHALL prove that perturbing a
bin's values does not change any score at or before that bin, and no report or
docstring SHALL describe a same-bin input-inclusive residual as a forecast.

#### Scenario: Target-leak invariant

- **WHEN** the values of bin `t` are arbitrarily perturbed
- **THEN** the model's prediction of bin `t` is byte-identical to the
  unperturbed run (the forecaster never reads its own target) and every score
  at bins < `t` is byte-identical, while bin `t`'s residual changes only
  through the observed-vs-forecast comparison — that difference is the
  detection signal, not a leak (automated unit test)

#### Scenario: Loss alignment

- **WHEN** training runs under the realigned objective
- **THEN** the horizon-1 loss compares the prediction formed from state(`t`)
  with the values of bin `t+1`, verified by a unit test showing a same-bin
  identity mapping does not minimize the horizon loss while the shifted target
  does

#### Scenario: Bin 0 unscored

- **WHEN** a cycle is scored
- **THEN** the bin-0 alarm path entry is NaN (no earlier state exists to
  forecast from), and alarm-rule `min_history` semantics are unchanged

#### Scenario: Honest terminology

- **WHEN** any report, docstring or finding describes the patch score
- **THEN** it names a one-step-ahead forecasting residual formed from bins
  ≤ `t-1`, and historical input-inclusive results are labeled as such

#### Scenario: Protocol identity and resume guard

- **WHEN** a checkpoint trained under the input-inclusive alignment is loaded
  by a v3-aligned scorer or trainer
- **THEN** validation fails on the protocol/config-hash mismatch instead of
  silently mixing alignments

#### Scenario: Supersession record

- **WHEN** the first v3-protocol full result lands (v2 store,
  `bands_consistent` primary basis)
- **THEN** findings 23/24 carry a superseded-annotation pointing at the new
  finding, which becomes the primary patch-level record
