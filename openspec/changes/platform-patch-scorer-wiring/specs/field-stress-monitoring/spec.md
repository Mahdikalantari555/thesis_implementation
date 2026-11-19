## ADDED Requirements

### Requirement: Trial-stage badge on the patch slot

The platform UI SHALL render the patch slot with an explicit trial-stage
badge and honest copy in both locales, so no viewer can mistake a synthetic-
anomaly L2 trial score for validated stress evidence.

#### Scenario: Badge copy (EN/FA)

- **WHEN** the patch slot is displayed (chart, chip, or comparison view)
- **THEN** a badge marks it "Trial — synthetic-anomaly benchmark (L2)" and
  the FA locale carries the equivalent Persian copy produced through the
  `persian-writing` skill (RTL-correct, no AI slop); both strings live in
  the i18n registries and are covered by the bilingual tests.

#### Scenario: Stress-flag source unchanged by default

- **WHEN** the change is applied
- **THEN** the per-observation `stressFlag`/`stressFlagModel` continue to
  come from the frozen `gru_ae_main` checkpoint; switching the flag source
  to the patch slot happens ONLY in the owner-gated task recorded in this
  change's tasks.md, and never silently.
