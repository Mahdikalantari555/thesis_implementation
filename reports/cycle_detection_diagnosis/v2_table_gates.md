# v2 cycle-table gates (D6) — 2026-11-20 19:20

```json
{
  "created": "2026-11-20 19:20",
  "git_sha": "cfa57d86d9c2",
  "input": "data/processed/cycles_490_v2.parquet (provenance sidecar cycles_490_v2.provenance.json)",
  "v1": {
    "total": 1755,
    "clean": 799,
    "len_median": 365.0
  },
  "v2": {
    "total": 4458,
    "clean": 3478,
    "len_median": 366.0
  },
  "gate1_length": {
    "median_days": 366.0,
    "share_200_500d": 0.9436,
    "pass": true
  },
  "gate2_season": {
    "new_starts": 2679,
    "share_nov_mar": 0.875,
    "pass": true
  },
  "gate3_per_field": {
    "max_cycles": 12,
    "mean_cycles": 7.1,
    "fields": 490,
    "pass": true
  },
  "gate4_subset": {
    "v1_clean_keys": 799,
    "missing_in_v2": 0,
    "pass": true
  },
  "new_start_month_hist": {
    "1": 438,
    "2": 583,
    "3": 490,
    "4": 156,
    "5": 50,
    "6": 12,
    "7": 5,
    "8": 19,
    "9": 52,
    "10": 41,
    "11": 232,
    "12": 601
  },
  "per_field_percentiles": {
    "50": 7,
    "75": 8,
    "90": 9,
    "95": 10,
    "99": 11
  },
  "ladder_note": "D6 table gates only \u2014 these are DATA-quality gates, not model claims; rung statements unchanged"
}
```
