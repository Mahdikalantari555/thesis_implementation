| arm | channels | stats | τ @5% | FAR achieved | alarms | top-field share | wall time (s) |
|---|---|---|---|---|---|---|---|
| `r05_level` | 8 | median | 15.658 | 0.043 | 30 | 0.07 | 0.28 |
| `r05_lower` | 8 | p10 | 14.412 | 0.043 | 31 | 0.06 | 0.29 |
| `r05_spread` | 14 | iqr, std | 18.401 | 0.043 | 32 | 0.06 | 0.42 |
| `r05_dist` | 45 | iqr, median, p10, p25, p75, std | 19.842 | 0.043 | 14 | 0.14 | 0.51 |

Calibration: τ chosen on dev clean cycles at a 5% cycle-FAR budget with k=2 consecutive bins (the unchanged alarm rule). No training anywhere in this table; every row is a rule-based arm on the same cycle population.
