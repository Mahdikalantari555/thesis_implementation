# Port `ssl_anomaly.baseline.score.cusum_detector` to Rust

You're in a Rust crate fast-anything generated for this: `/home/asus/Projects/thesis/thesis_implementation/src/.fast/ports/ssl_anomaly.baseline.score.cusum_detector`.
Write `cusum_detector` in `src/lib.rs` so it behaves **exactly** like the Python below, and runs fast.
Only edit files in this folder. Don't touch the Python project.

## The original (ssl_anomaly/baseline/score.py:116)

```python
def cusum_detector(
    s: np.ndarray,
    *,
    drift_k: float = 1.0,
    decision_h: float = 4.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Page's tabular CUSUM along observed phase bins for gradual slow-drift detection.

    Accumulates ``C_t = max(0, C_{t-1} + |s_t| - drift_k)`` solely across finite
    observed bins so unobserved NaN bins neither accumulate nor reset the statistic.
    Returns ``(flag_bool, cusum_stat)`` of shape ``(bins,)``, where unobserved bins
    have ``flag_bool=False`` and ``cusum_stat=NaN``.
    """
    arr = np.asarray(s, dtype=np.float64)
    stat = np.full_like(arr, np.nan, dtype=np.float64)
    flag = np.zeros(arr.shape, dtype=bool)
    c_pos = 0.0
    for idx in range(arr.size):
        val = arr[idx]
        if not np.isfinite(val):
            continue
        c_pos = max(0.0, c_pos + abs(float(val)) - float(drift_k))
        stat[idx] = c_pos
        if c_pos > decision_h:
            flag[idx] = True
    return flag, stat
```

## Real calls

fast-anything recorded 200 real calls (out of 80,000 while the app ran). A few:

```
cusum_detector(array([0.3514510571886237, 3.4991209885000902, 0.5940313796064494, -0.6186402578901297, 1.2657…  ->  (array([False, False, False, False, False, False, False, Fa…
cusum_detector(array([0.3525043570603179, 3.5096078645895017, 0.5958116934314438, -0.6204943245970931, 1.2695…  ->  (array([False, False, False, False, False, False, False, Fa…
cusum_detector(array([0.3514510571886237, 3.4991209885000902, 0.5940313796064494, -0.6186402578901297, 1.2657…  ->  (array([False, False, False, False, False, False, False, Fa…
cusum_detector(array([0.35320655697478065, 3.5165991153157754, 0.5969985693147734, -0.6217303690684021, 1.272…  ->  (array([False, False, False, False, False, False, False, Fa…
cusum_detector(array([0.3521532571030864, 3.5061122392263644, 0.595218255489779, -0.6198763023614385, 1.26827…  ->  (array([False, False, False, False, False, False, False, Fa…
cusum_detector(array([0.3514510571886237, 3.4991209885000902, 0.5940313796064494, -0.6186402578901297, 1.2657…  ->  (array([False, False, False, False, False, False, False, Fa…
cusum_detector(array([0.3510999572313923, 3.495625363136954, 0.5934379416647847, -0.6180222356544752, 1.26448…  ->  (array([False, False, False, False, False, False, False, Fa…
cusum_detector(array([0.3528554570175492, 3.513103489952638, 0.5964051313731086, -0.6211123468327475, 1.27080…  ->  (array([False, False, False, False, False, False, False, Fa…
```

## Rules

- Same result for every input, edge cases included: empty input, zero, negatives, NaN and infinity, non-ASCII text, very long input. verify tests these too.
- Python `str` indexes by Unicode code point: use `.chars()` (or bytes only after checking `is_ascii()`).
- Python ints don't overflow. `i64` is fine unless the recorded values are bigger.
- Float results are compared with a relative tolerance of 1e-9. Keep the same order of operations.
- Where Python raises, raise too (`PyValueError::new_err(...)`, `PyKeyError`, ...).
- Make it fast: no needless allocation or cloning, borrow inputs, `with_capacity`, fast paths for common cases.
- std only, unless a crate clearly pays off. Keep the function and module names as they are.
- Note: already uses np, less to gain.

## Check

Run `cargo check` here. Then fast-anything builds the crate in release mode and replays all 200 recorded calls against it
(`fast-anything verify ssl_anomaly.baseline.score.cusum_detector`).
