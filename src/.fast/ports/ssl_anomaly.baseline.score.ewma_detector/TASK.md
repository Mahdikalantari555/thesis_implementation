# Port `ssl_anomaly.baseline.score.ewma_detector` to Rust

You're in a Rust crate fast-anything generated for this: `/home/asus/Projects/thesis/thesis_implementation/src/.fast/ports/ssl_anomaly.baseline.score.ewma_detector`.
Write `ewma_detector` in `src/lib.rs` so it behaves **exactly** like the Python below, and runs fast.
Only edit files in this folder. Don't touch the Python project.

## The original (ssl_anomaly/baseline/score.py:144)

```python
def ewma_detector(
    s: np.ndarray,
    *,
    alpha: float = 0.35,
    L_sigma: float = 2.8,
    baseline_mu: float = 1.0,
    baseline_sigma: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    """EWMA control-chart slow-drift detector over observed phase bins.

    Updates ``E_t = alpha * |s_t| + (1 - alpha) * E_{t-1}`` across observed bins and
    flags bins exceeding exact finite-step upper control limit
    ``mu + L_sigma * sigma * sqrt((alpha / (2 - alpha)) * (1 - (1 - alpha)**(2*m)))``.
    """
    arr = np.asarray(s, dtype=np.float64)
    ewma = np.full_like(arr, np.nan, dtype=np.float64)
    flag = np.zeros(arr.shape, dtype=bool)
    a = float(np.clip(alpha, 0.05, 0.95))
    e_val = float(baseline_mu)
    step_m = 0
    for idx in range(arr.size):
        val = arr[idx]
        if not np.isfinite(val):
            continue
        step_m += 1
        e_val = a * abs(float(val)) + (1.0 - a) * e_val
        ewma[idx] = e_val
        asymp = (a / (2.0 - a)) * (1.0 - (1.0 - a) ** (2 * step_m))
        ucl = float(baseline_mu) + float(L_sigma) * float(baseline_sigma) * float(np.sqrt(max(asymp, 1e-8)))
        if e_val > ucl:
            flag[idx] = True
    return flag, ewma
```

## Real calls

fast-anything recorded 200 real calls (out of 80,000 while the app ran). A few:

```
ewma_detector(array([0.3514510571886237, 3.4991209885000902, 0.5940313796064494, -0.6186402578901297, 1.26574…  ->  (array([False, False, False, False, False, False, False, Fa…
ewma_detector(array([0.3525043570603179, 3.5096078645895017, 0.5958116934314438, -0.6204943245970931, 1.26953…  ->  (array([False, False, False, False, False, False, False, Fa…
ewma_detector(array([0.3514510571886237, 3.4991209885000902, 0.5940313796064494, -0.6186402578901297, 1.26574…  ->  (array([False, False, False, False, False, False, False, Fa…
ewma_detector(array([0.35320655697478065, 3.5165991153157754, 0.5969985693147734, -0.6217303690684021, 1.2720…  ->  (array([False, False, False, False, False, False, False, Fa…
ewma_detector(array([0.3521532571030864, 3.5061122392263644, 0.595218255489779, -0.6198763023614385, 1.268274…  ->  (array([False, False, False, False, False, False, False, Fa…
ewma_detector(array([0.3514510571886237, 3.4991209885000902, 0.5940313796064494, -0.6186402578901297, 1.26574…  ->  (array([False, False, False, False, False, False, False, Fa…
ewma_detector(array([0.3510999572313923, 3.495625363136954, 0.5934379416647847, -0.6180222356544752, 1.264480…  ->  (array([False, False, False, False, False, False, False, Fa…
ewma_detector(array([0.3528554570175492, 3.513103489952638, 0.5964051313731086, -0.6211123468327475, 1.270803…  ->  (array([False, False, False, False, False, False, False, Fa…
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
(`fast-anything verify ssl_anomaly.baseline.score.ewma_detector`).
