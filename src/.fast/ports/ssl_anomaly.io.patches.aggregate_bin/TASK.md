# Port `ssl_anomaly.io.patches.aggregate_bin` to Rust

You're in a Rust crate fast-anything generated for this: `/home/asus/Projects/thesis/thesis_implementation/src/.fast/ports/ssl_anomaly.io.patches.aggregate_bin`.
Write `aggregate_bin` in `src/lib.rs` so it behaves **exactly** like the Python below, and runs fast.
Only edit files in this folder. Don't touch the Python project.

## The original (ssl_anomaly/io/patches.py:234)

```python
def aggregate_bin(
    scenes: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
    *,
    in_field: np.ndarray | None = None,
    dtype: np.dtype | str = np.float16,
) -> BinAggregate:
    """Per-pixel median over the valid acquisitions inside one bin.

    Parameters
    ----------
    scenes
        One ``(C, S, S)`` float array per acquisition in the bin (already
        cropped/padded by the caller). NaN marks an invalid pixel value.
    valid_masks
        One boolean ``(S, S)`` array per acquisition — the QA mask.
    in_field
        Optional ``(S, S)`` 0/1 polygon mask. When given, pixels outside the
        field are forced invalid (they are never asserted as observed).

    Semantics (mirrors the field-level median convention in
    ``monitor/inputs.py``): a pixel-bin is valid iff at least one acquisition
    has QA-valid, finite values; the value is the median of those. No
    interpolation, no zero-fill. ``n_obs`` counts acquisitions with at least
    one valid in-field pixel.
    """
    if len(scenes) != len(valid_masks):
        raise ValueError(
            f"{len(scenes)} scenes but {len(valid_masks)} validity masks"
        )
    if not scenes:
        s = in_field.shape[0] if in_field is not None else 0
        return BinAggregate(
            values=np.full((N_CHANNELS, s, s), np.nan, dtype=dtype),
            pixel_valid=np.zeros((s, s), dtype=np.uint8),
            n_obs=0, sensor_id=SENSOR_IDS["unknown"],
        )

    first = np.asarray(scenes[0])
    if first.ndim != 3:
        raise ValueError(f"scene arrays must be (C, S, S), got {first.shape}")
    c, size, _ = first.shape
    stack = np.full((len(scenes), c, size, size), np.nan, dtype=np.float64)
    valid = np.zeros((len(scenes), size, size), dtype=bool)
    for i, (sc, vm) in enumerate(zip(scenes, valid_masks)):
        arr = np.asarray(sc, dtype=np.float64)
        if arr.shape != (c, size, size):
            raise ValueError(f"scene {i} has shape {arr.shape}, expected {(c, size, size)}")
        stack[i] = arr
        v = np.asarray(vm, dtype=bool)
        if v.shape != (size, size):
            raise ValueError(f"mask {i} has shape {v.shape}, expected {(size, size)}")
        valid[i] = v

    if in_field is not None:
        inside = np.asarray(in_field, dtype=bool)
        if inside.shape != (size, size):
            raise ValueError(f"in_field has shape {inside.shape}, expected {(size, size)}")
        valid &= inside[None, :, :]

    # A pixel is observed in an acquisition when it is QA-valid AND finite.
    observed = valid[:, None, :, :] & np.isfinite(stack)
    any_obs = observed.any(axis=0)                      # (C, S, S)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # All-NaN slices (whole pixels with no observation) are expected here and
        # stay NaN by design — silence the specific numpy warning, not all of them.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                category=RuntimeWarning)
        values = np.nanmedian(np.where(observed, stack, np.nan), axis=0)
    values[~any_obs] = np.nan

    pixel_valid = observed.any(axis=0).any(axis=0)       # (S, S) any channel observed
    n_obs = int(observed.any(axis=(1, 2, 3)).sum())

    return BinAggregate(
        values=values.astype(dtype),
        pixel_valid=pixel_valid.astype(np.uint8),
        n_obs=n_obs,
        sensor_id=SENSOR_IDS["unknown"],
    )
```

## Real calls

fast-anything recorded 200 real calls (out of 1,200 while the app ran). A few:

```
aggregate_bin([array(shape=(8, 32, 32), dtype=float32), array(shape=(8, 32, 32), dtype=float32)], [array(shap…  ->  BinAggregate(values=array(shape=(8, 32, 32), dtype=f2), pix…
aggregate_bin([array(shape=(8, 32, 32), dtype=float32), array(shape=(8, 32, 32), dtype=float32)], [array(shap…  ->  BinAggregate(values=array(shape=(8, 32, 32), dtype=f2), pix…
aggregate_bin([array(shape=(8, 32, 32), dtype=float32), array(shape=(8, 32, 32), dtype=float32)], [array(shap…  ->  BinAggregate(values=array(shape=(8, 32, 32), dtype=f2), pix…
aggregate_bin([array(shape=(8, 32, 32), dtype=float32), array(shape=(8, 32, 32), dtype=float32)], [array(shap…  ->  BinAggregate(values=array(shape=(8, 32, 32), dtype=f2), pix…
aggregate_bin([array(shape=(8, 32, 32), dtype=float32), array(shape=(8, 32, 32), dtype=float32)], [array(shap…  ->  BinAggregate(values=array(shape=(8, 32, 32), dtype=f2), pix…
aggregate_bin([array(shape=(8, 32, 32), dtype=float32), array(shape=(8, 32, 32), dtype=float32)], [array(shap…  ->  BinAggregate(values=array(shape=(8, 32, 32), dtype=f2), pix…
aggregate_bin([array(shape=(8, 32, 32), dtype=float32), array(shape=(8, 32, 32), dtype=float32)], [array(shap…  ->  BinAggregate(values=array(shape=(8, 32, 32), dtype=f2), pix…
aggregate_bin([array(shape=(8, 32, 32), dtype=float32), array(shape=(8, 32, 32), dtype=float32)], [array(shap…  ->  BinAggregate(values=array(shape=(8, 32, 32), dtype=f2), pix…
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
- Note: `BinAggregate` is a dataclass/NamedTuple in ssl_anomaly/io/patches.py (values: np.ndarray, pixel_valid: np.ndarray, n_obs: int, sensor_id: int). The result must be a `BinAggregate` too: get the class with `py.import("ssl_anomaly.io.patches")?.getattr("BinAggregate")?` and call it; verify checks the class and every field.
- Note: `dtype` defaults to `np.float16` in Python; the Rust signature takes `None` for it, use that default then.

## Check

Run `cargo check` here. Then fast-anything builds the crate in release mode and replays all 200 recorded calls against it
(`fast-anything verify ssl_anomaly.io.patches.aggregate_bin`).
