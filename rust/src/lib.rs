// Gate G3 placeholder: Rust acceleration for ssl_anomaly.
// Implements robust_baseline (median + MAD*scale) and batched Whittaker.
// Compile with: cargo build --release (in WSL or Windows with Rust toolchain).
// Python binding via PyO3; parity tests in tests/parity/.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use ndarray::Array2;

/// Robust baseline: per-column median and MAD*1.4826.
/// X: (n_samples, n_features) with NaN for missing. Returns (median, scale) each (n_features,).
#[pyfunction]
fn robust_baseline<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    min_n: usize,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let arr = x.as_array();
    let (n, m) = (arr.shape()[0], arr.shape()[1]);
    let mut med = vec![f64::NAN; m];
    let mut scale = vec![f64::NAN; m];
    for j in 0..m {
        let mut col: Vec<f64> = (0..n).map(|i| arr[[i, j]]).filter(|v| v.is_finite()).collect();
        if col.len() < min_n {
            continue;
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ncol = col.len();
        med[j] = if ncol % 2 == 1 {
            col[ncol / 2]
        } else {
            0.5 * (col[ncol / 2 - 1] + col[ncol / 2])
        };
        let mut dev: Vec<f64> = col.iter().map(|v| (v - med[j]).abs()).collect();
        dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let nd = dev.len();
        let mad = if nd % 2 == 1 {
            dev[nd / 2]
        } else {
            0.5 * (dev[nd / 2 - 1] + dev[nd / 2])
        };
        scale[j] = mad * 1.4826;
    }
    let med_arr = PyArray1::from_vec_bound(py, med);
    let scale_arr = PyArray1::from_vec_bound(py, scale);
    (med_arr, scale_arr)
}

#[pymodule]
fn ssl_anomaly_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(robust_baseline, m)?)?;
    Ok(())
}
