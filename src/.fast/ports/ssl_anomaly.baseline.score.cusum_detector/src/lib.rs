//! Rust port of ssl_anomaly.baseline.score.cusum_detector.
//! Faithful scalar loop; numpy only used to wrap the two output buffers.

use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (s, *, drift_k = 1.0, decision_h = 4.5))]
fn cusum_detector(
    py: Python<'_>,
    s: &Bound<'_, PyAny>,
    drift_k: f64,
    decision_h: f64,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let vals: Vec<f64> = s.extract()?;
    let n = vals.len();
    let mut stat = vec![f64::NAN; n];
    let mut flag = vec![false; n];

    let mut c_pos = 0.0f64;
    for (idx, &val) in vals.iter().enumerate() {
        if !val.is_finite() {
            continue;
        }
        c_pos = (c_pos + val.abs() - drift_k).max(0.0);
        stat[idx] = c_pos;
        if c_pos > decision_h {
            flag[idx] = true;
        }
    }

    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("bool_")?)?;
    let flag_arr = np.call_method("asarray", (flag,), Some(&kwargs))?;
    let stat_arr = np.call_method1("asarray", (stat,))?;
    Ok((flag_arr.unbind(), stat_arr.unbind()))
}

#[pymodule]
fn fa_ssl_anomaly_baseline_score_cusum_detector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cusum_detector, m)?)?;
    Ok(())
}
