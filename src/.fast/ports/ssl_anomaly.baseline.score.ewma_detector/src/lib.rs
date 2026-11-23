//! Rust port of ssl_anomaly.baseline.score.ewma_detector.
//! Faithful scalar loop; numpy only used to wrap the two output buffers.

use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (s, *, alpha = 0.35, L_sigma = 2.8, baseline_mu = 1.0, baseline_sigma = 0.65))]
fn ewma_detector(
    py: Python<'_>,
    s: &Bound<'_, PyAny>,
    alpha: f64,
    L_sigma: f64,
    baseline_mu: f64,
    baseline_sigma: f64,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let vals: Vec<f64> = s.extract()?;
    let n = vals.len();
    let mut ewma = vec![f64::NAN; n];
    let mut flag = vec![false; n];

    let a = alpha.clamp(0.05, 0.95);
    let one_minus_a = 1.0 - a;
    let scale = L_sigma * baseline_sigma;
    let mut e_val = baseline_mu;
    let mut step_m: i32 = 0;

    for (idx, &val) in vals.iter().enumerate() {
        if !val.is_finite() {
            continue;
        }
        step_m += 1;
        e_val = a * val.abs() + one_minus_a * e_val;
        ewma[idx] = e_val;
        let asymp = (a / (2.0 - a)) * (1.0 - one_minus_a.powi(2 * step_m));
        let ucl = baseline_mu + scale * asymp.max(1e-8).sqrt();
        if e_val > ucl {
            flag[idx] = true;
        }
    }

    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("bool_")?)?;
    let flag_arr = np.call_method("asarray", (flag,), Some(&kwargs))?;
    let ewma_arr = np.call_method1("asarray", (ewma,))?;
    Ok((flag_arr.unbind(), ewma_arr.unbind()))
}

#[pymodule]
fn fa_ssl_anomaly_baseline_score_ewma_detector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ewma_detector, m)?)?;
    Ok(())
}
