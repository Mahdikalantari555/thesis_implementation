//! Rust port of ssl_anomaly.io.patches.aggregate_bin.
//! Per-pixel median over valid acquisitions; numpy used only for I/O wrapping.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn shape_of(a: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    a.getattr("shape")?.extract::<Vec<usize>>()
}

// numpy scalars are not Python bool/float instances; convert through .tolist().
fn flat_f64(a: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let flat = a.call_method1("reshape", (-1isize,))?;
    flat.call_method0("tolist")?.extract::<Vec<f64>>()
}

fn flat_bool(a: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    // Accept bool, uint8 (0/1) and any numeric mask, like np.asarray(..., dtype=bool).
    let flat = a.call_method1("reshape", (-1isize,))?;
    let vals: Vec<i64> = flat.call_method0("tolist")?.extract()?;
    Ok(vals.into_iter().map(|v| v != 0).collect())
}

#[pyfunction]
#[pyo3(signature = (scenes, valid_masks, *, in_field = None, dtype = None))]
fn aggregate_bin(
    py: Python<'_>,
    scenes: &Bound<'_, PyAny>,
    valid_masks: &Bound<'_, PyAny>,
    in_field: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let patches = py.import("ssl_anomaly.io.patches")?;
    let n_channels: usize = patches.getattr("N_CHANNELS")?.extract()?;
    let unknown_sensor: i64 = patches
        .getattr("SENSOR_IDS")?
        .get_item("unknown")?
        .extract()?;

    let scenes: Vec<Bound<'_, PyAny>> = scenes.extract()?;
    let masks: Vec<Bound<'_, PyAny>> = valid_masks.extract()?;
    if scenes.len() != masks.len() {
        return Err(PyValueError::new_err(format!(
            "{} scenes but {} validity masks",
            scenes.len(),
            masks.len()
        )));
    }
    let n_scenes = scenes.len();

    let inside: Option<Vec<bool>> = match in_field {
        Some(a) => Some(flat_bool(a)?),
        None => None,
    };

    // Empty bin: all-NaN values, empty pixel_valid, zero counts.
    if n_scenes == 0 {
        let s = inside.as_ref().map(|v| {
            let mut side = 0usize;
            while (side + 1) * (side + 1) <= v.len() {
                side += 1;
            }
            side
        }).unwrap_or(0);
        let values = vec![f64::NAN; n_channels * s * s];
        let pixel_valid = vec![0i64; s * s];
        let dtype_arg = match dtype {
            Some(d) => d.clone().unbind(),
            None => np.getattr("float16")?.unbind(),
        };
        // astype (not asarray with dtype) matches numpy's NaN->int casting.
        let values_arr = np.call_method1("asarray", (values,))?
            .call_method1("astype", (dtype_arg,))?
            .call_method1("reshape", (n_channels, s, s))?;
        let kw2 = PyDict::new(py);
        kw2.set_item("dtype", np.getattr("uint8")?)?;
        let pv_arr = np.call_method("asarray", (pixel_valid,), Some(&kw2))?
            .call_method1("reshape", (s, s))?;
        let cls = patches.getattr("BinAggregate")?;
        return Ok(cls.call1((values_arr, pv_arr, 0i64, unknown_sensor))?.unbind());
    }

    // Decode first scene for geometry.
    let shape = shape_of(&scenes[0])?;
    if shape.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "scene arrays must be (C, S, S), got {:?}",
            shape
        )));
    }
    let c = shape[0];
    let s = shape[1];
    let plane = s * s;
    let out_len = c * plane;

    // Decode scenes into flat f64 planes; masks into flat bool planes.
    let mut scene_vals: Vec<Vec<f64>> = Vec::with_capacity(n_scenes);
    let mut scene_masks: Vec<Vec<bool>> = Vec::with_capacity(n_scenes);
    for (i, (sc, vm)) in scenes.iter().zip(masks.iter()).enumerate() {
        let sh = shape_of(sc)?;
        if sh.len() != 3 || sh[0] != c || sh[1] != s || sh[2] != s {
            return Err(PyValueError::new_err(format!(
                "scene {} has shape {:?}, expected [{}, {}, {}]",
                i, sh, c, s, s
            )));
        }
        let msh = shape_of(vm)?;
        if msh.len() != 2 || msh[0] != s || msh[1] != s {
            return Err(PyValueError::new_err(format!(
                "mask {} has shape {:?}, expected [{}, {}]",
                i, msh, s, s
            )));
        }
        scene_vals.push(flat_f64(sc)?);
        scene_masks.push(flat_bool(vm)?);
    }
    if let Some(ref f) = inside {
        if f.len() != plane {
            return Err(PyValueError::new_err(format!(
                "in_field has {} elements, expected {}",
                f.len(),
                plane
            )));
        }
    }

    let mut values = vec![f64::NAN; out_len];
    // i64, not u8: pyo3 maps Vec<u8> to Python bytes, which numpy cannot cast.
    let mut pixel_valid = vec![0i64; plane];
    let mut n_obs = 0i64;
    let mut buf: Vec<f64> = Vec::with_capacity(n_scenes);

    for a in 0..n_scenes {
        let mask = &scene_masks[a];
        let vals = &scene_vals[a];
        let mut any_in_this_scene = false;
        for ch in 0..c {
            let base = ch * plane;
            for p in 0..plane {
                if !mask[p] {
                    continue;
                }
                if let Some(ref f) = inside {
                    if !f[p] {
                        continue;
                    }
                }
                let v = vals[base + p];
                if !v.is_finite() {
                    continue;
                }
                any_in_this_scene = true;
                pixel_valid[p] = 1;
                // median is written in the separate pass below
            }
        }
        if any_in_this_scene {
            n_obs += 1;
        }
    }

    // Second pass: median of finite+masked values per (channel, pixel).
    // `buf` was only used for counting; recompute here to keep buffers simple.
    for ch in 0..c {
        let base = ch * plane;
        for p in 0..plane {
            buf.clear();
            for a in 0..n_scenes {
                let mask = &scene_masks[a];
                if !mask[p] {
                    continue;
                }
                if let Some(ref f) = inside {
                    if !f[p] {
                        continue;
                    }
                }
                let v = scene_vals[a][base + p];
                if v.is_finite() {
                    buf.push(v);
                }
            }
            if buf.is_empty() {
                continue;
            }
            buf.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            let k = buf.len();
            let med = if k % 2 == 1 {
                buf[k / 2]
            } else {
                (buf[k / 2 - 1] + buf[k / 2]) / 2.0
            };
            values[base + p] = med;
        }
    }

    let dtype_arg = match dtype {
        Some(d) => d.clone().unbind(),
        None => np.getattr("float16")?.unbind(),
    };
    let values_arr = np.call_method1("asarray", (values,))?
        .call_method1("astype", (dtype_arg,))?
        .call_method1("reshape", (c, s, s))?;
    let kw2 = PyDict::new(py);
    kw2.set_item("dtype", np.getattr("uint8")?)?;
    let pv_arr = np.call_method("asarray", (pixel_valid,), Some(&kw2))?
        .call_method1("reshape", (s, s))?;
    let cls = patches.getattr("BinAggregate")?;
    Ok(cls.call1((values_arr, pv_arr, n_obs, unknown_sensor))?.unbind())
}

#[pymodule]
fn fa_ssl_anomaly_io_patches_aggregate_bin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_bin, m)?)?;
    Ok(())
}
