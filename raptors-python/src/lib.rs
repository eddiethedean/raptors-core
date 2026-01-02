//! Raptors Python bindings
//!
//! This module provides Python bindings for Raptors Core using PyO3.

pub mod array;
pub mod dtype;
mod ufunc;
pub mod iterators;
mod numpy_interop;

use pyo3::prelude::*;

/// Python module definition
#[pymodule]
fn raptors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Add classes
    m.add_class::<array::PyArray>()?;
    m.add_class::<dtype::PyDType>()?;
    m.add_class::<iterators::PyArrayIterator>()?;
    
    // Add Array as alias for PyArray (NumPy-compatible naming)
    m.add("Array", m.getattr("PyArray")?)?;
    
    // Add module-level convenience functions
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    
    // Add DType class alias
    m.add("DType", m.getattr("PyDType")?)?;
    
    // Add ufunc functions
    ufunc::add_ufuncs(m)?;
    
    // Add NumPy interop functions
    m.add_function(wrap_pyfunction!(numpy_interop::from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_interop::to_numpy, m)?)?;
    
    // Add module-level constants
    dtype::add_dtype_constants(m)?;
    
    Ok(())
}

/// Create a zero-filled array (module-level function)
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn zeros(shape: Vec<i64>, dtype: Option<&dtype::PyDType>) -> PyResult<array::PyArray> {
    use raptors_core::{zeros as core_zeros, DType};
    use raptors_core::types::NpyType;
    use std::sync::Arc;
    
    let dtype_val = match dtype {
        Some(dt) => dt.get_inner().clone(),
        None => DType::new(NpyType::Double),
    };
    let array = core_zeros(shape, dtype_val)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(array::PyArray {
        #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting
        inner: Arc::new(array),
    })
}

/// Create a one-filled array (module-level function)
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn ones(shape: Vec<i64>, dtype: Option<&dtype::PyDType>) -> PyResult<array::PyArray> {
    use raptors_core::{ones as core_ones, DType};
    use raptors_core::types::NpyType;
    use std::sync::Arc;
    
    let dtype_val = match dtype {
        Some(dt) => dt.get_inner().clone(),
        None => DType::new(NpyType::Double),
    };
    let array = core_ones(shape, dtype_val)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(array::PyArray {
        #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting
        inner: Arc::new(array),
    })
}

/// Create an empty array (module-level function)
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn empty(shape: Vec<i64>, dtype: Option<&dtype::PyDType>) -> PyResult<array::PyArray> {
    use raptors_core::{empty as core_empty, DType};
    use raptors_core::types::NpyType;
    use std::sync::Arc;
    
    let dtype_val = match dtype {
        Some(dt) => dt.get_inner().clone(),
        None => DType::new(NpyType::Double),
    };
    let array = core_empty(shape, dtype_val)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(array::PyArray {
        #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting
        inner: Arc::new(array),
    })
}

