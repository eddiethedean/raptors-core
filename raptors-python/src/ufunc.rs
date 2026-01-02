//! Ufunc Python bindings
//!
//! This module provides Python bindings for universal functions.

#![allow(clippy::arc_with_non_send_sync)] // Arc used for Python reference counting, not thread safety

use pyo3::prelude::*;
use raptors_core::{empty, operations::{add, subtract, multiply, divide, equal, less}};
use raptors_core::ufunc::reduction::{sum_along_axis, mean_along_axis, min_along_axis, max_along_axis};
use raptors_core::ufunc::loop_exec::create_unary_ufunc_loop;
use raptors_core::ufunc::advanced::*;
use std::sync::Arc;

use crate::array::PyArray;

/// Add ufunc functions to module
pub fn add_ufuncs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Arithmetic ufuncs
    m.add_function(wrap_pyfunction!(add_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(divide_arrays, m)?)?;
    
    // Comparison ufuncs
    m.add_function(wrap_pyfunction!(equal_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(less_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(greater_arrays, m)?)?;
    
    // Math ufuncs
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    
    // Reductions
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    
    Ok(())
}

/// Add two arrays
#[pyfunction]
fn add_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = add(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Subtract two arrays
#[pyfunction]
fn subtract_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = subtract(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Multiply two arrays
#[pyfunction]
fn multiply_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = multiply(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Divide two arrays
#[pyfunction]
fn divide_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = divide(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Check if arrays are equal
#[pyfunction]
fn equal_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = equal(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Check if a < b
#[pyfunction]
fn less_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = less(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Check if a > b
#[pyfunction]
fn greater_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    // Use less with swapped arguments
    less_arrays(b, a)
}

/// Compute sine
#[pyfunction]
fn sin(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_sin_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute cosine
#[pyfunction]
fn cos(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_cos_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute tangent
#[pyfunction]
fn tan(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_tan_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute exponential
#[pyfunction]
fn exp(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_exp_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute natural logarithm
#[pyfunction]
fn log(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_log_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute square root
#[pyfunction]
fn sqrt(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_sqrt_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute absolute value
#[pyfunction]
fn abs(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_abs_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Sum array elements
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn sum(a: &PyArray, axis: Option<usize>) -> PyResult<PyArray> {
    let result = sum_along_axis(a.get_inner(), axis)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Mean of array elements
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn mean(a: &PyArray, axis: Option<usize>) -> PyResult<PyArray> {
    let result = mean_along_axis(a.get_inner(), axis)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Minimum of array elements
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn min(a: &PyArray, axis: Option<usize>) -> PyResult<PyArray> {
    let result = min_along_axis(a.get_inner(), axis)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Maximum of array elements
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn max(a: &PyArray, axis: Option<usize>) -> PyResult<PyArray> {
    let result = max_along_axis(a.get_inner(), axis)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

