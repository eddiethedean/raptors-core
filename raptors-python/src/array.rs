//! Array Python bindings
//!
//! This module provides Python bindings for the Array type.

#![allow(clippy::arc_with_non_send_sync)] // Arc used for Python reference counting, not thread safety

use pyo3::prelude::*;
use pyo3::types::PyAny;
use raptors_core::{Array, empty, zeros, ones};
use raptors_core::types::{DType, NpyType};
use raptors_core::indexing::index_array;
use raptors_core::operations::{add, subtract, multiply, divide};
use raptors_core::operations::{equal, less};
use std::sync::Arc;
use crate::dtype::PyDType;
use crate::iterators;

/// Python Array class
#[pyclass]
pub struct PyArray {
    #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting, not thread safety
    pub(crate) inner: Arc<Array>,
}

// SAFETY: Array contains raw pointers but they are managed safely through Arc
// The data is owned by the Array struct, and Arc provides thread-safe reference counting
unsafe impl Send for PyArray {}
unsafe impl Sync for PyArray {}

impl PyArray {
    /// Get reference to inner array (for internal use)
    pub(crate) fn get_inner(&self) -> &Arc<Array> {
        &self.inner
    }
}

#[pymethods]
impl PyArray {
    /// Create an empty array
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    fn empty(shape: Vec<i64>, dtype: Option<&PyDType>) -> PyResult<Self> {
        let dtype = match dtype {
            Some(dt) => dt.get_inner().clone(),
            None => DType::new(NpyType::Double),
        };
        let array = empty(shape, dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(array),
        })
    }
    
    /// Create a zero-filled array
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    fn zeros(shape: Vec<i64>, dtype: Option<&PyDType>) -> PyResult<Self> {
        let dtype = match dtype {
            Some(dt) => dt.get_inner().clone(),
            None => DType::new(NpyType::Double),
        };
        let array = zeros(shape, dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(array),
        })
    }
    
    /// Create a one-filled array
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    fn ones(shape: Vec<i64>, dtype: Option<&PyDType>) -> PyResult<Self> {
        let dtype = match dtype {
            Some(dt) => dt.get_inner().clone(),
            None => DType::new(NpyType::Double),
        };
        let array = ones(shape, dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(array),
        })
    }
    
    /// Get the shape of the array
    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.get_inner().shape().to_vec()
    }
    
    /// Get the dtype
    #[getter]
    fn dtype(&self) -> PyDType {
        PyDType {
            inner: self.inner.dtype().clone(),
        }
    }
    
    /// Create an iterator over array elements
    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<iterators::PyArrayIterator>> {
        // Create iterator directly
        let inner_ref = slf.get_inner();
        let iter = iterators::PyArrayIterator {
            array: inner_ref.clone(),
            index: 0,
            size: inner_ref.size(),
        };
        Py::new(slf.py(), iter)
    }
    
    /// Get the size (total number of elements)
    #[getter]
    fn size(&self) -> usize {
        self.get_inner().size()
    }
    
    /// Get the number of dimensions
    #[getter]
    fn ndim(&self) -> usize {
        self.get_inner().ndim()
    }
    
    /// Get the strides
    #[getter]
    fn strides(&self) -> Vec<i64> {
        self.get_inner().strides().to_vec()
    }
    
    /// Check if array is C-contiguous
    #[getter]
    fn is_c_contiguous(&self) -> bool {
        self.get_inner().is_c_contiguous()
    }
    
    /// Check if array is Fortran-contiguous
    #[getter]
    fn is_f_contiguous(&self) -> bool {
        self.get_inner().is_f_contiguous()
    }
    
    /// Check if array is writeable
    #[getter]
    fn is_writeable(&self) -> bool {
        self.get_inner().is_writeable()
    }
    
    /// Create a copy of the array
    fn copy(&self) -> PyResult<Self> {
        let copied = self.inner.copy();
        Ok(PyArray {
            inner: Arc::new(copied),
        })
    }
    
    /// Create a view of the array
    fn view(&self) -> PyResult<Self> {
        // Create a view with the same shape and strides
        let view = self.inner.view(
            self.get_inner().shape().to_vec(),
            self.get_inner().strides().to_vec()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(view),
        })
    }
    
    /// Reshape the array
    fn reshape(&self, shape: Vec<i64>) -> PyResult<Self> {
        use raptors_core::shape::shape::validate_reshape_shape;
        validate_reshape_shape(self.get_inner().shape(), &shape)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        // Create new array with new shape using view
        let itemsize = self.get_inner().itemsize();
        let new_strides = raptors_core::shape::shape::compute_reshape_strides(&shape, itemsize);
        
        let reshaped = self.get_inner().view(shape, new_strides)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        Ok(PyArray {
            inner: Arc::new(reshaped),
        })
    }
    
    /// Transpose the array
    fn transpose(&self) -> PyResult<Self> {
        use raptors_core::shape::shape::transpose_dimensions;
        let (new_shape, axes) = transpose_dimensions(self.get_inner().shape(), None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        // Create view with transposed shape and strides
        let mut new_strides = vec![0; new_shape.len()];
        for (i, &axis) in axes.iter().enumerate() {
            new_strides[i] = self.get_inner().strides()[axis as usize];
        }
        
        let transposed = self.get_inner().view(new_shape, new_strides)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        Ok(PyArray {
            inner: Arc::new(transposed),
        })
    }
    
    /// Get item at index
    fn __getitem__(&self, py: Python, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Simple integer indexing for now
        if let Ok(idx) = index.extract::<usize>() {
            if self.inner.ndim() == 1 {
                let indices = vec![idx as i64];
                let ptr = index_array(&self.inner, &indices)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("{}", e)))?;
                
                // Extract value based on dtype
                return self.extract_value(py, ptr);
            }
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Advanced indexing not yet implemented"
        ))
    }
    
    /// Set item at index
    fn __setitem__(&mut self, py: Python, index: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // Simple integer indexing for now
        if let Ok(idx) = index.extract::<usize>() {
            if self.inner.ndim() == 1 {
                let indices = vec![idx as i64];
                let ptr = index_array(&self.inner, &indices)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("{}", e)))?;
                
                // Set value based on dtype (cast to mutable)
                return Self::set_value_static(py, ptr as *mut u8, value, self.get_inner().dtype());
            }
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Advanced indexing not yet implemented"
        ))
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("Array(shape={:?}, dtype={})", self.get_inner().shape(), self.get_inner().dtype().name())
    }
    
    /// String representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
    
    /// Check if this is an instance of a type (isinstance equivalent)
    fn isinstance(&self, type_name: String) -> bool {
        // For now, always return true for "Array" or "PyArray"
        // Full implementation would check actual type hierarchy
        type_name == "Array" || type_name == "PyArray" || type_name == "raptors.Array"
    }
    
    /// Get the class name
    fn __class__(&self) -> String {
        "PyArray".to_string()
    }
    
    /// Addition operator
    fn __add__(&self, other: &PyArray) -> PyResult<Self> {
        let result = add(self.get_inner(), other.get_inner())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(result),
        })
    }
    
    /// Subtraction operator
    fn __sub__(&self, other: &PyArray) -> PyResult<Self> {
        let result = subtract(self.get_inner(), other.get_inner())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(result),
        })
    }
    
    /// Multiplication operator
    fn __mul__(&self, other: &PyArray) -> PyResult<Self> {
        let result = multiply(self.get_inner(), other.get_inner())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(result),
        })
    }
    
    /// True division operator
    fn __truediv__(&self, other: &PyArray) -> PyResult<Self> {
        let result = divide(self.get_inner(), other.get_inner())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(result),
        })
    }
    
    /// Equality operator
    fn __eq__(&self, other: &PyArray) -> PyResult<Self> {
        let result = equal(self.get_inner(), other.get_inner())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(result),
        })
    }
    
    /// Less than operator
    fn __lt__(&self, other: &PyArray) -> PyResult<Self> {
        let result = less(self.get_inner(), other.get_inner())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(result),
        })
    }
    
    /// Greater than operator
    fn __gt__(&self, other: &PyArray) -> PyResult<Self> {
        // Use less with swapped arguments
        let result = less(other.get_inner(), self.get_inner())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(result),
        })
    }
}

impl PyArray {
    /// Extract value from pointer based on dtype
    fn extract_value(&self, py: Python, ptr: *const u8) -> PyResult<Py<PyAny>> {
        use raptors_core::types::NpyType;
        use NpyType::*;
        match self.get_inner().dtype().type_() {
            Bool => {
                let val = unsafe { *(ptr as *const bool) };
                let py_obj = unsafe { pyo3::ffi::PyBool_FromLong(val as i64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Byte => {
                let val = unsafe { *(ptr as *const i8) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromLong(val as i64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            UByte => {
                let val = unsafe { *ptr };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLong(val as u64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Short => {
                let val = unsafe { *(ptr as *const i16) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromLong(val as i64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            UShort => {
                let val = unsafe { *(ptr as *const u16) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLong(val as u64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Int => {
                let val = unsafe { *(ptr as *const i32) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromLong(val as i64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            UInt => {
                let val = unsafe { *(ptr as *const u32) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLong(val as u64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Long | LongLong => {
                let val = unsafe { *(ptr as *const i64) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromLongLong(val) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            ULong | ULongLong => {
                let val = unsafe { *(ptr as *const u64) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLongLong(val) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Float | Half => {
                let val = unsafe { *(ptr as *const f32) };
                let py_obj = unsafe { pyo3::ffi::PyFloat_FromDouble(val as f64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Double | LongDouble => {
                let val = unsafe { *(ptr as *const f64) };
                let py_obj = unsafe { pyo3::ffi::PyFloat_FromDouble(val) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Dtype not supported for indexing"
            ))
        }
    }
    
    /// Set value at pointer based on dtype (static method)
    fn set_value_static(_py: Python, ptr: *mut u8, value: &Bound<'_, PyAny>, dtype: &raptors_core::types::DType) -> PyResult<()> {
        use raptors_core::types::NpyType;
        use NpyType::*;
        match dtype.type_() {
            Bool => {
                let val: bool = value.extract()?;
                unsafe { *(ptr as *mut bool) = val; }
            }
            Byte => {
                let val: i8 = value.extract()?;
                unsafe { *(ptr as *mut i8) = val; }
            }
            UByte => {
                let val: u8 = value.extract()?;
                unsafe { *ptr = val; }
            }
            Short => {
                let val: i16 = value.extract()?;
                unsafe { *(ptr as *mut i16) = val; }
            }
            UShort => {
                let val: u16 = value.extract()?;
                unsafe { *(ptr as *mut u16) = val; }
            }
            Int => {
                let val: i32 = value.extract()?;
                unsafe { *(ptr as *mut i32) = val; }
            }
            UInt => {
                let val: u32 = value.extract()?;
                unsafe { *(ptr as *mut u32) = val; }
            }
            Long | LongLong => {
                let val: i64 = value.extract()?;
                unsafe { *(ptr as *mut i64) = val; }
            }
            ULong | ULongLong => {
                let val: u64 = value.extract()?;
                unsafe { *(ptr as *mut u64) = val; }
            }
            Float | Half => {
                let val: f32 = value.extract()?;
                unsafe { *(ptr as *mut f32) = val; }
            }
            Double | LongDouble => {
                let val: f64 = value.extract()?;
                unsafe { *(ptr as *mut f64) = val; }
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Dtype not supported for indexing"
            ))
        }
        Ok(())
    }
}

