//! Conversion utilities between internal Array and PyArrayObject
//!
//! This module provides conversion functions to bridge the gap between
//! the internal Rust Array type and the C-compatible PyArrayObject

use crate::array::Array;
use crate::ffi::PyArrayObject;
use libc::c_int;

/// Convert internal Array to PyArrayObject for C API
///
/// This creates a PyArrayObject structure that shares data with the Array.
/// The caller must ensure the Array outlives the PyArrayObject.
/// Note: This is unsafe because it creates a mutable pointer from an immutable reference.
pub unsafe fn array_to_pyarray(array: &Array) -> PyArrayObject {
    let mut dimensions = [0i64; 64];
    let mut strides = [0i64; 64];
    
    let shape = array.shape();
    let array_strides = array.strides();
    
    for (i, &dim) in shape.iter().enumerate() {
        if i < 64 {
            dimensions[i] = dim;
        }
    }
    
    for (i, &stride) in array_strides.iter().enumerate() {
        if i < 64 {
            strides[i] = stride;
        }
    }
    
    // Cast immutable pointer to mutable for C API compatibility
    let data_ptr = array.data_ptr() as *mut u8;
    
    PyArrayObject {
        ob_base: std::ptr::null_mut(),
        data: data_ptr,
        nd: array.ndim() as c_int,
        descr: std::ptr::null_mut(), // TODO: Convert dtype to descriptor
        flags: array.flags().bits(),
        dimensions,
        strides,
        base: std::ptr::null_mut(),
        _descr: std::ptr::null_mut(),
        weakreflist: std::ptr::null_mut(),
    }
}

