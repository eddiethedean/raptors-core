//! Linear algebra C API
//!
//! This module provides C API wrappers for linear algebra operations,
//! equivalent to NumPy's linear algebra functions

use crate::array::Array;
use crate::ffi::{PyArrayObject, conversion};
use crate::linalg::{matmul, dot};
use libc::c_int;
use std::ptr;

/// Matrix multiplication
///
/// Equivalent to NumPy's PyArray_MatrixProduct function.
///
/// # Safety
/// The caller must ensure `arr1` and `arr2` are valid pointers to PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_MatrixProduct(
    arr1: *mut PyArrayObject,
    arr2: *mut PyArrayObject,
) -> *mut PyArrayObject {
    if arr1.is_null() || arr2.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array1 = match conversion::pyarray_to_array_view(arr1) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        let array2 = match conversion::pyarray_to_array_view(arr2) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Call matmul
        let result = match matmul(&array1, &array2) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&result)
    }
}

/// Inner product
///
/// Equivalent to NumPy's PyArray_InnerProduct function.
///
/// # Safety
/// The caller must ensure `arr1` and `arr2` are valid pointers to PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_InnerProduct(
    arr1: *mut PyArrayObject,
    arr2: *mut PyArrayObject,
) -> *mut PyArrayObject {
    if arr1.is_null() || arr2.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array1 = match conversion::pyarray_to_array_view(arr1) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        let array2 = match conversion::pyarray_to_array_view(arr2) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Call dot (inner product for 1D arrays)
        let result = match dot(&array1, &array2) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&result)
    }
}

/// Matrix multiplication (alias)
///
/// Equivalent to NumPy's PyArray_MatMul function.
///
/// # Safety
/// The caller must ensure `arr1` and `arr2` are valid pointers to PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_MatMul(
    arr1: *mut PyArrayObject,
    arr2: *mut PyArrayObject,
) -> *mut PyArrayObject {
    // Same as PyArray_MatrixProduct
    PyArray_MatrixProduct(arr1, arr2)
}

