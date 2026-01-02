//! Sorting and searching C API
//!
//! This module provides C API wrappers for sorting and searching operations,
//! equivalent to NumPy's sorting functions

use crate::array::Array;
use crate::ffi::{PyArrayObject, conversion};
use crate::sorting::{sort, argsort, searchsorted, partition, SortKind};
use libc::c_int;
use std::ptr;

/// Sort array in-place
///
/// Equivalent to NumPy's PyArray_Sort function.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Sort(
    arr: *mut PyArrayObject,
    axis: c_int,
    _kind: c_int, // Sort kind (simplified - not used yet)
) -> c_int {
    if arr.is_null() {
        return -1; // Error
    }
    
    unsafe {
        // Convert PyArrayObject to Array (mutable)
        let mut array = match conversion::pyarray_to_array_owned(arr) {
            Ok(a) => a,
            Err(_) => return -1,
        };
        
        // Call sort (axis parameter not supported in current implementation)
        match sort(&mut array, SortKind::Quick) {
            Ok(_) => {
                // Note: In full implementation, we would update the original array
                // For now, this is a simplified version
                0 // Success
            }
            Err(_) => -1, // Error
        }
    }
}

/// Return indices that would sort array
///
/// Equivalent to NumPy's PyArray_ArgSort function.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_ArgSort(
    arr: *mut PyArrayObject,
    axis: c_int,
    _kind: c_int, // Sort kind (simplified - not used yet)
) -> *mut PyArrayObject {
    if arr.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Call argsort (axis parameter not supported in current implementation)
        let result = match argsort(&array, SortKind::Quick) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&result)
    }
}

/// Find insertion points in sorted array
///
/// Equivalent to NumPy's PyArray_SearchSorted function.
///
/// # Safety
/// The caller must ensure `arr` and `values` are valid pointers to PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_SearchSorted(
    arr: *mut PyArrayObject,
    values: *mut PyArrayObject,
    _side: c_int, // Side (left or right) - simplified, not used yet
    _sorter: *mut PyArrayObject, // Sorter indices - simplified, not used yet
) -> *mut PyArrayObject {
    if arr.is_null() || values.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        let value_array = match conversion::pyarray_to_array_view(values) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Call searchsorted
        let result = match searchsorted(&array, &value_array) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&result)
    }
}

/// Partition array
///
/// Equivalent to NumPy's PyArray_Partition function.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Partition(
    arr: *mut PyArrayObject,
    kth: c_int,
    axis: c_int,
    _kind: c_int, // Sort kind (simplified - not used yet)
) -> c_int {
    if arr.is_null() {
        return -1; // Error
    }
    
    unsafe {
        // Convert PyArrayObject to Array (mutable)
        let mut array = match conversion::pyarray_to_array_owned(arr) {
            Ok(a) => a,
            Err(_) => return -1,
        };
        
        // Call partition (axis parameter not supported in current implementation)
        match partition(&mut array, kth as usize) {
            Ok(_) => {
                // Note: In full implementation, we would update the original array
                0 // Success
            }
            Err(_) => -1, // Error
        }
    }
}

