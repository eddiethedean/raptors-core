//! File I/O C API
//!
//! This module provides C API wrappers for file I/O operations,
//! equivalent to NumPy's file I/O functions

use crate::ffi::{PyArrayObject, conversion};
use crate::io::{save_npy, load_npy};
use libc::{c_char, c_int};
use std::ffi::CStr;
use std::ptr;

/// Save array to NPY file
///
/// Equivalent to NumPy's PyArray_Save function.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
/// The caller must ensure `filename` is a valid null-terminated C string.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Save(
    arr: *mut PyArrayObject,
    filename: *const c_char,
    _format: c_int, // Format (simplified - not used yet, always NPY)
) -> c_int {
    if arr.is_null() || filename.is_null() {
        return -1; // Error
    }
    
    unsafe {
        // Convert filename to Rust string
        let filename_str = match CStr::from_ptr(filename).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        };
        
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return -1,
        };
        
        // Call save_npy (filename_str implements AsRef<Path>)
        match save_npy(filename_str, &array) {
            Ok(_) => 0, // Success
            Err(_) => -1, // Error
        }
    }
}

/// Load array from NPY file
///
/// Equivalent to NumPy's PyArray_Load function.
///
/// # Safety
/// The caller must ensure `filename` is a valid null-terminated C string.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Load(
    filename: *const c_char,
) -> *mut PyArrayObject {
    if filename.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert filename to Rust string
        let filename_str = match CStr::from_ptr(filename).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        
        // Call load_npy
        let array = match load_npy(filename_str) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&array)
    }
}

