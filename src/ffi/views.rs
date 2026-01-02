//! Array views and copies C API
//!
//! This module provides C API wrappers for array view operations,
//! equivalent to NumPy's view and copy functions

use crate::array::Array;
use crate::ffi::{PyArrayObject, conversion};
use crate::shape::{squeeze_dims, flatten_shape};
use libc::c_int;
use std::ptr;

/// Create array view with new dtype
///
/// Equivalent to NumPy's PyArray_View function.
/// Creates a new view of the array with a different dtype.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
/// The caller is responsible for freeing the returned pointer using appropriate memory management.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_View(
    arr: *mut PyArrayObject,
    _descr: *mut libc::c_void, // Descriptor for new dtype (simplified - not used yet)
    _type: *mut libc::c_void,  // Type object (not used in pure C)
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
        
        // For now, return a view with the same dtype
        // Full implementation would use the descriptor to create a new dtype
        // Create a new array with the same data (simplified - copies data)
        let shape = array.shape().to_vec();
        let dtype = array.dtype().clone();
        let new_array = match Array::new(shape, dtype) {
            Ok(mut a) => {
                // Copy data
                let size = array.size() * array.itemsize();
                std::ptr::copy_nonoverlapping(
                    array.data_ptr(),
                    a.data_ptr_mut(),
                    size,
                );
                a
            }
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&new_array)
    }
}

/// Create new view with different shape/strides
///
/// Equivalent to NumPy's PyArray_NewView function.
/// Creates a new view with different shape and strides.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_NewView(
    arr: *mut PyArrayObject,
    _type: *mut libc::c_void, // Type object (not used in pure C)
    _descr: *mut libc::c_void, // Descriptor (not used yet)
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
        
        // Create a new view with same shape and dtype (simplified)
        // Full implementation would allow changing shape/strides
        let shape = array.shape().to_vec();
        let dtype = array.dtype().clone();
        let new_array = match Array::new(shape, dtype) {
            Ok(mut a) => {
                // Copy data
                let size = array.size() * array.itemsize();
                std::ptr::copy_nonoverlapping(
                    array.data_ptr(),
                    a.data_ptr_mut(),
                    size,
                );
                a
            }
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&new_array)
    }
}

/// Remove dimensions of size 1
///
/// Equivalent to NumPy's PyArray_Squeeze function.
/// Removes dimensions of size 1 from the array.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Squeeze(arr: *mut PyArrayObject) -> *mut PyArrayObject {
    if arr.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Compute squeezed shape
        let new_shape = squeeze_dims(array.shape(), None);
        
        // Create new array with squeezed shape
        let dtype = array.dtype().clone();
        let new_array = match Array::new(new_shape, dtype) {
            Ok(mut a) => {
                // Copy data (simplified - full implementation would create a view)
                let size = array.size().min(a.size()) * array.itemsize().min(a.itemsize());
                std::ptr::copy_nonoverlapping(
                    array.data_ptr(),
                    a.data_ptr_mut(),
                    size,
                );
                a
            }
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&new_array)
    }
}

/// Flatten array to 1D
///
/// Equivalent to NumPy's PyArray_Flatten function.
/// Returns a flattened copy of the array.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Flatten(arr: *mut PyArrayObject, _order: c_int) -> *mut PyArrayObject {
    if arr.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Compute flattened shape
        let new_shape = flatten_shape(array.shape());
        
        // Create new array with flattened shape
        let dtype = array.dtype().clone();
        let new_array = match Array::new(new_shape, dtype) {
            Ok(mut a) => {
                // Copy all data
                let size = array.size() * array.itemsize();
                std::ptr::copy_nonoverlapping(
                    array.data_ptr(),
                    a.data_ptr_mut(),
                    size,
                );
                a
            }
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&new_array)
    }
}

