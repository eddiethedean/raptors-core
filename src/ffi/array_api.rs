//! Extended C API functions for array operations
//!
//! This module provides additional C API functions beyond the basic ones in mod.rs

use super::PyArrayObject;
use super::conversion;
use libc::{c_int, c_void, size_t};
use crate::{empty, zeros, ones, types::{DType, NpyType}};

/// Get the number of dimensions of an array
///
/// Equivalent to NumPy's PyArray_NDIM macro/function
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_NDIM(arr: *mut PyArrayObject) -> c_int {
    if arr.is_null() {
        return 0;
    }
    
    unsafe {
        (*arr).nd
    }
}

/// Get the size of a specific dimension
///
/// Equivalent to NumPy's PyArray_DIM macro/function
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_DIM(arr: *mut PyArrayObject, idim: c_int) -> i64 {
    if arr.is_null() || idim < 0 {
        return 0;
    }
    
    unsafe {
        let arr_ref = &*arr;
        let dim = idim as usize;
        if dim < arr_ref.nd as usize && dim < 64 {
            arr_ref.dimensions[dim]
        } else {
            0
        }
    }
}

/// Get the stride of a specific dimension
///
/// Equivalent to NumPy's PyArray_STRIDE macro/function
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_STRIDE(arr: *mut PyArrayObject, istride: c_int) -> i64 {
    if arr.is_null() || istride < 0 {
        return 0;
    }
    
    unsafe {
        let arr_ref = &*arr;
        let stride_idx = istride as usize;
        if stride_idx < arr_ref.nd as usize && stride_idx < 64 {
            arr_ref.strides[stride_idx]
        } else {
            0
        }
    }
}

/// Get the data pointer of an array
///
/// Equivalent to NumPy's PyArray_DATA macro/function
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_DATA(arr: *mut PyArrayObject) -> *mut c_void {
    if arr.is_null() {
        return std::ptr::null_mut();
    }
    
    unsafe {
        (*arr).data as *mut c_void
    }
}

/// Get the item size in bytes
///
/// Equivalent to NumPy's PyArray_ITEMSIZE macro/function
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_ITEMSIZE(arr: *mut PyArrayObject) -> size_t {
    if arr.is_null() {
        return 0;
    }
    
    unsafe {
        // For now, try to infer itemsize from the array
        // If data is null (empty array), we can't convert, so infer from type
        let arr_ref = &*arr;
        if arr_ref.data.is_null() {
            // Empty array - infer from type_num or use default
            // In full implementation, would check descriptor
            return 8; // Default to double
        }
        
        // Try to convert to Array to get itemsize
        match conversion::pyarray_to_array_view(arr) {
            Ok(array) => array.itemsize() as size_t,
            Err(_) => {
                // Fallback: infer from type_num or use default
                // For now, return 8 as default (double)
                8
            }
        }
    }
}

/// Get pointer to dimensions array
///
/// Equivalent to NumPy's PyArray_DIMS macro/function
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_DIMS(arr: *mut PyArrayObject) -> *const i64 {
    if arr.is_null() {
        return std::ptr::null();
    }
    
    unsafe {
        (*arr).dimensions.as_ptr()
    }
}

/// Get pointer to strides array
///
/// Equivalent to NumPy's PyArray_STRIDES macro/function
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_STRIDES(arr: *mut PyArrayObject) -> *const i64 {
    if arr.is_null() {
        return std::ptr::null();
    }
    
    unsafe {
        (*arr).strides.as_ptr()
    }
}

/// Create an empty array (C API)
///
/// Equivalent to NumPy's PyArray_Empty function
/// Creates an array with uninitialized memory
///
/// # Safety
/// The caller must ensure `dims` is a valid pointer to an array of at least `_nd` elements
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Empty(
    _nd: c_int,
    dims: *const i64,
    type_num: c_int,
    _is_f_order: c_int,
) -> *mut PyArrayObject {
    if dims.is_null() || _nd <= 0 || _nd > 64 {
        return std::ptr::null_mut();
    }
    
    // Convert type_num to NpyType (simplified - only handles basic types)
    let npy_type = match conversion::type_num_to_npytype(type_num) {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    
    let dtype = DType::new(npy_type);
    
    // Build shape vector
    let mut shape = Vec::with_capacity(_nd as usize);
    for i in 0..(_nd as usize) {
        unsafe {
            shape.push(*dims.add(i));
        }
    }
    
    match empty(shape, dtype) {
        Ok(array) => {
            // Convert to PyArrayObject
            unsafe {
                conversion::array_to_pyarray_ptr(&array)
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create a zero-filled array (C API)
///
/// Equivalent to NumPy's PyArray_Zeros function
///
/// # Safety
/// The caller must ensure `dims` is a valid pointer to an array of at least `_nd` elements
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Zeros(
    _nd: c_int,
    dims: *const i64,
    type_num: c_int,
    _is_f_order: c_int,
) -> *mut PyArrayObject {
    if dims.is_null() || _nd <= 0 || _nd > 64 {
        return std::ptr::null_mut();
    }
    
    // Convert type_num to NpyType
    let npy_type = match conversion::type_num_to_npytype(type_num) {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    
    let dtype = DType::new(npy_type);
    
    // Build shape vector
    let mut shape = Vec::with_capacity(_nd as usize);
    unsafe {
        for i in 0..(_nd as usize) {
            shape.push(*dims.add(i));
        }
    }
    
    match zeros(shape, dtype) {
        Ok(array) => {
            // Convert to PyArrayObject
            unsafe {
                conversion::array_to_pyarray_ptr(&array)
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create a one-filled array (C API)
///
/// Equivalent to NumPy's PyArray_Ones function
///
/// # Safety
/// The caller must ensure `dims` is a valid pointer to an array of at least `_nd` elements
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Ones(
    _nd: c_int,
    dims: *const i64,
    type_num: c_int,
    _is_f_order: c_int,
) -> *mut PyArrayObject {
    if dims.is_null() || _nd <= 0 || _nd > 64 {
        return std::ptr::null_mut();
    }
    
    // Convert type_num to NpyType
    let npy_type = match conversion::type_num_to_npytype(type_num) {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    
    let dtype = DType::new(npy_type);
    
    // Build shape vector
    let mut shape = Vec::with_capacity(_nd as usize);
    unsafe {
        for i in 0..(_nd as usize) {
            shape.push(*dims.add(i));
        }
    }
    
    match ones(shape, dtype) {
        Ok(array) => {
            // Convert to PyArrayObject
            unsafe {
                conversion::array_to_pyarray_ptr(&array)
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create array from descriptor
///
/// Equivalent to NumPy's PyArray_NewFromDescr function.
///
/// # Safety
/// The caller must ensure `_descr` is a valid descriptor pointer (simplified - not fully implemented).
/// The caller must ensure `_dimensions` points to an array of at least `_nd` elements if not null.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_NewFromDescr(
    _subtype: *mut c_void,
    _descr: *mut c_void, // Descriptor (simplified - not used yet)
    _nd: c_int,
    _dimensions: *const i64,
    _strides: *const i64,
    _data: *mut c_void,
    _flags: c_int,
    _obj: *mut c_void,
) -> *mut PyArrayObject {
    // For now, simplified implementation - use type_num from descriptor if available
    // Full implementation would parse the descriptor properly
    // For now, default to Double type
    let type_num = 12; // Double
    
    // Use PyArray_New with default type (from mod.rs)
    super::PyArray_New(
        _subtype,
        _nd,
        _dimensions,
        type_num,
        _strides,
        _data,
        8, // itemsize for double
        _flags,
        _obj,
    )
}

/// Check if object is exactly an array type
///
/// Equivalent to NumPy's PyArray_CheckExact function
#[no_mangle]
pub extern "C" fn PyArray_CheckExact(op: *mut c_void) -> c_int {
    // TODO: Implement proper type checking
    // For now, simplified check
    if op.is_null() {
        return 0;
    }
    0
}
