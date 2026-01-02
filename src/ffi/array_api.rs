//! Extended C API functions for array operations
//!
//! This module provides additional C API functions beyond the basic ones in mod.rs

use super::PyArrayObject;
use libc::{c_int, c_void, size_t};

/// Get the number of dimensions of an array
///
/// Equivalent to NumPy's PyArray_NDIM macro/function
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
#[no_mangle]
pub extern "C" fn PyArray_ITEMSIZE(arr: *mut PyArrayObject) -> size_t {
    if arr.is_null() {
        return 0;
    }
    
    // For now, we'll need to compute this from the dtype
    // In a full implementation, we'd have proper dtype access
    // This is a placeholder that assumes we can get itemsize from the array
    unsafe {
        // TODO: Get actual itemsize from descriptor
        // For now, return 8 as a default (double)
        8
    }
}

/// Get pointer to dimensions array
///
/// Equivalent to NumPy's PyArray_DIMS macro/function
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
#[no_mangle]
pub extern "C" fn PyArray_Empty(
    _nd: c_int,
    dims: *const i64,
    type_num: c_int,
    _is_f_order: c_int,
) -> *mut PyArrayObject {
    use crate::{empty, types::{DType, NpyType}};
    
    if dims.is_null() || _nd <= 0 || _nd > 64 {
        return std::ptr::null_mut();
    }
    
    // Convert type_num to NpyType (simplified - only handles basic types)
    let npy_type = match type_num {
        0 => NpyType::Bool,
        1 => NpyType::Byte,
        2 => NpyType::UByte,
        3 => NpyType::Short,
        4 => NpyType::UShort,
        5 => NpyType::Int,
        6 => NpyType::UInt,
        7 => NpyType::Long,
        8 => NpyType::ULong,
        11 => NpyType::Float,
        12 => NpyType::Double,
        _ => return std::ptr::null_mut(), // Unsupported type
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
            // Convert to PyArrayObject - this is unsafe and leaks memory
            // In a real implementation, we'd need proper memory management
            // For now, this is a placeholder
            std::ptr::null_mut()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create a zero-filled array (C API)
///
/// Equivalent to NumPy's PyArray_Zeros function
#[no_mangle]
pub extern "C" fn PyArray_Zeros(
    _nd: c_int,
    dims: *const i64,
    type_num: c_int,
    _is_f_order: c_int,
) -> *mut PyArrayObject {
    use crate::{zeros, types::{DType, NpyType}};
    
    if dims.is_null() || _nd <= 0 || _nd > 64 {
        return std::ptr::null_mut();
    }
    
    // Convert type_num to NpyType
    let npy_type = match type_num {
        0 => NpyType::Bool,
        1 => NpyType::Byte,
        2 => NpyType::UByte,
        3 => NpyType::Short,
        4 => NpyType::UShort,
        5 => NpyType::Int,
        6 => NpyType::UInt,
        7 => NpyType::Long,
        8 => NpyType::ULong,
        11 => NpyType::Float,
        12 => NpyType::Double,
        _ => return std::ptr::null_mut(),
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
        Ok(_array) => {
            // TODO: Convert Array to PyArrayObject with proper memory management
            std::ptr::null_mut()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create a one-filled array (C API)
///
/// Equivalent to NumPy's PyArray_Ones function
#[no_mangle]
pub extern "C" fn PyArray_Ones(
    _nd: c_int,
    dims: *const i64,
    type_num: c_int,
    _is_f_order: c_int,
) -> *mut PyArrayObject {
    use crate::{ones, types::{DType, NpyType}};
    
    if dims.is_null() || _nd <= 0 || _nd > 64 {
        return std::ptr::null_mut();
    }
    
    // Convert type_num to NpyType
    let npy_type = match type_num {
        0 => NpyType::Bool,
        1 => NpyType::Byte,
        2 => NpyType::UByte,
        3 => NpyType::Short,
        4 => NpyType::UShort,
        5 => NpyType::Int,
        6 => NpyType::UInt,
        7 => NpyType::Long,
        8 => NpyType::ULong,
        11 => NpyType::Float,
        12 => NpyType::Double,
        _ => return std::ptr::null_mut(),
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
        Ok(_array) => {
            // TODO: Convert Array to PyArrayObject with proper memory management
            std::ptr::null_mut()
        }
        Err(_) => std::ptr::null_mut(),
    }
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

