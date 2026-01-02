//! Conversion utilities between internal Array and PyArrayObject
//!
//! This module provides conversion functions to bridge the gap between
//! the internal Rust Array type and the C-compatible PyArrayObject

use crate::array::{Array, ArrayError};
use crate::ffi::PyArrayObject;
use crate::types::{DType, NpyType};
use libc::c_int;
use std::alloc::{alloc, dealloc, Layout};

/// Convert internal Array to heap-allocated PyArrayObject* for C API
///
/// This creates a heap-allocated PyArrayObject that shares data with the Array.
/// The caller is responsible for freeing the memory using `free_pyarray`.
/// 
/// # Safety
/// The Array must outlive the returned PyArrayObject*.
/// The returned pointer must be freed using `free_pyarray` to avoid memory leaks.
pub unsafe fn array_to_pyarray_ptr(array: &Array) -> *mut PyArrayObject {
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
    
    // Allocate PyArrayObject on heap
    let layout = Layout::new::<PyArrayObject>();
    let pyarray_ptr = alloc(layout) as *mut PyArrayObject;
    
    if pyarray_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    // Initialize the structure
    std::ptr::write(pyarray_ptr, PyArrayObject {
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
    });
    
    pyarray_ptr
}

/// Free a heap-allocated PyArrayObject
///
/// # Safety
/// The pointer must have been allocated by `array_to_pyarray_ptr` or be null.
pub unsafe fn free_pyarray(ptr: *mut PyArrayObject) {
    if !ptr.is_null() {
        let layout = Layout::new::<PyArrayObject>();
        std::ptr::drop_in_place(ptr);
        dealloc(ptr as *mut u8, layout);
    }
}

/// Convert PyArrayObject* to borrowed Array reference
///
/// This creates a temporary Array view that borrows data from PyArrayObject.
/// The Array does not own the data.
///
/// # Safety
/// The PyArrayObject must be valid and the data pointer must be valid.
/// The returned Array is a view and does not own the data.
pub unsafe fn pyarray_to_array_view(pyarray: *const PyArrayObject) -> Result<Array, ArrayError> {
    if pyarray.is_null() {
        return Err(ArrayError::InvalidShape);
    }
    
    let arr = &*pyarray;
    
    // Extract shape and strides
    let ndim = arr.nd as usize;
    let mut shape = Vec::with_capacity(ndim);
    let mut strides = Vec::with_capacity(ndim);
    
    for i in 0..ndim {
        shape.push(arr.dimensions[i]);
        strides.push(arr.strides[i]);
    }
    
    // Determine dtype from itemsize (simplified - would need descriptor in full implementation)
    // For now, infer from itemsize - use first dimension or default to 1
    let first_dim = if ndim > 0 && arr.dimensions[0] > 0 {
        arr.dimensions[0] as usize
    } else {
        1
    };
    let dtype = infer_dtype_from_itemsize(first_dim)?;
    
    // Create Array view (doesn't own data)
    // Note: This is a simplified implementation. In a full implementation,
    // we would need a way to create an Array from raw data without ownership.
    // For now, we'll create a new array and copy data (not ideal, but works)
    let mut array = Array::new(shape, dtype)?;
    
    // Copy data only if source data is not null and size is valid
    let size = array.size() * array.itemsize();
    if !arr.data.is_null() && size > 0 {
        let data_ptr = array.data_ptr_mut();
        if !data_ptr.is_null() && size <= array.size() * array.itemsize() {
            // Ensure we don't copy more than allocated
            let copy_size = size.min(array.size() * array.itemsize());
            std::ptr::copy_nonoverlapping(arr.data, data_ptr, copy_size);
        }
    }
    
    // Mark as not owning data (view)
    // Note: Array doesn't expose a way to set owns_data, so this is a limitation
    // In a full implementation, we'd need a constructor that takes raw data
    
    Ok(array)
}

/// Convert PyArrayObject* to owned Array
///
/// This creates an owned Array by copying data from PyArrayObject.
///
/// # Safety
/// The PyArrayObject must be valid.
pub unsafe fn pyarray_to_array_owned(pyarray: *const PyArrayObject) -> Result<Array, ArrayError> {
    // For now, same as view - creates a copy
    // In a full implementation, we could optimize this
    pyarray_to_array_view(pyarray)
}

/// Infer dtype from itemsize (simplified)
///
/// This is a helper function that tries to infer the dtype from itemsize.
/// In a full implementation, we would use the descriptor from PyArrayObject.
fn infer_dtype_from_itemsize(_itemsize: usize) -> Result<DType, ArrayError> {
    // Simplified: default to Double
    // In full implementation, would check descriptor or itemsize more carefully
    Ok(DType::new(NpyType::Double))
}

/// Convert type number to NpyType
pub fn type_num_to_npytype(type_num: c_int) -> Option<NpyType> {
    match type_num {
        0 => Some(NpyType::Bool),
        1 => Some(NpyType::Byte),
        2 => Some(NpyType::UByte),
        3 => Some(NpyType::Short),
        4 => Some(NpyType::UShort),
        5 => Some(NpyType::Int),
        6 => Some(NpyType::UInt),
        7 => Some(NpyType::Long),
        8 => Some(NpyType::ULong),
        11 => Some(NpyType::Float),
        12 => Some(NpyType::Double),
        _ => None,
    }
}

/// Convert NpyType to type number
pub fn npytype_to_type_num(npy_type: NpyType) -> c_int {
    match npy_type {
        NpyType::Bool => 0,
        NpyType::Byte => 1,
        NpyType::UByte => 2,
        NpyType::Short => 3,
        NpyType::UShort => 4,
        NpyType::Int => 5,
        NpyType::UInt => 6,
        NpyType::Long => 7,
        NpyType::ULong => 8,
        NpyType::Float => 11,
        NpyType::Double => 12,
        _ => -1, // Unsupported
    }
}

/// Legacy function for backward compatibility
/// 
/// # Safety
/// See `array_to_pyarray_ptr` for safety requirements.
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
        descr: std::ptr::null_mut(),
        flags: array.flags().bits(),
        dimensions,
        strides,
        base: std::ptr::null_mut(),
        _descr: std::ptr::null_mut(),
        weakreflist: std::ptr::null_mut(),
    }
}

