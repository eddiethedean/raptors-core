//! Concatenation and splitting C API
//!
//! This module provides C API wrappers for concatenation and splitting operations,
//! equivalent to NumPy's concatenation functions

use crate::array::Array;
use crate::ffi::{PyArrayObject, conversion};
use crate::concatenation::concatenate;
use crate::concatenation::stack;
use crate::concatenation::split;
use crate::concatenation::SplitSpec;
use libc::c_int;
use std::ptr;

/// Concatenate arrays along axis
///
/// Equivalent to NumPy's PyArray_Concatenate function.
///
/// # Safety
/// The caller must ensure `arrays` points to an array of at least `n` valid PyArrayObject pointers.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Concatenate(
    arrays: *mut *mut PyArrayObject,
    n: c_int,
    axis: c_int,
) -> *mut PyArrayObject {
    if arrays.is_null() || n <= 0 {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert array of PyArrayObject pointers to slice of Array references
        let mut array_refs = Vec::with_capacity(n as usize);
        for i in 0..(n as usize) {
            let arr_ptr = *arrays.add(i);
            if arr_ptr.is_null() {
                return ptr::null_mut();
            }
            let array = match conversion::pyarray_to_array_view(arr_ptr) {
                Ok(a) => a,
                Err(_) => return ptr::null_mut(),
            };
            // Store in a vector that will outlive the slice
            // Note: This is simplified - in full implementation we'd need proper lifetime management
            array_refs.push(array);
        }
        
        // Create slice of references
        let array_slice: Vec<&Array> = array_refs.iter().collect();
        
        // Call concatenate
        let axis_opt = if axis < 0 { None } else { Some(axis as usize) };
        let result = match concatenate(&array_slice, axis_opt) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&result)
    }
}

/// Stack arrays along axis
///
/// Equivalent to NumPy's PyArray_Stack function.
///
/// # Safety
/// The caller must ensure `arrays` points to an array of at least `n` valid PyArrayObject pointers.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Stack(
    arrays: *mut *mut PyArrayObject,
    n: c_int,
    axis: c_int,
) -> *mut PyArrayObject {
    if arrays.is_null() || n <= 0 || axis < 0 {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert array of PyArrayObject pointers to slice of Array references
        let mut array_refs = Vec::with_capacity(n as usize);
        for i in 0..(n as usize) {
            let arr_ptr = *arrays.add(i);
            if arr_ptr.is_null() {
                return ptr::null_mut();
            }
            let array = match conversion::pyarray_to_array_view(arr_ptr) {
                Ok(a) => a,
                Err(_) => return ptr::null_mut(),
            };
            array_refs.push(array);
        }
        
        // Create slice of references
        let array_slice: Vec<&Array> = array_refs.iter().collect();
        
        // Call stack
        let result = match stack(&array_slice, axis as usize) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&result)
    }
}

/// Split array into multiple arrays
///
/// Equivalent to NumPy's PyArray_Split function.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
/// The caller must ensure `indices_or_sections` points to valid data.
/// The caller is responsible for freeing the returned array of pointers.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Split(
    arr: *mut PyArrayObject,
    indices_or_sections: *const i64,
    n: c_int,
    axis: c_int,
) -> *mut *mut PyArrayObject {
    if arr.is_null() || indices_or_sections.is_null() || n <= 0 || axis < 0 {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Extract indices or sections
        let mut indices = Vec::with_capacity(n as usize);
        for i in 0..(n as usize) {
            indices.push(*indices_or_sections.add(i));
        }
        
        // Create SplitSpec
        // Simplified: assume it's sections (number of equal parts)
        // Full implementation would distinguish between indices and sections
        let split_spec = if indices.len() == 1 && indices[0] > 0 {
            SplitSpec::Sections(indices[0] as usize)
        } else {
            SplitSpec::Indices(indices.iter().map(|&x| x as usize).collect())
        };
        
        // Call split
        let results = match split(&array, split_spec, axis as usize) {
            Ok(arrays) => arrays,
            Err(_) => return ptr::null_mut(),
        };
        
        // Allocate array of PyArrayObject pointers
        let layout = match std::alloc::Layout::from_size_align(
            (results.len() * std::mem::size_of::<*mut PyArrayObject>()) as usize,
            std::mem::align_of::<*mut PyArrayObject>(),
        ) {
            Ok(l) => l,
            Err(_) => return ptr::null_mut(),
        };
        
        let ptr_array = std::alloc::alloc(layout) as *mut *mut PyArrayObject;
        if ptr_array.is_null() {
            return ptr::null_mut();
        }
        
        // Convert each result to PyArrayObject and store pointer
        for (i, result) in results.iter().enumerate() {
            let pyarray_ptr = conversion::array_to_pyarray_ptr(result);
            *ptr_array.add(i) = pyarray_ptr;
        }
        
        ptr_array
    }
}

