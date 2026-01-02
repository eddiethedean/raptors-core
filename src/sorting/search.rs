//! Binary search implementation
//!
//! This module provides searchsorted functionality,
//! finding insertion points in sorted arrays

use crate::array::{Array, ArrayError};
use crate::types::{DType, NpyType};

use super::SortingError;

/// Find insertion points for values in a sorted array
///
/// Returns indices where values should be inserted to maintain sorted order
///
/// # Arguments
/// * `array` - Sorted array to search in
/// * `values` - Values to find insertion points for
///
/// # Returns
/// * `Ok(Array)` - Array of insertion indices
/// * `Err(SortingError)` if search fails
pub fn searchsorted(array: &Array, values: &Array) -> Result<Array, SortingError> {
    if array.dtype().type_() != values.dtype().type_() {
        return Err(SortingError::UnsupportedType);
    }
    
    let size = values.size();
    let dtype = DType::new(NpyType::Long);
    let mut indices = Array::new(vec![size as i64], dtype)?;
    
    match array.dtype().type_() {
        NpyType::Double => searchsorted_double(array, values, &mut indices),
        NpyType::Float => searchsorted_float(array, values, &mut indices),
        NpyType::Int => searchsorted_int(array, values, &mut indices),
        NpyType::Long => searchsorted_long(array, values, &mut indices),
        _ => Err(SortingError::UnsupportedType),
    }?;
    
    Ok(indices)
}

/// Searchsorted for double
fn searchsorted_double(array: &Array, values: &Array, indices: &mut Array) -> Result<(), SortingError> {
    let arr_size = array.size();
    let val_size = values.size();
    
    unsafe {
        let arr_ptr = array.data_ptr() as *const f64;
        let val_ptr = values.data_ptr() as *const f64;
        let idx_ptr = indices.data_ptr_mut() as *mut i64;
        
        for i in 0..val_size {
            let value = *val_ptr.add(i);
            let idx = binary_search_double(arr_ptr, arr_size, value);
            *idx_ptr.add(i) = idx as i64;
        }
    }
    
    Ok(())
}

/// Binary search for double
fn binary_search_double(arr: *const f64, len: usize, value: f64) -> usize {
    let mut left = 0;
    let mut right = len;
    
    while left < right {
        let mid = left + (right - left) / 2;
        unsafe {
            if *arr.add(mid) < value {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
    }
    
    left
}

/// Searchsorted for float
fn searchsorted_float(array: &Array, values: &Array, indices: &mut Array) -> Result<(), SortingError> {
    let arr_size = array.size();
    let val_size = values.size();
    
    unsafe {
        let arr_ptr = array.data_ptr() as *const f32;
        let val_ptr = values.data_ptr() as *const f32;
        let idx_ptr = indices.data_ptr_mut() as *mut i64;
        
        for i in 0..val_size {
            let value = *val_ptr.add(i);
            let idx = binary_search_float(arr_ptr, arr_size, value);
            *idx_ptr.add(i) = idx as i64;
        }
    }
    
    Ok(())
}

fn binary_search_float(arr: *const f32, len: usize, value: f32) -> usize {
    let mut left = 0;
    let mut right = len;
    
    while left < right {
        let mid = left + (right - left) / 2;
        unsafe {
            if *arr.add(mid) < value {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
    }
    
    left
}

/// Searchsorted for int
fn searchsorted_int(array: &Array, values: &Array, indices: &mut Array) -> Result<(), SortingError> {
    let arr_size = array.size();
    let val_size = values.size();
    
    unsafe {
        let arr_ptr = array.data_ptr() as *const i32;
        let val_ptr = values.data_ptr() as *const i32;
        let idx_ptr = indices.data_ptr_mut() as *mut i64;
        
        for i in 0..val_size {
            let value = *val_ptr.add(i);
            let idx = binary_search_int(arr_ptr, arr_size, value);
            *idx_ptr.add(i) = idx as i64;
        }
    }
    
    Ok(())
}

fn binary_search_int(arr: *const i32, len: usize, value: i32) -> usize {
    let mut left = 0;
    let mut right = len;
    
    while left < right {
        let mid = left + (right - left) / 2;
        unsafe {
            if *arr.add(mid) < value {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
    }
    
    left
}

/// Searchsorted for long
fn searchsorted_long(array: &Array, values: &Array, indices: &mut Array) -> Result<(), SortingError> {
    let arr_size = array.size();
    let val_size = values.size();
    
    unsafe {
        let arr_ptr = array.data_ptr() as *const i64;
        let val_ptr = values.data_ptr() as *const i64;
        let idx_ptr = indices.data_ptr_mut() as *mut i64;
        
        for i in 0..val_size {
            let value = *val_ptr.add(i);
            let idx = binary_search_long(arr_ptr, arr_size, value);
            *idx_ptr.add(i) = idx as i64;
        }
    }
    
    Ok(())
}

fn binary_search_long(arr: *const i64, len: usize, value: i64) -> usize {
    let mut left = 0;
    let mut right = len;
    
    while left < right {
        let mid = left + (right - left) / 2;
        unsafe {
            if *arr.add(mid) < value {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
    }
    
    left
}

