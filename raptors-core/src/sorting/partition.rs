//! Partition operations
//!
//! This module provides partition functionality,
//! partially sorting arrays

use crate::array::{Array, ArrayError};
use crate::types::NpyType;

use super::SortingError;

/// Partition array around kth element
///
/// Rearranges array so that element at kth position is in its final sorted position,
/// with all smaller elements before and larger elements after
///
/// # Arguments
/// * `array` - Array to partition
/// * `kth` - Index of element to partition around
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(SortingError)` if partition fails
pub fn partition(array: &mut Array, kth: usize) -> Result<(), SortingError> {
    if kth >= array.size() {
        return Err(SortingError::ArrayError(ArrayError::InvalidShape));
    }
    
    match array.dtype().type_() {
        NpyType::Double => partition_double(array, kth),
        NpyType::Float => partition_float(array, kth),
        NpyType::Int => partition_int(array, kth),
        NpyType::Long => partition_long(array, kth),
        _ => Err(SortingError::UnsupportedType),
    }
}

/// Partition for double
fn partition_double(array: &mut Array, kth: usize) -> Result<(), SortingError> {
    let size = array.size();
    
    unsafe {
        let data_ptr = array.data_ptr_mut() as *mut f64;
        let slice = std::slice::from_raw_parts_mut(data_ptr, size);
        quickselect_double(slice, kth);
    }
    
    Ok(())
}

/// Quickselect for double
fn quickselect_double(arr: &mut [f64], k: usize) {
    let mut left = 0;
    let mut right = arr.len() - 1;
    
    while left < right {
        let pivot_index = partition_double_slice(arr, left, right);
        
        if pivot_index == k {
            return;
        } else if pivot_index < k {
            left = pivot_index + 1;
        } else {
            right = pivot_index - 1;
        }
    }
}

fn partition_double_slice(arr: &mut [f64], left: usize, right: usize) -> usize {
    let pivot = arr[right];
    let mut i = left;
    
    for j in left..right {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    
    arr.swap(i, right);
    i
}

/// Partition for float
fn partition_float(array: &mut Array, kth: usize) -> Result<(), SortingError> {
    let size = array.size();
    
    unsafe {
        let data_ptr = array.data_ptr_mut() as *mut f32;
        let slice = std::slice::from_raw_parts_mut(data_ptr, size);
        quickselect_float(slice, kth);
    }
    
    Ok(())
}

fn quickselect_float(arr: &mut [f32], k: usize) {
    let mut left = 0;
    let mut right = arr.len() - 1;
    
    while left < right {
        let pivot_index = partition_float_slice(arr, left, right);
        
        if pivot_index == k {
            return;
        } else if pivot_index < k {
            left = pivot_index + 1;
        } else {
            right = pivot_index - 1;
        }
    }
}

fn partition_float_slice(arr: &mut [f32], left: usize, right: usize) -> usize {
    let pivot = arr[right];
    let mut i = left;
    
    for j in left..right {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    
    arr.swap(i, right);
    i
}

/// Partition for int
fn partition_int(array: &mut Array, kth: usize) -> Result<(), SortingError> {
    let size = array.size();
    
    unsafe {
        let data_ptr = array.data_ptr_mut() as *mut i32;
        let slice = std::slice::from_raw_parts_mut(data_ptr, size);
        quickselect_int(slice, kth);
    }
    
    Ok(())
}

fn quickselect_int(arr: &mut [i32], k: usize) {
    let mut left = 0;
    let mut right = arr.len() - 1;
    
    while left < right {
        let pivot_index = partition_int_slice(arr, left, right);
        
        if pivot_index == k {
            return;
        } else if pivot_index < k {
            left = pivot_index + 1;
        } else {
            right = pivot_index - 1;
        }
    }
}

fn partition_int_slice(arr: &mut [i32], left: usize, right: usize) -> usize {
    let pivot = arr[right];
    let mut i = left;
    
    for j in left..right {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    
    arr.swap(i, right);
    i
}

/// Partition for long
fn partition_long(array: &mut Array, kth: usize) -> Result<(), SortingError> {
    let size = array.size();
    
    unsafe {
        let data_ptr = array.data_ptr_mut() as *mut i64;
        let slice = std::slice::from_raw_parts_mut(data_ptr, size);
        quickselect_long(slice, kth);
    }
    
    Ok(())
}

fn quickselect_long(arr: &mut [i64], k: usize) {
    let mut left = 0;
    let mut right = arr.len() - 1;
    
    while left < right {
        let pivot_index = partition_long_slice(arr, left, right);
        
        if pivot_index == k {
            return;
        } else if pivot_index < k {
            left = pivot_index + 1;
        } else {
            right = pivot_index - 1;
        }
    }
}

fn partition_long_slice(arr: &mut [i64], left: usize, right: usize) -> usize {
    let pivot = arr[right];
    let mut i = left;
    
    for j in left..right {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    
    arr.swap(i, right);
    i
}

