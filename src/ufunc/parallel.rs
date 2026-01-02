//! Parallel ufunc operations
//!
//! This module provides parallel implementations of ufunc operations
//! for large arrays using Rayon, matching NumPy's threading behavior.

use crate::array::{Array, ArrayError};
use crate::types::NpyType;
use crate::performance::threading::should_parallelize;
use rayon::prelude::*;

/// Parallel element-wise addition for contiguous arrays
pub fn add_parallel(
    array1: &Array,
    array2: &Array,
    output: &mut Array,
) -> Result<(), ArrayError> {
    if array1.shape() != array2.shape() || array1.shape() != output.shape() {
        return Err(ArrayError::InvalidShape);
    }
    
    if !array1.is_c_contiguous() || !array2.is_c_contiguous() || !output.is_c_contiguous() {
        // For non-contiguous arrays, use sequential operations
        return Err(ArrayError::InvalidShape);
    }
    
    let size = array1.size();
    if !should_parallelize(size) {
        return Err(ArrayError::InvalidShape); // Use sequential path
    }
    
    match array1.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let ptr1 = array1.data_ptr() as *const f64;
                let ptr2 = array2.data_ptr() as *const f64;
                let out_ptr = output.data_ptr_mut() as *mut f64;
                
                let slice1 = std::slice::from_raw_parts(ptr1, size);
                let slice2 = std::slice::from_raw_parts(ptr2, size);
                let slice_out = std::slice::from_raw_parts_mut(out_ptr, size);
                
                slice1
                    .par_iter()
                    .zip(slice2.par_iter())
                    .zip(slice_out.par_iter_mut())
                    .for_each(|((&a, &b), out)| {
                        *out = a + b;
                    });
            }
            Ok(())
        }
        NpyType::Float => {
            unsafe {
                let ptr1 = array1.data_ptr() as *const f32;
                let ptr2 = array2.data_ptr() as *const f32;
                let out_ptr = output.data_ptr_mut() as *mut f32;
                
                let slice1 = std::slice::from_raw_parts(ptr1, size);
                let slice2 = std::slice::from_raw_parts(ptr2, size);
                let slice_out = std::slice::from_raw_parts_mut(out_ptr, size);
                
                slice1
                    .par_iter()
                    .zip(slice2.par_iter())
                    .zip(slice_out.par_iter_mut())
                    .for_each(|((&a, &b), out)| {
                        *out = a + b;
                    });
            }
            Ok(())
        }
        NpyType::Int => {
            unsafe {
                let ptr1 = array1.data_ptr() as *const i32;
                let ptr2 = array2.data_ptr() as *const i32;
                let out_ptr = output.data_ptr_mut() as *mut i32;
                
                let slice1 = std::slice::from_raw_parts(ptr1, size);
                let slice2 = std::slice::from_raw_parts(ptr2, size);
                let slice_out = std::slice::from_raw_parts_mut(out_ptr, size);
                
                slice1
                    .par_iter()
                    .zip(slice2.par_iter())
                    .zip(slice_out.par_iter_mut())
                    .for_each(|((&a, &b), out)| {
                        *out = a + b;
                    });
            }
            Ok(())
        }
        _ => Err(ArrayError::TypeMismatch),
    }
}

/// Parallel element-wise multiplication for contiguous arrays
pub fn multiply_parallel(
    array1: &Array,
    array2: &Array,
    output: &mut Array,
) -> Result<(), ArrayError> {
    if array1.shape() != array2.shape() || array1.shape() != output.shape() {
        return Err(ArrayError::InvalidShape);
    }
    
    if !array1.is_c_contiguous() || !array2.is_c_contiguous() || !output.is_c_contiguous() {
        return Err(ArrayError::InvalidShape);
    }
    
    let size = array1.size();
    if !should_parallelize(size) {
        return Err(ArrayError::InvalidShape);
    }
    
    match array1.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let ptr1 = array1.data_ptr() as *const f64;
                let ptr2 = array2.data_ptr() as *const f64;
                let out_ptr = output.data_ptr_mut() as *mut f64;
                
                let slice1 = std::slice::from_raw_parts(ptr1, size);
                let slice2 = std::slice::from_raw_parts(ptr2, size);
                let slice_out = std::slice::from_raw_parts_mut(out_ptr, size);
                
                slice1
                    .par_iter()
                    .zip(slice2.par_iter())
                    .zip(slice_out.par_iter_mut())
                    .for_each(|((&a, &b), out)| {
                        *out = a * b;
                    });
            }
            Ok(())
        }
        _ => Err(ArrayError::TypeMismatch),
    }
}

/// Check if parallel ufunc execution should be used
pub fn should_use_parallel_ufunc(array: &Array) -> bool {
    array.is_c_contiguous() && should_parallelize(array.size())
}

