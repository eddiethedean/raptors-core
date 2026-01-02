//! Reduction operations
//!
//! This module provides reduction operations like sum, mean, etc.,
//! equivalent to NumPy's reduction.c

use crate::array::{Array, ArrayError};
use crate::types::{DType, NpyType};

/// Reduction error
#[derive(Debug, Clone)]
pub enum ReductionError {
    /// Array error
    ArrayError(ArrayError),
    /// Invalid axis
    InvalidAxis,
}

impl std::fmt::Display for ReductionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReductionError::ArrayError(e) => write!(f, "Array error: {}", e),
            ReductionError::InvalidAxis => write!(f, "Invalid axis"),
        }
    }
}

impl std::error::Error for ReductionError {}

impl From<ArrayError> for ReductionError {
    fn from(err: ArrayError) -> Self {
        ReductionError::ArrayError(err)
    }
}

/// Sum reduction along axis
///
/// If axis is None, sums over all elements
pub fn sum_along_axis(array: &Array, axis: Option<usize>) -> Result<Array, ReductionError> {
    // Simplified implementation - for now, just sum all elements
    // Full implementation would handle axis specification
    let shape = array.shape();
    let dtype = array.dtype().clone();
    
    // Create output array (scalar for full reduction, reduced shape for axis reduction)
    let output_shape = if let Some(ax) = axis {
        if ax >= shape.len() {
            return Err(ReductionError::InvalidAxis);
        }
        let mut new_shape = shape.to_vec();
        new_shape.remove(ax);
        new_shape
    } else {
        vec![1] // Scalar result
    };
    
    let mut output = Array::new(output_shape, dtype)?;
    
    // For now, simple sum over all elements (would need proper axis handling)
    let size = array.size();
    if size == 0 {
        return Ok(output);
    }
    
    // Simple implementation: sum all elements
    // In full implementation, would use proper iteration and axis handling
    match array.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f64;
                let mut sum = 0.0f64;
                for i in 0..size {
                    sum += *data_ptr.add(i);
                }
                let out_ptr = output.data_ptr_mut() as *mut f64;
                *out_ptr = sum;
            }
        }
        NpyType::Int => {
            unsafe {
                let data_ptr = array.data_ptr() as *const i32;
                let mut sum = 0i32;
                for i in 0..size {
                    sum += *data_ptr.add(i);
                }
                let out_ptr = output.data_ptr_mut() as *mut i32;
                *out_ptr = sum;
            }
        }
        _ => return Err(ReductionError::ArrayError(ArrayError::TypeMismatch)),
    }
    
    Ok(output)
}

/// Mean reduction along axis
pub fn mean_along_axis(array: &Array, axis: Option<usize>) -> Result<Array, ReductionError> {
    let mut sum_result = sum_along_axis(array, axis)?;
    let size = array.size() as f64;
    
    // Divide sum by size to get mean
    // Simplified - would need proper type handling
    match sum_result.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let out_ptr = sum_result.data_ptr_mut() as *mut f64;
                *out_ptr /= size;
            }
        }
        _ => {
            // For integer types, convert to float
            // Simplified implementation
        }
    }
    
    Ok(sum_result)
}

/// Min reduction along axis
pub fn min_along_axis(array: &Array, axis: Option<usize>) -> Result<Array, ReductionError> {
    let shape = array.shape();
    let dtype = array.dtype().clone();
    
    let output_shape = if let Some(ax) = axis {
        if ax >= shape.len() {
            return Err(ReductionError::InvalidAxis);
        }
        let mut new_shape = shape.to_vec();
        new_shape.remove(ax);
        new_shape
    } else {
        vec![1]
    };
    
    let mut output = Array::new(output_shape, dtype)?;
    let size = array.size();
    
    if size == 0 {
        return Ok(output);
    }
    
    match array.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f64;
                let mut min_val = f64::INFINITY;
                for i in 0..size {
                    let val = *data_ptr.add(i);
                    if val < min_val {
                        min_val = val;
                    }
                }
                let out_ptr = output.data_ptr_mut() as *mut f64;
                *out_ptr = min_val;
            }
        }
        _ => return Err(ReductionError::ArrayError(ArrayError::TypeMismatch)),
    }
    
    Ok(output)
}

/// Max reduction along axis
pub fn max_along_axis(array: &Array, axis: Option<usize>) -> Result<Array, ReductionError> {
    let shape = array.shape();
    let dtype = array.dtype().clone();
    
    let output_shape = if let Some(ax) = axis {
        if ax >= shape.len() {
            return Err(ReductionError::InvalidAxis);
        }
        let mut new_shape = shape.to_vec();
        new_shape.remove(ax);
        new_shape
    } else {
        vec![1]
    };
    
    let mut output = Array::new(output_shape, dtype)?;
    let size = array.size();
    
    if size == 0 {
        return Ok(output);
    }
    
    match array.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f64;
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..size {
                    let val = *data_ptr.add(i);
                    if val > max_val {
                        max_val = val;
                    }
                }
                let out_ptr = output.data_ptr_mut() as *mut f64;
                *out_ptr = max_val;
            }
        }
        _ => return Err(ReductionError::ArrayError(ArrayError::TypeMismatch)),
    }
    
    Ok(output)
}

