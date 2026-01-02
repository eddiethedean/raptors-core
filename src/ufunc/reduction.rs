//! Reduction operations
//!
//! This module provides reduction operations like sum, mean, etc.,
//! equivalent to NumPy's reduction.c

use crate::array::{Array, ArrayError};
use crate::types::NpyType;
use crate::performance::threading::should_parallelize;
use rayon::prelude::*;

/// Pairwise summation for floating-point values
/// This reduces floating-point error accumulation compared to naive summation
fn pairwise_sum_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    if data.len() == 1 {
        return data[0];
    }
    
    // Recursive pairwise summation
    let mid = data.len() / 2;
    pairwise_sum_f64(&data[..mid]) + pairwise_sum_f64(&data[mid..])
}

/// Optimized sum for contiguous double arrays using pairwise summation
unsafe fn sum_contiguous_f64(data_ptr: *const f64, size: usize) -> f64 {
    // For small arrays, use simple sum
    if size < 64 {
        let mut sum = 0.0f64;
        for i in 0..size {
            sum += *data_ptr.add(i);
        }
        return sum;
    }
    
    // For larger arrays, use pairwise summation for better accuracy
    let slice = std::slice::from_raw_parts(data_ptr, size);
    pairwise_sum_f64(slice)
}

/// Optimized sum for contiguous float arrays
unsafe fn sum_contiguous_f32(data_ptr: *const f32, size: usize) -> f32 {
    // For small arrays, use simple sum
    if size < 64 {
        let mut sum = 0.0f32;
        for i in 0..size {
            sum += *data_ptr.add(i);
        }
        return sum;
    }
    
    // For larger arrays, use pairwise summation
    let slice = std::slice::from_raw_parts(data_ptr, size);
    pairwise_sum_f32(slice)
}

fn pairwise_sum_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    if data.len() == 1 {
        return data[0];
    }
    
    let mid = data.len() / 2;
    pairwise_sum_f32(&data[..mid]) + pairwise_sum_f32(&data[mid..])
}

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

/// Parallel sum for contiguous f64 arrays using Rayon
unsafe fn sum_parallel_f64(data_ptr: *const f64, size: usize) -> f64 {
    let chunk_size = (size / rayon::current_num_threads()).max(1);
    let chunks: Vec<&[f64]> = (0..size)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(size);
            std::slice::from_raw_parts(data_ptr.add(start), end - start)
        })
        .collect();
    
    chunks
        .par_iter()
        .map(|chunk| {
            if chunk.len() < 64 {
                chunk.iter().sum()
            } else {
                pairwise_sum_f64(chunk)
            }
        })
        .sum()
}

/// Parallel sum for contiguous f32 arrays using Rayon
unsafe fn sum_parallel_f32(data_ptr: *const f32, size: usize) -> f32 {
    let chunk_size = (size / rayon::current_num_threads()).max(1);
    let chunks: Vec<&[f32]> = (0..size)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(size);
            std::slice::from_raw_parts(data_ptr.add(start), end - start)
        })
        .collect();
    
    chunks
        .par_iter()
        .map(|chunk| {
            if chunk.len() < 64 {
                chunk.iter().sum()
            } else {
                pairwise_sum_f32(chunk)
            }
        })
        .sum()
}

/// Parallel sum for contiguous i32 arrays using Rayon
unsafe fn sum_parallel_i32(data_ptr: *const i32, size: usize) -> i32 {
    let chunk_size = (size / rayon::current_num_threads()).max(1);
    let chunks: Vec<&[i32]> = (0..size)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(size);
            std::slice::from_raw_parts(data_ptr.add(start), end - start)
        })
        .collect();
    
    chunks.par_iter().map(|chunk| chunk.iter().sum::<i32>()).sum()
}

/// Parallel min for contiguous f64 arrays
unsafe fn min_parallel_f64(data_ptr: *const f64, size: usize) -> f64 {
    let chunk_size = (size / rayon::current_num_threads()).max(1);
    let chunks: Vec<&[f64]> = (0..size)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(size);
            std::slice::from_raw_parts(data_ptr.add(start), end - start)
        })
        .collect();
    
    chunks
        .par_iter()
        .map(|chunk| chunk.iter().fold(f64::INFINITY, |acc, &x| acc.min(x)))
        .reduce(|| f64::INFINITY, |acc, x| acc.min(x))
}

/// Parallel max for contiguous f64 arrays
unsafe fn max_parallel_f64(data_ptr: *const f64, size: usize) -> f64 {
    let chunk_size = (size / rayon::current_num_threads()).max(1);
    let chunks: Vec<&[f64]> = (0..size)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(size);
            std::slice::from_raw_parts(data_ptr.add(start), end - start)
        })
        .collect();
    
    chunks
        .par_iter()
        .map(|chunk| chunk.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)))
        .reduce(|| f64::NEG_INFINITY, |acc, x| acc.max(x))
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
    
    // Optimized implementation: use contiguous path, pairwise summation, and parallelization
    let is_contiguous = array.is_c_contiguous();
    let should_par = should_parallelize(size);
    
    match array.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f64;
                let sum = if is_contiguous && should_par {
                    // Parallel contiguous path for large arrays
                    sum_parallel_f64(data_ptr, size)
                } else if is_contiguous {
                    // Sequential contiguous path with pairwise summation for accuracy
                    sum_contiguous_f64(data_ptr, size)
                } else {
                    // Strided path - simple sum (could be optimized further)
                    let mut sum = 0.0f64;
                    for i in 0..size {
                        sum += *data_ptr.add(i);
                    }
                    sum
                };
                let out_ptr = output.data_ptr_mut() as *mut f64;
                *out_ptr = sum;
            }
        }
        NpyType::Float => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f32;
                let sum = if is_contiguous && should_par {
                    sum_parallel_f32(data_ptr, size)
                } else if is_contiguous {
                    sum_contiguous_f32(data_ptr, size)
                } else {
                    let mut sum = 0.0f32;
                    for i in 0..size {
                        sum += *data_ptr.add(i);
                    }
                    sum
                };
                let out_ptr = output.data_ptr_mut() as *mut f32;
                *out_ptr = sum;
            }
        }
        NpyType::Int => {
            unsafe {
                let data_ptr = array.data_ptr() as *const i32;
                let sum = if is_contiguous && should_par {
                    // Parallel contiguous path for large integer arrays
                    sum_parallel_i32(data_ptr, size)
                } else if is_contiguous {
                    // Sequential contiguous path for integers
                    let mut sum = 0i32;
                    let slice = std::slice::from_raw_parts(data_ptr, size);
                    for &val in slice {
                        sum += val;
                    }
                    sum
                } else {
                    let mut sum = 0i32;
                    for i in 0..size {
                        sum += *data_ptr.add(i);
                    }
                    sum
                };
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
    
    let is_contiguous = array.is_c_contiguous();
    let should_par = should_parallelize(size);
    
    match array.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f64;
                let min_val = if is_contiguous && should_par {
                    // Parallel contiguous path for large arrays
                    min_parallel_f64(data_ptr, size)
                } else if is_contiguous {
                    // Sequential contiguous path
                    let slice = std::slice::from_raw_parts(data_ptr, size);
                    slice.iter().fold(f64::INFINITY, |acc, &x| acc.min(x))
                } else {
                    let mut min_val = f64::INFINITY;
                    for i in 0..size {
                        let val = *data_ptr.add(i);
                        if val < min_val {
                            min_val = val;
                        }
                    }
                    min_val
                };
                let out_ptr = output.data_ptr_mut() as *mut f64;
                *out_ptr = min_val;
            }
        }
        NpyType::Float => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f32;
                let min_val = if is_contiguous {
                    let slice = std::slice::from_raw_parts(data_ptr, size);
                    slice.iter().fold(f32::INFINITY, |acc, &x| acc.min(x))
                } else {
                    let mut min_val = f32::INFINITY;
                    for i in 0..size {
                        let val = *data_ptr.add(i);
                        if val < min_val {
                            min_val = val;
                        }
                    }
                    min_val
                };
                let out_ptr = output.data_ptr_mut() as *mut f32;
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
    
    let is_contiguous = array.is_c_contiguous();
    let should_par = should_parallelize(size);
    
    match array.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f64;
                let max_val = if is_contiguous && should_par {
                    // Parallel contiguous path for large arrays
                    max_parallel_f64(data_ptr, size)
                } else if is_contiguous {
                    // Sequential contiguous path
                    let slice = std::slice::from_raw_parts(data_ptr, size);
                    slice.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
                } else {
                    let mut max_val = f64::NEG_INFINITY;
                    for i in 0..size {
                        let val = *data_ptr.add(i);
                        if val > max_val {
                            max_val = val;
                        }
                    }
                    max_val
                };
                let out_ptr = output.data_ptr_mut() as *mut f64;
                *out_ptr = max_val;
            }
        }
        NpyType::Float => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f32;
                let max_val = if is_contiguous {
                    let slice = std::slice::from_raw_parts(data_ptr, size);
                    slice.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
                } else {
                    let mut max_val = f32::NEG_INFINITY;
                    for i in 0..size {
                        let val = *data_ptr.add(i);
                        if val > max_val {
                            max_val = val;
                        }
                    }
                    max_val
                };
                let out_ptr = output.data_ptr_mut() as *mut f32;
                *out_ptr = max_val;
            }
        }
        _ => return Err(ReductionError::ArrayError(ArrayError::TypeMismatch)),
    }
    
    Ok(output)
}

