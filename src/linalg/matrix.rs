//! Matrix operations
//!
//! This module provides matrix multiplication and dot product operations

use crate::array::{Array, ArrayError};
use crate::types::DType;

/// Linear algebra error
#[derive(Debug, Clone)]
pub enum LinalgError {
    /// Array error
    ArrayError(ArrayError),
    /// Shape mismatch
    ShapeMismatch,
    /// Invalid dimension
    InvalidDimension,
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinalgError::ArrayError(e) => write!(f, "Array error: {}", e),
            LinalgError::ShapeMismatch => write!(f, "Shape mismatch"),
            LinalgError::InvalidDimension => write!(f, "Invalid dimension"),
        }
    }
}

impl std::error::Error for LinalgError {}

impl From<ArrayError> for LinalgError {
    fn from(err: ArrayError) -> Self {
        LinalgError::ArrayError(err)
    }
}

/// Compute dot product of two arrays
///
/// Handles various cases:
/// - 1D-1D: scalar result (inner product)
/// - 1D-2D: matrix-vector product
/// - 2D-1D: matrix-vector product
/// - 2D-2D: matrix multiplication
pub fn dot(a: &Array, b: &Array) -> Result<Array, LinalgError> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();
    
    // Validate dtype compatibility  
    if a.dtype().type_() != b.dtype().type_() {
        return Err(LinalgError::ShapeMismatch);
    }
    
    match (a_ndim, b_ndim) {
        (1, 1) => {
            // 1D-1D: inner product (scalar result)
            if a_shape[0] != b_shape[0] {
                return Err(LinalgError::ShapeMismatch);
            }
            
            let dtype = a.dtype().clone();
            let mut result = Array::new(vec![1], dtype)?;
            
            unsafe {
                let a_ptr = a.data_ptr() as *const f64;
                let b_ptr = b.data_ptr() as *const f64;
                let result_ptr = result.data_ptr_mut() as *mut f64;
                
                let mut sum = 0.0;
                for i in 0..(a_shape[0] as usize) {
                    sum += *a_ptr.add(i) * *b_ptr.add(i);
                }
                *result_ptr = sum;
            }
            
            Ok(result)
        }
        (2, 2) => {
            // 2D-2D: matrix multiplication
            if a_shape[1] != b_shape[0] {
                return Err(LinalgError::ShapeMismatch);
            }
            
            let dtype = a.dtype().clone();
            let output_shape = vec![a_shape[0], b_shape[1]];
            let mut result = Array::new(output_shape, dtype)?;
            
            unsafe {
                let a_ptr = a.data_ptr() as *const f64;
                let b_ptr = b.data_ptr() as *const f64;
                let result_ptr = result.data_ptr_mut() as *mut f64;
                
                let m = a_shape[0] as usize;
                let n = a_shape[1] as usize;
                let p = b_shape[1] as usize;
                
                for i in 0..m {
                    for j in 0..p {
                        let mut sum = 0.0;
                        for k in 0..n {
                            sum += *a_ptr.add(i * n + k) * *b_ptr.add(k * p + j);
                        }
                        *result_ptr.add(i * p + j) = sum;
                    }
                }
            }
            
            Ok(result)
        }
        (1, 2) => {
            // 1D-2D: vector-matrix product
            if a_shape[0] != b_shape[0] {
                return Err(LinalgError::ShapeMismatch);
            }
            
            let dtype = a.dtype().clone();
            let output_shape = vec![b_shape[1]];
            let mut result = Array::new(output_shape, dtype)?;
            
            unsafe {
                let a_ptr = a.data_ptr() as *const f64;
                let b_ptr = b.data_ptr() as *const f64;
                let result_ptr = result.data_ptr_mut() as *mut f64;
                
                let n = a_shape[0] as usize;
                let p = b_shape[1] as usize;
                
                for j in 0..p {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += *a_ptr.add(k) * *b_ptr.add(k * p + j);
                    }
                    *result_ptr.add(j) = sum;
                }
            }
            
            Ok(result)
        }
        (2, 1) => {
            // 2D-1D: matrix-vector product
            if a_shape[1] != b_shape[0] {
                return Err(LinalgError::ShapeMismatch);
            }
            
            let dtype = a.dtype().clone();
            let output_shape = vec![a_shape[0]];
            let mut result = Array::new(output_shape, dtype)?;
            
            unsafe {
                let a_ptr = a.data_ptr() as *const f64;
                let b_ptr = b.data_ptr() as *const f64;
                let result_ptr = result.data_ptr_mut() as *mut f64;
                
                let m = a_shape[0] as usize;
                let n = a_shape[1] as usize;
                
                for i in 0..m {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += *a_ptr.add(i * n + k) * *b_ptr.add(k);
                    }
                    *result_ptr.add(i) = sum;
                }
            }
            
            Ok(result)
        }
        _ => Err(LinalgError::InvalidDimension),
    }
}

/// Matrix multiplication
///
/// Similar to dot but with stricter broadcasting rules
pub fn matmul(a: &Array, b: &Array) -> Result<Array, LinalgError> {
    // For now, delegate to dot for 2D case
    // Full implementation would handle ND arrays with broadcasting
    dot(a, b)
}

