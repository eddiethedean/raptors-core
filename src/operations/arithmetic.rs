//! Arithmetic operations
//!
//! High-level arithmetic operations built on ufuncs

use crate::array::{Array, ArrayError};
use crate::broadcasting::broadcast_shapes;
use crate::types::NpyType;
use crate::conversion::promote_dtypes;

/// Add two arrays
///
/// Returns a new array with the result of element-wise addition
pub fn add(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    // Compute broadcast shape
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_| ArrayError::InvalidShape)?;
    
    // Promote types
    let output_dtype = promote_dtypes(a1.dtype(), a2.dtype())
        .map_err(|_| ArrayError::TypeMismatch)?;
    
    // Create output array
    let mut output = Array::new(broadcast_shape, output_dtype)?;
    
    // For now, simplified implementation - would use ufunc system
    // Simple element-wise addition for same-shaped arrays
    if a1.shape() == a2.shape() && a1.dtype().type_() == a2.dtype().type_() {
        let size = a1.size();
        match a1.dtype().type_() {
            NpyType::Double => {
                unsafe {
                    let in1_ptr = a1.data_ptr() as *const f64;
                    let in2_ptr = a2.data_ptr() as *const f64;
                    let out_ptr = output.data_ptr_mut() as *mut f64;
                    for i in 0..size {
                        *out_ptr.add(i) = *in1_ptr.add(i) + *in2_ptr.add(i);
                    }
                }
            }
            _ => return Err(ArrayError::TypeMismatch),
        }
    } else {
        return Err(ArrayError::TypeMismatch);
    }
    
    Ok(output)
}

/// Subtract two arrays
pub fn subtract(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_| ArrayError::InvalidShape)?;
    
    let output_dtype = promote_dtypes(a1.dtype(), a2.dtype())
        .map_err(|_| ArrayError::TypeMismatch)?;
    
    let mut output = Array::new(broadcast_shape, output_dtype)?;
    
    if a1.shape() == a2.shape() && a1.dtype().type_() == a2.dtype().type_() {
        let size = a1.size();
        match a1.dtype().type_() {
            NpyType::Double => {
                unsafe {
                    let in1_ptr = a1.data_ptr() as *const f64;
                    let in2_ptr = a2.data_ptr() as *const f64;
                    let out_ptr = output.data_ptr_mut() as *mut f64;
                    for i in 0..size {
                        *out_ptr.add(i) = *in1_ptr.add(i) - *in2_ptr.add(i);
                    }
                }
            }
            _ => return Err(ArrayError::TypeMismatch),
        }
    } else {
        return Err(ArrayError::TypeMismatch);
    }
    
    Ok(output)
}

/// Multiply two arrays
pub fn multiply(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_| ArrayError::InvalidShape)?;
    
    let output_dtype = promote_dtypes(a1.dtype(), a2.dtype())
        .map_err(|_| ArrayError::TypeMismatch)?;
    
    let mut output = Array::new(broadcast_shape, output_dtype)?;
    
    if a1.shape() == a2.shape() && a1.dtype().type_() == a2.dtype().type_() {
        let size = a1.size();
        match a1.dtype().type_() {
            NpyType::Double => {
                unsafe {
                    let in1_ptr = a1.data_ptr() as *const f64;
                    let in2_ptr = a2.data_ptr() as *const f64;
                    let out_ptr = output.data_ptr_mut() as *mut f64;
                    for i in 0..size {
                        *out_ptr.add(i) = *in1_ptr.add(i) * *in2_ptr.add(i);
                    }
                }
            }
            _ => return Err(ArrayError::TypeMismatch),
        }
    } else {
        return Err(ArrayError::TypeMismatch);
    }
    
    Ok(output)
}

/// Divide two arrays
pub fn divide(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_| ArrayError::InvalidShape)?;
    
    let output_dtype = promote_dtypes(a1.dtype(), a2.dtype())
        .map_err(|_| ArrayError::TypeMismatch)?;
    
    let mut output = Array::new(broadcast_shape, output_dtype)?;
    
    if a1.shape() == a2.shape() && a1.dtype().type_() == a2.dtype().type_() {
        let size = a1.size();
        match a1.dtype().type_() {
            NpyType::Double => {
                unsafe {
                    let in1_ptr = a1.data_ptr() as *const f64;
                    let in2_ptr = a2.data_ptr() as *const f64;
                    let out_ptr = output.data_ptr_mut() as *mut f64;
                    for i in 0..size {
                        *out_ptr.add(i) = *in1_ptr.add(i) / *in2_ptr.add(i);
                    }
                }
            }
            _ => return Err(ArrayError::TypeMismatch),
        }
    } else {
        return Err(ArrayError::TypeMismatch);
    }
    
    Ok(output)
}

