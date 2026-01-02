//! Unique element finding

use crate::array::{Array, ArrayError};

use super::ManipulationError;

/// Find unique elements in array
///
/// # Arguments
/// * `array` - Array to find unique elements in
///
/// # Returns
/// * `Ok(Array)` - Array of unique elements (sorted)
/// * `Err(ManipulationError)` if unique fails
pub fn unique(array: &Array) -> Result<Array, ManipulationError> {
    // For now, flatten array and find unique elements
    // Full implementation would handle multi-dimensional arrays properly
    
    let size = array.size();
    if size == 0 {
        let dtype = array.dtype().clone();
        return Ok(Array::new(vec![0], dtype)?);
    }
    
    match array.dtype().type_() {
        crate::types::NpyType::Double => unique_double(array),
        crate::types::NpyType::Float => unique_float(array),
        crate::types::NpyType::Int => unique_int(array),
        crate::types::NpyType::Long => unique_long(array),
        _ => Err(ManipulationError::ArrayError(ArrayError::TypeMismatch)),
    }
}

/// Unique for double
fn unique_double(array: &Array) -> Result<Array, ManipulationError> {
    let size = array.size();
    let dtype = array.dtype().clone();
    
    unsafe {
        let ptr = array.data_ptr() as *const f64;
        let mut values: Vec<f64> = (0..size).map(|i| *ptr.add(i)).collect();
        
        // Sort and deduplicate
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        
        let unique_size = values.len();
        let mut output = Array::new(vec![unique_size as i64], dtype)?;
        let out_ptr = output.data_ptr_mut() as *mut f64;
        
        for (i, &val) in values.iter().enumerate() {
            *out_ptr.add(i) = val;
        }
        
        Ok(output)
    }
}

/// Unique for float
fn unique_float(array: &Array) -> Result<Array, ManipulationError> {
    let size = array.size();
    let dtype = array.dtype().clone();
    
    unsafe {
        let ptr = array.data_ptr() as *const f32;
        let mut values: Vec<f32> = (0..size).map(|i| *ptr.add(i)).collect();
        
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        
        let unique_size = values.len();
        let mut output = Array::new(vec![unique_size as i64], dtype)?;
        let out_ptr = output.data_ptr_mut() as *mut f32;
        
        for (i, &val) in values.iter().enumerate() {
            *out_ptr.add(i) = val;
        }
        
        Ok(output)
    }
}

/// Unique for int
fn unique_int(array: &Array) -> Result<Array, ManipulationError> {
    let size = array.size();
    let dtype = array.dtype().clone();
    
    unsafe {
        let ptr = array.data_ptr() as *const i32;
        let mut values: Vec<i32> = (0..size).map(|i| *ptr.add(i)).collect();
        
        values.sort();
        values.dedup();
        
        let unique_size = values.len();
        let mut output = Array::new(vec![unique_size as i64], dtype)?;
        let out_ptr = output.data_ptr_mut() as *mut i32;
        
        for (i, &val) in values.iter().enumerate() {
            *out_ptr.add(i) = val;
        }
        
        Ok(output)
    }
}

/// Unique for long
fn unique_long(array: &Array) -> Result<Array, ManipulationError> {
    let size = array.size();
    let dtype = array.dtype().clone();
    
    unsafe {
        let ptr = array.data_ptr() as *const i64;
        let mut values: Vec<i64> = (0..size).map(|i| *ptr.add(i)).collect();
        
        values.sort();
        values.dedup();
        
        let unique_size = values.len();
        let mut output = Array::new(vec![unique_size as i64], dtype)?;
        let out_ptr = output.data_ptr_mut() as *mut i64;
        
        for (i, &val) in values.iter().enumerate() {
            *out_ptr.add(i) = val;
        }
        
        Ok(output)
    }
}

