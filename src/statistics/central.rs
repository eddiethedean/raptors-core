//! Central tendency measures (median, mode)

use crate::array::{Array, ArrayError};

use super::{StatisticsError, percentile};

/// Compute median of array
///
/// # Arguments
/// * `array` - Array to compute median for
/// * `axis` - Axis along which to compute (None means all elements)
///
/// # Returns
/// * `Ok(Array)` - Median value(s)
/// * `Err(StatisticsError)` if computation fails
pub fn median(array: &Array, axis: Option<usize>) -> Result<Array, StatisticsError> {
    // Median is 50th percentile
    percentile(array, 50.0, axis)
}

/// Compute mode of array
///
/// Returns the most frequently occurring value
pub fn mode(array: &Array, axis: Option<usize>) -> Result<Array, StatisticsError> {
    // Simplified implementation - for full array only
    if axis.is_some() {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    match array.dtype().type_() {
        crate::types::NpyType::Double => mode_double(array),
        crate::types::NpyType::Float => mode_float(array),
        crate::types::NpyType::Int => mode_int(array),
        crate::types::NpyType::Long => mode_long(array),
        _ => Err(StatisticsError::UnsupportedType),
    }
}

/// Mode for double
fn mode_double(array: &Array) -> Result<Array, StatisticsError> {
    let size = array.size();
    if size == 0 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    unsafe {
        let ptr = array.data_ptr() as *const f64;
        let mut counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        
        // Count occurrences (using rounded values for floating point)
        for i in 0..size {
            let val = *ptr.add(i);
            let key = (val * 1e10) as i64; // Round to avoid floating point issues
            *counts.entry(key).or_insert(0) += 1;
        }
        
        // Find most frequent
        let (most_frequent_key, _) = counts.iter().max_by_key(|(_, &count)| count)
            .ok_or(StatisticsError::ArrayError(ArrayError::InvalidShape))?;
        
        let mode_val = *most_frequent_key as f64 / 1e10;
        
        let dtype = array.dtype().clone();
        let mut output = Array::new(vec![1], dtype)?;
        let out_ptr = output.data_ptr_mut() as *mut f64;
        *out_ptr = mode_val;
        
        Ok(output)
    }
}

/// Mode for float
fn mode_float(array: &Array) -> Result<Array, StatisticsError> {
    let size = array.size();
    if size == 0 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    unsafe {
        let ptr = array.data_ptr() as *const f32;
        let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        
        for i in 0..size {
            let val = *ptr.add(i);
            let key = (val * 1e6) as i32;
            *counts.entry(key).or_insert(0) += 1;
        }
        
        let (most_frequent_key, _) = counts.iter().max_by_key(|(_, &count)| count)
            .ok_or(StatisticsError::ArrayError(ArrayError::InvalidShape))?;
        
        let mode_val = *most_frequent_key as f32 / 1e6;
        
        let dtype = array.dtype().clone();
        let mut output = Array::new(vec![1], dtype)?;
        let out_ptr = output.data_ptr_mut() as *mut f32;
        *out_ptr = mode_val;
        
        Ok(output)
    }
}

/// Mode for int
fn mode_int(array: &Array) -> Result<Array, StatisticsError> {
    let size = array.size();
    if size == 0 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    unsafe {
        let ptr = array.data_ptr() as *const i32;
        let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        
        for i in 0..size {
            let val = *ptr.add(i);
            *counts.entry(val).or_insert(0) += 1;
        }
        
        let (mode_val, _) = counts.iter().max_by_key(|(_, &count)| count)
            .ok_or(StatisticsError::ArrayError(ArrayError::InvalidShape))?;
        
        let dtype = array.dtype().clone();
        let mut output = Array::new(vec![1], dtype)?;
        let out_ptr = output.data_ptr_mut() as *mut i32;
        *out_ptr = *mode_val;
        
        Ok(output)
    }
}

/// Mode for long
fn mode_long(array: &Array) -> Result<Array, StatisticsError> {
    let size = array.size();
    if size == 0 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    unsafe {
        let ptr = array.data_ptr() as *const i64;
        let mut counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        
        for i in 0..size {
            let val = *ptr.add(i);
            *counts.entry(val).or_insert(0) += 1;
        }
        
        let (mode_val, _) = counts.iter().max_by_key(|(_, &count)| count)
            .ok_or(StatisticsError::ArrayError(ArrayError::InvalidShape))?;
        
        let dtype = array.dtype().clone();
        let mut output = Array::new(vec![1], dtype)?;
        let out_ptr = output.data_ptr_mut() as *mut i64;
        *out_ptr = *mode_val;
        
        Ok(output)
    }
}

