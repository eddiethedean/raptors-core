//! Masked array reduction operations

use crate::array::Array;

use super::{MaskedArray, MaskedError};

/// Sum masked array, skipping masked values
///
/// # Arguments
/// * `array` - Masked array
/// * `axis` - Axis along which to sum (None means all elements)
///
/// # Returns
/// * `Ok(Array)` - Sum result
/// * `Err(MaskedError)` if reduction fails
pub fn masked_sum(array: &MaskedArray, _axis: Option<usize>) -> Result<Array, MaskedError> {
    // Simplified implementation - sums all valid (unmasked) values
    let data = array.data();
    let mask = array.mask();
    let size = array.size();
    
    match data.dtype().type_() {
        crate::types::NpyType::Double => {
            let mut sum = 0.0;
            
            unsafe {
                let data_ptr = data.data_ptr() as *const f64;
                let mask_ptr = mask.data_ptr() as *const bool;
                
                for i in 0..size {
                    if !*mask_ptr.add(i) {
                        sum += *data_ptr.add(i);
                    }
                }
            }
            
            let dtype = data.dtype().clone();
            let mut result = Array::new(vec![1], dtype)?;
            unsafe {
                let result_ptr = result.data_ptr_mut() as *mut f64;
                *result_ptr = sum;
            }
            Ok(result)
        }
        crate::types::NpyType::Float => {
            let mut sum = 0.0f32;
            
            unsafe {
                let data_ptr = data.data_ptr() as *const f32;
                let mask_ptr = mask.data_ptr() as *const bool;
                
                for i in 0..size {
                    if !*mask_ptr.add(i) {
                        sum += *data_ptr.add(i);
                    }
                }
            }
            
            let dtype = data.dtype().clone();
            let mut result = Array::new(vec![1], dtype)?;
            unsafe {
                let result_ptr = result.data_ptr_mut() as *mut f32;
                *result_ptr = sum;
            }
            Ok(result)
        }
        _ => Err(MaskedError::ArrayError(crate::array::ArrayError::TypeMismatch)),
    }
}

/// Mean of masked array, skipping masked values
pub fn masked_mean(array: &MaskedArray, axis: Option<usize>) -> Result<Array, MaskedError> {
    let mut sum_result = masked_sum(array, axis)?;
    let valid_count = array.count_valid() as f64;
    
    if valid_count == 0.0 {
        return Err(MaskedError::ArrayError(crate::array::ArrayError::InvalidShape));
    }
    
    match sum_result.dtype().type_() {
        crate::types::NpyType::Double => {
            unsafe {
                let ptr = sum_result.data_ptr_mut() as *mut f64;
                *ptr /= valid_count;
            }
        }
        crate::types::NpyType::Float => {
            unsafe {
                let ptr = sum_result.data_ptr_mut() as *mut f32;
                *ptr /= valid_count as f32;
            }
        }
        _ => {}
    }
    
    Ok(sum_result)
}

/// Minimum of masked array, skipping masked values
pub fn masked_min(array: &MaskedArray, _axis: Option<usize>) -> Result<Array, MaskedError> {
    let data = array.data();
    let mask = array.mask();
    let size = array.size();
    
    match data.dtype().type_() {
        crate::types::NpyType::Double => {
            let mut min_val = f64::INFINITY;
            
            unsafe {
                let data_ptr = data.data_ptr() as *const f64;
                let mask_ptr = mask.data_ptr() as *const bool;
                
                for i in 0..size {
                    if !*mask_ptr.add(i) {
                        let val = *data_ptr.add(i);
                        if val < min_val {
                            min_val = val;
                        }
                    }
                }
            }
            
            let dtype = data.dtype().clone();
            let mut result = Array::new(vec![1], dtype)?;
            unsafe {
                let result_ptr = result.data_ptr_mut() as *mut f64;
                *result_ptr = min_val;
            }
            Ok(result)
        }
        _ => Err(MaskedError::ArrayError(crate::array::ArrayError::TypeMismatch)),
    }
}

/// Maximum of masked array, skipping masked values
pub fn masked_max(array: &MaskedArray, _axis: Option<usize>) -> Result<Array, MaskedError> {
    let data = array.data();
    let mask = array.mask();
    let size = array.size();
    
    match data.dtype().type_() {
        crate::types::NpyType::Double => {
            let mut max_val = f64::NEG_INFINITY;
            
            unsafe {
                let data_ptr = data.data_ptr() as *const f64;
                let mask_ptr = mask.data_ptr() as *const bool;
                
                for i in 0..size {
                    if !*mask_ptr.add(i) {
                        let val = *data_ptr.add(i);
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
            }
            
            let dtype = data.dtype().clone();
            let mut result = Array::new(vec![1], dtype)?;
            unsafe {
                let result_ptr = result.data_ptr_mut() as *mut f64;
                *result_ptr = max_val;
            }
            Ok(result)
        }
        _ => Err(MaskedError::ArrayError(crate::array::ArrayError::TypeMismatch)),
    }
}

