//! Masked array creation functions

use crate::array::Array;
use crate::types::DType;

use super::{MaskedArray, MaskedError};

/// Create masked array from array and mask
///
/// # Arguments
/// * `array` - Data array
/// * `mask` - Boolean mask array
///
/// # Returns
/// * `Ok(MaskedArray)` if successful
/// * `Err(MaskedError)` if creation fails
pub fn masked_array(array: Array, mask: Array) -> Result<MaskedArray, MaskedError> {
    MaskedArray::new(array, mask)
}

/// Create masked array with specific indices masked
///
/// # Arguments
/// * `array` - Data array
/// * `mask_indices` - Indices to mask
///
/// # Returns
/// * `Ok(MaskedArray)` if successful
/// * `Err(MaskedError)` if creation fails
pub fn masked_array_with_indices(array: Array, mask_indices: &[usize]) -> Result<MaskedArray, MaskedError> {
    let shape = array.shape().to_vec();
    let size = array.size();
    let bool_dtype = DType::new(crate::types::NpyType::Bool);
    let mut mask = Array::new(shape, bool_dtype)?;
    
    // Initialize mask to all false (unmasked)
    unsafe {
        let mask_ptr = mask.data_ptr_mut() as *mut bool;
        for i in 0..size {
            *mask_ptr.add(i) = false;
        }
    }
    
    // Set specified indices to masked
    unsafe {
        let mask_ptr = mask.data_ptr_mut() as *mut bool;
        for &idx in mask_indices {
            if idx < size {
                *mask_ptr.add(idx) = true;
            }
        }
    }
    
    MaskedArray::new(array, mask)
}

/// Create masked array where mask is true for invalid values (NaN, etc.)
///
/// For floating point arrays, masks NaN and infinity values
pub fn masked_array_invalid(array: Array) -> Result<MaskedArray, MaskedError> {
    let shape = array.shape().to_vec();
    let size = array.size();
    let bool_dtype = DType::new(crate::types::NpyType::Bool);
    let mut mask = Array::new(shape, bool_dtype)?;
    
    match array.dtype().type_() {
        crate::types::NpyType::Double => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f64;
                let mask_ptr = mask.data_ptr_mut() as *mut bool;
                for i in 0..size {
                    let val = *data_ptr.add(i);
                    *mask_ptr.add(i) = val.is_nan() || val.is_infinite();
                }
            }
        }
        crate::types::NpyType::Float => {
            unsafe {
                let data_ptr = array.data_ptr() as *const f32;
                let mask_ptr = mask.data_ptr_mut() as *mut bool;
                for i in 0..size {
                    let val = *data_ptr.add(i);
                    *mask_ptr.add(i) = val.is_nan() || val.is_infinite();
                }
            }
        }
        _ => {
            // For non-float types, no invalid values to mask
            unsafe {
                let mask_ptr = mask.data_ptr_mut() as *mut bool;
                for i in 0..size {
                    *mask_ptr.add(i) = false;
                }
            }
        }
    }
    
    MaskedArray::new(array, mask)
}

