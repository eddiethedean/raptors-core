//! Masked array operations

use crate::array::Array;
use crate::types::DType;

use super::{MaskedArray, MaskedError};

/// Add two masked arrays with mask propagation
///
/// Result is masked where either input is masked
pub fn masked_add(a1: &MaskedArray, a2: &MaskedArray) -> Result<MaskedArray, MaskedError> {
    if a1.shape() != a2.shape() {
        return Err(MaskedError::ArrayError(crate::array::ArrayError::InvalidShape));
    }
    
    // Perform addition on data arrays (simplified - would use proper broadcasting)
    let result_data = crate::operations::add(a1.data(), a2.data())
        .map_err(MaskedError::ArrayError)?;
    
    // Combine masks (OR operation - masked if either is masked)
    let combined_mask = combine_masks(a1.mask(), a2.mask())?;
    
    MaskedArray::new(result_data, combined_mask)
}

/// Subtract two masked arrays
pub fn masked_subtract(a1: &MaskedArray, a2: &MaskedArray) -> Result<MaskedArray, MaskedError> {
    if a1.shape() != a2.shape() {
        return Err(MaskedError::ArrayError(crate::array::ArrayError::InvalidShape));
    }
    
    let result_data = crate::operations::subtract(a1.data(), a2.data())
        .map_err(MaskedError::ArrayError)?;
    let combined_mask = combine_masks(a1.mask(), a2.mask())?;
    
    MaskedArray::new(result_data, combined_mask)
}

/// Multiply two masked arrays
pub fn masked_multiply(a1: &MaskedArray, a2: &MaskedArray) -> Result<MaskedArray, MaskedError> {
    if a1.shape() != a2.shape() {
        return Err(MaskedError::ArrayError(crate::array::ArrayError::InvalidShape));
    }
    
    let result_data = crate::operations::multiply(a1.data(), a2.data())
        .map_err(MaskedError::ArrayError)?;
    let combined_mask = combine_masks(a1.mask(), a2.mask())?;
    
    MaskedArray::new(result_data, combined_mask)
}

/// Combine two masks using OR operation
fn combine_masks(mask1: &Array, mask2: &Array) -> Result<Array, MaskedError> {
    let shape = mask1.shape().to_vec();
    let size = mask1.size();
    let bool_dtype = DType::new(crate::types::NpyType::Bool);
    let mut result = Array::new(shape, bool_dtype)?;
    
    unsafe {
        let m1_ptr = mask1.data_ptr() as *const bool;
        let m2_ptr = mask2.data_ptr() as *const bool;
        let result_ptr = result.data_ptr_mut() as *mut bool;
        
        for i in 0..size {
            *result_ptr.add(i) = *m1_ptr.add(i) || *m2_ptr.add(i);
        }
    }
    
    Ok(result)
}

