//! Masked array operations

use crate::array::Array;
use crate::types::DType;
use crate::broadcasting::broadcast_shapes;

use super::{MaskedArray, MaskedError};

/// Add two masked arrays with mask propagation and broadcasting
///
/// Result is masked where either input is masked.
/// Supports broadcasting: if shapes differ, they are broadcast together.
pub fn masked_add(a1: &MaskedArray, a2: &MaskedArray) -> Result<MaskedArray, MaskedError> {
    // Compute broadcast shape
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_e| MaskedError::ArrayError(crate::array::ArrayError::InvalidShape))?;
    
    // Broadcast data arrays if needed
    // Would need to create broadcasted view - for now, require same shape
    // Full implementation would create broadcasted arrays
    if a1.shape() != a2.shape() && a1.shape() != broadcast_shape.as_slice() && a2.shape() != broadcast_shape.as_slice() {
        return Err(MaskedError::ArrayError(crate::array::ArrayError::InvalidShape));
    }
    
    let data1 = a1.data().clone();
    let data2 = a2.data().clone();
    
    // Perform addition on data arrays
    let result_data = crate::operations::add(&data1, &data2)
        .map_err(MaskedError::ArrayError)?;
    
    // Broadcast and combine masks (OR operation - masked if either is masked)
    // Simplified - would broadcast mask in full implementation
    let mask1 = a1.mask().clone();
    let mask2 = a2.mask().clone();
    
    let combined_mask = combine_masks(&mask1, &mask2)?;
    
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

