//! Masked array structure

use crate::array::{Array, ArrayError};

/// Masked array error
#[derive(Debug, Clone)]
pub enum MaskedError {
    /// Array error
    ArrayError(ArrayError),
    /// Mask shape mismatch
    MaskShapeMismatch,
    /// Invalid mask (not boolean)
    InvalidMask,
}

impl std::fmt::Display for MaskedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaskedError::ArrayError(e) => write!(f, "Array error: {}", e),
            MaskedError::MaskShapeMismatch => write!(f, "Mask shape does not match array shape"),
            MaskedError::InvalidMask => write!(f, "Mask must be boolean array"),
        }
    }
}

impl std::error::Error for MaskedError {}

impl From<ArrayError> for MaskedError {
    fn from(err: ArrayError) -> Self {
        MaskedError::ArrayError(err)
    }
}

/// Masked array structure
///
/// Wraps an array with a boolean mask indicating which values are valid
pub struct MaskedArray {
    /// The underlying data array
    data: Array,
    /// Boolean mask (true = masked/invalid, false = valid)
    mask: Array,
}

impl MaskedArray {
    /// Create a new masked array
    ///
    /// # Arguments
    /// * `data` - Data array
    /// * `mask` - Boolean mask array (must match data shape)
    ///
    /// # Returns
    /// * `Ok(MaskedArray)` if successful
    /// * `Err(MaskedError)` if mask shape doesn't match or mask is not boolean
    pub fn new(data: Array, mask: Array) -> Result<Self, MaskedError> {
        // Validate mask is boolean
        if mask.dtype().type_() != crate::types::NpyType::Bool {
            return Err(MaskedError::InvalidMask);
        }
        
        // Validate mask shape matches data shape
        if mask.shape() != data.shape() {
            return Err(MaskedError::MaskShapeMismatch);
        }
        
        Ok(MaskedArray { data, mask })
    }
    
    /// Get reference to data array
    pub fn data(&self) -> &Array {
        &self.data
    }
    
    /// Get mutable reference to data array
    pub fn data_mut(&mut self) -> &mut Array {
        &mut self.data
    }
    
    /// Get reference to mask array
    pub fn mask(&self) -> &Array {
        &self.mask
    }
    
    /// Get shape of the masked array
    pub fn shape(&self) -> &[i64] {
        self.data.shape()
    }
    
    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }
    
    /// Get size (total number of elements)
    pub fn size(&self) -> usize {
        self.data.size()
    }
    
    /// Check if element at index is masked
    pub fn is_masked(&self, index: usize) -> Result<bool, MaskedError> {
        if index >= self.size() {
            return Err(MaskedError::ArrayError(ArrayError::InvalidShape));
        }
        
        unsafe {
            let mask_ptr = self.mask.data_ptr() as *const bool;
            Ok(*mask_ptr.add(index))
        }
    }
    
    /// Count number of masked elements
    pub fn count_masked(&self) -> usize {
        let size = self.size();
        let mut count = 0;
        
        unsafe {
            let mask_ptr = self.mask.data_ptr() as *const bool;
            for i in 0..size {
                if *mask_ptr.add(i) {
                    count += 1;
                }
            }
        }
        
        count
    }
    
    /// Count number of valid (unmasked) elements
    pub fn count_valid(&self) -> usize {
        self.size() - self.count_masked()
    }
}

