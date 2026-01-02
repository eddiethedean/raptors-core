//! Array indexing implementation
//!
//! This module provides indexing operations for arrays

use crate::array::Array;

/// Indexing error
#[derive(Debug, Clone)]
pub enum IndexError {
    /// Index out of bounds
    OutOfBounds,
    /// Invalid index type
    InvalidIndex,
    /// Dimension mismatch
    DimensionMismatch,
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::OutOfBounds => write!(f, "Index out of bounds"),
            IndexError::InvalidIndex => write!(f, "Invalid index"),
            IndexError::DimensionMismatch => write!(f, "Dimension mismatch"),
        }
    }
}

impl std::error::Error for IndexError {}

impl From<crate::array::ArrayError> for IndexError {
    fn from(_: crate::array::ArrayError) -> Self {
        IndexError::InvalidIndex
    }
}

/// Index an array element by integer indices
///
/// Returns a pointer to the element at the given indices.
/// The indices must match the array's dimensionality.
pub fn index_array(array: &Array, indices: &[i64]) -> Result<*const u8, IndexError> {
    // Validate dimension count
    if indices.len() != array.ndim() {
        return Err(IndexError::DimensionMismatch);
    }
    
    // Validate indices are within bounds
    let shape = array.shape();
    for (i, &idx) in indices.iter().enumerate() {
        if idx < 0 || idx >= shape[i] {
            return Err(IndexError::OutOfBounds);
        }
    }
    
    // Calculate offset using strides
    let strides = array.strides();
    let mut offset: i64 = 0;
    for (i, &idx) in indices.iter().enumerate() {
        offset += idx * strides[i];
    }
    
    // Return pointer to the element
    unsafe {
        Ok(array.data_ptr().add(offset as usize))
    }
}

/// Index an array element by integer indices (mutable)
///
/// Returns a mutable pointer to the element at the given indices.
pub fn index_array_mut(array: &mut Array, indices: &[i64]) -> Result<*mut u8, IndexError> {
    // Validate dimension count
    if indices.len() != array.ndim() {
        return Err(IndexError::DimensionMismatch);
    }
    
    // Validate indices are within bounds
    let shape = array.shape();
    for (i, &idx) in indices.iter().enumerate() {
        if idx < 0 || idx >= shape[i] {
            return Err(IndexError::OutOfBounds);
        }
    }
    
    // Check if array is writeable
    if !array.is_writeable() {
        return Err(IndexError::InvalidIndex);
    }
    
    // Calculate offset using strides
    let strides = array.strides();
    let mut offset: i64 = 0;
    for (i, &idx) in indices.iter().enumerate() {
        offset += idx * strides[i];
    }
    
    // Return mutable pointer to the element
    unsafe {
        Ok(array.data_ptr_mut().add(offset as usize))
    }
}

/// Validate index for a single dimension
pub fn validate_index(index: i64, dim_size: i64) -> Result<(), IndexError> {
    if index < 0 || index >= dim_size {
        Err(IndexError::OutOfBounds)
    } else {
        Ok(())
    }
}

