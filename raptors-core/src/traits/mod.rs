//! Traits for array extensibility
//!
//! This module provides traits for extending array functionality.

use crate::array::{Array, ArrayError};
use crate::types::DType;

/// Trait for array-like objects
///
/// This trait defines common operations that can be performed on array-like objects.
pub trait ArrayLike {
    /// Get the shape of the array
    fn shape(&self) -> &[i64];
    
    /// Get the dtype
    fn dtype(&self) -> &DType;
    
    /// Get the size (total number of elements)
    fn size(&self) -> usize;
    
    /// Get the number of dimensions
    fn ndim(&self) -> usize;
}

/// Trait for indexable objects
///
/// This trait defines indexing operations.
pub trait Indexable {
    /// Get a pointer to an element at the given indices
    fn index(&self, indices: &[i64]) -> Result<*const u8, ArrayError>;
    
    /// Get a mutable pointer to an element at the given indices
    fn index_mut(&mut self, indices: &[i64]) -> Result<*mut u8, ArrayError>;
}

/// Trait for broadcastable objects
///
/// This trait defines broadcasting operations.
pub trait Broadcastable {
    /// Get the broadcast shape with another array
    fn broadcast_shape(&self, other: &dyn ArrayLike) -> Result<Vec<i64>, ArrayError>;
}

/// Trait for reducible objects
///
/// This trait defines reduction operations.
pub trait Reducible {
    /// Sum along an axis
    fn sum_axis(&self, axis: Option<usize>) -> Result<Array, ArrayError>;
    
    /// Mean along an axis
    fn mean_axis(&self, axis: Option<usize>) -> Result<Array, ArrayError>;
    
    /// Min along an axis
    fn min_axis(&self, axis: Option<usize>) -> Result<Array, ArrayError>;
    
    /// Max along an axis
    fn max_axis(&self, axis: Option<usize>) -> Result<Array, ArrayError>;
}

// Implement ArrayLike for Array
impl ArrayLike for Array {
    fn shape(&self) -> &[i64] {
        self.shape()
    }
    
    fn dtype(&self) -> &DType {
        self.dtype()
    }
    
    fn size(&self) -> usize {
        self.size()
    }
    
    fn ndim(&self) -> usize {
        self.ndim()
    }
}

// Implement Indexable for Array
impl Indexable for Array {
    fn index(&self, indices: &[i64]) -> Result<*const u8, ArrayError> {
        use crate::indexing::index_array;
        index_array(self, indices)
            .map_err(|_| ArrayError::InvalidShape)
    }
    
    fn index_mut(&mut self, indices: &[i64]) -> Result<*mut u8, ArrayError> {
        use crate::indexing::index_array;
        // For now, use the const version and cast
        // In a full implementation, would have a mutable version
        let ptr = index_array(self, indices)
            .map_err(|_| ArrayError::InvalidShape)?;
        Ok(ptr as *mut u8)
    }
}

// Implement Broadcastable for Array
impl Broadcastable for Array {
    fn broadcast_shape(&self, other: &dyn ArrayLike) -> Result<Vec<i64>, ArrayError> {
        use crate::broadcasting::broadcast_shapes;
        broadcast_shapes(self.shape(), other.shape())
            .map_err(|_| ArrayError::InvalidShape)
    }
}

// Implement Reducible for Array
impl Reducible for Array {
    fn sum_axis(&self, axis: Option<usize>) -> Result<Array, ArrayError> {
        use crate::ufunc::reduction::sum_along_axis;
        sum_along_axis(self, axis)
            .map_err(|_| ArrayError::TypeMismatch)
    }
    
    fn mean_axis(&self, axis: Option<usize>) -> Result<Array, ArrayError> {
        use crate::ufunc::reduction::mean_along_axis;
        mean_along_axis(self, axis)
            .map_err(|_| ArrayError::TypeMismatch)
    }
    
    fn min_axis(&self, axis: Option<usize>) -> Result<Array, ArrayError> {
        use crate::ufunc::reduction::min_along_axis;
        min_along_axis(self, axis)
            .map_err(|_| ArrayError::TypeMismatch)
    }
    
    fn max_axis(&self, axis: Option<usize>) -> Result<Array, ArrayError> {
        use crate::ufunc::reduction::max_along_axis;
        max_along_axis(self, axis)
            .map_err(|_| ArrayError::TypeMismatch)
    }
}

