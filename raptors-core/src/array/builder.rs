//! Array builder for convenient array creation
//!
//! This module provides a builder pattern for creating arrays with various options.

use crate::array::{Array, ArrayError};
use crate::types::DType;

/// Memory order for array layout
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    /// C-contiguous (row-major)
    C,
    /// Fortran-contiguous (column-major)
    F,
}

/// Array builder for convenient array creation
///
/// Note: This is an internal implementation detail and not part of the public API.
pub struct ArrayBuilder {
    shape: Option<Vec<i64>>,
    dtype: Option<DType>,
    order: Option<MemoryOrder>,
    fill_value: Option<f64>,
}

impl ArrayBuilder {
    /// Create a new array builder
    pub fn new() -> Self {
        ArrayBuilder {
            shape: None,
            dtype: None,
            order: None,
            fill_value: None,
        }
    }
    
    /// Set the shape of the array
    pub fn with_shape(mut self, shape: Vec<i64>) -> Self {
        self.shape = Some(shape);
        self
    }
    
    /// Set the dtype of the array
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    
    /// Set the memory order (C or Fortran)
    pub fn with_order(mut self, order: MemoryOrder) -> Self {
        self.order = Some(order);
        self
    }
    
    /// Set a fill value (for creating filled arrays)
    pub fn with_fill_value(mut self, value: f64) -> Self {
        self.fill_value = Some(value);
        self
    }
    
    /// Build the array
    pub fn build(self) -> Result<Array, ArrayError> {
        let shape = self.shape.ok_or(ArrayError::InvalidShape)?;
        let dtype = self.dtype.unwrap_or_else(|| DType::new(crate::types::NpyType::Double));
        
        let mut array = Array::new(shape, dtype)?;
        
        // Apply fill value if specified
        if let Some(fill) = self.fill_value {
            // Simple fill for double precision
            if array.dtype().type_() == crate::types::NpyType::Double {
                unsafe {
                    let ptr = array.data_ptr_mut() as *mut f64;
                    let size = array.size();
                    for i in 0..size {
                        *ptr.add(i) = fill;
                    }
                }
            }
        }
        
        // Note: Memory order (C vs F) is determined by strides,
        // which are computed automatically. For now, we always use C order.
        // Full F-order support would require stride computation changes.
        
        Ok(array)
    }
}

impl Default for ArrayBuilder {
    fn default() -> Self {
        Self::new()
    }
}

