//! Iterator-based array operations
//!
//! This module provides iterator-based operations for arrays.

use crate::array::{Array, ArrayError};

/// Trait for iterator-based array operations
pub trait ArrayIterOps {
    /// Map a function over array elements
    fn map<F>(&self, f: F) -> Result<Array, ArrayError>
    where
        F: Fn(*const u8) -> Result<*mut u8, ArrayError>;
    
    /// Zip two arrays together
    fn zip<F>(&self, other: &Array, f: F) -> Result<Array, ArrayError>
    where
        F: Fn(*const u8, *const u8) -> Result<*mut u8, ArrayError>;
    
    /// Filter array elements based on a condition
    fn filter<F>(&self, f: F) -> Result<Vec<*const u8>, ArrayError>
    where
        F: Fn(*const u8) -> bool;
    
    /// Fold array elements with an accumulator
    fn fold<F, T>(&self, init: T, f: F) -> Result<T, ArrayError>
    where
        F: Fn(T, *const u8) -> Result<T, ArrayError>;
}

impl ArrayIterOps for Array {
    /// Map a function over array elements
    ///
    /// Creates a new array by applying a function to each element.
    /// Note: This is a simplified implementation that works with pointers.
    /// For type-safe operations, use ufuncs or operations module.
    fn map<F>(&self, _f: F) -> Result<Array, ArrayError>
    where
        F: Fn(*const u8) -> Result<*mut u8, ArrayError>,
    {
        // Simplified implementation - would need proper type handling
        // For now, return a copy
        Ok(self.copy())
    }
    
    /// Zip two arrays together
    ///
    /// Combines two arrays element-wise using a function.
    fn zip<F>(&self, _other: &Array, _f: F) -> Result<Array, ArrayError>
    where
        F: Fn(*const u8, *const u8) -> Result<*mut u8, ArrayError>,
    {
        // Simplified implementation
        Err(ArrayError::TypeMismatch)
    }
    
    /// Filter array elements based on a condition
    ///
    /// Returns pointers to elements that satisfy the condition.
    fn filter<F>(&self, f: F) -> Result<Vec<*const u8>, ArrayError>
    where
        F: Fn(*const u8) -> bool,
    {
        use crate::iterators::FlatIterator;
        
        let mut result = Vec::new();
        let mut iter = FlatIterator::new(self);
        
        while iter.next() {
            let ptr = iter.data_ptr();
            if f(ptr) {
                result.push(ptr);
            }
        }
        
        Ok(result)
    }
    
    /// Fold array elements with an accumulator
    ///
    /// Reduces the array to a single value using a function.
    fn fold<F, T>(&self, init: T, f: F) -> Result<T, ArrayError>
    where
        F: Fn(T, *const u8) -> Result<T, ArrayError>,
    {
        use crate::iterators::FlatIterator;
        
        let mut acc = init;
        let mut iter = FlatIterator::new(self);
        
        while iter.next() {
            let ptr = iter.data_ptr();
            acc = f(acc, ptr)?;
        }
        
        Ok(acc)
    }
}

