//! Utility functions module
//!
//! This module provides various utility functions used throughout the crate

/// Compute the total size from a shape
pub fn compute_size(shape: &[i64]) -> usize {
    shape.iter().product::<i64>() as usize
}

/// Check if shape is valid
pub fn is_valid_shape(shape: &[i64]) -> bool {
    shape.iter().all(|&dim| dim >= 0) && shape.len() <= crate::array::MAXDIMS
}

