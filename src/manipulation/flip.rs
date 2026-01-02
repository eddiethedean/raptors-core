//! Array flipping operations
//!
//! This module provides flip operations (flipud, fliplr, flip)

use crate::array::{Array, ArrayError};

/// Manipulation error
#[derive(Debug, Clone)]
pub enum ManipulationError {
    /// Array error
    ArrayError(ArrayError),
    /// Invalid axis
    InvalidAxis,
}

impl std::fmt::Display for ManipulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManipulationError::ArrayError(e) => write!(f, "Array error: {}", e),
            ManipulationError::InvalidAxis => write!(f, "Invalid axis"),
        }
    }
}

impl std::error::Error for ManipulationError {}

impl From<ArrayError> for ManipulationError {
    fn from(err: ArrayError) -> Self {
        ManipulationError::ArrayError(err)
    }
}

/// Flip array along specified axis
///
/// # Arguments
/// * `array` - Array to flip
/// * `axis` - Axis along which to flip
///
/// # Returns
/// * `Ok(Array)` - Flipped array
/// * `Err(ManipulationError)` if flip fails
pub fn flip(array: &Array, axis: usize) -> Result<Array, ManipulationError> {
    if axis >= array.ndim() {
        return Err(ManipulationError::InvalidAxis);
    }
    
    let shape = array.shape().to_vec();
    let dtype = array.dtype().clone();
    let mut output = Array::new(shape.clone(), dtype)?;
    
    let size = array.size();
    if size == 0 {
        return Ok(output);
    }
    
    // Copy data with axis flipped
    copy_flipped(array, &mut output, axis)?;
    
    Ok(output)
}

/// Flip array vertically (along first axis)
pub fn flipud(array: &Array) -> Result<Array, ManipulationError> {
    if array.ndim() == 0 {
        return Err(ManipulationError::InvalidAxis);
    }
    flip(array, 0)
}

/// Flip array horizontally (along last axis)
pub fn fliplr(array: &Array) -> Result<Array, ManipulationError> {
    if array.ndim() == 0 {
        return Err(ManipulationError::InvalidAxis);
    }
    flip(array, array.ndim() - 1)
}

/// Copy array with axis flipped
fn copy_flipped(src: &Array, dst: &mut Array, axis: usize) -> Result<(), ManipulationError> {
    let shape = src.shape();
    let src_strides = src.strides().to_vec();
    let dst_strides = dst.strides().to_vec();
    let itemsize = src.itemsize();
    
    // Iterate through all elements
    let mut src_coords = vec![0; shape.len()];
    let mut dst_coords = vec![0; shape.len()];
    
    let axis_size = shape[axis] as usize;
    
    for flat_idx in 0..src.size() {
        // Convert flat index to coordinates
        index_to_coords(flat_idx, shape, &mut src_coords);
        
        // Flip coordinate along axis
        dst_coords.copy_from_slice(&src_coords);
        dst_coords[axis] = (axis_size - 1) as i64 - src_coords[axis];
        
        // Compute offsets
        let src_offset = coords_to_offset(&src_coords, src_strides.as_slice());
        let dst_offset = coords_to_offset(&dst_coords, dst_strides.as_slice());
        
        // Copy element
        unsafe {
            let src_ptr = src.data_ptr().add(src_offset);
            let dst_ptr = dst.data_ptr_mut().add(dst_offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize);
        }
    }
    
    Ok(())
}

/// Convert flat index to coordinates
fn index_to_coords(index: usize, shape: &[i64], coords: &mut [i64]) {
    let mut idx = index;
    for i in (0..shape.len()).rev() {
        coords[i] = (idx % shape[i] as usize) as i64;
        idx /= shape[i] as usize;
    }
}

/// Convert coordinates to byte offset
fn coords_to_offset(coords: &[i64], strides: &[i64]) -> usize {
    let mut offset = 0;
    for (i, &coord) in coords.iter().enumerate() {
        offset += (coord * strides[i]) as usize;
    }
    offset
}

