//! Array rolling (circular shift) operations

use crate::array::Array;

use super::ManipulationError;

/// Roll array elements along specified axis
///
/// Elements that roll beyond the last position are re-introduced at the first
///
/// # Arguments
/// * `array` - Array to roll
/// * `shift` - Number of positions to roll (can be negative)
/// * `axis` - Axis along which to roll (None means flatten first)
///
/// # Returns
/// * `Ok(Array)` - Rolled array
/// * `Err(ManipulationError)` if roll fails
pub fn roll(array: &Array, shift: i64, axis: Option<usize>) -> Result<Array, ManipulationError> {
    let shape = array.shape().to_vec();
    let dtype = array.dtype().clone();
    let mut output = Array::new(shape.clone(), dtype)?;
    
    if let Some(ax) = axis {
        if ax >= array.ndim() {
            return Err(ManipulationError::InvalidAxis);
        }
        roll_along_axis(array, &mut output, shift, ax)?;
    } else {
        // Roll flattened array
        roll_flat(array, &mut output, shift)?;
    }
    
    Ok(output)
}

/// Roll along specific axis
fn roll_along_axis(
    src: &Array,
    dst: &mut Array,
    shift: i64,
    axis: usize,
) -> Result<(), ManipulationError> {
    let shape = src.shape();
    let axis_size = shape[axis];
    let normalized_shift = ((shift % axis_size) + axis_size) % axis_size;
    
    let itemsize = src.itemsize();
    let src_strides = src.strides().to_vec();
    let dst_strides = dst.strides().to_vec();
    
    // Iterate through all elements
    let mut src_coords = vec![0; shape.len()];
    let mut dst_coords = vec![0; shape.len()];
    
    for flat_idx in 0..src.size() {
        index_to_coords(flat_idx, shape, &mut src_coords);
        
        dst_coords.copy_from_slice(&src_coords);
        dst_coords[axis] = (src_coords[axis] + normalized_shift) % axis_size;
        
        let src_offset = coords_to_offset(&src_coords, src_strides.as_slice());
        let dst_offset = coords_to_offset(&dst_coords, dst_strides.as_slice());
        
        unsafe {
            let src_ptr = src.data_ptr().add(src_offset);
            let dst_ptr = dst.data_ptr_mut().add(dst_offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize);
        }
    }
    
    Ok(())
}

/// Roll flattened array
fn roll_flat(src: &Array, dst: &mut Array, shift: i64) -> Result<(), ManipulationError> {
    let size = src.size() as i64;
    let normalized_shift = ((shift % size) + size) % size;
    let itemsize = src.itemsize();
    
    unsafe {
        let src_ptr = src.data_ptr();
        let dst_ptr = dst.data_ptr_mut();
        
        // Copy elements with shift
        for i in 0..size {
            let src_idx = ((i - normalized_shift + size) % size) as usize;
            let dst_idx = i as usize;
            
            std::ptr::copy_nonoverlapping(
                src_ptr.add(src_idx * itemsize),
                dst_ptr.add(dst_idx * itemsize),
                itemsize,
            );
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

