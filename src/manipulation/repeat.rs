//! Array repeat operations

use crate::array::Array;

use super::ManipulationError;

/// Repeat elements of array along specified axis
///
/// # Arguments
/// * `array` - Array to repeat
/// * `repeats` - Number of times to repeat each element
/// * `axis` - Axis along which to repeat (None means flatten first)
///
/// # Returns
/// * `Ok(Array)` - Repeated array
/// * `Err(ManipulationError)` if repeat fails
pub fn repeat(array: &Array, repeats: usize, axis: Option<usize>) -> Result<Array, ManipulationError> {
    if let Some(ax) = axis {
        if ax >= array.ndim() {
            return Err(ManipulationError::InvalidAxis);
        }
        repeat_along_axis(array, repeats, ax)
    } else {
        repeat_flat(array, repeats)
    }
}

/// Repeat along specific axis
fn repeat_along_axis(
    array: &Array,
    repeats: usize,
    axis: usize,
) -> Result<Array, ManipulationError> {
    let shape = array.shape();
    let mut new_shape = shape.to_vec();
    new_shape[axis] *= repeats as i64;
    
    let dtype = array.dtype().clone();
    let mut output = Array::new(new_shape, dtype)?;
    
    let itemsize = array.itemsize();
    let src_strides = array.strides().to_vec();
    let dst_strides = output.strides().to_vec();
    
    // Iterate through source and repeat each element
    let mut src_coords = vec![0; shape.len()];
    let mut dst_coords = vec![0; shape.len()];
    
    for flat_idx in 0..array.size() {
        index_to_coords(flat_idx, shape, &mut src_coords);
        
        let src_offset = coords_to_offset(&src_coords, src_strides.as_slice());
        
        // Repeat this element 'repeats' times
        for r in 0..repeats {
            dst_coords.copy_from_slice(&src_coords);
            dst_coords[axis] = src_coords[axis] * repeats as i64 + r as i64;
            
            let dst_offset = coords_to_offset(&dst_coords, dst_strides.as_slice());
            
            unsafe {
                let src_ptr = array.data_ptr().add(src_offset);
                let dst_ptr = output.data_ptr_mut().add(dst_offset);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize);
            }
        }
    }
    
    Ok(output)
}

/// Repeat flattened array
fn repeat_flat(array: &Array, repeats: usize) -> Result<Array, ManipulationError> {
    let size = array.size();
    let new_size = size * repeats;
    
    let dtype = array.dtype().clone();
    let mut output = Array::new(vec![new_size as i64], dtype)?;
    
    let itemsize = array.itemsize();
    
    unsafe {
        let src_ptr = array.data_ptr();
        let dst_ptr = output.data_ptr_mut();
        
        for i in 0..size {
            for r in 0..repeats {
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(i * itemsize),
                    dst_ptr.add((i * repeats + r) * itemsize),
                    itemsize,
                );
            }
        }
    }
    
    Ok(output)
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

