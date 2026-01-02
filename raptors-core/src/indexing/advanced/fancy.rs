//! Fancy indexing implementation
//!
//! This module provides fancy indexing (integer array indexing),
//! equivalent to NumPy's fancy indexing functionality

use crate::array::Array;
use crate::types::NpyType;
use crate::indexing::IndexError;

/// Fancy index an array using an index array
///
/// Returns a new array with elements selected by the index array.
/// The index array must be 1D and contain integer indices.
pub fn fancy_index_array(array: &Array, indices: &Array) -> Result<Array, IndexError> {
    // Validate index array
    if indices.ndim() != 1 {
        return Err(IndexError::DimensionMismatch);
    }
    
    // Index array must be integer type
    match indices.dtype().type_() {
        NpyType::Int => {}
        _ => return Err(IndexError::InvalidIndex),
    }
    
    let array_shape = array.shape();
    let index_size = indices.size();
    let array_size = array.size() as i64;
    
    // Handle negative indices and validate bounds
    // For 1D fancy indexing, check against total array size
    let normalized_indices: Vec<i64> = unsafe {
        let index_ptr = indices.data_ptr() as *const i32;
        (0..index_size)
            .map(|i| {
                let mut idx = *index_ptr.add(i) as i64;
                // Normalize negative indices
                if idx < 0 {
                    idx += array_size;
                }
                // Validate bounds against total size (for 1D indexing)
                if idx < 0 || idx >= array_size {
                    return Err(IndexError::OutOfBounds);
                }
                Ok(idx)
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    
    // Create output array with shape from indices
    let output_shape = vec![index_size as i64];
    let output_dtype = array.dtype().clone();
    let mut output = Array::new(output_shape, output_dtype)?;
    
    // Copy selected elements
    let itemsize = array.itemsize();
    unsafe {
        let array_data = array.data_ptr();
        let output_data = output.data_ptr_mut();
        let array_strides = array.strides();
        
        for (i, &idx) in normalized_indices.iter().enumerate() {
            // For 1D indexing, calculate offset using flat index
            // This works for both 1D and multi-dimensional arrays when treating as 1D
            let src_offset = if array.ndim() == 1 {
                (idx * array_strides[0]) as usize
            } else {
                // For multi-dimensional, calculate flat offset using row-major indexing
                // For a flat index, convert to multi-dimensional coordinates
                let mut offset = 0i64;
                let mut remaining = idx;
                // Iterate from first dimension to last (row-major order)
                for (dim_idx, (dim_size, &stride)) in array_shape.iter().zip(array_strides.iter()).enumerate() {
                    let dim_size_i64 = *dim_size;
                    // Calculate coordinate for this dimension
                    let coord = if dim_idx == array_shape.len() - 1 {
                        // Last dimension: remaining is the coordinate
                        remaining
                    } else {
                        // Calculate how many "rows" we've passed
                        let product_after: i64 = array_shape[(dim_idx + 1)..].iter().product::<i64>();
                        remaining / product_after
                    };
                    offset += coord * stride;
                    // Update remaining for next dimension
                    if dim_idx < array_shape.len() - 1 {
                        let product_after: i64 = array_shape[(dim_idx + 1)..].iter().product::<i64>();
                        remaining %= product_after;
                    }
                }
                offset as usize
            };
            let dst_offset = i * itemsize;
            
            std::ptr::copy_nonoverlapping(
                array_data.add(src_offset),
                output_data.add(dst_offset),
                itemsize,
            );
        }
    }
    
    Ok(output)
}

/// Fancy index a multi-dimensional array
///
/// For now, simplified implementation - treats array as 1D for indexing
/// Full implementation would handle multi-dimensional fancy indexing
pub fn fancy_index_ndarray(array: &Array, indices: &Array, axis: usize) -> Result<Array, IndexError> {
    if axis >= array.ndim() {
        return Err(IndexError::DimensionMismatch);
    }
    
    // For now, simplified: just use 1D fancy indexing on first axis
    // Full implementation would handle arbitrary axis and multi-dimensional indexing
    fancy_index_array(array, indices)
}

