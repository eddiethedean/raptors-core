//! Histogram operations

use crate::array::{Array, ArrayError};
use crate::types::DType;

use super::StatisticsError;

/// Compute histogram of array
///
/// # Arguments
/// * `array` - Array to compute histogram for
/// * `bins` - Number of bins
///
/// # Returns
/// * `Ok((Array, Array))` - Tuple of (counts, bin_edges)
/// * `Err(StatisticsError)` if computation fails
pub fn histogram(array: &Array, bins: usize) -> Result<(Array, Array), StatisticsError> {
    if bins == 0 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    match array.dtype().type_() {
        crate::types::NpyType::Double => histogram_double(array, bins),
        crate::types::NpyType::Float => histogram_float(array, bins),
        _ => Err(StatisticsError::UnsupportedType),
    }
}

/// Histogram for double
fn histogram_double(array: &Array, bins: usize) -> Result<(Array, Array), StatisticsError> {
    let size = array.size();
    if size == 0 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    unsafe {
        let ptr = array.data_ptr() as *const f64;
        
        // Find min and max
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        
        for i in 0..size {
            let val = *ptr.add(i);
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }
        
        // Create bins
        let bin_width = (max_val - min_val) / bins as f64;
        let mut counts = vec![0; bins];
        
        // Count values in each bin
        for i in 0..size {
            let val = *ptr.add(i);
            let bin_idx = (((val - min_val) / bin_width) as usize).min(bins - 1);
            counts[bin_idx] += 1;
        }
        
        // Create output arrays
        let counts_dtype = DType::new(crate::types::NpyType::Long);
        let mut counts_array = Array::new(vec![bins as i64], counts_dtype)?;
        let counts_ptr = counts_array.data_ptr_mut() as *mut i64;
        for (i, &count) in counts.iter().enumerate() {
            *counts_ptr.add(i) = count as i64;
        }
        
        let edges_dtype = DType::new(crate::types::NpyType::Double);
        let mut edges_array = Array::new(vec![(bins + 1) as i64], edges_dtype)?;
        let edges_ptr = edges_array.data_ptr_mut() as *mut f64;
        for i in 0..=bins {
            *edges_ptr.add(i) = min_val + i as f64 * bin_width;
        }
        
        Ok((counts_array, edges_array))
    }
}

/// Histogram for float
fn histogram_float(array: &Array, bins: usize) -> Result<(Array, Array), StatisticsError> {
    let size = array.size();
    if size == 0 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    unsafe {
        let ptr = array.data_ptr() as *const f32;
        
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        
        for i in 0..size {
            let val = *ptr.add(i);
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }
        
        let bin_width = (max_val - min_val) / bins as f32;
        let mut counts = vec![0; bins];
        
        for i in 0..size {
            let val = *ptr.add(i);
            let bin_idx = (((val - min_val) / bin_width) as usize).min(bins - 1);
            counts[bin_idx] += 1;
        }
        
        let counts_dtype = DType::new(crate::types::NpyType::Long);
        let mut counts_array = Array::new(vec![bins as i64], counts_dtype)?;
        let counts_ptr = counts_array.data_ptr_mut() as *mut i64;
        for (i, &count) in counts.iter().enumerate() {
            *counts_ptr.add(i) = count as i64;
        }
        
        let edges_dtype = DType::new(crate::types::NpyType::Float);
        let mut edges_array = Array::new(vec![(bins + 1) as i64], edges_dtype)?;
        let edges_ptr = edges_array.data_ptr_mut() as *mut f32;
        for i in 0..=bins {
            *edges_ptr.add(i) = min_val + i as f32 * bin_width;
        }
        
        Ok((counts_array, edges_array))
    }
}

