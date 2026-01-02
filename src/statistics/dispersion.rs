//! Dispersion measures (std, var)

use crate::array::{Array, ArrayError};
use crate::ufunc::mean_along_axis;

use super::StatisticsError;

/// Compute standard deviation
///
/// # Arguments
/// * `array` - Array to compute std for
/// * `axis` - Axis along which to compute (None means all elements)
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample)
///
/// # Returns
/// * `Ok(Array)` - Standard deviation value(s)
/// * `Err(StatisticsError)` if computation fails
pub fn std(array: &Array, axis: Option<usize>, ddof: usize) -> Result<Array, StatisticsError> {
    let mut var_result = var(array, axis, ddof)?;
    
    // Take square root of variance
    match var_result.dtype().type_() {
        crate::types::NpyType::Double => {
            unsafe {
                let ptr = var_result.data_ptr_mut() as *mut f64;
                let size = var_result.size();
                for i in 0..size {
                    *ptr.add(i) = (*ptr.add(i)).sqrt();
                }
            }
        }
        crate::types::NpyType::Float => {
            unsafe {
                let ptr = var_result.data_ptr_mut() as *mut f32;
                let size = var_result.size();
                for i in 0..size {
                    *ptr.add(i) = (*ptr.add(i)).sqrt();
                }
            }
        }
        _ => return Err(StatisticsError::UnsupportedType),
    }
    
    Ok(var_result)
}

/// Compute variance
///
/// # Arguments
/// * `array` - Array to compute variance for
/// * `axis` - Axis along which to compute (None means all elements)
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
/// * `Ok(Array)` - Variance value(s)
/// * `Err(StatisticsError)` if computation fails
pub fn var(array: &Array, axis: Option<usize>, ddof: usize) -> Result<Array, StatisticsError> {
    let mean_result = mean_along_axis(array, axis)
        .map_err(|_| StatisticsError::ArrayError(ArrayError::InvalidShape))?;
    let size = array.size();
    let n = size - ddof;
    
    if n == 0 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    match array.dtype().type_() {
        crate::types::NpyType::Double => var_double(array, &mean_result, n),
        crate::types::NpyType::Float => var_float(array, &mean_result, n),
        _ => Err(StatisticsError::UnsupportedType),
    }
}

/// Variance for double
fn var_double(array: &Array, mean: &Array, n: usize) -> Result<Array, StatisticsError> {
    let size = array.size();
    let dtype = array.dtype().clone();
    let mut output = Array::new(vec![1], dtype)?;
    
    unsafe {
        let mean_val = *(mean.data_ptr() as *const f64);
        let arr_ptr = array.data_ptr() as *const f64;
        let mut sum_sq_diff = 0.0;
        
        for i in 0..size {
            let diff = *arr_ptr.add(i) - mean_val;
            sum_sq_diff += diff * diff;
        }
        
        let var_val = sum_sq_diff / n as f64;
        let out_ptr = output.data_ptr_mut() as *mut f64;
        *out_ptr = var_val;
    }
    
    Ok(output)
}

/// Variance for float
fn var_float(array: &Array, mean: &Array, n: usize) -> Result<Array, StatisticsError> {
    let size = array.size();
    let dtype = array.dtype().clone();
    let mut output = Array::new(vec![1], dtype)?;
    
    unsafe {
        let mean_val = *(mean.data_ptr() as *const f32);
        let arr_ptr = array.data_ptr() as *const f32;
        let mut sum_sq_diff = 0.0f32;
        
        for i in 0..size {
            let diff = *arr_ptr.add(i) - mean_val;
            sum_sq_diff += diff * diff;
        }
        
        let var_val = sum_sq_diff / n as f32;
        let out_ptr = output.data_ptr_mut() as *mut f32;
        *out_ptr = var_val;
    }
    
    Ok(output)
}

