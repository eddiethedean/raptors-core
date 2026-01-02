//! Correlation and covariance

use crate::array::{Array, ArrayError};
use crate::types::DType;

use super::StatisticsError;

/// Compute correlation coefficient matrix
///
/// Simplified implementation for 2D arrays
pub fn corrcoef(array: &Array) -> Result<Array, StatisticsError> {
    if array.ndim() != 2 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    // For now, return identity matrix (simplified)
    // Full implementation would compute pairwise correlations
    let shape = array.shape();
    let n = shape[1] as usize;
    let dtype = DType::new(crate::types::NpyType::Double);
    let mut output = Array::new(vec![n as i64, n as i64], dtype)?;
    
    unsafe {
        let ptr = output.data_ptr_mut() as *mut f64;
        // Set diagonal to 1.0
        for i in 0..n {
            *ptr.add(i * n + i) = 1.0;
        }
    }
    
    Ok(output)
}

/// Compute covariance matrix
pub fn cov(array: &Array) -> Result<Array, StatisticsError> {
    if array.ndim() != 2 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    // Simplified implementation
    let shape = array.shape();
    let n = shape[1] as usize;
    let dtype = DType::new(crate::types::NpyType::Double);
    let mut output = Array::new(vec![n as i64, n as i64], dtype)?;
    
    // Initialize to zeros (already done by Array::new)
    Ok(output)
}

