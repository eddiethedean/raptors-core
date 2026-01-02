//! Memory-mapped array I/O operations

use crate::array::Array;
use crate::types::DType;
use std::path::Path;

use super::{MemMapArray, MemMapError};

/// Save array to memory-mapped file
///
/// # Arguments
/// * `array` - Array to save
/// * `file_path` - Path to save file
///
/// # Returns
/// * `Ok(MemMapArray)` - Memory-mapped array
/// * `Err(MemMapError)` if save fails
pub fn save_memmap(array: &Array, file_path: &Path) -> Result<MemMapArray, MemMapError> {
    // Save array data to file
    let data_size = array.size() * array.itemsize();
    unsafe {
        let data = std::slice::from_raw_parts(array.data_ptr(), data_size);
        std::fs::write(file_path, data)
            .map_err(|e| MemMapError::IoError(e.to_string()))?;
    }
    
    // Create memory-mapped array from saved file
    MemMapArray::new(file_path, array.dtype().clone(), array.shape().to_vec(), super::MapMode::ReadWrite)
}

/// Load array from memory-mapped file
///
/// # Arguments
/// * `file_path` - Path to file
/// * `dtype` - Expected data type
/// * `shape` - Expected shape
///
/// # Returns
/// * `Ok(MemMapArray)` - Memory-mapped array
/// * `Err(MemMapError)` if load fails
pub fn load_memmap(file_path: &Path, dtype: DType, shape: Vec<i64>) -> Result<MemMapArray, MemMapError> {
    MemMapArray::new(file_path, dtype, shape, super::MapMode::ReadOnly)
}

