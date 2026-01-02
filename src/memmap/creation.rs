//! Memory-mapped array creation functions

use crate::types::DType;
use std::path::Path;

use super::{MemMapArray, MapMode, MemMapError};

/// Create memory-mapped array from file
///
/// # Arguments
/// * `file_path` - Path to file
/// * `dtype` - Data type
/// * `shape` - Shape of array
///
/// # Returns
/// * `Ok(MemMapArray)` - Memory-mapped array
/// * `Err(MemMapError)` if creation fails
pub fn memmap_array(file_path: &Path, dtype: DType, shape: Vec<i64>) -> Result<MemMapArray, MemMapError> {
    MemMapArray::new(file_path, dtype, shape, MapMode::ReadOnly)
}

/// Create memory-mapped array for writing
pub fn memmap_array_writable(file_path: &Path, dtype: DType, shape: Vec<i64>) -> Result<MemMapArray, MemMapError> {
    // For writable array, file should exist or we should create it
    if !file_path.exists() {
        // Create file with appropriate size
        let itemsize = dtype.itemsize();
        let total_elements: usize = shape.iter().product::<i64>() as usize;
        let file_size = total_elements * itemsize;
        
        // Create empty file
        std::fs::File::create(file_path)
            .map_err(|e| MemMapError::IoError(e.to_string()))?;
        
        // Extend file to required size
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(file_path)
            .map_err(|e| MemMapError::IoError(e.to_string()))?;
        file.set_len(file_size as u64)
            .map_err(|e| MemMapError::IoError(e.to_string()))?;
    }
    
    MemMapArray::new(file_path, dtype, shape, MapMode::ReadWrite)
}

