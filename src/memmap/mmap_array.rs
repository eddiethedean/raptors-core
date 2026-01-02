//! Memory-mapped array structure

use crate::array::{Array, ArrayError};
use crate::types::DType;
use std::path::Path;

/// Memory mapping mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapMode {
    /// Read-only mapping
    ReadOnly,
    /// Read-write mapping
    ReadWrite,
    /// Copy-on-write mapping
    CopyOnWrite,
}

/// Memory-mapped array error
#[derive(Debug, Clone)]
pub enum MemMapError {
    /// Array error
    ArrayError(ArrayError),
    /// File I/O error
    IoError(String),
    /// Memory mapping failed
    MappingFailed(String),
    /// File not found
    FileNotFound,
}

impl std::fmt::Display for MemMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemMapError::ArrayError(e) => write!(f, "Array error: {}", e),
            MemMapError::IoError(msg) => write!(f, "I/O error: {}", msg),
            MemMapError::MappingFailed(msg) => write!(f, "Memory mapping failed: {}", msg),
            MemMapError::FileNotFound => write!(f, "File not found"),
        }
    }
}

impl std::error::Error for MemMapError {}

impl From<ArrayError> for MemMapError {
    fn from(err: ArrayError) -> Self {
        MemMapError::ArrayError(err)
    }
}

/// Memory-mapped array
///
/// Wraps an array backed by a memory-mapped file
pub struct MemMapArray {
    /// The underlying array
    array: Array,
    /// File path (for reference)
    file_path: std::path::PathBuf,
    /// Mapping mode
    mode: MapMode,
}

impl MemMapArray {
    /// Create a new memory-mapped array
    ///
    /// # Arguments
    /// * `file_path` - Path to file
    /// * `dtype` - Data type
    /// * `shape` - Shape of array
    /// * `mode` - Mapping mode
    ///
    /// # Returns
    /// * `Ok(MemMapArray)` if successful
    /// * `Err(MemMapError)` if creation fails
    pub fn new(
        file_path: &Path,
        dtype: DType,
        shape: Vec<i64>,
        mode: MapMode,
    ) -> Result<Self, MemMapError> {
        // For now, simplified implementation that reads file into memory
        // Full implementation would use memmap2 crate for actual memory mapping
        
        // Check if file exists
        if !file_path.exists() {
            return Err(MemMapError::FileNotFound);
        }
        
        // Read file data (simplified - full impl would memory-map)
        let data = std::fs::read(file_path)
            .map_err(|e| MemMapError::IoError(e.to_string()))?;
        
        let itemsize = dtype.itemsize();
        let total_elements: usize = shape.iter().product::<i64>() as usize;
        let required_size = total_elements * itemsize;
        
        if data.len() < required_size {
            return Err(MemMapError::IoError("File too small".to_string()));
        }
        
        // Create array and copy data
        let mut array = Array::new(shape, dtype)?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), array.data_ptr_mut(), required_size);
        }
        
        Ok(MemMapArray {
            array,
            file_path: file_path.to_path_buf(),
            mode,
        })
    }
    
    /// Get reference to underlying array
    pub fn array(&self) -> &Array {
        &self.array
    }
    
    /// Get mutable reference to underlying array
    pub fn array_mut(&mut self) -> &mut Array {
        &mut self.array
    }
    
    /// Get file path
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }
    
    /// Get mapping mode
    pub fn mode(&self) -> MapMode {
        self.mode
    }
    
    /// Flush changes to file (for write modes)
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err(MemMapError)` if flush fails
    pub fn flush(&self) -> Result<(), MemMapError> {
        // For simplified implementation, write data to file
        if self.mode == MapMode::ReadOnly {
            return Ok(()); // No-op for read-only
        }
        
        let data_size = self.array.size() * self.array.itemsize();
        unsafe {
            let data = std::slice::from_raw_parts(self.array.data_ptr(), data_size);
            std::fs::write(&self.file_path, data)
                .map_err(|e| MemMapError::IoError(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Sync changes to file
    ///
    /// Similar to flush, but ensures data is written to disk
    pub fn sync(&self) -> Result<(), MemMapError> {
        self.flush()
    }
}

