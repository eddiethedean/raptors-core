//! String array representation and operations

use crate::array::{Array, ArrayError};
use crate::types::{DType, NpyType};

/// String operation error
#[derive(Debug, Clone)]
pub enum StringError {
    /// Array error
    ArrayError(ArrayError),
    /// Type mismatch (not a string array)
    TypeMismatch,
    /// Invalid string data
    InvalidStringData,
    /// Encoding error
    EncodingError(String),
}

impl std::fmt::Display for StringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StringError::ArrayError(e) => write!(f, "Array error: {}", e),
            StringError::TypeMismatch => write!(f, "Type mismatch - expected string array"),
            StringError::InvalidStringData => write!(f, "Invalid string data"),
            StringError::EncodingError(msg) => write!(f, "Encoding error: {}", msg),
        }
    }
}

impl std::error::Error for StringError {}

impl From<ArrayError> for StringError {
    fn from(err: ArrayError) -> Self {
        StringError::ArrayError(err)
    }
}

/// Create a string array from vector of strings
///
/// For simplicity, this uses fixed-width strings (padded to max length)
/// or stores strings as variable-length with offset array
///
/// # Arguments
/// * `data` - Vector of strings
/// * `shape` - Shape of the array
///
/// # Returns
/// * `Ok(Array)` - String array
/// * `Err(StringError)` if creation fails
pub fn create_string_array(data: Vec<String>, shape: Vec<i64>) -> Result<Array, StringError> {
    // Simplified implementation using fixed-width strings
    // Find maximum string length (minimum 1 to avoid zero itemsize)
    let max_len = data.iter().map(|s| s.len()).max().unwrap_or(0).max(1);
    
    // Create dtype with appropriate itemsize for fixed-width strings
    let dtype = DType::string_with_itemsize(max_len);
    let total_size = shape.iter().product::<i64>() as usize;
    
    // Create array with correct itemsize
    let mut array = Array::new(shape, dtype)?;
    
    // Copy string data into array using proper itemsize
    unsafe {
        let ptr = array.data_ptr_mut();
        let itemsize = array.itemsize();
        for (i, s) in data.iter().enumerate().take(total_size) {
            let bytes = s.as_bytes();
            let offset = i * itemsize;
            // Copy bytes (truncate if too long)
            let copy_len = bytes.len().min(itemsize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset), copy_len);
            // Zero-fill remaining bytes
            if copy_len < itemsize {
                std::ptr::write_bytes(ptr.add(offset + copy_len), 0, itemsize - copy_len);
            }
        }
    }
    
    Ok(array)
}

/// Check if array is a string array
pub fn is_string_array(array: &Array) -> bool {
    matches!(array.dtype().type_(), NpyType::String | NpyType::Unicode)
}

/// Get string from array at index (simplified - for fixed-width strings)
pub fn get_string(array: &Array, index: usize) -> Result<String, StringError> {
    if !is_string_array(array) {
        return Err(StringError::TypeMismatch);
    }
    
    let itemsize = array.itemsize();
    unsafe {
        let ptr = array.data_ptr().add(index * itemsize);
        // Read null-terminated or fixed-width string
        let mut bytes = Vec::with_capacity(itemsize);
        for i in 0..itemsize {
            let byte = *ptr.add(i);
            if byte == 0 {
                break;
            }
            bytes.push(byte);
        }
        String::from_utf8(bytes).map_err(|e| StringError::EncodingError(e.to_string()))
    }
}

