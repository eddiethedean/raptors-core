//! String concatenation operations

use crate::array::{Array, ArrayError};
use crate::types::DType;

use super::StringError;

/// Concatenate two string arrays element-wise
///
/// # Arguments
/// * `a1` - First string array
/// * `a2` - Second string array
///
/// # Returns
/// * `Ok(Array)` - Concatenated string array
/// * `Err(StringError)` if concatenation fails
pub fn str_concat(a1: &Array, a2: &Array) -> Result<Array, StringError> {
    if !super::is_string_array(a1) || !super::is_string_array(a2) {
        return Err(StringError::TypeMismatch);
    }
    
    // For now, simplified implementation
    // Full implementation would handle broadcasting and variable-length strings
    if a1.shape() != a2.shape() {
        return Err(StringError::ArrayError(ArrayError::InvalidShape));
    }
    
    let size = a1.size();
    
    // For concatenation, we need to calculate the max length needed
    let mut max_result_len = 0;
    for i in 0..size {
        let s1_len = super::get_string(a1, i).map(|s| s.len()).unwrap_or(0);
        let s2_len = super::get_string(a2, i).map(|s| s.len()).unwrap_or(0);
        max_result_len = max_result_len.max(s1_len + s2_len);
    }
    max_result_len = max_result_len.max(1); // At least 1
    
    // Create result array with appropriate itemsize
    let result_dtype = DType::string_with_itemsize(max_result_len);
    let mut result = Array::new(a1.shape().to_vec(), result_dtype)?;
    
    // Concatenate strings element-wise
    for i in 0..size {
        let s1 = super::get_string(a1, i)?;
        let s2 = super::get_string(a2, i)?;
        let concatenated = s1 + &s2;
        
        // Write back (simplified - would need proper offset handling)
        unsafe {
            let itemsize = result.itemsize();
            let ptr = result.data_ptr_mut().add(i * itemsize);
            let bytes = concatenated.as_bytes();
            let copy_len = bytes.len().min(itemsize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, copy_len);
            if copy_len < itemsize {
                std::ptr::write_bytes(ptr.add(copy_len), 0, itemsize - copy_len);
            }
        }
    }
    
    Ok(result)
}

/// Join strings in array with separator
///
/// # Arguments
/// * `array` - String array to join
/// * `separator` - Separator string
///
/// # Returns
/// * `Ok(String)` - Joined string
/// * `Err(StringError)` if join fails
pub fn str_join(array: &Array, separator: &str) -> Result<String, StringError> {
    if !super::is_string_array(array) {
        return Err(StringError::TypeMismatch);
    }
    
    let size = array.size();
    let mut parts = Vec::with_capacity(size);
    
    for i in 0..size {
        let s = super::get_string(array, i)?;
        parts.push(s);
    }
    
    Ok(parts.join(separator))
}

