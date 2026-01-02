//! String formatting operations

use crate::array::Array;
use crate::types::DType;

use super::StringError;

/// Convert strings to uppercase
///
/// # Arguments
/// * `array` - String array
///
/// # Returns
/// * `Ok(Array)` - Uppercase string array
/// * `Err(StringError)` if conversion fails
pub fn str_upper(array: &Array) -> Result<Array, StringError> {
    if !super::is_string_array(array) {
        return Err(StringError::TypeMismatch);
    }
    
    let size = array.size();
    let dtype = array.dtype().clone();
    let mut result = Array::new(array.shape().to_vec(), dtype)?;
    
    for i in 0..size {
        let s = super::get_string(array, i)?;
        let upper = s.to_uppercase();
        
        unsafe {
            let ptr = result.data_ptr_mut().add(i * array.itemsize());
            let bytes = upper.as_bytes();
            let copy_len = bytes.len().min(array.itemsize());
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, copy_len);
            if copy_len < array.itemsize() {
                std::ptr::write_bytes(ptr.add(copy_len), 0, array.itemsize() - copy_len);
            }
        }
    }
    
    Ok(result)
}

/// Convert strings to lowercase
pub fn str_lower(array: &Array) -> Result<Array, StringError> {
    if !super::is_string_array(array) {
        return Err(StringError::TypeMismatch);
    }
    
    let size = array.size();
    let dtype = array.dtype().clone();
    let mut result = Array::new(array.shape().to_vec(), dtype)?;
    
    for i in 0..size {
        let s = super::get_string(array, i)?;
        let lower = s.to_lowercase();
        
        unsafe {
            let ptr = result.data_ptr_mut().add(i * array.itemsize());
            let bytes = lower.as_bytes();
            let copy_len = bytes.len().min(array.itemsize());
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, copy_len);
            if copy_len < array.itemsize() {
                std::ptr::write_bytes(ptr.add(copy_len), 0, array.itemsize() - copy_len);
            }
        }
    }
    
    Ok(result)
}

/// Convert strings to title case (first letter uppercase, rest lowercase)
pub fn str_title(array: &Array) -> Result<Array, StringError> {
    if !super::is_string_array(array) {
        return Err(StringError::TypeMismatch);
    }
    
    let size = array.size();
    let dtype = array.dtype().clone();
    let mut result = Array::new(array.shape().to_vec(), dtype)?;
    
    for i in 0..size {
        let s = super::get_string(array, i)?;
        let title = s
            .split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str().to_lowercase().as_str(),
                }
            })
            .collect::<Vec<String>>()
            .join(" ");
        
        unsafe {
            let ptr = result.data_ptr_mut().add(i * array.itemsize());
            let bytes = title.as_bytes();
            let copy_len = bytes.len().min(array.itemsize());
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, copy_len);
            if copy_len < array.itemsize() {
                std::ptr::write_bytes(ptr.add(copy_len), 0, array.itemsize() - copy_len);
            }
        }
    }
    
    Ok(result)
}

/// Pad strings to specified width
///
/// # Arguments
/// * `array` - String array
/// * `width` - Target width
/// * `pad_char` - Padding character
///
/// # Returns
/// * `Ok(Array)` - Padded string array
/// * `Err(StringError)` if padding fails
pub fn str_pad(array: &Array, width: usize, pad_char: char) -> Result<Array, StringError> {
    if !super::is_string_array(array) {
        return Err(StringError::TypeMismatch);
    }
    
    let size = array.size();
    
    // Calculate max length needed (original max or width, whichever is larger)
    let mut max_len = width;
    for i in 0..size {
        if let Ok(s) = super::get_string(array, i) {
            max_len = max_len.max(s.len());
        }
    }
    max_len = max_len.max(1);
    
    // Create result array with appropriate itemsize
    let result_dtype = DType::string_with_itemsize(max_len);
    let mut result = Array::new(array.shape().to_vec(), result_dtype)?;
    
    for i in 0..size {
        let s = super::get_string(array, i)?;
        let padded = if s.len() < width {
            format!("{}{}", s, pad_char.to_string().repeat(width - s.len()))
        } else {
            s
        };
        
        unsafe {
            let itemsize = result.itemsize();
            let ptr = result.data_ptr_mut().add(i * itemsize);
            let bytes = padded.as_bytes();
            let copy_len = bytes.len().min(itemsize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, copy_len);
            if copy_len < itemsize {
                std::ptr::write_bytes(ptr.add(copy_len), 0, itemsize - copy_len);
            }
        }
    }
    
    Ok(result)
}

