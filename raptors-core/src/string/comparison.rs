//! String comparison operations

use crate::array::{Array, ArrayError};
use crate::types::DType;

use super::StringError;

/// Element-wise string equality comparison
///
/// # Arguments
/// * `a1` - First string array
/// * `a2` - Second string array
///
/// # Returns
/// * `Ok(Array)` - Boolean array with comparison results
/// * `Err(StringError)` if comparison fails
pub fn str_equal(a1: &Array, a2: &Array) -> Result<Array, StringError> {
    if !super::is_string_array(a1) || !super::is_string_array(a2) {
        return Err(StringError::TypeMismatch);
    }
    
    if a1.shape() != a2.shape() {
        return Err(StringError::ArrayError(ArrayError::InvalidShape));
    }
    
    let size = a1.size();
    let bool_dtype = DType::new(crate::types::NpyType::Bool);
    let mut result = Array::new(a1.shape().to_vec(), bool_dtype)?;
    
    unsafe {
        let result_ptr = result.data_ptr_mut() as *mut bool;
        for i in 0..size {
            let s1 = super::get_string(a1, i)?;
            let s2 = super::get_string(a2, i)?;
            *result_ptr.add(i) = s1 == s2;
        }
    }
    
    Ok(result)
}

/// Element-wise string less-than comparison (lexicographic)
///
/// # Arguments
/// * `a1` - First string array
/// * `a2` - Second string array
///
/// # Returns
/// * `Ok(Array)` - Boolean array with comparison results
/// * `Err(StringError)` if comparison fails
pub fn str_less(a1: &Array, a2: &Array) -> Result<Array, StringError> {
    if !super::is_string_array(a1) || !super::is_string_array(a2) {
        return Err(StringError::TypeMismatch);
    }
    
    if a1.shape() != a2.shape() {
        return Err(StringError::ArrayError(ArrayError::InvalidShape));
    }
    
    let size = a1.size();
    let bool_dtype = DType::new(crate::types::NpyType::Bool);
    let mut result = Array::new(a1.shape().to_vec(), bool_dtype)?;
    
    unsafe {
        let result_ptr = result.data_ptr_mut() as *mut bool;
        for i in 0..size {
            let s1 = super::get_string(a1, i)?;
            let s2 = super::get_string(a2, i)?;
            *result_ptr.add(i) = s1 < s2;
        }
    }
    
    Ok(result)
}

/// Element-wise string greater-than comparison
pub fn str_greater(a1: &Array, a2: &Array) -> Result<Array, StringError> {
    // a > b is equivalent to b < a
    str_less(a2, a1)
}

/// Case-insensitive string equality
pub fn str_equal_case_insensitive(a1: &Array, a2: &Array) -> Result<Array, StringError> {
    if !super::is_string_array(a1) || !super::is_string_array(a2) {
        return Err(StringError::TypeMismatch);
    }
    
    if a1.shape() != a2.shape() {
        return Err(StringError::ArrayError(ArrayError::InvalidShape));
    }
    
    let size = a1.size();
    let bool_dtype = DType::new(crate::types::NpyType::Bool);
    let mut result = Array::new(a1.shape().to_vec(), bool_dtype)?;
    
    unsafe {
        let result_ptr = result.data_ptr_mut() as *mut bool;
        for i in 0..size {
            let s1 = super::get_string(a1, i)?;
            let s2 = super::get_string(a2, i)?;
            *result_ptr.add(i) = s1.eq_ignore_ascii_case(&s2);
        }
    }
    
    Ok(result)
}

