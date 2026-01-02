//! Set operations on arrays

use crate::array::{Array, ArrayError};
use crate::types::DType;

use super::ManipulationError;

/// Union of two 1D arrays
///
/// Returns sorted unique values that are in either array
pub fn union1d(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    if a1.ndim() != 1 || a2.ndim() != 1 {
        return Err(ManipulationError::ArrayError(ArrayError::InvalidShape));
    }
    
    if a1.dtype().type_() != a2.dtype().type_() {
        return Err(ManipulationError::ArrayError(ArrayError::TypeMismatch));
    }
    
    match a1.dtype().type_() {
        crate::types::NpyType::Double => union1d_double(a1, a2),
        crate::types::NpyType::Float => union1d_float(a1, a2),
        crate::types::NpyType::Int => union1d_int(a1, a2),
        crate::types::NpyType::Long => union1d_long(a1, a2),
        _ => Err(ManipulationError::ArrayError(ArrayError::TypeMismatch)),
    }
}

/// Intersection of two 1D arrays
pub fn intersect1d(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    if a1.ndim() != 1 || a2.ndim() != 1 {
        return Err(ManipulationError::ArrayError(ArrayError::InvalidShape));
    }
    
    if a1.dtype().type_() != a2.dtype().type_() {
        return Err(ManipulationError::ArrayError(ArrayError::TypeMismatch));
    }
    
    match a1.dtype().type_() {
        crate::types::NpyType::Double => intersect1d_double(a1, a2),
        crate::types::NpyType::Float => intersect1d_float(a1, a2),
        crate::types::NpyType::Int => intersect1d_int(a1, a2),
        crate::types::NpyType::Long => intersect1d_long(a1, a2),
        _ => Err(ManipulationError::ArrayError(ArrayError::TypeMismatch)),
    }
}

/// Set difference (elements in a1 but not in a2)
pub fn setdiff1d(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    if a1.ndim() != 1 || a2.ndim() != 1 {
        return Err(ManipulationError::ArrayError(ArrayError::InvalidShape));
    }
    
    if a1.dtype().type_() != a2.dtype().type_() {
        return Err(ManipulationError::ArrayError(ArrayError::TypeMismatch));
    }
    
    match a1.dtype().type_() {
        crate::types::NpyType::Double => setdiff1d_double(a1, a2),
        crate::types::NpyType::Float => setdiff1d_float(a1, a2),
        crate::types::NpyType::Int => setdiff1d_int(a1, a2),
        crate::types::NpyType::Long => setdiff1d_long(a1, a2),
        _ => Err(ManipulationError::ArrayError(ArrayError::TypeMismatch)),
    }
}

/// Set exclusive or (elements in either array but not both)
pub fn setxor1d(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    if a1.ndim() != 1 || a2.ndim() != 1 {
        return Err(ManipulationError::ArrayError(ArrayError::InvalidShape));
    }
    
    if a1.dtype().type_() != a2.dtype().type_() {
        return Err(ManipulationError::ArrayError(ArrayError::TypeMismatch));
    }
    
    match a1.dtype().type_() {
        crate::types::NpyType::Double => setxor1d_double(a1, a2),
        crate::types::NpyType::Float => setxor1d_float(a1, a2),
        crate::types::NpyType::Int => setxor1d_int(a1, a2),
        crate::types::NpyType::Long => setxor1d_long(a1, a2),
        _ => Err(ManipulationError::ArrayError(ArrayError::TypeMismatch)),
    }
}

// Double implementations
fn union1d_double(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let mut combined = collect_double(a1, a2)?;
    combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    combined.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    array_from_vec_double(&combined, a1.dtype().clone())
}

fn intersect1d_double(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let v1 = collect_double(a1, a1)?;
    let v2 = collect_double(a2, a2)?;
    let mut result: Vec<f64> = Vec::new();
    
    for &val in &v1 {
        if v2.iter().any(|&x| (x - val).abs() < 1e-10) && !result.iter().any(|&x| (x - val).abs() < 1e-10) {
            result.push(val);
        }
    }
    
    result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    array_from_vec_double(&result, a1.dtype().clone())
}

fn setdiff1d_double(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let v1 = collect_double(a1, a1)?;
    let v2 = collect_double(a2, a2)?;
    let mut result: Vec<f64> = Vec::new();
    
    for &val in &v1 {
        if !v2.iter().any(|&x| (x - val).abs() < 1e-10) && !result.iter().any(|&x| (x - val).abs() < 1e-10) {
            result.push(val);
        }
    }
    
    result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    array_from_vec_double(&result, a1.dtype().clone())
}

fn setxor1d_double(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let diff1 = setdiff1d_double(a1, a2)?;
    let diff2 = setdiff1d_double(a2, a1)?;
    
    // Combine and deduplicate
    let mut combined = collect_double(&diff1, &diff2)?;
    combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    combined.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    array_from_vec_double(&combined, a1.dtype().clone())
}

fn collect_double(a1: &Array, a2: &Array) -> Result<Vec<f64>, ManipulationError> {
    let mut result = Vec::new();
    unsafe {
        let ptr1 = a1.data_ptr() as *const f64;
        for i in 0..a1.size() {
            result.push(*ptr1.add(i));
        }
        let ptr2 = a2.data_ptr() as *const f64;
        for i in 0..a2.size() {
            result.push(*ptr2.add(i));
        }
    }
    Ok(result)
}

fn array_from_vec_double(values: &[f64], dtype: DType) -> Result<Array, ManipulationError> {
    let mut array = Array::new(vec![values.len() as i64], dtype)?;
    unsafe {
        let ptr = array.data_ptr_mut() as *mut f64;
        for (i, &val) in values.iter().enumerate() {
            *ptr.add(i) = val;
        }
    }
    Ok(array)
}

// Similar implementations for other types (simplified)
fn union1d_float(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let mut combined = collect_float(a1, a2)?;
    combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    combined.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    array_from_vec_float(&combined, a1.dtype().clone())
}

fn intersect1d_float(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let v1 = collect_float(a1, a1)?;
    let v2 = collect_float(a2, a2)?;
    let mut result: Vec<f32> = v1.into_iter().filter(|&x| v2.iter().any(|&y| (y - x).abs() < 1e-6)).collect();
    result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    result.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    array_from_vec_float(&result, a1.dtype().clone())
}

fn setdiff1d_float(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let v1 = collect_float(a1, a1)?;
    let v2 = collect_float(a2, a2)?;
    let mut result: Vec<f32> = v1.into_iter().filter(|&x| !v2.iter().any(|&y| (y - x).abs() < 1e-6)).collect();
    result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    result.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    array_from_vec_float(&result, a1.dtype().clone())
}

fn setxor1d_float(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let diff1 = setdiff1d_float(a1, a2)?;
    let diff2 = setdiff1d_float(a2, a1)?;
    let mut combined = collect_float(&diff1, &diff2)?;
    combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    combined.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    array_from_vec_float(&combined, a1.dtype().clone())
}

fn collect_float(a1: &Array, a2: &Array) -> Result<Vec<f32>, ManipulationError> {
    let mut result = Vec::new();
    unsafe {
        let ptr1 = a1.data_ptr() as *const f32;
        for i in 0..a1.size() {
            result.push(*ptr1.add(i));
        }
        let ptr2 = a2.data_ptr() as *const f32;
        for i in 0..a2.size() {
            result.push(*ptr2.add(i));
        }
    }
    Ok(result)
}

fn array_from_vec_float(values: &[f32], dtype: DType) -> Result<Array, ManipulationError> {
    let mut array = Array::new(vec![values.len() as i64], dtype)?;
    unsafe {
        let ptr = array.data_ptr_mut() as *mut f32;
        for (i, &val) in values.iter().enumerate() {
            *ptr.add(i) = val;
        }
    }
    Ok(array)
}

// Int implementations
fn union1d_int(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let mut combined = collect_int(a1, a2)?;
    combined.sort();
    combined.dedup();
    array_from_vec_int(&combined, a1.dtype().clone())
}

fn intersect1d_int(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let v1 = collect_int(a1, a1)?;
    let v2 = collect_int(a2, a2)?;
    let mut result: Vec<i32> = v1.into_iter().filter(|&x| v2.contains(&x)).collect();
    result.sort();
    result.dedup();
    array_from_vec_int(&result, a1.dtype().clone())
}

fn setdiff1d_int(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let v1 = collect_int(a1, a1)?;
    let v2 = collect_int(a2, a2)?;
    let mut result: Vec<i32> = v1.into_iter().filter(|&x| !v2.contains(&x)).collect();
    result.sort();
    result.dedup();
    array_from_vec_int(&result, a1.dtype().clone())
}

fn setxor1d_int(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let diff1 = setdiff1d_int(a1, a2)?;
    let diff2 = setdiff1d_int(a2, a1)?;
    let mut combined = collect_int(&diff1, &diff2)?;
    combined.sort();
    combined.dedup();
    array_from_vec_int(&combined, a1.dtype().clone())
}

fn collect_int(a1: &Array, a2: &Array) -> Result<Vec<i32>, ManipulationError> {
    let mut result = Vec::new();
    unsafe {
        let ptr1 = a1.data_ptr() as *const i32;
        for i in 0..a1.size() {
            result.push(*ptr1.add(i));
        }
        let ptr2 = a2.data_ptr() as *const i32;
        for i in 0..a2.size() {
            result.push(*ptr2.add(i));
        }
    }
    Ok(result)
}

fn array_from_vec_int(values: &[i32], dtype: DType) -> Result<Array, ManipulationError> {
    let mut array = Array::new(vec![values.len() as i64], dtype)?;
    unsafe {
        let ptr = array.data_ptr_mut() as *mut i32;
        for (i, &val) in values.iter().enumerate() {
            *ptr.add(i) = val;
        }
    }
    Ok(array)
}

// Long implementations
fn union1d_long(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let mut combined = collect_long(a1, a2)?;
    combined.sort();
    combined.dedup();
    array_from_vec_long(&combined, a1.dtype().clone())
}

fn intersect1d_long(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let v1 = collect_long(a1, a1)?;
    let v2 = collect_long(a2, a2)?;
    let mut result: Vec<i64> = v1.into_iter().filter(|&x| v2.contains(&x)).collect();
    result.sort();
    result.dedup();
    array_from_vec_long(&result, a1.dtype().clone())
}

fn setdiff1d_long(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let v1 = collect_long(a1, a1)?;
    let v2 = collect_long(a2, a2)?;
    let mut result: Vec<i64> = v1.into_iter().filter(|&x| !v2.contains(&x)).collect();
    result.sort();
    result.dedup();
    array_from_vec_long(&result, a1.dtype().clone())
}

fn setxor1d_long(a1: &Array, a2: &Array) -> Result<Array, ManipulationError> {
    let diff1 = setdiff1d_long(a1, a2)?;
    let diff2 = setdiff1d_long(a2, a1)?;
    let mut combined = collect_long(&diff1, &diff2)?;
    combined.sort();
    combined.dedup();
    array_from_vec_long(&combined, a1.dtype().clone())
}

fn collect_long(a1: &Array, a2: &Array) -> Result<Vec<i64>, ManipulationError> {
    let mut result = Vec::new();
    unsafe {
        let ptr1 = a1.data_ptr() as *const i64;
        for i in 0..a1.size() {
            result.push(*ptr1.add(i));
        }
        let ptr2 = a2.data_ptr() as *const i64;
        for i in 0..a2.size() {
            result.push(*ptr2.add(i));
        }
    }
    Ok(result)
}

fn array_from_vec_long(values: &[i64], dtype: DType) -> Result<Array, ManipulationError> {
    let mut array = Array::new(vec![values.len() as i64], dtype)?;
    unsafe {
        let ptr = array.data_ptr_mut() as *mut i64;
        for (i, &val) in values.iter().enumerate() {
            *ptr.add(i) = val;
        }
    }
    Ok(array)
}

