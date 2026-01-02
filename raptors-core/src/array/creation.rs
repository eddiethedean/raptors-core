//! Array creation functions
//!
//! This module provides array creation functionality,
//! equivalent to NumPy's array creation functions from ctors.c

use crate::array::{Array, ArrayError};
use crate::types::{DType, NpyType};

/// Create an empty array with the specified shape and dtype
///
/// The array is allocated but not initialized (contains uninitialized memory).
pub fn empty(shape: Vec<i64>, dtype: DType) -> Result<Array, ArrayError> {
    Array::new(shape, dtype)
}

/// Create a zero-filled array with the specified shape and dtype
pub fn zeros(shape: Vec<i64>, dtype: DType) -> Result<Array, ArrayError> {
    let mut array = Array::new(shape, dtype)?;
    
    // Zero-fill the memory
    unsafe {
        let size = array.size() * array.itemsize();
        std::ptr::write_bytes(array.data_ptr_mut(), 0, size);
    }
    
    Ok(array)
}

/// Create a one-filled array with the specified shape and dtype
pub fn ones(shape: Vec<i64>, dtype: DType) -> Result<Array, ArrayError> {
    let mut array = Array::new(shape, dtype.clone())?;
    
    // Fill with ones based on dtype
    match dtype.type_() {
        NpyType::Bool => fill_with_value(&mut array, true)?,
        NpyType::Byte => fill_with_value(&mut array, 1i8)?,
        NpyType::UByte => fill_with_value(&mut array, 1u8)?,
        NpyType::Short => fill_with_value(&mut array, 1i16)?,
        NpyType::UShort => fill_with_value(&mut array, 1u16)?,
        NpyType::Int => fill_with_value(&mut array, 1i32)?,
        NpyType::UInt => fill_with_value(&mut array, 1u32)?,
        NpyType::Long | NpyType::LongLong => fill_with_value(&mut array, 1i64)?,
        NpyType::ULong | NpyType::ULongLong => fill_with_value(&mut array, 1u64)?,
        NpyType::Float | NpyType::Half => fill_with_value(&mut array, 1.0f32)?,
        NpyType::Double | NpyType::LongDouble => fill_with_value(&mut array, 1.0f64)?,
        _ => return Err(ArrayError::TypeMismatch),
    }
    
    Ok(array)
}

/// Fill array with a value (helper function)
fn fill_with_value<T: Copy>(array: &mut Array, value: T) -> Result<(), ArrayError> {
    let size = array.size();
    if size == 0 {
        return Ok(());
    }
    
    let itemsize = std::mem::size_of::<T>();
    if itemsize != array.itemsize() {
        return Err(ArrayError::TypeMismatch);
    }
    
    unsafe {
        let data = array.data_ptr_mut() as *mut T;
        for i in 0..size {
            *data.add(i) = value;
        }
    }
    
    Ok(())
}

