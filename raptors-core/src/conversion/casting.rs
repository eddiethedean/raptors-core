//! Type casting implementation
//!
//! Type casting converts values between different types

use crate::types::NpyType;
use crate::array::Array;
use crate::types::DType;

/// Casting safety level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastingSafety {
    /// Safe cast (no information loss)
    Safe,
    /// Same kind cast (e.g., int to int)
    SameKind,
    /// Unsafe cast (may lose information)
    Unsafe,
    /// Casting not allowed
    No,
}

/// Check if casting from one type to another is allowed
pub fn can_cast(from_type: NpyType, to_type: NpyType, casting: CastingSafety) -> bool {
    if from_type == to_type {
        return true;
    }
    
    match casting {
        CastingSafety::No => false,
        CastingSafety::Safe => is_safe_cast(from_type, to_type),
        CastingSafety::SameKind => is_same_kind_cast(from_type, to_type),
        CastingSafety::Unsafe => true, // Allow all casts
    }
}

/// Check if a cast is safe (no information loss)
fn is_safe_cast(from_type: NpyType, to_type: NpyType) -> bool {
    // Simplified safe cast rules
    // Safe casts are when the destination type can represent all values
    // from the source type without loss
    
    match (from_type, to_type) {
        // Integer widening
        (NpyType::Byte, NpyType::Short | NpyType::Int | NpyType::Long | NpyType::LongLong) => true,
        (NpyType::Short, NpyType::Int | NpyType::Long | NpyType::LongLong) => true,
        (NpyType::Int, NpyType::Long | NpyType::LongLong) => true,
        (NpyType::Long, NpyType::LongLong) => true,
        
        // Unsigned widening
        (NpyType::UByte, NpyType::UShort | NpyType::UInt | NpyType::ULong | NpyType::ULongLong) => true,
        (NpyType::UShort, NpyType::UInt | NpyType::ULong | NpyType::ULongLong) => true,
        (NpyType::UInt, NpyType::ULong | NpyType::ULongLong) => true,
        (NpyType::ULong, NpyType::ULongLong) => true,
        
        // Integer to float (within precision)
        (NpyType::Byte | NpyType::Short | NpyType::Int, NpyType::Float) => true,
        (NpyType::Byte | NpyType::Short | NpyType::Int | NpyType::Long, NpyType::Double) => true,
        
        // Float widening
        (NpyType::Half, NpyType::Float | NpyType::Double | NpyType::LongDouble) => true,
        (NpyType::Float, NpyType::Double | NpyType::LongDouble) => true,
        (NpyType::Double, NpyType::LongDouble) => true,
        
        // Complex widening
        (NpyType::CFloat, NpyType::CDouble | NpyType::CLongDouble) => true,
        (NpyType::CDouble, NpyType::CLongDouble) => true,
        
        _ => false,
    }
}

/// Check if a cast is same-kind (both are integers, both are floats, etc.)
fn is_same_kind_cast(from_type: NpyType, to_type: NpyType) -> bool {
    
    let from_kind = type_kind(from_type);
    let to_kind = type_kind(to_type);
    from_kind == to_kind
}

/// Get the kind of a type (integer, float, complex, etc.)
fn type_kind(ty: NpyType) -> &'static str {
    match ty {
        NpyType::Bool | NpyType::Byte | NpyType::UByte
        | NpyType::Short | NpyType::UShort
        | NpyType::Int | NpyType::UInt
        | NpyType::Long | NpyType::ULong
        | NpyType::LongLong | NpyType::ULongLong => "integer",
        
        NpyType::Half | NpyType::Float | NpyType::Double | NpyType::LongDouble => "float",
        
        NpyType::CFloat | NpyType::CDouble | NpyType::CLongDouble => "complex",
        
        _ => "other",
    }
}

/// Conversion error
#[derive(Debug, Clone)]
pub enum ConversionError {
    /// Conversion not supported
    UnsupportedConversion,
    /// Array error
    ArrayError(crate::array::ArrayError),
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConversionError::UnsupportedConversion => write!(f, "Unsupported conversion"),
            ConversionError::ArrayError(e) => write!(f, "Array error: {}", e),
        }
    }
}

impl std::error::Error for ConversionError {}

impl From<crate::array::ArrayError> for ConversionError {
    fn from(e: crate::array::ArrayError) -> Self {
        ConversionError::ArrayError(e)
    }
}

/// Convert an array to a different dtype
///
/// This function creates a new array with the target dtype and converts
/// all elements from the source type to the target type.
pub fn convert_array(array: &Array, target_dtype: DType) -> Result<Array, ConversionError> {
    use crate::array::ArrayError;
    
    let source_dtype = array.dtype();
    let source_type = source_dtype.type_();
    let target_type = target_dtype.type_();
    
    // If types are the same, just copy the array
    if source_type == target_type {
        return Ok(array.copy());
    }
    
    // Create output array with target dtype
    let shape = array.shape().to_vec();
    let mut output = Array::new(shape.clone(), target_dtype.clone())
        .map_err(ConversionError::ArrayError)?;
    
    // Convert elements based on type combination
    let size = array.size();
    let src_strides = array.strides().to_vec();
    let dst_strides = output.strides().to_vec();
    
    // Helper to convert flat index to coordinates
    fn index_to_coords(index: usize, shape: &[i64], coords: &mut [i64]) {
        let mut idx = index;
        for i in (0..shape.len()).rev() {
            coords[i] = (idx % shape[i] as usize) as i64;
            idx /= shape[i] as usize;
        }
    }
    
    // Helper to convert coordinates to byte offset
    fn coords_to_offset(coords: &[i64], strides: &[i64]) -> usize {
        let mut offset = 0;
        for (i, &coord) in coords.iter().enumerate() {
            offset += (coord * strides[i]) as usize;
        }
        offset
    }
    
    let mut coords = vec![0; shape.len()];
    
    // Convert each element
    for flat_idx in 0..size {
        index_to_coords(flat_idx, &shape, &mut coords);
        let src_offset = coords_to_offset(&coords, &src_strides);
        let dst_offset = coords_to_offset(&coords, &dst_strides);
        
        unsafe {
            let src_ptr = array.data_ptr().add(src_offset);
            let dst_ptr = output.data_ptr_mut().add(dst_offset);
            
            // Perform type conversion based on source and target types
            convert_element(src_ptr, dst_ptr, source_type, target_type)?;
        }
    }
    
    Ok(output)
}

/// Convert a single element from source type to target type
///
/// # Safety
/// The source and destination pointers must be valid and properly aligned
/// for their respective types.
unsafe fn convert_element(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    src_type: NpyType,
    dst_type: NpyType,
) -> Result<(), ConversionError> {
    use NpyType::*;
    
    match (src_type, dst_type) {
        // Integer to integer conversions
        (Int, Double) => {
            let val = *(src_ptr as *const i32);
            *(dst_ptr as *mut f64) = val as f64;
        }
        (Int, Float) => {
            let val = *(src_ptr as *const i32);
            *(dst_ptr as *mut f32) = val as f32;
        }
        (Double, Int) => {
            let val = *(src_ptr as *const f64);
            *(dst_ptr as *mut i32) = val as i32;
        }
        (Float, Int) => {
            let val = *(src_ptr as *const f32);
            *(dst_ptr as *mut i32) = val as i32;
        }
        (Double, Float) => {
            let val = *(src_ptr as *const f64);
            *(dst_ptr as *mut f32) = val as f32;
        }
        (Float, Double) => {
            let val = *(src_ptr as *const f32);
            *(dst_ptr as *mut f64) = val as f64;
        }
        // Same type (shouldn't happen, but handle gracefully)
        _ if src_type == dst_type => {
            // For same types, copy the element
            let src_itemsize = crate::types::DType::new(src_type).itemsize();
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, src_itemsize);
        }
        _ => {
            return Err(ConversionError::UnsupportedConversion);
        }
    }
    
    Ok(())
}

