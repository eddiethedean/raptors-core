//! Type casting implementation
//!
//! Type casting converts values between different types

use crate::types::NpyType;

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

