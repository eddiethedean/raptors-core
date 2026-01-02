//! Type promotion implementation
//!
//! Type promotion finds the common type for two types, following NumPy's rules

use crate::types::{DType, NpyType};

/// Type promotion error
#[derive(Debug, Clone)]
pub enum PromotionError {
    /// Types cannot be promoted
    CannotPromote,
}

impl std::fmt::Display for PromotionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromotionError::CannotPromote => write!(f, "Types cannot be promoted"),
        }
    }
}

impl std::error::Error for PromotionError {}

/// Promote two types to a common type
///
/// Returns the promoted type that can safely hold values from both input types.
/// This follows NumPy's type promotion rules.
pub fn promote_types(type1: NpyType, type2: NpyType) -> Result<NpyType, PromotionError> {
    // If types are the same, return that type
    if type1 == type2 {
        return Ok(type1);
    }
    
    // Type promotion hierarchy (simplified version)
    // The order matters - later types can safely represent earlier types
    let type_hierarchy: &[NpyType] = &[
        NpyType::Bool,
        NpyType::Byte,
        NpyType::UByte,
        NpyType::Short,
        NpyType::UShort,
        NpyType::Int,
        NpyType::UInt,
        NpyType::Long,
        NpyType::ULong,
        NpyType::LongLong,
        NpyType::ULongLong,
        NpyType::Half,
        NpyType::Float,
        NpyType::Double,
        NpyType::LongDouble,
    ];
    
    let pos1 = type_hierarchy.iter().position(|&t| t == type1);
    let pos2 = type_hierarchy.iter().position(|&t| t == type2);
    
    match (pos1, pos2) {
        (Some(p1), Some(p2)) => {
            // Return the type with higher position (more general)
            Ok(type_hierarchy[p1.max(p2)])
        }
        _ => {
            // Handle special cases
            match (type1, type2) {
                // Complex types
                (NpyType::CFloat, _) | (_, NpyType::CFloat) => {
                    if type2 == NpyType::Float || type1 == NpyType::Float {
                        Ok(NpyType::CFloat)
                    } else {
                        Err(PromotionError::CannotPromote)
                    }
                }
                (NpyType::CDouble, _) | (_, NpyType::CDouble) => {
                    if type2 == NpyType::Double || type1 == NpyType::Double {
                        Ok(NpyType::CDouble)
                    } else {
                        Err(PromotionError::CannotPromote)
                    }
                }
                _ => Err(PromotionError::CannotPromote),
            }
        }
    }
}

/// Promote two dtypes to a common dtype
pub fn promote_dtypes(dtype1: &DType, dtype2: &DType) -> Result<DType, PromotionError> {
    let promoted_type = promote_types(dtype1.type_(), dtype2.type_())?;
    Ok(DType::new(promoted_type))
}

