//! String encoding operations

use crate::array::Array;

use super::StringError;

/// Character encoding types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// UTF-8 encoding
    Utf8,
    /// ASCII encoding
    Ascii,
    /// Latin-1 encoding
    Latin1,
}

/// Convert string array encoding
///
/// # Arguments
/// * `array` - String array to convert
/// * `from_encoding` - Source encoding
/// * `to_encoding` - Target encoding
///
/// # Returns
/// * `Ok(Array)` - Converted string array
/// * `Err(StringError)` if conversion fails
pub fn convert_encoding(
    array: &Array,
    from_encoding: Encoding,
    to_encoding: Encoding,
) -> Result<Array, StringError> {
    if from_encoding == to_encoding {
        // No conversion needed
        return Ok(array.clone());
    }
    
    // Simplified implementation - Rust strings are UTF-8
    // Full implementation would handle encoding conversion properly
    if from_encoding != Encoding::Utf8 || to_encoding != Encoding::Utf8 {
        return Err(StringError::EncodingError(
            "Only UTF-8 encoding is currently supported".to_string(),
        ));
    }
    
    Ok(array.clone())
}

/// Validate string array encoding
///
/// # Arguments
/// * `array` - String array to validate
/// * `encoding` - Expected encoding
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(StringError)` if invalid
pub fn validate_encoding(array: &Array, encoding: Encoding) -> Result<(), StringError> {
    if !super::is_string_array(array) {
        return Err(StringError::TypeMismatch);
    }
    
    // For UTF-8, validate each string
    if encoding == Encoding::Utf8 {
        let size = array.size();
        for i in 0..size {
            // get_string already validates UTF-8
            super::get_string(array, i)?;
        }
    }
    
    Ok(())
}

/// Check if string array contains valid UTF-8
pub fn is_valid_utf8(array: &Array) -> bool {
    validate_encoding(array, Encoding::Utf8).is_ok()
}

