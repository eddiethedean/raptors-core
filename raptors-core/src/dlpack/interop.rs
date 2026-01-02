//! DLPack interoperability functions

use crate::array::Array;

use super::{DLTensor, DLPackError};

/// Import array from DLPack tensor
///
/// Equivalent to NumPy's `from_dlpack`
///
/// # Arguments
/// * `dlpack` - DLPack tensor to import
///
/// # Returns
/// * `Ok(Array)` - Imported array
/// * `Err(DLPackError)` if import fails
///
/// # Safety
/// This function assumes the DLTensor is valid
pub unsafe fn from_dlpack(dlpack: *mut DLTensor) -> Result<Array, DLPackError> {
    super::dlpack_to_array(dlpack)
}

/// Export array to DLPack tensor
///
/// Equivalent to NumPy's `__dlpack__`
///
/// # Arguments
/// * `array` - Array to export
///
/// # Returns
/// * `Ok(*mut DLTensor)` - Exported DLPack tensor
/// * `Err(DLPackError)` if export fails
///
/// # Safety
/// The returned DLTensor must be properly managed (freed when done)
pub unsafe fn to_dlpack(array: &Array) -> Result<*mut DLTensor, DLPackError> {
    super::array_to_dlpack(array)
}

/// Delete DLPack tensor
///
/// # Safety
/// This function assumes the pointer is valid and was allocated by array_to_dlpack
pub unsafe fn delete_dlpack_tensor(tensor: *mut DLTensor) {
    if !tensor.is_null() {
        drop(Box::from_raw(tensor));
    }
}

