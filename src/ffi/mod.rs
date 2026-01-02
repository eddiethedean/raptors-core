//! Foreign Function Interface (FFI) module
//!
//! This module provides the C API compatibility layer,
//! exposing NumPy-compatible C functions for use as a drop-in replacement.

mod array_api;
mod conversion;

pub use array_api::*;
pub use conversion::*;

use libc::{c_int, c_void, size_t};

/// C-compatible array object structure
///
/// This matches NumPy's PyArrayObject structure for C API compatibility.
/// Fields are public for C API compatibility.
#[repr(C)]
pub struct PyArrayObject {
    /// Object header (for Python compatibility, will be NULL in pure C usage)
    pub ob_base: *mut c_void,
    /// Data pointer
    pub data: *mut u8,
    /// Number of dimensions
    pub nd: c_int,
    /// Type descriptor (simplified for now)
    pub descr: *mut c_void,
    /// Flags
    pub flags: u32,
    /// Shape array (MAXDIMS elements)
    pub dimensions: [i64; 64],
    /// Strides array (MAXDIMS elements)
    pub strides: [i64; 64],
    /// Base object
    pub base: *mut PyArrayObject,
    /// Descriptor for the array element type
    pub _descr: *mut c_void,
    /// Weak references (for Python compatibility)
    pub weakreflist: *mut c_void,
}

/// Array creation function
///
/// Creates a new array with the specified parameters.
/// This is a placeholder that will be expanded as we implement more functionality.
#[no_mangle]
pub extern "C" fn PyArray_New(
    _subtype: *mut c_void,
    _nd: c_int,
    _dimensions: *const i64,
    _type_num: c_int,
    _strides: *const i64,
    _data: *mut c_void,
    _itemsize: c_int,
    _flags: c_int,
    _obj: *mut c_void,
) -> *mut PyArrayObject {
    // TODO: Implement full PyArray_New functionality
    std::ptr::null_mut()
}

/// Get array size
///
/// Returns the total number of elements in the array.
#[no_mangle]
pub extern "C" fn PyArray_SIZE(arr: *mut PyArrayObject) -> size_t {
    if arr.is_null() {
        return 0;
    }
    
    unsafe {
        let arr_ref = &*arr;
        let mut size: size_t = 1;
        for i in 0..(arr_ref.nd as usize) {
            if arr_ref.dimensions[i] > 0 {
                size *= arr_ref.dimensions[i] as size_t;
            }
        }
        size
    }
}

/// Check if object is an array
#[no_mangle]
pub extern "C" fn PyArray_Check(op: *mut c_void) -> c_int {
    // Simplified check - in real implementation would check object type
    if op.is_null() {
        return 0;
    }
    // TODO: Implement proper type checking
    0
}

