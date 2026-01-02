//! DLPack tensor structure

use std::os::raw::{c_int, c_void};

/// DLPack device type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDeviceType {
    /// CPU device
    CPU = 1,
    /// CUDA GPU device
    CUDA = 2,
    /// OpenCL device
    OpenCL = 4,
    /// Vulkan device
    Vulkan = 7,
    /// Metal device
    Metal = 8,
    /// VPI device
    VPI = 9,
    /// ROCm device
    ROCm = 10,
}

/// DLPack device structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    /// Device type
    pub device_type: DLDeviceType,
    /// Device ID
    pub device_id: c_int,
}

/// DLPack data type code
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDataTypeCode {
    /// Integer type
    Int = 0,
    /// Unsigned integer type
    UInt = 1,
    /// Floating point type
    Float = 2,
    /// Opaque handle type
    OpaqueHandle = 3,
}

/// DLPack data type structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    /// Type code
    pub code: u8,
    /// Bits (e.g., 32 for int32)
    pub bits: u8,
    /// Number of lanes (for vector types)
    pub lanes: u16,
}

/// DLPack tensor structure
///
/// This matches the DLPack specification for tensor representation
#[repr(C)]
pub struct DLTensor {
    /// Data pointer
    pub data: *mut c_void,
    /// Device where the data resides
    pub device: DLDevice,
    /// Number of dimensions
    pub ndim: c_int,
    /// Data type
    pub dtype: DLDataType,
    /// Shape array (length = ndim)
    pub shape: *mut i64,
    /// Strides array (length = ndim, can be NULL for compact tensors)
    pub strides: *mut i64,
    /// Byte offset into data
    pub byte_offset: u64,
}

/// DLPack error
#[derive(Debug, Clone)]
pub enum DLPackError {
    /// Array error
    ArrayError(crate::array::ArrayError),
    /// Invalid device type
    InvalidDevice,
    /// Unsupported dtype
    UnsupportedDtype,
    /// Invalid tensor structure
    InvalidTensor,
    /// Memory management error
    MemoryError(String),
}

impl std::fmt::Display for DLPackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DLPackError::ArrayError(e) => write!(f, "Array error: {}", e),
            DLPackError::InvalidDevice => write!(f, "Invalid device type"),
            DLPackError::UnsupportedDtype => write!(f, "Unsupported dtype"),
            DLPackError::InvalidTensor => write!(f, "Invalid tensor structure"),
            DLPackError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
        }
    }
}

impl std::error::Error for DLPackError {}

impl From<crate::array::ArrayError> for DLPackError {
    fn from(err: crate::array::ArrayError) -> Self {
        DLPackError::ArrayError(err)
    }
}

