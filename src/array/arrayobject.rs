//! Core array object implementation
//!
//! This module implements the core array structure, equivalent to
//! NumPy's PyArrayObject and related functionality from arrayobject.c

use crate::types::*;
use super::flags::ArrayFlags;

/// Maximum number of dimensions for arrays
pub const MAXDIMS: usize = 64;

/// Core array structure
///
/// This is the Rust internal representation of an array.
/// For C API compatibility, see `ffi::PyArrayObject`.
#[derive(Debug, Clone)]
pub struct Array {
    /// Data pointer
    data: *mut u8,
    /// Number of dimensions
    ndim: usize,
    /// Shape (dimensions)
    shape: Vec<i64>,
    /// Strides (bytes per element in each dimension)
    strides: Vec<i64>,
    /// Data type descriptor
    dtype: DType,
    /// Array flags
    flags: ArrayFlags,
    /// Size of each element in bytes
    itemsize: usize,
    /// Base array (for views)
    base: Option<*mut Array>,
    /// Owned data flag
    owns_data: bool,
}

impl Array {
    /// Create a new array
    pub fn new(shape: Vec<i64>, dtype: DType) -> Result<Self, ArrayError> {
        let itemsize = dtype.itemsize();
        let size = shape.iter().product::<i64>() as usize;
        let data_size = size * itemsize;
        
        let data = unsafe {
            let layout = std::alloc::Layout::from_size_align(data_size, itemsize)
                .map_err(|_| ArrayError::InvalidLayout)?;
            std::alloc::alloc(layout) as *mut u8
        };
        
        if data.is_null() {
            return Err(ArrayError::AllocationFailed);
        }
        
        let ndim = shape.len();
        let strides = compute_strides(&shape, itemsize);
        
        let mut array = Array {
            data,
            ndim,
            shape,
            strides,
            dtype,
            flags: ArrayFlags::empty(),
            itemsize,
            base: None,
            owns_data: true,
        };
        
        // Update flags based on memory layout
        array.update_flags();
        
        Ok(array)
    }
    
    /// Get the shape of the array
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    
    /// Get the strides of the array
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }
    
    /// Get the data type
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
    
    /// Get the data pointer
    pub fn data_ptr(&self) -> *const u8 {
        self.data
    }
    
    /// Get the data pointer as mutable
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        self.data
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.ndim
    }
    
    /// Get the total size (number of elements)
    pub fn size(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }
    
    /// Get the item size in bytes
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }
    
    /// Get the array flags
    pub fn flags(&self) -> ArrayFlags {
        self.flags
    }
    
    /// Check if array is C-contiguous
    pub fn is_c_contiguous(&self) -> bool {
        self.flags.contains(ArrayFlags::C_CONTIGUOUS)
    }
    
    /// Check if array is Fortran-contiguous
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.contains(ArrayFlags::F_CONTIGUOUS)
    }
    
    /// Check if array is writeable
    pub fn is_writeable(&self) -> bool {
        self.flags.contains(ArrayFlags::WRITEABLE)
    }
    
    /// Check if array owns its data
    pub fn owns_data(&self) -> bool {
        self.flags.contains(ArrayFlags::OWNDATA)
    }
    
    /// Update flags based on memory layout
    pub fn update_flags(&mut self) {
        let mut new_flags = ArrayFlags::empty();
        
        // Check C-contiguity
        if is_c_contiguous_layout(&self.shape, &self.strides, self.itemsize) {
            new_flags |= ArrayFlags::C_CONTIGUOUS;
        }
        
        // Check F-contiguity
        if is_f_contiguous_layout(&self.shape, &self.strides, self.itemsize) {
            new_flags |= ArrayFlags::F_CONTIGUOUS;
        }
        
        // Set default flags for new arrays
        new_flags |= ArrayFlags::WRITEABLE | ArrayFlags::ALIGNED;
        
        // Set OWNDATA based on internal flag
        if self.owns_data {
            new_flags |= ArrayFlags::OWNDATA;
        }
        
        // Preserve any other existing flags that should be preserved
        new_flags |= self.flags & (ArrayFlags::WRITEBACKIFCOPY | ArrayFlags::UPDATEIFCOPY);
        
        self.flags = new_flags;
    }
}

/// Check if memory layout is C-contiguous
fn is_c_contiguous_layout(shape: &[i64], strides: &[i64], itemsize: usize) -> bool {
    if shape.is_empty() {
        return true;
    }
    
    // Last dimension stride should equal itemsize
    if strides[shape.len() - 1] != itemsize as i64 {
        return false;
    }
    
    // Each stride should be the product of subsequent dimensions * itemsize
    for i in (0..shape.len() - 1).rev() {
        let expected_stride = strides[i + 1] * shape[i + 1];
        if strides[i] != expected_stride {
            return false;
        }
    }
    
    true
}

/// Check if memory layout is Fortran-contiguous
fn is_f_contiguous_layout(shape: &[i64], strides: &[i64], itemsize: usize) -> bool {
    if shape.is_empty() {
        return true;
    }
    
    // First dimension stride should equal itemsize
    if strides[0] != itemsize as i64 {
        return false;
    }
    
    // Each stride should be the product of previous dimensions * itemsize
    for i in 1..shape.len() {
        let expected_stride = strides[i - 1] * shape[i - 1];
        if strides[i] != expected_stride {
            return false;
        }
    }
    
    true
}

impl Drop for Array {
    fn drop(&mut self) {
        if self.owns_data && !self.data.is_null() {
            let size = self.size() * self.itemsize;
            unsafe {
                let layout = std::alloc::Layout::from_size_align(size, self.itemsize)
                    .unwrap();
                std::alloc::dealloc(self.data, layout);
            }
        }
    }
}

/// Compute strides from shape and itemsize
fn compute_strides(shape: &[i64], itemsize: usize) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = itemsize as i64;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

/// Array errors
#[derive(Debug, Clone)]
pub enum ArrayError {
    /// Memory allocation failed
    AllocationFailed,
    /// Invalid layout for allocation
    InvalidLayout,
    /// Invalid shape or dimension
    InvalidShape,
    /// Type mismatch
    TypeMismatch,
}

impl std::fmt::Display for ArrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrayError::AllocationFailed => write!(f, "Memory allocation failed"),
            ArrayError::InvalidLayout => write!(f, "Invalid memory layout"),
            ArrayError::InvalidShape => write!(f, "Invalid array shape"),
            ArrayError::TypeMismatch => write!(f, "Type mismatch"),
        }
    }
}

impl std::error::Error for ArrayError {}

