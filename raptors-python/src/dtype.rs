//! DType Python bindings
//!
//! This module provides Python bindings for the DType type.

use pyo3::prelude::*;
use raptors_core::types::{DType, NpyType, register_custom_type, create_custom_dtype, get_custom_type_id, CustomType, CustomTypeError};

/// Python DType class
#[pyclass]
pub struct PyDType {
    pub(crate) inner: DType,
}

impl PyDType {
    /// Get reference to inner dtype (for internal use)
    pub(crate) fn get_inner(&self) -> &DType {
        &self.inner
    }
}

#[pymethods]
impl PyDType {
    /// Create a new dtype
    #[new]
    fn new(type_name: String) -> PyResult<Self> {
        let npy_type = match type_name.as_str() {
            "bool" => NpyType::Bool,
            "int8" => NpyType::Byte,
            "uint8" => NpyType::UByte,
            "int16" => NpyType::Short,
            "uint16" => NpyType::UShort,
            "int32" => NpyType::Int,
            "uint32" => NpyType::UInt,
            "int64" => NpyType::LongLong,
            "uint64" => NpyType::ULongLong,
            "float32" | "float" => NpyType::Float,
            "float64" | "double" => NpyType::Double,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown dtype: {}", type_name)
            )),
        };
        Ok(PyDType {
            inner: DType::new(npy_type),
        })
    }
    
    /// Get the dtype name
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }
    
    /// Get the item size in bytes
    #[getter]
    fn itemsize(&self) -> usize {
        self.inner.itemsize()
    }
    
    /// Get the dtype kind
    #[getter]
    fn kind(&self) -> String {
        match self.inner.type_() {
            NpyType::Bool => "b".to_string(),
            NpyType::Byte | NpyType::UByte | NpyType::Short | NpyType::UShort |
            NpyType::Int | NpyType::UInt | NpyType::Long | NpyType::ULong |
            NpyType::LongLong | NpyType::ULongLong => "i".to_string(),
            NpyType::Float | NpyType::Double | NpyType::LongDouble | NpyType::Half => "f".to_string(),
            NpyType::CFloat | NpyType::CDouble | NpyType::CLongDouble => "c".to_string(),
            NpyType::String | NpyType::Unicode => "S".to_string(),
            NpyType::DateTime => "M".to_string(),
            NpyType::Timedelta => "m".to_string(),
            _ => "O".to_string(),
        }
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("dtype('{}')", self.inner.name())
    }
    
    /// String representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
    
    /// Create a custom dtype from a registered custom type ID
    #[staticmethod]
    fn from_custom_type_id(type_id: u32) -> PyResult<Self> {
        let dtype = create_custom_dtype(type_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to create custom dtype: {}", e)
            ))?;
        Ok(PyDType { inner: dtype })
    }
    
    /// Create a custom dtype from a registered custom type name
    #[staticmethod]
    fn from_custom_type_name(name: String) -> PyResult<Self> {
        let type_id = get_custom_type_id(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Custom type '{}' not found", name)
            ))?;
        Self::from_custom_type_id(type_id)
    }
    
    /// Check if this is a custom dtype
    fn is_custom(&self) -> bool {
        self.inner.custom_type_id().is_some()
    }
    
    /// Get the custom type ID if this is a custom dtype
    fn custom_type_id(&self) -> Option<u32> {
        self.inner.custom_type_id()
    }
}

/// Simple custom type implementation for Python
/// This allows Python users to register custom types with basic information
struct SimpleCustomType {
    itemsize: usize,
    align: usize,
    name: String,
}

impl CustomType for SimpleCustomType {
    fn itemsize(&self) -> usize {
        self.itemsize
    }
    
    fn align(&self) -> usize {
        self.align
    }
    
    fn to_string(&self) -> String {
        self.name.clone()
    }
    
    fn from_bytes(&self, bytes: &[u8]) -> Result<Vec<u8>, CustomTypeError> {
        if bytes.len() < self.itemsize {
            return Err(CustomTypeError::InvalidType(
                format!("Not enough bytes: expected {}, got {}", self.itemsize, bytes.len())
            ));
        }
        Ok(bytes[..self.itemsize].to_vec())
    }
    
    fn to_bytes(&self, value: &[u8]) -> Result<Vec<u8>, CustomTypeError> {
        if value.len() < self.itemsize {
            return Err(CustomTypeError::InvalidType(
                format!("Not enough bytes: expected {}, got {}", self.itemsize, value.len())
            ));
        }
        Ok(value[..self.itemsize].to_vec())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Register a custom type from Python
///
/// # Arguments
/// * `name` - Type name
/// * `itemsize` - Size in bytes
/// * `align` - Alignment requirement in bytes
///
/// # Returns
/// The custom type ID assigned to this type
#[pyfunction]
pub fn register_custom_dtype(name: String, itemsize: usize, align: usize) -> PyResult<u32> {
    let custom_type = SimpleCustomType {
        itemsize,
        align,
        name: name.clone(),
    };
    
    let type_id = register_custom_type(custom_type)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to register custom dtype '{}': {}", name, e)
        ))?;
    
    Ok(type_id)
}

/// Get custom type ID by name
#[pyfunction]
pub fn get_custom_dtype_id(name: String) -> PyResult<u32> {
    get_custom_type_id(&name)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Custom type '{}' not found", name)
        ))
}

/// Add dtype constants to module
pub fn add_dtype_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add dtype constants matching NumPy
    m.add("bool_", PyDType { inner: DType::new(NpyType::Bool) })?;
    m.add("int8", PyDType { inner: DType::new(NpyType::Byte) })?;
    m.add("uint8", PyDType { inner: DType::new(NpyType::UByte) })?;
    m.add("int16", PyDType { inner: DType::new(NpyType::Short) })?;
    m.add("uint16", PyDType { inner: DType::new(NpyType::UShort) })?;
    m.add("int32", PyDType { inner: DType::new(NpyType::Int) })?;
    m.add("uint32", PyDType { inner: DType::new(NpyType::UInt) })?;
    m.add("int64", PyDType { inner: DType::new(NpyType::LongLong) })?;
    m.add("uint64", PyDType { inner: DType::new(NpyType::ULongLong) })?;
    m.add("float32", PyDType { inner: DType::new(NpyType::Float) })?;
    m.add("float64", PyDType { inner: DType::new(NpyType::Double) })?;
    m.add("float_", PyDType { inner: DType::new(NpyType::Double) })?;
    m.add("int_", PyDType { inner: DType::new(NpyType::LongLong) })?;
    
    Ok(())
}

