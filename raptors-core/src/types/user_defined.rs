//! User-defined types implementation
//!
//! This module provides support for custom dtypes beyond the built-in NumPy types

use crate::array::ArrayError;
use crate::types::{DType, NpyType};
use std::collections::HashMap;
use std::sync::Mutex;

/// Custom type trait
///
/// Types implementing this trait can be used as custom array element types
pub trait CustomType: Send + Sync {
    /// Size in bytes
    fn itemsize(&self) -> usize;
    
    /// Alignment requirement in bytes
    fn align(&self) -> usize;
    
    /// String representation of the type
    fn to_string(&self) -> String;
    
    /// Convert from bytes
    #[allow(clippy::wrong_self_convention)] // This is an intentional naming convention
    fn from_bytes(&self, bytes: &[u8]) -> Result<Vec<u8>, CustomTypeError>;
    
    /// Convert to bytes
    fn to_bytes(&self, value: &[u8]) -> Result<Vec<u8>, CustomTypeError>;
    
    /// Type name
    fn name(&self) -> &str;
    
    /// Convert from another type (default implementation returns error)
    ///
    /// This allows custom types to define conversions from built-in types
    fn convert_from(&self, _source_type: &NpyType, _data: &[u8]) -> Result<Vec<u8>, CustomTypeError> {
        Err(CustomTypeError::InvalidType(
            "Conversion from this type not supported".to_string()
        ))
    }
    
    /// Convert to another type (default implementation returns error)
    ///
    /// This allows custom types to define conversions to built-in types
    fn convert_to(&self, _target_type: &NpyType, _data: &[u8]) -> Result<Vec<u8>, CustomTypeError> {
        Err(CustomTypeError::InvalidType(
            "Conversion to this type not supported".to_string()
        ))
    }
    
    /// Perform an optimized operation (optional)
    ///
    /// This allows custom types to provide optimized implementations
    /// for common operations. Returns None if no optimization is available.
    fn optimized_operation(
        &self,
        _op_name: &str,
        _inputs: &[&[u8]],
        _output: &mut [u8],
    ) -> Option<Result<(), CustomTypeError>> {
        None // Default: no optimization available
    }
}

/// Custom type error
#[derive(Debug, Clone)]
pub enum CustomTypeError {
    /// Array error
    ArrayError(ArrayError),
    /// Invalid type
    InvalidType(String),
    /// Serialization error
    SerializationError(String),
}

impl std::fmt::Display for CustomTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CustomTypeError::ArrayError(e) => write!(f, "Array error: {}", e),
            CustomTypeError::InvalidType(msg) => write!(f, "Invalid type: {}", msg),
            CustomTypeError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for CustomTypeError {}

impl From<ArrayError> for CustomTypeError {
    fn from(err: ArrayError) -> Self {
        CustomTypeError::ArrayError(err)
    }
}

/// Type ID for custom types
pub type CustomTypeId = u32;

/// Type metadata stored in registry
#[derive(Debug, Clone)]
pub struct TypeMetadata {
    /// Size in bytes
    pub itemsize: usize,
    /// Alignment requirement in bytes
    pub align: usize,
    /// Type name
    pub name: String,
}

/// Global type registry
static TYPE_REGISTRY: Mutex<Option<TypeRegistry>> = Mutex::new(None);

/// Conversion function type
pub type ConversionFunction = Box<dyn Fn(&[u8]) -> Result<Vec<u8>, CustomTypeError> + Send + Sync>;

/// Type registry for custom types
pub struct TypeRegistry {
    /// Next available type ID
    next_id: CustomTypeId,
    /// Registered types
    types: HashMap<CustomTypeId, Box<dyn CustomType>>,
    /// Type metadata (itemsize, align, name)
    metadata: HashMap<CustomTypeId, TypeMetadata>,
    /// Name to ID mapping
    name_to_id: HashMap<String, CustomTypeId>,
    /// Conversion registry: (source_type_id, target_type_id) -> conversion function
    conversions: HashMap<(CustomTypeId, CustomTypeId), ConversionFunction>,
    /// Conversions from built-in types: (NpyType, target_type_id) -> conversion function
    conversions_from_builtin: HashMap<(NpyType, CustomTypeId), ConversionFunction>,
    /// Conversions to built-in types: (source_type_id, NpyType) -> conversion function
    conversions_to_builtin: HashMap<(CustomTypeId, NpyType), ConversionFunction>,
}

impl TypeRegistry {
    /// Create a new type registry
    fn new() -> Self {
        TypeRegistry {
            next_id: 1000, // Start custom types at 1000
            types: HashMap::new(),
            metadata: HashMap::new(),
            name_to_id: HashMap::new(),
            conversions: HashMap::new(),
            conversions_from_builtin: HashMap::new(),
            conversions_to_builtin: HashMap::new(),
        }
    }
    
    /// Get the global registry (lazy initialization)
    fn get() -> std::sync::MutexGuard<'static, Option<TypeRegistry>> {
        let mut registry = TYPE_REGISTRY.lock().unwrap();
        if registry.is_none() {
            *registry = Some(TypeRegistry::new());
        }
        registry
    }
    
    /// Register a custom type
    pub fn register<T: CustomType + 'static>(custom_type: T) -> Result<CustomTypeId, CustomTypeError> {
        let mut registry = TypeRegistry::get();
        let reg = registry.as_mut().unwrap();
        
        let id = reg.next_id;
        reg.next_id += 1;
        
        let name = custom_type.name().to_string();
        
        if reg.name_to_id.contains_key(&name) {
            return Err(CustomTypeError::InvalidType(
                format!("Type '{}' already registered", name)
            ));
        }
        
        // Extract and store metadata
        let metadata = TypeMetadata {
            itemsize: custom_type.itemsize(),
            align: custom_type.align(),
            name: name.clone(),
        };
        
        reg.name_to_id.insert(name.clone(), id);
        reg.types.insert(id, Box::new(custom_type));
        reg.metadata.insert(id, metadata);
        
        Ok(id)
    }
    
    /// Get type metadata by ID
    pub fn get_metadata_by_id(id: CustomTypeId) -> Option<TypeMetadata> {
        let registry = TypeRegistry::get();
        let reg = registry.as_ref().unwrap();
        reg.metadata.get(&id).cloned()
    }
    
    /// Get a custom type by ID
    pub fn get_by_id(_id: CustomTypeId) -> Option<Box<dyn CustomType>> {
        // Clone the type (simplified - in practice would need better cloning)
        // For now, return None as we can't easily clone trait objects
        // This is a limitation of the current implementation
        None
    }
    
    /// Get a custom type by name
    pub fn get_by_name(name: &str) -> Option<CustomTypeId> {
        let registry = TypeRegistry::get();
        let reg = registry.as_ref().unwrap();
        
        reg.name_to_id.get(name).copied()
    }
    
    /// Register a conversion function between two custom types
    pub fn register_conversion(
        from_id: CustomTypeId,
        to_id: CustomTypeId,
        conversion_fn: ConversionFunction,
    ) -> Result<(), CustomTypeError> {
        let mut registry = TypeRegistry::get();
        let reg = registry.as_mut().unwrap();
        
        // Verify both types exist
        if !reg.types.contains_key(&from_id) {
            return Err(CustomTypeError::InvalidType(
                format!("Source type ID {} not found", from_id)
            ));
        }
        if !reg.types.contains_key(&to_id) {
            return Err(CustomTypeError::InvalidType(
                format!("Target type ID {} not found", to_id)
            ));
        }
        
        reg.conversions.insert((from_id, to_id), conversion_fn);
        Ok(())
    }
    
    /// Register a conversion function from a built-in type to a custom type
    pub fn register_conversion_from_builtin(
        from_type: NpyType,
        to_id: CustomTypeId,
        conversion_fn: ConversionFunction,
    ) -> Result<(), CustomTypeError> {
        let mut registry = TypeRegistry::get();
        let reg = registry.as_mut().unwrap();
        
        // Verify target type exists
        if !reg.types.contains_key(&to_id) {
            return Err(CustomTypeError::InvalidType(
                format!("Target type ID {} not found", to_id)
            ));
        }
        
        reg.conversions_from_builtin.insert((from_type, to_id), conversion_fn);
        Ok(())
    }
    
    /// Register a conversion function from a custom type to a built-in type
    pub fn register_conversion_to_builtin(
        from_id: CustomTypeId,
        to_type: NpyType,
        conversion_fn: ConversionFunction,
    ) -> Result<(), CustomTypeError> {
        let mut registry = TypeRegistry::get();
        let reg = registry.as_mut().unwrap();
        
        // Verify source type exists
        if !reg.types.contains_key(&from_id) {
            return Err(CustomTypeError::InvalidType(
                format!("Source type ID {} not found", from_id)
            ));
        }
        
        reg.conversions_to_builtin.insert((from_id, to_type), conversion_fn);
        Ok(())
    }
    
    /// Convert data from one custom type to another
    pub fn convert(
        from_id: CustomTypeId,
        to_id: CustomTypeId,
        data: &[u8],
    ) -> Result<Vec<u8>, CustomTypeError> {
        let registry = TypeRegistry::get();
        let reg = registry.as_ref().unwrap();
        
        // Try direct conversion
        if let Some(conversion_fn) = reg.conversions.get(&(from_id, to_id)) {
            return conversion_fn(data);
        }
        
        // Try using the type's convert_from/convert_to methods
        // (This would require getting the type instance, which is currently limited)
        Err(CustomTypeError::InvalidType(
            format!("No conversion registered from type {} to type {}", from_id, to_id)
        ))
    }
    
    /// Convert data from a built-in type to a custom type
    pub fn convert_from_builtin(
        from_type: NpyType,
        to_id: CustomTypeId,
        data: &[u8],
    ) -> Result<Vec<u8>, CustomTypeError> {
        let registry = TypeRegistry::get();
        let reg = registry.as_ref().unwrap();
        
        // Try registered conversion
        if let Some(conversion_fn) = reg.conversions_from_builtin.get(&(from_type, to_id)) {
            return conversion_fn(data);
        }
        
        Err(CustomTypeError::InvalidType(
            format!("No conversion registered from built-in type {:?} to custom type {}", from_type, to_id)
        ))
    }
    
    /// Convert data from a custom type to a built-in type
    pub fn convert_to_builtin(
        from_id: CustomTypeId,
        to_type: NpyType,
        data: &[u8],
    ) -> Result<Vec<u8>, CustomTypeError> {
        let registry = TypeRegistry::get();
        let reg = registry.as_ref().unwrap();
        
        // Try registered conversion
        if let Some(conversion_fn) = reg.conversions_to_builtin.get(&(from_id, to_type)) {
            return conversion_fn(data);
        }
        
        Err(CustomTypeError::InvalidType(
            format!("No conversion registered from custom type {} to built-in type {:?}", from_id, to_type)
        ))
    }
}

/// Register a custom type
///
/// Returns the type ID assigned to this custom type
pub fn register_custom_type<T: CustomType + 'static>(
    custom_type: T,
) -> Result<CustomTypeId, CustomTypeError> {
    TypeRegistry::register(custom_type)
}

/// Get custom type ID by name
pub fn get_custom_type_id(name: &str) -> Option<CustomTypeId> {
    TypeRegistry::get_by_name(name)
}

/// Create a DType for a custom type
pub fn create_custom_dtype(
    type_id: CustomTypeId,
) -> Result<DType, CustomTypeError> {
    // Get type metadata from registry
    let metadata = TypeRegistry::get_metadata_by_id(type_id)
        .ok_or_else(|| CustomTypeError::InvalidType(
            format!("Custom type with ID {} not found in registry", type_id)
        ))?;
    
    // Create DType with custom type information
    Ok(DType::custom(
        type_id,
        metadata.itemsize,
        metadata.align,
        metadata.name,
    ))
}

/// Check if a custom type has an optimized operation
pub fn has_optimized_operation(type_id: CustomTypeId, op_name: &str) -> bool {
    // This would require accessing the type instance, which is currently limited
    // For now, return false - full implementation would check the type's optimized_operation
    let _ = (type_id, op_name);
    false
}

/// Execute an optimized operation for a custom type
pub fn execute_optimized_operation(
    type_id: CustomTypeId,
    op_name: &str,
    inputs: &[&[u8]],
    output: &mut [u8],
) -> Option<Result<(), CustomTypeError>> {
    // This would require accessing the type instance, which is currently limited
    // For now, return None - full implementation would call the type's optimized_operation
    let _ = (type_id, op_name, inputs, output);
    None
}


