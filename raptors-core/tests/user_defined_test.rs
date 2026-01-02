//! Tests for user-defined types functionality

use raptors_core::types::{
    CustomType, CustomTypeError, register_custom_type, get_custom_type_id, create_custom_dtype,
    TypeRegistry, NpyType,
};

// Example custom type implementation
struct TestCustomType {
    itemsize: usize,
    align: usize,
    name: String,
}

impl CustomType for TestCustomType {
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
        Ok(bytes.to_vec())
    }
    
    fn to_bytes(&self, value: &[u8]) -> Result<Vec<u8>, CustomTypeError> {
        Ok(value.to_vec())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

#[test]
fn test_register_custom_type() {
    let custom_type = TestCustomType {
        itemsize: 16,
        align: 8,
        name: "TestType".to_string(),
    };
    
    let type_id = register_custom_type(custom_type).unwrap();
    
    // Type ID should be assigned
    assert!(type_id >= 1000); // Custom types start at 1000
}

#[test]
fn test_get_custom_type_id() {
    let custom_type = TestCustomType {
        itemsize: 16,
        align: 8,
        name: "GetTestType".to_string(),
    };
    
    let type_id = register_custom_type(custom_type).unwrap();
    
    // Should be able to retrieve by name
    let retrieved_id = get_custom_type_id("GetTestType");
    assert_eq!(retrieved_id, Some(type_id));
}

#[test]
fn test_register_duplicate_name() {
    let custom_type1 = TestCustomType {
        itemsize: 16,
        align: 8,
        name: "DuplicateType".to_string(),
    };
    
    let custom_type2 = TestCustomType {
        itemsize: 16,
        align: 8,
        name: "DuplicateType".to_string(),
    };
    
    let _type_id1 = register_custom_type(custom_type1).unwrap();
    
    // Should fail to register duplicate name
    let result = register_custom_type(custom_type2);
    assert!(result.is_err());
}

#[test]
fn test_custom_type_properties() {
    let custom_type = TestCustomType {
        itemsize: 24,
        align: 16,
        name: "PropertyTest".to_string(),
    };
    
    assert_eq!(custom_type.itemsize(), 24);
    assert_eq!(custom_type.align(), 16);
    assert_eq!(custom_type.name(), "PropertyTest");
}

#[test]
fn test_create_custom_dtype() {
    let custom_type = TestCustomType {
        itemsize: 16,
        align: 8,
        name: "DTypeTest".to_string(),
    };
    
    let type_id = register_custom_type(custom_type).unwrap();
    
    // Should now successfully create dtype
    let dtype = create_custom_dtype(type_id).unwrap();
    assert_eq!(dtype.itemsize(), 16);
    assert_eq!(dtype.align(), 8);
    assert_eq!(dtype.name(), "DTypeTest");
    assert_eq!(dtype.custom_type_id(), Some(type_id));
}

#[test]
fn test_create_custom_dtype_nonexistent() {
    // Should return error for non-existent type ID
    let result = create_custom_dtype(99999);
    assert!(result.is_err());
}

#[test]
fn test_get_nonexistent_type_id() {
    // Should return None for non-existent type
    let result = get_custom_type_id("NonExistentType");
    assert_eq!(result, None);
}

#[test]
fn test_multiple_custom_types() {
    let custom_type1 = TestCustomType {
        itemsize: 8,
        align: 8,
        name: "Type1".to_string(),
    };
    
    let custom_type2 = TestCustomType {
        itemsize: 16,
        align: 16,
        name: "Type2".to_string(),
    };
    
    let type_id1 = register_custom_type(custom_type1).unwrap();
    let type_id2 = register_custom_type(custom_type2).unwrap();
    
    // Should have different type IDs
    assert_ne!(type_id1, type_id2);
    
    // Both should be retrievable
    assert_eq!(get_custom_type_id("Type1"), Some(type_id1));
    assert_eq!(get_custom_type_id("Type2"), Some(type_id2));
}

#[test]
fn test_type_conversion_hooks() {
    let custom_type = TestCustomType {
        itemsize: 8,
        align: 8,
        name: "ConvertTest".to_string(),
    };
    
    let type_id = register_custom_type(custom_type).unwrap();
    
    // Test conversion from built-in type
    let conversion_fn = Box::new(|data: &[u8]| -> Result<Vec<u8>, CustomTypeError> {
        // Simple conversion: just copy the data
        Ok(data.to_vec())
    });
    
    let result = TypeRegistry::register_conversion_from_builtin(
        NpyType::Double,
        type_id,
        conversion_fn,
    );
    assert!(result.is_ok());
    
    // Test conversion execution
    let test_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let converted = TypeRegistry::convert_from_builtin(NpyType::Double, type_id, &test_data).unwrap();
    assert_eq!(converted, test_data);
}

#[test]
fn test_type_optimized_operation() {
    let custom_type = TestCustomType {
        itemsize: 8,
        align: 8,
        name: "OptimizedTest".to_string(),
    };
    
    // Test that optimized_operation returns None by default
    let result = custom_type.optimized_operation("add", &[], &mut []);
    assert!(result.is_none());
}

#[test]
fn test_type_metadata_storage() {
    let custom_type = TestCustomType {
        itemsize: 32,
        align: 16,
        name: "MetadataTest".to_string(),
    };
    
    let type_id = register_custom_type(custom_type).unwrap();
    
    // Retrieve metadata
    let metadata = TypeRegistry::get_metadata_by_id(type_id).unwrap();
    assert_eq!(metadata.itemsize, 32);
    assert_eq!(metadata.align, 16);
    assert_eq!(metadata.name, "MetadataTest");
}

#[test]
fn test_custom_dtype_integration() {
    let custom_type = TestCustomType {
        itemsize: 12,
        align: 4,
        name: "IntegrationTest".to_string(),
    };
    
    let type_id = register_custom_type(custom_type).unwrap();
    
    // Create dtype
    let dtype = create_custom_dtype(type_id).unwrap();
    
    // Verify dtype properties
    assert_eq!(dtype.itemsize(), 12);
    assert_eq!(dtype.align(), 4);
    assert_eq!(dtype.name(), "IntegrationTest");
    assert_eq!(dtype.custom_type_id(), Some(type_id));
    
    // Verify dtype can be used (basic check)
    assert!(dtype.custom_type_id().is_some());
}

