//! Tests for array subclassing functionality

use raptors_core::array::{Array, ArrayBase, SubclassableArray, CustomArray, isinstance};
use raptors_core::types::{DType, NpyType};

#[test]
fn test_subclassable_array_creation() {
    let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    let subclassable = SubclassableArray::new(array, "TestArray");
    
    assert_eq!(subclassable.type_name(), "TestArray");
    assert_eq!(subclassable.shape(), &[3, 4]);
}

#[test]
fn test_isinstance() {
    let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    let subclassable = SubclassableArray::new(array, "TestArray");
    
    assert!(subclassable.isinstance("TestArray"));
    assert!(!subclassable.isinstance("OtherArray"));
}

#[test]
fn test_custom_array() {
    let array = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    let custom = CustomArray::new(array, "test metadata".to_string());
    
    assert_eq!(custom.type_name(), "CustomArray");
    assert_eq!(custom.metadata(), "test metadata");
    assert!(custom.isinstance("CustomArray"));
    assert!(custom.isinstance("ArrayBase"));
}

#[test]
fn test_array_base_trait() {
    let array = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    let subclassable = SubclassableArray::new(array, "BaseTest");
    
    // Test trait methods
    assert_eq!(subclassable.ndim(), 1);
    assert_eq!(subclassable.size(), 5);
    assert_eq!(subclassable.dtype().type_(), NpyType::Double);
}

#[test]
fn test_mro() {
    let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    let subclassable = SubclassableArray::new(array, "MROTest");
    
    let mro = subclassable.mro();
    assert!(mro.contains("MROTest"));
}

#[test]
fn test_isinstance_function() {
    let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    let subclassable = SubclassableArray::new(array, "FunctionTest");
    
    assert!(isinstance(&subclassable, "FunctionTest"));
    assert!(!isinstance(&subclassable, "OtherType"));
}

