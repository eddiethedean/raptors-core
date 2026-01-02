//! NumPy string array tests
//!
//! Ported from NumPy's test_strings.py
//! Tests cover string operations, encoding, formatting

#![allow(unused_unsafe)]

mod numpy_port {
    pub mod helpers;
}

use numpy_port::helpers::*;
use raptors_core::array::Array;
use raptors_core::types::{DType, NpyType};
use raptors_core::string::{create_string_array, is_string_array, get_string};
use raptors_core::string::{str_upper, str_lower};
use raptors_core::string::{Encoding, convert_encoding, validate_encoding, is_valid_utf8};

// String array creation tests

#[test]
fn test_string_array_creation_basic() {
    let data = vec!["hello".to_string(), "world".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    assert_eq!(arr.shape(), &[2]);
    assert!(is_string_array(&arr));
}

#[test]
fn test_string_array_creation_1d() {
    let data = vec![
        "a".to_string(),
        "bb".to_string(),
        "ccc".to_string(),
    ];
    let arr = create_string_array(data, vec![3]).unwrap();
    
    assert_eq!(arr.shape(), &[3]);
    assert!(is_string_array(&arr));
}

#[test]
fn test_string_array_creation_2d() {
    let data = vec![
        "a".to_string(), "b".to_string(),
        "c".to_string(), "d".to_string(),
    ];
    let arr = create_string_array(data, vec![2, 2]).unwrap();
    
    assert_eq!(arr.shape(), &[2, 2]);
    assert!(is_string_array(&arr));
}

#[test]
fn test_string_array_fixed_width() {
    let data = vec!["short".to_string(), "verylongstring".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    // Itemsize should be at least the length of the longest string
    assert!(arr.itemsize() >= 14);
}

#[test]
fn test_string_array_empty_strings() {
    let data = vec!["".to_string(), "test".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    assert_eq!(arr.shape(), &[2]);
    assert!(is_string_array(&arr));
}

#[test]
fn test_string_array_single_element() {
    let data = vec!["hello".to_string()];
    let arr = create_string_array(data, vec![1]).unwrap();
    
    assert_eq!(arr.shape(), &[1]);
    assert!(is_string_array(&arr));
}

// String access tests

#[test]
fn test_get_string_basic() {
    let data = vec!["hello".to_string(), "world".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    let s0 = get_string(&arr, 0).unwrap();
    assert!(s0.starts_with("hello"));
    
    let s1 = get_string(&arr, 1).unwrap();
    assert!(s1.starts_with("world"));
}

#[test]
fn test_get_string_out_of_bounds() {
    let data = vec!["test".to_string()];
    let arr = create_string_array(data, vec![1]).unwrap();
    
    let result = get_string(&arr, 10);
    assert!(result.is_err());
}

#[test]
fn test_get_string_non_string_array() {
    let arr = Array::new(vec![5], DType::new(NpyType::Int)).unwrap();
    
    let result = get_string(&arr, 0);
    assert!(result.is_err());
}

// String formatting tests

#[test]
fn test_str_upper_basic() {
    let data = vec!["hello".to_string(), "world".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    let result = str_upper(&arr).unwrap();
    assert_eq!(result.shape(), &[2]);
    
    let s0 = get_string(&result, 0).unwrap();
    assert_eq!(s0, "HELLO");
    
    let s1 = get_string(&result, 1).unwrap();
    assert_eq!(s1, "WORLD");
}

#[test]
fn test_str_lower_basic() {
    let data = vec!["HELLO".to_string(), "WORLD".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    let result = str_lower(&arr).unwrap();
    assert_eq!(result.shape(), &[2]);
    
    let s0 = get_string(&result, 0).unwrap();
    assert_eq!(s0, "hello");
    
    let s1 = get_string(&result, 1).unwrap();
    assert_eq!(s1, "world");
}

#[test]
fn test_str_upper_mixed_case() {
    let data = vec!["Hello".to_string(), "WoRlD".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    let result = str_upper(&arr).unwrap();
    
    let s0 = get_string(&result, 0).unwrap();
    assert_eq!(s0, "HELLO");
    
    let s1 = get_string(&result, 1).unwrap();
    assert_eq!(s1, "WORLD");
}

#[test]
fn test_str_lower_mixed_case() {
    let data = vec!["Hello".to_string(), "WoRlD".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    let result = str_lower(&arr).unwrap();
    
    let s0 = get_string(&result, 0).unwrap();
    assert_eq!(s0, "hello");
    
    let s1 = get_string(&result, 1).unwrap();
    assert_eq!(s1, "world");
}

// String encoding tests

#[test]
fn test_is_string_array_true() {
    let data = vec!["test".to_string()];
    let arr = create_string_array(data, vec![1]).unwrap();
    
    assert!(is_string_array(&arr));
}

#[test]
fn test_is_string_array_false() {
    let arr = Array::new(vec![5], DType::new(NpyType::Int)).unwrap();
    
    assert!(!is_string_array(&arr));
}

#[test]
fn test_validate_encoding_utf8() {
    let data = vec!["hello".to_string(), "world".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    let result = validate_encoding(&arr, Encoding::Utf8);
    assert!(result.is_ok());
}

#[test]
fn test_is_valid_utf8() {
    let data = vec!["hello".to_string(), "world".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    assert!(is_valid_utf8(&arr));
}

#[test]
fn test_convert_encoding_same() {
    let data = vec!["test".to_string()];
    let arr = create_string_array(data, vec![1]).unwrap();
    
    let result = convert_encoding(&arr, Encoding::Utf8, Encoding::Utf8).unwrap();
    assert_eq!(result.shape(), arr.shape());
}

// Edge cases

#[test]
fn test_string_array_empty() {
    let data = vec![];
    let arr = create_string_array(data, vec![0]).unwrap();
    
    assert_eq!(arr.shape(), &[0]);
    assert!(is_string_array(&arr));
}

#[test]
fn test_string_array_unicode() {
    let data = vec!["hello".to_string(), "世界".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    assert_eq!(arr.shape(), &[2]);
    assert!(is_string_array(&arr));
}

#[test]
fn test_string_array_special_characters() {
    let data = vec!["hello\nworld".to_string(), "tab\there".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    assert_eq!(arr.shape(), &[2]);
    assert!(is_string_array(&arr));
}

#[test]
fn test_string_array_very_long() {
    let long_string = "a".repeat(1000);
    let data = vec![long_string.clone()];
    let arr = create_string_array(data, vec![1]).unwrap();
    
    assert_eq!(arr.shape(), &[1]);
    assert!(arr.itemsize() >= 1000);
}

#[test]
fn test_string_array_different_lengths() {
    let data = vec![
        "a".to_string(),
        "bb".to_string(),
        "ccc".to_string(),
        "dddd".to_string(),
    ];
    let arr = create_string_array(data, vec![4]).unwrap();
    
    // Itemsize should accommodate longest string
    assert!(arr.itemsize() >= 4);
}

// Consistency tests

#[test]
fn test_string_array_consistency() {
    let data = vec!["test".to_string()];
    let arr1 = create_string_array(data.clone(), vec![1]).unwrap();
    let arr2 = create_string_array(data, vec![1]).unwrap();
    
    assert_eq!(arr1.shape(), arr2.shape());
    assert_eq!(arr1.itemsize(), arr2.itemsize());
}

#[test]
fn test_str_upper_lower_roundtrip() {
    let data = vec!["Hello".to_string()];
    let arr = create_string_array(data, vec![1]).unwrap();
    
    let upper = str_upper(&arr).unwrap();
    let lower = str_lower(&upper).unwrap();
    
    let s = get_string(&lower, 0).unwrap();
    assert_eq!(s, "hello");
}

// Test with helpers

#[test]
fn test_string_array_with_helpers() {
    let data = vec!["test1".to_string(), "test2".to_string()];
    let arr = create_string_array(data, vec![2]).unwrap();
    
    assert_eq!(arr.shape(), &[2]);
    assert!(is_string_array(&arr));
}

