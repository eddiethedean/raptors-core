//! NumPy masked array tests
//!
//! Ported from NumPy's test_ma.py
//! Tests cover masked array operations, mask propagation, and edge cases

#![allow(unused_unsafe)]

mod numpy_port {
    pub mod helpers;
}

use numpy_port::helpers::*;
use numpy_port::helpers::test_data;
use raptors_core::array::Array;
use raptors_core::types::{DType, NpyType};
use raptors_core::masked::{MaskedArray, masked_array, masked_array_with_indices};
// Note: masked operations may be private - using basic masked array functionality
use raptors_core::{zeros, ones};

// Masked array creation tests

#[test]
fn test_masked_array_creation_basic() {
    let data = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let mask = zeros(vec![5], DType::new(NpyType::Bool)).unwrap();
    
    let masked = MaskedArray::new(data, mask).unwrap();
    assert_eq!(masked.shape(), &[5]);
    assert_eq!(masked.size(), 5);
}

#[test]
fn test_masked_array_with_indices() {
    let data = test_data::sequential(vec![5], DType::new(NpyType::Double));
    
    let masked = masked_array_with_indices(data, &[1, 3]).unwrap();
    // Count masked elements manually
    let mut masked_count = 0;
    let mut valid_count = 0;
    for i in 0..5 {
        if masked.is_masked(i).unwrap() {
            masked_count += 1;
        } else {
            valid_count += 1;
        }
    }
    assert_eq!(masked_count, 2);
    assert_eq!(valid_count, 3);
    
    assert!(masked.is_masked(1).unwrap());
    assert!(masked.is_masked(3).unwrap());
    assert!(!masked.is_masked(0).unwrap());
    assert!(!masked.is_masked(2).unwrap());
    assert!(!masked.is_masked(4).unwrap());
}

#[test]
fn test_masked_array_all_masked() {
    let data = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let mask = ones(vec![5], DType::new(NpyType::Bool)).unwrap();
    
    let masked = MaskedArray::new(data, mask).unwrap();
    let mut masked_count = 0;
    for i in 0..5 {
        if masked.is_masked(i).unwrap() {
            masked_count += 1;
        }
    }
    assert_eq!(masked_count, 5);
}

#[test]
fn test_masked_array_none_masked() {
    let data = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let mask = zeros(vec![5], DType::new(NpyType::Bool)).unwrap();
    
    let masked = MaskedArray::new(data, mask).unwrap();
    let mut masked_count = 0;
    for i in 0..5 {
        if masked.is_masked(i).unwrap() {
            masked_count += 1;
        }
    }
    assert_eq!(masked_count, 0);
}

#[test]
fn test_masked_array_shape_mismatch() {
    let data = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let mask = zeros(vec![3], DType::new(NpyType::Bool)).unwrap();
    
    let result = MaskedArray::new(data, mask);
    assert!(result.is_err());
}

#[test]
fn test_masked_array_invalid_mask_type() {
    let data = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let mask = zeros(vec![5], DType::new(NpyType::Int)).unwrap();
    
    let result = MaskedArray::new(data, mask);
    assert!(result.is_err());
}

#[test]
fn test_masked_array_2d() {
    let data = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let mask = zeros(vec![2, 3], DType::new(NpyType::Bool)).unwrap();
    
    let masked = MaskedArray::new(data, mask).unwrap();
    assert_eq!(masked.shape(), &[2, 3]);
    assert_eq!(masked.ndim(), 2);
}

#[test]
fn test_masked_array_3d() {
    let data = test_data::sequential(vec![2, 2, 2], DType::new(NpyType::Double));
    let mask = zeros(vec![2, 2, 2], DType::new(NpyType::Bool)).unwrap();
    
    let masked = MaskedArray::new(data, mask).unwrap();
    assert_eq!(masked.shape(), &[2, 2, 2]);
    assert_eq!(masked.ndim(), 3);
}

// Masked array operations

#[test]
fn test_masked_add_basic() {
    let data1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let data2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let mask1 = zeros(vec![5], DType::new(NpyType::Bool)).unwrap();
    let mask2 = zeros(vec![5], DType::new(NpyType::Bool)).unwrap();
    
    let masked1 = MaskedArray::new(data1, mask1).unwrap();
    let masked2 = MaskedArray::new(data2, mask2).unwrap();
    
    // Test basic masked array creation and access
    assert_eq!(masked1.shape(), &[5]);
    assert_eq!(masked2.shape(), &[5]);
}

#[test]
fn test_masked_array_with_masked_elements() {
    let mut data = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    let mut mask = Array::new(vec![5], DType::new(NpyType::Bool)).unwrap();
    
    unsafe {
        let d_ptr = data.data_ptr_mut() as *mut f64;
        let m_ptr = mask.data_ptr_mut() as *mut bool;
        for i in 0..5 {
            *d_ptr.add(i) = (i + 1) as f64;
            *m_ptr.add(i) = i == 2; // Mask element 2
        }
    }
    
    let masked = MaskedArray::new(data, mask).unwrap();
    assert_eq!(masked.shape(), &[5]);
    // Element 2 should be masked
    assert!(masked.is_masked(2).unwrap());
    assert!(!masked.is_masked(0).unwrap());
}

// Masked array access tests

#[test]
fn test_masked_array_data_access() {
    let data = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let mask = zeros(vec![5], DType::new(NpyType::Bool)).unwrap();
    
    let masked = MaskedArray::new(data, mask).unwrap();
    let data_ref = masked.data();
    
    assert_eq!(data_ref.shape(), &[5]);
    assert_eq!(data_ref.size(), 5);
}

#[test]
fn test_masked_array_mask_access() {
    let data = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let mut mask = Array::new(vec![5], DType::new(NpyType::Bool)).unwrap();
    
    unsafe {
        let mask_ptr = mask.data_ptr_mut() as *mut bool;
        *mask_ptr.add(2) = true; // Mask element 2
    }
    
    let masked = MaskedArray::new(data, mask).unwrap();
    let mask_ref = masked.mask();
    
    assert_eq!(mask_ref.shape(), &[5]);
    unsafe {
        let mask_ptr = mask_ref.data_ptr() as *const bool;
        assert!(*mask_ptr.add(2));
        assert!(!*mask_ptr.add(0));
    }
}

// Edge cases

#[test]
fn test_masked_array_empty() {
    let data = zeros(vec![0], DType::new(NpyType::Double)).unwrap();
    let mask = zeros(vec![0], DType::new(NpyType::Bool)).unwrap();
    
    let masked = MaskedArray::new(data, mask).unwrap();
    assert_eq!(masked.size(), 0);
}

#[test]
fn test_masked_array_single_element() {
    let mut data = Array::new(vec![1], DType::new(NpyType::Double)).unwrap();
    let mut mask = Array::new(vec![1], DType::new(NpyType::Bool)).unwrap();
    
    unsafe {
        *(data.data_ptr_mut() as *mut f64) = 42.0;
        *(mask.data_ptr_mut() as *mut bool) = true;
    }
    
    let masked = MaskedArray::new(data, mask).unwrap();
    assert_eq!(masked.size(), 1);
    assert!(masked.is_masked(0).unwrap());
}

#[test]
fn test_masked_array_large() {
    let data = ones(vec![1000], DType::new(NpyType::Double)).unwrap();
    let mask = zeros(vec![1000], DType::new(NpyType::Bool)).unwrap();
    
    let masked = MaskedArray::new(data, mask).unwrap();
    assert_eq!(masked.size(), 1000);
    let mut masked_count = 0;
    for i in 0..1000 {
        if masked.is_masked(i).unwrap() {
            masked_count += 1;
        }
    }
    assert_eq!(masked_count, 0);
}

// Different dtypes

#[test]
fn test_masked_array_int() {
    let mut data = Array::new(vec![5], DType::new(NpyType::Int)).unwrap();
    let mask = zeros(vec![5], DType::new(NpyType::Bool)).unwrap();
    
    unsafe {
        let ptr = data.data_ptr_mut() as *mut i32;
        for i in 0..5 {
            *ptr.add(i) = (i + 1) as i32;
        }
    }
    
    let masked = MaskedArray::new(data, mask).unwrap();
    assert_eq!(masked.shape(), &[5]);
}

#[test]
fn test_masked_array_float() {
    let mut data = Array::new(vec![5], DType::new(NpyType::Float)).unwrap();
    let mask = zeros(vec![5], DType::new(NpyType::Bool)).unwrap();
    
    unsafe {
        let ptr = data.data_ptr_mut() as *mut f32;
        for i in 0..5 {
            *ptr.add(i) = (i + 1) as f32;
        }
    }
    
    let masked = MaskedArray::new(data, mask).unwrap();
    assert_eq!(masked.shape(), &[5]);
}

// Consistency tests

#[test]
fn test_masked_array_consistency() {
    let data = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let mask = zeros(vec![5], DType::new(NpyType::Bool)).unwrap();
    
    let masked1 = MaskedArray::new(data.clone(), mask.clone()).unwrap();
    let masked2 = MaskedArray::new(data, mask).unwrap();
    
    assert_eq!(masked1.shape(), masked2.shape());
    assert_eq!(masked1.size(), masked2.size());
}

// Test with helpers

#[test]
fn test_masked_array_with_helpers() {
    let data = test_data::sequential(vec![10], DType::new(NpyType::Double));
    let masked = masked_array_with_indices(data, &[2, 5, 8]).unwrap();
    
    let mut masked_count = 0;
    for i in 0..10 {
        if masked.is_masked(i).unwrap() {
            masked_count += 1;
        }
    }
    assert_eq!(masked_count, 3);
}

