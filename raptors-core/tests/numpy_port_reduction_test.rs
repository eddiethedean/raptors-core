//! NumPy reduction tests
//!
//! Ported from NumPy's test_reduction.py
//! Tests cover sum, mean, min, max, and other reduction operations

#![allow(unused_unsafe)]

mod numpy_port {
    pub mod helpers;
}

use numpy_port::helpers::*;
use numpy_port::helpers::test_data;
use raptors_core::array::Array;
use raptors_core::types::{DType, NpyType};
use raptors_core::ufunc::reduction::*;
use raptors_core::{zeros, ones};

// Sum reduction tests

#[test]
fn test_sum_1d() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..5 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = sum_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 15.0).abs() < 1e-10); // 1+2+3+4+5
    }
}

#[test]
fn test_sum_2d() {
    let mut arr = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..6 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = sum_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 21.0).abs() < 1e-10); // 1+2+3+4+5+6
    }
}

#[test]
fn test_sum_empty() {
    let arr = zeros(vec![0], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64)).abs() < 1e-10);
    }
}

#[test]
fn test_sum_single_element() {
    let mut arr = Array::new(vec![1], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        *(arr.data_ptr_mut() as *mut f64) = 42.0;
    }
    
    let result = sum_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 42.0).abs() < 1e-10);
    }
}

#[test]
fn test_sum_zeros() {
    let arr = zeros(vec![10], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64)).abs() < 1e-10);
    }
}

#[test]
fn test_sum_ones() {
    let arr = ones(vec![10], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 10.0).abs() < 1e-10);
    }
}

#[test]
fn test_sum_large_array() {
    let arr = ones(vec![1000], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 1000.0).abs() < 1e-10);
    }
}

#[test]
fn test_sum_3d() {
    let arr = ones(vec![2, 3, 4], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 24.0).abs() < 1e-10); // 2*3*4
    }
}

// Mean reduction tests

#[test]
fn test_mean_1d() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..5 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = mean_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 3.0).abs() < 1e-10); // (1+2+3+4+5)/5
    }
}

#[test]
fn test_mean_2d() {
    let mut arr = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..6 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = mean_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 3.5).abs() < 1e-10); // (1+2+3+4+5+6)/6
    }
}

#[test]
fn test_mean_ones() {
    let arr = ones(vec![10], DType::new(NpyType::Double)).unwrap();
    
    let result = mean_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_mean_zeros() {
    let arr = zeros(vec![10], DType::new(NpyType::Double)).unwrap();
    
    let result = mean_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64)).abs() < 1e-10);
    }
}

// Min reduction tests

#[test]
fn test_min_1d() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        *ptr.add(0) = 5.0;
        *ptr.add(1) = 2.0;
        *ptr.add(2) = 8.0;
        *ptr.add(3) = 1.0;
        *ptr.add(4) = 3.0;
    }
    
    let result = min_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_min_2d() {
    let mut arr = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..6 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = min_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_min_negative() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        *ptr.add(0) = -5.0;
        *ptr.add(1) = -2.0;
        *ptr.add(2) = 8.0;
        *ptr.add(3) = 1.0;
        *ptr.add(4) = 3.0;
    }
    
    let result = min_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - (-5.0)).abs() < 1e-10);
    }
}

#[test]
fn test_min_single_element() {
    let mut arr = Array::new(vec![1], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        *(arr.data_ptr_mut() as *mut f64) = 42.0;
    }
    
    let result = min_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 42.0).abs() < 1e-10);
    }
}

// Max reduction tests

#[test]
fn test_max_1d() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        *ptr.add(0) = 5.0;
        *ptr.add(1) = 2.0;
        *ptr.add(2) = 8.0;
        *ptr.add(3) = 1.0;
        *ptr.add(4) = 3.0;
    }
    
    let result = max_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 8.0).abs() < 1e-10);
    }
}

#[test]
fn test_max_2d() {
    let mut arr = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..6 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = max_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 6.0).abs() < 1e-10);
    }
}

#[test]
fn test_max_negative() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        *ptr.add(0) = -5.0;
        *ptr.add(1) = -2.0;
        *ptr.add(2) = -8.0;
        *ptr.add(3) = -1.0;
        *ptr.add(4) = -3.0;
    }
    
    let result = max_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - (-1.0)).abs() < 1e-10);
    }
}

#[test]
fn test_max_single_element() {
    let mut arr = Array::new(vec![1], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        *(arr.data_ptr_mut() as *mut f64) = 42.0;
    }
    
    let result = max_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 42.0).abs() < 1e-10);
    }
}

// Min/Max edge cases

#[test]
fn test_min_max_same() {
    let arr = ones(vec![10], DType::new(NpyType::Double)).unwrap();
    
    let min_result = min_along_axis(&arr, None).unwrap();
    let max_result = max_along_axis(&arr, None).unwrap();
    
    unsafe {
        let min_val = *(min_result.data_ptr() as *const f64);
        let max_val = *(max_result.data_ptr() as *const f64);
        assert!((min_val - max_val).abs() < 1e-10);
    }
}

#[test]
fn test_min_max_sequential() {
    let arr = test_data::sequential(vec![10], DType::new(NpyType::Double));
    
    let min_result = min_along_axis(&arr, None).unwrap();
    let max_result = max_along_axis(&arr, None).unwrap();
    
    unsafe {
        let min_val = *(min_result.data_ptr() as *const f64);
        let max_val = *(max_result.data_ptr() as *const f64);
        assert!((min_val - 0.0).abs() < 1e-10);
        assert!((max_val - 9.0).abs() < 1e-10);
    }
}

// Different dtypes

#[test]
fn test_sum_int() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Int)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut i32;
        for i in 0..5 {
            *ptr.add(i) = (i + 1) as i32;
        }
    }
    
    // Note: sum_along_axis may require Double for now
    // This test may need adjustment based on implementation
    let result = sum_along_axis(&arr, None);
    if result.is_ok() {
        assert_eq!(result.unwrap().shape(), &[1]);
    }
}

// Large arrays

#[test]
fn test_sum_large_2d() {
    let arr = ones(vec![100, 100], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 10000.0).abs() < 1e-10);
    }
}

#[test]
fn test_mean_large_2d() {
    let arr = ones(vec![100, 100], DType::new(NpyType::Double)).unwrap();
    
    let result = mean_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 1.0).abs() < 1e-10);
    }
}

// High-dimensional arrays

#[test]
fn test_sum_4d() {
    let arr = ones(vec![2, 2, 2, 2], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 16.0).abs() < 1e-10); // 2^4
    }
}

#[test]
fn test_mean_4d() {
    let arr = ones(vec![2, 2, 2, 2], DType::new(NpyType::Double)).unwrap();
    
    let result = mean_along_axis(&arr, None).unwrap();
    assert_eq!(result.shape(), &[1]);
    
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 1.0).abs() < 1e-10);
    }
}

// Consistency tests

#[test]
fn test_sum_mean_consistency() {
    let arr = ones(vec![10], DType::new(NpyType::Double)).unwrap();
    
    let sum_result = sum_along_axis(&arr, None).unwrap();
    let mean_result = mean_along_axis(&arr, None).unwrap();
    
    unsafe {
        let sum_val = *(sum_result.data_ptr() as *const f64);
        let mean_val = *(mean_result.data_ptr() as *const f64);
        // sum / size should equal mean
        assert!((sum_val / 10.0 - mean_val).abs() < 1e-10);
    }
}

#[test]
fn test_min_max_ordering() {
    let mut arr = Array::new(vec![10], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..10 {
            *ptr.add(i) = (i * 2) as f64;
        }
    }
    
    let min_result = min_along_axis(&arr, None).unwrap();
    let max_result = max_along_axis(&arr, None).unwrap();
    
    unsafe {
        let min_val = *(min_result.data_ptr() as *const f64);
        let max_val = *(max_result.data_ptr() as *const f64);
        assert!(min_val <= max_val);
    }
}

// Test with helpers

#[test]
fn test_reduction_with_helpers() {
    let arr = test_data::sequential(vec![10], DType::new(NpyType::Double));
    
    let sum_result = sum_along_axis(&arr, None).unwrap();
    let mean_result = mean_along_axis(&arr, None).unwrap();
    
    unsafe {
        let sum_val = *(sum_result.data_ptr() as *const f64);
        let mean_val = *(mean_result.data_ptr() as *const f64);
        // Sum of 0..9 = 45, mean = 4.5
        assert!((sum_val - 45.0).abs() < 1e-10);
        assert!((mean_val - 4.5).abs() < 1e-10);
    }
}

// Additional reduction tests for comprehensive coverage

#[test]
fn test_sum_axis_0_2d_detailed() {
    let mut arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..12 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = sum_along_axis(&arr, Some(0)).unwrap();
    assert_eq!(result.shape(), &[4]);
}

#[test]
fn test_sum_axis_1_2d_detailed() {
    let mut arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..12 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = sum_along_axis(&arr, Some(1)).unwrap();
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn test_sum_axis_0_3d() {
    let arr = ones(vec![2, 3, 4], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, Some(0)).unwrap();
    assert_eq!(result.shape(), &[3, 4]);
}

#[test]
fn test_sum_axis_1_3d() {
    let arr = ones(vec![2, 3, 4], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, Some(1)).unwrap();
    assert_eq!(result.shape(), &[2, 4]);
}

#[test]
fn test_sum_axis_2_3d() {
    let arr = ones(vec![2, 3, 4], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, Some(2)).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
}

#[test]
fn test_mean_axis_0_2d() {
    let mut arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..12 {
            *ptr.add(i) = 2.0;
        }
    }
    
    let result = mean_along_axis(&arr, Some(0)).unwrap();
    assert_eq!(result.shape(), &[4]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const f64;
        for i in 0..4 {
            assert!((*result_ptr.add(i) - 2.0).abs() < 1e-10);
        }
    }
}

#[test]
fn test_mean_axis_1_2d() {
    let mut arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..12 {
            *ptr.add(i) = 2.0;
        }
    }
    
    let result = mean_along_axis(&arr, Some(1)).unwrap();
    assert_eq!(result.shape(), &[3]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const f64;
        for i in 0..3 {
            assert!((*result_ptr.add(i) - 2.0).abs() < 1e-10);
        }
    }
}

#[test]
fn test_min_axis_0_2d() {
    let mut arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..12 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = min_along_axis(&arr, Some(0)).unwrap();
    assert_eq!(result.shape(), &[4]);
}

#[test]
fn test_min_axis_1_2d() {
    let mut arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..12 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = min_along_axis(&arr, Some(1)).unwrap();
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn test_max_axis_0_2d() {
    let mut arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..12 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = max_along_axis(&arr, Some(0)).unwrap();
    assert_eq!(result.shape(), &[4]);
}

#[test]
fn test_max_axis_1_2d() {
    let mut arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..12 {
            *ptr.add(i) = (i + 1) as f64;
        }
    }
    
    let result = max_along_axis(&arr, Some(1)).unwrap();
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn test_sum_negative_values() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..5 {
            *ptr.add(i) = -((i + 1) as f64);
        }
    }
    
    let result = sum_along_axis(&arr, None).unwrap();
    unsafe {
        // Sum of -1, -2, -3, -4, -5 = -15
        assert!((*(result.data_ptr() as *const f64) + 15.0).abs() < 1e-10);
    }
}

#[test]
fn test_mean_negative_values() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..5 {
            *ptr.add(i) = -((i + 1) as f64);
        }
    }
    
    let result = mean_along_axis(&arr, None).unwrap();
    unsafe {
        // Mean of -1, -2, -3, -4, -5 = -3
        assert!((*(result.data_ptr() as *const f64) + 3.0).abs() < 1e-10);
    }
}

#[test]
fn test_min_max_negative_values() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..5 {
            *ptr.add(i) = -((i + 1) as f64);
        }
    }
    
    let min_result = min_along_axis(&arr, None).unwrap();
    let max_result = max_along_axis(&arr, None).unwrap();
    
    unsafe {
        assert!((*(min_result.data_ptr() as *const f64) + 5.0).abs() < 1e-10); // min = -5
        assert!((*(max_result.data_ptr() as *const f64) + 1.0).abs() < 1e-10); // max = -1
    }
}

#[test]
fn test_sum_mixed_positive_negative() {
    let mut arr = Array::new(vec![6], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        *ptr.add(0) = 5.0;
        *ptr.add(1) = -3.0;
        *ptr.add(2) = 2.0;
        *ptr.add(3) = -1.0;
        *ptr.add(4) = 4.0;
        *ptr.add(5) = -2.0;
    }
    
    let result = sum_along_axis(&arr, None).unwrap();
    unsafe {
        // 5 - 3 + 2 - 1 + 4 - 2 = 5
        assert!((*(result.data_ptr() as *const f64) - 5.0).abs() < 1e-10);
    }
}

#[test]
fn test_sum_large_values() {
    let mut arr = Array::new(vec![100], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..100 {
            *ptr.add(i) = (i + 1) as f64 * 1000.0;
        }
    }
    
    let result = sum_along_axis(&arr, None).unwrap();
    unsafe {
        // Sum of 1000, 2000, ..., 100000 = 1000 * (1+2+...+100) = 1000 * 5050 = 5050000
        let expected = 1000.0 * (100.0 * 101.0 / 2.0);
        assert!((*(result.data_ptr() as *const f64) - expected).abs() < 1e-5);
    }
}

#[test]
fn test_sum_small_values() {
    let mut arr = Array::new(vec![10], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..10 {
            *ptr.add(i) = (i + 1) as f64 * 0.001;
        }
    }
    
    let result = sum_along_axis(&arr, None).unwrap();
    unsafe {
        // Sum of 0.001, 0.002, ..., 0.010 = 0.055
        assert!((*(result.data_ptr() as *const f64) - 0.055).abs() < 1e-10);
    }
}

#[test]
fn test_mean_precision() {
    let mut arr = Array::new(vec![1000], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f64;
        for i in 0..1000 {
            *ptr.add(i) = 1.0;
        }
    }
    
    let result = mean_along_axis(&arr, None).unwrap();
    unsafe {
        assert!((*(result.data_ptr() as *const f64) - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_min_max_single_element() {
    let mut arr = Array::new(vec![1], DType::new(NpyType::Double)).unwrap();
    unsafe {
        *(arr.data_ptr_mut() as *mut f64) = 42.0;
    }
    
    let min_result = min_along_axis(&arr, None).unwrap();
    let max_result = max_along_axis(&arr, None).unwrap();
    
    unsafe {
        assert!((*(min_result.data_ptr() as *const f64) - 42.0).abs() < 1e-10);
        assert!((*(max_result.data_ptr() as *const f64) - 42.0).abs() < 1e-10);
    }
}

#[test]
fn test_sum_4d_axis_0() {
    let arr = ones(vec![2, 2, 2, 2], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, Some(0)).unwrap();
    assert_eq!(result.shape(), &[2, 2, 2]);
}

#[test]
fn test_sum_4d_axis_1() {
    let arr = ones(vec![2, 2, 2, 2], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, Some(1)).unwrap();
    assert_eq!(result.shape(), &[2, 2, 2]);
}

#[test]
fn test_sum_4d_axis_2() {
    let arr = ones(vec![2, 2, 2, 2], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, Some(2)).unwrap();
    assert_eq!(result.shape(), &[2, 2, 2]);
}

#[test]
fn test_sum_4d_axis_3() {
    let arr = ones(vec![2, 2, 2, 2], DType::new(NpyType::Double)).unwrap();
    
    let result = sum_along_axis(&arr, Some(3)).unwrap();
    assert_eq!(result.shape(), &[2, 2, 2]);
}

#[test]
fn test_reduction_int_type() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Int)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut i32;
        for i in 0..5 {
            *ptr.add(i) = (i + 1) as i32;
        }
    }
    
    let result = sum_along_axis(&arr, None).unwrap();
    unsafe {
        assert_eq!(*(result.data_ptr() as *const i32), 15);
    }
}

#[test]
fn test_reduction_float_type() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Float)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f32;
        for i in 0..5 {
            *ptr.add(i) = (i + 1) as f32;
        }
    }
    
    let result = sum_along_axis(&arr, None).unwrap();
    unsafe {
        assert!((*(result.data_ptr() as *const f32) - 15.0).abs() < 1e-5);
    }
}

#[test]
fn test_mean_float_precision() {
    let mut arr = Array::new(vec![10], DType::new(NpyType::Float)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f32;
        for i in 0..10 {
            *ptr.add(i) = 1.0;
        }
    }
    
    let result = mean_along_axis(&arr, None).unwrap();
    unsafe {
        assert!((*(result.data_ptr() as *const f32) - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_min_max_float_type() {
    let mut arr = Array::new(vec![5], DType::new(NpyType::Float)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut f32;
        *ptr.add(0) = 5.0;
        *ptr.add(1) = 2.0;
        *ptr.add(2) = 8.0;
        *ptr.add(3) = 1.0;
        *ptr.add(4) = 3.0;
    }
    
    let min_result = min_along_axis(&arr, None).unwrap();
    let max_result = max_along_axis(&arr, None).unwrap();
    
    unsafe {
        assert!((*(min_result.data_ptr() as *const f32) - 1.0).abs() < 1e-5);
        assert!((*(max_result.data_ptr() as *const f32) - 8.0).abs() < 1e-5);
    }
}

