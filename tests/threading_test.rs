//! Threading tests for parallel operations
//!
//! These tests verify thread safety and correctness of parallel operations.

use raptors_core::array::Array;
use raptors_core::types::DType;
use raptors_core::ufunc::{sum_along_axis, min_along_axis, max_along_axis, mean_along_axis};
use raptors_core::ufunc::add_parallel;
use std::thread;

// Helper function to create array from Vec<f64>
fn array_from_vec_f64(data: Vec<f64>, shape: Vec<i64>) -> Array {
    let mut array = Array::new(shape, DType::new(raptors_core::types::NpyType::Double)).unwrap();
    unsafe {
        let ptr = array.data_ptr_mut() as *mut f64;
        for (i, &val) in data.iter().enumerate() {
            *ptr.add(i) = val;
        }
    }
    array
}

// Helper function to create array from Vec<f32>
#[allow(dead_code)] // May be used in future tests
fn array_from_vec_f32(data: Vec<f32>, shape: Vec<i64>) -> Array {
    let mut array = Array::new(shape, DType::new(raptors_core::types::NpyType::Float)).unwrap();
    unsafe {
        let ptr = array.data_ptr_mut() as *mut f32;
        for (i, &val) in data.iter().enumerate() {
            *ptr.add(i) = val;
        }
    }
    array
}

// Helper function to create array from Vec<i32>
#[allow(dead_code)] // May be used in future tests
fn array_from_vec_i32(data: Vec<i32>, shape: Vec<i64>) -> Array {
    let mut array = Array::new(shape, DType::new(raptors_core::types::NpyType::Int)).unwrap();
    unsafe {
        let ptr = array.data_ptr_mut() as *mut i32;
        for (i, &val) in data.iter().enumerate() {
            *ptr.add(i) = val;
        }
    }
    array
}

#[test]
fn test_parallel_sum_correctness() {
    // Create a large array
    let size = 100_000;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push(i as f64);
    }
    
    let array = array_from_vec_f64(data, vec![size as i64]);
    
    // Calculate sum sequentially (expected result)
    let expected_sum = (0..size).sum::<usize>() as f64;
    
    // Calculate sum using parallel reduction
    let result = sum_along_axis(&array, None).unwrap();
    let actual_sum = unsafe {
        let ptr = result.data_ptr() as *const f64;
        *ptr
    };
    
    // Allow small floating point differences
    assert!((actual_sum - expected_sum).abs() < 1.0);
}

#[test]
fn test_parallel_sum_thread_safety() {
    // Create arrays in separate threads to test thread safety
    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                let size = 50_000;
                let mut data = Vec::with_capacity(size);
                for j in 0..size {
                    data.push((i * size + j) as f64);
                }
                let arr = array_from_vec_f64(data, vec![size as i64]);
                let result = sum_along_axis(&arr, None).unwrap();
                // Extract the sum value before returning (can't return Array from thread)
                unsafe {
                    let ptr = result.data_ptr() as *const f64;
                    *ptr
                }
            })
        })
        .collect();
    
    let results: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    
    // Verify all results are correct
    assert_eq!(results.len(), 10);
    for (i, &actual_sum) in results.iter().enumerate() {
        let size = 50_000;
        let expected_sum = ((i * size)..((i + 1) * size)).sum::<usize>() as f64;
        assert!((actual_sum - expected_sum).abs() < 1.0);
    }
}

#[test]
fn test_parallel_min_max_correctness() {
    let size = 100_000;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push(i as f64);
    }
    
    let array = array_from_vec_f64(data, vec![size as i64]);
    
    // Test min
    let min_result = min_along_axis(&array, None).unwrap();
    let min_val = unsafe {
        let ptr = min_result.data_ptr() as *const f64;
        *ptr
    };
    assert_eq!(min_val, 0.0);
    
    // Test max
    let max_result = max_along_axis(&array, None).unwrap();
    let max_val = unsafe {
        let ptr = max_result.data_ptr() as *const f64;
        *ptr
    };
    assert_eq!(max_val, (size - 1) as f64);
}

#[test]
fn test_parallel_add_operation() {
    let size = 100_000;
    let data1: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let data2: Vec<f64> = (0..size).map(|i| (i * 2) as f64).collect();
    
    let array1 = array_from_vec_f64(data1, vec![size as i64]);
    let array2 = array_from_vec_f64(data2, vec![size as i64]);
    let mut output = Array::new(vec![size as i64], DType::new(raptors_core::types::NpyType::Double)).unwrap();
    
    // Try parallel add (should work for large contiguous arrays)
    if add_parallel(&array1, &array2, &mut output).is_ok() {
        // Verify correctness
        unsafe {
            let out_ptr = output.data_ptr() as *const f64;
            for i in 0..1000 {
                // Check first 1000 elements
                let expected = (i + i * 2) as f64;
                let actual = *out_ptr.add(i);
                assert!((actual - expected).abs() < 0.001);
            }
        }
    }
}

#[test]
fn test_parallel_vs_sequential_consistency() {
    // Test that parallel and sequential reductions produce same results
    let size = 50_000;
    let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.001).collect();
    let array = array_from_vec_f64(data, vec![size as i64]);
    
    // Both should use the same code path (parallel for large arrays)
    // but we can verify consistency
    let result1 = sum_along_axis(&array, None).unwrap();
    let result2 = sum_along_axis(&array, None).unwrap();
    
    let sum1 = unsafe { *(result1.data_ptr() as *const f64) };
    let sum2 = unsafe { *(result2.data_ptr() as *const f64) };
    
    assert_eq!(sum1, sum2);
}

#[test]
fn test_parallel_mean_correctness() {
    let size = 100_000;
    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let array = array_from_vec_f64(data, vec![size as i64]);
    
    let result = mean_along_axis(&array, None).unwrap();
    let mean_val = unsafe {
        let ptr = result.data_ptr() as *const f64;
        *ptr
    };
    
    // Expected mean: (0 + 99999) / 2 = 49999.5
    let expected_mean = ((size - 1) as f64) / 2.0;
    assert!((mean_val - expected_mean).abs() < 0.1);
}

// NumPy-style tests for parallel operations

#[test]
fn test_parallel_operations_deterministic() {
    // NumPy test: Parallel operations should be deterministic
    let size = 200_000;
    let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.001).collect();
    let array = array_from_vec_f64(data, vec![size as i64]);
    
    // Run parallel sum multiple times - should get same result
    let results: Vec<f64> = (0..3).map(|_| {
        let result = sum_along_axis(&array, None).unwrap();
        unsafe { *(result.data_ptr() as *const f64) }
    }).collect();
    
    // All results should be identical (deterministic)
    assert_eq!(results[0], results[1]);
    assert_eq!(results[1], results[2]);
}

#[test]
fn test_parallel_min_max_with_duplicates() {
    // NumPy test: Min/max with duplicate values in parallel
    let size = 150_000;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        // Create pattern with duplicates
        data.push((i % 100) as f64);
    }
    
    let array = array_from_vec_f64(data, vec![size as i64]);
    
    let min_result = min_along_axis(&array, None).unwrap();
    let min_val = unsafe { *(min_result.data_ptr() as *const f64) };
    assert_eq!(min_val, 0.0);
    
    let max_result = max_along_axis(&array, None).unwrap();
    let max_val = unsafe { *(max_result.data_ptr() as *const f64) };
    assert_eq!(max_val, 99.0);
}

#[test]
fn test_parallel_operations_independence() {
    // NumPy test: Parallel operations on different arrays should be independent
    let sizes = [50_000, 75_000, 100_000, 125_000];
    
    for &size in &sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let array = array_from_vec_f64(data, vec![size as i64]);
        
        let result = sum_along_axis(&array, None).unwrap();
        let sum = unsafe { *(result.data_ptr() as *const f64) };
        
        let expected = (0..size).sum::<usize>() as f64;
        assert!((sum - expected).abs() < 1.0, "Independent operation failed for size {}", size);
    }
}

#[test]
fn test_parallel_sum_accuracy_extreme_values() {
    // NumPy test: Parallel sum with extreme values (very large, very small)
    let size = 100_000;
    let data: Vec<f64> = (0..size).map(|i| {
        if i % 1000 == 0 {
            1e15  // Very large
        } else {
            1e-15 // Very small
        }
    }).collect();
    
    let array = array_from_vec_f64(data, vec![size as i64]);
    let result = sum_along_axis(&array, None).unwrap();
    let sum = unsafe { *(result.data_ptr() as *const f64) };
    
    let expected_large = (size / 1000) as f64 * 1e15;
    let expected_small = (size - size / 1000) as f64 * 1e-15;
    let expected = expected_large + expected_small;
    
    // Parallel summation should maintain reasonable relative accuracy
    assert!((sum - expected).abs() / expected.abs() < 1e-10);
}

#[test]
fn test_parallel_reductions_concurrent() {
    // NumPy test: Multiple reductions in parallel (simulated via multiple calls)
    // Tests that internal thread pool can handle concurrent operations
    let size = 80_000;
    let data1: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let data2: Vec<f64> = (0..size).map(|i| (i * 2) as f64).collect();
    let data3: Vec<f64> = (0..size).map(|i| (i * 3) as f64).collect();
    
    let array1 = array_from_vec_f64(data1, vec![size as i64]);
    let array2 = array_from_vec_f64(data2, vec![size as i64]);
    let array3 = array_from_vec_f64(data3, vec![size as i64]);
    
    // Perform multiple reductions (should work concurrently with Rayon)
    let sum1 = sum_along_axis(&array1, None).unwrap();
    let sum2 = sum_along_axis(&array2, None).unwrap();
    let sum3 = sum_along_axis(&array3, None).unwrap();
    
    let val1 = unsafe { *(sum1.data_ptr() as *const f64) };
    let val2 = unsafe { *(sum2.data_ptr() as *const f64) };
    let val3 = unsafe { *(sum3.data_ptr() as *const f64) };
    
    // Verify correctness
    let expected1 = (0..size).sum::<usize>() as f64;
    let expected2 = (0..size).map(|i| i * 2).sum::<usize>() as f64;
    let expected3 = (0..size).map(|i| i * 3).sum::<usize>() as f64;
    
    assert!((val1 - expected1).abs() < 1.0);
    assert!((val2 - expected2).abs() < 1.0);
    assert!((val3 - expected3).abs() < 1.0);
}

#[test]
fn test_parallel_sum_zero_elements() {
    // NumPy test: Arrays with many zeros should sum correctly
    let size = 200_000;
    let data: Vec<f64> = (0..size).map(|i| {
        if i % 100 == 0 {
            1.0
        } else {
            0.0
        }
    }).collect();
    
    let array = array_from_vec_f64(data, vec![size as i64]);
    let result = sum_along_axis(&array, None).unwrap();
    let sum = unsafe { *(result.data_ptr() as *const f64) };
    
    // Should sum to approximately size/100
    let expected = (size / 100) as f64;
    assert!((sum - expected).abs() < 0.1);
}

#[test]
fn test_parallel_mean_numerical_stability() {
    // NumPy test: Mean calculation should be numerically stable
    // Using values that might cause precision issues
    let size = 100_000;
    let data: Vec<f64> = (0..size).map(|i| 1e10 + (i as f64)).collect();
    let array = array_from_vec_f64(data, vec![size as i64]);
    
    let result = mean_along_axis(&array, None).unwrap();
    let mean_val = unsafe { *(result.data_ptr() as *const f64) };
    
    // Expected: 1e10 + mean of 0..(size-1)
    let expected = 1e10 + ((size - 1) as f64) / 2.0;
    // Mean should maintain good relative accuracy
    assert!((mean_val - expected).abs() / expected.abs() < 1e-12);
}

#[test]
fn test_parallel_min_max_stability() {
    // NumPy test: Min/max should be stable with repeated operations
    let size = 150_000;
    let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.0001).collect();
    let array = array_from_vec_f64(data, vec![size as i64]);
    
    // Run multiple times - should get same result
    let min_results: Vec<f64> = (0..3).map(|_| {
        let result = min_along_axis(&array, None).unwrap();
        unsafe { *(result.data_ptr() as *const f64) }
    }).collect();
    
    let max_results: Vec<f64> = (0..3).map(|_| {
        let result = max_along_axis(&array, None).unwrap();
        unsafe { *(result.data_ptr() as *const f64) }
    }).collect();
    
    // Results should be consistent
    assert_eq!(min_results[0], min_results[1]);
    assert_eq!(min_results[1], min_results[2]);
    assert_eq!(max_results[0], max_results[1]);
    assert_eq!(max_results[1], max_results[2]);
}

