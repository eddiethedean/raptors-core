//! NumPy array creation tests
//!
//! Ported from NumPy's test_creation.py and test_array.py
//! Tests cover array constructors, shape variations, dtype variations, and edge cases

#![allow(unused_unsafe)]

mod numpy_port {
    pub mod helpers;
}

use numpy_port::helpers::*;
use raptors_core::array::Array;
use raptors_core::types::{DType, NpyType};
use raptors_core::{empty, zeros, ones};

// Test array() constructor equivalent - Array::new
#[test]
fn test_array_new_basic() {
    let arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[3, 4]);
    assert_eq!(arr.ndim(), 2);
    assert_eq!(arr.size(), 12);
}

#[test]
fn test_array_new_0d() {
    // 0-dimensional array (scalar)
    let arr = Array::new(vec![], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[] as &[i64]);
    assert_eq!(arr.ndim(), 0);
    assert_eq!(arr.size(), 1);
}

#[test]
fn test_array_new_1d() {
    let arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[5]);
    assert_eq!(arr.ndim(), 1);
    assert_eq!(arr.size(), 5);
}

#[test]
fn test_array_new_3d() {
    let arr = Array::new(vec![2, 3, 4], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4]);
    assert_eq!(arr.ndim(), 3);
    assert_eq!(arr.size(), 24);
}

#[test]
fn test_array_new_high_dimensional() {
    // Test higher dimensional arrays
    let arr = Array::new(vec![2, 2, 2, 2], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[2, 2, 2, 2]);
    assert_eq!(arr.ndim(), 4);
    assert_eq!(arr.size(), 16);
}

// Test zeros() - NumPy np.zeros
#[test]
fn test_zeros_basic() {
    let arr = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[3, 4]);
    assert_eq!(arr.size(), 12);
    
    // Verify all elements are zero
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        for i in 0..12 {
            assert_eq!(*ptr.add(i), 0.0);
        }
    }
}

#[test]
fn test_zeros_0d() {
    let arr = zeros(vec![], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[] as &[i64]);
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        assert_eq!(*ptr, 0.0);
    }
}

#[test]
fn test_zeros_different_dtypes() {
    // Test zeros with different dtypes
    let arr_int = zeros(vec![3], DType::new(NpyType::Int)).unwrap();
    assert_eq!(arr_int.dtype().type_(), NpyType::Int);
    
    let arr_float = zeros(vec![3], DType::new(NpyType::Float)).unwrap();
    assert_eq!(arr_float.dtype().type_(), NpyType::Float);
    
    let arr_bool = zeros(vec![3], DType::new(NpyType::Bool)).unwrap();
    assert_eq!(arr_bool.dtype().type_(), NpyType::Bool);
}

#[test]
fn test_zeros_empty_array() {
    // Array with zero-size dimension
    let arr = zeros(vec![0], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[0]);
    assert_eq!(arr.size(), 0);
}

#[test]
fn test_zeros_large_array() {
    // Test with larger array
    let arr = zeros(vec![100], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[100]);
    assert_eq!(arr.size(), 100);
    
    // Verify first and last elements are zero
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        assert_eq!(*ptr, 0.0);
        assert_eq!(*ptr.add(99), 0.0);
    }
}

// Test ones() - NumPy np.ones
#[test]
fn test_ones_basic() {
    let arr = ones(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[3, 4]);
    assert_eq!(arr.size(), 12);
    
    // Verify all elements are one
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        for i in 0..12 {
            assert!((*ptr.add(i) - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn test_ones_0d() {
    let arr = ones(vec![], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[] as &[i64]);
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        assert!((*ptr - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_ones_different_dtypes() {
    let arr_int = ones(vec![3], DType::new(NpyType::Int)).unwrap();
    assert_eq!(arr_int.dtype().type_(), NpyType::Int);
    unsafe {
        let ptr = arr_int.data_ptr() as *const i32;
        assert_eq!(*ptr, 1);
    }
    
    let arr_float = ones(vec![3], DType::new(NpyType::Float)).unwrap();
    assert_eq!(arr_float.dtype().type_(), NpyType::Float);
    unsafe {
        let ptr = arr_float.data_ptr() as *const f32;
        assert!((*ptr - 1.0).abs() < 1e-6);
    }
}

// Test empty() - NumPy np.empty
#[test]
fn test_empty_basic() {
    let arr = empty(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[3, 4]);
    assert_eq!(arr.size(), 12);
    // Note: empty() doesn't initialize memory, so we can't verify values
}

#[test]
fn test_empty_0d() {
    let arr = empty(vec![], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[] as &[i64]);
    assert_eq!(arr.size(), 1);
}

#[test]
fn test_empty_different_dtypes() {
    let arr_int = empty(vec![3], DType::new(NpyType::Int)).unwrap();
    assert_eq!(arr_int.dtype().type_(), NpyType::Int);
    
    let arr_float = empty(vec![3], DType::new(NpyType::Float)).unwrap();
    assert_eq!(arr_float.dtype().type_(), NpyType::Float);
}

// Test array properties
#[test]
fn test_array_shape_property() {
    let arr = Array::new(vec![2, 3, 4], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4]);
}

#[test]
fn test_array_ndim_property() {
    let arr0d = Array::new(vec![], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr0d.ndim(), 0);
    
    let arr1d = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr1d.ndim(), 1);
    
    let arr2d = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr2d.ndim(), 2);
}

#[test]
fn test_array_size_property() {
    let arr = Array::new(vec![2, 3, 4], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.size(), 24); // 2 * 3 * 4
}

#[test]
fn test_array_itemsize() {
    let arr_double = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr_double.itemsize(), 8); // f64 is 8 bytes
    
    let arr_float = Array::new(vec![3], DType::new(NpyType::Float)).unwrap();
    assert_eq!(arr_float.itemsize(), 4); // f32 is 4 bytes
    
    let arr_int = Array::new(vec![3], DType::new(NpyType::Int)).unwrap();
    assert_eq!(arr_int.itemsize(), 4); // i32 is 4 bytes
}

#[test]
fn test_array_dtype_property() {
    let arr = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.dtype().type_(), NpyType::Double);
    assert_eq!(arr.dtype().name(), "float64");
}

// Test edge cases
#[test]
fn test_array_zero_size_dimension() {
    // Array with one dimension of size 0
    let arr = Array::new(vec![0], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[0]);
    assert_eq!(arr.size(), 0);
}

#[test]
fn test_array_multiple_zero_dimensions() {
    // Array with multiple zero-size dimensions
    let arr = Array::new(vec![0, 5], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[0, 5]);
    assert_eq!(arr.size(), 0);
}

#[test]
fn test_array_single_element() {
    // Single element array
    let arr = Array::new(vec![1], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[1]);
    assert_eq!(arr.size(), 1);
}

#[test]
fn test_array_contiguity_new() {
    // New arrays should be C-contiguous
    let arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    assert!(arr.is_c_contiguous());
}

#[test]
fn test_array_contiguity_zeros() {
    let arr = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    assert!(arr.is_c_contiguous());
}

#[test]
fn test_array_contiguity_ones() {
    let arr = ones(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    assert!(arr.is_c_contiguous());
}

// Test dtype variations
#[test]
fn test_creation_all_numeric_dtypes() {
    let dtypes = vec![
        NpyType::Bool,
        NpyType::Byte,
        NpyType::UByte,
        NpyType::Short,
        NpyType::UShort,
        NpyType::Int,
        NpyType::UInt,
        NpyType::LongLong,
        NpyType::ULongLong,
        NpyType::Float,
        NpyType::Double,
    ];
    
    for dtype_enum in dtypes {
        let dtype = DType::new(dtype_enum);
        let arr = Array::new(vec![3], dtype).unwrap();
        assert_eq!(arr.dtype().type_(), dtype_enum);
    }
}

// Test from_slice equivalent patterns
#[test]
fn test_array_from_data_pattern() {
    // Test pattern similar to np.array([1, 2, 3])
    let data = vec![1.0, 2.0, 3.0];
    let arr = Array::from_slice(&data, vec![3], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[3]);
    
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        assert_eq!(*ptr.add(0), 1.0);
        assert_eq!(*ptr.add(1), 2.0);
        assert_eq!(*ptr.add(2), 3.0);
    }
}

#[test]
fn test_array_from_2d_data_pattern() {
    // Test pattern similar to np.array([[1, 2], [3, 4]])
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let arr = Array::from_slice(&data, vec![2, 2], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[2, 2]);
    
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        assert_eq!(*ptr.add(0), 1.0);
        assert_eq!(*ptr.add(1), 2.0);
        assert_eq!(*ptr.add(2), 3.0);
        assert_eq!(*ptr.add(3), 4.0);
    }
}

#[test]
fn test_array_very_large_shape() {
    // Test with reasonably large shape (not too large to avoid OOM)
    let arr = Array::new(vec![1000], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[1000]);
    assert_eq!(arr.size(), 1000);
}

// Test memory layout
#[test]
fn test_array_strides_c_contiguous() {
    let arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    let strides = arr.strides();
    // For C-contiguous [3, 4] array with f64 (8 bytes):
    // stride[0] = 4 * 8 = 32
    // stride[1] = 1 * 8 = 8
    assert_eq!(strides[0], 32);
    assert_eq!(strides[1], 8);
}

#[test]
fn test_array_strides_1d() {
    let arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    let strides = arr.strides();
    assert_eq!(strides[0], 8); // itemsize for f64
}

// Test with helpers
#[test]
fn test_creation_with_helpers() {
    use numpy_port::helpers::test_data;
    
    let arr = test_data::sequential(vec![5], DType::new(NpyType::Double));
    assert_eq!(arr.shape(), &[5]);
    
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        assert_eq!(*ptr.add(0), 0.0);
        assert_eq!(*ptr.add(4), 4.0);
    }
}

// Additional edge cases and variations

#[test]
fn test_array_creation_very_large_dimension() {
    // Test with very large single dimension
    let arr = Array::new(vec![10000], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[10000]);
    assert_eq!(arr.size(), 10000);
}

#[test]
fn test_array_creation_single_dimension_large() {
    // Test single large dimension
    let arr = Array::new(vec![1000], DType::new(NpyType::Int)).unwrap();
    assert_eq!(arr.shape(), &[1000]);
    assert_eq!(arr.size(), 1000);
}

#[test]
fn test_zeros_all_dtypes() {
    // Test zeros with all numeric dtypes
    let dtypes = vec![
        NpyType::Bool,
        NpyType::Byte,
        NpyType::UByte,
        NpyType::Short,
        NpyType::UShort,
        NpyType::Int,
        NpyType::UInt,
        NpyType::LongLong,
        NpyType::ULongLong,
        NpyType::Float,
        NpyType::Double,
    ];
    
    for dtype_enum in dtypes {
        let dtype = DType::new(dtype_enum);
        let arr = zeros(vec![5], dtype).unwrap();
        assert_eq!(arr.dtype().type_(), dtype_enum);
        assert_eq!(arr.size(), 5);
    }
}

#[test]
fn test_ones_all_dtypes() {
    // Test ones with all numeric dtypes
    let dtypes = vec![
        NpyType::Bool,
        NpyType::Int,
        NpyType::Float,
        NpyType::Double,
    ];
    
    for dtype_enum in dtypes {
        let dtype = DType::new(dtype_enum);
        let arr = ones(vec![5], dtype).unwrap();
        assert_eq!(arr.dtype().type_(), dtype_enum);
        assert_eq!(arr.size(), 5);
    }
}

#[test]
fn test_empty_all_dtypes() {
    // Test empty with all numeric dtypes
    let dtypes = vec![
        NpyType::Int,
        NpyType::Float,
        NpyType::Double,
    ];
    
    for dtype_enum in dtypes {
        let dtype = DType::new(dtype_enum);
        let arr = empty(vec![5], dtype).unwrap();
        assert_eq!(arr.dtype().type_(), dtype_enum);
        assert_eq!(arr.size(), 5);
    }
}

#[test]
fn test_array_creation_rectangular_2d() {
    // Test rectangular 2D arrays (non-square)
    let arr1 = Array::new(vec![2, 10], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr1.shape(), &[2, 10]);
    assert_eq!(arr1.size(), 20);
    
    let arr2 = Array::new(vec![10, 2], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr2.shape(), &[10, 2]);
    assert_eq!(arr2.size(), 20);
}

#[test]
fn test_array_creation_rectangular_3d() {
    // Test rectangular 3D arrays
    let arr = Array::new(vec![2, 3, 5], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 5]);
    assert_eq!(arr.size(), 30);
}

#[test]
fn test_array_creation_very_high_dimensional() {
    // Test arrays with many dimensions
    let arr = Array::new(vec![2, 2, 2, 2, 2], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[2, 2, 2, 2, 2]);
    assert_eq!(arr.ndim(), 5);
    assert_eq!(arr.size(), 32);
}

#[test]
fn test_zeros_multiple_zero_dimensions() {
    // Array with multiple zero-size dimensions
    let arr = zeros(vec![0, 0, 5], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[0, 0, 5]);
    assert_eq!(arr.size(), 0);
}

#[test]
fn test_ones_multiple_zero_dimensions() {
    // Array with multiple zero-size dimensions
    let arr = ones(vec![0, 0, 5], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[0, 0, 5]);
    assert_eq!(arr.size(), 0);
}

#[test]
fn test_array_creation_memory_alignment() {
    // Test that arrays are properly aligned
    let arr = Array::new(vec![10], DType::new(NpyType::Double)).unwrap();
    let ptr = arr.data_ptr() as usize;
    // For f64, alignment should be 8 bytes
    assert_eq!(ptr % 8, 0);
}

#[test]
fn test_array_creation_dtype_names() {
    // Test dtype names for all types
    let test_cases = vec![
        (NpyType::Bool, "bool"),
        (NpyType::Double, "float64"),
        (NpyType::Float, "float32"),
        (NpyType::Int, "int32"),
        (NpyType::LongLong, "int64"),
    ];
    
    for (npy_type, expected_name) in test_cases {
        let dtype = DType::new(npy_type);
        let arr = Array::new(vec![1], dtype).unwrap();
        assert_eq!(arr.dtype().name(), expected_name);
    }
}

#[test]
fn test_array_creation_itemsize_consistency() {
    // Test that itemsize matches dtype
    let test_cases = vec![
        (NpyType::Double, 8),
        (NpyType::Float, 4),
        (NpyType::Int, 4),
        (NpyType::LongLong, 8),
    ];
    
    for (npy_type, expected_itemsize) in test_cases {
        let dtype = DType::new(npy_type);
        let arr = Array::new(vec![1], dtype).unwrap();
        assert_eq!(arr.itemsize(), expected_itemsize);
        assert_eq!(arr.dtype().itemsize(), expected_itemsize);
    }
}

#[test]
fn test_array_creation_writeable_flag() {
    // New arrays should be writeable
    let arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    assert!(arr.is_writeable());
}

#[test]
fn test_array_creation_owns_data_flag() {
    // New arrays should own their data
    let arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    assert!(arr.owns_data());
}

#[test]
fn test_zeros_writeable() {
    // Zeros arrays should be writeable
    let arr = zeros(vec![5], DType::new(NpyType::Double)).unwrap();
    assert!(arr.is_writeable());
}

#[test]
fn test_ones_writeable() {
    // Ones arrays should be writeable
    let arr = ones(vec![5], DType::new(NpyType::Double)).unwrap();
    assert!(arr.is_writeable());
}

#[test]
fn test_empty_writeable() {
    // Empty arrays should be writeable
    let arr = empty(vec![5], DType::new(NpyType::Double)).unwrap();
    assert!(arr.is_writeable());
}

#[test]
fn test_array_creation_from_slice_all_dtypes() {
    // Test from_slice with different dtypes
    let int_data = vec![1i32, 2, 3, 4, 5];
    let arr_int = Array::from_slice(&int_data, vec![5], DType::new(NpyType::Int)).unwrap();
    assert_eq!(arr_int.dtype().type_(), NpyType::Int);
    
    let float_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let arr_float = Array::from_slice(&float_data, vec![5], DType::new(NpyType::Float)).unwrap();
    assert_eq!(arr_float.dtype().type_(), NpyType::Float);
}

#[test]
fn test_array_creation_from_slice_2d() {
    // Test from_slice with 2D shape
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = Array::from_slice(&data, vec![2, 3], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[2, 3]);
    assert_eq!(arr.size(), 6);
}

#[test]
fn test_array_creation_from_slice_3d() {
    // Test from_slice with 3D shape
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let arr = Array::from_slice(&data, vec![2, 2, 2], DType::new(NpyType::Double)).unwrap();
    assert_eq!(arr.shape(), &[2, 2, 2]);
    assert_eq!(arr.size(), 8);
}

#[test]
fn test_array_creation_strides_2d() {
    // Test strides for 2D array
    let arr = Array::new(vec![4, 3], DType::new(NpyType::Double)).unwrap();
    let strides = arr.strides();
    // For C-contiguous [4, 3] with f64 (8 bytes):
    // stride[0] = 3 * 8 = 24
    // stride[1] = 1 * 8 = 8
    assert_eq!(strides[0], 24);
    assert_eq!(strides[1], 8);
}

#[test]
fn test_array_creation_strides_3d() {
    // Test strides for 3D array
    let arr = Array::new(vec![2, 3, 4], DType::new(NpyType::Double)).unwrap();
    let strides = arr.strides();
    // For C-contiguous [2, 3, 4] with f64 (8 bytes):
    // stride[0] = 3 * 4 * 8 = 96
    // stride[1] = 4 * 8 = 32
    // stride[2] = 1 * 8 = 8
    assert_eq!(strides[0], 96);
    assert_eq!(strides[1], 32);
    assert_eq!(strides[2], 8);
}

#[test]
fn test_array_creation_ndim_consistency() {
    // Test that ndim matches shape length
    let test_shapes = vec![
        vec![],
        vec![5],
        vec![3, 4],
        vec![2, 3, 4],
        vec![2, 2, 2, 2],
    ];
    
    for shape in test_shapes {
        let arr = Array::new(shape.clone(), DType::new(NpyType::Double)).unwrap();
        assert_eq!(arr.ndim(), shape.len());
    }
}

#[test]
fn test_array_creation_size_calculation() {
    // Test that size is calculated correctly
    let test_cases = vec![
        (vec![], 1),
        (vec![5], 5),
        (vec![3, 4], 12),
        (vec![2, 3, 4], 24),
        (vec![2, 2, 2, 2], 16),
    ];
    
    for (shape, expected_size) in test_cases {
        let arr = Array::new(shape, DType::new(NpyType::Double)).unwrap();
        assert_eq!(arr.size(), expected_size);
    }
}

#[test]
fn test_array_creation_zero_size_edge_cases() {
    // Test various zero-size edge cases
    let test_shapes = vec![
        vec![0],
        vec![0, 5],
        vec![5, 0],
        vec![0, 0],
        vec![0, 0, 5],
    ];
    
    for shape in test_shapes {
        let arr = Array::new(shape.clone(), DType::new(NpyType::Double)).unwrap();
        assert_eq!(arr.shape(), &shape);
        // Size should be 0 if any dimension is 0
        let has_zero = shape.iter().any(|&s| s == 0);
        if has_zero {
            assert_eq!(arr.size(), 0);
        }
    }
}

#[test]
fn test_zeros_verify_all_elements() {
    // Verify all elements in zeros array are actually zero
    let arr = zeros(vec![100], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        for i in 0..100 {
            assert_eq!(*ptr.add(i), 0.0);
        }
    }
}

#[test]
fn test_ones_verify_all_elements() {
    // Verify all elements in ones array are actually one
    let arr = ones(vec![100], DType::new(NpyType::Double)).unwrap();
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        for i in 0..100 {
            assert!((*ptr.add(i) - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn test_array_creation_dtype_clone() {
    // Test that dtype can be cloned and used
    let dtype = DType::new(NpyType::Double);
    let arr1 = Array::new(vec![5], dtype.clone()).unwrap();
    let arr2 = Array::new(vec![5], dtype).unwrap();
    assert_eq!(arr1.dtype().type_(), arr2.dtype().type_());
}

#[test]
fn test_array_creation_contiguity_flags() {
    // Test contiguity flags for new arrays
    let arr = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    assert!(arr.is_c_contiguous());
    // New arrays may or may not be F-contiguous depending on implementation
}

#[test]
fn test_array_creation_flags_consistency() {
    // Test that flags are consistent
    let arr = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    let flags = arr.flags();
    assert!(flags.contains(raptors_core::array::ArrayFlags::C_CONTIGUOUS));
    assert!(flags.contains(raptors_core::array::ArrayFlags::OWNDATA));
    assert!(flags.contains(raptors_core::array::ArrayFlags::WRITEABLE));
}


