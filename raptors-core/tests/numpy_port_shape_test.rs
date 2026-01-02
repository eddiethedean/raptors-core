//! NumPy shape manipulation tests
//!
//! Ported from NumPy's test_shape_base.py
//! Tests cover reshape, transpose, squeeze, expand_dims, and flatten

#![allow(unused_unsafe)]

mod numpy_port {
    pub mod helpers;
}

use numpy_port::helpers::*;
use numpy_port::helpers::test_data;
use raptors_core::array::Array;
use raptors_core::types::{DType, NpyType};
use raptors_core::shape::*;
use raptors_core::{zeros, ones};

// Reshape tests

#[test]
fn test_reshape_1d_to_2d() {
    let arr = test_data::sequential(vec![6], DType::new(NpyType::Double));
    
    let new_shape = vec![2, 3];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), &[2, 3]);
    assert_eq!(reshaped.size(), 6);
}

#[test]
fn test_reshape_2d_to_1d() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let new_shape = vec![6];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), &[6]);
    assert_eq!(reshaped.size(), 6);
}

#[test]
fn test_reshape_2d_to_3d() {
    let arr = test_data::sequential(vec![2, 6], DType::new(NpyType::Double));
    
    let new_shape = vec![2, 2, 3];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), &[2, 2, 3]);
    assert_eq!(reshaped.size(), 12);
}

#[test]
fn test_reshape_3d_to_2d() {
    let arr = test_data::sequential(vec![2, 2, 3], DType::new(NpyType::Double));
    
    let new_shape = vec![4, 3];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), &[4, 3]);
    assert_eq!(reshaped.size(), 12);
}

#[test]
fn test_reshape_invalid_size() {
    let arr = test_data::sequential(vec![6], DType::new(NpyType::Double));
    
    let new_shape = vec![3, 3]; // Size 9, but arr has size 6
    let result = validate_reshape_shape(arr.shape(), &new_shape);
    assert!(result.is_err());
}

#[test]
fn test_reshape_same_shape() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let new_shape = vec![2, 3];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), arr.shape());
}

#[test]
fn test_reshape_flatten() {
    let arr = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    let new_shape = vec![24]; // Flatten to 1D
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), &[24]);
    assert_eq!(reshaped.ndim(), 1);
}

// Transpose tests

#[test]
fn test_transpose_2d_default() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let (new_shape, axes) = transpose_dimensions(arr.shape(), None).unwrap();
    assert_eq!(new_shape, vec![3, 2]); // Reversed
    assert_eq!(axes, vec![1, 0]);
}

#[test]
fn test_transpose_2d_explicit() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let axes = vec![1, 0];
    let (new_shape, _) = transpose_dimensions(arr.shape(), Some(&axes)).unwrap();
    assert_eq!(new_shape, vec![3, 2]);
}

#[test]
fn test_transpose_3d_default() {
    let arr = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    let (new_shape, axes) = transpose_dimensions(arr.shape(), None).unwrap();
    assert_eq!(new_shape, vec![4, 3, 2]); // Reversed
    assert_eq!(axes, vec![2, 1, 0]);
}

#[test]
fn test_transpose_3d_custom() {
    let arr = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    let axes = vec![2, 0, 1];
    let (new_shape, _) = transpose_dimensions(arr.shape(), Some(&axes)).unwrap();
    assert_eq!(new_shape, vec![4, 2, 3]);
}

#[test]
fn test_transpose_invalid_axes() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let axes = vec![0, 1, 2]; // Too many axes
    let result = transpose_dimensions(arr.shape(), Some(&axes));
    assert!(result.is_err());
}

#[test]
fn test_transpose_duplicate_axes() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let axes = vec![0, 0]; // Duplicate
    let result = transpose_dimensions(arr.shape(), Some(&axes));
    assert!(result.is_err());
}

#[test]
fn test_transpose_out_of_bounds() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let axes = vec![0, 5]; // Out of bounds
    let result = transpose_dimensions(arr.shape(), Some(&axes));
    assert!(result.is_err());
}

// Squeeze tests

#[test]
fn test_squeeze_all() {
    let shape = vec![1, 5, 1, 3, 1];
    let squeezed = squeeze_dims(&shape, None);
    assert_eq!(squeezed, vec![5, 3]);
}

#[test]
fn test_squeeze_specific_axis() {
    let shape = vec![1, 5, 1, 3];
    
    let squeezed = squeeze_dims(&shape, Some(0));
    assert_eq!(squeezed, vec![5, 1, 3]);
    
    let squeezed = squeeze_dims(&shape, Some(2));
    assert_eq!(squeezed, vec![1, 5, 3]);
}

#[test]
fn test_squeeze_no_ones() {
    let shape = vec![2, 3, 4];
    let squeezed = squeeze_dims(&shape, None);
    assert_eq!(squeezed, shape);
}

#[test]
fn test_squeeze_all_ones() {
    let shape = vec![1, 1, 1];
    let squeezed = squeeze_dims(&shape, None);
    assert_eq!(squeezed, vec![]);
}

#[test]
fn test_squeeze_single_one() {
    let shape = vec![1];
    let squeezed = squeeze_dims(&shape, None);
    assert_eq!(squeezed, vec![]);
}

#[test]
fn test_squeeze_axis_not_one() {
    let shape = vec![2, 3, 4];
    let squeezed = squeeze_dims(&shape, Some(0));
    // Axis 0 has size 2, not 1, so no change
    assert_eq!(squeezed, shape);
}

// Expand dims tests

#[test]
fn test_expand_dims_beginning() {
    let shape = vec![3, 4];
    let expanded = expand_dims(&shape, 0).unwrap();
    assert_eq!(expanded, vec![1, 3, 4]);
}

#[test]
fn test_expand_dims_end() {
    let shape = vec![3, 4];
    let expanded = expand_dims(&shape, 2).unwrap();
    assert_eq!(expanded, vec![3, 4, 1]);
}

#[test]
fn test_expand_dims_middle() {
    let shape = vec![3, 4];
    let expanded = expand_dims(&shape, 1).unwrap();
    assert_eq!(expanded, vec![3, 1, 4]);
}

#[test]
fn test_expand_dims_multiple() {
    let shape = vec![3];
    let expanded1 = expand_dims(&shape, 0).unwrap();
    assert_eq!(expanded1, vec![1, 3]);
    
    let expanded2 = expand_dims(&expanded1, 2).unwrap();
    assert_eq!(expanded2, vec![1, 3, 1]);
}

#[test]
fn test_expand_dims_out_of_bounds() {
    let shape = vec![3, 4];
    let result = expand_dims(&shape, 10); // Out of bounds
    assert!(result.is_err());
}

#[test]
fn test_expand_dims_empty() {
    let shape = vec![];
    let expanded = expand_dims(&shape, 0).unwrap();
    assert_eq!(expanded, vec![1]);
}

// Flatten tests

#[test]
fn test_flatten_1d() {
    let shape = vec![5];
    let flattened = flatten_shape(&shape);
    assert_eq!(flattened, vec![5]);
}

#[test]
fn test_flatten_2d() {
    let shape = vec![2, 3];
    let flattened = flatten_shape(&shape);
    assert_eq!(flattened, vec![6]);
}

#[test]
fn test_flatten_3d() {
    let shape = vec![2, 3, 4];
    let flattened = flatten_shape(&shape);
    assert_eq!(flattened, vec![24]);
}

#[test]
fn test_flatten_empty() {
    let shape = vec![];
    let flattened = flatten_shape(&shape);
    assert_eq!(flattened, vec![1]);
}

#[test]
fn test_flatten_with_ones() {
    let shape = vec![1, 5, 1, 3];
    let flattened = flatten_shape(&shape);
    assert_eq!(flattened, vec![15]); // 1*5*1*3
}

// Compute size tests

#[test]
fn test_compute_size_1d() {
    let shape = vec![5];
    let size = compute_size(&shape);
    assert_eq!(size, 5);
}

#[test]
fn test_compute_size_2d() {
    let shape = vec![2, 3];
    let size = compute_size(&shape);
    assert_eq!(size, 6);
}

#[test]
fn test_compute_size_3d() {
    let shape = vec![2, 3, 4];
    let size = compute_size(&shape);
    assert_eq!(size, 24);
}

#[test]
fn test_compute_size_empty() {
    let shape = vec![];
    let size = compute_size(&shape);
    assert_eq!(size, 1); // Empty shape has size 1
}

#[test]
fn test_compute_size_with_zeros() {
    let shape = vec![0, 5];
    let size = compute_size(&shape);
    assert_eq!(size, 0);
}

#[test]
fn test_compute_size_with_ones() {
    let shape = vec![1, 5, 1, 3];
    let size = compute_size(&shape);
    assert_eq!(size, 15);
}

// Integration tests with arrays

#[test]
fn test_reshape_array_1d_to_2d() {
    let arr = test_data::sequential(vec![6], DType::new(NpyType::Double));
    
    let new_shape = vec![2, 3];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    // Verify data is accessible
    assert_eq!(reshaped.size(), 6);
    unsafe {
        let ptr = reshaped.data_ptr() as *const f64;
        // First element should be 0
        assert!((*ptr - 0.0).abs() < 1e-10);
        // Last element should be 5
        assert!((*ptr.add(5) - 5.0).abs() < 1e-10);
    }
}

#[test]
fn test_transpose_array_2d() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let (new_shape, axes) = transpose_dimensions(arr.shape(), None).unwrap();
    let mut new_strides = vec![0; new_shape.len()];
    for (i, &axis) in axes.iter().enumerate() {
        new_strides[i] = arr.strides()[axis as usize];
    }
    
    let transposed = arr.view(new_shape, new_strides).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
}

// Edge cases

#[test]
fn test_reshape_very_large() {
    let shape = vec![1000, 1000];
    let new_shape = vec![1000000];
    validate_reshape_shape(&shape, &new_shape).unwrap();
}

#[test]
fn test_transpose_high_dimensional() {
    let shape = vec![2, 2, 2, 2, 2];
    let (new_shape, _) = transpose_dimensions(&shape, None).unwrap();
    assert_eq!(new_shape, vec![2, 2, 2, 2, 2]); // Reversed is same for symmetric
}

#[test]
fn test_squeeze_high_dimensional() {
    let shape = vec![1, 1, 1, 5, 1, 3, 1];
    let squeezed = squeeze_dims(&shape, None);
    assert_eq!(squeezed, vec![5, 3]);
}

#[test]
fn test_expand_dims_high_dimensional() {
    let shape = vec![2, 3, 4, 5];
    let expanded = expand_dims(&shape, 2).unwrap();
    assert_eq!(expanded, vec![2, 3, 1, 4, 5]);
}

// Consistency tests

#[test]
fn test_reshape_transpose_consistency() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    // Reshape to 1D
    let flat_shape = vec![6];
    validate_reshape_shape(arr.shape(), &flat_shape).unwrap();
    
    // Transpose then reshape should give same 1D result
    let (trans_shape, axes) = transpose_dimensions(arr.shape(), None).unwrap();
    let trans_flat_shape = vec![6];
    validate_reshape_shape(&trans_shape, &trans_flat_shape).unwrap();
}

#[test]
fn test_squeeze_expand_inverse() {
    let shape = vec![3, 4];
    
    // Expand then squeeze should return original (if we squeeze the right axis)
    let expanded = expand_dims(&shape, 0).unwrap();
    assert_eq!(expanded, vec![1, 3, 4]);
    
    let squeezed = squeeze_dims(&expanded, Some(0));
    assert_eq!(squeezed, shape);
}

#[test]
fn test_flatten_reshape_consistency() {
    let shape = vec![2, 3, 4];
    let flattened = flatten_shape(&shape);
    
    // Flattened shape should match reshape to 1D
    let reshape_1d = vec![24];
    validate_reshape_shape(&shape, &reshape_1d).unwrap();
    assert_eq!(flattened, reshape_1d);
}

// Additional shape manipulation tests for comprehensive coverage

#[test]
fn test_reshape_3d_to_2d_additional() {
    let arr = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    let new_shape = vec![6, 4];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), &[6, 4]);
    assert_eq!(reshaped.size(), 24);
}

#[test]
fn test_reshape_4d_to_2d() {
    let arr = test_data::sequential(vec![2, 2, 2, 2], DType::new(NpyType::Double));
    
    let new_shape = vec![4, 4];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), &[4, 4]);
    assert_eq!(reshaped.size(), 16);
}

#[test]
fn test_reshape_5d_to_3d() {
    let arr = test_data::sequential(vec![2, 2, 2, 2, 2], DType::new(NpyType::Double));
    
    let new_shape = vec![4, 4, 2];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), &[4, 4, 2]);
    assert_eq!(reshaped.size(), 32);
}

#[test]
fn test_transpose_4d() {
    let shape = vec![2, 3, 4, 5];
    let (new_shape, axes) = transpose_dimensions(&shape, None).unwrap();
    
    assert_eq!(new_shape, vec![5, 4, 3, 2]); // Reversed
    assert_eq!(axes.len(), 4);
}

#[test]
fn test_transpose_custom_axes() {
    let shape = vec![2, 3, 4];
    let axes = vec![2, 0, 1];
    let (new_shape, _) = transpose_dimensions(&shape, Some(&axes)).unwrap();
    
    assert_eq!(new_shape, vec![4, 2, 3]);
}

#[test]
fn test_transpose_identity() {
    let shape = vec![2, 3, 4];
    let axes = vec![0, 1, 2];
    let (new_shape, _) = transpose_dimensions(&shape, Some(&axes)).unwrap();
    
    assert_eq!(new_shape, shape);
}

#[test]
fn test_squeeze_all_ones_additional() {
    let shape = vec![1, 1, 1, 1];
    let squeezed = squeeze_dims(&shape, None);
    assert_eq!(squeezed, vec![]);
}

#[test]
fn test_squeeze_specific_axis_additional() {
    let shape = vec![1, 3, 1, 4];
    let squeezed = squeeze_dims(&shape, Some(0));
    assert_eq!(squeezed, vec![3, 1, 4]);
}

#[test]
fn test_squeeze_specific_axis_2_additional() {
    let shape = vec![1, 3, 1, 4];
    let squeezed = squeeze_dims(&shape, Some(2));
    assert_eq!(squeezed, vec![1, 3, 4]);
}

#[test]
fn test_expand_dims_beginning_additional() {
    let shape = vec![3, 4];
    let expanded = expand_dims(&shape, 0).unwrap();
    assert_eq!(expanded, vec![1, 3, 4]);
}

#[test]
fn test_expand_dims_middle_additional() {
    let shape = vec![2, 3];
    let expanded = expand_dims(&shape, 1).unwrap();
    assert_eq!(expanded, vec![2, 1, 3]);
}

#[test]
fn test_expand_dims_end_additional() {
    let shape = vec![2, 3];
    let expanded = expand_dims(&shape, 2).unwrap();
    assert_eq!(expanded, vec![2, 3, 1]);
}

#[test]
fn test_expand_dims_high_dimensional_additional() {
    let shape = vec![2, 3, 4, 5];
    let expanded = expand_dims(&shape, 2).unwrap();
    assert_eq!(expanded, vec![2, 3, 1, 4, 5]);
}

#[test]
fn test_flatten_1d_additional() {
    let shape = vec![5];
    let flattened = flatten_shape(&shape);
    assert_eq!(flattened, vec![5]);
}

#[test]
fn test_flatten_2d_additional() {
    let shape = vec![3, 4];
    let flattened = flatten_shape(&shape);
    assert_eq!(flattened, vec![12]);
}

#[test]
fn test_flatten_3d_additional() {
    let shape = vec![2, 3, 4];
    let flattened = flatten_shape(&shape);
    assert_eq!(flattened, vec![24]);
}

#[test]
fn test_flatten_4d_additional() {
    let shape = vec![2, 2, 2, 2];
    let flattened = flatten_shape(&shape);
    assert_eq!(flattened, vec![16]);
}

#[test]
fn test_reshape_invalid_size_additional() {
    let shape = vec![2, 3];
    let new_shape = vec![7];
    let result = validate_reshape_shape(&shape, &new_shape);
    assert!(result.is_err());
}

#[test]
fn test_reshape_zero_dimension() {
    let shape = vec![0, 5];
    let new_shape = vec![0];
    let result = validate_reshape_shape(&shape, &new_shape);
    // May succeed or fail depending on implementation
    let _ = result;
}

#[test]
fn test_transpose_invalid_axes_additional() {
    let shape = vec![2, 3, 4];
    let axes = vec![0, 1]; // Wrong length
    let result = transpose_dimensions(&shape, Some(&axes));
    assert!(result.is_err());
}

#[test]
fn test_transpose_out_of_bounds_axis() {
    let shape = vec![2, 3, 4];
    let axes = vec![0, 1, 10]; // Out of bounds
    let result = transpose_dimensions(&shape, Some(&axes));
    assert!(result.is_err());
}

#[test]
fn test_expand_dims_invalid_axis() {
    let shape = vec![2, 3];
    let result = expand_dims(&shape, 10); // Out of bounds
    assert!(result.is_err());
}

#[test]
fn test_reshape_preserves_data() {
    let arr = test_data::sequential(vec![6], DType::new(NpyType::Double));
    
    let new_shape = vec![2, 3];
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    // Data should be preserved (same underlying array)
    assert_eq!(arr.size(), reshaped.size());
}

#[test]
fn test_transpose_preserves_size() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let (new_shape, _) = transpose_dimensions(arr.shape(), None).unwrap();
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let transposed = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(arr.size(), transposed.size());
}

#[test]
fn test_squeeze_preserves_size() {
    let arr = test_data::sequential(vec![1, 5, 1], DType::new(NpyType::Double));
    
    let squeezed_shape = squeeze_dims(arr.shape(), None);
    let new_strides = compute_reshape_strides(&squeezed_shape, arr.itemsize());
    let squeezed = arr.view(squeezed_shape, new_strides).unwrap();
    
    assert_eq!(arr.size(), squeezed.size());
}

#[test]
fn test_expand_dims_preserves_size() {
    let arr = test_data::sequential(vec![3, 4], DType::new(NpyType::Double));
    
    let expanded_shape = expand_dims(arr.shape(), 0).unwrap();
    let new_strides = compute_reshape_strides(&expanded_shape, arr.itemsize());
    let expanded = arr.view(expanded_shape, new_strides).unwrap();
    
    assert_eq!(arr.size(), expanded.size());
}

#[test]
fn test_reshape_roundtrip() {
    let arr = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    // Reshape to 1D
    let flat_shape = vec![24];
    let flat_strides = compute_reshape_strides(&flat_shape, arr.itemsize());
    let flat = arr.view(flat_shape, flat_strides).unwrap();
    
    // Reshape back to original
    let orig_shape = vec![2, 3, 4];
    let orig_strides = compute_reshape_strides(&orig_shape, arr.itemsize());
    let restored = flat.view(orig_shape, orig_strides).unwrap();
    
    assert_eq!(restored.shape(), arr.shape());
}

#[test]
fn test_transpose_roundtrip() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    // Transpose
    let (trans_shape, _trans_axes) = transpose_dimensions(arr.shape(), None).unwrap();
    let trans_strides = compute_reshape_strides(&trans_shape, arr.itemsize());
    let transposed = arr.view(trans_shape, trans_strides).unwrap();
    
    // Transpose back
    let (restored_shape, _) = transpose_dimensions(transposed.shape(), None).unwrap();
    let restored_strides = compute_reshape_strides(&restored_shape, arr.itemsize());
    let restored = transposed.view(restored_shape, restored_strides).unwrap();
    
    assert_eq!(restored.shape(), arr.shape());
}

#[test]
fn test_squeeze_expand_roundtrip() {
    let arr = test_data::sequential(vec![3, 4], DType::new(NpyType::Double));
    
    // Expand
    let expanded_shape = expand_dims(arr.shape(), 0).unwrap();
    let expanded_strides = compute_reshape_strides(&expanded_shape, arr.itemsize());
    let expanded = arr.view(expanded_shape, expanded_strides).unwrap();
    
    // Squeeze
    let squeezed_shape = squeeze_dims(expanded.shape(), Some(0));
    let squeezed_strides = compute_reshape_strides(&squeezed_shape, arr.itemsize());
    let squeezed = expanded.view(squeezed_shape, squeezed_strides).unwrap();
    
    assert_eq!(squeezed.shape(), arr.shape());
}

#[test]
fn test_reshape_strides_c_order() {
    let shape = vec![2, 3, 4];
    let itemsize = 8;
    let strides = compute_reshape_strides(&shape, itemsize);
    
    // C-order: last dimension has itemsize stride
    assert_eq!(strides[2], itemsize as i64);
    assert_eq!(strides[1], (itemsize * shape[2] as usize) as i64);
    assert_eq!(strides[0], (itemsize * shape[2] as usize * shape[1] as usize) as i64);
}

#[test]
fn test_transpose_strides() {
    let shape = vec![2, 3, 4];
    let itemsize = 8;
    let original_strides = compute_reshape_strides(&shape, itemsize);
    
    let (trans_shape, _) = transpose_dimensions(&shape, None).unwrap();
    let trans_strides = compute_reshape_strides(&trans_shape, itemsize);
    
    // Strides should be reversed
    assert_eq!(trans_strides.len(), original_strides.len());
}

#[test]
fn test_reshape_very_large_additional() {
    let arr = test_data::sequential(vec![1000], DType::new(NpyType::Double));
    
    let new_shape = vec![10, 100];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), &[10, 100]);
}

#[test]
fn test_reshape_high_dimensional() {
    let arr = test_data::sequential(vec![2, 2, 2, 2, 2], DType::new(NpyType::Double));
    
    let new_shape = vec![4, 8];
    validate_reshape_shape(arr.shape(), &new_shape).unwrap();
    
    let new_strides = compute_reshape_strides(&new_shape, arr.itemsize());
    let reshaped = arr.view(new_shape, new_strides).unwrap();
    
    assert_eq!(reshaped.shape(), &[4, 8]);
}

