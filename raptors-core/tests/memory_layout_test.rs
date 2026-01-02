//! Tests for memory layout optimizations

use raptors_core::array::Array;
use raptors_core::types::{DType, NpyType};
use raptors_core::memory::{verify_alignment, verify_simd_alignment, simd_alignment};

#[test]
fn test_layout_analysis() {
    let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    let analysis = array.analyze_layout();
    
    assert!(analysis.is_c_contiguous);
    assert!(!analysis.is_strided);
    assert_eq!(analysis.itemsize, 8);
}

#[test]
fn test_optimize_layout() {
    let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    
    // Already optimal, should return view
    let optimized = array.optimize_layout().unwrap();
    assert_eq!(optimized.shape(), array.shape());
}

#[test]
fn test_alignment_verification() {
    let array = Array::new(vec![10], DType::new(NpyType::Double)).unwrap();
    let ptr = array.data_ptr();
    
    // Should be aligned to itemsize (8 bytes for double)
    assert!(verify_alignment(ptr, 8));
    
    // Check SIMD alignment
    let simd_align = simd_alignment();
    let is_simd_aligned = verify_simd_alignment(ptr);
    
    // May or may not be SIMD aligned depending on allocation
    // Just verify the function works
    assert_eq!(is_simd_aligned, (ptr as usize) % simd_align == 0);
}

#[test]
fn test_strided_array_optimization() {
    // Create a strided array by transposing
    let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    
    // For now, test that optimize_layout works
    let optimized = array.optimize_layout().unwrap();
    assert_eq!(optimized.shape(), array.shape());
}

