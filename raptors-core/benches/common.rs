//! Common utilities for benchmarks

use raptors_core::{Array, zeros};
use raptors_core::types::{DType, NpyType};

/// Create a test array for benchmarking
pub fn create_test_array(shape: Vec<i64>) -> Array {
    zeros(shape, DType::new(NpyType::Double)).unwrap()
}

/// Create a large test array
pub fn create_large_array() -> Array {
    create_test_array(vec![1000, 1000])
}

/// Create a small test array
pub fn create_small_array() -> Array {
    create_test_array(vec![10, 10])
}

