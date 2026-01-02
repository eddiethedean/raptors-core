//! Basic usage examples for Raptors Core

use raptors_core::{Array, zeros, ones, empty, ArrayBuilder};
use raptors_core::types::{DType, NpyType};
use raptors_core::operations::{add, multiply};

fn main() {
    println!("=== Raptors Core Basic Usage Examples ===\n");
    
    // Example 1: Array Creation
    println!("1. Array Creation:");
    let shape = vec![3, 4];
    let dtype = DType::new(NpyType::Double);
    
    let zeros_arr = zeros(shape.clone(), dtype.clone()).unwrap();
    println!("   Zeros array shape: {:?}", zeros_arr.shape());
    
    let ones_arr = ones(shape.clone(), dtype.clone()).unwrap();
    println!("   Ones array shape: {:?}", ones_arr.shape());
    
    let empty_arr = empty(shape, dtype).unwrap();
    println!("   Empty array created\n");
    
    // Example 2: Array Properties
    println!("2. Array Properties:");
    let array = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    println!("   Shape: {:?}", array.shape());
    println!("   Size: {}", array.size());
    println!("   NDim: {}", array.ndim());
    println!("   DType: {}", array.dtype().name());
    println!("   Is C-contiguous: {}\n", array.is_c_contiguous());
    
    // Example 3: Array Operations
    println!("3. Array Operations:");
    let a = ones(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    let b = ones(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    
    let sum = add(&a, &b).unwrap();
    println!("   Sum array shape: {:?}", sum.shape());
    
    let prod = multiply(&a, &b).unwrap();
    println!("   Product array shape: {:?}\n", prod.shape());
    
    // Example 4: Shape Manipulation
    println!("4. Shape Manipulation:");
    let array = zeros(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    
    // Reshape by creating a new array with new shape
    use raptors_core::shape::shape::validate_reshape_shape;
    let new_shape = vec![12];
    validate_reshape_shape(array.shape(), &new_shape).unwrap();
    let reshaped = Array::new(new_shape.clone(), array.dtype().clone()).unwrap();
    println!("   Reshaped from [3, 4] to {:?}", reshaped.shape());
    
    // Transpose using shape manipulation
    use raptors_core::shape::shape::transpose_dimensions;
    let (transposed_shape, _) = transpose_dimensions(array.shape(), None).unwrap();
    let transposed = Array::new(transposed_shape.clone(), array.dtype().clone()).unwrap();
    println!("   Transposed shape: {:?}\n", transposed.shape());
    
    // Example 5: Using Builder Pattern
    println!("5. Builder Pattern:");
    let array = ArrayBuilder::new()
        .with_shape(vec![2, 3])
        .with_dtype(DType::new(NpyType::Double))
        .with_fill_value(5.0)
        .build()
        .unwrap();
    println!("   Built array with shape: {:?}\n", array.shape());
    
    println!("=== Examples Complete ===");
}

