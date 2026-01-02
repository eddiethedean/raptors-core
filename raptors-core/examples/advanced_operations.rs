//! Advanced operations examples for Raptors Core

use raptors_core::{Array, zeros, ones};
use raptors_core::types::{DType, NpyType};
use raptors_core::ufunc::reduction::{sum_along_axis, mean_along_axis};
use raptors_core::ufunc::loop_exec::create_unary_ufunc_loop;
use raptors_core::ufunc::advanced::math_ufuncs::*;

fn main() {
    println!("=== Raptors Core Advanced Operations Examples ===\n");
    
    // Example 1: Reductions
    println!("1. Reduction Operations:");
    let array = ones(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    
    let sum_axis0 = sum_along_axis(&array, Some(0)).unwrap();
    println!("   Sum along axis 0 shape: {:?}", sum_axis0.shape());
    
    let mean_axis1 = mean_along_axis(&array, Some(1)).unwrap();
    println!("   Mean along axis 1 shape: {:?}", mean_axis1.shape());
    
    let total_sum = sum_along_axis(&array, None).unwrap();
    println!("   Total sum: {:?}\n", total_sum.shape());
    
    // Example 2: Mathematical Functions
    println!("2. Mathematical Functions:");
    let array = zeros(vec![10], DType::new(NpyType::Double)).unwrap();
    
    // Create sin ufunc
    let sin_ufunc = create_sin_ufunc();
    let mut output = zeros(vec![10], DType::new(NpyType::Double)).unwrap();
    create_unary_ufunc_loop(&sin_ufunc, &array, &mut output).unwrap();
    println!("   Applied sin function");
    
    // Create exp ufunc
    let exp_ufunc = create_exp_ufunc();
    let mut output = zeros(vec![10], DType::new(NpyType::Double)).unwrap();
    create_unary_ufunc_loop(&exp_ufunc, &array, &mut output).unwrap();
    println!("   Applied exp function\n");
    
    // Example 3: Iteration
    println!("3. Array Iteration:");
    use raptors_core::iterators::FlatIterator;
    let array = zeros(vec![5], DType::new(NpyType::Double)).unwrap();
    let mut iter = FlatIterator::new(&array);
    let mut count = 0;
            while iter.next() {
        count += 1;
    }
    println!("   Iterated over {} elements\n", count);
    
    // Example 4: Convenience Methods
    println!("4. Convenience Methods:");
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let array = Array::from_slice(&data, vec![2, 2], DType::new(NpyType::Double)).unwrap();
    println!("   Created array from slice");
    
    let vec: Vec<f64> = unsafe { array.to_vec().unwrap() };
    println!("   Converted to Vec: {:?}\n", vec);
    
    println!("=== Advanced Examples Complete ===");
}

