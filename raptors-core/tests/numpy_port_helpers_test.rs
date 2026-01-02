//! Test file to verify NumPy porting helpers work correctly

#![allow(unused_unsafe)]

mod numpy_port {
    pub mod helpers;
}

use numpy_port::helpers::*;
use raptors_core::array::Array;
use raptors_core::types::{DType, NpyType};
use raptors_core::zeros;

#[test]
fn test_assert_array_equal() {
    let a = zeros(vec![3], DType::new(NpyType::Double)).unwrap();
    let b = zeros(vec![3], DType::new(NpyType::Double)).unwrap();
    
    assert_array_equal(&a, &b);
}

#[test]
fn test_assert_array_almost_equal() {
    let mut a = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        *a_ptr.add(0) = 1.0;
        *a_ptr.add(1) = 2.0;
        *a_ptr.add(2) = 3.0;
        *b_ptr.add(0) = 1.0000001;
        *b_ptr.add(1) = 2.0000001;
        *b_ptr.add(2) = 3.0000001;
    }
    
    assert_array_almost_equal(&a, &b, 1e-5, 1e-8);
}

#[test]
fn test_assert_array_less() {
    let mut a = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    let mut b = Array::new(vec![3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let a_ptr = a.data_ptr_mut() as *mut f64;
        let b_ptr = b.data_ptr_mut() as *mut f64;
        *a_ptr.add(0) = 1.0;
        *a_ptr.add(1) = 2.0;
        *a_ptr.add(2) = 3.0;
        *b_ptr.add(0) = 2.0;
        *b_ptr.add(1) = 3.0;
        *b_ptr.add(2) = 4.0;
    }
    
    assert_array_less(&a, &b);
}

#[test]
fn test_random_array() {
    let arr = random_array(vec![5], DType::new(NpyType::Double));
    assert_eq!(arr.shape(), &[5]);
    assert_eq!(arr.size(), 5);
}

#[test]
fn test_nan_array() {
    let arr = nan_array(vec![3], DType::new(NpyType::Double));
    assert_eq!(arr.shape(), &[3]);
    
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        assert!(ptr.add(0).read().is_nan());
        assert!(ptr.add(1).read().is_nan());
        assert!(ptr.add(2).read().is_nan());
    }
}

#[test]
fn test_inf_array() {
    let arr = inf_array(vec![3], DType::new(NpyType::Double));
    assert_eq!(arr.shape(), &[3]);
    
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        assert!(ptr.add(0).read().is_infinite());
        assert!(ptr.add(1).read().is_infinite());
        assert!(ptr.add(2).read().is_infinite());
    }
}

#[test]
fn test_sequential_data() {
    let arr = test_data::sequential(vec![5], DType::new(NpyType::Double));
    assert_eq!(arr.shape(), &[5]);
    
    unsafe {
        let ptr = arr.data_ptr() as *const f64;
        assert_eq!(*ptr.add(0), 0.0);
        assert_eq!(*ptr.add(1), 1.0);
        assert_eq!(*ptr.add(2), 2.0);
        assert_eq!(*ptr.add(3), 3.0);
        assert_eq!(*ptr.add(4), 4.0);
    }
}


